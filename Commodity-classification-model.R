# This example demonstrates only the Level 1 commodity classification model (Non-metals, Iron, chalcophile-related metals, and miscellaneous metals); the methodology for the other levels remains the same.

# Install and load necessary libraries
# install.packages("smotefamily")
# install.packages("randomForest")
# install.packages("gee")
# install.packages("caret")
# install.packages("DMwR2")
# install.packages("stats")
# install.packages("dplyr")
# install.packages("ggplot2")

library(smotefamily)
library(randomForest)
library(gee)
library(caret)
library(DMwR2)
library(stats)
library(dplyr)
library(ggplot2)

# Load training data
data <- read.csv("Example_Data_2.csv")

# Identify the columns to exclude (ID and label_1) for further data processing.
exclude_columns <- c("uni_id_ast", "L1", "Com")

class1_data_oo <- data[data$L1 == "1", ]
class1_data_oo$L1<-as.numeric(class1_data_oo$L1)

class2_data_oo <- data[data$L1 == "2", ]
class2_data_oo$L1<-as.numeric(class2_data_oo$L1)

class3_data_oo <- data[data$L1 == "3", ]
class3_data_oo$L1<-as.numeric(class3_data_oo$L1)

class4_data_oo <- data[data$L1 == "4", ]
class4_data_oo$L1<-as.numeric(class4_data_oo$L1)


set.seed(128)  # For reproducibility

# Set the desired number of clusters (k) - This is how many samples you'll end up with
k <- 100  # Adjust 'k' as needed
test_ratio <- 0.20
test_size  <- as.integer(round(k * test_ratio))  # exact target per class

# ---- stratified-by-Com splitter ----
split_fixed_test_by_Com <- function(df, test_size, class_name = "class", com_col = "Com") {
  n <- nrow(df)
  if (n < test_size) {
    stop(sprintf("%s has only %d rows; cannot take test_size = %d. Adjust k or test_ratio.",
                 class_name, n, test_size))
  }
  # Tabulate Com groups
  com_vals <- df[[com_col]]
  tab <- table(com_vals)
  coms <- names(tab)
  G <- length(tab)
  
  # Base equal quota per Com, then fill remainder
  base_quota <- floor(test_size / G)
  # First pass: sample up to base_quota from each Com (or all if fewer)
  picked_idx <- integer(0)
  per_com_picked <- integer(G)
  names(per_com_picked) <- coms
  
  for (j in seq_along(coms)) {
    cj <- coms[j]
    idx_j <- which(com_vals == cj)
    qj <- min(length(idx_j), base_quota)
    if (qj > 0) {
      picked_idx <- c(picked_idx, sample(idx_j, qj, replace = FALSE))
    }
    per_com_picked[j] <- qj
  }
  
  # How many more do we need after the first pass?
  remaining_needed <- test_size - length(picked_idx)
  
  if (remaining_needed > 0) {
    # Build a pool of leftover indices (not yet picked)
    all_idx <- seq_len(n)
    already <- rep(FALSE, n)
    already[picked_idx] <- TRUE
    pool <- all_idx[!already]
    
    # Sample the remaining from the pool (random across Com; simple & fair)
    picked_idx <- c(picked_idx, sample(pool, remaining_needed, replace = FALSE))
  }
  
  # Build test/train data.frames
  test_df  <- df[picked_idx, , drop = FALSE]
  train_df <- df[-picked_idx, , drop = FALSE]
  
  list(train = train_df, test = test_df, idx_test = picked_idx)
}

# Apply to four classes
s1 <- split_fixed_test_by_Com(class1_data_oo, test_size, "class1", "Com")
s2 <- split_fixed_test_by_Com(class2_data_oo, test_size, "class2", "Com")
s3 <- split_fixed_test_by_Com(class3_data_oo, test_size, "class3", "Com")
s4 <- split_fixed_test_by_Com(class4_data_oo, test_size, "class4", "Com")

# Train/test per class
train_class1 <- s1$train; test_class1 <- s1$test
train_class2 <- s2$train; test_class2 <- s2$test
train_class3 <- s3$train; test_class3 <- s3$test
train_class4 <- s4$train; test_class4 <- s4$test

class1_data_o <- train_class1
class2_data_o <- train_class2
class3_data_o <- train_class3
class4_data_o <- train_class4

# Given the class imbalances, the Synthetic Minority Oversampling Technique (SMOTE) was used to generate synthetic samples and oversample minority classes. Additionally, clustering-based k-means undersampling was applied to majority classes to reduce their dominance in the dataset when there was a significant difference.
# In this example, let's balance the number of all classes to 80.

set.seed(128)  # For reproducibility

# Set the desired number of clusters (k) - This is how many samples you'll end up with
k <- k*0.8  # Adjust 'k' as needed
# Apply K-means clustering
set.seed(45)

# Apply K-means clustering
class1_data_o_k <- kmeans(class1_data_o[, !names(class1_data_o) %in% exclude_columns], centers = k)

# Add cluster labels to the data
class1_data_o$cluster <- class1_data_o_k$cluster

# Sample one or a few instances from each cluster
# Although the variable name includes "smote", this step actually performs k-means undersampling. 
# The name is kept only for consistency with later code where SMOTE results are combined.
class1_data_smote <- class1_data_o %>%
  group_by(cluster) %>%
  sample_n(1)  # Adjust 'n' for more samples per cluster if needed

class1_data_smote<-class1_data_smote[-41]
class1_data_smote<-class1_data_smote[-42]
colnames(class1_data_smote)[41]<-"class"
class1_data_smote<-class1_data_smote[-1]

# Apply K-means clustering
class2_data_o_k <- kmeans(class2_data_o[, !names(class2_data_o) %in% exclude_columns], centers = k)

# Add cluster labels to the data
class2_data_o$cluster <- class2_data_o_k$cluster

# Sample one or a few instances from each cluster
# Although the variable name includes "smote", this step actually performs k-means undersampling. 
# The name is kept only for consistency with later code where SMOTE results are combined.
class2_data_smote <- class2_data_o %>%
  group_by(cluster) %>%
  sample_n(1)  # Adjust 'n' for more samples per cluster if needed

class2_data_smote<-class2_data_smote[-41]
class2_data_smote<-class2_data_smote[-42]
colnames(class2_data_smote)[41]<-"class"
class2_data_smote<-class2_data_smote[-1]

# Apply K-means clustering
class3_data_o_k <- kmeans(class3_data_o[, !names(class3_data_o) %in% exclude_columns], centers = k)
class3_data_o$cluster <- class3_data_o_k$cluster

# Sample one or a few instances from each cluster
# Although the variable name includes "smote", this step actually performs k-means undersampling. 
# The name is kept only for consistency with later code where SMOTE results are combined.
class3_data_smote <- class3_data_o %>%
  group_by(cluster) %>%
  sample_n(1)  # Adjust 'n' for more samples per cluster if needed

class3_data_smote<-class3_data_smote[-41]
class3_data_smote<-class3_data_smote[-42]
colnames(class3_data_smote)[41]<-"class"
class3_data_smote<-class3_data_smote[-1]

# Apply SMOTE with K = 5, excluding the ID and label columns
smote_class4 <- SMOTE(X = class4_data_o[, !names(class4_data_o) %in% exclude_columns], 
                      target = class4_data_o$L1, K = 5, dup_size =2)
synthetic_samples4 <- smote_class4$syn_data[1:43,]

class4_data_o<-class4_data_o[-41]
colnames(class4_data_o)[41] <- "class"
class4_data_o<-class4_data_o[-1]
class4_data_smote<-rbind(class4_data_o, synthetic_samples4)

# Merge classes
balanced_data <- rbind(class1_data_smote, class2_data_smote, class3_data_smote, class4_data_smote)
balanced_data$ID <- paste0("ID_", seq_len(nrow(balanced_data)))
balanced_data$class <- as.factor(balanced_data$class)

train_data<-balanced_data

test_data<-rbind(test_class1,test_class2,test_class3,test_class4)
test_data$ID <- paste0("ID_", seq_len(nrow(test_data)))
test_data<-test_data[-1]
test_data<-test_data[-40]
colnames(test_data)[40] <- "class"

test_data$class <- as.factor(test_data$class)

# Train the Random Forest Model for classification
# All parameter settings (e.g. mtry, ntree and number of nodes) should be calibrated to find the most suitable configuration for the dataset. 
# The values below are provided only as example settings.
# Set up cross-validation
control <- trainControl(method = "cv", number = 10)  # 10-fold cross-validation

# Expand the tuning grid for mtry
tunegrid <- expand.grid(.mtry = c(15))

# Train the Random Forest model with caret
rf_model_cv <- train(
  class ~ . - ID,
  data = train_data,
  method = "rf",
  tuneGrid = tunegrid,
  trControl = control,
  ntree = 500
)

# Print cross-validation results
print(rf_model_cv)

# ---- Predict with probabilities ----
rf_prob <- predict(rf_model_cv, newdata = test_data, type = "prob")

# Convert probabilities -> predicted class (take the max per row)
rf_pred <- colnames(rf_prob)[max.col(rf_prob, ties.method = "first")]
rf_pred <- factor(rf_pred, levels = levels(train_data$class))

# ---- Confusion matrix ----
conf_matrix <- confusionMatrix(rf_pred, test_data$class)
print(conf_matrix)

library(dplyr)
library(stringr)
library(randomForest)

# --- pull importance from the trained model (caret -> randomForest) ---
best_rf_model <- rf_model_cv$finalModel
imp_mat <- randomForest::importance(best_rf_model, type = 2)  # MeanDecreaseGini

importance_df <- data.frame(
  feature    = rownames(imp_mat),
  importance = imp_mat[, "MeanDecreaseGini"],
  row.names  = NULL
) %>% arrange(desc(importance))


## ---- Helper function definition ----
align_for_rf <- function(newdf, train_df, drop_cols = c("class","ID")) {
  predictors <- setdiff(names(train_df), drop_cols)
  
  missing <- setdiff(predictors, names(newdf))
  for (m in missing) {
    tr_col <- train_df[[m]]
    if (is.factor(tr_col)) {
      mf <- names(sort(table(tr_col), decreasing = TRUE))[1]
      newdf[[m]] <- factor(rep(mf, nrow(newdf)), levels = levels(tr_col))
    } else {
      med <- suppressWarnings(median(tr_col, na.rm = TRUE))
      if (is.na(med)) med <- 0
      newdf[[m]] <- rep(med, nrow(newdf))
    }
  }
  
  for (p in predictors) {
    tr_col <- train_df[[p]]
    if (is.factor(tr_col)) {
      newdf[[p]] <- as.character(newdf[[p]])
      newdf[[p]] <- factor(newdf[[p]], levels = levels(tr_col))
      if (anyNA(newdf[[p]])) {
        mf <- names(sort(table(tr_col), decreasing = TRUE))[1]
        newdf[[p]][is.na(newdf[[p]])] <- mf
        newdf[[p]] <- factor(newdf[[p]], levels = levels(tr_col))
      }
    } else {
      newdf[[p]] <- suppressWarnings(as.numeric(newdf[[p]]))
      med <- suppressWarnings(median(tr_col, na.rm = TRUE))
      if (is.na(med)) med <- 0
      na_idx <- which(!is.finite(newdf[[p]]) | is.na(newdf[[p]]))
      if (length(na_idx)) newdf[[p]][na_idx] <- med
    }
  }
  
  X <- newdf[, predictors, drop = FALSE]
  
  if (anyNA(X)) {
    for (p in predictors) {
      if (is.numeric(X[[p]])) {
        med <- suppressWarnings(median(train_df[[p]], na.rm = TRUE))
        if (is.na(med)) med <- 0
        X[[p]][is.na(X[[p]])] <- med
      } else if (is.factor(X[[p]])) {
        mf <- names(sort(table(train_df[[p]]), decreasing = TRUE))[1]
        X[[p]][is.na(X[[p]])] <- mf
        X[[p]] <- factor(X[[p]], levels = levels(train_df[[p]]))
      }
    }
  }
  X
}

# best_rf_model from caret
best_rf_model <- rf_model_cv$finalModel

# Make predictions on the unknown data  
# Since this is just an example, we used the training sample as the prediction sample to demonstrate the methodology.  
# In the actual study, an exclusive prediction dataset was used instead, ensuring that the model was applied to classify commodities in unlabelled polygons. 
# Align your unknown data to training predictors (no 'class' needed)
unknown <- data[,-42]
unknown_clean <- na.omit(unknown)  # optional; you can skip if you want full imputation

Xnew <- align_for_rf(unknown_clean, train_data, drop_cols = c("class","ID"))

# Predict probabilities ONLY (no voting)
prob_unknown <- predict(best_rf_model, newdata = Xnew, type = "prob")

# Take the highest-probability class
pred_unknown <- colnames(prob_unknown)[max.col(prob_unknown, ties.method = "first")]
conf_unknown <- apply(prob_unknown, 1, max)

# Combine & save
out <- cbind(unknown_clean,
             PredictedClass = pred_unknown,
             MaxProb = conf_unknown)
table(out$PredictedClass)

# --- OOB metrics (Overfitting Detection) --- 
# This section compares out-of-bag (OOB) and independent test accuracies to detect potential overfitting. 
# A smaller performance gap (OOB − Test) indicates better generalisation. In the study, a threshold of 0.025 was used to determine whether the model was overfitted. 
# The example values shown here do not necessarily fall below 0.025, as this script is only a demonstration using example data.

library(caret)

rf <- rf_model_cv$finalModel  # underlying randomForest object

oob_err <- rf$err.rate[nrow(rf$err.rate), "OOB"]
oob_acc <- 1 - oob_err

# Balanced Accuracy helper (works for binary or multiclass)
bal_acc_from_cm <- function(cm) {
  bc <- cm$byClass
  if (is.matrix(bc)) {
    if ("Balanced Accuracy" %in% colnames(bc)) return(mean(bc[, "Balanced Accuracy"], na.rm=TRUE))
    if (all(c("Sensitivity","Specificity") %in% colnames(bc))) return(mean((bc[,"Sensitivity"]+bc[,"Specificity"])/2, na.rm=TRUE))
  } else {
    nms <- names(bc)
    if ("Balanced Accuracy" %in% nms) return(as.numeric(bc["Balanced Accuracy"]))
    if (all(c("Sensitivity","Specificity") %in% nms)) return(mean(c(bc["Sensitivity"], bc["Specificity"])))
  }
  NA_real_
}

# OOB confusion matrix and Balanced Accuracy
oob_pred <- factor(rf$predicted, levels = levels(train_data$class))
cm_oob   <- confusionMatrix(oob_pred, train_data$class)
oob_bal  <- bal_acc_from_cm(cm_oob)

# --- Test metrics ---
pred_test <- predict(rf_model_cv, newdata = test_data, type = "raw")
cm_test   <- confusionMatrix(pred_test, test_data$class)
test_acc  <- as.numeric(cm_test$overall["Accuracy"])
test_bal  <- bal_acc_from_cm(cm_test)

# --- Compare (gaps) ---
cat(sprintf("OOB  Acc=%.3f | Test Acc=%.3f | Gap(OOB−Test)=%.3f\n", oob_acc, test_acc, oob_acc - test_acc))
cat(sprintf("OOB  Bal=%.3f | Test Bal=%.3f | Gap(OOB−Test)=%.3f\n", oob_bal,  test_bal,  oob_bal  - test_bal))

# The results at this level (L1) will then be used as the prediction dataset for the next levels.
# The resulting model outputs from all levels were then used to quantify deforestation and biodiversity risk. 
# The mining area, deforestation-to-mining area ratio, and Extinction Risk Index (ERI) for 20 commodities were further calculated based on the polygons from Maus et al. (2022) and Tang and Werner (2023) (licensed under CC BY-SA 4.0) and are available in this repository as Results.zip.
