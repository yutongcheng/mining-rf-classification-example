# This example demonstrates only the Level 1 commodity classification model (Non-metals, Non-iron metals, Iron); the methodology for the other levels remains the same.

# Install and load necessary libraries
install.packages("smotefamily")
install.packages("randomForest")
install.packages("gee")
install.packages("caret")
install.packages("DMwR2")
install.packages("stats")
install.packages("dplyr")
install.packages("ggplot2")

library(smotefamily)
library(randomForest)
library(gee)
library(caret)
library(DMwR2)
library(stats)
library(dplyr)
library(ggplot2)

# Load training data
data <- read.csv("Example_Data_4.csv")

# Identify the columns to exclude (ID and label_1) for further data processing.
exclude_columns <- c("ID", "label_1")

class1_data_o <- data[data$label_1 == "1", ]
class1_data_o$label_1<-as.numeric(class1_data_o$label_1)

class2_data_o <- data[data$label_1 == "2", ]
class2_data_o$label_1<-as.numeric(class2_data_o$label_1)

class3_data_o <- data[data$label_1 == "3", ]
class3_data_o$label_1<-as.numeric(class3_data_o$label_1)

# Given the class imbalances, the Synthetic Minority Oversampling Technique (SMOTE) was used to generate synthetic samples and oversample minority classes. Additionally, clustering-based k-means undersampling was applied to majority classes to reduce their dominance in the dataset when there was a significant difference.
# In this example, let's balance the number of all classes to 50.

set.seed(128)  # For reproducibility

# Set the desired number of clusters (k) - This is how many samples you'll end up with
k <- 50  # Adjust 'k' as needed
# Apply K-means clustering
set.seed(45)

# Apply K-means clustering
class2_data_o_k <- kmeans(class2_data_o[, !names(class2_data_o) %in% exclude_columns], centers = k)

# Add cluster labels to the data
class2_data_o$cluster <- class2_data_o_k$cluster

# Sample one or a few instances from each cluster
class2_data_smote <- class2_data_o %>%
  group_by(cluster) %>%
  sample_n(1)  # Adjust 'n' for more samples per cluster if needed

class2_data_smote<-class2_data_smote[-29]
class2_data_smote<-class2_data_smote[-27]
colnames(class2_data_smote)[27]<-"class"

# Apply SMOTE with K = 5, excluding the ID and label columns
smote_class1 <- SMOTE(X = class1_data_o[, !names(class1_data_o) %in% exclude_columns], 
                      target = class1_data_o$label_1, K = 5, dup_size = 2)
synthetic_samples1 <- smote_class1$syn_data[1:32,]
colnames(class1_data_o)[28] <- "class"
class1_data_smote<-rbind(class1_data_o[-27], synthetic_samples1)

# Apply SMOTE with K = 5, excluding the ID and label columns
smote_class3 <- SMOTE(X = class3_data_o[, !names(class3_data_o) %in% exclude_columns], 
                      target = class3_data_o$label_1, K = 5, dup_size = 4)
synthetic_samples3 <- smote_class3$syn_data[1:38,]
colnames(class3_data_o)[28] <- "class"
class3_data_smote<-rbind(class3_data_o[-27], synthetic_samples3)

# Merge classes
balanced_data <- rbind(class1_data_smote, class2_data_smote, class3_data_smote)
balanced_data$ID <- paste0("ID_", seq_len(nrow(balanced_data)))
balanced_data$class <- as.factor(balanced_data$class)

# Separate the data by class
class1_data <- balanced_data[balanced_data$class == "1", ]
class2_data <- balanced_data[balanced_data$class == "2", ]
class3_data <- balanced_data[balanced_data$class == "3", ]

set.seed(123) 
# Sample 80% for training and 20% for testing from each class

# Class 1
train_indices_class1 <- sample(1:nrow(class1_data), 0.8 * nrow(class1_data))
train_class1 <- class1_data[train_indices_class1, ]
test_class1 <- class1_data[-train_indices_class1, ]

# Class 2
train_indices_class2 <- sample(1:nrow(class2_data), 0.8 * nrow(class2_data))
train_class2 <- class2_data[train_indices_class2, ]
test_class2 <- class2_data[-train_indices_class2, ]

# Class 3
train_indices_class3 <- sample(1:nrow(class3_data), 0.8 * nrow(class3_data))
train_class3 <- class3_data[train_indices_class3, ]
test_class3 <- class3_data[-train_indices_class3, ]

# Combine the training and testing sets from both classes
train_data <- rbind(train_class1, train_class2, train_class3)
test_data <- rbind(test_class1, test_class2, test_class3)

# Train the Random Forest Model for classification
train_data$class<-as.factor(train_data$class)
test_data$class<-as.factor(test_data$class)

# Set up cross-validation
control <- trainControl(method = "cv", number = 10)  # 10-fold cross-validation

# Expand the tuning grid for mtry
tunegrid <- expand.grid(.mtry = c(5))

# Train the Random Forest model using cross-validation with expanded mtry values
rf_model_cv <- train(class ~ . - ID, 
                     data = train_data, 
                     method = "rf", 
                     tuneGrid = tunegrid, 
                     trControl = control,
                     ntree = 500,
                     maxnodes = 300)   # Limiting the complexity of the trees

# Print the results of cross-validation
print(rf_model_cv)

# Make predictions on the test data using the best model
best_rf_model <- rf_model_cv$finalModel
predictions <- predict(best_rf_model, newdata = test_data)

# Calculate the confusion matrix
conf_matrix <- confusionMatrix(predictions, test_data$class)

# Print the confusion matrix
print(conf_matrix)

# Make predictions on the unknown data  
# Since this is just an example, we used the training sample as the prediction sample to demonstrate the methodology.  
# In the actual study, an exclusive prediction dataset was used instead, ensuring that the model was applied to classify commodities in unlabelled polygons. 
unknown_data<-data[,-28]
colSums(is.na(unknown_data))
unknown_data_clean <- na.omit(unknown_data)
predictions <- predict(best_rf_model, newdata = unknown_data_clean)
summary(predictions)
# The prediction results at this level will then be used as the prediction dataset for the next levels.

