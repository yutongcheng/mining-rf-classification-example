# mining-rf-classification-example

# Commodity Classification

## Overview

This repository provides a methodology for a **Commodity classification model** using **ASTER** and **HISUI imagery** based on spectral properties (band ratios). The workflow involves two main scripts:

1. **Extract_band_ratios.js** - Extracts band ratios from open-pit locations identified by Cheng et al. (2025) for subsequent classification. ASTER is used as an example because HISUI data, although freely accessible, require registration and approval for download.
2. **Commodity-classification-model.R** - Implements a **hierarchical random forest model** for commodity classification.

## Data

Due to data confidentiality, not all datasets used in the original study are publicly available. Instead, example datasets (**Example_Data_1**, and **Example_Data_2**) are provided to ensure reproducibility. Additionally, the commodity classification results (in the grid level) based on the polygons from Maus et al. (2022) and Tang and Werner (2023) (**licensed under CC BY-SA 4.0**) are available in this repository.

- **Example_Data_1**: Open-pit locations/pixels, used for deriving ASTER band ratios (**available in the GEE JavaScript code**).
- **Example_Data_2**: ASTER band ratio data for commodity classification (**available in this repository as `Example_Data_2.csv`**).
- **Results**: Final grid-level classification results of commodity types based on the polygons from Maus et al. (2022) and Tang and Werner (2023) (**licensed under CC BY-SA 4.0**) (**available in this repository as `Results.zip`**).

## Workflow

### 1. ASTER Band Ratio Extraction

- Uses **ASTER imagery from 2003-2005**.
- Extracts **band ratios (mineral indices)** using **Example_Data_1**.
- Produces **Example_Data_2_pre**, which contains ASTER band ratios for each identified open-pit location. Please note that the open-pit locations used in this example do not necessarily represent real ones. For the actual identified locations, please refer to Cheng et al. (2025).

### 2. Commodity Classification

- Implements a **hierarchical Random Forest model**.
- Splits the dataset into **80% for training and 20% for independent validation** to assess model generalisation.
- Applies **10-fold cross-validation (CV)** within the 80% training subset for internal model tuning and evaluation.
- Performs **SMOTE** and **k-means undersampling** after data splitting to balance class distribution and avoid data leakage.
- Trains the **Level 1 commodity classification model** (Non-metals, Iron, chalcophile-related metals, and miscellaneous metals).
- Outputs **Level 1 classification results**.

## Scripts

### **Extract_ASTER_band_ratios.js**

- **Extracts ASTER band ratios** from open-pit locations identified by Cheng et al. (2025).
- **Computes spectral indices** based on **Geoscience Australia (2004)**.
- **Outputs a dataset with 26 band ratios**.

### **Commodity-classification-model.R**

- **Reads Example_Data_2**.
- **Balances the dataset** using **SMOTE and k-means clustering**.
- **Trains a Random Forest model** with **10-fold cross-validation**.
- **Outputs classification results for commodities**.

## Requirements

### **Software & Dependencies**
- Google Earth Engine (Tested in GEE Code Editor, May 2024)
- R version 4.4.0 (Tested on RStudio)
  
### **Google Earth Engine (GEE)**
- **Required** for running `Extract_band_ratios.js`.
- **Data is stored in an EE FeatureCollection**.
  
### **R Environment**
- **Required** for `Commodity-classification-model.R`.
- Install the following **R packages**:
  ```r
  install.packages("smotefamily")
  install.packages("randomForest")
  install.packages("gee")
  install.packages("caret")
  install.packages("DMwR2")
  install.packages("stats")
  install.packages("dplyr")
  install.packages("ggplot2")
  
## Outputs
- Example_Data_2_pre.csv - ASTER band ratios for commodity classification.
- Example_L1_Commodity_classification_results - Predictions for identified commodities.

## Notes
  Ensure that Google Earth Engine (GEE) assets are correctly linked.
  Modify folder paths before exporting results to Google Drive.
  This repository provides an example methodology; adjustments may be required for different datasets and study areas.
## Licenses:
- The code in this repository is licensed under **CC BY-NC 4.0**.
- The results (**Results.zip**) are licensed under **CC BY-SA 4.0**.
