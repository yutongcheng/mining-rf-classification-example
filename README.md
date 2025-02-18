# mining-rf-classification-example

# Open-Pit Identification and Commodity Classification

## Overview

This repository provides a methodology for **identifying open-pit mining sites** using **ASTER imagery** and classifying commodities based on spectral properties (band ratios). The workflow involves three main scripts:

1. **Open-pit-identification.js** - Identifies open pits using ASTER imagery and machine learning.
2. **Extract_ASTER_band_ratios.js** - Extracts ASTER band ratios for further classification.
3. **Commodity-classification-model.R** - Implements a **hierarchical random forest model** for commodity classification.

## Data

Due to data confidentiality, not all datasets used in the original study are publicly available. Instead, example datasets (**Example_Data_1**, **Example_Data_2**, **Example_Data_3**, and **Example_Data_4**) are provided to ensure reproducibility. Additionally, the commodity classification results based on Maus et al. (2022)'s polygons from FINEPRINT (**licensed under CC BY-SA 4.0**) are available in this repository.

- **Example_Data_1**: Training data for open-pit identification (**available in the GEE JavaScript code**).
- **Example_Data_2**: Unknown regions for open-pit prediction (**available in the GEE JavaScript code**).
- **Example_Data_3**: Identified open pits, used for extracting ASTER band ratios (**available in the GEE JavaScript code**).
- **Example_Data_4**: ASTER band ratio data for commodity classification (**available in this repository as `Example_Data_4.csv`**).
- **Commodity Classification Results**: The final classification results of commodity types, with mining polygon IDs following Maus et al. (2022)'s polygons from FINEPRINT (**licensed under CC BY-SA 4.0**) (**available in this repository as `Commodity-classification-results.csv`**).

## Workflow

### 1. Open-Pit Identification

- Uses **ASTER imagery from 2003-2005**.
- Applies **pre-processing functions** (radiance, surface temperature, reflectance calculation, masking, cloud detection, and scaling).
- Trains a **Random Forest classifier** using **Example_Data_1**.
- Classifies open pits in **Example_Data_2**.
- Outputs **Example_Data_3_pre**, which contains probabilities for land-use categories.

### 2. ASTER Band Ratio Extraction

- Uses **ASTER imagery from 2003-2005**.
- Extracts **band ratios (mineral indices)** using **Example_Data_3**.
- Outputs **Example_Data_4_pre**, which contains **ASTER band ratios** for each identified open-pit site.

### 3. Commodity Classification

- Implements a **hierarchical Random Forest model**.
- Balances class distribution using **SMOTE and k-means undersampling**.
- Uses **cross-validation (10-fold CV)** for model evaluation.
- Trains the **Level 1 commodity classification model** (Non-metals, Non-iron metals, Iron).
- Outputs **classification results** for labelled and unlabelled datasets.

## Scripts

### **Open-pit-identification.js**

- **Imports the training dataset and the dataset for prediction**.
- **Processes ASTER imagery**.
- **Implements a Random Forest model** for open-pit identification.
- **Outputs results as a CSV file**.

### **Extract_ASTER_band_ratios.js**

- **Extracts ASTER band ratios** from identified open pits.
- **Computes spectral indices** based on **Geoscience Australia (2004)**.
- **Outputs a dataset with 26 band ratios**.

### **Commodity-classification-model.R**

- **Reads Example_Data_4**.
- **Balances the dataset** using **SMOTE and k-means clustering**.
- **Trains a Random Forest model** with **10-fold cross-validation**.
- **Outputs classification results for commodities**.

## Requirements

### **Software & Dependencies**
- Google Earth Engine (Tested in GEE Code Editor, May 2024)
- R version 4.4.0 (Tested on RStudio)
  
### **Google Earth Engine (GEE)**
- **Required** for running `Open-pit-identification.js` and `Extract_ASTER_band_ratios.js`.
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
- Example_Data_3_pre.csv - Open-pit identification results.
- Example_Data_4_pre.csv - ASTER band ratios for commodity classification.
- Example_Commodity_classification_results - Predictions for identified commodities.

## Notes
  Ensure that Google Earth Engine (GEE) assets are correctly linked.
  Modify folder paths before exporting results to Google Drive.
  This repository provides an example methodology; adjustments may be required for different datasets and study areas.
