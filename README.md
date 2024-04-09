# ICU Length of Stay Prediction

## Overview
This project utilizes neural network and ensemble models to predict the length of stay for patients in an ICU. Drawing on data from the MIMIC-III database, the models aim to accurately forecast ICU durations, aiding healthcare professionals in managing ICU resources effectively.

## Data Description
- **Source:** MIMIC-III Database, encompassing over forty thousand patients from Beth Israel Deaconess Medical Center's critical care units (2001-2012).
-  Train, test and metadata datasets
- **Features:**
  - Vital signs and general characteristics (age, gender, etc.) at the time of ICU admission.
  - Disease codes (`ICD9_diagnosis`) indicating primary and secondary conditions.
  - Exclusions: Features not known on the first day of ICU admission.

## Data Processing (same process for Neural Network and Ensembles)
- Merge metadata with train and test sets
- Exploratory Data Analysis
- Cleaning, normalizing and imputing patient data: e.g. Setting max age of 93 for extreme DOB data, simply ethnicity add "WHITE - BRAZILIAN" to "WHITE", Merge 'OTHER' religion into 'UNSPECIFIED' religion
- Handling missing values and outliers.
- Define Numerical and Categorical Columns

## Models
1. **Neural Network Model:**
   - Architecture details (layers, nodes, activation functions).
   - Training process and hyperparameter tuning.
2. **Ensemble Models:**
   - Description of ensemble techniques (e.g., Random Forest, Gradient Boosting).
   - Strategy for model combination and optimization.

## Model Evaluation
- Employing RMSE (Root Mean Squared Error) to measure model accuracy.
- Comparisons between neural network and ensemble model performances.
- Analysis of model residuals and error distribution.

## Insights and Observations
- Summary of key findings from model predictions.
- Discussion on model strengths and limitations.
- Implications for ICU length of stay prediction in healthcare settings.

## Future Directions
- Potential for further model refinement.
- Exploration of additional features and advanced modeling techniques.

## Repository Structure
- `E_Monbiot_NN_Project.ipynb`: Jupyter notebook containing the neural network model.
- `E_Monbiot_Ensembles_Project.ipynb`: Jupyter notebook detailing ensemble modeling.

## How to Run
- Instructions on setting up the environment (e.g., required libraries and dependencies).
- Steps to execute the notebooks and reproduce the results.

## Contributions
- [Your Full Name]: Design and development of the predictive models and analysis.
