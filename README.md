# ICU Length of Stay Prediction

## By Edward Monbiot

## Overview
This project utilizes neural network and ensemble models to predict the length of stay for patients in an ICU, drawing on data from the MIMIC-III database.

## Data Description
- **Source:** MIMIC-III Database from Beth Israel Deaconess Medical Center's critical care units (2001-2012).
- **Components:** Train, test, and metadata datasets.
- **Features:** Vital signs, general characteristics (age, gender, etc.), and disease codes (`ICD9_diagnosis`).
- **Exclusions:** Features unknown on the first day of ICU admission.

## Data Pre-Processing
- Merged metadata with train and test sets.
- Conducted Exploratory Data Analysis (EDA).
- Cleaned, normalized, and imputed patient data (e.g., adjusting extreme ages, simplifying ethnicity and religion categories, merging insurance types).
- Defined numerical and categorical columns for analysis.
- Applied one-hot/binary encoding to numerical values.
- Grouped medical columns and dropped unused columns.
- Scaled variables based on characteristics using ColumnTransformer for simultaneous processing.
- Utilized MultiColumnTargetEncoder with a smoothing parameter.

## Analysis of Models
1. **Neural Network Model:**
   - Initial model: SKLearn MLPRegressor with sigmoid activation function.
   - Enhanced with ReLU activation function, adam solver, and small learning rate (0.01).
   - Keras Neural Networks trials, including early stopping and experiments with exponential activation.
   - Best model: Keras with 16 neurons in the first layer and exponential activation.
   
2. **Ensemble Models:**
   - Grid search for optimal hyperparameters in ridge regression, KNN, and Random Forest.
   - Stacking ensemble of Ridge regressor, KNN, and Random Forest, with Gradient Boosting as the meta-learner.
   - Exploration of log transformation on y_train, and analysis of its impact.

## Future Improvements
- **Neural Networks:** Investigate parameter regularization in Keras and implement dropout features.
- **Ensembles:** Test feature propagation to the meta-learner and apply SelectKBest for feature selection.

## Repository Structure
- `E_Monbiot_NN_Project.ipynb`: Neural network model notebook.
- `E_Monbiot_Ensembles_Project.ipynb`: Ensemble modeling notebook.
