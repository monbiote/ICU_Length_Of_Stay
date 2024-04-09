# ICU Length of Stay Prediction via Neural Networks and Ensemble Methods

## By Edward Monbiot

## Overview
This project utilizes neural network and ensemble models to predict the length of stay for patients in an ICU. Drawing on data from the MIMIC-III database, the models aim to accurately forecast ICU durations, aiding healthcare professionals in managing ICU resources effectively.

## Data Description
- **Source:** MIMIC-III Database, encompassing over forty thousand patients from Beth Israel Deaconess Medical Center's critical care units (2001-2012).
- **Datasets:** Train, test, and metadata.
- **Features:** 
  - Vital signs and general characteristics (age, gender, etc.) at the time of ICU admission.
  - Disease codes (`ICD9_diagnosis`) for primary and secondary conditions.
  - **Exclusions:** Features not known on the first day of ICU admission.

## Data Pre-Processing (for Both Neural Network and Ensembles)
- Merging metadata with train and test sets.
- Conducting Exploratory Data Analysis.
- Cleaning, normalizing, and imputing patient data. Examples include:
  - Setting max age to 93 for extreme DOB data.
  - Simplifying ethnicity (e.g., adding "WHITE - BRAZILIAN" to "WHITE").
  - Merging 'OTHER' religion into 'UNSPECIFIED', 'Medicaid' and 'Medicare' into 'Government' insurance.
- Defining numerical and categorical columns.
- One-hot/binary encoding numerical values.
- Grouping medical columns and dropping unused columns.
- Scaling variables based on characteristics (skewed/binary/dummy), multi-target encoding.
- Creating pipelines via `ColumnTransformer` for simultaneous processing.
- Using `MultiColumnTargetEncoder` with a smoothing parameter to balance overall mean and mean per category.

## Analysis of Models
1. **Neural Network Model:**
   - Initial simple SKLearn `MLPRegressor` with sigmoid activation function to introduce non-linearity.
   - Enhanced with ReLU activation via adam solver; small learning rate (0.01) for best Kaggle score.
   - Experimented with Keras Neural Networks; used early stopping to mitigate overfitting.
   - Best Keras model: exponential activation, first dense layer with 16 neurons and last dense layer with a single neuron.
2. **Ensemble Models:**
   - Utilized grid search with Ridge regression estimator, KNN, and Random Forest for hyperparameter tuning.
   - Base learners: Ridge regressor, K-Nearest Neighbors regressor, and Random Forest regressor. Gradient Boosting Regressor as the meta-learner.
   - Stacking Regressor ensemble combining base & meta-learner predictions.
   - Experimented with log-transformed `y_train`, but nominal `y_train` value yielded better results.

## Future Improvements
- **Neural Networks:** Explore parameter regularization in Keras and dropout features.
- **Ensembles:** Propagate features to the meta-learner, in addition to the existing weak learners; use `SelectKBest` to identify top correlated features.

## Repository Structure
- [`E_Monbiot_NN_Project.ipynb`](E_Monbiot_NN_Project.ipynb): Neural network model notebook.
- [`E_Monbiot_Ensembles_Project.ipynb`](E_Monbiot_Ensembles_Project.ipynb): Ensemble modeling notebook.
