# ICU Length of Stay Prediction

## By Edward Monbiot

## Overview
This project utilizes neural network and ensemble models to predict the length of stay for patients in an ICU. Drawing on data from the MIMIC-III database, the models aim to accurately forecast ICU durations, aiding healthcare professionals in managing ICU resources effectively.

## Data Description
- **Source:** MIMIC-III Database, encompassing over forty thousand patients from Beth Israel Deaconess Medical Center's critical care units (2001-2012).
-  Train, test and metadata datasets
- **Features:**
  - Vital signs and general characteristics (age, gender, etc.) at the time of ICU admission.
  - Disease codes (`ICD9_diagnosis`) indicating primary and secondary conditions.
  - Exclusions: Features not known on the first day of ICU admission.

## Data Pre-Processing (same process for Neural Network and Ensembles)
- Merge metadata with train and test sets
- Exploratory Data Analysis
- Cleaning, normalizing and imputing patient data: e.g. Setting max age of 93 for extreme DOB data, simplify ethnicity add "WHITE - BRAZILIAN" to "WHITE", Merge 'OTHER' religion into 'UNSPECIFIED' religion, "Medicaid" "Medicare" to "Government" insurance. 
- Define Numerical and Categorical Columns
- One-hot/binary encoding numerical values
- Group medical columns and drop unused columns
- Scaled variables based on characteristics (skewed/binary/dummy), multi-target encoding, created pipelines via ColumnTransformer for simultaneous processing
- MultiColumnTargetEncoder used with smoothing parameter to balance between the overall mean and the mean per category

## Analysis of Models
1. **Neural Network Model:**
   - Started with simple SKLearn MLPRegressor model with sigmoid activation function, idea was to introduce non-linearity, enabling the model to learn complex patterns, 5,5 hidden layer size striking balance between complexity and the risk of overfitting.
   - Enchanced using ReLU (Rectified Linear Unit) activation function, via adam solver,  known for being effective on large datasets and handling sparse gradients well, chose a small learning rate (0.01) for precise adjustments to weights (produced best Kaggle score).
   - Experimented with Keras Neural Networks, started off with ReLU activation and first and second dense layers with 64 neurons with early stopping to stop training if it doesn't improve after 3 epochs, but results showed validation loss to be higher than the training       loss which suggests the model could be overfitting to the training data.
   - Best Keras Model involved using exponential activation, first dense layer having 16 neurons and last dense layer having just a single neuron. The training loss decreases sharply and then plateaus, which indicates that the model is learning from the training data         and then stabilizing.
2. **Ensemble Models:**
   - Began with defining grid search using ridge regression estimator (6-fold cross validation to identify optimun alpha), KNN (5-cross validation) and Random Forest number of trees in the forest (100, 200, 400), maximum depth of each tree (None, 3, 5, 20) (6-cross           validation)
   - Defined list of base lerners for a stacking ensemble, consisting of a Ridge regressor, a K-Nearest Neighbors regressor, and a Random Forest regressor, each configured with specific hyperparameters. Gradient Boosting Regressor also setup as the meta-learner, which        to learn how to best combine the predictions from the base learners.
   - Created a Stacking Regressor ensemble model, which combines the predictions from specified base & meta-learners.
   - Also experimented with the same process but this time taking the taking the log of y_train with the hope that the log transformation can help the model perform better by giving equal weight to all data points and help to normalize the distribution of the target.
   - The results of the log transformation however produced worse kaggle scores compared to the nominal y_train value. It is likely that transforming y_train distorted the data in a way that impaired the model's ability to learn effectively. 

## Future Improvements
- Neural Networks: explore regularising the parameters in the Keras implementation & use drop out features.
- Ensembles: Experiment with propagating features to the meta-learner in addition the exisitng usage of the weak learners, also utilize selectKBest method to select the top K features that have the highest correlation with target variable.

## Repository Structure
- `E_Monbiot_NN_Project.ipynb`: Jupyter notebook containing the neural network model.
- `E_Monbiot_Ensembles_Project.ipynb`: Jupyter notebook detailing ensemble modeling.




