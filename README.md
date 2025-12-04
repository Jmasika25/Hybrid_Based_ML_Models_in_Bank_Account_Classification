## Problem of Statement

In East Africa, the financial sector has made significant progress towards financial inclusion, encouraging individuals to embrace technological advancements in the financial sector like owning a bank account. However, most of the population in the region do not own bank accounts despite the efforts, but rather prefer the traditional means of managing their income. Determining the factors that contributes to an individual either owning a bank account or not is crucial, as this will enable the financial sector, specifically banks to craft certain solutions that could draw these individuals into the financial system, and thus be bank account holders. Single ML models incapably capture complex, nonlinear relationships among socioeconomic, demographic, and geographic factors influencing financial inclusion. They often overfit or underfit, perform poorly on imbalanced data, and fail to detect variable interactions, limiting their predictive power. There is a need to evaluate hybrid-based ML models that combine multiple algorithms to enhance accuracy, robustness, and interpretability in predicting bank account across East African countries.

## Objectives of Study

1. To determine key factors influencing individual bank account ownership.
2. To compare the predictive power of hybrid-based ML models to single ML models.

## Project Usage

Banks and government agencies can use these insights to design targeted financial-inclusion strategies focused on groups with lower predicted access to bank accounts. For example, individuals with limited education, informal employment, or no cellphone access appear less likely to be financially included. Tailored outreach—such as simplified account-opening processes, mobile-based banking tools, or fee-free starter accounts-can address these barriers and make financial services more accessible to underserved populations.

Policymakers can also leverage the strong influence of job type and education by promoting digital-literacy programs, expanding connectivity, and encouraging employers-especially in the informal sector-to integrate digital payments. Community-based campaigns, partnerships with mobile operators, and support for fintech innovation can further bridge these gaps. By acting on the dominant predictors, institutions can accelerate financial inclusion and build more equitable access to economic opportunities.

Also, the results show that the hybrid model (AUC = 0.854) performs far better than the best single model, XGBoost (AUC = 0.665). This large improvement indicates that combining multiple algorithms captures more complex patterns that a single model cannot learn on its own. Overall, the findings strongly support the study objective by demonstrating that hybrid-based ML models offer significantly higher predictive power for bank-account classification.

## ML Practices Performed
#### Data Cleaning
This involves checking for missing values, duplicates and data types. Practices like removing duplicates, imputing missing values using statistical measures such as mean, median and mode are performed on the data. Also, data types are checked to ensure there is consistency.
#### Descriptive statistics
#### Exploratory Data Analysis
This is the use of visuals to represent the features in the data. Features like location type can be represented using a count plot, Age using a histogram etc. Basically, numerical and continuous variables are represented using either a histogram or a boxplot, numerical but discrete variables are represented using a count plot. Categorical variables are represented using also count plots and bar plots.
#### Feature Engineering and Selection
Here, variable encoding, scaling and selection is performed. Label encoding is conducted on ordinal and/or binary variables, One hot encoding is performed on nominal variables, then a variable importance analysis is performed to select the most important features to be used in a model.

Example of a simple python syntax for encoding and scaling.
```
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler
le = LabelEncoder() # for label encoding
data = le.fit_transform(data)
data_encoded = pd.get_dummies(data, columns = data.columns, drop_first = True).astype(int) # for one hot encoding
scaler = StandardScaler()
data = scaler.fit_transform(data)
```
#### Modeling
The best single ML model after the analysis was the XGBoost Classifier. The evaluation metric used was the ROC-AUC score.

Some of the libraries used included:

```
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
```
## Performance of the best Single ML model

Although the XGB Model's AUC score is weak, showing a weak ability to distingush between an individual having a bank account or not, It performs better than Random Forest Classifier and Logistic Regression.

<img width="846" height="547" alt="7e6c1f93-ce19-444b-8901-dbcf0affc3a5" src="https://github.com/user-attachments/assets/f68685b8-3474-4db9-a450-ce1bac784f4e" />

## Performance of Hybrid-based ML models

### RF+Logistic Regression

<img width="846" height="547" alt="052b8a93-ef0a-4f85-be6f-5b258fab0ccd" src="https://github.com/user-attachments/assets/8d7100d4-f4d0-4756-8d89-9945801a2c28" />

RF and LR as base estimators and LR as the final estimator records an AUC score of ~0.84. This is a great improvement compared to the single models.

An AUC of 0.842 means the hybrid model can correctly distinguish bankers vs. non-bankers about 84% of the time, even under varying classification thresholds. This indicates a strong predictive performance, showing that the model reliably ranks positive cases higher than negative ones.

### SVM+XGBoost

<img width="846" height="547" alt="2afeb305-fa1b-44a8-be8b-1841399f8b6d" src="https://github.com/user-attachments/assets/b14d333b-2a74-41a4-a2f8-8a4c092cf1de" />

The SVM-XGBoost Hybrid model recorded an AUC of 0.854, which is an improvement from the AUC recorded by the RF-Logistic regression hybrid model.
This shows how powerful the combination of  SVM+XGB is in classifying whether an individual has a bank account or not.

## Variable Importance

<img width="790" height="690" alt="a6f6d35d-87f4-4f06-90ae-f995c7f2b53c" src="https://github.com/user-attachments/assets/855a3675-ab09-4b61-8fff-44a6506938f8" />

The feature-importance plot shows that job type (especially formal government employment), cellphone access, and education level are the strongest predictors of whether someone has a bank account. These socioeconomic factors contribute the highest model gain, meaning they significantly improve XGBoost’s ability to classify individuals correctly. Lower-ranked variables such as gender, household size, and marital status still contribute but have much weaker predictive power.





