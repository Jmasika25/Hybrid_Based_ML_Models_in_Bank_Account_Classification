# Hybrid_Based_ML_Models_in_Bank_Account_Classification

## Problem of Statement

In East Africa, the financial sector has made significant progress towards financial inclusion, encouraging individuals to embrace technological advancements in the financial sector like owning a bank account. However, most of the population in the region do not own bank accounts despite the efforts, but rather prefer the traditional means of managing their income. Determining the factors that contributes to an individual either owning a bank account or not is crucial, as this will enable the financial sector, specifically banks to craft certain solutions that could draw these individuals into the financial system, and thus be bank account holders. Single ML models incapably capture complex, nonlinear relationships among socioeconomic, demographic, and geographic factors influencing financial inclusion. They often overfit or underfit, perform poorly on imbalanced data, and fail to detect variable interactions, limiting their predictive power. There is a need to evaluate hybrid-based ML models that combine multiple algorithms to enhance accuracy, robustness, and interpretability in predicting bank account across East African countries.

## Objectives of Study

1. To determine key factors influencing individual bank account ownership.
2. To compare the predictive power of hybrid-based ML models to single ML models.

## ML Practices Performed
1. Data Cleaning; This involves checking for missing values, duplicates and data types. Practices like removing duplicates, imputing missing values using statistical measures such as mean, median and mode are performed on the data. Also, data types are checked to ensure there is consistency.
2. Descriptive statistics
3. Exploratory Data Analysis; this is the use of visuals to represent the features in the data. Features like location type can be represented using a count plot, Age using a histogram etc. Basically, numerical and continuous variables are represented using either a histogram or a boxplot, numerical but discrete variables are represented using a count plot. Categorical variables are represented using also count plots and bar plots.
An example of a plot from the analysis is as below.

<img width="580" height="502" alt="1012fda5-1023-4c8e-bd89-493ef738cffd" src="https://github.com/user-attachments/assets/7cfb7456-f343-447e-8d86-8d4e4abbce9d"/>

Across the four countries, individuals without formal bank accounts consistently outnumber those who are banked. Rwanda has the highest proportion of unbanked people, followed by Tanzania, then Kenya, and lastly Uganda, reflecting varying levels of financial inclusion across the region. Kenya has the highest number of individuals with bank accounts among the four countries, demonstrating relatively stronger financial inclusion. It is followed by Rwanda, then Tanzania, and finally Uganda, which has the lowest share of banked individuals.
5. Feature Engineering and Selection; Here, variable encoding, scaling and selection is performed. Label encoding is conducted on ordinal and/or binary variables, One hot encoding is performed on nominal variables, then a variable importance analysis is performed to select the most important features to be used in a model.

```
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler
le = LabelEncoder() # for label encoding
data = le.fit_transform(data)
data_encoded = pd.get_dummies(data, columns = data.columns, drop_first = True).astype(int) # for one hot encoding
scaler = StandardScaler()
data = scaler.fit_transform(data)
```
6. Modeling
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

<img width="846" height="547" alt="7e6c1f93-ce19-444b-8901-dbcf0affc3a5" src="https://github.com/user-attachments/assets/f68685b8-3474-4db9-a450-ce1bac784f4e" />

Although the XGB Model's AUC score is weak, showing a weak ability to distingush between an individual having a bank account or not, It performs better than Random Forest Classifier and Logistic Regression.

The best Hybrid-based ML model after analysis was the SVM-XGBoost. 

The SVM-XGBoost Hybrid model recorded an AUC of 0.854, which is an improvement from the AUC recorded by the RF-Logistic regression hybrid model.
This shows how powerful the combination of  SVM+XGB is in classifying whether an individual has a bank account or not.

<img width="846" height="547" alt="2afeb305-fa1b-44a8-be8b-1841399f8b6d" src="https://github.com/user-attachments/assets/b14d333b-2a74-41a4-a2f8-8a4c092cf1de" />

7. Variable Importance

<img width="790" height="690" alt="a6f6d35d-87f4-4f06-90ae-f995c7f2b53c" src="https://github.com/user-attachments/assets/855a3675-ab09-4b61-8fff-44a6506938f8" />

From the Variable Importance plot, the features that were of very much importance during modeling were employment type, job type, education level, cellphone access and relationship with head. Marital Status, Location type, household size, gender and age were of less significance during modeling. They contributed less to the prediction of whether an individual has a bank account or not.


