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
<img width="580" height="502" alt="1012fda5-1023-4c8e-bd89-493ef738cffd" src="https://github.com/user-attachments/assets/7cfb7456-f343-447e-8d86-8d4e4abbce9d" />
Across the four countries, individuals without formal bank accounts consistently outnumber those who are banked. Rwanda has the highest proportion of unbanked people, followed by Tanzania, then Kenya, and lastly Uganda, reflecting varying levels of financial inclusion across the region. Kenya has the highest number of individuals with bank accounts among the four countries, demonstrating relatively stronger financial inclusion. It is followed by Rwanda, then Tanzania, and finally Uganda, which has the lowest share of banked individuals.
5. Feature Engineering and Selection; Here, variable encoding, scaling and selection is performed. Label encoding is conducted on ordinal and/or binary variables, One hot encoding is performed on nominal variables, then a variable importance analysis is performed to select the most important features to be used in a model.
```
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler
le = LabelEncoder() # for label encoding
data = le.fit_transform(data)
data_encoded = pd.get_dummies(data, columns = data.columns, drop_first = True).astype(int) # for one hot encoding
```
5. Feature Scaling
```
scaler = StandardScaler()
data = scaler.fit_transform(data)
```


