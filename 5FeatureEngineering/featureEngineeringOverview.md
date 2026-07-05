# Feature Engineering in Machine Learning (Detailed Guide)

Feature engineering is the process of **creating, transforming, selecting, and preparing input variables (features)** so that machine learning models can learn patterns more effectively.

It is often said:

> **Better features > More complex models**

A simple model with well-engineered features often outperforms a complex model trained on poor features.

---

# What is a Feature?

A **feature** is an input variable used by the model.

Example:

Suppose we're predicting house prices.

| Area | Bedrooms | Age | Distance to City | Price |
| ---- | -------- | --- | ---------------- | ----- |
| 1500 | 3        | 5   | 2 km             | $250k |
| 2200 | 4        | 8   | 5 km             | $340k |

Features:

* Area
* Bedrooms
* Age
* Distance

Target:

* Price

---

# Why Feature Engineering is Important

Real-world data is rarely clean.

Problems include:

* Missing values
* Wrong scales
* Dates stored as strings
* Categories stored as text
* Outliers
* Duplicate information
* Noise

Feature engineering converts raw data into a form suitable for ML algorithms.

Example

Instead of

```
Date = 2026-07-05
```

Create

```
Year = 2026
Month = 7
Day = 5
Weekday = Sunday
Quarter = Q3
```

The model can now learn seasonal patterns.

---

# Complete Feature Engineering Pipeline

```
Raw Data
     │
Cleaning
     │
Handling Missing Values
     │
Encoding Categories
     │
Scaling
     │
Feature Creation
     │
Feature Transformation
     │
Feature Selection
     │
Final Dataset
     │
Machine Learning Model
```

---

# Step 1: Handling Missing Values

Suppose

| Age | Salary |
| --- | ------ |
| 23  | 50000  |
| 25  | NaN    |
| 30  | 65000  |
| NaN | 70000  |

## Option 1: Remove rows

```python
df.dropna()
```

Good when very few rows are missing.

---

## Option 2: Fill with Mean

```python
df["Age"].fillna(df["Age"].mean(), inplace=True)
```

Example

```
23
25
30
NaN
```

Mean

```
(23+25+30)/3 = 26
```

Replace

```
26
```

---

## Option 3: Fill with Median

Useful for skewed data.

```python
df["Salary"].fillna(df["Salary"].median(), inplace=True)
```

---

## Option 4: Fill with Mode

For categorical features

```python
df["City"].fillna(df["City"].mode()[0])
```

---

# Step 2: Encoding Categorical Variables

Models cannot understand text.

Example

| Color |
| ----- |
| Red   |
| Blue  |
| Green |

Need numerical values.

---

## Label Encoding

```
Red → 0
Blue → 1
Green → 2
```

```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

df["Color"] = le.fit_transform(df["Color"])
```

Best for **ordinal** categories.

Example

```
Small
Medium
Large
```

---

## One-Hot Encoding

Original

| Color |
| ----- |
| Red   |
| Blue  |
| Green |

Becomes

| Red | Blue | Green |
| --- | ---- | ----- |
| 1   | 0    | 0     |
| 0   | 1    | 0     |
| 0   | 0    | 1     |

```python
pd.get_dummies(df, columns=["Color"])
```

Best for **nominal** categories.

---

## Ordinal Encoding

```
Poor = 1
Average = 2
Good = 3
Excellent = 4
```

```python
mapping = {
    "Poor":1,
    "Average":2,
    "Good":3,
    "Excellent":4
}

df["Quality"] = df["Quality"].map(mapping)
```

---

# Step 3: Feature Scaling

Some algorithms depend heavily on feature magnitudes.

Example

| Age | Income |
| --- | ------ |
| 20  | 500000 |
| 30  | 700000 |

Income dominates because its scale is much larger.

---

## Standardization

Formula

[
z = \frac{x-\mu}{\sigma}
]

Mean becomes 0.

Standard deviation becomes 1.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)
```

Used in

* Linear Regression
* Logistic Regression
* SVM
* Neural Networks
* PCA
* KNN

---

## Min-Max Scaling

Formula

[
\frac{x-min}{max-min}
]

Range

```
0 → 1
```

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
```

Useful for deep learning and image data.

---

## Robust Scaling

Uses

* Median
* Interquartile Range (IQR)

Less affected by outliers.

```python
from sklearn.preprocessing import RobustScaler
```

---

# Step 4: Feature Transformation

Sometimes data isn't normally distributed.

Example Salary

```
1000
2000
3000
500000
```

Very skewed.

---

## Log Transformation

```python
import numpy as np

df["Salary"] = np.log1p(df["Salary"])
```

Turns

```
1000
10000
100000
```

into

```
6.9
9.2
11.5
```

The distribution becomes more balanced.

---

## Square Root

```python
np.sqrt(df["Area"])
```

---

## Box-Cox

Requires positive values.

```python
from scipy.stats import boxcox

df["Salary"], _ = boxcox(df["Salary"])
```

---

## Yeo-Johnson

Works for positive and negative values.

```python
from sklearn.preprocessing import PowerTransformer

pt = PowerTransformer(method="yeo-johnson")
```

---

# Step 5: Feature Creation

One of the most important parts of feature engineering.

Create new features from existing ones.

---

## Example 1

Instead of

```
Weight
Height
```

Create

```
BMI
```

Formula

```
BMI = weight / height²
```

---

## Example 2

House data

Instead of

```
Bedrooms
Bathrooms
```

Create

```
Bathrooms per bedroom
```

---

## Example 3

Date

```
2026-07-05
```

Create

```
Year
Month
Day
Weekend
Quarter
Season
Holiday
```

```python
df["Date"] = pd.to_datetime(df["Date"])

df["Month"] = df["Date"].dt.month
df["Day"] = df["Date"].dt.day
df["Weekday"] = df["Date"].dt.day_name()
```

---

## Example 4

Age

```
DOB
```

Create

```
Current Age
```

---

## Example 5

Shopping

```
Total Spend
```

Create

```
Average Purchase Value
Purchase Frequency
```

---

# Step 6: Handling Outliers

Example

```
20
22
24
25
1000
```

1000 is an outlier.

---

## IQR Method

```python
Q1 = df["Age"].quantile(0.25)
Q3 = df["Age"].quantile(0.75)

IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR
```

Filter

```python
df = df[(df["Age"] >= lower) & (df["Age"] <= upper)]
```

---

## Z-score

```python
from scipy.stats import zscore

df = df[(zscore(df["Age"]) < 3)]
```

---

# Step 7: Binning

Convert continuous values into categories.

Age

```
18
23
35
60
```

Becomes

```
Young
Adult
Middle-aged
Senior
```

```python
bins = [0,18,35,60,100]

labels = ["Child","Young","Adult","Senior"]

df["AgeGroup"] = pd.cut(df["Age"], bins=bins, labels=labels)
```

---

# Step 8: Polynomial Features

Suppose

```
Area
```

Create

```
Area²
Area³
```

Useful for nonlinear relationships.

```python
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2)
```

Creates

```
x

x²

x1*x2
```

---

# Step 9: Interaction Features

Example

```
Age
Income
```

Create

```
Age × Income
```

Sometimes the interaction between features is more predictive than either feature alone.

```python
df["AgeIncome"] = df["Age"] * df["Income"]
```

---

# Step 10: Feature Selection

Remove irrelevant features.

Benefits

* Faster training
* Less overfitting
* Better accuracy
* Easier interpretation

---

## Filter Methods

Correlation

```python
corr = df.corr(numeric_only=True)
```

Remove highly correlated features.

---

## Chi-Square Test

```python
from sklearn.feature_selection import chi2
```

For categorical features.

---

## Mutual Information

```python
from sklearn.feature_selection import mutual_info_classif
```

Captures nonlinear relationships.

---

## Wrapper Methods

Recursive Feature Elimination (RFE)

```python
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression

model = LinearRegression()

selector = RFE(model, n_features_to_select=5)

selector.fit(X, y)
```

---

## Embedded Methods

Some models perform feature selection automatically.

Examples

* Lasso Regression (L1 regularization)
* Decision Trees
* Random Forests
* Gradient Boosting models

---

# Domain-Specific Feature Engineering Examples

## 1. House Price Prediction

Original features:

* Area
* Bedrooms
* Bathrooms
* Year Built

Engineered features:

* House Age = Current Year − Year Built
* Rooms per Area = (Bedrooms + Bathrooms) / Area
* Luxury Flag = Area > 3000
* Price per Square Foot (if historical prices are available)

---

## 2. Titanic Survival Prediction

Original:

* Name
* Age
* Ticket
* Cabin

Engineered:

* Family Size = SibSp + Parch + 1
* Is Alone = Family Size == 1
* Title extracted from Name (Mr., Mrs., Miss., Dr., etc.)
* Cabin Known = Cabin is not null
* Ticket Group Size = Number of passengers sharing a ticket

---

## 3. Credit Card Fraud Detection

Original:

* Transaction Time
* Amount
* Merchant

Engineered:

* Hour of Day
* Day of Week
* Number of transactions in the last hour
* Average spending over the last 7 days
* Distance from previous transaction location
* Merchant risk score

---

## 4. Movie Recommendation

Original:

* Genres
* Cast
* Release Date

Engineered:

* Number of Genres
* Release Year
* Director popularity
* Average actor rating
* Genre combinations (e.g., Action + Comedy)

---

# Complete Example Using Pandas and Scikit-learn

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

# Example data
df = pd.DataFrame({
    "Age": [22, 38, None, 35],
    "Fare": [7.25, 71.83, 8.05, None],
    "Sex": ["male", "female", "female", "male"],
    "Survived": [0, 1, 1, 0]
})

X = df.drop("Survived", axis=1)
y = df["Survived"]

numeric_features = ["Age", "Fare"]
categorical_features = ["Sex"]

numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression())
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

model.fit(X_train, y_train)

print(model.score(X_test, y_test))
```

This pipeline automatically imputes missing values, scales numerical features, one-hot encodes categorical features, and trains a logistic regression model. The same preprocessing is consistently applied during prediction, reducing the risk of data leakage.

# Best Practices

* Understand the data before creating features; domain knowledge often leads to the most useful features.
* Split the dataset into training and testing sets **before** fitting preprocessing steps like scalers or imputers to avoid data leakage.
* Use pipelines to combine preprocessing and model training, ensuring consistency between training and inference.
* Evaluate whether engineered features improve performance using cross-validation rather than assuming they help.
* Remove redundant or highly correlated features when appropriate to simplify models.
* Document every transformation so your preprocessing can be reproduced in production.

# Summary

| Technique                | Purpose                            | Example                        |
| ------------------------ | ---------------------------------- | ------------------------------ |
| Missing Value Imputation | Handle incomplete data             | Fill age with median           |
| Encoding                 | Convert categories to numbers      | One-hot encoding for city      |
| Scaling                  | Normalize feature ranges           | StandardScaler                 |
| Transformation           | Reduce skewness                    | Log transform income           |
| Feature Creation         | Add informative variables          | BMI, house age                 |
| Outlier Handling         | Reduce influence of extreme values | IQR filtering                  |
| Binning                  | Group continuous values            | Age groups                     |
| Polynomial Features      | Capture nonlinear patterns         | (x^2), (x_1x_2)                |
| Interaction Features     | Model combined effects             | Age × Income                   |
| Feature Selection        | Keep only useful features          | RFE, Lasso, Mutual Information |

Feature engineering is an iterative process. You start with basic preprocessing, create features based on data understanding and domain knowledge, evaluate their impact on validation performance, and refine the feature set until you achieve a good balance between predictive power, simplicity, and generalization.
