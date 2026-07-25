# Feature Construction and Feature Splitting in Machine Learning (with Scikit-learn)

Feature engineering is one of the most important steps in machine learning. A famous quote says:

> **"Better features beat better algorithms."**

Feature engineering has two major parts:

1. **Feature Construction (Feature Creation)** → Creating new features from existing ones.
2. **Feature Splitting (Feature Decomposition)** → Breaking one feature into multiple meaningful features.

These techniques improve model performance by exposing hidden information to the algorithm.

---

# 1. Feature Construction

## Definition

Feature construction is the process of **creating new variables (features)** using existing features.

Instead of directly feeding raw data into the model, we create better representations.

For example:

Original Data

| Age | Salary |
| --- | ------ |
| 25  | 40000  |
| 35  | 70000  |

Constructed Feature

| Age | Salary | Salary_per_Age |
| --- | ------ | -------------- |
| 25  | 40000  | 1600           |
| 35  | 70000  | 2000           |

The new feature may capture relationships the model could not easily learn.

---

# Why Feature Construction?

Suppose we are predicting house prices.

Original Features

```
Length = 20 ft
Width = 30 ft
```

Instead of using both separately,

Create

```
Area = Length × Width
```

Area is much more informative.

---

Another example

Predicting loan approval

Original

```
Income
Loan Amount
```

Construct

```
Debt Ratio = Loan Amount / Income
```

This ratio is more useful than either variable alone.

---

# Types of Feature Construction

There are many methods.

---

# A. Mathematical Features

Using arithmetic operations.

Original

```
Height
Weight
```

Construct

```
BMI = Weight / Height²
```

Python

```python
df["BMI"] = df["Weight"] / (df["Height"] ** 2)
```

---

Another example

```python
df["Area"] = df["Length"] * df["Width"]
```

---

# B. Polynomial Features

Sometimes relationships are nonlinear.

Instead of

```
y = ax
```

Maybe

```
y = ax² + bx + c
```

Need squared terms.

Example

Original

```
x
```

Construct

```
x²
```

or

```
x³
```

or

```
x₁x₂
```

---

Example

Original

| x |
| - |
| 2 |
| 3 |

Constructed

| x | x² |
| - | -- |
| 2 | 4  |
| 3 | 9  |

---

## Using sklearn

```python
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(
    degree=2,
    include_bias=False
)

X_poly = poly.fit_transform(X)
```

Example

```python
import pandas as pd

df = pd.DataFrame({
    "x1":[1,2,3],
    "x2":[4,5,6]
})

poly = PolynomialFeatures(degree=2, include_bias=False)

new = poly.fit_transform(df)

print(poly.get_feature_names_out())
```

Output

```
['x1'
 'x2'
 'x1^2'
 'x1 x2'
 'x2^2']
```

Notice interaction terms are automatically generated.

---

Parameters

```
degree=3
```

Creates

```
x
x²
x³
```

---

```
interaction_only=True
```

Creates

```
x1*x2
```

But NOT

```
x1²
```

---

```
include_bias=False
```

Avoids adding a constant column of ones.

---

# C. Interaction Features

Sometimes two variables together are important.

Example

```
Age
Income
```

Construct

```
Age × Income
```

Scikit-learn

```python
PolynomialFeatures(
    degree=2,
    interaction_only=True,
    include_bias=False
)
```

---

# D. Log Transformation

Income often has huge outliers.

Instead of

```
Income
```

Create

```
log(Income)
```

```python
import numpy as np

df["log_income"] = np.log1p(df["Income"])
```

`log1p(x)` computes `log(1 + x)`, making it safe when `x = 0`.

Benefits

* reduces skewness
* handles outliers
* stabilizes variance

---

# E. Date-Time Features

Original

```
2025-06-15
```

Construct

```
Year
Month
Day
Weekday
Quarter
Weekend
```

```python
df["date"] = pd.to_datetime(df["date"])

df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["day"] = df["date"].dt.day
df["weekday"] = df["date"].dt.dayofweek
df["quarter"] = df["date"].dt.quarter
```

---

Another useful feature

```python
df["is_weekend"] = df["weekday"].isin([5,6]).astype(int)
```

---

# F. Domain Knowledge Features

The most powerful features usually come from domain knowledge.

Examples

Healthcare

```
BMI
Heart Risk Score
```

Finance

```
Debt Ratio
Savings Rate
```

Retail

```
Average Purchase
Days Since Last Purchase
```

No library can automatically create these.

---

# 2. Feature Splitting

Feature splitting means dividing one feature into multiple useful features.

---

Example

Original

```
John Smith
```

Split into

```
First Name
Last Name
```

---

Original

```
Kathmandu, Nepal
```

Split

```
City
Country
```

---

Original

```
2025-06-15
```

Split

```
Year
Month
Day
```

---

Original

```
1920x1080
```

Split

```
Width
Height
```

---

# Why Split Features?

Many algorithms cannot understand complex strings.

Example

```
"2025-05-17"
```

Model cannot understand it.

Instead provide

```
Year
Month
Weekday
```

Now patterns become visible.

---

# String Splitting

Example

```python
df = pd.DataFrame({
    "Name":[
        "John Smith",
        "Alice Brown"
    ]
})
```

Split

```python
df[["First","Last"]] = df["Name"].str.split(
    " ",
    expand=True
)
```

Result

| First | Last  |
| ----- | ----- |
| John  | Smith |
| Alice | Brown |

---

# Email Splitting

Original

```
john@gmail.com
```

Split

```
Username
Domain
```

```python
df[["Username","Domain"]] = df["Email"].str.split(
    "@",
    expand=True
)
```

---

# Address Splitting

Original

```
New York, USA
```

Split

```
City
Country
```

```python
df[["City","Country"]] = df["Address"].str.split(
    ",",
    expand=True
)
```

---

# Date Splitting

```python
df["date"] = pd.to_datetime(df["date"])

df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["day"] = df["date"].dt.day
```

---

# URL Splitting

Original

```
https://example.com/blog/page
```

Can extract

```
Protocol
Domain
Path
```

Using Python

```python
from urllib.parse import urlparse

parsed = urlparse("https://example.com/blog/page")

print(parsed.scheme)
print(parsed.netloc)
print(parsed.path)
```

---

# Using FunctionTransformer in Scikit-learn

Scikit-learn doesn't have dedicated transformers for arbitrary feature construction or string splitting, but you can build custom transformations with `FunctionTransformer`.

Example: Add BMI

```python
from sklearn.preprocessing import FunctionTransformer

def add_bmi(X):
    X = X.copy()
    X["BMI"] = X["Weight"] / (X["Height"] ** 2)
    return X

transformer = FunctionTransformer(add_bmi)

X_new = transformer.fit_transform(df)
```

---

# Using ColumnTransformer

Different columns often need different preprocessing.

Example

```
Age
Salary
Date
City
```

Pipeline

```
Numeric
    ↓
Scaling

Date
    ↓
Split into Year Month Day

City
    ↓
One-Hot Encoding
```

Example

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), ["Age", "Salary"]),
    ("cat", OneHotEncoder(handle_unknown="ignore"), ["City"])
])
```

To include custom feature construction (such as extracting date parts), you can create a custom transformer (using `FunctionTransformer` or by implementing `BaseEstimator` and `TransformerMixin`) and place it inside a `Pipeline` before the `ColumnTransformer`.

---

# Example Workflow

Suppose you have this dataset:

| Age | Salary | Date       | Email                                   |
| --- | ------ | ---------- | --------------------------------------- |
| 25  | 50000  | 2025-03-15 | [john@gmail.com](mailto:john@gmail.com) |

### Step 1: Feature Construction

Create

```python
df["Salary_per_Age"] = df["Salary"] / df["Age"]
```

Result

```
Age
Salary
Salary_per_Age
```

---

### Step 2: Feature Splitting

Split date

```python
df["Date"] = pd.to_datetime(df["Date"])

df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["Day"] = df["Date"].dt.day
```

Split email

```python
df[["Username","Domain"]] = (
    df["Email"]
    .str.split("@", expand=True)
)
```

Final dataset

| Age | Salary | Salary_per_Age | Year | Month | Day | Username | Domain    |
| --- | ------ | -------------- | ---- | ----- | --- | -------- | --------- |
| 25  | 50000  | 2000           | 2025 | 3     | 15  | john     | gmail.com |

---

# Best Practices

* Construct features only when they have a meaningful relationship to the target.
* Use domain knowledge whenever possible—domain-specific features often outperform generic transformations.
* Avoid creating too many polynomial features, as they can dramatically increase dimensionality and lead to overfitting.
* Apply the same feature construction and splitting steps to both training and test data. The easiest way to ensure this is by using `Pipeline` and custom transformers.
* Handle missing values before performing operations that require valid inputs (for example, extracting date parts or taking logarithms).

---

# Summary

| Aspect             | Feature Construction                                                         | Feature Splitting                                              |
| ------------------ | ---------------------------------------------------------------------------- | -------------------------------------------------------------- |
| Purpose            | Create new informative features                                              | Break one feature into multiple features                       |
| Input              | One or more existing features                                                | One complex feature                                            |
| Output             | New derived columns                                                          | Multiple simpler columns                                       |
| Examples           | BMI, Area, Polynomial Features, Log Transform, Interaction Terms             | Year/Month/Day, First/Last Name, Username/Domain, City/Country |
| Scikit-learn Tools | `PolynomialFeatures`, `FunctionTransformer`, custom transformers, `Pipeline` | `FunctionTransformer`, custom transformers, `Pipeline`         |
| Benefit            | Captures hidden relationships and nonlinearities                             | Makes complex data understandable for ML algorithms            |

In modern Scikit-learn workflows, **`Pipeline` + `ColumnTransformer` + custom transformers** is the recommended approach because it keeps preprocessing reproducible, prevents data leakage, and ensures that the exact same transformations are applied during both training and inference.
