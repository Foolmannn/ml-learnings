# Handling Missing Data in Machine Learning (Scikit-learn) - Complete Guide

Missing data is one of the most common problems in real-world datasets. If not handled properly, it can reduce model accuracy or even prevent algorithms from training.

---

# What is Missing Data?

Missing data refers to values that are unavailable for some observations.

Example dataset:

| Age | Salary | City      | Purchased |
| --- | ------ | --------- | --------- |
| 25  | 50000  | Kathmandu | Yes       |
| 30  | NaN    | Pokhara   | No        |
| NaN | 65000  | Butwal    | Yes       |
| 40  | 80000  | NaN       | No        |

Here,

* Age has one missing value
* Salary has one missing value
* City has one missing value

In pandas,

```python
import numpy as np

df = pd.DataFrame({
    "Age":[25,30,np.nan,40],
    "Salary":[50000,np.nan,65000,80000],
    "City":["Kathmandu","Pokhara","Butwal",np.nan]
})
```

---

# Why Missing Data Matters

Many ML algorithms cannot work with NaN values.

Examples:

* Logistic Regression ❌
* Linear Regression ❌
* SVM ❌
* KNN ❌

Some algorithms can naturally handle missing values:

* HistGradientBoosting
* XGBoost
* LightGBM
* CatBoost

Still, handling missing values before training is generally a good practice.

---

# Types of Missing Data

Understanding *why* data is missing helps choose the best strategy.

## 1. MCAR (Missing Completely At Random)

Missingness has no relationship with any variable.

Example:

A survey page accidentally failed to load.

```
Age      Salary
25       50000
NaN      65000
30       70000
```

The missing age happened randomly.

Good news:

Deleting rows often doesn't introduce bias.

---

## 2. MAR (Missing At Random)

Missingness depends on another observed variable.

Example:

Women are less likely to disclose salary.

```
Gender    Salary
Male      60000
Female    NaN
Female    NaN
Male      70000
```

Salary is missing because of Gender.

---

## 3. MNAR (Missing Not At Random)

Missingness depends on the missing value itself.

Example:

Rich people refuse to disclose income.

```
Salary
40000
50000
NaN
NaN
```

Higher salaries are more likely to be missing.

This is the hardest type to handle.

---

# Detecting Missing Values

## Count missing values

```python
df.isnull().sum()
```

Output

```
Age       1
Salary    1
City      1
```

---

## Percentage of missing values

```python
(df.isnull().mean())*100
```

Output

```
Age       25%
Salary    25%
City      25%
```

---

## Visualizing missing values

Using seaborn

```python
import seaborn as sns

sns.heatmap(df.isnull(), cbar=False)
```

Dark cells indicate missing values.

---

# Approaches for Handling Missing Data

There are two main approaches:

1. Delete missing data
2. Fill (Impute) missing data

---

# Method 1: Delete Missing Data

## Remove Rows

```python
df.dropna()
```

Before

| Age | Salary |
| --- | ------ |
| 25  | 50000  |
| 30  | NaN    |
| NaN | 70000  |

After

| Age | Salary |
| --- | ------ |
| 25  | 50000  |

### Advantages

* Very simple
* No assumptions

### Disadvantages

* Lose information
* Bad when dataset is small

Use when

* Less than 5% missing values
* Large dataset
* MCAR

---

## Remove Columns

```python
df.dropna(axis=1)
```

Removes columns containing missing values.

Useful when

A feature has too many missing values.

Example

```
Address
95% missing
```

Removing may be better than imputing.

---

# Method 2: Fill Missing Values (Imputation)

Imputation means replacing missing values with estimated values.

Scikit-learn provides several imputers.

---

# SimpleImputer

```python
from sklearn.impute import SimpleImputer
```

Most commonly used.

---

# Mean Imputation

Suitable for numerical data.

Example

```
Age

20
30
40
NaN
50
```

Mean

```
(20+30+40+50)/4 = 35
```

Replace NaN with 35.

```python
from sklearn.impute import SimpleImputer

imp = SimpleImputer(strategy="mean")

df["Age"] = imp.fit_transform(df[["Age"]])
```

Why `df[["Age"]]`?

Because sklearn expects a 2D input.

---

# Median Imputation

Better for skewed distributions.

Example

```
Age

20
25
30
35
200
NaN
```

Mean

```
62
```

Median

```
30
```

Median is better because of the outlier.

```python
imp = SimpleImputer(strategy="median")

df["Age"] = imp.fit_transform(df[["Age"]])
```

Use median when

* Outliers exist
* Data is skewed

---

# Most Frequent Imputation

Used for categorical variables.

Example

```
City

Pokhara
Kathmandu
Pokhara
NaN
```

Fill with

```
Pokhara
```

```python
imp = SimpleImputer(strategy="most_frequent")

df["City"] = imp.fit_transform(df[["City"]])
```

---

# Constant Value Imputation

Replace missing values with a fixed value.

Example

```
City

Kathmandu
NaN
Pokhara
```

Replace NaN with

```
Unknown
```

```python
imp = SimpleImputer(
    strategy="constant",
    fill_value="Unknown"
)

df["City"] = imp.fit_transform(df[["City"]])
```

For numerical data

```python
SimpleImputer(
    strategy="constant",
    fill_value=0
)
```

---

# Complete Example

```python
import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer

df = pd.DataFrame({
    "Age":[25,np.nan,35,40],
    "Salary":[40000,50000,np.nan,70000],
    "City":["A","B",np.nan,"A"]
})

num_imp = SimpleImputer(strategy="median")

cat_imp = SimpleImputer(strategy="most_frequent")

df[["Age","Salary"]] = num_imp.fit_transform(
    df[["Age","Salary"]]
)

df[["City"]] = cat_imp.fit_transform(
    df[["City"]]
)

print(df)
```

---

# Why Use `fit_transform()`?

During training

```python
imputer.fit(X_train)
```

Calculates statistics.

Example

```
Mean Age = 32
Median Salary = 45000
```

Then

```python
imputer.transform(X_train)
```

fills missing values.

`fit_transform()` does both.

---

# Train-Test Split Rule

**Always split your data before fitting an imputer.**

❌ Wrong

```python
imputer.fit(df)
```

Then split.

This leaks information from the test set.

✅ Correct

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

imputer.fit(X_train)

X_train = imputer.transform(X_train)

X_test = imputer.transform(X_test)
```

Notice that the imputer is **fit only on the training data**, then reused to transform both the training and test sets. This prevents **data leakage**.

---

# KNN Imputation

Instead of replacing with the mean or median, estimate the missing value using the nearest samples.

```python
from sklearn.impute import KNNImputer

imputer = KNNImputer(
    n_neighbors=5
)

X = imputer.fit_transform(X)
```

Example

| Height | Weight |
| ------ | ------ |
| 170    | 70     |
| 175    | 75     |
| 172    | NaN    |

The missing weight is estimated from the nearest neighbors.

### Advantages

* More accurate than mean/median in many cases.
* Preserves relationships between features.

### Disadvantages

* Computationally expensive.
* Sensitive to feature scaling.

---

# Iterative Imputation (MICE)

Missing values are predicted using the other features.

Example

```
Salary = f(Age, Experience, Education)
```

Scikit-learn provides `IterativeImputer` (currently experimental).

```python
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer

imp = IterativeImputer(random_state=42)

X = imp.fit_transform(X)
```

The imputer repeatedly models each feature with missing values as a function of the other features until the estimates stabilize.

### Advantages

* Often more accurate than simple imputation.
* Captures relationships between variables.

### Disadvantages

* Slower than `SimpleImputer`.
* More complex to tune.

---

# MissingIndicator

Sometimes, the fact that a value is missing is itself informative.

Example

| Income |
| ------ |
| 50000  |
| NaN    |
| 70000  |

Create an additional feature:

| Income | Income_missing |
| ------ | -------------- |
| 50000  | 0              |
| NaN    | 1              |
| 70000  | 0              |

This allows the model to learn whether missingness carries predictive information.

---

# Using Pipelines

The recommended way to preprocess data is with a pipeline.

```python
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("model", LogisticRegression())
])

pipe.fit(X_train, y_train)
```

Benefits:

* Prevents data leakage.
* Keeps preprocessing and modeling together.
* Easier cross-validation and deployment.

---

# Different Strategies for Different Column Types

A `ColumnTransformer` lets you apply different imputers to different feature types.

```python
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

numeric_features = ["Age", "Salary"]
categorical_features = ["City"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", SimpleImputer(strategy="median"), numeric_features),
        ("cat", SimpleImputer(strategy="most_frequent"), categorical_features),
    ]
)
```

This is the preferred approach for mixed-type datasets.

---

# Which Strategy Should You Choose?

| Situation                              | Recommended Strategy                                  |
| -------------------------------------- | ----------------------------------------------------- |
| < 5% missing (MCAR)                    | Drop rows or simple imputation                        |
| Numerical, symmetric distribution      | Mean imputation                                       |
| Numerical, skewed or with outliers     | Median imputation                                     |
| Categorical                            | Most frequent or constant ("Unknown")                 |
| Need to preserve feature relationships | KNN or Iterative imputation                           |
| Feature has > 50–70% missing values    | Consider dropping the feature (depends on importance) |
| Missingness itself may be informative  | Add a missing indicator                               |

---

# Best Practices

* Split into training and test sets **before** fitting an imputer.
* Fit imputers **only on the training data**.
* Use `Pipeline` and `ColumnTransformer` for clean, leak-free workflows.
* Use **median** instead of mean when data contains outliers.
* Consider **KNN** or **IterativeImputer** when simple methods are not sufficient.
* Investigate **why** values are missing (MCAR, MAR, or MNAR); the cause often determines the best handling strategy.
* Evaluate different imputation methods with cross-validation—there is no universally best approach.

---

# Summary

| Imputer                                   | Best For                        | Pros                                  | Cons                                                |
| ----------------------------------------- | ------------------------------- | ------------------------------------- | --------------------------------------------------- |
| `SimpleImputer(strategy="mean")`          | Symmetric numerical data        | Fast, simple                          | Sensitive to outliers                               |
| `SimpleImputer(strategy="median")`        | Skewed numerical data           | Robust to outliers                    | Ignores relationships between features              |
| `SimpleImputer(strategy="most_frequent")` | Categorical data                | Easy to use                           | Can increase the frequency of the dominant category |
| `SimpleImputer(strategy="constant")`      | Any type                        | Explicitly marks missing values       | Choice of constant may affect the model             |
| `KNNImputer`                              | Numerical data                  | Uses similar samples                  | Slower, affected by feature scaling                 |
| `IterativeImputer`                        | Numerical data                  | Models relationships between features | Computationally intensive, experimental             |
| `MissingIndicator`                        | When missingness is informative | Adds useful signal                    | Increases feature count                             |

In practice, a common and effective scikit-learn preprocessing pipeline is:

1. Split the data into training and test sets.
2. Use a `ColumnTransformer` to apply **median imputation** to numerical features and **most frequent** (or **constant**) imputation to categorical features.
3. Add encoding and scaling if needed.
4. Train the model using a `Pipeline` to ensure preprocessing is applied consistently and without data leakage.
