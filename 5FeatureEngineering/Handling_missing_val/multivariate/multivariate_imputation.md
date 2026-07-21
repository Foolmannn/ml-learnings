# Handling Missing Values using Multivariate Imputation (Machine Learning)

Multivariate imputation is one of the most advanced techniques for handling missing values. Unlike simple methods (mean, median, mode), it **uses relationships between multiple features** to estimate missing values.

---

# Why do we need Multivariate Imputation?

Suppose we have this dataset:

| Age | Salary | Experience |
| --- | ------ | ---------- |
| 25  | 40,000 | 2          |
| 30  | 55,000 | 5          |
| 35  | 70,000 | 8          |
| 40  | ?      | 12         |
| 45  | 95,000 | 15         |

Mean imputation would simply replace Salary with the average salary.

But notice:

* Higher age → Higher salary
* More experience → Higher salary

Instead of ignoring these relationships, multivariate imputation uses them.

It might estimate

Salary = **82,000**

instead of the mean (65,000).

This usually produces much better results.

---

# What is Multivariate Imputation?

It predicts missing values using **all the remaining features**.

Instead of

```
Salary = Mean(Salary)
```

it learns

```
Salary = f(Age, Experience, Education, City, ...)
```

where **f** is a machine learning model.

---

# Types of Multivariate Imputation

The most common methods are

1. Iterative Imputer (MICE)
2. KNN Imputer
3. Regression Imputation
4. Random Forest Imputation (MissForest)

---

# 1. Iterative Imputer (MICE)

This is the most widely used multivariate method.

MICE means

> Multiple Imputation by Chained Equations

Scikit-learn implements a simplified version called

```
IterativeImputer
```

---

## Idea

Suppose we have

| Age | Salary | Experience |
| --- | ------ | ---------- |
| 25  | 40000  | 2          |
| 30  | 55000  | 5          |
| ?   | 70000  | 8          |
| 40  | 85000  | 12         |
| 45  | ?      | 15         |

Both Age and Salary have missing values.

The algorithm predicts each missing column one after another.

---

## Step 1

Initialize missing values.

Usually with

* mean
* median

Dataset becomes

| Age | Salary | Experience |
| --- | ------ | ---------- |
| 25  | 40000  | 2          |
| 30  | 55000  | 5          |
| 35  | 70000  | 8          |
| 40  | 85000  | 12         |
| 45  | 62500  | 15         |

---

## Step 2

Choose one feature with missing values.

Example

Predict **Age**

Target

```
Age
```

Features

```
Salary
Experience
```

Train a regression model

```
Age = f(Salary, Experience)
```

Replace missing Age.

---

## Step 3

Now predict Salary.

Target

```
Salary
```

Features

```
Age
Experience
```

Replace missing Salary.

---

## Step 4

Repeat again and again until values stop changing.

This is why it is called **Iterative Imputation**.

---

# Visualization

```
Missing Values

Age
 ↓
Predict using
Salary + Experience

↓

Update dataset

↓

Salary
 ↓
Predict using
Age + Experience

↓

Update dataset

↓

Repeat
```

---

# Mathematics

Suppose

```
X1 = Age
X2 = Salary
X3 = Experience
```

Missing Salary

Regression model

```
Salary = β0
       + β1(Age)
       + β2(Experience)
```

The predicted salary replaces the missing value.

---

# Python Example

```python
import pandas as pd

data = {
    'Age':[25,30,None,40,45],
    'Salary':[40000,55000,70000,85000,None],
    'Experience':[2,5,8,12,15]
}

df = pd.DataFrame(data)
```

---

## Import IterativeImputer

It is experimental.

```python
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
```

---

## Create imputer

```python
imputer = IterativeImputer(random_state=42)
```

---

## Fit and Transform

```python
df_imputed = imputer.fit_transform(df)
```

Convert back to DataFrame

```python
df_imputed = pd.DataFrame(
    df_imputed,
    columns=df.columns
)

print(df_imputed)
```

Output might be

```
Age   Salary  Experience

25    40000     2
30    55000     5
35.2  70000     8
40    85000    12
45    97900    15
```

Notice

Neither Age nor Salary is replaced by the mean.

---

# Important Parameters

```python
IterativeImputer(
    estimator=None,
    max_iter=10,
    random_state=42,
    initial_strategy='mean'
)
```

### estimator

Model used for prediction.

Default

```
BayesianRidge()
```

Can also use

```python
RandomForestRegressor()
```

or

```python
ExtraTreesRegressor()
```

Example

```python
from sklearn.ensemble import RandomForestRegressor

imputer = IterativeImputer(
    estimator=RandomForestRegressor(),
    random_state=42
)
```

---

### max_iter

Number of iterations.

Default

```
10
```

Higher

```
20
30
```

may improve results but increases computation.

---

### initial_strategy

Initial guess before iterations.

Options

```
mean
median
most_frequent
constant
```

---

### random_state

For reproducibility.

---

# Example with Train/Test Split

Never fit on the entire dataset before splitting.

Correct approach:

```python
from sklearn.model_selection import train_test_split

X_train, X_test = train_test_split(df, test_size=0.2, random_state=42)

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

imputer = IterativeImputer(random_state=42)

X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)
```

Always use:

* `fit_transform()` on training data
* `transform()` on test data

This prevents data leakage.

---

# 2. KNN Imputer

Instead of regression, it finds the **K nearest rows**.

Suppose

| Age | Salary | Experience |
| --- | ------ | ---------- |
| 25  | 40k    | 2          |
| 30  | 55k    | 5          |
| 35  | ?      | 7          |
| 36  | 71k    | 8          |
| 40  | 80k    | 12         |

Missing salary.

The nearest rows are

```
30
36
40
```

Average

```
(55000+71000+80000)/3
```

Used as the new salary.

---

Python

```python
from sklearn.impute import KNNImputer

imputer = KNNImputer(n_neighbors=3)

X = imputer.fit_transform(df)
```

---

## Important Parameters

```python
KNNImputer(
    n_neighbors=5,
    weights='uniform'
)
```

`n_neighbors`

Number of neighbors.

Example

```
3
5
7
```

`weights`

```
uniform
distance
```

Distance gives closer neighbors more influence.

---

# Distance Used

Usually Euclidean distance.

```
distance =
√((x1−y1)^2 + (x2−y2)^2 + ...)
```

Because KNN relies on distances, **feature scaling is strongly recommended** before applying `KNNImputer`, especially when features have very different ranges (e.g., Age vs. Salary).

---

# 3. Regression Imputation

Train a regression model.

Example

```
Salary = f(Age, Experience)
```

Predict only missing salaries.

Simple but does not naturally handle uncertainty and can underestimate variability.

---

# 4. Random Forest Imputation (MissForest)

Uses Random Forest instead of linear regression.

Advantages

* Handles nonlinear relationships
* High accuracy
* Robust to outliers
* Works with mixed feature types (numeric and categorical, depending on the implementation)

Not directly available in scikit-learn, but implementations exist in third-party libraries.

---

# Comparison

| Method           | Uses Other Features | Iterative | Handles Nonlinear Data      | Speed | Best For                      |
| ---------------- | ------------------- | --------- | --------------------------- | ----- | ----------------------------- |
| Mean             | ❌                   | ❌         | ❌                           | ⭐⭐⭐⭐⭐ | Baseline                      |
| Median           | ❌                   | ❌         | ❌                           | ⭐⭐⭐⭐⭐ | Skewed numeric data           |
| Mode             | ❌                   | ❌         | ❌                           | ⭐⭐⭐⭐⭐ | Categorical data              |
| KNN              | ✅                   | ❌         | ✅                           | ⭐⭐⭐   | Local similarity              |
| Iterative (MICE) | ✅                   | ✅         | Depends on estimator        | ⭐⭐    | Complex relationships         |
| Regression       | ✅                   | ❌         | Limited (linear by default) | ⭐⭐⭐⭐  | Linear relationships          |
| MissForest       | ✅                   | ✅         | ✅                           | ⭐     | High accuracy on complex data |

---

# Advantages of Multivariate Imputation

* Uses information from other variables.
* Usually more accurate than mean/median imputation.
* Preserves relationships between variables.
* Can improve downstream model performance.
* MICE can estimate multiple variables with missing values simultaneously.

---

# Disadvantages

* Computationally more expensive.
* Assumes the relationships between features are informative and reasonably stable.
* Can overfit if the estimator is too complex for a small dataset.
* More difficult to interpret than simple imputation.
* Results depend on the chosen estimator and its hyperparameters.

---

# When to Use Which Method

| Dataset Situation                          | Recommended Method                                          |
| ------------------------------------------ | ----------------------------------------------------------- |
| Few missing values (<5%)                   | Mean/Median/Mode often suffice                              |
| Numeric features with meaningful neighbors | KNN Imputer                                                 |
| Multiple correlated features               | IterativeImputer (MICE)                                     |
| Strong nonlinear relationships             | IterativeImputer with tree-based estimator or MissForest    |
| Very large datasets                        | Simple imputation or carefully tuned KNN (MICE may be slow) |
| Mixed numeric/categorical data             | MissForest or separate pipelines for each feature type      |

---

# Interview Questions

**1. Why is multivariate imputation usually better than mean imputation?**
Because it leverages relationships among features instead of replacing all missing values with a single statistic, leading to more realistic estimates.

**2. Why should the imputer be fit only on the training set?**
To avoid data leakage. Statistics and learned relationships should come only from the training data; the test data should only be transformed.

**3. What is the default estimator in `IterativeImputer`?**
`BayesianRidge`.

**4. Is `IterativeImputer` deterministic?**
It is deterministic with a fixed `random_state` and default settings. If `sample_posterior=True` is used, the imputations become stochastic.

**5. When is KNN Imputer preferred?**
When similar samples are expected to have similar values and local neighborhood information is more useful than fitting a global regression model.
