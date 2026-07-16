# Handling Mixed Variables in Machine Learning with Scikit-learn

In real-world datasets, you almost never have only one type of feature. Most datasets contain a **mixture of numerical and categorical variables**, and sometimes datetime, text, or boolean variables as well. Before training a machine learning model, these variables need different preprocessing techniques.

For example, consider this dataset:

| Age | Salary | Gender | City      | Purchased |
| --- | ------ | ------ | --------- | --------- |
| 25  | 40000  | Male   | Kathmandu | Yes       |
| 32  | 65000  | Female | Pokhara   | No        |
| 45  | 90000  | Male   | Lalitpur  | Yes       |
| 28  | 52000  | Female | Butwal    | No        |

Here,

* **Age** → Numerical
* **Salary** → Numerical
* **Gender** → Categorical (Binary)
* **City** → Categorical (Multi-class)
* **Purchased** → Target variable

Since numerical and categorical columns require different preprocessing, Scikit-learn provides **ColumnTransformer** to apply different transformations to different columns.

---

# Why Can't We Apply the Same Transformation to Every Column?

Suppose we use StandardScaler on all columns.

```
Age      Salary      Gender
25       40000       Male
```

StandardScaler only works on numbers.

```
ValueError:
could not convert string to float: 'Male'
```

Similarly,

Applying OneHotEncoder on numerical columns makes no sense.

Therefore we preprocess each type separately.

---

# Common Preprocessing Needed

## Numerical Features

Usually require

* Missing value imputation
* Scaling
* Power transformation
* Polynomial features (optional)

Example

```
Age
Salary
Height
Weight
```

---

## Categorical Features

Usually require

* Missing value imputation
* Encoding

Example

```
Gender
City
Department
```

---

# ColumnTransformer

This is the most important tool.

It allows different preprocessing pipelines on different columns.

```
Dataset
       |
ColumnTransformer
      /      \
Numeric    Categorical
Pipeline     Pipeline
      \      /
   Combined Dataset
          |
       ML Model
```

---

# Basic Syntax

```python
from sklearn.compose import ColumnTransformer

ct = ColumnTransformer(
    transformers=[
        ('num', transformer1, numeric_columns),
        ('cat', transformer2, categorical_columns)
    ]
)
```

Each tuple contains

```
(name,
 transformer,
 columns)
```

---

# Example Dataset

```python
import pandas as pd

df = pd.DataFrame({
    "Age":[25,30,35,40],
    "Salary":[30000,50000,70000,90000],
    "Gender":["Male","Female","Female","Male"],
    "City":["KTM","PKR","BRT","KTM"]
})

print(df)
```

Output

```
Age Salary Gender City

25 30000 Male KTM
30 50000 Female PKR
35 70000 Female BRT
40 90000 Male KTM
```

---

# Example 1: Scale Numeric + OneHot Encode Categorical

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(
    transformers=[
        ('num',
         StandardScaler(),
         ['Age','Salary']),

        ('cat',
         OneHotEncoder(),
         ['Gender','City'])
    ]
)

X = ct.fit_transform(df)
```

Result

```
Scaled Age
Scaled Salary
Gender_Female
Gender_Male
City_BRT
City_KTM
City_PKR
```

Everything becomes numerical.

---

# What Happens Internally?

Input

```
Age Salary Gender City

25 30000 Male KTM
30 50000 Female PKR
```

↓

Numeric Pipeline

```
Age Salary

-1.2 -1.4
-0.4 -0.2
```

↓

Categorical Pipeline

```
Male Female KTM PKR

1 0 1 0
0 1 0 1
```

↓

Combined

```
-1.2 -1.4 1 0 1 0
-0.4 -0.2 0 1 0 1
```

---

# Handling Missing Values Too

Suppose

```
Age

25
NaN
35
```

and

```
Gender

Male
NaN
Female
```

We use separate imputers.

```python
from sklearn.impute import SimpleImputer

num_imputer = SimpleImputer(strategy="mean")

cat_imputer = SimpleImputer(strategy="most_frequent")
```

---

# Multiple Steps Using Pipeline

Instead of a single transformer, each column type usually has a pipeline.

```
Numerical

Missing Values
      ↓
Scaling
      ↓
Output
```

```
Categorical

Missing Values
      ↓
OneHotEncoder
      ↓
Output
```

---

## Numeric Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])
```

---

## Categorical Pipeline

```python
cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder())
])
```

---

## Combine Both

```python
ct = ColumnTransformer([
    ('num', num_pipeline, ['Age','Salary']),
    ('cat', cat_pipeline, ['Gender','City'])
])
```

Now one call performs everything.

```python
X = ct.fit_transform(df)
```

---

# Complete Example with Model

```python
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

df = pd.DataFrame({
    "Age":[25,30,35,40],
    "Salary":[30000,50000,70000,90000],
    "Gender":["Male","Female","Female","Male"],
    "City":["KTM","PKR","BRT","KTM"],
    "Purchased":[0,1,1,0]
})

X = df.drop("Purchased", axis=1)
y = df["Purchased"]

num_cols = ["Age","Salary"]
cat_cols = ["Gender","City"]

num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", num_pipeline, num_cols),
    ("cat", cat_pipeline, cat_cols)
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression())
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model.fit(X_train, y_train)

pred = model.predict(X_test)
```

Notice that the `Pipeline` ensures the preprocessing is fit only on the training data and automatically applied to the test data, preventing data leakage.

---

# Column Selection Methods

### By Name

```python
["Age","Salary"]
```

---

### By Index

```python
[0,1]
```

---

### Using make_column_selector

Useful for large datasets.

```python
from sklearn.compose import make_column_selector

numeric_selector = make_column_selector(dtype_include='number')

categorical_selector = make_column_selector(dtype_include='object')
```

Example

```python
preprocessor = ColumnTransformer([
    ("num", num_pipeline, numeric_selector),
    ("cat", cat_pipeline, categorical_selector)
])
```

This automatically detects column types.

---

# Handling Different Categorical Variables

Different categorical columns may need different encoders:

* **Nominal (no order):** `OneHotEncoder`
* **Ordinal (ordered):** `OrdinalEncoder`

Example:

```python
from sklearn.preprocessing import OrdinalEncoder

preprocessor = ColumnTransformer([
    ("num", num_pipeline, ["Age", "Salary"]),
    ("nominal", OneHotEncoder(handle_unknown="ignore"), ["City"]),
    ("ordinal", OrdinalEncoder(categories=[["Low", "Medium", "High"]]), ["Education"])
])
```

---

# Useful Parameters

### `remainder`

Controls what happens to columns not listed.

```python
ColumnTransformer(
    [...],
    remainder='drop'      # default
)
```

or

```python
ColumnTransformer(
    [...],
    remainder='passthrough'
)
```

Example:

```
Age
Salary
Gender
City
ID
```

If `ID` isn't specified:

* `drop` → remove it
* `passthrough` → keep it unchanged

---

### `verbose_feature_names_out`

Controls feature names after transformation.

```python
ColumnTransformer(
    ...,
    verbose_feature_names_out=False
)
```

---

### `sparse_threshold`

If most outputs are sparse (common with one-hot encoding), the output will be a sparse matrix. Setting `sparse_threshold=0` forces a dense array.

---

# Getting Feature Names

```python
feature_names = preprocessor.get_feature_names_out()

print(feature_names)
```

Output

```
num__Age
num__Salary
cat__Gender_Female
cat__Gender_Male
cat__City_BRT
cat__City_KTM
cat__City_PKR
```

---

# Best Practices

* Use **Pipeline** for sequential transformations on the same columns.
* Use **ColumnTransformer** to apply different preprocessing to different column groups.
* Always fit preprocessors on the **training data only**.
* Set `handle_unknown="ignore"` in `OneHotEncoder` to avoid errors with unseen categories.
* Use `make_column_selector` for large datasets where selecting columns manually is cumbersome.
* Combine preprocessing and the estimator into one `Pipeline` to make training, prediction, cross-validation, and hyperparameter tuning cleaner and safer.

---

# Summary Workflow

```text
Raw Dataset
     │
     ▼
Train-Test Split
     │
     ▼
ColumnTransformer
 ┌──────────────┬─────────────────┐
 │ Numeric      │ Categorical     │
 │ Pipeline     │ Pipeline        │
 │ Imputer      │ Imputer         │
 │ Scaler       │ Encoder         │
 └──────────────┴─────────────────┘
     │
     ▼
Processed Features
     │
     ▼
Machine Learning Model
     │
     ▼
Predictions
```

This `ColumnTransformer` + `Pipeline` pattern is the standard, scalable approach in scikit-learn for handling mixed-variable datasets and is widely used in production ML workflows.
