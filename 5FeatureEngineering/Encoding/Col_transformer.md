`ColumnTransformer` is one of the most important preprocessing tools in **scikit-learn**. It allows you to **apply different preprocessing steps to different columns of your dataset** in a single pipeline.

This is especially useful because real-world datasets usually contain:

* Numerical features (Age, Salary)
* Categorical features (Gender, City)
* Text features (Review)
* Date features

Each type requires different preprocessing.

---

# Why do we need ColumnTransformer?

Suppose we have the following dataset:

| Age | Salary | Gender | City      |
| --- | ------ | ------ | --------- |
| 25  | 40000  | Male   | Kathmandu |
| 30  | 50000  | Female | Pokhara   |
| 22  | 35000  | Female | Butwal    |

Different columns require different preprocessing.

* Age → StandardScaler
* Salary → MinMaxScaler
* Gender → OneHotEncoder
* City → OneHotEncoder

Without `ColumnTransformer`, we would need to preprocess each column separately and then combine everything manually.

That becomes messy.

`ColumnTransformer` does everything automatically.

---

# Syntax

```python
from sklearn.compose import ColumnTransformer

ct = ColumnTransformer(
    transformers=[
        ("name1", transformer1, columns1),
        ("name2", transformer2, columns2)
    ]
)
```

where

```
(name, transformer, columns)
```

means

* **name**

  * Just an identifier.
  * Can be anything.

```python
"scale"
```

or

```python
"cat"
```

---

**transformer**

The preprocessing object.

Example

```python
StandardScaler()
```

or

```python
OneHotEncoder()
```

---

**columns**

The columns on which the transformer will be applied.

Example

```python
["Age", "Salary"]
```

or

```python
["Gender"]
```

---

# Example 1

Suppose

```python
import pandas as pd

df = pd.DataFrame({
    "Age":[20,25,30],
    "Salary":[30000,50000,70000],
    "Gender":["Male","Female","Female"]
})
```

We want

* Scale Age
* Scale Salary
* One Hot Encode Gender

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), ["Age","Salary"]),
        ("cat", OneHotEncoder(), ["Gender"])
    ]
)
```

Now transform

```python
X = ct.fit_transform(df)

print(X)
```

Output

```
array([
[-1.22, -1.22, 1, 0],
[ 0.00,  0.00, 0, 1],
[ 1.22,  1.22, 0, 1]
])
```

Notice

```
Age
Salary
Gender_Male
Gender_Female
```

are all combined automatically.

---

# Understanding the Flow

```
                Dataset
                    |
     --------------------------------
     |                              |
 Numerical Columns          Categorical Columns
     |                              |
StandardScaler()           OneHotEncoder()
     |                              |
     -----------Concatenate-----------
                    |
             Final Feature Matrix
```

---

# Example 2

Using different scalers

```python
ct = ColumnTransformer(
    [
        ("standard", StandardScaler(), ["Age"]),
        ("minmax", MinMaxScaler(), ["Salary"]),
        ("encode", OneHotEncoder(), ["Gender"])
    ]
)
```

Result

```
Age -> StandardScaler

Salary -> MinMaxScaler

Gender -> OneHotEncoder
```

---

# Example 3

Ignoring some columns

Suppose

| Age | Salary | Gender | Name |
| --- | ------ | ------ | ---- |

We only want to preprocess

```
Age
Salary
Gender
```

and ignore Name.

```python
ct = ColumnTransformer(
    [
        ("num", StandardScaler(), ["Age","Salary"]),
        ("cat", OneHotEncoder(), ["Gender"])
    ],
    remainder="drop"
)
```

Output

```
Age
Salary
Gender
```

Only these columns remain.

---

# Keeping remaining columns

Sometimes we want to transform only some columns.

Keep everything else unchanged.

```python
ct = ColumnTransformer(
    [
        ("num", StandardScaler(), ["Age"])
    ],
    remainder="passthrough"
)
```

Suppose

| Age | Salary | Gender |
| --- | ------ | ------ |

Output

```
Scaled Age
Salary
Gender
```

Salary and Gender remain unchanged.

---

# Using Column Index

Instead of names

```python
ct = ColumnTransformer(
    [
        ("num", StandardScaler(), [0,1]),
        ("cat", OneHotEncoder(), [2])
    ]
)
```

---

# Selecting Columns by Data Type

This is useful when you don't want to hardcode column names.

```python
from sklearn.compose import make_column_selector

ct = ColumnTransformer(
    [
        (
            "num",
            StandardScaler(),
            make_column_selector(dtype_include="number")
        ),
        (
            "cat",
            OneHotEncoder(),
            make_column_selector(dtype_include="object")
        )
    ]
)
```

It automatically selects

* Numeric columns
* Object (categorical) columns

---

# Getting Feature Names

After one-hot encoding, many new columns are created.

```python
ct.fit(df)

ct.get_feature_names_out()
```

Output

```
[
'num__Age',
'num__Salary',
'cat__Gender_Female',
'cat__Gender_Male'
]
```

Very useful for converting the transformed array back into a DataFrame.

```python
import pandas as pd

X = ct.fit_transform(df)

df_new = pd.DataFrame(
    X,
    columns=ct.get_feature_names_out()
)

print(df_new)
```

---

# Sparse Output

`OneHotEncoder` often returns a sparse matrix to save memory.

```python
OneHotEncoder()
```

returns sparse output by default.

To force a dense array:

```python
OneHotEncoder(sparse_output=False)
```

This is convenient when working with pandas DataFrames.

---

# Using ColumnTransformer with Pipeline

This is the recommended approach because it prevents **data leakage** by ensuring preprocessing is learned only from the training data.

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression

preprocessor = ColumnTransformer(
    [
        ("num", StandardScaler(), ["Age","Salary"]),
        ("cat", OneHotEncoder(), ["Gender"])
    ]
)

model = Pipeline(
    [
        ("preprocess", preprocessor),
        ("classifier", LogisticRegression())
    ]
)

model.fit(X_train, y_train)
```

When you later call:

```python
pred = model.predict(X_test)
```

the pipeline automatically:

1. Applies the same preprocessing learned from `X_train`.
2. Transforms `X_test`.
3. Makes predictions with the trained model.

---

# Advantages of ColumnTransformer

* Applies different preprocessing to different column groups.
* Keeps preprocessing code organized and maintainable.
* Integrates seamlessly with `Pipeline`.
* Reduces the risk of inconsistent preprocessing and data leakage.
* Automatically combines transformed outputs into a single feature matrix.
* Works with column names, indices, or automatic column selectors.

---

# Common Parameters

| Parameter                   | Description                                                                                     |
| --------------------------- | ----------------------------------------------------------------------------------------------- |
| `transformers`              | List of `(name, transformer, columns)` tuples.                                                  |
| `remainder`                 | `"drop"` (default) drops unspecified columns, `"passthrough"` keeps them.                       |
| `verbose_feature_names_out` | If `False`, removes prefixes like `num__` from output feature names (when names remain unique). |
| `n_jobs`                    | Number of CPU cores to use for parallel preprocessing.                                          |
| `transformer_weights`       | Assigns weights to the outputs of specific transformers (used less frequently).                 |

---

# Common Mistakes

1. **Calling `fit_transform()` on the test set**

   ```python
   X_train = ct.fit_transform(X_train)
   X_test = ct.fit_transform(X_test)   # ❌ Wrong
   ```

   This fits the preprocessing twice and can cause data leakage.

   Correct approach:

   ```python
   X_train = ct.fit_transform(X_train)
   X_test = ct.transform(X_test)       # ✅
   ```

2. **Forgetting `remainder="passthrough"`**

   * By default, columns not listed in `transformers` are dropped.

3. **Not setting `sparse_output=False` when a dense array is needed**

   ```python
   OneHotEncoder(sparse_output=False)
   ```

4. **Using incorrect column names or indices**

   * Ensure the specified columns exist in the input DataFrame.

---

## Summary

`ColumnTransformer` acts like a traffic controller for your dataset:

```
                Dataset
                    │
        ┌───────────┴───────────┐
        │                       │
 Numeric Columns         Categorical Columns
        │                       │
StandardScaler()       OneHotEncoder()
        │                       │
        └───────────┬───────────┘
                    │
        Combined Feature Matrix
                    │
              Machine Learning Model
```

In modern scikit-learn workflows, the most common pattern is:

```python
Pipeline(
    [
        ("preprocess", ColumnTransformer(...)),
        ("model", SomeEstimator())
    ]
)
```

This creates a reproducible, leakage-free preprocessing and training workflow that works consistently for both training and prediction.
