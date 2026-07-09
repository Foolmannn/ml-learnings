This is one of the most important concepts in scikit-learn. Almost every transformer (such as `StandardScaler`, `OneHotEncoder`, `SimpleImputer`, and `ColumnTransformer`) follows the same pattern.

---

# 1. `fit()`

`fit()` **learns** information from the data but **does not change the data**.

Think of it as **studying** the dataset.

Example with `StandardScaler`:

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(X_train)
```

The scaler learns:

* Mean of each column
* Standard deviation of each column

Suppose

| Age |
| --- |
| 20  |
| 30  |
| 40  |

After

```python
scaler.fit(X_train)
```

it stores

```text
Mean = 30
Std = 8.16
```

but

```python
X_train
```

is still

| Age |
| --- |
| 20  |
| 30  |
| 40  |

Nothing has changed.

---

# 2. `transform()`

`transform()` **uses the information learned by `fit()`** to transform the data.

```python
X_scaled = scaler.transform(X_train)
```

Output

| Original | Scaled |
| -------- | ------ |
| 20       | -1.22  |
| 30       | 0      |
| 40       | 1.22   |

Notice that `transform()` does **not** calculate the mean again. It uses the mean and standard deviation already learned during `fit()`.

---

# 3. `fit_transform()`

This is simply a shortcut for:

```python
scaler.fit(X_train)
X_scaled = scaler.transform(X_train)
```

Instead, you write:

```python
X_scaled = scaler.fit_transform(X_train)
```

Both are equivalent.

---

# Visual Flow

```
Training Data
      │
      ▼
    fit()
      │
 Learns parameters
(mean, std, categories, etc.)
      │
      ▼
 transform()
      │
Transforms the data
```

or simply

```
Training Data
      │
      ▼
fit_transform()
      │
Learns + Transforms
```

---

# Example with `OneHotEncoder`

Suppose

```python
Gender

Male
Female
Male
Female
```

### Step 1

```python
encoder.fit(X_train)
```

The encoder learns

```
Categories

Female
Male
```

Nothing changes yet.

---

### Step 2

```python
encoder.transform(X_train)
```

Output

| Male | Female |
| ---- | ------ |
| 1    | 0      |
| 0    | 1      |
| 1    | 0      |
| 0    | 1      |

---

# Example with `SimpleImputer`

Data

| Age |
| --- |
| 20  |
| NaN |
| 40  |

### fit()

Learns

```
Mean = 30
```

---

### transform()

Replaces missing value

| Age |
| --- |
| 20  |
| 30  |
| 40  |

---

# Example with `ColumnTransformer`

```python
ct = ColumnTransformer([
    ("scale", StandardScaler(), ["age"]),
    ("encode", OneHotEncoder(), ["gender"])
])
```

### fit()

```python
ct.fit(X_train)
```

It learns:

* Mean and standard deviation for `age`
* Categories of `gender`

No transformed data is returned.

---

### transform()

```python
X_train_new = ct.transform(X_train)
```

Now it:

* Scales `age`
* Encodes `gender`
* Combines the results into one feature matrix

---

### fit_transform()

```python
X_train_new = ct.fit_transform(X_train)
```

This does both learning and transforming in one step.

---

# Why do we use `fit_transform()` only on the training set?

Suppose

```
Training

20
30
40
```

```
Testing

100
```

If you do

```python
scaler.fit(X_test)
```

the scaler learns

```
Mean = 100
```

instead of using the training mean (30).

Your model has now used information from the test set before evaluation. This is called **data leakage**, and it can make your model evaluation overly optimistic.

The correct workflow is:

```python
scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

Or more commonly:

```python
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

---

# Summary Table

| Method            | Learns from data? | Changes data? | Typical use                                                      |
| ----------------- | ----------------- | ------------- | ---------------------------------------------------------------- |
| `fit()`           | ✅ Yes             | ❌ No          | Learn preprocessing parameters from the training data.           |
| `transform()`     | ❌ No              | ✅ Yes         | Apply the learned transformation to new data (training or test). |
| `fit_transform()` | ✅ Yes             | ✅ Yes         | Learn and transform in one step, usually on the training data.   |

---

## A simple analogy

Imagine you're preparing for an exam:

* **`fit()`**: You study the textbook and learn the concepts.
* **`transform()`**: You use what you learned to solve practice questions.
* **`fit_transform()`**: You study the textbook and immediately solve the practice questions.

The key idea is that you **learn only once** (on the training data), then **apply that same knowledge** to any future data (validation, test, or new unseen samples).


Excellent question. The short answer is:

> **No, you should not use `LabelEncoder` inside a `ColumnTransformer`.**

Here's why.

## Why does it give an error?

`ColumnTransformer` expects every transformer to implement **both**:

* `fit()`
* `transform()`

`LabelEncoder` is different. It is designed for **1-dimensional target vectors (`y`)**, not for DataFrame columns (`X`).

It expects input like:

```python
y = ["Yes", "No", "Yes"]

le = LabelEncoder()
le.fit_transform(y)
```

But inside `ColumnTransformer`, the column is passed as a **2D DataFrame**:

```python
[['Yes'],
 ['No'],
 ['Yes']]
```

Since `LabelEncoder` doesn't support this, it raises an error.

---

## `LabelEncoder` is meant for the target (`y`)

The intended usage is:

```python
X = df.drop(columns=['has_covid'])
y = df['has_covid']

le = LabelEncoder()
y = le.fit_transform(y)
```

Then preprocess `X` with your `ColumnTransformer`:

```python
transformer = ColumnTransformer(
    transformers=[
        ('imputed', SimpleImputer(), ['fever']),
        ('o_encoded', OrdinalEncoder(categories=[['Mild', 'Strong']]), ['cough']),
        ('ohe', OneHotEncoder(sparse_output=False, drop='first'), ['gender', 'city'])
    ],
    remainder='passthrough'
)
```

---

## What if I want to encode a feature column?

If the column is an **input feature**, don't use `LabelEncoder`.

Instead use:

### For unordered categories

```python
OneHotEncoder()
```

Example:

```
City
-----
Delhi
Mumbai
Kolkata
```

↓

```
Delhi  Mumbai  Kolkata
1       0       0
0       1       0
0       0       1
```

---

### For ordered categories

```python
OrdinalEncoder()
```

Example:

```
Education

High School
Bachelor
Master
PhD
```

↓

```
0
1
2
3
```

---

## Why not use `LabelEncoder` on features?

Suppose you have

```
Color

Red
Blue
Green
```

`LabelEncoder` produces

```
Red    -> 2
Blue   -> 0
Green  -> 1
```

A machine learning model might incorrectly assume:

```
Blue < Green < Red
```

which has **no real meaning**. That's why `OneHotEncoder` is preferred for nominal (unordered) features.

---

## Correct workflow

```python
X = df.drop(columns=['has_covid'])
y = df['has_covid']

# Encode target
le = LabelEncoder()
y = le.fit_transform(y)

# Preprocess features
transformer = ColumnTransformer(
    [
        ('imputed', SimpleImputer(), ['fever']),
        ('ordinal', OrdinalEncoder(categories=[['Mild', 'Strong']]), ['cough']),
        ('ohe', OneHotEncoder(drop='first', sparse_output=False), ['gender', 'city'])
    ],
    remainder='passthrough'
)

X = transformer.fit_transform(X)
```

---

### Rule of thumb

| Transformer      | Use on `X` (features)? | Use on `y` (target)? |
| ---------------- | ---------------------- | -------------------- |
| `SimpleImputer`  | ✅                      | ❌                    |
| `StandardScaler` | ✅                      | ❌                    |
| `MinMaxScaler`   | ✅                      | ❌                    |
| `OneHotEncoder`  | ✅                      | ❌                    |
| `OrdinalEncoder` | ✅                      | ❌                    |
| `LabelEncoder`   | ❌                      | ✅                    |

This distinction—**`LabelEncoder` is for `y`, while `OneHotEncoder` and `OrdinalEncoder` are for `X`**—is a common interview question and an important scikit-learn convention.
