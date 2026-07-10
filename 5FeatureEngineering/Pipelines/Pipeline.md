# Pipelines in Machine Learning using scikit-learn

A **Pipeline** in **scikit-learn** is a way to **chain multiple preprocessing steps and a machine learning model into a single object**.

Instead of writing code like this:

```python
# Step 1
imputer.fit_transform(X_train)

# Step 2
encoder.fit_transform(X_train)

# Step 3
scaler.fit_transform(X_train)

# Step 4
model.fit(X_train, y_train)
```

you can combine everything into one pipeline:

```python
pipeline.fit(X_train, y_train)
```

The pipeline automatically performs every step in the correct order.

---

# Why use Pipelines?

Suppose your ML workflow is

```
Dataset
   ↓
Missing Value Imputation
   ↓
Encoding
   ↓
Scaling
   ↓
Feature Selection
   ↓
Model Training
```

Without a pipeline, you have to manually remember every transformation.

With a pipeline:

```
Pipeline
 ├── Imputer
 ├── Encoder
 ├── Scaler
 ├── Feature Selection
 └── Classifier
```

Everything becomes one object.

---

# Advantages

## 1. Cleaner Code

Without pipeline

```python
X_train = imputer.fit_transform(X_train)
X_train = encoder.fit_transform(X_train)
X_train = scaler.fit_transform(X_train)

model.fit(X_train, y_train)
```

With pipeline

```python
pipe.fit(X_train, y_train)
```

---

## 2. Prevents Data Leakage

This is one of the biggest advantages.

Imagine scaling before splitting the dataset.

Wrong

```python
scaler.fit(X)

X_train, X_test = train_test_split(...)
```

The scaler has already seen the test data.

This causes **data leakage**.

Instead

```python
pipe.fit(X_train, y_train)

pipe.predict(X_test)
```

Only training data is used for learning the transformations.

---

## 3. Easy Cross Validation

Without pipeline

```python
scores = cross_val_score(model, X_scaled, y)
```

The scaling was done before CV.

Wrong.

With pipeline

```python
scores = cross_val_score(pipe, X, y)
```

Each fold performs preprocessing separately.

---

## 4. Hyperparameter Tuning

Works directly with GridSearchCV.

```python
GridSearchCV(pipe, param_grid)
```

---

## 5. Easy Deployment

Save one object.

```python
joblib.dump(pipe, "model.pkl")
```

Later

```python
pipe = joblib.load("model.pkl")
```

Everything is restored.

---

# Structure of a Pipeline

```
Pipeline

Step 1
↓

Step 2
↓

Step 3
↓

Final Model
```

All intermediate steps must implement:

```
fit()
transform()
```

The last step must implement

```
fit()
predict()
```

---

# sklearn Pipeline

Import

```python
from sklearn.pipeline import Pipeline
```

Syntax

```python
pipe = Pipeline([
    ('step_name1', transformer1),
    ('step_name2', transformer2),
    ('model', estimator)
])
```

Each tuple contains

```
(name, object)
```

Example

```python
Pipeline([
    ('imputer', SimpleImputer()),
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression())
])
```

---

# Example 1: Numerical Dataset

Dataset

```
Age    Salary
23     20000
30     NaN
40     50000
```

Pipeline

```python
from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression())
])
```

Training

```python
pipe.fit(X_train, y_train)
```

Prediction

```python
pred = pipe.predict(X_test)
```

Internally

```
X_train

↓

Imputer

↓

Scaler

↓

Logistic Regression
```

---

# Example 2: Encoding + Scaling + Model

Suppose

```
Gender
City
Age
Salary
```

Need

* Encode Gender
* Encode City
* Scale Age
* Scale Salary
* Train model

This requires a `ColumnTransformer`.

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(), ['Gender','City']),
    ('num', StandardScaler(), ['Age','Salary'])
])
```

Now combine with model.

```python
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression())
])
```

Training

```python
pipe.fit(X_train, y_train)
```

Prediction

```python
pipe.predict(X_test)
```

---

# Pipeline Execution

Suppose

```python
Pipeline([
    ('imputer', SimpleImputer()),
    ('scaler', StandardScaler()),
    ('model', RandomForestClassifier())
])
```

Calling

```python
pipe.fit(X_train, y_train)
```

Internally

```
Step 1

imputer.fit()
imputer.transform()

↓

Step 2

scaler.fit()
scaler.transform()

↓

Step 3

model.fit()
```

Prediction

```
pipe.predict(X_test)
```

Internally

```
imputer.transform()

↓

scaler.transform()

↓

model.predict()
```

Notice

The model is **not fitted again**.

---

# Access Individual Steps

```python
pipe.named_steps
```

Output

```python
{
 'imputer': SimpleImputer(),
 'scaler': StandardScaler(),
 'classifier': LogisticRegression()
}
```

Access scaler

```python
pipe.named_steps['scaler']
```

---

# Changing Parameters

```python
pipe.set_params(
    classifier__C=10
)
```

Double underscore (`__`) separates the step name from the parameter.

---

# Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'classifier__C':[0.1,1,10],
    'classifier__solver':['lbfgs','liblinear']
}

grid = GridSearchCV(pipe, param_grid)

grid.fit(X_train, y_train)
```

Notice

```
classifier__C
```

means

```
Pipeline

↓

classifier

↓

parameter C
```

---

# Using Cross Validation

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(pipe, X, y, cv=5)
```

Every fold does

```
Training Fold

↓

Imputer.fit()

↓

Scaler.fit()

↓

Train Model

↓

Validation Fold

↓

Imputer.transform()

↓

Scaler.transform()

↓

Prediction
```

This avoids leakage.

---

# Pipeline with ColumnTransformer

A common pattern for mixed data is:

```python
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

numeric_features = ['Age', 'Salary']
categorical_features = ['Gender', 'City']

numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression())
])

pipe.fit(X_train, y_train)

predictions = pipe.predict(X_test)
```

This nested structure is very common in real-world projects.

```
Pipeline
│
├── ColumnTransformer
│   ├── Numeric Pipeline
│   │   ├── Imputer
│   │   └── StandardScaler
│   │
│   └── Categorical Pipeline
│       ├── Imputer
│       └── OneHotEncoder
│
└── Logistic Regression
```

---

# Commonly Used Transformers in Pipelines

| Transformer          | Purpose                                            |
| -------------------- | -------------------------------------------------- |
| `SimpleImputer`      | Fill missing values                                |
| `StandardScaler`     | Standardize numerical features                     |
| `MinMaxScaler`       | Scale features to a range (e.g., 0–1)              |
| `OneHotEncoder`      | Encode nominal categorical variables               |
| `OrdinalEncoder`     | Encode ordinal categorical variables               |
| `PolynomialFeatures` | Create polynomial feature combinations             |
| `PCA`                | Reduce dimensionality                              |
| `SelectKBest`        | Select top features                                |
| `ColumnTransformer`  | Apply different preprocessing to different columns |

---

# Best Practices

* Split your data into training and testing sets **before** fitting the pipeline.
* Use `ColumnTransformer` when numerical and categorical features require different preprocessing.
* Use pipelines with `cross_val_score` and `GridSearchCV` to prevent data leakage.
* Name pipeline steps clearly (e.g., `"preprocessor"`, `"classifier"`), as these names are used when tuning hyperparameters.
* Save the entire fitted pipeline with `joblib` so preprocessing and the model stay together.

A well-designed pipeline makes your workflow cleaner, reproducible, and much less prone to preprocessing mistakes.
