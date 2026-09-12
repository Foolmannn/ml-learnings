`GridSearchCV` has **two different kinds of parameters**, and this distinction is very important:

1. **Parameters of `GridSearchCV` itself** → control *how the search is performed*
2. **Parameters of the model** → specify *what configurations GridSearchCV should try*

---

# 1. `GridSearchCV` parameters

The basic structure is:

```python
GridSearchCV(
    estimator,
    param_grid,
    scoring=None,
    n_jobs=None,
    cv=None,
    verbose=0,
    refit=True,
    return_train_score=False
)
```

Let's understand each one.

---

## 1. `estimator`

The ML algorithm you want to tune.

Example:

```python
GridSearchCV(
    BaggingClassifier(),
    ...
)
```

Here:

```python
BaggingClassifier()
```

is the estimator.

Other examples:

```python
GridSearchCV(
    LogisticRegression(),
    ...
)
```

```python
GridSearchCV(
    DecisionTreeClassifier(),
    ...
)
```

```python
GridSearchCV(
    RandomForestClassifier(),
    ...
)
```

So:

> **Estimator = Which model are we tuning?**

---

# 2. `param_grid`

This is the **most important parameter**.

It contains the hyperparameters and values that GridSearchCV should try.

For your Bagging example:

```python
parameters = {
    'n_estimators': [50, 100, 500],
    'max_samples': [0.1, 0.4, 0.7, 1.0],
    'bootstrap': [True, False],
    'max_features': [0.1, 0.4, 0.7, 1.0]
}
```

Then:

```python
GridSearchCV(
    BaggingClassifier(),
    parameters,
    ...
)
```

GridSearchCV tries every possible combination.

For example:

```text
n_estimators = 50
max_samples = 0.1
bootstrap = True
max_features = 0.1
```

Then:

```text
n_estimators = 50
max_samples = 0.1
bootstrap = True
max_features = 0.4
```

and so on.

This is called the **parameter grid**.

---

# 3. `scoring`

This tells GridSearchCV:

> **How should I decide which model is best?**

For classification:

```python
scoring='accuracy'
```

Example:

```python
search = GridSearchCV(
    BaggingClassifier(),
    parameters,
    scoring='accuracy',
    cv=5
)
```

Other common scoring metrics:

```python
scoring='precision'
```

```python
scoring='recall'
```

```python
scoring='f1'
```

```python
scoring='roc_auc'
```

For regression:

```python
scoring='r2'
```

```python
scoring='neg_mean_squared_error'
```

```python
scoring='neg_mean_absolute_error'
```

### Important

If you don't specify `scoring`:

```python
scoring=None
```

GridSearchCV uses the estimator's default scoring method.

For `LogisticRegression` or `BaggingClassifier`, this is generally **accuracy**.

---

# 4. `cv`

`cv` means **cross-validation**.

You used:

```python
cv=5
```

This means **5-fold cross-validation**.

Suppose your training data is:

```text
        Training Data
             │
    ┌────────┴────────┐
    │                  │
  Fold 1             Fold 2
  Fold 3             Fold 4
  Fold 5
```

Each model configuration gets trained and evaluated 5 times.

For example:

```python
parameters = {
    'n_estimators': [50, 100]
}
```

There are 2 configurations.

With:

```python
cv=5
```

GridSearchCV performs:

$$
2 \times 5 = 10
$$

model fits.

Your previous grid had:

$$
96 \times 5 = 480
$$

fits.

---

# 5. `n_jobs`

Controls **how many CPU cores are used**.

```python
n_jobs=None
```

means normally use one job.

You can use:

```python
n_jobs=-1
```

which means:

> Use all available CPU cores.

For your Bagging GridSearch, I strongly recommend:

```python
n_jobs=-1
```

Example:

```python
search = GridSearchCV(
    BaggingClassifier(),
    parameters,
    cv=5,
    n_jobs=-1
)
```

This can substantially reduce the runtime.

---

# 6. `verbose`

Controls how much information GridSearchCV prints while running.

```python
verbose=0
```

means no progress information.

Use:

```python
verbose=2
```

For example:

```python
search = GridSearchCV(
    BaggingClassifier(),
    parameters,
    cv=5,
    n_jobs=-1,
    verbose=2
)
```

You'll see information such as:

```text
Fitting 5 folds for each of 96 candidates, totalling 480 fits
```

This is very useful when a search takes a long time.

---

# 7. `refit`

This is an important one.

By default:

```python
refit=True
```

After GridSearchCV finds the best hyperparameters, it **trains the best model again on the entire training dataset**.

For example:

```python
search.fit(X_train, y_train)
```

Suppose it discovers:

```python
{
    'n_estimators': 100,
    'max_samples': 0.7
}
```

GridSearchCV then creates the best model:

```python
BaggingClassifier(
    n_estimators=100,
    max_samples=0.7
)
```

and fits it on the complete `X_train`.

That's why you can do:

```python
search.predict(X_test)
```

because `search` contains the refitted best estimator.

You can access it with:

```python
search.best_estimator_
```

---

# 8. `return_train_score`

By default:

```python
return_train_score=False
```

If you set:

```python
return_train_score=True
```

GridSearchCV also stores the training scores for each configuration.

Then you can inspect:

```python
search.cv_results_
```

For example:

```python
search.cv_results_['mean_train_score']
```

This can be useful for understanding **overfitting**.

---

# 9. `error_score`

This controls what happens if one particular parameter combination causes an error.

For example:

```python
error_score='raise'
```

means:

> Stop and raise the error.

The default behavior uses:

```python
error_score=np.nan
```

meaning the failed configuration gets a `NaN` score instead of immediately stopping the entire search.

---

# 10. `return_train_score`

As mentioned above:

```python
return_train_score=True
```

allows you to compare:

```text
Training score
Validation score
```

which is useful for diagnosing:

```text
Underfitting
Overfitting
Good fit
```

---

# The most important ones to remember

You don't need to memorize every parameter.

For ML learning, focus on these:

| Parameter    | Purpose                                   |
| ------------ | ----------------------------------------- |
| `estimator`  | Model you want to tune                    |
| `param_grid` | Hyperparameters and values to try         |
| `scoring`    | Metric used to select the best model      |
| `cv`         | Number/strategy of cross-validation folds |
| `n_jobs`     | Number of parallel CPU jobs               |
| `verbose`    | Progress information                      |
| `refit`      | Retrain best model after search           |

---

# Your Bagging example

A good practical version would be:

```python
parameters = {
    'n_estimators': [50, 100, 500],
    'max_samples': [0.1, 0.4, 0.7, 1.0],
    'bootstrap': [True, False],
    'max_features': [0.1, 0.4, 0.7, 1.0]
}

search = GridSearchCV(
    estimator=BaggingClassifier(random_state=42),
    param_grid=parameters,
    scoring='accuracy',
    cv=5,
    n_jobs=-1,
    verbose=2,
    refit=True
)

search.fit(X_train, y_train)
```

Then the most useful results are:

```python
search.best_params_
```

→ **Best hyperparameters**

```python
search.best_score_
```

→ **Best mean CV score**

```python
search.best_estimator_
```

→ **Best trained model**

---

### One conceptual distinction to remember

Don't confuse:

```python
GridSearchCV(..., cv=5)
```

with:

```python
parameters = {
    'n_estimators': [50, 100, 500]
}
```

`cv`, `scoring`, `n_jobs`, `verbose`, etc. are **GridSearchCV parameters**.

`n_estimators`, `max_samples`, `bootstrap`, `max_features` are **BaggingClassifier hyperparameters** that GridSearchCV is searching through.

So the workflow is:

```text
                 GridSearchCV
                      │
             ┌────────┴────────┐
             │                 │
       How to search?     What to search?
             │                 │
       cv, scoring,       n_estimators
       n_jobs, verbose    max_samples
             │            bootstrap
             │            max_features
             │                 │
             └────────┬────────┘
                      ↓
              Best Parameters
                      ↓
              Best Estimator
```

This distinction becomes **very important when you start using GridSearchCV with Random Forest, SVM, Logistic Regression, XGBoost, etc.**
