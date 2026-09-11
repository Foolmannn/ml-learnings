
. `cross_val_score` is one of the most important tools for understanding **whether your ML model actually generalizes well**.

# Cross-Validation and `cross_val_score()` in Detail

First understand the problem it solves.

Suppose you do:

```python
model.fit(X_train, y_train)
model.predict(X_test)
```

You get one test score.

For example:

```text
Accuracy = 0.87
```

But there is a problem:

> **What if your particular train/test split was unusually easy or unusually difficult?**

If you randomly split the data again, you might get:

```text
Accuracy = 0.82
```

or:

```text
Accuracy = 0.91
```

So we want a more reliable estimate of model performance.

That's where **cross-validation** comes in.

---

# 1. What is Cross-Validation?

Cross-validation means:

> **Split the training data into multiple parts, train the model multiple times, and evaluate it on different parts each time.**

The most common type is:

$$
\boxed{\text{K-Fold Cross-Validation}}
$$

For example, with:

```python
cv=5
```

we perform **5-fold cross-validation**.

---

# 2. Imagine Your Dataset

Suppose you have 100 samples:

```text
1  2  3  4  5  6  7  8  9  10
11 12 13 14 15 16 17 18 19 20
...
91 92 93 94 95 96 97 98 99 100
```

With 5-fold cross-validation, divide them into 5 groups:

```text
Fold 1 → 1-20
Fold 2 → 21-40
Fold 3 → 41-60
Fold 4 → 61-80
Fold 5 → 81-100
```

Then we train and validate 5 times.

---

# 3. How 5-Fold CV Works

### Round 1

Train on:

```text
Fold 2
Fold 3
Fold 4
Fold 5
```

Validate on:

```text
Fold 1
```

---

### Round 2

Train on:

```text
Fold 1
Fold 3
Fold 4
Fold 5
```

Validate on:

```text
Fold 2
```

---

### Round 3

Train on:

```text
Fold 1
Fold 2
Fold 4
Fold 5
```

Validate on:

```text
Fold 3
```

---

### Round 4

Train on:

```text
Fold 1
Fold 2
Fold 3
Fold 5
```

Validate on:

```text
Fold 4
```

---

### Round 5

Train on:

```text
Fold 1
Fold 2
Fold 3
Fold 4
```

Validate on:

```text
Fold 5
```

So every sample gets used:

- for training multiple times
- for validation exactly once

---

# 4. The Scores

Suppose our model gets:

```text
Fold 1 → 0.85
Fold 2 → 0.89
Fold 3 → 0.83
Fold 4 → 0.91
Fold 5 → 0.87
```

Then cross-validation score is usually the mean:

$$
CV\ Score =
\frac{0.85+0.89+0.83+0.91+0.87}{5}
$$

$$
=0.87
$$

So:

```text
Mean CV Score = 87%
```

---

# 5. What Does `cross_val_score()` Do?

Scikit-learn provides:

```python
from sklearn.model_selection import cross_val_score
```

Basic usage:

```python
scores = cross_val_score(
    model,
    X,
    y,
    cv=5
)
```

It will:

1. Split your data into 5 folds.
2. Train the model 5 times.
3. Evaluate each time.
4. Return the 5 scores.

For example:

```python
print(scores)
```

might give:

```text
[0.85 0.89 0.83 0.91 0.87]
```

Then:

```python
print(scores.mean())
```

gives:

```text
0.87
```

And:

```python
print(scores.std())
```

might give:

```text
0.027
```

---

# 6. Complete Example

Let's use Logistic Regression.

```python
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

X, y = load_iris(return_X_y=True)

model = LogisticRegression(max_iter=1000)

scores = cross_val_score(
    model,
    X,
    y,
    cv=5
)

print("Scores:", scores)
print("Mean:", scores.mean())
print("Std:", scores.std())
```

Possible output:

```text
Scores: [0.9667 1.0000 0.9333 0.9667 1.0000]

Mean: 0.9733
Std: 0.0249
```

Interpretation:

```text
Average performance ≈ 97.33%
```

---

# 7. Why Not Just Use `train_test_split()`?

Suppose:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

You get:

```text
Training → 80%
Testing  → 20%
```

Then:

```python
model.fit(X_train, y_train)

score = model.score(X_test, y_test)
```

Maybe:

```text
Accuracy = 92%
```

But this is based on **one particular split**.

Cross-validation:

```text
Fold 1 → 91%
Fold 2 → 94%
Fold 3 → 89%
Fold 4 → 93%
Fold 5 → 92%
```

Mean:

```text
91.8%
```

This gives you a better understanding of how the model behaves across different subsets of the data.

---

# 8. Very Important: Cross-Validation Does NOT Mean Test Set

This is one of the most important things to understand.

Suppose you have:

```text
Entire Dataset
       │
       ├───────────────┐
       ↓               ↓
 Training             Test
   80%                 20%
    │
    ↓
Cross Validation
    │
 ┌──┼──┬──┬──┐
 ↓  ↓  ↓  ↓  ↓
F1 F2 F3 F4 F5
```

You should generally keep the **test set untouched** until the end.

Cross-validation happens on the **training data**.

---

# 9. Proper ML Workflow

A good workflow is:

### Step 1 — Split data

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

### Step 2 — Perform CV on training data

```python
scores = cross_val_score(
    model,
    X_train,
    y_train,
    cv=5
)
```

### Step 3 — Look at CV performance

```python
print(scores.mean())
```

### Step 4 — Train final model on all training data

```python
model.fit(X_train, y_train)
```

### Step 5 — Evaluate once on test data

```python
test_score = model.score(X_test, y_test)
```

So:

$$
\boxed{
CV \rightarrow Model\ Selection
}
$$

while:

$$
\boxed{
Test\ Set \rightarrow Final\ Evaluation
}
$$

---

# 10. Why Is This Important?

Imagine you are comparing three models:

```text
Logistic Regression
Decision Tree
SVM
```

You shouldn't choose the model simply based on one train/test split.

Instead:

```text
                    Training Data
                         │
                ┌────────┼────────┐
                ↓        ↓        ↓
                LR       DT       SVM
                │        │        │
                ↓        ↓        ↓
              CV=91%   CV=87%   CV=94%
                │        │        │
                └────────┼────────┘
                         ↓
                       SVM
```

You would probably choose SVM for further tuning.

---

# 11. `cv=5` Doesn't Mean 5 Models

This can be slightly confusing.

When you write:

```python
cross_val_score(model, X, y, cv=5)
```

the **same model type** is trained 5 times on different training subsets.

Conceptually:

```text
Logistic Regression
        │
 ┌──────┼──────┬──────┬──────┐
 ↓      ↓      ↓      ↓      ↓
Train1 Train2 Train3 Train4 Train5
```

These are separate fitted instances during CV, not five different algorithms.

---

# 12. What Does `cv` Mean?

`cv` controls the cross-validation strategy.

For example:

```python
cv=3
```

means 3-fold CV.

```python
cv=5
```

means 5-fold CV.

```python
cv=10
```

means 10-fold CV.

---

# 13. Which `cv` Should You Use?

A common choice:

```python
cv=5
```

or:

```python
cv=10
```

### `cv=5`

Good general-purpose choice.

Advantages:

- relatively fast
- enough validation variation
- commonly used

### `cv=10`

Provides more folds.

Advantages:

- more training/validation evaluations
- potentially more reliable estimate

Disadvantage:

- computationally more expensive

---

# 14. Small Dataset vs Large Dataset

### Small dataset

You may prefer:

```python
cv=5
```

or:

```python
cv=10
```

because you want to use your limited data efficiently.

### Very large dataset

You might use:

```python
cv=5
```

because each model training can already be expensive.

There is no universal rule like:

> "Always use 10-fold."

You choose based on:

- dataset size
- computational cost
- problem type

---

# 15. Classification: Stratified K-Fold

Here's an important detail.

Suppose your dataset contains:

```text
Class 0 → 900 samples
Class 1 → 100 samples
```

This is imbalanced.

A normal K-Fold split could accidentally create folds with different class distributions.

For classification, scikit-learn generally uses **StratifiedKFold** when `cv` is an integer.

Stratification tries to preserve the class distribution in each fold.

For example:

```text
Entire dataset:
90% Class 0
10% Class 1

Fold 1:
90% Class 0
10% Class 1

Fold 2:
90% Class 0
10% Class 1

...
```

This is usually what you want for classification.

---

# 16. Explicit `StratifiedKFold`

You can also define it yourself:

```python
from sklearn.model_selection import StratifiedKFold

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)
```

Then:

```python
scores = cross_val_score(
    model,
    X,
    y,
    cv=cv
)
```

This gives you more control.

---

# 17. Why `shuffle=True`?

Suppose your dataset is sorted:

```text
Rows 1-100 → Class 0
Rows 101-200 → Class 1
```

If you don't shuffle, your folds might have undesirable distributions.

So:

```python
shuffle=True
```

randomizes the samples before creating folds.

And:

```python
random_state=42
```

makes the split reproducible.

---

# 18. Regression Cross-Validation

For regression, there are no classes to stratify.

You can use:

```python
from sklearn.model_selection import KFold

cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)
```

Then:

```python
scores = cross_val_score(
    model,
    X,
    y,
    cv=cv,
    scoring="r2"
)
```

---

# 19. `scoring` Parameter

This is extremely important.

By default, `cross_val_score()` uses the estimator's default scoring method.

But you can explicitly tell it what metric to use.

For classification:

```python
scoring="accuracy"
```

Example:

```python
scores = cross_val_score(
    model,
    X,
    y,
    cv=5,
    scoring="accuracy"
)
```

---

# 20. Other Classification Metrics

You can use:

```python
scoring="precision"
```

```python
scoring="recall"
```

```python
scoring="f1"
```

```python
scoring="roc_auc"
```

For example:

```python
scores = cross_val_score(
    model,
    X,
    y,
    cv=5,
    scoring="f1"
)
```

---

# 21. Regression Metrics

Common ones:

```python
scoring="r2"
```

```python
scoring="neg_mean_squared_error"
```

```python
scoring="neg_mean_absolute_error"
```

Notice something strange:

```python
neg_mean_squared_error
```

Why negative?

---

# 22. Why Does Scikit-Learn Use `neg_`?

Scikit-learn's scoring system follows the idea:

> **Higher score is better.**

But MSE is:

$$
MSE = \frac{1}{n}\sum(y_i-\hat y_i)^2
$$

Lower is better.

So scikit-learn returns:

```text
-MSE
```

For example:

```text
Actual MSE = 25
cross_val_score → -25
```

You can convert it:

```python
mse_scores = -scores
```

Then:

```python
print(mse_scores.mean())
```

gives:

```text
25
```

---

# 23. Multiple Metrics

Sometimes you want:

```text
Accuracy
Precision
Recall
F1
ROC-AUC
```

In that situation, `cross_val_score()` isn't enough because it returns one metric at a time.

Use:

```python
cross_validate()
```

Example:

```python
from sklearn.model_selection import cross_validate

scoring = [
    "accuracy",
    "precision",
    "recall",
    "f1"
]

results = cross_validate(
    model,
    X,
    y,
    cv=5,
    scoring=scoring
)
```

Then:

```python
results["test_accuracy"]
results["test_precision"]
results["test_recall"]
results["test_f1"]
```

---

# 24. Mean and Standard Deviation

Don't just look at:

```python
scores.mean()
```

Also look at:

```python
scores.std()
```

Suppose Model A:

```text
[0.91, 0.92, 0.90, 0.91, 0.92]
```

Mean:

```text
0.912
```

Std:

```text
~0.007
```

Very stable.

Model B:

```text
[0.70, 0.99, 0.82, 0.95, 0.87]
```

Mean might be similar:

```text
~0.866
```

but standard deviation is much higher.

This means Model B's performance varies considerably across folds.

So:

$$
\boxed{\text{Mean = average performance}}
$$

$$
\boxed{\text{Std = stability/variation}}
$$

---

# 25. A Very Useful Interpretation

Suppose:

```python
scores = [0.91, 0.92, 0.90, 0.91, 0.92]
```

Then:

```text
Mean = 0.912
Std  = 0.007
```

You can think:

> "My model generally performs around 91.2%, and its performance is quite stable."

---

# 26. Comparing Models with CV

This is where `cross_val_score()` becomes very useful.

```python
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "SVM": SVC()
}
```

Then:

```python
for name, model in models.items():

    scores = cross_val_score(
        model,
        X,
        y,
        cv=5,
        scoring="accuracy"
    )

    print(
        name,
        scores.mean(),
        scores.std()
    )
```

You might get:

```text
Logistic Regression  0.95  0.02
Decision Tree        0.92  0.04
SVM                  0.97  0.01
```

Now you have much more information than a single train/test score.

---

# 27. Cross-Validation in Hyperparameter Tuning

This is one of the **most important uses**.

Suppose you're tuning a Decision Tree.

You want to determine:

```python
max_depth
```

Try:

```text
max_depth=2
max_depth=3
max_depth=4
max_depth=5
max_depth=10
```

Instead of one train/test split, you can evaluate each using CV.

For example:

```text
max_depth    CV Accuracy

2            0.82
3            0.87
4            0.90
5            0.91
10           0.86
```

You'd choose:

```text
max_depth=5
```

because it performs best under CV.

---

# 28. `GridSearchCV`

Instead of manually doing this, use:

```python
from sklearn.model_selection import GridSearchCV
```

Example:

```python
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(random_state=42)

params = {
    "max_depth": [2, 3, 4, 5, 10],
    "min_samples_split": [2, 5, 10]
}
```

Then:

```python
grid = GridSearchCV(
    model,
    params,
    cv=5,
    scoring="accuracy"
)
```

Train:

```python
grid.fit(X_train, y_train)
```

Then:

```python
print(grid.best_params_)
```

and:

```python
print(grid.best_score_)
```

Here `best_score_` is the **mean cross-validation score of the best parameter combination**.

---

# 29. Cross-Validation and Overfitting

CV is also useful for detecting overfitting.

Imagine:

```text
Training Accuracy = 99%
CV Accuracy       = 82%
```

That's a warning sign.

The model performs extremely well on training data but much worse on unseen folds.

Likely:

$$
\boxed{\text{Overfitting}}
$$

---

# 30. Underfitting

Suppose:

```text
Training Accuracy = 70%
CV Accuracy       = 68%
```

Both are poor.

That could indicate:

$$
\boxed{\text{Underfitting}}
$$

The model isn't sufficiently learning the underlying patterns.

---

# 31. The Relationship

A useful mental model:

```text
                  Model
                    │
          ┌─────────┴─────────┐
          ↓                   ↓
   Training Performance   CV Performance
          │                   │
          ↓                   ↓
       How well             How well
       did it learn?        does it generalize?
```

Roughly:

| Training | CV | Possible situation |
|---:|---:|---|
| High | High | Good |
| High | Low | Overfitting |
| Low | Low | Underfitting |
| Low | High | Usually investigate; may indicate unusual split/data issues |

---

# 32. VERY Important: Data Leakage

Cross-validation does **not automatically protect you from data leakage**.

Consider scaling.

Bad:

```python
scaler.fit_transform(X)
```

before CV.

Why?

Because the scaler learns:

$$
mean,\ standard\ deviation
$$

from the **entire dataset**, including validation folds.

That's information leakage.

Instead, use a Pipeline:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=1000))
])
```

Then:

```python
scores = cross_val_score(
    model,
    X,
    y,
    cv=5
)
```

Now the scaler is fitted separately inside each training fold.

This is the **correct approach**.

---

# 33. Cross-Validation with Your ML Workflow

For your current ML learning, think of the workflow like this:

```text
                 Dataset
                    │
                    ↓
             Train/Test Split
                    │
          ┌─────────┴──────────┐
          ↓                    ↓
      Training                Test
          │                    │
          ↓                    │
    Cross Validation           │
          │                    │
     ┌────┼────┐               │
     ↓    ↓    ↓               │
    CV1  CV2  CV3 ...           │
     │    │    │               │
     └────┼────┘               │
          ↓                    │
      Mean CV Score             │
          │                    │
          ↓                    │
   Select/Tune Model            │
          │                    │
          ↓                    │
   Fit on ALL Training          │
          │                    │
          └──────────┐         │
                     ↓         ↓
                  Final Test Evaluation
```

This is the workflow you should remember.

---

# 34. When Should You Use `cross_val_score()`?

### Use it when:

**1. Comparing models**

```text
LR vs SVM vs DT
```

**2. Estimating generalization performance**

You want a more reliable performance estimate than one split.

**3. Checking model stability**

Look at:

```python
scores.mean()
scores.std()
```

**4. Evaluating a baseline model**

Before doing complicated hyperparameter tuning.

**5. Evaluating a model after preprocessing**

Especially with Pipelines.

---

# 35. When Should You NOT Use Normal K-Fold CV?

There are important exceptions.

## Time-Series Data

Suppose you're predicting:

```text
2024 → 2025 → 2026
```

You cannot randomly mix future data into training.

For example:

```text
Train: 2026
Validate: 2024
```

would be nonsense because you're using the future to predict the past.

Instead, use:

```python
TimeSeriesSplit
```

Conceptually:

```text
Fold 1:
Train → Jan
Valid → Feb

Fold 2:
Train → Jan + Feb
Valid → Mar

Fold 3:
Train → Jan + Feb + Mar
Valid → Apr
```

This respects temporal ordering.

This is especially important for **forecasting problems**, such as an AQI predictor.

---

# 36. Grouped Data

Suppose you have medical data:

```text
Patient A → 10 records
Patient B → 10 records
Patient C → 10 records
```

You don't want records from Patient A in both training and validation.

Otherwise the model might effectively see the same person's information in both sets.

Use:

```python
GroupKFold
```

The same principle applies to:

- multiple measurements per person
- multiple images from the same subject
- multiple samples from the same machine
- multiple observations from the same location

---

# 37. Cross-Validation vs Train/Test Split

| Feature | Train/Test Split | Cross-Validation |
|---|---|---|
| Number of validation splits | 1 | Multiple |
| Training runs | 1 | K times |
| Computational cost | Low | Higher |
| Performance estimate | More dependent on split | More robust |
| Good for quick testing | ✅ | |
| Model comparison | Okay | ✅ |
| Hyperparameter tuning | Limited | ✅ |
| Final evaluation | ✅ Test set | ❌ Usually not the final test |

---

# 38. One Common Mistake

Don't do:

```python
scores = cross_val_score(
    model,
    X_test,
    y_test,
    cv=5
)
```

That's not how you normally use CV.

Your test set should be kept untouched.

Instead:

```python
scores = cross_val_score(
    model,
    X_train,
    y_train,
    cv=5
)
```

Then finally:

```python
model.fit(X_train, y_train)

model.score(X_test, y_test)
```

---

# 39. The One-Line Definition

For your notes:

> **Cross-validation is a resampling technique used to evaluate a machine-learning model by repeatedly training it on different subsets of the training data and validating it on the remaining subset.**

And:

> **`cross_val_score()` performs cross-validation and returns the evaluation score for each fold.**

---

# 40. Most Important Code to Remember

### Basic

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(
    model,
    X_train,
    y_train,
    cv=5
)

print(scores)
print("Mean:", scores.mean())
print("Std:", scores.std())
```

### With metric

```python
scores = cross_val_score(
    model,
    X_train,
    y_train,
    cv=5,
    scoring="accuracy"
)
```

### Classification with explicit StratifiedKFold

```python
from sklearn.model_selection import StratifiedKFold

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scores = cross_val_score(
    model,
    X_train,
    y_train,
    cv=cv,
    scoring="accuracy"
)
```

### Regression

```python
scores = cross_val_score(
    model,
    X_train,
    y_train,
    cv=5,
    scoring="r2"
)
```

---

## 🧠 Final Mental Model

Don't think of:

```python
cv=5
```

as simply "split my dataset into 5."

Think:

> **"Train and validate my model 5 different times, using a different part of the training data for validation each time, then summarize how consistently the model performs."**

So:

$$
\boxed{
\text{CV Score}
=
\text{Average performance across validation folds}
}
$$

and:

$$
\boxed{
\text{CV Standard Deviation}
=
\text{How much performance varies across folds}
}
$$

And the practical workflow is:

```text
Train/Test Split
       ↓
Keep Test Set untouched
       ↓
Cross-Validation on Training Set
       ↓
Compare/Tune Models
       ↓
Fit Best Model on Entire Training Set
       ↓
Evaluate ONCE on Test Set
```

That distinction between **CV score and final test score** is one of the most important concepts to understand before moving into `GridSearchCV`, `RandomizedSearchCV`, and hyperparameter tuning.