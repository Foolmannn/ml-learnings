# Stacking Ensemble Learning — In Detail

**Stacking (Stacked Generalization)** is an ensemble learning technique where we train **multiple different models (base learners)** and then train another model called a **meta-model** to learn how to combine their predictions.

The key idea is:

> **Instead of manually deciding how to combine models, let another machine-learning model learn the best combination.**

For example, suppose we have:

* Logistic Regression
* Decision Tree
* Random Forest
* SVM

Instead of choosing one model, stacking allows these models to make predictions and then gives their predictions to another model, such as Logistic Regression, which makes the **final prediction**.

---

# 1. Why Do We Need Stacking?

Suppose you are solving a classification problem.

You train:

```text
Logistic Regression → 85% accuracy
Decision Tree       → 82% accuracy
Random Forest       → 88% accuracy
SVM                 → 86% accuracy
```

Each model has different strengths.

For example:

* Logistic Regression may capture linear relationships well.
* Decision Tree may capture nonlinear rules.
* Random Forest may be robust to noise.
* SVM may find useful decision boundaries.

The important observation is:

> **Different models make different mistakes.**

If we can intelligently combine their predictions, the combined model may generalize better.

Stacking attempts to learn **how much each model should contribute depending on the input**.

---

# 2. Basic Architecture

A stacking ensemble has at least two levels.

```text
                 Original Features
                       X
                       │
          ┌────────────┼────────────┐
          │            │            │
          ▼            ▼            ▼
     Model 1       Model 2       Model 3
     Logistic      Decision       Random
     Regression      Tree         Forest
          │            │            │
          ▼            ▼            ▼
       pred₁         pred₂        pred₃
          │            │            │
          └────────────┼────────────┘
                       │
                       ▼
                Meta Features
                       │
                       ▼
                 Meta Model
                (Logistic Reg.)
                       │
                       ▼
                Final Prediction
```

There are therefore two major levels:

### Level 0 — Base models

These are the original models.

```text
Model 1
Model 2
Model 3
...
Model n
```

### Level 1 — Meta-model

The meta-model receives predictions from the base models.

```text
Base predictions
      ↓
Meta-model
      ↓
Final prediction
```

---

# 3. Example

Imagine a binary classification problem:

```text
Predict whether a customer will purchase a product.

0 → No
1 → Yes
```

We train three base models:

```text
Logistic Regression
Random Forest
SVM
```

For a new customer:

```text
Logistic Regression → 0.70
Random Forest       → 0.90
SVM                 → 0.60
```

These predictions become input to the meta-model:

```text
[0.70, 0.90, 0.60]
```

The meta-model might learn something like:

```text
Final probability = f(0.70, 0.90, 0.60)
```

and produce:

```text
0.82
```

Therefore:

```text
0.82 > 0.5

Final class = 1
```

---

# 4. Stacking vs Voting

This is one of the most important distinctions.

## Voting

In voting, we specify how models should be combined.

For example:

```text
Model 1 → Yes
Model 2 → Yes
Model 3 → No
```

Majority vote:

```text
Yes
```

The combination rule is predetermined.

---

## Stacking

Stacking uses another ML model to learn the combination.

```text
Model 1 → 0.70
Model 2 → 0.90
Model 3 → 0.60
             ↓
        Meta-model
             ↓
          0.82
```

So:

| Voting                        | Stacking            |
| ----------------------------- | ------------------- |
| Fixed combination rule        | Learned combination |
| Majority/weighted voting      | Meta-model          |
| No separate ML model required | Requires meta-model |
| Simpler                       | More flexible       |
| Usually easier to interpret   | More complex        |

A useful way to remember:

> **Voting combines predictions directly. Stacking learns how to combine predictions.**

---

# 5. Stacking vs Bagging

These are also very different.

### Bagging

Bagging generally trains multiple models using different bootstrap samples.

```text
Dataset
   │
   ├── Sample 1 → Model 1
   ├── Sample 2 → Model 2
   ├── Sample 3 → Model 3
   └── Sample 4 → Model 4
                    │
                    ▼
                 Combine
```

The models are usually trained **independently/in parallel**.

Example:

```text
Random Forest
```

is a classic bagging-based algorithm.

---

### Stacking

Stacking has a hierarchy:

```text
Dataset
   │
   ├── Model 1 ──┐
   ├── Model 2 ──┤
   └── Model 3 ──┤
                 ▼
             Meta-model
                 ▼
            Final output
```

So:

> **Bagging focuses on reducing variance through multiple learners, while stacking focuses on learning how to combine different learners.**

---

# 6. Stacking vs Boosting

### Boosting

Models are usually trained sequentially.

```text
Model 1
   ↓
Model 2
   ↓
Model 3
   ↓
Model 4
```

Each new learner attempts to improve the ensemble by focusing on errors/residuals from previous learners, depending on the boosting algorithm.

Examples:

* AdaBoost
* Gradient Boosting
* XGBoost
* LightGBM
* CatBoost

---

### Stacking

Models generally learn different patterns independently.

```text
Model 1 ──┐
Model 2 ──┼──→ Meta-model
Model 3 ──┘
```

The meta-model learns how to combine them.

---

# 7. The Most Important Problem: Data Leakage

This is the most important concept when learning stacking.

Suppose we train a base model:

```python
model.fit(X_train, y_train)

pred = model.predict(X_train)
```

Now we use:

```text
pred
```

as input to the meta-model.

The problem is that the base model already saw those training samples.

Therefore, these predictions can be overly optimistic.

For example:

```text
Training data
      ↓
Base model
      ↓
Prediction on SAME training data
      ↓
Meta-model
```

This can cause **data leakage**.

---

# 8. Out-of-Fold Predictions

The standard solution is to use **out-of-fold (OOF) predictions**.

This is one of the most important concepts in stacking.

Suppose:

```text
Training dataset = 1000 samples
```

Use 5-fold cross-validation.

```text
Fold 1 → validation
Fold 2 → validation
Fold 3 → validation
Fold 4 → validation
Fold 5 → validation
```

For each fold:

### Fold 1

```text
Train → folds 2,3,4,5
Validate → fold 1
```

Generate predictions for fold 1.

### Fold 2

```text
Train → folds 1,3,4,5
Validate → fold 2
```

Generate predictions for fold 2.

Continue until every sample has received a prediction from a model that **did not train on that sample**.

---

# 9. OOF Prediction Example

Suppose we have:

```text
10 samples
```

and 5-fold CV.

```text
Fold 1 → samples 1,2
Fold 2 → samples 3,4
Fold 3 → samples 5,6
Fold 4 → samples 7,8
Fold 5 → samples 9,10
```

For Fold 1:

```text
Train → 3,4,5,6,7,8,9,10
Predict → 1,2
```

For Fold 2:

```text
Train → 1,2,5,6,7,8,9,10
Predict → 3,4
```

Eventually:

```text
Sample    OOF Prediction
1         0.71
2         0.65
3         0.83
4         0.42
5         0.91
...
```

Now these predictions are much more appropriate for training the meta-model.

---

# 10. Why OOF Predictions Matter

Without OOF:

```text
X_train
   ↓
Base model
   ↓
prediction on X_train
   ↓
Meta-model
```

Potential leakage.

With OOF:

```text
X_train
   ↓
5-fold CV
   ↓
Out-of-fold predictions
   ↓
Meta-model
```

Each OOF prediction is produced by a model that did not train on that particular sample.

Therefore:

> **The meta-model receives predictions that better represent how the base model behaves on unseen data.**

---

# 11. Mathematical Representation

Suppose our original dataset is:

$$
X \in \mathbb{R}^{n \times p}
$$

and target:

$$
y
$$

We have \(K\) base models:

$$
f_1, f_2, ..., f_K
$$

Each base model produces:

$$
\hat{y}_1=f_1(X)
$$

$$
\hat{y}_2=f_2(X)
$$

$$
...
$$

$$
\hat{y}_K=f_K(X)
$$

We construct a new dataset:

$$
Z =
[\hat{y}_1,\hat{y}_2,...,\hat{y}_K]
$$

Then train a meta-model:

$$
g(Z)
$$

Final prediction:

$$
\hat{y}_{final}
=
g(f_1(X),f_2(X),...,f_K(X))
$$

This is the core mathematical idea of stacking.

---

# 12. Classification Stacking

For classification, base models can provide:

### Class predictions

```text
0
1
0
1
```

or probabilities:

```text
0.72
0.91
0.31
0.64
```

Usually, probability predictions are more informative.

For binary classification:

$$
P(y=1|X)
$$

might be:

```text
Logistic Regression → 0.72
Random Forest       → 0.91
SVM                 → 0.63
```

Meta-model receives:

$$
[0.72,0.91,0.63]
$$

---

# 13. Multiclass Classification

Suppose there are three classes:

```text
Cat
Dog
Horse
```

A base model might output:

```text
Cat    = 0.20
Dog    = 0.70
Horse  = 0.10
```

Another:

```text
Cat    = 0.10
Dog    = 0.60
Horse  = 0.30
```

If we have three base models, the meta-model can receive:

```text
Model 1:
[0.20, 0.70, 0.10]

Model 2:
[0.10, 0.60, 0.30]

Model 3:
[0.15, 0.75, 0.10]
```

These can be combined into meta-features.

---

# 14. Stacking for Regression

Stacking isn't limited to classification.

Suppose we're predicting house prices.

Base models:

```text
Linear Regression
Random Forest Regressor
Gradient Boosting Regressor
SVR
```

For one house:

```text
Linear Regression       → $250,000
Random Forest           → $280,000
Gradient Boosting       → $275,000
SVR                     → $260,000
```

Meta-model receives:

```text
[250000, 280000, 275000, 260000]
```

and might predict:

```text
$272,000
```

Mathematically:

$$
\hat{y}
=
g(\hat{y}_1,\hat{y}_2,\hat{y}_3,\hat{y}_4)
$$

---

# 15. What Should We Use as the Meta-Model?

The meta-model is sometimes called:

* Meta-learner
* Level-1 learner
* Blender

Common choices include:

### Classification

```text
Logistic Regression
```

### Regression

```text
Linear Regression
```

These are often good starting choices because they are relatively simple.

You can also use:

```text
Random Forest
Gradient Boosting
XGBoost
SVM
Neural Network
```

But a highly complex meta-model can overfit.

A common principle is:

> **Keep the meta-model relatively simple initially.**

---

# 16. Why Logistic Regression Is Common as a Meta-Model

Suppose three base classifiers produce:

$$
p_1,p_2,p_3
$$

Logistic regression can learn:

$$
z =
w_1p_1+w_2p_2+w_3p_3+b
$$

Then:

$$
P(y=1)=\sigma(z)
$$

where:

$$
\sigma(z)=\frac{1}{1+e^{-z}}
$$

This allows the meta-model to learn different weights.

For example:

```text
Base model             Learned importance

Logistic Regression        0.20
Random Forest              0.55
SVM                        0.25
```

These values are illustrative, not something guaranteed by stacking.

The key point is that the meta-model learns the combination from data.

---

# 17. Homogeneous vs Heterogeneous Stacking

## Homogeneous stacking

Uses similar types of models.

Example:

```text
Decision Tree
Decision Tree
Decision Tree
```

But with different hyperparameters or training conditions.

---

## Heterogeneous stacking

Uses different algorithms.

For example:

```text
Logistic Regression
Random Forest
SVM
KNN
```

This is often particularly useful because different algorithms may learn different structures in the data.

---

# 18. Choosing Base Models

Don't simply add as many models as possible.

Good base models should ideally have:

### 1. Good individual performance

A model should have some predictive value.

### 2. Diversity

Models should make somewhat different errors.

For example:

```text
Logistic Regression → linear patterns
Decision Tree       → rule-based nonlinear patterns
SVM                 → margin-based boundaries
Random Forest       → ensemble of trees
```

If all models make exactly the same predictions, stacking provides little additional information.

---

# 19. Diversity Is Extremely Important

Consider:

```text
Model A → 90%
Model B → 90%
Model C → 90%
```

That looks good.

But suppose all three make exactly the same mistakes.

Then stacking has little additional information.

Compare:

```text
Model A → 88%
Model B → 87%
Model C → 86%
```

but their errors are substantially different.

The meta-model may potentially benefit more from this diversity.

So:

> **Performance + diversity is often more useful than performance alone when selecting base learners.**

---

# 20. Feature-Level vs Prediction-Level Learning

Base models work with the original features:

```text
X
 ↓
Base models
```

The meta-model works primarily with the **predictions produced by the base models**:

```text
X
 ↓
Base models
 ↓
Predictions
 ↓
Meta-model
```

Therefore, stacking creates a second feature space.

Original:

```text
Age
Income
Education
Experience
```

Meta features:

```text
LR_prediction
RF_prediction
SVM_prediction
```

---

# 21. Important: `passthrough`

In scikit-learn stacking, there is an option:

```python
passthrough=True
```

Normally:

```text
Original X
   ↓
Base models
   ↓
Predictions
   ↓
Meta-model
```

With `passthrough=True`:

```text
                ┌───────────────┐
Original X ─────┤               │
                │  Meta-model   │
Base predictions┤               │
                └───────────────┘
```

So the meta-model receives:

```text
[original features + base predictions]
```

This can sometimes improve performance because the meta-model has access to the original information as well.

---

# 22. Stacking in Scikit-Learn

For classification:

```python
from sklearn.ensemble import StackingClassifier
```

For regression:

```python
from sklearn.ensemble import StackingRegressor
```

---

# 23. Classification Implementation

Let's use the Iris dataset.

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import StackingClassifier
```

Load data:

```python
X, y = load_iris(return_X_y=True)
```

Split:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

---

# 24. Define Base Models

```python
estimators = [
    (
        "lr",
        make_pipeline(
            StandardScaler(),
            LogisticRegression()
        )
    ),

    (
        "dt",
        DecisionTreeClassifier(
            max_depth=5,
            random_state=42
        )
    ),

    (
        "rf",
        RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )
    )
]
```

Notice the structure:

```python
[
    ("name", model),
    ("name", model),
    ("name", model)
]
```

---

# 25. Define the Meta-Model

```python
meta_model = LogisticRegression()
```

Then:

```python
stacking_model = StackingClassifier(
    estimators=estimators,
    final_estimator=meta_model
)
```

Here:

```text
estimators
     ↓
Base models

final_estimator
     ↓
Meta-model
```

---

# 26. Train

```python
stacking_model.fit(X_train, y_train)
```

Then:

```python
y_pred = stacking_model.predict(X_test)
```

Evaluate:

```python
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)

print(accuracy)
```

---

# 27. Cross-Validation Parameter

One important parameter is:

```python
cv=5
```

Example:

```python
stacking_model = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(),
    cv=5
)
```

This controls the cross-validation used to generate the predictions for the meta-model.

Conceptually:

```text
Training data
     ↓
    5-fold CV
     ↓
OOF predictions
     ↓
Meta-model
```

---

# 28. `stack_method`

For classification, another important parameter is:

```python
stack_method
```

Possible values include:

```text
'auto'
'predict_proba'
'decision_function'
'predict'
```

### `auto`

Scikit-learn chooses an appropriate method.

### `predict_proba`

Uses class probabilities.

Example:

```text
[0.10, 0.80, 0.10]
```

### `decision_function`

Uses the model's decision scores where supported.

### `predict`

Uses class labels.

Example:

```text
0
1
1
```

Generally, probabilities contain more information than hard class labels when available.

---

# 29. `passthrough`

Example:

```python
stacking_model = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(),
    cv=5,
    passthrough=True
)
```

Now the final estimator gets:

```text
Original features
       +
Base-model predictions
```

instead of only:

```text
Base-model predictions
```

---

# 30. Regression Implementation

For regression:

```python
from sklearn.ensemble import StackingRegressor
```

Example:

```python
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
```

Base models:

```python
estimators = [
    (
        "lr",
        LinearRegression()
    ),

    (
        "dt",
        DecisionTreeRegressor(
            max_depth=5,
            random_state=42
        )
    ),

    (
        "rf",
        RandomForestRegressor(
            n_estimators=100,
            random_state=42
        )
    )
]
```

Meta-model:

```python
final_estimator = LinearRegression()
```

Create stack:

```python
stacking_regressor = StackingRegressor(
    estimators=estimators,
    final_estimator=final_estimator,
    cv=5
)
```

Train:

```python
stacking_regressor.fit(X_train, y_train)
```

Predict:

```python
y_pred = stacking_regressor.predict(X_test)
```

---

# 31. Complete Conceptual Workflow

The entire stacking process looks like this:

```text
                  Training Data
                       │
             ┌─────────┼─────────┐
             │         │         │
             ▼         ▼         ▼
          Model A    Model B    Model C
             │         │         │
             ▼         ▼         ▼
           OOF-A     OOF-B     OOF-C
             │         │         │
             └─────────┼─────────┘
                       │
                       ▼
              Meta Training Data
                       │
                       ▼
                  Meta-model
                       │
                       ▼
               Final prediction
```

For test data:

```text
                 X_test
                    │
          ┌─────────┼─────────┐
          ▼         ▼         ▼
       Model A   Model B   Model C
          │         │         │
          ▼         ▼         ▼
        pred A    pred B    pred C
          │         │         │
          └─────────┼─────────┘
                    │
                    ▼
                Meta-model
                    │
                    ▼
              Final prediction
```

---

# 32. What Happens Internally?

This is worth understanding because `StackingClassifier` can look like a black box.

Suppose:

```python
estimators = [
    ("lr", LogisticRegression()),
    ("rf", RandomForestClassifier()),
    ("svm", SVC(probability=True))
]
```

and:

```python
cv=5
```

Conceptually, scikit-learn generates cross-validated predictions from each base estimator.

You might end up with something conceptually like:

```text
Sample     LR       RF       SVM       Target

1          0.72     0.81     0.65       1
2          0.21     0.15     0.30       0
3          0.83     0.91     0.77       1
4          0.40     0.25     0.35       0
...
```

The meta-model learns:

```text
Target
  ↑
  │
LR prediction
RF prediction
SVM prediction
```

Then for unseen data:

```text
New sample
    ↓
LR → 0.65
RF → 0.84
SVM → 0.72
    ↓
Meta-model
    ↓
Final probability
```

---

# 33. Stacking with Your Previous Ensemble Topics

Since you've been studying ensemble learning in sequence, you can organize them like this:

```text
                    Ensemble Learning
                           │
           ┌───────────────┼────────────────┐
           │               │                │
        Bagging          Boosting         Stacking
           │               │                │
     Random Forest     AdaBoost          Meta-model
           │           Gradient Boosting      │
           │              XGBoost             │
           │                                  │
           └──────────────┐        ┌──────────┘
                          │        │
                       Voting   Stacking
```

More specifically:

### Bagging

```text
Many models
     ↓
Aggregation
```

### Boosting

```text
Model 1
   ↓
Model 2
   ↓
Model 3
   ↓
...
```

### Voting

```text
Model 1 ──┐
Model 2 ──┼──→ Vote
Model 3 ──┘
```

### Stacking

```text
Model 1 ──┐
Model 2 ──┼──→ Meta-model
Model 3 ──┘
```

---

# 34. Advantages of Stacking

### 1. Combines different algorithms

You can combine:

```text
Linear models
Tree models
Kernel models
Nearest-neighbor models
```

---

### 2. Learns the combination

Unlike simple voting, the meta-model learns how to combine predictions.

---

### 3. Can capture complementary patterns

For example:

```text
Logistic Regression
      ↓
Linear relationships

Random Forest
      ↓
Nonlinear interactions

SVM
      ↓
Complex decision boundaries
```

The meta-model can potentially exploit these complementary predictions.

---

### 4. Flexible

You can choose different:

* base models
* meta-models
* hyperparameters
* cross-validation strategies

---

# 35. Disadvantages

### 1. More computationally expensive

If you have:

```text
5 base models
5-fold CV
```

many model fits are required.

This becomes especially expensive when base models themselves are expensive.

---

### 2. More complex

A simple model:

```text
RandomForest → prediction
```

is easier to understand than:

```text
Model A
Model B
Model C
   ↓
OOF predictions
   ↓
Meta-model
   ↓
Prediction
```

---

### 3. Can overfit

Especially if:

* dataset is small
* meta-model is too complex
* base models are too similar
* cross-validation is poorly configured

---

### 4. More difficult to tune

You potentially need to tune:

```text
Base model 1
Base model 2
Base model 3
Meta-model
CV strategy
```

---

# 36. When Should You Use Stacking?

Stacking can be useful when:

### You have several reasonably strong models

For example:

```text
Logistic Regression
Random Forest
SVM
Gradient Boosting
```

and their predictions differ meaningfully.

---

### You want to improve predictive performance

Stacking is particularly useful when individual models capture different aspects of the data.

---

### You have enough data

Because stacking adds another learning layer, having sufficient training data is useful.

---

# 37. When Should You Avoid It?

Avoid unnecessary stacking when:

```text
Dataset is extremely small
```

or:

```text
One model already performs very well
```

or:

```text
Models are highly correlated
```

or:

```text
Computational resources are limited
```

Also remember:

> More models does **not** automatically mean better performance.

---

# 38. Important Hyperparameters in `StackingClassifier`

You should know these for practical ML work.

### `estimators`

Base models.

```python
estimators=[
    ("lr", LogisticRegression()),
    ("rf", RandomForestClassifier())
]
```

---

### `final_estimator`

Meta-model.

```python
final_estimator=LogisticRegression()
```

---

### `cv`

Cross-validation strategy.

```python
cv=5
```

---

### `stack_method`

How base predictions are generated.

```python
stack_method="auto"
```

or:

```python
stack_method="predict_proba"
```

---

### `passthrough`

Whether original features are also given to the meta-model.

```python
passthrough=False
```

Default behavior is to use base-model outputs without directly passing the original features.

---

### `n_jobs`

Controls parallel processing where supported.

```python
n_jobs=-1
```

This can be useful when training multiple base models is computationally expensive.

---

# 39. A Very Important Practical Rule

Suppose you have:

```text
Model A → 95%
Model B → 94%
Model C → 93%
```

You might think:

> "Let's stack them."

But first check whether their predictions/errors are different.

If:

```text
A errors ≈ B errors ≈ C errors
```

then stacking may add little.

If:

```text
A catches some cases B misses
B catches some cases C misses
C catches some cases A misses
```

then stacking has more useful information to learn from.

---

# 40. Stacking in One Sentence

If you remember only one thing:

> **Stacking trains multiple base models, uses their out-of-fold predictions as features, and trains a meta-model to learn how to combine those predictions into the final prediction.**

The complete mental model is:

```text
                 ORIGINAL DATA
                      │
        ┌─────────────┼─────────────┐
        │             │             │
        ▼             ▼             ▼
      Model A       Model B       Model C
        │             │             │
        └─────────────┼─────────────┘
                      │
              OOF PREDICTIONS
                      │
                      ▼
               META FEATURES
                      │
                      ▼
                META MODEL
                      │
                      ▼
              FINAL PREDICTION
```

### The key concepts to study next

For your ensemble-learning progression, I would focus on these in order:

1. **Stacking architecture**
2. **OOF predictions and why they prevent leakage**
3. **Stacking vs Voting**
4. **Classification with `StackingClassifier`**
5. **Regression with `StackingRegressor`**
6. **`cv`, `stack_method`, `passthrough`, `n_jobs`**
7. **Choosing diverse base learners**
8. **Hyperparameter tuning of stacking**
9. **Evaluation with cross-validation**
10. **Blending vs Stacking**
