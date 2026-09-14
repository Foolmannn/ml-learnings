 **Complete study note for Logistic Regression in `scikit-learn`**, including the important hyperparameters, what they mean mathematically, how they affect the model, and how they interact with **polynomial features, regularization, multiclass classification, optimization, and preprocessing**.

# Logistic Regression in Scikit-Learn — Complete Detailed Notes

---

# 1. What is Logistic Regression?

Logistic Regression is a supervised learning algorithm primarily used for **classification**.

For binary classification:

$$
y\in\{0,1\}
$$

The model first computes:

$$
z=w^Tx+b
$$

and then applies the sigmoid function:

$$
\sigma(z)=\frac{1}{1+e^{-z}}
$$

Therefore:

$$
P(y=1|X)=\frac{1}{1+e^{-(w^TX+b)}}
$$

The output is a probability between 0 and 1.

For example:

```text
P(y=1) = 0.92
```

means the model estimates a 92% probability of class 1.

---

# 2. Basic Scikit-Learn implementation

The main class is:

```python
from sklearn.linear_model import LogisticRegression
```

Basic usage:

```python
model = LogisticRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
```

Probability prediction:

```python
y_prob = model.predict_proba(X_test)
```

Decision scores:

```python
scores = model.decision_function(X_test)
```

---

# 3. Complete syntax

The constructor contains many hyperparameters:

```python
LogisticRegression(
    penalty='l2',
    *,
    dual=False,
    tol=1e-4,
    C=1.0,
    fit_intercept=True,
    intercept_scaling=1,
    class_weight=None,
    random_state=None,
    solver='lbfgs',
    max_iter=100,
    multi_class='deprecated',
    verbose=0,
    warm_start=False,
    n_jobs=None,
    l1_ratio=None
)
```

Some defaults and supported options vary by scikit-learn version, so always check the version-specific documentation when exact API behavior matters.

The **most important parameters to understand** are:

```text
penalty
C
solver
max_iter
tol
class_weight
fit_intercept
random_state
```

For polynomial Logistic Regression, also understand:

```text
PolynomialFeatures(degree)
PolynomialFeatures(include_bias)
PolynomialFeatures(interaction_only)
```

---

# 4. `penalty`

```python
penalty='l2'
```

Controls **regularization**.

Available choices depend on the solver:

```text
'l1'
'l2'
'elasticnet'
None
```

---

# 5. Why do we need regularization?

Suppose our Logistic Regression has:

$$
z=w_1x_1+w_2x_2+b
$$

The standard binary cross-entropy loss is:

$$
J(w)=
-\frac{1}{m}
\sum_{i=1}^{m}
[
y_i\log(\hat y_i)
+
(1-y_i)\log(1-\hat y_i)
]
$$

Without regularization, the model may produce very large coefficients, especially when:

- there are many features
- features are highly correlated
- data is nearly perfectly separable
- polynomial features are used

Regularization adds a penalty.

---

# 6. L2 Regularization

With L2:

$$
J(w)
=
Loss+
\lambda\sum_jw_j^2
$$

Large coefficients are penalized.

In sklearn:

```python
LogisticRegression(
    penalty='l2'
)
```

L2 generally:

- reduces coefficient magnitude
- improves stability
- helps prevent overfitting
- works well with correlated features

---

# 7. L1 Regularization

L1 adds:

$$
J(w)
=
Loss+
\lambda\sum_j|w_j|
$$

The important property is that L1 can make coefficients exactly zero.

Therefore it can perform a form of:

> **Feature selection**

Example:

```python
model = LogisticRegression(
    penalty='l1',
    solver='liblinear'
)
```

Suppose:

```text
feature       coefficient

age              1.42
income           0.00
education       -0.83
noise_feature    0.00
```

The zero coefficients mean those features aren't being used by the fitted linear predictor.

---

# 8. L1 vs L2

| Property | L1 | L2 |
|---|---|---|
| Penalty | $|w|$ | $w^2$ |
| Sparse coefficients | ✅ | Usually no |
| Feature selection | ✅ | ❌ |
| Correlated features | Can select one/some | Tends to distribute weights |
| Stability | Good | Very good |
| Common default | No | Yes |

---

# 9. Elastic Net

Elastic Net combines L1 and L2:

$$
J(w)=
Loss+
\lambda
\left[
\alpha\sum_j|w_j|
+
(1-\alpha)\sum_jw_j^2
\right]
$$

In sklearn:

```python
model = LogisticRegression(
    penalty='elasticnet',
    solver='saga',
    l1_ratio=0.5
)
```

Here:

```python
l1_ratio=0
```

is pure L2-like behavior, while:

```python
l1_ratio=1
```

is pure L1-like behavior.

Values between 0 and 1 mix the two.

---

# 10. `C` — one of the most important hyperparameters

```python
C=1.0
```

This is extremely important.

In sklearn:

$$
\boxed{C=\frac{1}{\lambda}}
$$

conceptually.

Therefore:

### Small C

```python
C=0.01
```

means:

$$
C\downarrow
\Rightarrow
regularization\uparrow
$$

The model is strongly constrained.

### Large C

```python
C=100
```

means:

$$
C\uparrow
\Rightarrow
regularization\downarrow
$$

The model is allowed to fit the training data more aggressively.

---

# 11. Effect of `C`

Think of:

```text
C = 0.001
```

Very strong regularization.

```text
C = 0.01
```

Strong regularization.

```text
C = 1
```

Moderate/default-style setting.

```text
C = 100
```

Weak regularization.

```text
C = 1000
```

Very weak regularization.

The exact best value depends on the dataset.

---

# 12. `C` and underfitting/overfitting

Generally:

$$
C\downarrow
$$

means a simpler/more constrained model.

Potential problem:

> Underfitting

While:

$$
C\uparrow
$$

allows a more flexible fit.

Potential problem:

> Overfitting

But remember: this interaction depends on the data and other settings.

---

# 13. `solver`

The solver is the optimization algorithm used to find the model parameters.

Common choices include:

```text
lbfgs
liblinear
newton-cg
newton-cholesky
sag
saga
```

This is another **very important hyperparameter**.

---

# 14. `solver='lbfgs'`

```python
LogisticRegression(
    solver='lbfgs'
)
```

`lbfgs` is a quasi-Newton optimization method.

It is a very good general-purpose choice.

Advantages:

- good default for many problems
- works well for dense data
- supports L2 regularization
- supports multinomial classification

For many normal datasets:

```python
LogisticRegression(
    solver='lbfgs',
    max_iter=1000
)
```

is an excellent starting point.

---

# 15. `solver='liblinear'`

```python
LogisticRegression(
    solver='liblinear'
)
```

`liblinear` is useful particularly for:

- relatively small datasets
- binary classification
- L1 regularization

Example:

```python
LogisticRegression(
    solver='liblinear',
    penalty='l1'
)
```

However, it is not generally the first choice for large-scale multiclass problems.

---

# 16. `solver='saga'`

```python
LogisticRegression(
    solver='saga'
)
```

SAGA is particularly useful for:

- large datasets
- sparse datasets
- L1
- L2
- Elastic Net

For Elastic Net:

```python
LogisticRegression(
    solver='saga',
    penalty='elasticnet',
    l1_ratio=0.5
)
```

This is a very important combination to remember.

---

# 17. `solver='sag'`

SAG = Stochastic Average Gradient.

Useful for:

- large datasets
- relatively large sample sizes
- scaled features

Feature scaling is particularly important for SAG/SAGA convergence.

---

# 18. `solver='newton-cg'`

Newton-CG uses a Newton-style optimization approach.

Useful when:

- the dataset is suitable for second-order optimization
- you want an alternative to L-BFGS

Supports L2-style regularization rather than L1/Elastic Net.

---

# 19. `solver='newton-cholesky'`

This solver uses a Newton method with a Cholesky-based approach.

It can be especially useful when the number of samples is large compared with the number of features, although the Hessian-related memory cost can become important when the feature dimension is large.

This becomes particularly relevant after polynomial feature expansion because polynomial features can dramatically increase dimensionality.

---

# 20. Solver compatibility

A useful mental table:

| Solver | L1 | L2 | Elastic Net | None | Multinomial |
|---|---:|---:|---:|---:|---:|
| `lbfgs` | ❌ | ✅ | ❌ | ✅ | ✅ |
| `liblinear` | ✅ | ✅ | ❌ | ❌ | ❌ |
| `newton-cg` | ❌ | ✅ | ❌ | ✅ | ✅ |
| `newton-cholesky` | ❌ | ✅ | ❌ | ✅ | depends on sklearn version |
| `sag` | ❌ | ✅ | ❌ | ✅ | ✅ |
| `saga` | ✅ | ✅ | ✅ | ✅ | ✅ |

Check your installed sklearn version for exact current compatibility.

---

# 21. `max_iter`

```python
max_iter=100
```

Controls the maximum number of optimization iterations.

If you get:

```text
ConvergenceWarning:
```

you can often increase:

```python
max_iter=1000
```

or:

```python
max_iter=5000
```

For example:

```python
model = LogisticRegression(
    max_iter=1000
)
```

---

# 22. Does increasing `max_iter` improve accuracy?

Not necessarily.

This is important.

`max_iter` controls **how long the optimizer is allowed to search**, not model complexity directly.

If your model hasn't converged:

```text
max_iter too low
       ↓
optimization stops early
       ↓
parameters may not be fully optimized
```

Increasing it allows more optimization steps.

---

# 23. `tol`

```python
tol=1e-4
```

`tol` controls the stopping tolerance for optimization.

Conceptually:

```text
optimization progress becomes sufficiently small
                    ↓
               stop training
```

Smaller `tol`:

```python
tol=1e-5
```

usually means stricter convergence.

Larger `tol`:

```python
tol=1e-3
```

can stop earlier.

---

# 24. `tol` vs `max_iter`

These two are often confused.

### `max_iter`

Maximum number of iterations.

### `tol`

How close the optimization needs to get before stopping.

So:

```text
max_iter → maximum allowed work
tol      → convergence precision
```

---

# 25. `fit_intercept`

Default:

```python
fit_intercept=True
```

The Logistic Regression equation is:

$$
z=w^Tx+b
$$

Here:

$$
b
$$

is the intercept.

With:

```python
fit_intercept=True
```

the model learns $b$.

With:

```python
fit_intercept=False
```

the model becomes:

$$
z=w^Tx
$$

There is no learned intercept.

Generally:

> Keep `fit_intercept=True` unless you have a specific reason not to.

---

# 26. `intercept_scaling`

This is primarily relevant to the `liblinear` solver when `fit_intercept=True`.

It changes how the intercept is represented internally.

For most users:

```python
intercept_scaling=1
```

can simply be left alone.

It is not one of the first hyperparameters you should tune.

---

# 27. `class_weight`

This is extremely important for **imbalanced datasets**.

Suppose:

```text
Class 0 → 950 samples
Class 1 → 50 samples
```

The model may become biased toward class 0.

You can use:

```python
class_weight='balanced'
```

Example:

```python
model = LogisticRegression(
    class_weight='balanced'
)
```

Sklearn automatically assigns weights inversely proportional to class frequencies.

Conceptually:

$$
w_k\propto\frac{1}{n_k}
$$

where $n_k$ is the number of samples in class $k$.

---

# 28. Manual class weights

You can also specify:

```python
class_weight={
    0: 1,
    1: 5
}
```

This makes errors on class 1 more costly.

Useful when:

- false negatives are expensive
- one class is rare
- classes have different importance

---

# 29. `random_state`

```python
random_state=42
```

Controls randomness where the selected solver/model procedure uses randomness.

For reproducibility:

```python
model = LogisticRegression(
    random_state=42
)
```

However, not every solver relies on randomness in the same way.

---

# 30. `multi_class`

This parameter has changed across recent scikit-learn versions, with multinomial behavior becoming increasingly standardized/deprecated as a user-selectable option.

Conceptually, Logistic Regression supports:

### Binary classification

$$
K=2
$$

### Multiclass classification

$$
K>2
$$

For multiclass classification, the model can learn a probability for each class.

For example:

```text
Class 0 → 0.10
Class 1 → 0.70
Class 2 → 0.20
```

Prediction:

```text
Class 1
```

---

# 31. Multinomial Logistic Regression

For $K$ classes:

$$
P(y=k|x)
=
\frac{e^{w_k^Tx+b_k}}
{\sum_{j=1}^{K}e^{w_j^Tx+b_j}}
$$

This is the **softmax function**.

So:

```text
Binary Logistic Regression
        ↓
      Sigmoid

Multiclass Logistic Regression
        ↓
      Softmax
```

This connects directly to the Softmax Regression topic you studied.

---

# 32. `n_jobs`

```python
n_jobs=None
```

Controls parallelism for parts of the computation where supported.

For example:

```python
n_jobs=-1
```

means use all available CPU cores where the implementation supports it.

But don't assume every Logistic Regression solver benefits equally from `n_jobs`.

---

# 33. `warm_start`

```python
warm_start=False
```

If:

```python
warm_start=True
```

the previous solution can be reused as initialization for a subsequent `.fit()` call.

This can be useful when fitting related models sequentially.

Example idea:

```text
C=0.1
   ↓
fit

C=1
   ↓
reuse previous coefficients

C=10
   ↓
reuse previous coefficients
```

It's more specialized and usually isn't necessary for beginners.

---

# 34. `verbose`

```python
verbose=0
```

Controls the amount of solver output.

For example:

```python
verbose=1
```

can provide optimization information for supported solvers.

Mostly useful for debugging or understanding convergence.

---

# 35. `l1_ratio`

Only relevant for:

```python
penalty='elasticnet'
```

and compatible solvers such as:

```python
solver='saga'
```

Example:

```python
LogisticRegression(
    penalty='elasticnet',
    solver='saga',
    l1_ratio=0.5
)
```

Interpretation:

$$
l1\_ratio=0
$$

→ L2 component dominates.

$$
l1\_ratio=1
$$

→ L1 component dominates.

$$
0<l1\_ratio<1
$$

→ mixture.

---

# 36. Important parameter combinations

### Standard Logistic Regression

```python
LogisticRegression(
    penalty='l2',
    C=1.0,
    solver='lbfgs',
    max_iter=1000
)
```

---

### L1 Logistic Regression

```python
LogisticRegression(
    penalty='l1',
    C=1.0,
    solver='liblinear',
    max_iter=1000
)
```

or `saga` when appropriate.

---

### Elastic Net

```python
LogisticRegression(
    penalty='elasticnet',
    C=1.0,
    solver='saga',
    l1_ratio=0.5,
    max_iter=1000
)
```

---

### Imbalanced classification

```python
LogisticRegression(
    class_weight='balanced',
    max_iter=1000
)
```

---

# 37. `coef_`

After fitting:

```python
model.fit(X_train, y_train)

print(model.coef_)
```

For binary Logistic Regression:

```text
[[w1, w2, w3, ...]]
```

These are the learned coefficients.

Suppose:

$$
w_1=2
$$

Then increasing $x_1$ generally increases the log-odds, holding other features fixed.

---

# 38. `intercept_`

```python
print(model.intercept_)
```

This gives:

$$
b
$$

from:

$$
z=w^Tx+b
$$

---

# 39. Log-odds interpretation

This is a very important statistical interpretation.

Logistic Regression models:

$$
\log\left(\frac{p}{1-p}\right)
=
w^Tx+b
$$

where:

$$
\frac{p}{1-p}
$$

is the **odds**.

For one feature:

$$
\log\left(\frac{p}{1-p}\right)
=
w_0+w_1x
$$

Therefore:

$$
e^{w_1}
$$

is the multiplicative change in odds for a one-unit increase in $x$, holding the other features fixed.

---

# 40. `predict()`

```python
y_pred = model.predict(X_test)
```

Returns the predicted class.

For binary classification, conceptually:

$$
P(y=1)\geq0.5
\Rightarrow
1
$$

otherwise:

$$
0
$$

---

# 41. `predict_proba()`

```python
model.predict_proba(X_test)
```

Example:

```text
[[0.20, 0.80]]
```

means:

```text
P(class 0) = 0.20
P(class 1) = 0.80
```

This is often more useful than `predict()` when you care about risk/probability.

---

# 42. `decision_function()`

```python
model.decision_function(X_test)
```

Returns the raw score:

$$
z=w^Tx+b
$$

For binary Logistic Regression:

$$
p=\sigma(z)
$$

Therefore:

```text
decision_function
        ↓
       z
        ↓
     sigmoid
        ↓
 probability
```

---

# 43. Changing the classification threshold

The default classification threshold is commonly:

$$
0.5
$$

But you don't always need 0.5.

Suppose:

```text
P(y=1) = 0.40
```

Default:

```text
0.40 < 0.50
→ class 0
```

If you choose:

$$
threshold=0.30
$$

then:

```text
0.40 > 0.30
→ class 1
```

Lower threshold generally increases recall for the positive class while potentially increasing false positives.

This is especially important for:

- medical screening
- fraud detection
- anomaly detection
- safety systems

---

# 44. Polynomial Logistic Regression in sklearn

Now combine this with your previous topic.

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
```

Create:

```python
poly = PolynomialFeatures(
    degree=2
)

X_poly = poly.fit_transform(X)
```

Then:

```python
model = LogisticRegression()

model.fit(X_poly, y)
```

---

# 45. Better: Pipeline

Always prefer a pipeline in a real project.

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

model = Pipeline([
    ("poly", PolynomialFeatures(degree=2)),
    ("scaler", StandardScaler()),
    ("logistic", LogisticRegression(
        C=1.0,
        penalty="l2",
        solver="lbfgs",
        max_iter=1000
    ))
])
```

Then:

```python
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
```

---

# 46. Why scaling is important with polynomial features

Suppose:

$$
x=100
$$

Then:

$$
x^2=10,000
$$

and:

$$
x^3=1,000,000
$$

Now different polynomial features can have drastically different scales.

Therefore:

```text
PolynomialFeatures
       ↓
StandardScaler
       ↓
LogisticRegression
```

is often a good pipeline.

---

# 47. `PolynomialFeatures` hyperparameters

The important parameters are:

```python
PolynomialFeatures(
    degree=2,
    interaction_only=False,
    include_bias=True
)
```

---

# 48. `degree`

Controls maximum polynomial degree.

```python
PolynomialFeatures(degree=1)
```

→ original features.

```python
PolynomialFeatures(degree=2)
```

→ quadratic terms.

```python
PolynomialFeatures(degree=3)
```

→ cubic terms.

Higher degree:

$$
\Rightarrow
\text{more features}
\Rightarrow
\text{more complexity}
\Rightarrow
\text{higher overfitting risk}
$$

---

# 49. `interaction_only`

Default:

```python
interaction_only=False
```

Then:

```text
x1
x2
x1²
x1*x2
x2²
```

With:

```python
interaction_only=True
```

you don't generate powers such as:

$$
x_1^2
$$

but retain interaction terms such as:

$$
x_1x_2
$$

This can be useful when you specifically want feature interactions.

---

# 50. `include_bias`

Default:

```python
include_bias=True
```

This adds a column of ones:

$$
1
$$

If Logistic Regression already has:

```python
fit_intercept=True
```

you generally don't need a separate polynomial bias column.

A common setup is:

```python
PolynomialFeatures(
    degree=2,
    include_bias=False
)
```

followed by:

```python
LogisticRegression(
    fit_intercept=True
)
```

This avoids an unnecessary duplicate constant feature.

---

# 51. Full Polynomial Logistic Regression pipeline

```python
model = Pipeline([
    (
        "poly",
        PolynomialFeatures(
            degree=2,
            include_bias=False
        )
    ),

    (
        "scaler",
        StandardScaler()
    ),

    (
        "logistic",
        LogisticRegression(
            penalty="l2",
            C=1.0,
            solver="lbfgs",
            max_iter=1000
        )
    )
])
```

This is a very strong baseline.

---

# 52. Hyperparameter tuning

You shouldn't manually guess the best `C`.

Use:

```python
from sklearn.model_selection import GridSearchCV
```

Example:

```python
param_grid = {
    "logistic__C": [
        0.001,
        0.01,
        0.1,
        1,
        10,
        100
    ]
}
```

Then:

```python
grid = GridSearchCV(
    model,
    param_grid,
    cv=5,
    scoring="accuracy"
)

grid.fit(X_train, y_train)
```

Get the best model:

```python
best_model = grid.best_estimator_
```

Best hyperparameters:

```python
print(grid.best_params_)
```

---

# 53. Tuning polynomial Logistic Regression

You can tune both degree and `C`.

```python
param_grid = {
    "poly__degree": [1, 2, 3, 4],
    "logistic__C": [
        0.01,
        0.1,
        1,
        10,
        100
    ]
}
```

Then:

```python
grid = GridSearchCV(
    model,
    param_grid,
    cv=5,
    scoring="accuracy"
)

grid.fit(X_train, y_train)
```

This searches combinations such as:

```text
degree=1, C=0.01
degree=1, C=0.1
degree=1, C=1
...
degree=2, C=0.01
degree=2, C=0.1
...
```

---

# 54. Important: avoid data leakage

Don't do this before splitting:

```python
X_poly = poly.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(...)
```

Because transformations have been fitted using the entire dataset.

Instead:

```text
Raw X
 │
 ├──── Train
 │      │
 │      ▼
 │   fit transformations
 │
 └──── Test
        │
        ▼
     transform only
```

The pipeline automatically handles this correctly.

---

# 55. Evaluation metrics

Don't rely only on accuracy.

For binary Logistic Regression, commonly use:

### Accuracy

$$
Accuracy=
\frac{TP+TN}{TP+TN+FP+FN}
$$

### Precision

$$
Precision=
\frac{TP}{TP+FP}
$$

### Recall

$$
Recall=
\frac{TP}{TP+FN}
$$

### F1

$$
F1=
2\frac{Precision\cdot Recall}
{Precision+Recall}
$$

And:

### ROC-AUC

Measures ranking/discrimination across thresholds.

### PR-AUC / Average Precision

Often especially informative when the positive class is rare.

---

# 56. A good practical configuration

For a normal tabular dataset:

```python
model = Pipeline([
    ("scaler", StandardScaler()),
    ("logistic", LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="lbfgs",
        max_iter=1000
    ))
])
```

For polynomial features:

```python
model = Pipeline([
    ("poly", PolynomialFeatures(
        degree=2,
        include_bias=False
    )),

    ("scaler", StandardScaler()),

    ("logistic", LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="lbfgs",
        max_iter=1000
    ))
])
```

For imbalanced data:

```python
model = Pipeline([
    ("scaler", StandardScaler()),
    ("logistic", LogisticRegression(
        penalty="l2",
        C=1.0,
        class_weight="balanced",
        solver="lbfgs",
        max_iter=1000
    ))
])
```

---

# 57. Which hyperparameters should you actually tune?

You **don't need to tune everything**.

Start with:

### 1. `C`

Most important.

Try:

```python
[0.001, 0.01, 0.1, 1, 10, 100]
```

### 2. `penalty`

Try:

```text
l1
l2
elasticnet
```

only with compatible solvers.

### 3. `solver`

Usually choose based on:

```text
dataset size
sparsity
penalty
multiclass requirements
```

rather than blindly tuning it.

### 4. `class_weight`

Consider:

```python
None
"balanced"
```

when the dataset is imbalanced.

### 5. `degree`

For polynomial Logistic Regression:

```python
[1, 2, 3, 4]
```

Usually don't jump immediately to very high degrees.

### 6. `l1_ratio`

Only for Elastic Net:

```python
[0.1, 0.3, 0.5, 0.7, 0.9]
```

---

# 58. A complete GridSearch example

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

pipeline = Pipeline([
    ("poly", PolynomialFeatures(
        include_bias=False
    )),

    ("scaler", StandardScaler()),

    ("logistic", LogisticRegression(
        max_iter=2000
    ))
])

param_grid = {
    "poly__degree": [1, 2, 3],

    "logistic__C": [
        0.01,
        0.1,
        1,
        10,
        100
    ],

    "logistic__penalty": [
        "l2"
    ],

    "logistic__solver": [
        "lbfgs"
    ]
}

grid = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

grid.fit(X_train, y_train)

print("Best parameters:")
print(grid.best_params_)

print("Best CV score:")
print(grid.best_score_)
```

---

# 59. Common mistakes

### Mistake 1 — Forgetting scaling

Especially problematic with:

```text
SAG
SAGA
polynomial features
large-valued features
```

Use:

```python
StandardScaler()
```

when appropriate.

---

### Mistake 2 — Using L1 with `lbfgs`

This is invalid:

```python
LogisticRegression(
    penalty="l1",
    solver="lbfgs"
)
```

Use a compatible solver such as:

```python
solver="liblinear"
```

or:

```python
solver="saga"
```

depending on the problem.

---

### Mistake 3 — Polynomial expansion before train/test split

Can cause leakage.

Use a pipeline.

---

### Mistake 4 — Very high polynomial degree

For example:

```python
degree=15
```

can explode the number of features and produce severe overfitting.

---

### Mistake 5 — Increasing `C` blindly

Large:

```python
C=100000
```

isn't automatically better.

It means very weak regularization.

---

### Mistake 6 — Treating `max_iter` as a regularization parameter

It isn't.

`max_iter` controls optimization iterations.

`C` controls regularization strength.

---

# 60. Hyperparameter cheat sheet

| Hyperparameter | Purpose | Typical values |
|---|---|---|
| `penalty` | Regularization type | `l1`, `l2`, `elasticnet`, `None` |
| `C` | Inverse regularization strength | `0.001`–`100+` |
| `solver` | Optimization algorithm | `lbfgs`, `liblinear`, `saga`, etc. |
| `max_iter` | Maximum optimization iterations | `100–5000` |
| `tol` | Convergence tolerance | `1e-3`–`1e-5` |
| `fit_intercept` | Learn $b$ | `True/False` |
| `class_weight` | Handle class imbalance | `None`, `"balanced"` |
| `random_state` | Reproducibility | `42`, etc. |
| `l1_ratio` | L1/L2 mixture | `0–1` |
| `n_jobs` | Parallelism where supported | `None`, `-1` |
| `warm_start` | Reuse previous solution | `True/False` |
| `verbose` | Solver output | `0`, `1`, etc. |

For polynomial features:

| Parameter | Purpose |
|---|---|
| `degree` | Polynomial complexity |
| `interaction_only` | Only interaction terms |
| `include_bias` | Include constant feature |

---

# 61. The parameters I would memorize

For ML interviews and practical work, prioritize these:

```text
LogisticRegression
│
├── penalty
│
├── C ⭐⭐⭐
│
├── solver ⭐⭐⭐
│
├── max_iter ⭐⭐
│
├── tol ⭐⭐
│
├── class_weight ⭐⭐⭐
│
├── fit_intercept
│
└── l1_ratio
```

And understand this relationship:

$$
\boxed{
C=\frac{1}{\lambda}
}
$$

so:

$$
\boxed{
C\uparrow\Rightarrow Regularization\downarrow
}
$$

$$
\boxed{
C\downarrow\Rightarrow Regularization\uparrow
}
$$

For your **Polynomial Logistic Regression**:

$$
\boxed{
Degree\uparrow
\Rightarrow
Features\uparrow
\Rightarrow
Complexity\uparrow
\Rightarrow
Overfitting\ risk\uparrow
}
$$

while:

$$
\boxed{
C\downarrow
\Rightarrow
Regularization\uparrow
\Rightarrow
Model\ constrained
}
$$

---

# 62. The big picture

You can now think of the complete sklearn workflow as:

```text
                 DATA
                   │
                   ▼
            Train/Test Split
                   │
                   ▼
        ┌─────────────────────┐
        │ Feature Engineering │
        │                     │
        │ PolynomialFeatures  │
        └──────────┬──────────┘
                   │
                   ▼
          StandardScaler
                   │
                   ▼
          LogisticRegression
                   │
        ┌──────────┼──────────┐
        ▼          ▼          ▼
     penalty       C        solver
        │          │          │
        └──────────┼──────────┘
                   ▼
              Optimization
                   │
                   ▼
             Probability
                   │
                   ▼
              Threshold
                   │
                   ▼
              Prediction
                   │
                   ▼
             Evaluation
```

The **three most important ideas** to carry forward are:

1. **`penalty` determines the type of regularization.**
2. **`C` controls the strength of regularization (`C` is inverse to regularization strength).**
3. **`solver` determines how the optimization is performed and must be compatible with the chosen penalty.**

And for your current topic:

$$
\boxed{
\text{Polynomial Features}
+
\text{Logistic Regression}
=
\text{Nonlinear decision boundary}
}
$$

while the Logistic Regression optimization is still fundamentally **linear in the transformed features**.