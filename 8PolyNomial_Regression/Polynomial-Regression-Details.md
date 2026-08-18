
# Polynomial Regression — In Detail

Polynomial Regression is one of the most important extensions of **Linear Regression**. It is used when the relationship between the input feature(s) and target is **non-linear**, but we still want to use a linear-regression-based model.

The key idea is:

> **Transform the input features into polynomial features, then apply Linear Regression to those transformed features.**

---

## 1. Why do we need Polynomial Regression?

Suppose we have this dataset:

| Experience | Salary |
|---:|---:|
| 1 | 35 |
| 2 | 42 |
| 3 | 50 |
| 4 | 61 |
| 5 | 75 |
| 6 | 92 |

If we plot it, the relationship may look curved rather than like a straight line.

Ordinary Linear Regression tries to fit:

$$
y = b_0 + b_1x
$$

So it can only produce a **straight line**.



For a curved relationship, we can instead use:

$$
y = b_0 + b_1x + b_2x^2
$$

or

$$
y = b_0 + b_1x + b_2x^2 + b_3x^3
$$

and so on.

This is Polynomial Regression.

---

# 2. What exactly is Polynomial Regression?

For a single feature $x$, a polynomial regression model of degree $n$ is:

$$
\hat y =
b_0+b_1x+b_2x^2+b_3x^3+\cdots+b_nx^n
$$

For example, degree 2:

$$
\boxed{\hat y=b_0+b_1x+b_2x^2}
$$

Degree 3:

$$
\boxed{\hat y=b_0+b_1x+b_2x^2+b_3x^3}
$$

Degree 4:

$$
\boxed{\hat y=b_0+b_1x+b_2x^2+b_3x^3+b_4x^4}
$$

The important thing to notice is that the model is **non-linear with respect to $x$** but **linear with respect to the parameters $b_0,b_1,b_2,\ldots$**.

That's why we can still use Linear Regression.

---

# 3. The most important concept

Consider:

$$
y=b_0+b_1x+b_2x^2
$$

You might think:

> "This isn't Linear Regression because there is $x^2$."

But look at the parameters:

$$
y=b_0+b_1(x)+b_2(x^2)
$$

We can define:

$$
x_1=x
$$

$$
x_2=x^2
$$

Then:

$$
y=b_0+b_1x_1+b_2x_2
$$

Now it is simply Linear Regression with **two features**:

$$
X=
\begin{bmatrix}
x_1 & x_2
\end{bmatrix}
$$

Therefore:

> **Polynomial Regression is Linear Regression performed on polynomially transformed features.**

This is probably the single most important thing to understand.

---

# 4. Example

Suppose:

```python
X = [1, 2, 3, 4, 5]
```

For degree 2, we transform it into:

| X | X² |
|---:|---:|
| 1 | 1 |
| 2 | 4 |
| 3 | 9 |
| 4 | 16 |
| 5 | 25 |

So instead of giving Linear Regression:

```text
X
```

we give:

```text
X   X²
```

The model learns:

$$
y=b_0+b_1X+b_2X^2
$$

---

# 5. How PolynomialFeatures works in Scikit-Learn

Scikit-learn provides:

```python
from sklearn.preprocessing import PolynomialFeatures
```

Let's create some data:

```python
import numpy as np

X = np.array([
    [1],
    [2],
    [3],
    [4],
    [5]
])
```

Now:

```python
poly = PolynomialFeatures(degree=2)

X_poly = poly.fit_transform(X)

print(X_poly)
```

Output:

```text
[[ 1.  1.  1.]
 [ 1.  2.  4.]
 [ 1.  3.  9.]
 [ 1.  4. 16.]
 [ 1.  5. 25.]]
```

The columns represent:

```text
1     X     X²
```

The first column is the **bias/intercept column**.

So:

$$
X_{poly}=
\begin{bmatrix}
1&1&1\\
1&2&4\\
1&3&9\\
1&4&16\\
1&5&25
\end{bmatrix}
$$

---

# 6. Now apply Linear Regression

```python
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2)

X_poly = poly.fit_transform(X)

model = LinearRegression()

model.fit(X_poly, y)
```

Prediction:

```python
y_pred = model.predict(X_poly)
```

The model has effectively learned:

$$
\hat y=b_0+b_1x+b_2x^2
$$

---

# 7. Why is it called "Polynomial" Regression?

Because the model contains polynomial terms:

$$
x^2,\quad x^3,\quad x^4,\ldots
$$

For example:

### Degree 1

$$
y=b_0+b_1x
$$

Straight line.

### Degree 2

$$
y=b_0+b_1x+b_2x^2
$$

Parabolic curve.

### Degree 3

$$
y=b_0+b_1x+b_2x^2+b_3x^3
$$

Can create more complex curves.

### Degree 4

$$
y=b_0+b_1x+b_2x^2+b_3x^3+b_4x^4
$$

Even more flexibility.

---

# 8. Degree is extremely important

The **degree** controls the complexity of the model.

Suppose the actual data follows a curved pattern.

### Degree 1

```text
      *
    *
  *
 *
*____________
```

The model is too simple.

This is **underfitting**.

---

### Degree 2

```text
*          *
 \        /
  \______/
```

Much better.

---

### Degree 10

The model can become extremely flexible:

```text
*\/\___/\__/\/*
```

It may start following every small fluctuation in the training data.

This is **overfitting**.

Therefore:

$$
\boxed{\text{Higher degree} \neq \text{always better}}
$$

---

# 9. Underfitting vs Overfitting

This is one of the most important parts of Polynomial Regression.

## Underfitting

Model is too simple.

For example:

```python
PolynomialFeatures(degree=1)
```

The model cannot capture the curvature.

Typically:

```text
Training error: HIGH
Validation error: HIGH
```

---

## Good fit

A suitable degree captures the underlying relationship.

```text
Training error: LOW
Validation error: LOW
```

---

## Overfitting

Degree is too high.

For example:

```python
PolynomialFeatures(degree=15)
```

The model may memorize the training data.

```text
Training error: VERY LOW
Validation error: HIGH
```

The model performs well on data it has seen but poorly on unseen data.

---

# 10. Visual intuition

Imagine the actual data is:

```text
y
│
│             *
│          *
│       *
│    *
│  *
│ *
└──────────────── x
```

Linear Regression might produce:

```text
│             *
│          * /
│       *  /
│    *   /
│  *    /
│ *    /
└──────/──────── x
```

Polynomial Regression can bend itself:

```text
│             *
│          *
│       *
│    *
│  *
│ *
└──────────────── x
```

The important difference is that the polynomial model can capture **curvature**.

---

# 11. Polynomial Regression with a real dataset

Let's create a simple dataset.

```python
import numpy as np
import matplotlib.pyplot as plt

X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1)

y = np.array([
    2, 5, 10, 17, 26,
    37, 50, 65, 82, 101
])
```

Notice the relationship approximately follows:

$$
y=x^2+1
$$

Now let's fit Linear Regression.

```python
from sklearn.linear_model import LinearRegression

linear_model = LinearRegression()

linear_model.fit(X, y)

y_pred = linear_model.predict(X)
```

The model can only fit:

$$
y=b_0+b_1x
$$

It won't capture the curve very well.

---

# 12. Polynomial Regression

Let's use degree 2.

```python
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2)

X_poly = poly.fit_transform(X)

model = LinearRegression()

model.fit(X_poly, y)

y_pred = model.predict(X_poly)
```

Now the model is:

$$
y=b_0+b_1x+b_2x^2
$$

Since our data is approximately quadratic, this should fit very well.

---

# 13. Plotting the Polynomial Regression

```python
plt.scatter(X, y)

plt.plot(X, y_pred)

plt.xlabel("X")
plt.ylabel("Y")

plt.show()
```

You should see a curved fitted relationship.

For a cleaner curve, sort the X values and generate many points:

```python
X_grid = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)

X_grid_poly = poly.transform(X_grid)

y_grid_pred = model.predict(X_grid_poly)

plt.scatter(X, y)

plt.plot(X_grid, y_grid_pred)

plt.show()
```

This is preferable because the line is drawn using many closely spaced points.

---

# 14. Polynomial Regression mathematically

Suppose we have:

$$
X=
\begin{bmatrix}
x_1\\
x_2\\
x_3\\
\vdots\\
x_m
\end{bmatrix}
$$

For degree 2, transform it into:

$$
X'=
\begin{bmatrix}
1&x_1&x_1^2\\
1&x_2&x_2^2\\
1&x_3&x_3^2\\
\vdots&\vdots&\vdots\\
1&x_m&x_m^2
\end{bmatrix}
$$

Then ordinary Linear Regression solves:

$$
\hat y=X'\theta
$$

where:

$$
\theta=
\begin{bmatrix}
b_0\\
b_1\\
b_2
\end{bmatrix}
$$

The ordinary least-squares solution is:

$$
\theta=(X'^TX')^{-1}X'^Ty
$$

So Polynomial Regression itself doesn't require a completely different regression algorithm.

The major change is:

$$
\boxed{X\rightarrow PolynomialFeatures(X)}
$$

followed by:

$$
\boxed{LinearRegression(X_{poly},y)}
$$

---

# 15. Multiple Features

This is where Polynomial Regression becomes particularly interesting.

Suppose we have two features:

$$
x_1,\quad x_2
$$

For degree 2, PolynomialFeatures generates:

$$
1,x_1,x_2,x_1^2,x_1x_2,x_2^2
$$

So the model becomes:

$$
y =
b_0
+b_1x_1
+b_2x_2
+b_3x_1^2
+b_4x_1x_2
+b_5x_2^2
$$

Notice the new term:

$$
x_1x_2
$$

This is called an **interaction term**.

---

# 16. Example with two features

Suppose:

```python
X = [
    [2, 3],
    [3, 4],
    [4, 5]
]
```

Using:

```python
poly = PolynomialFeatures(degree=2)

X_poly = poly.fit_transform(X)
```

The generated features are:

```text
1
x1
x2
x1²
x1*x2
x2²
```

So PolynomialFeatures doesn't just generate powers.

It also generates **feature interactions**.

---

# 17. What does `degree=3` generate?

For two features:

```python
PolynomialFeatures(degree=3)
```

generates:

$$
1
$$

$$
x_1,x_2
$$

$$
x_1^2,x_1x_2,x_2^2
$$

$$
x_1^3,x_1^2x_2,x_1x_2^2,x_2^3
$$

So the number of features can increase rapidly.

---

# 18. Why feature explosion is dangerous

Suppose you have:

```text
10 original features
```

and use:

```python
PolynomialFeatures(degree=5)
```

You can end up with a **large number of generated features**.

This causes:

### 1. Higher computational cost

Training becomes more expensive.

### 2. More memory usage

The transformed matrix becomes much larger.

### 3. Overfitting

The model becomes extremely flexible.

### 4. Multicollinearity

Polynomial features can be highly correlated.

For example:

$$
x,\ x^2,\ x^3
$$

can have strong relationships with each other.

---

# 19. `include_bias`

You will often see:

```python
PolynomialFeatures(
    degree=2,
    include_bias=False
)
```

Why?

By default:

```python
PolynomialFeatures(degree=2)
```

creates:

```text
1
X
X²
```

But `LinearRegression` already has an intercept by default.

Therefore, we can avoid generating the `1` column:

```python
poly = PolynomialFeatures(
    degree=2,
    include_bias=False
)
```

Then:

```text
X
X²
```

and Linear Regression handles:

$$
b_0
$$

itself.

This is often cleaner.

---

# 20. Complete Scikit-Learn implementation

```python
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Data
X = np.array([
    [1],
    [2],
    [3],
    [4],
    [5],
    [6],
    [7],
    [8],
    [9],
    [10]
])

y = np.array([
    2,
    5,
    10,
    17,
    26,
    37,
    50,
    65,
    82,
    101
])

# Polynomial transformation
poly = PolynomialFeatures(
    degree=2,
    include_bias=False
)

X_poly = poly.fit_transform(X)

# Linear Regression
model = LinearRegression()

model.fit(X_poly, y)

# Prediction
y_pred = model.predict(X_poly)

# Plot
plt.scatter(X, y)

plt.plot(X, y_pred)

plt.xlabel("X")
plt.ylabel("Y")

plt.show()
```

---

# 21. Understanding the pipeline

The whole process is:

```text
Original Data
     ↓
X
     ↓
PolynomialFeatures
     ↓
X, X², X³, ...
     ↓
LinearRegression
     ↓
Prediction
```

In code:

```python
poly = PolynomialFeatures(degree=2)

X_poly = poly.fit_transform(X)

model = LinearRegression()

model.fit(X_poly, y)
```

This is the fundamental implementation.

---

# 22. Using Pipeline

Instead of manually transforming the data, we can use:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

model = Pipeline([
    ("poly", PolynomialFeatures(degree=2)),
    ("linear", LinearRegression())
])

model.fit(X, y)

y_pred = model.predict(X)
```

This is usually better practice.

Why?

Because the transformation and model stay together.

---

# 23. Train/Test Split

You should **not** evaluate your polynomial model only on training data.

Use:

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

Then:

```python
model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)
```

And evaluate both.

---

# 24. Evaluation metrics

For regression, you can use:

### R²

$$
R^2=1-\frac{SS_{res}}{SS_{tot}}
$$

In Python:

```python
from sklearn.metrics import r2_score

print("Train R2:", r2_score(y_train, y_train_pred))
print("Test R2:", r2_score(y_test, y_test_pred))
```

You can also use:

```python
from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y_test, y_test_pred)

print(mse)
```

And:

```python
from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(y_test, y_test_pred)

print(mae)
```

---

# 25. How to select the polynomial degree

This is a major practical question.

Try:

```python
degrees = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
```

For each degree:

```python
for degree in degrees:

    model = Pipeline([
        ("poly", PolynomialFeatures(degree=degree)),
        ("linear", LinearRegression())
    ])

    model.fit(X_train, y_train)

    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)

    print(
        degree,
        train_score,
        test_score
    )
```

You might observe something like:

```text
Degree    Train R²    Test R²

1         0.72        0.68
2         0.94        0.91
3         0.97        0.89
4         0.99        0.82
5         1.00        0.71
```

Here, degree 2 would probably be a better choice than degree 5.

---

# 26. Why does training R² keep increasing?

This is important.

As you increase polynomial degree, you give the model more flexibility.

Therefore, training performance generally doesn't get worse.

For example:

```text
Degree 1 → R² = 0.70
Degree 2 → R² = 0.85
Degree 3 → R² = 0.94
Degree 4 → R² = 0.98
Degree 5 → R² = 0.999
```

But test performance can behave differently:

```text
Degree 1 → 0.68
Degree 2 → 0.90
Degree 3 → 0.91
Degree 4 → 0.82
Degree 5 → 0.65
```

This is the classic **bias-variance tradeoff**.

---

# 27. Polynomial Regression and Regularization

Polynomial Regression can easily overfit.

One solution is regularization.

Instead of:

```python
LinearRegression()
```

you can use:

```python
Ridge()
```

or:

```python
Lasso()
```

For example:

```python
from sklearn.linear_model import Ridge

model = Pipeline([
    ("poly", PolynomialFeatures(degree=5)),
    ("ridge", Ridge(alpha=1.0))
])
```

Now the model has high polynomial flexibility but regularization discourages excessively large coefficients.

---

# 28. Polynomial Regression + Ridge

The polynomial model might be:

$$
y=b_0+b_1x+b_2x^2+\cdots+b_5x^5
$$

Ordinary Linear Regression minimizes:

$$
MSE
$$

Ridge Regression minimizes:

$$
MSE+\lambda\sum_{j=1}^{n}b_j^2
$$

The additional term penalizes large coefficients.

This can make high-degree polynomial models much more stable.

---

# 29. Scaling becomes important

Polynomial features can have very different magnitudes.

For example:

$$
x=100
$$

Then:

$$
x^2=10,000
$$

and:

$$
x^5=10^{10}
$$

This can cause numerical problems and make optimization harder.

Therefore, when using polynomial features—especially with regularized models—feature scaling is often useful.

A common pipeline is:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge

model = Pipeline([
    ("poly", PolynomialFeatures(degree=5)),
    ("scaler", StandardScaler()),
    ("ridge", Ridge(alpha=1.0))
])
```

---

# 30. Polynomial Regression vs Linear Regression

| Feature | Linear Regression | Polynomial Regression |
|---|---|---|
| Relationship | Linear | Non-linear/curved |
| Equation | $b_0+b_1x$ | $b_0+b_1x+b_2x^2+\cdots$ |
| Shape | Straight line | Curve |
| Features | Original | Polynomially transformed |
| Complexity | Low | Can be high |
| Overfitting risk | Lower | Higher |
| Main hyperparameter | Usually few | Degree |
| Regularization | Optional | Often useful |

---

# 31. A common misconception

Polynomial Regression is **not** a completely different regression algorithm.

Think of it as:

```text
Feature Engineering
        +
Linear Regression
```

More precisely:

$$
\boxed{
\text{Polynomial Regression}
=
\text{Polynomial Feature Transformation}
+
\text{Linear Regression}
}
$$

This is why the scikit-learn implementation is so simple.

---

# 32. Polynomial Regression vs other nonlinear models

Polynomial Regression is useful when the relationship is reasonably smooth and can be approximated by a polynomial.

But it isn't always the best choice.

For complicated relationships, you might use:

- Decision Trees
- Random Forest
- Gradient Boosting
- XGBoost
- Neural Networks
- SVM with nonlinear kernels

Polynomial Regression is particularly useful when you want a model that is:

- relatively simple
- interpretable
- mathematically understandable
- capable of capturing moderate curvature

---

# 33. Important practical workflow

When solving a Polynomial Regression problem, follow this pattern:

```text
1. Load data
      ↓
2. Explore relationship
      ↓
3. Train Linear Regression baseline
      ↓
4. Create Polynomial Features
      ↓
5. Train Polynomial Regression
      ↓
6. Evaluate on validation/test data
      ↓
7. Try different degrees
      ↓
8. Detect underfitting/overfitting
      ↓
9. Add regularization if necessary
      ↓
10. Select final model
```

---

# 34. The most important code to remember

If you're learning this for ML, remember these three versions.

### Basic

```python
poly = PolynomialFeatures(degree=2)

X_poly = poly.fit_transform(X)

model = LinearRegression()

model.fit(X_poly, y)
```

### Pipeline

```python
model = Pipeline([
    ("poly", PolynomialFeatures(degree=2)),
    ("linear", LinearRegression())
])

model.fit(X, y)
```

### Polynomial + Ridge

```python
model = Pipeline([
    ("poly", PolynomialFeatures(degree=5)),
    ("scaler", StandardScaler()),
    ("ridge", Ridge(alpha=1.0))
])

model.fit(X, y)
```

---

# 35. Final mental model

Think about Polynomial Regression like this:

```text
                 ORIGINAL DATA
                      │
                      ▼
                X = [x]
                      │
                      ▼
          Polynomial Transformation
                      │
          ┌───────────┼───────────┐
          ▼           ▼           ▼
          x          x²          x³ ...
          │           │           │
          └───────────┼───────────┘
                      ▼
              Linear Regression
                      │
                      ▼
                y = prediction
```

So if you understand **Linear Regression**, the next thing to understand is:

> **Polynomial Regression doesn't change the fundamental Linear Regression algorithm. It changes the feature representation before Linear Regression sees the data.**

And the three concepts you should focus on most are:

1. **Polynomial feature transformation**
   $$
   x\rightarrow x,x^2,x^3,\ldots
   $$

2. **Degree selection**
   - Too low → underfitting
   - Appropriate → good generalization
   - Too high → overfitting

3. **Regularization**
   - Especially useful when using high-degree polynomial features.