
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
