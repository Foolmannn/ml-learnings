> From: https://chatgpt.com/c/6a75dbc6-89c4-83ee-9cef-e468d6e8a7fe

# you asked

message time: 2026-08-07 19:06:14

Simple linear regression mathematical derivation

---

# chatgpt response

The mathematical derivation of **Simple Linear Regression (SLR)** is based on the **Ordinary Least Squares (OLS)** method. The objective is to find the values of the slope and intercept that minimize the sum of squared prediction errors.

The fitted regression equation is:



---

# 1. Problem Setup

Suppose we have a dataset of **n observations**:

$$
(x_1,y_1), (x_2,y_2), \ldots, (x_n,y_n)
$$

where

- $x$ = independent variable (feature)
- $y$ = dependent variable (target)

We want to fit a straight line:

$$
\boxed{\hat y = \beta_0+\beta_1x}
$$

where

- $\beta_0$ = intercept
- $\beta_1$ = slope

Our goal is to estimate

$$
\hat\beta_0,\quad \hat\beta_1
$$

from the given data.

---

# 2. Actual Model

The true relationship is

$$
y_i=\beta_0+\beta_1x_i+\varepsilon_i
$$

where

- $\varepsilon_i$ = random error
- Mean of error = 0

Since we don't know the true parameters,

our prediction is

$$
\hat y_i=b_0+b_1x_i
$$

where

- $b_0$ = estimated intercept
- $b_1$ = estimated slope

---

# 3. Residual

Residual means

$$
\text{Actual}-\text{Predicted}
$$

Therefore,

$$
e_i=y_i-\hat y_i
$$

Substituting

$$
\hat y_i=b_0+b_1x_i
$$

gives

$$
e_i=y_i-(b_0+b_1x_i)
$$

---

# 4. Why Not Minimize Residuals Directly?

Suppose

Residuals

$$
3,\,-3,\,5,\,-5
$$

Their sum

$$
3-3+5-5=0
$$

Even though predictions are poor.

So minimizing

$$
\sum e_i
$$

doesn't work.

---

# 5. Square the Residuals

Instead,

$$
e_i^2
$$

Now all values become positive.

The objective becomes

$$
\boxed{
S=\sum_{i=1}^{n}(y_i-\hat y_i)^2
}
$$

Substitute

$$
\hat y_i=b_0+b_1x_i
$$

$$
\boxed{
S=\sum_{i=1}^{n}(y_i-b_0-b_1x_i)^2
}
$$

This is called the **Residual Sum of Squares (RSS)** or **Sum of Squared Errors (SSE)**.

---

# 6. Objective Function

We need to find

$$
b_0,b_1
$$

such that

$$
S
$$

is minimum.

So we solve

$$
\frac{\partial S}{\partial b_0}=0
$$

and

$$
\frac{\partial S}{\partial b_1}=0
$$

---

# 7. Differentiate with Respect to $b_0$

Start with

$$
S=\sum(y_i-b_0-b_1x_i)^2
$$

Differentiate

$$
\frac{\partial S}{\partial b_0}
=
\sum
2(y_i-b_0-b_1x_i)(-1)
$$

$$
=-2\sum(y_i-b_0-b_1x_i)
$$

Set equal to zero

$$
-2\sum(y_i-b_0-b_1x_i)=0
$$

Divide by −2

$$
\boxed{
\sum(y_i-b_0-b_1x_i)=0
}
$$

Expand

$$
\sum y_i
-
\sum b_0
-
b_1\sum x_i
=0
$$

Since

$$
\sum b_0=nb_0
$$

we obtain

$$
\boxed{
\sum y_i
=
nb_0+b_1\sum x_i
}
$$

This is the **First Normal Equation**.

---

# 8. Differentiate with Respect to $b_1$

Again

$$
S=\sum(y_i-b_0-b_1x_i)^2
$$

Differentiate

$$
\frac{\partial S}{\partial b_1}
=
-2\sum
x_i(y_i-b_0-b_1x_i)
$$

Set equal to zero

$$
\boxed{
\sum x_i(y_i-b_0-b_1x_i)=0
}
$$

Expand

$$
\sum x_iy_i
-
b_0\sum x_i
-
b_1\sum x_i^2
=0
$$

Therefore

$$
\boxed{
\sum x_iy_i
=
b_0\sum x_i
+
b_1\sum x_i^2
}
$$

This is the **Second Normal Equation**.

---

# 9. Solve the Two Equations

The two equations are

$$
\sum y_i
=
nb_0+b_1\sum x_i
$$

$$
\sum x_iy_i
=
b_0\sum x_i
+
b_1\sum x_i^2
$$

From the first equation,

$$
b_0
=
\frac{\sum y_i-b_1\sum x_i}{n}
$$

Notice that

$$
\bar x=\frac{\sum x_i}{n},
\qquad
\bar y=\frac{\sum y_i}{n}
$$

Substitute these into the expression for $b_0$:

$$
\boxed{
b_0=\bar y-b_1\bar x
}
$$

---

# 10. Derivation of the Slope

Substitute

$$
b_0=\bar y-b_1\bar x
$$

into the second normal equation:

$$
\sum x_iy_i
=
(\bar y-b_1\bar x)\sum x_i
+
b_1\sum x_i^2
$$

Since

$$
\sum x_i=n\bar x
$$

we get

$$
\sum x_iy_i
=
n\bar x\bar y
-
nb_1\bar x^2
+
b_1\sum x_i^2
$$

Rearrange:

$$
\sum x_iy_i
-
n\bar x\bar y
=
b_1
\left(
\sum x_i^2
-
n\bar x^2
\right)
$$

Recognize that

$$
\sum (x_i-\bar x)(y_i-\bar y)
=
\sum x_iy_i
-
n\bar x\bar y
$$

and

$$
\sum(x_i-\bar x)^2
=
\sum x_i^2
-
n\bar x^2
$$

Hence,

$$
\boxed{
b_1
=
\frac{\sum(x_i-\bar x)(y_i-\bar y)}
{\sum(x_i-\bar x)^2}
}
$$

This is the **least-squares estimate of the slope**.

---

# 11. Intercept Formula

After computing $b_1$,

$$
\boxed{
b_0=\bar y-b_1\bar x
}
$$

---

# 12. Final Regression Equation

The fitted regression line is

$$
\boxed{
\hat y=b_0+b_1x
}
$$

where

$$
\boxed{
b_1=
\frac{\sum(x_i-\bar x)(y_i-\bar y)}
{\sum(x_i-\bar x)^2}
}
$$

and

$$
\boxed{
b_0=\bar y-b_1\bar x
}
$$

---

# 13. Matrix Interpretation (Preview)

Although simple linear regression is often derived using calculus, it can also be expressed in matrix form:

$$
\mathbf{y}=X\boldsymbol{\beta}+\boldsymbol{\varepsilon}
$$

where

$$
X=
\begin{bmatrix}
1 & x_1\\
1 & x_2\\
\vdots & \vdots\\
1 & x_n
\end{bmatrix},
\qquad
\boldsymbol{\beta}=
\begin{bmatrix}
\beta_0\\
\beta_1
\end{bmatrix}
$$

The least-squares solution is

$$
\boxed{
\hat{\boldsymbol{\beta}}
=
(X^TX)^{-1}X^Ty
}
$$

For simple linear regression, expanding this matrix equation leads exactly to the same formulas for $b_0$ and $b_1$.

---

## Summary

| Quantity | Formula |
|---------|---------|
| Regression line | $\hat y=b_0+b_1x$ |
| Residual | $e_i=y_i-\hat y_i$ |
| Cost function (RSS) | $\sum(y_i-\hat y_i)^2$ |
| Slope | $\displaystyle b_1=\frac{\sum(x_i-\bar x)(y_i-\bar y)}{\sum(x_i-\bar x)^2}$ |
| Intercept | $\displaystyle b_0=\bar y-b_1\bar x$ |

This derivation is the foundation for multiple linear regression as well. In multiple linear regression, the same least-squares principle is used, but the partial derivatives are taken with respect to multiple coefficients, leading naturally to the matrix solution $(X^\top X)^{-1}X^\top y$.

---
Powered by [AI Exporter](https://saveai.net)