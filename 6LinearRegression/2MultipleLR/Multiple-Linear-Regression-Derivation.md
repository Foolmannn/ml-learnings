

The mathematical derivation of **Multiple Linear Regression (MLR)** is one of the most important topics in machine learning because it explains **how the optimal coefficients are obtained**. The derivation is based on the **Least Squares Method** and leads to the famous **Normal Equation**.

---

# 1. Problem Statement

Suppose we have:

- **m** training examples
- **n** independent variables (features)

Instead of a single feature (Simple Linear Regression), we now have multiple features.

For one training example:

$$
(x_1,x_2,\dots,x_n,y)
$$

The model predicts

$$
\hat y=\beta_0+\beta_1x_1+\beta_2x_2+\cdots+\beta_nx_n
$$

where

- $\beta_0$ = intercept (bias)
- $\beta_1,\beta_2,\dots,\beta_n$ = feature coefficients

---

# Example

Suppose we predict house price.

Features:

- Size
- Bedrooms
- Age

Then

$$
\hat y=
\beta_0+
\beta_1(Size)+
\beta_2(Bedrooms)+
\beta_3(Age)
$$

---

# 2. Writing All Training Samples

Suppose we have m observations.

The model becomes

For observation 1

$$
y_1=\beta_0+\beta_1x_{11}+\beta_2x_{12}+\cdots+\beta_nx_{1n}+\epsilon_1
$$

Observation 2

$$
y_2=\beta_0+\beta_1x_{21}+\beta_2x_{22}+\cdots+\beta_nx_{2n}+\epsilon_2
$$

...

Observation m

$$
y_m=\beta_0+\beta_1x_{m1}+\beta_2x_{m2}+\cdots+\beta_nx_{mn}+\epsilon_m
$$

where

$$
\epsilon_i
$$

is the error.

---

# 3. Matrix Representation

Instead of writing m equations separately, we write everything in matrix form.

Define

Target vector

$$
Y=
\begin{bmatrix}
y_1\\
y_2\\
\vdots\\
y_m
\end{bmatrix}
$$

Coefficient vector

$$
\beta=
\begin{bmatrix}
\beta_0\\
\beta_1\\
\vdots\\
\beta_n
\end{bmatrix}
$$

Feature matrix

The first column contains all ones for the intercept.

$$
X=
\begin{bmatrix}
1&x_{11}&x_{12}&\cdots&x_{1n}\\
1&x_{21}&x_{22}&\cdots&x_{2n}\\
\vdots&\vdots&\vdots&&\vdots\\
1&x_{m1}&x_{m2}&\cdots&x_{mn}
\end{bmatrix}
$$

Prediction

$$
\hat Y=X\beta
$$

Thus the model becomes

$$
Y=X\beta+\epsilon
$$

---

# 4. Objective

We want to choose β such that the prediction error is minimum.

Residual

$$
e=Y-\hat Y
$$

Since

$$
\hat Y=X\beta
$$

Residual becomes

$$
e=Y-X\beta
$$

---

# 5. Cost Function

We minimize the Sum of Squared Errors.

$$
J(\beta)
=\sum_{i=1}^{m}(y_i-\hat y_i)^2
$$

Matrix form

$$
J(\beta)
=(Y-X\beta)^T(Y-X\beta)
$$

This is the Least Squares objective.

---

# Why Square the Error?

Because

- Positive and negative errors don't cancel.
- Larger errors are penalized more.
- The function becomes differentiable.
- The optimization has a unique solution (when $X^TX$ is invertible).

---

# 6. Expand the Cost Function

Expand

$$
(Y-X\beta)^T(Y-X\beta)
$$

Using matrix multiplication

$$
=
Y^TY
-
Y^TX\beta
-
(X\beta)^TY
+
(X\beta)^TX\beta
$$

Since

$$
(X\beta)^T=\beta^TX^T
$$

we get

$$
=
Y^TY
-
Y^TX\beta
-
\beta^TX^TY
+
\beta^TX^TX\beta
$$

Notice

$$
Y^TX\beta
$$

is a scalar.

A scalar equals its transpose.

Therefore

$$
Y^TX\beta
=
\beta^TX^TY
$$

Hence

$$
J(\beta)
=
Y^TY
-
2\beta^TX^TY
+
\beta^TX^TX\beta
$$

---

# 7. Differentiate

Take derivative with respect to β.

Useful matrix derivatives:

$$
\frac{\partial}{\partial\beta}
(c)=0
$$

$$
\frac{\partial}{\partial\beta}
(a^T\beta)
=a
$$

$$
\frac{\partial}{\partial\beta}
(\beta^TA\beta)
=
(A+A^T)\beta
$$

If A is symmetric

$$
=
2A\beta
$$

Since

$$
X^TX
$$

is symmetric,

$$
(X^TX)^T=X^TX
$$

---

Differentiate

First term

$$
\frac{\partial}{\partial\beta}
(Y^TY)
=0
$$

Second term

$$
\frac{\partial}{\partial\beta}
(-2\beta^TX^TY)
=
-2X^TY
$$

Third term

$$
\frac{\partial}{\partial\beta}
(\beta^TX^TX\beta)
=
2X^TX\beta
$$

Therefore

$$
\frac{\partial J}{\partial\beta}
=
-2X^TY
+
2X^TX\beta
$$

---

# 8. Set Gradient to Zero

At minimum,

$$
\frac{\partial J}{\partial\beta}=0
$$

So

$$
-2X^TY
+
2X^TX\beta
=
0
$$

Divide by 2

$$
X^TX\beta=X^TY
$$

These are called the **Normal Equations**.

---

# 9. Solve for β

Multiply both sides by

$$
(X^TX)^{-1}
$$

assuming it exists:

$$
\beta
=
(X^TX)^{-1}X^TY
$$

This is the closed-form solution for Multiple Linear Regression.



---

# 10. Final Formula

$$
\boxed{\beta=(X^TX)^{-1}X^TY}
$$

This computes the coefficients that minimize the sum of squared errors.

---

# Why Does This Work?

The cost function

$$
J(\beta)=(Y-X\beta)^T(Y-X\beta)
$$

is a convex quadratic function. Its graph is a multidimensional bowl (a paraboloid). A convex function has a single global minimum, so solving

$$
\nabla J(\beta)=0
$$

finds the unique optimal coefficient vector.

---

# Geometric Interpretation

Imagine each column of $X$ as a vector in an $m$-dimensional space. The prediction

$$
\hat{Y}=X\beta
$$

must lie in the **column space** of $X$. The least-squares solution chooses $\hat{Y}$ to be the **orthogonal projection** of the observed vector $Y$ onto that column space. This implies the residual vector

$$
e=Y-\hat{Y}
$$

is perpendicular to every column of $X$, giving

$$
X^T(Y-X\beta)=0,
$$

which rearranges directly to the normal equations:

$$
X^TX\beta=X^TY.
$$

---

# Assumptions for the Normal Equation

The inverse $(X^TX)^{-1}$ exists only if the columns of $X$ are linearly independent. Problems arise when:

- **Perfect multicollinearity:** One feature is an exact linear combination of others.
- **More features than samples:** $n > m$, making $X^TX$ singular.
- **Redundant or duplicate features.**

In practice, these cases are handled by:
- Using the **Moore–Penrose pseudoinverse** $X^+$, so $\beta = X^+Y$.
- Adding **Ridge Regression**, which solves

$$
\beta=(X^TX+\lambda I)^{-1}X^TY,
$$

where $\lambda$ stabilizes the inversion and reduces overfitting.

---

# Complete Derivation Flow

```
Linear Model
      │
      ▼
ŷ = Xβ
      │
      ▼
Residual = Y − Xβ
      │
      ▼
Cost Function
J(β) = (Y − Xβ)ᵀ(Y − Xβ)
      │
      ▼
Expand
      │
      ▼
Differentiate
      │
      ▼
∇J = −2XᵀY + 2XᵀXβ
      │
      ▼
Set Gradient = 0
      │
      ▼
XᵀXβ = XᵀY
      │
      ▼
β = (XᵀX)⁻¹XᵀY
```

This derivation forms the mathematical foundation of **Ordinary Least Squares (OLS)**, and many advanced regression methods—such as **Ridge Regression**, **Lasso**, **Elastic Net**, and even parts of **Generalized Linear Models (GLMs)**—are extensions or modifications of this framework.

---
