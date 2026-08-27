
# Lasso Regression and Regularization — In Detail

Lasso Regression is one of the most important **regularization techniques** used in machine learning, especially when you have many features and want the model to automatically remove some of them.

Since you’ve already been studying **Linear Regression and regularization**, think of Lasso as:

> **Linear Regression + L1 penalty**

---

# 1. Why do we need Regularization?

Suppose we have a dataset:

| Feature | Meaning |
|---|---|
| $x_1$ | house area |
| $x_2$ | number of rooms |
| $x_3$ | house age |
| $x_4$ | distance from city |
| $x_5$ | number of windows |
| ... | ... |
| $x_{100}$ | some other feature |

We want to predict house price.

Ordinary Linear Regression tries to minimize:

$$
MSE = \frac{1}{n}\sum_{i=1}^{n}(y_i-\hat y_i)^2
$$

The problem is that with many features, the model can start fitting **noise** rather than actual patterns.

This can lead to:

- large coefficients
- overfitting
- poor generalization
- unstable models
- unnecessary features

Regularization solves this by adding a **penalty for large coefficients**.

---

# 2. General idea of Regularization

Ordinary Linear Regression:

$$
Loss = MSE
$$

Regularized Linear Regression:

$$
Loss = MSE + Penalty
$$

The penalty discourages the model from making coefficients unnecessarily large.

There are two major forms:

### Ridge Regression

Uses **L2 regularization**:

$$
Loss = MSE + \lambda\sum_{j=1}^{p}\beta_j^2
$$

### Lasso Regression

Uses **L1 regularization**:

$$
Loss = MSE + \lambda\sum_{j=1}^{p}|\beta_j|
$$

The important difference is:

> **Ridge shrinks coefficients toward zero. Lasso can make coefficients exactly zero.**

That distinction is extremely important.

---

# 3. What is Lasso Regression?

**LASSO** stands for:

> **Least Absolute Shrinkage and Selection Operator**

Lasso is a linear regression technique that adds an **L1 penalty** to the ordinary least-squares objective.

The objective can be written as:

$$
\boxed{
J(\beta)=
\frac{1}{2n}\sum_{i=1}^{n}
(y_i-\hat y_i)^2
+
\lambda\sum_{j=1}^{p}|\beta_j|
}
$$

where:

- $y_i$ = actual value
- $\hat y_i$ = predicted value
- $\beta_j$ = coefficient
- $p$ = number of features
- $\lambda$ = regularization strength

The first term wants to make predictions accurate.

The second term wants to make coefficients small.

So Lasso has to find a compromise.

---

# 4. Understanding the L1 Penalty

Suppose your model is:

$$
\hat y = 5 + 10x_1 + 8x_2 + 2x_3
$$

The coefficient penalty is:

$$
|10|+|8|+|2|=20
$$

If:

$$
\lambda=0.1
$$

the regularization contribution is:

$$
0.1(20)=2
$$

So the optimization tries to balance:

$$
\text{prediction error} + 2
$$

The model may discover that reducing some coefficients slightly increases prediction error but significantly reduces the penalty.

This produces **simpler models**.

---

# 5. What does $\lambda$ do?

This is one of the most important concepts.

$$
\boxed{\lambda = \text{regularization strength}}
$$

### Small $\lambda$

The penalty is weak.

The model behaves more like ordinary Linear Regression.

$$
\lambda \rightarrow 0
$$

approximately gives:

$$
\text{Lasso} \rightarrow \text{Linear Regression}
$$

### Large $\lambda$

The penalty becomes strong.

The model strongly tries to reduce coefficients.

Some coefficients can become:

$$
\beta_j=0
$$

Therefore, those features effectively disappear from the model.

---

# 6. Why can Lasso produce exactly zero coefficients?

This is the special property of Lasso.

Consider:

$$
\lambda |\beta|
$$

The absolute value function has a sharp corner at:

$$
\beta=0
$$

Because of this shape, optimization can settle exactly at zero.

For example:

Before Lasso:

$$
\beta =
[5.2,\;3.8,\;0.9,\;0.2,\;-4.1]
$$

After Lasso:

$$
\beta =
[4.7,\;3.1,\;0,\;0,\;-3.6]
$$

Features 3 and 4 have coefficients of zero.

They have effectively been **removed**.

This is called:

> **Feature Selection**

---

# 7. Lasso performs feature selection

Suppose you have 20 features.

Ordinary Linear Regression:

```text
X1  →  coefficient 4.2
X2  →  coefficient 1.8
X3  →  coefficient -0.7
X4  →  coefficient 0.2
X5  →  coefficient 3.1
...
X20 → coefficient 0.05
```

Lasso might produce:

```text
X1  →  coefficient 3.9
X2  →  coefficient 1.4
X3  →  coefficient 0
X4  →  coefficient 0
X5  →  coefficient 2.8
...
X20 → coefficient 0
```

So instead of using all 20 features, the model may effectively use only 8.

This makes Lasso particularly useful for:

- high-dimensional datasets
- feature selection
- sparse models
- reducing model complexity

---

# 8. Lasso vs Ordinary Linear Regression

### Linear Regression

Objective:

$$
MSE
$$

It only cares about prediction error.

### Lasso

Objective:

$$
MSE+\lambda\sum|\beta_j|
$$

It cares about:

1. prediction error
2. coefficient size

Therefore:

```text
Linear Regression
       ↓
Minimize prediction error
       ↓
Can produce large coefficients
```

while:

```text
Lasso
       ↓
Minimize prediction error
       +
Minimize coefficient magnitude
       ↓
Smaller coefficients
       ↓
Some coefficients become exactly 0
```

---

# 9. Lasso vs Ridge

This is probably the most important comparison in regularization.

| Property | Linear Regression | Ridge | Lasso |
|---|---|---|---|
| Regularization | No | L2 | L1 |
| Penalty | None | $\sum\beta^2$ | $\sum|\beta|$ |
| Shrinks coefficients | ❌ | ✅ | ✅ |
| Can produce exactly zero | ❌ | Usually no | ✅ |
| Feature selection | ❌ | ❌ | ✅ |
| Useful with many features | Sometimes | ✅ | ✅ |
| Handles multicollinearity | Poorly | Very well | Can |
| Produces sparse model | ❌ | Usually no | ✅ |

---

# 10. Geometric intuition

There is a very useful geometric explanation.

For Ridge, the constraint looks like a circle in two dimensions:

$$
\beta_1^2+\beta_2^2\leq t
$$

For Lasso, the constraint looks like a diamond:

$$
|\beta_1|+|\beta_2|\leq t
$$

The corners of the Lasso diamond lie directly on the axes.

That means the optimization has a higher chance of reaching:

$$
\beta_1=0
$$

or

$$
\beta_2=0
$$

Hence Lasso naturally produces zero coefficients.

Conceptually:

```text
Ridge constraint          Lasso constraint

       ○                       /\
     /   \                    /  \
    |     |                  /    \
     \   /                  \      /
       ○                     \    /
                              \  /
                               \/
```

The sharp corners of the Lasso constraint are the reason for sparsity.

---
