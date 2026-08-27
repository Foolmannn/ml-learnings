
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

# 11. Another way to understand Lasso

Imagine you have two features:

$$
x_1 = \text{age}
$$

$$
x_2 = \text{income}
$$

and your model is:

$$
\hat y = \beta_0+\beta_1x_1+\beta_2x_2
$$

Suppose ordinary Linear Regression gives:

$$
\beta_1=8
$$

$$
\beta_2=3
$$

Lasso might decide:

$$
\beta_1=6.5
$$

$$
\beta_2=0
$$

Why?

Because perhaps $x_2$ isn't contributing enough to justify the penalty.

Therefore:

```text
Feature 1 → important → keep
Feature 2 → not useful enough → remove
```

---

# 12. Important: Lasso does NOT simply remove the least correlated features

A common misconception is:

> "Lasso calculates correlation and removes features."

That's not how it works.

Lasso considers the features **jointly while optimizing the loss function**.

Therefore, a feature with relatively low individual correlation with $y$ could still be useful when combined with other features.

Feature selection happens as a consequence of the optimization process.

---

# 13. Lasso and multicollinearity

Suppose:

$$
X_1 = \text{house area}
$$

and

$$
X_2 = \text{number of rooms}
$$

These might be highly correlated.

Lasso may choose one feature and push the other toward zero.

For example:

```text
Before:

Area       → 5.2
Rooms      → 4.7
Age        → -1.3


After Lasso:

Area       → 7.1
Rooms      → 0
Age        → -1.0
```

However, this behavior can be unstable when predictors are highly correlated.

If you have groups of strongly correlated features, **Elastic Net** can often be preferable because it combines L1 and L2 penalties.

---

# 14. Elastic Net

Elastic Net combines Ridge and Lasso.

Its objective is approximately:

$$
Loss =
MSE
+
\lambda
\left[
\alpha\sum|\beta_j|
+
(1-\alpha)\sum\beta_j^2
\right]
$$

So:

```text
Elastic Net
    │
    ├── L1 component → feature selection
    │
    └── L2 component → coefficient stability
```

Special cases:

$$
\alpha=1
$$

gives approximately Lasso.

$$
\alpha=0
$$

gives approximately Ridge.

---

# 15. Why Feature Scaling is important for Lasso

This is extremely important when implementing Lasso.

Suppose:

```text
Age:       20 - 80
Salary:    20,000 - 200,000
Experience: 0 - 30
```

Lasso penalizes coefficients based on their magnitude.

Without scaling, features measured on different scales can receive unfair treatment.

Therefore, you normally standardize features:

$$
X_{scaled}=
\frac{X-\mu}{\sigma}
$$

using something such as `StandardScaler`.

Typical pipeline:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso

model = Pipeline([
    ("scaler", StandardScaler()),
    ("lasso", Lasso(alpha=0.1))
])
```

Then:

```python
model.fit(X_train, y_train)

predictions = model.predict(X_test)
```

---

# 16. `alpha` in scikit-learn vs $\lambda$

This often causes confusion.

Mathematically, we usually write:

$$
\lambda
$$

for regularization strength.

In scikit-learn's `Lasso`, the parameter is called:

```python
alpha
```

So:

```python
Lasso(alpha=0.1)
```

means regularization strength is 0.1 under scikit-learn's objective formulation.

Don't confuse this with the mixing parameter often called `alpha` in Elastic Net literature.

---

# 17. What happens as alpha increases?

Consider:

```python
Lasso(alpha=0.001)
Lasso(alpha=0.01)
Lasso(alpha=0.1)
Lasso(alpha=1)
Lasso(alpha=10)
```

Generally:

```text
alpha small
     ↓
weak regularization
     ↓
more non-zero coefficients
     ↓
more complex model
```

and:

```text
alpha large
     ↓
strong regularization
     ↓
more coefficients → 0
     ↓
simpler model
```

But there's a trade-off.

Too much regularization can cause **underfitting**.

---

# 18. Bias-Variance perspective

Regularization changes the bias-variance trade-off.

### No regularization

Usually:

- lower bias
- higher variance

Potentially:

$$
\text{Overfitting}
$$

### Moderate regularization

Usually:

- slightly higher bias
- lower variance

Often:

$$
\text{Better generalization}
$$

### Excessive regularization

- very high bias
- very low variance

Potentially:

$$
\text{Underfitting}
$$

Conceptually:

```text
Regularization
     ↑
     │
     │       Variance ↓
     │
     │       Bias ↑
     │
     └────────────────→
```

The goal isn't:

> "Make coefficients as small as possible."

The goal is:

> **Find the regularization strength that gives the best generalization performance.**

---

# 19. How do we choose alpha?

We generally don't manually guess it.

Use **cross-validation**.

For example:

```python
from sklearn.linear_model import LassoCV

model = LassoCV(cv=5)

model.fit(X_train, y_train)

print(model.alpha_)
```

`LassoCV` tries different alpha values and selects one based on cross-validation performance.

For example, it might determine:

```text
alpha = 0.037
```

is better than:

```text
0.001
0.01
0.1
1
```

---

# 20. LassoCV

A common workflow:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV

model = Pipeline([
    ("scaler", StandardScaler()),
    ("lasso", LassoCV(cv=5))
])

model.fit(X_train, y_train)
```

Then:

```python
pred = model.predict(X_test)
```

You can inspect the selected alpha:

```python
print(model["lasso"].alpha_)
```

---

# 21. Exam-level mathematical understanding

Suppose:

$$
\hat y = \beta_0+\beta_1x_1+\beta_2x_2
$$

Lasso minimizes:

$$
J(\beta_0,\beta_1,\beta_2)
=
\sum_{i=1}^{n}
(y_i-\hat y_i)^2
+
\lambda
(|\beta_1|+|\beta_2|)
$$

Notice something important:

### The intercept is generally NOT penalized.

We penalize:

$$
\beta_1,\beta_2,\ldots,\beta_p
$$

but generally not:

$$
\beta_0
$$

because the intercept isn't a feature coefficient.

---

# 22. Why is Lasso harder mathematically than Linear Regression?

Ordinary Linear Regression has a differentiable squared-error objective.

Lasso contains:

$$
|\beta|
$$

The absolute-value function is not differentiable at:

$$
\beta=0
$$

because:

$$
\frac{d}{d\beta}|\beta|
=
\begin{cases}
1 & \beta>0\\
-1 & \beta<0
\end{cases}
$$

and it has no ordinary derivative exactly at zero.

This is one reason Lasso is commonly solved using specialized optimization techniques such as:

- coordinate descent
- proximal methods
- LARS-related algorithms

You don't need to implement these from scratch to use Lasso effectively, but understanding why the optimization is different is useful.

---

# 23. Soft Thresholding

A very important mathematical idea behind Lasso is **soft thresholding**.

Imagine an estimated coefficient:

$$
z=5
$$

and the regularization threshold is:

$$
\lambda=2
$$

The coefficient may become:

$$
5-2=3
$$

If:

$$
z=-5
$$

it becomes:

$$
-5+2=-3
$$

If:

$$
z=1
$$

and:

$$
\lambda=2
$$

then:

$$
z\rightarrow0
$$

So conceptually:

$$
S(z,\lambda)
=
sign(z)\max(|z|-\lambda,0)
$$

This is called the **soft-thresholding operator**.

It explains beautifully why Lasso can make coefficients exactly zero.

---

# 24. Lasso example

Suppose we have:

```python
import numpy as np

X = np.array([
    [1, 2, 10],
    [2, 3, 20],
    [3, 4, 30],
    [4, 5, 40],
    [5, 6, 50]
])

y = np.array([10, 20, 30, 40, 50])
```

Fit Lasso:

```python
from sklearn.linear_model import Lasso

model = Lasso(alpha=0.1)

model.fit(X, y)
```

Inspect coefficients:

```python
print(model.coef_)
```

You might get something conceptually like:

```text
[... ... 0]
```

The exact coefficients depend on the data and alpha.

A coefficient of zero means:

```text
That feature is not being used by the fitted Lasso model.
```

---
