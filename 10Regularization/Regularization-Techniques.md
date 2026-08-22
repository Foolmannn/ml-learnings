

# Regularization Techniques in Machine Learning

Regularization is one of the most important techniques for controlling **overfitting** in machine learning. Since you're working through regression and gradient descent, it's especially useful to understand regularization mathematically rather than treating it as just an sklearn parameter.

---

# 1. What is Regularization?

**Regularization is a technique used to reduce overfitting by adding a penalty for model complexity to the loss function.**

Normally, a model tries to minimize:

$$
Loss = Error
$$

With regularization:

$$
Loss = Error + Regularization\ Penalty
$$

So the model is encouraged to:

1. Fit the training data well.
2. Keep its parameters/weights relatively small.
3. Avoid unnecessarily complex relationships.

### The basic idea

Suppose a linear regression model is:

$$
\hat y = w_0 + w_1x_1+w_2x_2+\cdots+w_nx_n
$$

A model with very large weights might be fitting noise in the training data.

Regularization says:

> "You can minimize the prediction error, but I'll also penalize you if your weights become unnecessarily large."

---

# 2. Why Do We Need Regularization?

Consider a regression problem.

### Model A

$$
y = 2 + 3x
$$

Simple model.

### Model B

$$
y = 2 + 3x + 100x^2 - 50x^3 + 200x^4
$$

The second model is much more complex.

It might fit the training data extremely well:

$$
Training\ R^2 = 0.99
$$

but perform poorly on unseen data:

$$
Testing\ R^2 = 0.55
$$

This is **overfitting**.

Regularization discourages Model B from developing excessively large coefficients.

---

# 3. Regularization and Bias-Variance Tradeoff

Regularization is closely related to the **bias-variance tradeoff**.

Without regularization:

- Low training error
- Potentially high variance
- Higher chance of overfitting

With stronger regularization:

- Training error usually increases
- Variance decreases
- Bias increases
- Generalization can improve

Conceptually:

$$
\boxed{\text{Regularization} \rightarrow \text{lower variance} \rightarrow \text{less overfitting}}
$$

But too much regularization can cause:

$$
\boxed{\text{high bias} \rightarrow \text{underfitting}}
$$

So the goal isn't:

> "Use maximum regularization."

The goal is:

> **Find an appropriate regularization strength.**

---

# 4. Regularization in Linear Regression

Ordinary Linear Regression minimizes the Mean Squared Error:

$$
MSE = \frac{1}{m}\sum_{i=1}^{m}(y_i-\hat y_i)^2
$$

For:

$$
\hat y = w_0+w_1x_1+\cdots+w_nx_n
$$

regularization modifies the objective function.

There are three particularly important approaches:

1. **Ridge Regression — L2 regularization**
2. **Lasso Regression — L1 regularization**
3. **Elastic Net — combination of L1 and L2**

---

# 5. Ridge Regression — L2 Regularization

Ridge regression adds the squared weights to the loss function.

Instead of:

$$
MSE
$$

we minimize:

$$
\boxed{
MSE+\lambda\sum_{j=1}^{n}w_j^2
}
$$

where:

- $w_j$ = model coefficients
- $\lambda$ = regularization strength

The intercept $w_0$ is generally **not regularized**.

---

## 5.1 Why Squared Weights?

Suppose:

$$
w_1=10
$$

Then:

$$
w_1^2=100
$$

Suppose:

$$
w_1=2
$$

Then:

$$
w_1^2=4
$$

Therefore, large coefficients receive a much larger penalty.

This encourages:

$$
w_j \rightarrow 0
$$

but usually doesn't make them exactly zero.

---

# 6. Effect of Lambda in Ridge

Consider:

$$
Loss = MSE+\lambda\sum w_j^2
$$

### Small $\lambda$

For example:

$$
\lambda=0.001
$$

Regularization is weak.

The model behaves similarly to Linear Regression.

### Large $\lambda$

For example:

$$
\lambda=100
$$

Large weights are heavily penalized.

The coefficients shrink considerably.

### Extremely large $\lambda$

The coefficients can become very close to zero.

The model becomes too simple and may underfit.

So:

$$
\lambda \uparrow
\Rightarrow
Weights \downarrow
\Rightarrow
Model\ Complexity \downarrow
$$

---

# 7. Ridge Regression Example

Suppose your model is:

$$
y = 5+10x_1+20x_2
$$

The regularization penalty is:

$$
10^2+20^2
$$

$$
=100+400
$$

$$
=500
$$

If:

$$
\lambda=0.1
$$

then:

$$
Penalty=0.1(500)=50
$$

The model therefore tries to balance:

$$
Prediction\ Error + 50
$$

The optimization may prefer something like:

$$
y=5+7x_1+14x_2
$$

if that sufficiently reduces the overall objective.

---

# 8. Ridge Regression in sklearn

```python
from sklearn.linear_model import Ridge

model = Ridge(alpha=1.0)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
```

In sklearn:

$$
\boxed{\alpha \approx \text{regularization strength}}
$$

So:

```python
Ridge(alpha=0.01)
```

means weak regularization, while:

```python
Ridge(alpha=100)
```

means strong regularization.

---

# 9. Important Point: Feature Scaling

Regularization is heavily affected by feature scale.

Suppose:

```text
Age       = 20 - 60
Salary    = 20,000 - 500,000
```

The numerical scales are very different.

Because regularization operates on coefficients, the model can unfairly penalize some features.

Therefore, when using Ridge or Lasso, **feature scaling is usually important**.

A common pipeline is:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

model = Pipeline([
    ("scaler", StandardScaler()),
    ("ridge", Ridge(alpha=1.0))
])

model.fit(X_train, y_train)
```

---

# 10. Lasso Regression — L1 Regularization

Lasso uses the absolute values of the coefficients.

Its objective function is:

$$
\boxed{
MSE+\lambda\sum_{j=1}^{n}|w_j|
}
$$

Compare:

### Ridge

$$
MSE+\lambda\sum w_j^2
$$

### Lasso

$$
MSE+\lambda\sum |w_j|
$$

The key difference is:

$$
\boxed{L2: w^2}
$$

versus:

$$
\boxed{L1: |w|}
$$

---

# 11. Why Lasso is Special

Lasso can force some coefficients to become **exactly zero**.

For example:

Before Lasso:

$$
w=
[10,\ 5,\ 2,\ 0.5,\ 0.1]
$$

After Lasso:

$$
w=
[8.2,\ 3.7,\ 0,\ 0,\ 0]
$$

That means the model effectively removes some features.

Therefore:

$$
\boxed{\text{Lasso performs feature selection}}
$$

This is one of its biggest advantages.

---

# 12. Lasso Example

Suppose you have 100 features:

```text
Age
Salary
Experience
Height
Weight
City
Temperature
...
```

but only 10 are actually useful.

Lasso might produce:

```text
Feature        Coefficient

Age             0.82
Salary          1.42
Experience      0.67
Height          0
Weight          0
City             0
Temperature      0
...
```

The zero coefficients mean those features aren't being used by the model.

---

# 13. Lasso in sklearn

```python
from sklearn.linear_model import Lasso

model = Lasso(alpha=0.1)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
```

You can inspect the coefficients:

```python
print(model.coef_)
```

You'll often see:

```text
[1.24, 0.82, 0, 0, 0.31, 0, ...]
```

---

# 14. Ridge vs Lasso

| Property | Ridge | Lasso |
|---|---|---|
| Regularization | L2 | L1 |
| Penalty | $w^2$ | $|w|$ |
| Shrinks coefficients | Yes | Yes |
| Makes coefficients exactly zero | Usually no | Yes |
| Feature selection | No | Yes |
| Useful with correlated features | Generally good | Can select one and ignore others |
| Stability with many correlated features | Often better | Can be less stable |

The simplest way to remember:

$$
\boxed{\text{Ridge = shrink}}
$$

$$
\boxed{\text{Lasso = shrink + select}}
$$

---

# 15. Elastic Net

Elastic Net combines Ridge and Lasso.

Its objective function contains both penalties:

$$
\boxed{
MSE+
\lambda_1\sum |w_j|
+
\lambda_2\sum w_j^2
}
$$

Or commonly expressed using a total regularization parameter and mixing ratio:

$$
MSE+
\alpha
\left(
l1\_ratio\sum|w_j|
+
(1-l1\_ratio)\sum w_j^2
\right)
$$

---

# 16. Why Elastic Net?

Imagine you have highly correlated features:

```text
x1 = years of experience
x2 = age
x3 = career duration
```

These features may contain very similar information.

Lasso may arbitrarily choose one:

```text
x1 = 2.4
x2 = 0
x3 = 0
```

Elastic Net can behave more smoothly by combining L1 and L2 regularization.

For example:

```text
x1 = 1.4
x2 = 0.9
x3 = 0.7
```

So Elastic Net is especially useful when you have:

- Many features
- Correlated features
- Some irrelevant features
- A need for feature selection

---

# 17. Elastic Net in sklearn

```python
from sklearn.linear_model import ElasticNet

model = ElasticNet(
    alpha=0.1,
    l1_ratio=0.5
)

model.fit(X_train, y_train)
```

Interpretation:

```text
l1_ratio = 1
```

is essentially Lasso.

```text
l1_ratio = 0
```

is essentially Ridge.

```text
l1_ratio = 0.5
```

is an equal mixture.

---

# 18. Visual Intuition: Why Lasso Creates Zero Coefficients

This is one of the most interesting parts of regularization.

For two coefficients $w_1$ and $w_2$:

### Ridge constraint

$$
w_1^2+w_2^2\leq c
$$

creates a circular constraint.

### Lasso constraint

$$
|w_1|+|w_2|\leq c
$$

creates a diamond-shaped constraint.

The diamond has sharp corners exactly on the axes.

During optimization, the solution is therefore more likely to land on:

$$
w_1=0
$$

or:

$$
w_2=0
$$

Hence Lasso naturally produces sparse models.

---

# 19. Regularization and Gradient Descent

Since you've been studying Gradient Descent, this connection is important.

Suppose ordinary Linear Regression has:

$$
J(w)=MSE
$$

For Ridge:

$$
J(w)=MSE+\lambda\sum w_j^2
$$

The derivative of the regularization term is:

$$
\frac{\partial}{\partial w_j}
\lambda w_j^2
=
2\lambda w_j
$$

So the gradient becomes:

$$
\frac{\partial J}{\partial w_j}
=
\frac{\partial MSE}{\partial w_j}
+
2\lambda w_j
$$

Gradient descent:

$$
w_j := w_j-\eta
\left(
\frac{\partial MSE}{\partial w_j}
+
2\lambda w_j
\right)
$$

where $\eta$ is the learning rate.

---

# 20. Why Ridge "Shrinks" Weights

Look at:

$$
w_j := w_j-\eta
\left(
\frac{\partial MSE}{\partial w_j}
+
2\lambda w_j
\right)
$$

Expand:

$$
w_j :=
w_j
-\eta\frac{\partial MSE}{\partial w_j}
-2\eta\lambda w_j
$$

Therefore:

$$
w_j :=
(1-2\eta\lambda)w_j
-\eta\frac{\partial MSE}{\partial w_j}
$$

The term:

$$
(1-2\eta\lambda)w_j
$$

directly shrinks the weight.

This is why L2 regularization is often called **weight decay**.

---
