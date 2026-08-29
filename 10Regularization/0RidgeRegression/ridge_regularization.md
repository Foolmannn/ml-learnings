
# Ridge Regression in Detail

Ridge Regression is one of the most important **regularization techniques for linear regression**. Since you're learning gradient descent and polynomial regression, Ridge is especially important because it connects all three ideas:

$$
\boxed{\text{Linear Regression} + \text{L2 Regularization} = \text{Ridge Regression}}
$$

The main purpose of Ridge Regression is to **reduce overfitting by penalizing large model coefficients**.

---

# 1. First: Ordinary Linear Regression

Suppose we have one feature:

$$
y = b_0+b_1x
$$

For multiple features:

$$
\hat y=b_0+b_1x_1+b_2x_2+\cdots+b_nx_n
$$

Ordinary Linear Regression tries to find the coefficients that minimize the squared prediction error:

$$
J(b)=\sum_{i=1}^{m}(y_i-\hat y_i)^2
$$

or using MSE:

$$
MSE=\frac{1}{m}\sum_{i=1}^{m}(y_i-\hat y_i)^2
$$

The model's only concern is:

> "How can I make my predictions as close to the training data as possible?"

This can cause a problem.

---

# 2. The Problem: Overfitting

Imagine you have a polynomial model:

$$
\hat y =
b_0+b_1x+b_2x^2+b_3x^3+\cdots+b_{10}x^{10}
$$

The model has many parameters.

It can potentially learn extremely complicated patterns from the training data.

For example:

```text
Training R² = 0.99
Testing R²  = 0.65
```

This indicates that the model has learned the training data extremely well but doesn't generalize well.

One reason is that the coefficients can become very large.

For example:

$$
b_1=15
$$

$$
b_2=-82
$$

$$
b_3=240
$$

$$
b_4=-510
$$

These large coefficients can make the model highly sensitive to small changes in input.

---

# 3. Ridge's Basic Idea

Ridge says:

> "Don't just minimize prediction error. Also penalize large coefficients."

So instead of:

$$
Loss=MSE
$$

we use:

$$
Loss=MSE+\text{Penalty}
$$

Ridge uses the **L2 penalty**:

$$
\boxed{
Loss=MSE+\lambda\sum_{j=1}^{n}b_j^2
}
$$

This is the fundamental Ridge equation.

Here:

- $MSE$ = prediction error
- $b_j$ = coefficient
- $\lambda$ = regularization strength
- $\sum b_j^2$ = L2 penalty

Notice that $b_0$, the intercept, is generally excluded from the penalty.

---

# 4. What Does the Lambda Mean?

The most important Ridge hyperparameter is:

$$
\boxed{\lambda}
$$

It controls how strongly we penalize large coefficients.

### Small λ

$$
\lambda\approx0
$$

Very little regularization.

Ridge behaves similarly to ordinary Linear Regression.

### Medium λ

The coefficients are reduced.

The model becomes less complex and may generalize better.

### Very large λ

The coefficients are pushed strongly toward zero.

The model can become too simple.

So:

$$
\lambda\uparrow
\Rightarrow
|b_j|\downarrow
$$

But remember:

$$
\boxed{\lambda\text{ usually does not make coefficients exactly zero}}
$$

That's an important difference from Lasso.

---

# 5. Simple Numerical Example

Suppose our model is:

$$
\hat y=5+10x_1+20x_2
$$

The coefficients are:

$$
b_1=10
$$

$$
b_2=20
$$

The Ridge penalty is:

$$
b_1^2+b_2^2
$$

Therefore:

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

then the regularization penalty is:

$$
0.1(500)=50
$$

Therefore:

$$
Total\ Loss=MSE+50
$$

The optimization algorithm now has two objectives:

1. Reduce MSE.
2. Reduce coefficient magnitude.

---

# 6. Why Doesn't Ridge Simply Set Every Weight to Zero?

This is an important question.

Suppose:

$$
b_1=10
$$

Ridge penalizes:

$$
10^2=100
$$

So it has an incentive to reduce the coefficient.

But reducing $b_1$ also affects prediction error.

Therefore, the model has to balance:

$$
\boxed{\text{Prediction accuracy}}
$$

against:

$$
\boxed{\text{Small coefficients}}
$$

The optimal solution might be:

$$
b_1=6.8
$$

rather than:

$$
b_1=0
$$

So Ridge generally **shrinks** coefficients rather than eliminating them.

---

# 7. Ridge and Gradient Descent

This is where Ridge connects directly with the Gradient Descent you've been studying.

Suppose ordinary Linear Regression has:

$$
J(w)=MSE
$$

Ridge changes it to:

$$
J(w)=MSE+\lambda\sum_{j=1}^{n}w_j^2
$$

For one weight:

$$
J(w)=MSE+\lambda w^2
$$

Take the derivative:

$$
\frac{\partial J}{\partial w}
=
\frac{\partial MSE}{\partial w}
+
\frac{\partial}{\partial w}(\lambda w^2)
$$

Since:

$$
\frac{d}{dw}w^2=2w
$$

we get:

$$
\boxed{
\frac{\partial J}{\partial w}
=
\frac{\partial MSE}{\partial w}+2\lambda w
}
$$

Therefore, Gradient Descent becomes:

$$
w:=w-\eta
\left(
\frac{\partial MSE}{\partial w}+2\lambda w
\right)
$$

where:

- $w$ = coefficient
- $\eta$ = learning rate
- $\lambda$ = regularization strength

---

# 8. The Weight Decay Interpretation

Let's expand that update:

$$
w:=w-\eta\frac{\partial MSE}{\partial w}-2\eta\lambda w
$$

Rearrange:

$$
w:=
(1-2\eta\lambda)w
-
\eta\frac{\partial MSE}{\partial w}
$$

Look at:

$$
(1-2\eta\lambda)w
$$

This term directly reduces the magnitude of the weight.

That's why L2 regularization is often associated with **weight decay**.

The intuition is:

```text
Ordinary Gradient Descent:

      prediction error
             ↓
           update
             ↓
             w


Ridge:

      prediction error
             ↓
           update
             ↓
             w
             ↑
       weight shrinking
```

So every update has both:

- an error-correction component
- a coefficient-shrinking component

---

# 9. Ridge Closed-Form Solution

Ordinary Linear Regression has the famous normal equation:

$$
\boxed{
\theta=(X^TX)^{-1}X^Ty
}
$$

Ridge modifies this to:

$$
\boxed{
\theta=(X^TX+\lambda I)^{-1}X^Ty
}
$$

This is one of the most important equations for Ridge Regression.

Where:

- $X$ = feature matrix
- $y$ = target vector
- $\theta$ = coefficient vector
- $I$ = identity matrix
- $\lambda$ = regularization strength

---

# 10. Why Add λI?

This is a very important mathematical reason for Ridge.

Suppose:

$$
X^TX
$$

is singular or nearly singular.

Then:

$$
(X^TX)^{-1}
$$

may not exist or may be numerically unstable.

Ridge changes it to:

$$
X^TX+\lambda I
$$

The addition of:

$$
\lambda I
$$

makes the matrix better conditioned in many problematic cases.

This makes Ridge particularly useful when features are highly correlated.

---

# 11. Multicollinearity

Suppose your dataset contains:

```text
Age
Years of Experience
Years Since Graduation
Career Duration
```

These features may be strongly correlated.

For example:

$$
Age \approx Experience + 22
$$

Ordinary Linear Regression can have unstable coefficients.

You might get:

```text
Experience       = 150
Age              = -130
GraduationYear   = 200
```

The individual coefficients can become strange even if the predictions are reasonably good.

Ridge stabilizes them by discouraging very large coefficients.

This is one of Ridge's biggest advantages.

---

# 12. Ridge and Multicollinearity

Suppose:

$$
x_1\approx x_2
$$

Ordinary Linear Regression might distribute weights like:

$$
w_1=100
$$

$$
w_2=-95
$$

Their combined contribution could be reasonable, but the individual coefficients are unstable.

Ridge might instead produce:

$$
w_1=3.2
$$

$$
w_2=3.0
$$

The predictions can remain good while the coefficients become much more stable.

So:

$$
\boxed{
\text{Ridge is very useful when features are correlated}
}
$$

---

# 13. Geometric Interpretation

This is one of the best ways to understand Ridge.

Suppose we have two coefficients:

$$
w_1,w_2
$$

Ridge imposes a constraint:

$$
w_1^2+w_2^2\leq c
$$

Geometrically, this is a **circle**.

The ordinary regression objective has its own contours.

Ridge searches for the best solution inside that circle.

The optimum therefore tends to have smaller coefficients.

Compare this with Lasso:

$$
|w_1|+|w_2|\leq c
$$

which creates a diamond.

The corners of the Lasso diamond lie on the axes, making zero coefficients much more likely.

Ridge has a smooth circular boundary, so it generally shrinks coefficients without making them exactly zero.

---

# 14. Ridge vs Ordinary Linear Regression

Suppose ordinary Linear Regression gives:

$$
w=
[15,-20,30,5]
$$

Ridge might produce:

$$
w=
[8.2,-11.5,17.4,3.1]
$$

The weights have been reduced.

But they are still non-zero.

That's the characteristic behavior of Ridge.

---

# 15. Ridge vs Lasso

This distinction is extremely important.

| Property | Linear Regression | Ridge | Lasso |
|---|---:|---:|---:|
| Regularization | ❌ | L2 | L1 |
| Penalizes large coefficients | ❌ | ✅ | ✅ |
| Shrinks coefficients | ❌ | ✅ | ✅ |
| Can make coefficient exactly 0 | — | Usually ❌ | ✅ |
| Feature selection | ❌ | ❌ | ✅ |
| Handles multicollinearity | Less stable | Very good | Can be unstable with correlated features |

Memory trick:

$$
\boxed{\text{Ridge = shrink}}
$$

$$
\boxed{\text{Lasso = shrink + eliminate}}
$$

---

# 16. What Happens When λ = 0?

Start with:

$$
\theta=(X^TX+\lambda I)^{-1}X^Ty
$$

If:

$$
\lambda=0
$$

then:

$$
\theta=(X^TX)^{-1}X^Ty
$$

which is exactly ordinary Linear Regression.

Therefore:

$$
\boxed{
Ridge(\lambda=0)=Linear\ Regression
}
$$

This is a useful theoretical relationship.

---

# 17. What Happens as λ → ∞?

Suppose:

$$
\lambda\rightarrow\infty
$$

The penalty for large coefficients becomes enormous.

The optimization increasingly prefers:

$$
w_1,w_2,\ldots,w_n\rightarrow0
$$

The model approaches something like:

$$
\hat y\approx b_0
$$

In other words, the model becomes extremely simple.

Therefore:

$$
\boxed{
\lambda\rightarrow\infty
\Rightarrow
\text{strong underfitting}
}
$$

---

# 18. Bias-Variance Tradeoff

Ridge provides a classic example of the bias-variance tradeoff.

As regularization increases:

$$
\lambda\uparrow
$$

generally:

$$
Bias\uparrow
$$

and:

$$
Variance\downarrow
$$

Initially, reducing variance can improve test performance.

But eventually, excessive bias causes underfitting.

Conceptually:

```text
Regularization

Low λ                    High λ
  │                         │
  ▼                         ▼

Complex model          Simple model

Low bias               High bias
High variance          Low variance

Overfitting            Underfitting
```

The ideal $\lambda$ is somewhere between the two extremes.

---

# 19. Training Error vs Validation Error

Suppose we try different values of alpha:

| Alpha | Train MSE | Validation MSE |
|---:|---:|---:|
| 0 | 2.1 | 8.5 |
| 0.01 | 2.2 | 7.2 |
| 0.1 | 2.4 | 5.1 |
| 1 | 3.0 | 3.8 |
| 10 | 6.5 | 6.2 |
| 100 | 20.0 | 20.5 |

Here:

$$
\alpha=1
$$

would be preferable because it gives the lowest validation error.

Notice something important:

> The best regularized model doesn't necessarily have the lowest training error.

That's perfectly normal.

---

# 20. Ridge in sklearn

The implementation is straightforward:

```python
from sklearn.linear_model import Ridge

model = Ridge(alpha=1.0)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
```

Here:

```python
alpha=1.0
```

controls regularization strength.

---

# 21. `alpha` vs `lambda`

Mathematically, we often write:

$$
\lambda
$$

for regularization strength.

Scikit-learn's Ridge implementation uses:

```python
alpha
```

So conceptually:

$$
\boxed{\alpha\approx\lambda}
$$

Don't confuse this with Logistic Regression's `C`.

For Ridge:

```text
alpha ↑ → stronger regularization
alpha ↓ → weaker regularization
```

For Logistic Regression's `C`:

```text
C ↑ → weaker regularization
C ↓ → stronger regularization
```

---

# 22. Feature Scaling Before Ridge

This is extremely important.

Suppose:

```text
Age:       18 → 60
Salary:    20,000 → 2,000,000
Experience: 0 → 40
```

The features have completely different scales.

Because Ridge penalizes coefficient magnitude:

$$
\lambda\sum w_j^2
$$

the scale of the features affects the coefficient values.

Therefore, you generally want to scale numerical features before applying Ridge.

A good sklearn implementation is:

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

# 23. Why Scaling Changes the Coefficients

Consider:

$$
y=10x
$$

If:

$$
x=0-1
$$

the coefficient might be around:

$$
w=10
$$

But if we define:

$$
z=1000x
$$

then:

$$
y=0.01z
$$

Now the coefficient is:

$$
w=0.01
$$

Same underlying relationship.

But Ridge sees:

$$
10^2=100
$$

versus:

$$
0.01^2=0.0001
$$

So without scaling, the regularization penalty isn't treating coefficients in a comparable way.

That's why scaling is generally recommended.

---

# 24. Ridge with Polynomial Regression

This is one of the most useful applications.

Suppose you create:

$$
x,x^2,x^3,\ldots,x^{20}
$$

using:

```python
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=20)
```

You may get severe overfitting.

Instead of reducing the degree immediately, you can use Ridge:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge

model = Pipeline([
    ("poly", PolynomialFeatures(degree=20)),
    ("scaler", StandardScaler()),
    ("ridge", Ridge(alpha=1.0))
])

model.fit(X_train, y_train)
```

Now the model has access to a rich polynomial feature space but is discouraged from using enormous coefficients.

This is a powerful combination:

$$
\boxed{
Polynomial\ Features + Ridge
}
$$

---

# 25. Ridge Doesn't Automatically "Remove" Features

Suppose you have:

```text
Age
Salary
Experience
Height
Weight
City
```

Ridge might produce:

```text
Age          0.72
Salary       1.41
Experience   0.63
Height       0.04
Weight       0.02
City        -0.01
```

Even irrelevant features can retain small non-zero coefficients.

That's because Ridge's primary goal is:

$$
\boxed{\text{coefficient shrinkage}}
$$

not feature elimination.

If you specifically want coefficients to become zero, Lasso is more appropriate.

---

# 26. Ridge with Gradient Descent: From Scratch

You can implement a simplified Ridge Regression yourself.

For ordinary linear regression:

$$
\frac{\partial MSE}{\partial w_j}
=
-\frac{2}{m}
\sum_{i=1}^{m}
x_{ij}(y_i-\hat y_i)
$$

For Ridge:

$$
\boxed{
\frac{\partial J}{\partial w_j}
=
-\frac{2}{m}
\sum_{i=1}^{m}
x_{ij}(y_i-\hat y_i)
+
2\lambda w_j
}
$$

Therefore:

```python
for epoch in range(epochs):

    y_pred = X @ weights + intercept

    error = y_pred - y

    dw = (2 / m) * X.T @ error + 2 * alpha * weights

    db = (2 / m) * error.sum()

    weights -= learning_rate * dw
    intercept -= learning_rate * db
```

Notice:

```python
+ 2 * alpha * weights
```

That's the Ridge regularization term.

The intercept isn't penalized.

---

# 27. Important Detail for Your Gradient Descent Learning

Suppose:

```python
weights = [10, 20]
alpha = 0.1
```

The Ridge contribution to the gradient is:

$$
2\alpha w
$$

Therefore:

$$
2(0.1)[10,20]
$$

$$
=[2,4]
$$

So the regularization gradient itself pushes the weights downward.

If:

$$
w=-20
$$

then:

$$
2\lambda w
$$

is negative.

When Gradient Descent subtracts that negative quantity, it pushes the coefficient toward zero.

So both positive and negative coefficients are pushed toward zero.

---

# 28. Ridge and Your MBGD Implementation

If you're implementing Mini-Batch Gradient Descent, the concept is almost identical.

Suppose your existing update is:

```python
dw = (2 / batch_size) * X_batch.T @ error
```

For Ridge you conceptually modify it to:

```python
dw = (
    (2 / batch_size) * X_batch.T @ error
    + 2 * alpha * weights
)
```

Then:

```python
weights -= learning_rate * dw
```

The regularization term is added to the gradient **for the weights**.

Don't apply it to the intercept.

---

# 29. Choosing Alpha

Don't arbitrarily choose:

```python
alpha=1
```

You can search over multiple values.

For example:

```python
alphas = [
    0.0001,
    0.001,
    0.01,
    0.1,
    1,
    10,
    100,
    1000
]
```

Then evaluate with cross-validation.

```python
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge

params = {
    "alpha": [0.001, 0.01, 0.1, 1, 10, 100]
}

grid = GridSearchCV(
    Ridge(),
    params,
    cv=5,
    scoring="r2"
)

grid.fit(X_train, y_train)

print(grid.best_params_)
```

---

# 30. RidgeCV

Scikit-learn also provides:

```python
from sklearn.linear_model import RidgeCV

model = RidgeCV(
    alphas=[0.001, 0.01, 0.1, 1, 10, 100],
    cv=5
)

model.fit(X_train, y_train)

print(model.alpha_)
```

This automatically selects the best alpha according to cross-validation.

---

# 31. Ridge and the Normal Equation

Let's compare the two formulas.

### Ordinary Linear Regression

$$
\boxed{
\theta=(X^TX)^{-1}X^Ty
}
$$

### Ridge Regression

$$
\boxed{
\theta=(X^TX+\lambda I)^{-1}X^Ty
}
$$

The only major mathematical change is:

$$
\boxed{X^TX\rightarrow X^TX+\lambda I}
$$

This tiny-looking modification has significant consequences:

- reduces coefficient magnitude
- reduces variance
- improves stability
- handles multicollinearity better
- can improve generalization

---

# 32. Bayesian Interpretation

There is also a beautiful statistical interpretation.

Ridge Regression corresponds to placing a **Gaussian prior** on the coefficients:

$$
w_j\sim N(0,\sigma^2)
$$

In other words, before seeing the data, we're expressing a preference for coefficients close to zero.

The model then combines:

$$
\text{Data evidence}
$$

with:

$$
\text{Prior belief that large coefficients are unlikely}
$$

This leads to the Ridge solution.

You don't need this interpretation to use Ridge, but it's useful when you later study Bayesian Machine Learning.

---

# 33. When Should You Use Ridge?

Ridge is particularly useful when:

### 1. You have many features

Especially when:

$$
n\gg1
$$

or even:

$$
n>m
$$

where features can be numerous relative to observations.

### 2. Features are correlated

Ridge is excellent for multicollinearity.

### 3. You believe most features contain useful information

Unlike Lasso, Ridge doesn't try to eliminate features.

### 4. Your model is overfitting

For example:

$$
Train\ R^2=0.98
$$

but:

$$
Test\ R^2=0.60
$$

### 5. Polynomial features are causing large coefficients

Ridge can stabilize high-degree polynomial models.

---

# 34. When Might Ridge Not Be the Best Choice?

If you have thousands of features and believe that only a small subset are useful, Lasso may be more appropriate.

For example:

```text
10,000 features
    ↓
Only 100 actually useful
```

Lasso can potentially produce:

```text
9,900 coefficients = 0
100 coefficients ≠ 0
```

Ridge generally won't do that.

In such a case:

$$
\boxed{\text{Lasso or Elastic Net}}
$$

may be preferable.

---

# 35. Complete Practical Pipeline

A typical Ridge workflow looks like:

```python
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("ridge", Ridge())
])

params = {
    "ridge__alpha": [
        0.001,
        0.01,
        0.1,
        1,
        10,
        100
    ]
}

grid = GridSearchCV(
    pipeline,
    params,
    cv=5,
    scoring="r2"
)

grid.fit(X_train, y_train)

print("Best alpha:", grid.best_params_)
print("Test R²:", grid.score(X_test, y_test))
```

The important architecture is:

$$
\boxed{
X
\rightarrow
Train/Test Split
\rightarrow
Scaling
\rightarrow
Ridge
\rightarrow
Cross Validation
\rightarrow
Best\ \alpha
\rightarrow
Test
}
$$

---

# 36. Ridge vs Increasing Data

Regularization isn't the only way to reduce overfitting.

You can also increase the amount of training data.

Conceptually:

$$
\text{More data}
\rightarrow
\text{better estimation}
\rightarrow
\text{less variance}
$$

while:

$$
\text{Ridge}
\rightarrow
\text{constrain coefficients}
\rightarrow
\text{less variance}
$$

So both can help reduce overfitting, but they do it differently.

---

# 37. The Core Intuition

Imagine ordinary Linear Regression says:

> "Fit the training data as accurately as possible."

Ridge says:

> "Fit the training data accurately, **but don't use unnecessarily large coefficients to do it.**"

Mathematically:

$$
\boxed{
\underbrace{MSE}_{\text{fit data}}
+
\underbrace{\lambda\sum w_j^2}_{\text{control complexity}}
}
$$

That's Ridge Regression in one equation.

---

# 38. Final Mental Model

Keep this chain in your head:

```text
Linear Regression
       ↓
Can overfit
       ↓
Add penalty
       ↓
L2 penalty
       ↓
Ridge Regression
       ↓
Large weights are penalized
       ↓
Weights shrink
       ↓
Variance decreases
       ↓
Generalization can improve
```

And remember these five facts:

$$
\boxed{1.\ Ridge=L2\ Regularization}
$$

$$
\boxed{2.\ Penalty=\lambda\sum w_j^2}
$$

$$
\boxed{3.\ Large\ weights\ are\ penalized}
$$

$$
\boxed{4.\ Coefficients\ shrink\ but\ usually\ don't\ become\ zero}
$$

$$
\boxed{5.\ Larger\ \lambda\rightarrow stronger\ regularization}
$$

The most important connection for your current ML studies is:

$$
\boxed{
\text{Ridge}
\rightarrow
\text{L2 penalty}
\rightarrow
\text{extra gradient }2\lambda w
\rightarrow
\text{weight shrinkage}
\rightarrow
\text{lower variance}
\rightarrow
\text{less overfitting}
}
$$

That connection between the **loss function → derivative → Gradient Descent update → coefficient shrinkage** is the part worth mastering, because the exact same reasoning will make **Lasso, Elastic Net, and regularization in neural networks** much easier to understand.