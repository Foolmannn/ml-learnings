

# Elastic Net Regression — In Detail

**Elastic Net Regression** is a regularized linear regression technique that combines the advantages of **Ridge Regression (L2 regularization)** and **Lasso Regression (L1 regularization)**.

If you already understand Ridge and Lasso, the easiest way to remember Elastic Net is:

> **Elastic Net = Lasso + Ridge**

It is especially useful when you have **many features**, **multicollinearity**, or when you want **some coefficients to become exactly zero** while also keeping groups of correlated features.

---

# 1. Why do we need Elastic Net?

Start with ordinary Linear Regression:

$$
\hat y = \beta_0+\beta_1x_1+\beta_2x_2+\cdots+\beta_px_p
$$

Linear Regression tries to minimize the **sum of squared errors**:

$$
\boxed{
J(\beta)=\sum_{i=1}^{n}(y_i-\hat y_i)^2
}
$$

The problem is that ordinary Linear Regression can suffer from:

- Overfitting
- Large coefficients
- Multicollinearity
- Unstable coefficients
- Poor performance when the number of features is large

This is where regularization comes in.

---

# 2. Quick review: Ridge vs Lasso

### Ridge Regression

Ridge adds an **L2 penalty**:

$$
J(\beta)=RSS+\lambda\sum_{j=1}^{p}\beta_j^2
$$

It tends to:

- Shrink coefficients
- Reduce overfitting
- Handle multicollinearity well
- Keep **all features**
- Usually does **not** make coefficients exactly zero

---

### Lasso Regression

Lasso adds an **L1 penalty**:

$$
J(\beta)=RSS+\lambda\sum_{j=1}^{p}|\beta_j|
$$

It tends to:

- Shrink coefficients
- Reduce overfitting
- Perform feature selection
- Make some coefficients **exactly 0**

For example:

```text
Before Lasso:

Age          2.31
Income       5.72
Experience   1.84
Height       0.03
Weight       0.01

After Lasso:

Age          1.91
Income       4.83
Experience   1.27
Height       0
Weight       0
```

So Lasso creates **sparsity**.

---

# 3. The problem with using only Lasso

Suppose we have highly correlated features:

```text
X1 = House area
X2 = Number of rooms
X3 = Number of bedrooms
```

These features may contain very similar information.

Lasso can behave somewhat unpredictably with correlated features.

For example:

```text
X1 → 4.5
X2 → 0
X3 → 0
```

It might select one feature and eliminate the others.

But perhaps you want to retain the information from the correlated group.

This is one of the situations where **Elastic Net** becomes useful.

---

# 4. What is Elastic Net?

Elastic Net combines:

- **L1 regularization** → Lasso
- **L2 regularization** → Ridge

Conceptually:

$$
\boxed{
\text{Elastic Net}
=
\text{Linear Regression}
+
\text{L1 penalty}
+
\text{L2 penalty}
}
$$

A common formulation is:

$$
\boxed{
J(\beta)
=
RSS
+
\lambda
\left[
\alpha\sum_{j=1}^{p}|\beta_j|
+
(1-\alpha)\sum_{j=1}^{p}\beta_j^2
\right]
}
$$

where:

- $\lambda$ = overall regularization strength
- $\alpha$ = balance between Lasso and Ridge
- $RSS$ = residual sum of squares

---

# 5. Understanding α

The most important parameter in Elastic Net is often called **alpha**.

$$
0\leq\alpha\leq1
$$

It controls how much L1 and L2 regularization are used.

### α = 1

$$
\alpha=1
$$

Then:

$$
\text{Elastic Net} \approx \text{Lasso}
$$

because:

$$
(1-\alpha)=0
$$

---

### α = 0

$$
\alpha=0
$$

Then:

$$
\text{Elastic Net} \approx \text{Ridge}
$$

because the L1 component disappears.

---

### α = 0.5

$$
\alpha=0.5
$$

You have an equal mixture of L1 and L2.

```text
α

0                     0.5                     1
|----------------------|----------------------|
Ridge              Elastic Net              Lasso
L2                   L1 + L2                  L1
```

---

# 6. What does λ do?

While $\alpha$ controls **the type of regularization**, $\lambda$ controls **how strong the regularization is**.

Consider:

$$
J = RSS + \lambda(Penalty)
$$

### Small λ

The penalty is weak:

```text
λ → 0

Model behaves more like Linear Regression
```

Potential problem:

> Overfitting

---

### Large λ

The penalty becomes strong:

```text
λ → large

Coefficients get strongly shrunk
```

Potential problem:

> Underfitting

So:

$$
\boxed{
\lambda \uparrow
\Rightarrow
\text{more regularization}
}
$$

and

$$
\boxed{
\lambda \downarrow
\Rightarrow
\text{less regularization}
}
$$

---

# 7. α vs λ — very important

These two parameters are easy to confuse.

| Parameter | Controls |
|---|---|
| $\alpha$ | Balance between L1 and L2 |
| $\lambda$ | Overall regularization strength |

Think:

```text
α → "What kind of regularization?"

λ → "How much regularization?"
```

For example:

```text
α = 0.8
λ = 0.1
```

means:

- Mostly L1
- Some L2
- Overall regularization strength = 0.1

---

# 8. Elastic Net objective function

Let's write it more carefully.

Suppose:

$$
\hat y_i
=
\beta_0+\beta_1x_{i1}+\cdots+\beta_px_{ip}
$$

The Elastic Net objective can be written as:

$$
\boxed{
J(\beta)
=
\frac{1}{2n}
\sum_{i=1}^{n}
(y_i-\hat y_i)^2
+
\lambda
\left[
\alpha\sum_{j=1}^{p}|\beta_j|
+
\frac{1-\alpha}{2}\sum_{j=1}^{p}\beta_j^2
\right]
}
$$

Different libraries/papers use slightly different scaling conventions.

**Don't get stuck on the factors $1/2$ or $1/n$.**

The important idea is:

$$
\boxed{
\text{Loss}
+
\text{L1 penalty}
+
\text{L2 penalty}
}
$$

---

# 9. Why combine L1 and L2?

Because they solve different problems.

### L1

$$
\sum|\beta_j|
$$

encourages:

$$
\beta_j=0
$$

Therefore:

> **Feature selection**

---

### L2

$$
\sum\beta_j^2
$$

encourages coefficients to become smaller without necessarily becoming zero.

Therefore:

> **Coefficient stability + multicollinearity handling**

---

### Elastic Net

Combines both:

```text
             Elastic Net
                  │
          ┌───────┴───────┐
          ↓               ↓
         L1              L2
          │               │
          ↓               ↓
 Feature selection    Shrinkage
 Sparsity             Stability
          │               │
          └───────┬───────┘
                  ↓
        Better handling of
        correlated features
```

---

# 10. The "grouping effect"

This is one of the most important advantages of Elastic Net.

Suppose:

$$
X_1 \approx X_2
$$

meaning they are highly correlated.

For example:

```text
X1 = house size
X2 = house area in square feet
```

They may contain almost identical information.

Elastic Net tends to give correlated variables **similar coefficients** rather than arbitrarily selecting only one.

For example:

### Lasso

```text
X1 → 5.2
X2 → 0
X3 → 0
```

### Elastic Net

```text
X1 → 3.1
X2 → 2.8
X3 → 0
```

This behavior is called the **grouping effect**.

It is a major reason to use Elastic Net when features are strongly correlated.

---

# 11. Why does L1 create sparsity?

This connects directly to your previous question about Lasso.

The L1 penalty is:

$$
|\beta|
$$

It has a sharp point at:

$$
\beta=0
$$

Because of this geometry, optimization can push coefficients directly onto zero.

L2:

$$
\beta^2
$$

has a smooth curve and generally shrinks coefficients toward zero without producing as many exact zeros.

Therefore:

```text
L1 → many exact zeros
L2 → small coefficients
```

Elastic Net:

```text
L1 + L2
   ↓
some exact zeros
+
stable/shrunk coefficients
```

---

# 12. Geometric intuition

For two coefficients $\beta_1,\beta_2$:

### Ridge constraint

The L2 constraint creates a circle:

```text
        ______
      /        \
     |          |
      \________/
```

### Lasso constraint

The L1 constraint creates a diamond:

```text
         /\
        /  \
       /    \
       \    /
        \  /
         \/
```

The corners of the Lasso diamond lie on the axes.

Therefore, the optimum frequently occurs at:

$$
\beta_1=0
$$

or

$$
\beta_2=0
$$

which produces sparsity.

---

### Elastic Net

Elastic Net combines the L1 diamond-like behavior with the L2 circular behavior.

Conceptually:

```text
Lasso                Ridge
  ◇                    ○
   \                  /
    \                /
     └── Elastic ───┘
          Net
```

So you get both:

- sparsity from L1
- stability from L2

---

# 13. Example

Suppose we have:

```text
10,000 features
1,000 training examples
```

Many features are correlated.

Ordinary Linear Regression may struggle because:

$$
p > n
$$

where:

- $p$ = number of features
- $n$ = number of observations

Elastic Net can be very useful here.

It can produce something like:

```text
Feature       Coefficient

X1             2.31
X2             2.15
X3             0
X4             0
X5            -1.72
X6             0
X7             0.83
...
X10000         0
```

So perhaps only 300 of the 10,000 features remain non-zero.

This gives a **sparse model** while still handling correlated variables better than pure Lasso.

---

# 14. Elastic Net vs Linear Regression

| Property | Linear Regression | Elastic Net |
|---|---|---|
| Regularization | ❌ | ✅ |
| L1 penalty | ❌ | ✅ |
| L2 penalty | ❌ | ✅ |
| Feature selection | ❌ | ✅ |
| Handles multicollinearity | Poorer | Better |
| Coefficients shrink | ❌ | ✅ |
| Can create zeros | ❌ | ✅ |
| Overfitting control | Limited | Stronger |

---

# 15. Elastic Net vs Ridge

| Property | Ridge | Elastic Net |
|---|---|---|
| L1 | ❌ | ✅ |
| L2 | ✅ | ✅ |
| Feature selection | ❌ | ✅ |
| Exact zero coefficients | Rare | Common |
| Multicollinearity | Excellent | Excellent |
| Correlated features | Keeps them | Can keep groups |
| Sparse model | ❌ | ✅ |

---

# 16. Elastic Net vs Lasso

| Property | Lasso | Elastic Net |
|---|---|---|
| L1 | ✅ | ✅ |
| L2 | ❌ | ✅ |
| Feature selection | ✅ | ✅ |
| Sparsity | Strong | Strong |
| Multicollinearity | Can be problematic | Better |
| Correlated features | May select one | Often keeps groups |
| Stability | Lower with correlated features | Higher |

---

# 17. When should you use Elastic Net?

Elastic Net is particularly useful when:

### 1. You have many features

For example:

```text
10,000 features
1,000 samples
```

---

### 2. Features are highly correlated

For example:

```text
income
salary
annual earnings
monthly income
```

---

### 3. You want feature selection

You want:

```text
irrelevant features → coefficient = 0
```

---

### 4. You want regularization

You want to reduce:

> Overfitting

---

### 5. You have high-dimensional data

Common examples include:

- Genomics
- Text classification
- Image-derived features
- Financial modeling
- Sensor data
- Marketing data

---

# 18. When should you NOT necessarily use Elastic Net?

If you have:

```text
5 features
100,000 observations
little multicollinearity
```

ordinary Linear Regression may already work very well.

Also, if you have strong domain knowledge and don't want automated feature selection, regularization may not always be desirable.

---

# 19. Feature scaling is important

This is **very important** for Elastic Net.

Suppose:

```text
Age       = 20–60
Income    = 20,000–2,000,000
Experience = 0–30
```

The regularization penalty operates directly on coefficients.

Therefore, features should generally be standardized:

$$
z=\frac{x-\mu}{\sigma}
$$

Using:

```python
StandardScaler
```

before Elastic Net is usually a good practice.

---

# 20. Scikit-learn implementation

In scikit-learn:

```python
from sklearn.linear_model import ElasticNet

model = ElasticNet(
    alpha=0.1,
    l1_ratio=0.5
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
```

Notice something important:

### Scikit-learn uses `l1_ratio`

rather than calling the parameter `alpha` for the L1/L2 mixture.

Its parameters are:

```python
ElasticNet(
    alpha=0.1,
    l1_ratio=0.5
)
```

Here:

```text
alpha     → overall regularization strength
l1_ratio  → L1/L2 balance
```

This differs from the notation used in many mathematical explanations.

---

# 21. Scikit-learn parameter interpretation

Suppose:

```python
ElasticNet(
    alpha=0.5,
    l1_ratio=0.8
)
```

means approximately:

```text
Overall regularization strength
        ↓
      alpha
       0.5

L1 contribution
        ↓
   l1_ratio
       0.8

L2 contribution
        ↓
    1 - 0.8
       0.2
```

So:

```text
80% L1
20% L2
```

---
