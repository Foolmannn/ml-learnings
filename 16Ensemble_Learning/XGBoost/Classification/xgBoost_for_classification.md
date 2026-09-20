# XGBoost for Classification — In Detail

**XGBoost (Extreme Gradient Boosting)** is one of the most powerful and widely used algorithms for **classification problems**, especially for structured/tabular data.

Since you've already studied **Decision Trees, Random Forest, Bagging, AdaBoost, and Gradient Boosting**, XGBoost is a natural next step. It builds on gradient boosting but adds several improvements that make it faster, more regularized, and often more accurate.

---

# 1. What is XGBoost?

**XGBoost = Extreme Gradient Boosting**

It is an implementation of **gradient-boosted decision trees (GBDT)**.

The basic idea is:

> Instead of building one very powerful tree, build many relatively weak trees sequentially, where each new tree tries to correct the mistakes made by the previous trees.

For classification:

```text
Training Data
     ↓
Tree 1
     ↓
Errors / Gradients
     ↓
Tree 2
     ↓
Updated predictions
     ↓
Tree 3
     ↓
...
     ↓
Final prediction
```

The final model combines the predictions of all trees.

Mathematically:

$$
\hat{y}_i = \sum_{k=1}^{K} f_k(x_i)
$$

where:

* \(K\) = number of trees
* \(f_k\) = kth decision tree
* \(x_i\) = features of sample \(i\)
* \(\hat y_i\) = model prediction

For classification, this tree output is ultimately converted into a **probability**.

---

# 2. Why do we need XGBoost?

Suppose we want to classify whether a customer will leave a company.

Features:

| Age | Income | Tenure | Complaints | Churn |
| --: | -----: | -----: | ---------: | ----: |
|  22 |    30k |      1 |          4 |     1 |
|  45 |    80k |      8 |          0 |     0 |
|  31 |    45k |      3 |          2 |     1 |
|  50 |    90k |     10 |          0 |     0 |

A single decision tree might make mistakes.

Random Forest solves this using **parallel independent trees**:

```text
       Tree 1
       Tree 2
       Tree 3
       Tree 4
       ...
          ↓
      Voting
          ↓
     Prediction
```

Gradient Boosting instead builds trees **sequentially**:

```text
Tree 1
  ↓
Find mistakes
  ↓
Tree 2 focuses on mistakes
  ↓
Find remaining mistakes
  ↓
Tree 3 focuses on them
  ↓
...
```

XGBoost improves this process using:

* Gradient-based optimization
* Regularization
* Shrinkage / learning rate
* Subsampling
* Efficient tree construction
* Missing-value handling
* Parallel computation
* Early stopping
* Second-order derivatives

---

# 3. XGBoost Classification vs Regression

XGBoost can solve both.

### Regression

Output:

$$
\hat y \in \mathbb{R}
$$

Example:

```text
House price = $250,000
```

### Classification

Output is a class probability.

For binary classification:

$$
P(y=1|x)
$$

Example:

```text
P(churn) = 0.82
```

Then using a threshold:

```text
0.82 > 0.5
```

Prediction:

```text
Churn = 1
```

---

# 4. Types of Classification with XGBoost

XGBoost supports several classification scenarios.

## 4.1 Binary Classification

Two classes:

```text
0 / 1
```

Examples:

* Spam / Not Spam
* Disease / No Disease
* Fraud / Not Fraud
* Churn / No Churn

Typical objective:

```python
objective="binary:logistic"
```

---

# 5. Multiclass Classification

Suppose:

```text
Class 0 → Cat
Class 1 → Dog
Class 2 → Horse
```

XGBoost can output:

$$
P(Cat), P(Dog), P(Horse)
$$

For example:

```text
Cat   = 0.10
Dog   = 0.75
Horse = 0.15
```

Prediction:

```text
Dog
```

Typical objective:

```python
objective="multi:softprob"
```

with:

```python
num_class=3
```

---

# 6. Main Idea Behind XGBoost Classification

Let's understand the process conceptually before the mathematics.

Suppose we have:

```text
Actual:

0 1 1 0 1
```

Initially, XGBoost makes some predictions.

For example:

```text
Predicted probability:

0.5
0.5
0.5
0.5
0.5
```

Then we calculate how wrong these predictions are.

The next tree tries to learn the **direction and magnitude of correction needed**.

Then:

```text
Initial prediction
       +
Tree 1 correction
       +
Tree 2 correction
       +
Tree 3 correction
       +
...
       =
Final prediction
```

This is the fundamental idea behind boosting.

---

# 7. Why is it called "Gradient" Boosting?

Because XGBoost uses the **gradient of the loss function** to determine how the model should improve.

For classification, a common loss function is **log loss / binary cross-entropy**.

$$
L = -[y\log(p)+(1-y)\log(1-p)]
$$

where:

* \(y\) = actual class
* \(p\) = predicted probability

XGBoost calculates derivatives of this loss.

It uses:

### First derivative

$$
g_i = \frac{\partial L}{\partial \hat y_i}
$$

### Second derivative

$$
h_i = \frac{\partial^2 L}{\partial \hat y_i^2}
$$

This is one of the important differences between standard gradient boosting and XGBoost.

---

# 8. XGBoost Uses Second-Order Optimization

This is one of the most important concepts.

Traditional gradient boosting mainly uses the **first derivative**.

XGBoost uses:

$$
\boxed{\text{Gradient + Hessian}}
$$

That means:

```text
First derivative  → direction of improvement
Second derivative → curvature / how rapidly loss changes
```

This gives XGBoost more information when deciding how to construct the next tree.

---

# 9. Objective Function

XGBoost optimizes an objective function consisting of two parts:

$$
\boxed{
Obj = Loss + Regularization
}
$$

More formally:

$$
Obj =
\sum_{i=1}^{n} L(y_i,\hat y_i)
+
\sum_{k=1}^{K}\Omega(f_k)
$$

where:

* \(L\) = prediction loss
* \(\Omega\) = tree complexity penalty
* \(K\) = number of trees

This is extremely important.

XGBoost doesn't simply try to minimize prediction error.

It tries to minimize:

> **prediction error + model complexity**

---

# 10. Why Regularization?

Suppose a tree becomes extremely complicated:

```text
Tree
 ├── many branches
 │    ├── branches
 │    │    ├── branches
 │    │    └── ...
```

It might perfectly memorize training data.

Training accuracy:

```text
99.9%
```

But test accuracy:

```text
72%
```

That's overfitting.

XGBoost adds penalties for complicated trees.

So it tries to find:

```text
Good prediction
      +
Reasonable tree complexity
```

---

# 11. Tree Complexity in XGBoost

A simplified regularization term is:

$$
\Omega(f)
=
\gamma T
+
\frac{1}{2}\lambda \sum_{j=1}^{T}w_j^2
$$

where:

* \(T\) = number of leaves
* \(w_j\) = weight of leaf \(j\)
* \(\gamma\) = penalty for adding leaves
* \(\lambda\) = L2 regularization

XGBoost can also use L1 regularization through:

$$
\alpha
$$

So:

```text
gamma → controls tree complexity
lambda → L2 regularization
alpha  → L1 regularization
```

---
