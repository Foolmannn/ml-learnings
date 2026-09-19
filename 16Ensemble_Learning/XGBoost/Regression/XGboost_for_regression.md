# XGBoost for Regression — In Detail

**XGBoost Regression** means using **Extreme Gradient Boosting** to predict a continuous numerical value such as:

* House price → `Rs. 12,500,000`
* Temperature → `24.7°C`
* AQI → `156.3`
* Salary → `Rs. 85,000`
* Sales → `125,430`
* Delivery time → `42.5 minutes`

XGBoost is one of the most powerful implementations of **gradient-boosted decision trees (GBDT)**.

---

# 1. Where XGBoost Regression fits

You have already studied:

**Decision Tree → Random Forest → Bagging → Boosting → AdaBoost → Gradient Boosting → XGBoost**

The important progression is:

```text
Decision Tree
     ↓
Multiple Trees
     ↓
Gradient Boosting
     ↓
XGBoost
```

The fundamental idea remains:

> Build trees sequentially, where each new tree tries to correct the errors made by the previous trees.

But XGBoost adds several important improvements to ordinary Gradient Boosting:

* Regularization
* Second-order optimization
* Shrinkage
* Column subsampling
* Row subsampling
* Efficient tree construction
* Missing-value handling
* Early stopping
* Parallelization
* Better computational efficiency

---

# 2. What is XGBoost Regression?

Suppose we want to predict house prices.

Our dataset is:

| Area | Bedrooms | Age | Price |
| ---: | -------: | --: | ----: |
| 1000 |        2 |  10 |    50 |
| 1500 |        3 |   5 |    80 |
| 2000 |        4 |   3 |   120 |
| 1200 |        2 |  15 |    55 |
| 2500 |        5 |   2 |   160 |

The goal is:

$$
X \rightarrow y
$$

where:

$$
X = \text{features}
$$

and

$$
y = \text{continuous target}
$$

XGBoost creates many decision trees and combines their predictions.

Conceptually:

```text
Tree 1
   ↓
Initial prediction

Tree 2
   ↓
Correct Tree 1 errors

Tree 3
   ↓
Correct remaining errors

Tree 4
   ↓
Correct remaining errors

...

Final Prediction
```

The final prediction is the **sum of contributions from many trees**.

---

# 3. Why does XGBoost use multiple trees?

Consider a simple regression problem.

Suppose the actual values are:

```text
Actual:
100
200
300
400
500
```

Initially, the model might predict:

```text
Prediction:
300
300
300
300
300
```

Errors:

```text
Actual       Prediction       Error
100             300            -200
200             300            -100
300             300               0
400             300            +100
500             300            +200
```

The first tree is not perfect.

XGBoost builds another tree specifically to improve these predictions.

After Tree 2:

```text
Prediction:
180
230
300
370
420
```

The errors become smaller.

Then Tree 3 attempts to improve them again.

So:

```text
Initial model
     ↓
Tree 1
     ↓
Better model
     ↓
Tree 2
     ↓
Better model
     ↓
Tree 3
     ↓
Better model
     ↓
...
```

This is the core idea behind boosting.

---

# 4. How XGBoost Regression works

The overall process can be understood in these steps:

```text
Training Data
      ↓
Initial Prediction
      ↓
Calculate Errors
      ↓
Calculate Gradient + Hessian
      ↓
Build Decision Tree
      ↓
Calculate Leaf Weights
      ↓
Regularize Tree
      ↓
Apply Learning Rate
      ↓
Update Prediction
      ↓
Repeat
      ↓
Final XGBoost Model
```

Let's understand each step.

---

# 5. Step 1 — Initial prediction

For regression using the standard squared-error objective, XGBoost starts with an initial prediction.

For example:

```text
Actual values:

10
20
30
40
50
```

The initial prediction could be based on the mean:

$$
\hat y = 30
$$

So initially:

```text
Actual    Prediction
10           30
20           30
30           30
40           30
50           30
```

This is essentially our starting model.

---

# 6. Step 2 — Calculate the loss

For regression, a common objective is **squared error**.

For one observation:

$$
L(y,\hat y)=\frac{1}{2}(y-\hat y)^2
$$

The factor \(1/2\) is convenient mathematically.

For the entire dataset:

$$
L = \sum_i \frac{1}{2}(y_i-\hat y_i)^2
$$

The model wants to minimize this loss.

---

# 7. Step 3 — Calculate gradients

This is where XGBoost becomes different from a simple residual-based explanation of Gradient Boosting.

XGBoost calculates the **first derivative** of the loss:

$$
g_i =
\frac{\partial L(y_i,\hat y_i)}
{\partial \hat y_i}
$$

For squared error:

$$
L_i=\frac{1}{2}(y_i-\hat y_i)^2
$$

Therefore:

$$
g_i = \hat y_i-y_i
$$

So:

```text
gradient = prediction - actual
```

For example:

```text
Actual       Prediction       Gradient
100             120              20
200             180             -20
300             290             -10
400             450              50
```

The gradient tells XGBoost the **direction of the error**.

---

# 8. Step 4 — Calculate Hessian

This is one of the major mathematical improvements in XGBoost.

XGBoost doesn't only use the first derivative.

It also uses the **second derivative**, called the Hessian:

$$
h_i =
\frac{\partial^2 L(y_i,\hat y_i)}
{\partial \hat y_i^2}
$$

For squared error:

$$
h_i=1
$$

So for each training example:

```text
Gradient    Hessian
   g            h
```

XGBoost uses both.

That's why XGBoost is often described as using:

> **Second-order gradient boosting**

---

# 9. Gradient vs Hessian

Think of optimization like driving down a mountain.

The gradient tells you:

> Which direction should I move?

The Hessian tells you:

> How does the slope change?

Therefore:

```text
Gradient → direction
Hessian  → curvature
```

Using both gives XGBoost more information about the loss surface.

---

# 10. Step 5 — Build a new decision tree

Now XGBoost uses the gradients and Hessians to determine how a new tree should split the data.

Suppose we have:

```text
Feature: Area

1000
1200
1500
1800
2200
2500
```

XGBoost considers possible splits:

```text
Area < 1200
Area < 1500
Area < 1800
Area < 2200
...
```

For each possible split, it calculates how much that split improves the objective.

The best split is selected.

---

# 11. The XGBoost objective function

This is one of the most important concepts.

At boosting iteration \(t\):

$$
Obj^{(t)}
=
\sum_i L(y_i,\hat y_i^{(t)})
+
\sum_k \Omega(f_k)
$$

There are two major components:

### Loss

$$
\sum_i L(y_i,\hat y_i)
$$

Measures prediction error.

### Regularization

$$
\sum_k\Omega(f_k)
$$

Controls model complexity.

Therefore:

```text
XGBoost Objective
       =
Prediction Error
       +
Model Complexity
```

This is extremely important.

Ordinary optimization might focus primarily on:

> "Make predictions as accurate as possible."

XGBoost says:

> "Make predictions accurate while keeping the trees sufficiently simple."

---

# 12. Regularization in XGBoost

A tree can become extremely complicated.

For example:

```text
                 Root
                /    \
              ...    ...
             /          \
          Leaf          Leaf
```

If we allow unlimited complexity, the model can memorize the training dataset.

This is **overfitting**.

XGBoost therefore penalizes complex trees.

The tree complexity term can be represented as:

$$
\Omega(f_t)
=
\gamma T
+
\frac{1}{2}\lambda\sum_{j=1}^{T}w_j^2
+
\alpha\sum_{j=1}^{T}|w_j|
$$

where:

* \(T\) = number of leaves
* \(w_j\) = weight of leaf \(j\)
* \(\gamma\) = penalty for adding leaves
* \(\lambda\) = L2 regularization
* \(\alpha\) = L1 regularization

So XGBoost can control:

```text
Number of leaves
       +
Magnitude of leaf weights
```

---
