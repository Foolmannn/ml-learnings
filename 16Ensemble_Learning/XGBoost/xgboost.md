# XGBoost in Detail

**XGBoost** stands for **Extreme Gradient Boosting**. It is an optimized implementation of **gradient boosting with decision trees**, designed to be efficient, scalable, and strongly regularized. The original XGBoost paper introduced several engineering and algorithmic improvements such as sparsity-aware learning, approximate tree construction, and parallel/distributed computation. ([arXiv][1])

Since you're studying **ensemble learning → bagging → boosting → AdaBoost → Gradient Boosting**, the easiest way to understand XGBoost is:

> **XGBoost = Gradient Boosting + second-order optimization + explicit regularization + efficient tree construction + several practical improvements.**

---

# 1. Where XGBoost fits

You have already seen:

```text
Ensemble Learning
│
├── Bagging
│   └── Random Forest
│
└── Boosting
    │
    ├── AdaBoost
    │
    ├── Gradient Boosting
    │
    └── XGBoost
```

The important distinction is:

### Random Forest

Trees are generally built **independently/in parallel**.

```text
Dataset
  │
  ├── Tree 1
  ├── Tree 2
  ├── Tree 3
  ├── Tree 4
  └── Tree 5
       ↓
    Average/Vote
```

### Gradient Boosting

Trees are built **sequentially**.

```text
Tree 1
  ↓
Errors
  ↓
Tree 2
  ↓
New errors
  ↓
Tree 3
  ↓
...
```

### XGBoost

XGBoost follows the same sequential boosting idea but makes the optimization and implementation considerably more sophisticated.

---

# 2. Basic idea of XGBoost

Suppose we want to predict house prices.

Our dataset:

| House | Area | Bedrooms | Actual Price |
| ----- | ---: | -------: | -----------: |
| A     | 1000 |        2 |           50 |
| B     | 1500 |        3 |           75 |
| C     | 2000 |        4 |          110 |
| D     | 1200 |        2 |           60 |

Instead of trying to build one huge tree, XGBoost builds many small trees.

Initially:

```text
Prediction = initial prediction
```

Then:

```text
Tree 1
↓
Prediction improves

Tree 2
↓
Corrects remaining errors

Tree 3
↓
Corrects remaining errors

...
```

Final prediction:

$$
\hat y_i =
f_1(x_i)+f_2(x_i)+f_3(x_i)+\cdots+f_K(x_i)
$$

or

$$
\boxed{
\hat y_i=\sum_{k=1}^{K} f_k(x_i)
}
$$

where each \(f_k\) is a decision tree.

This **additive model** is fundamental to XGBoost. ([GitHub][2])

---

# 3. Why is it called Gradient Boosting?

Because each new tree is trained to move the model in the direction that reduces the loss.

Suppose:

$$
L(y,\hat y)
$$

is our loss.

For example, for regression:

$$
L = \frac12(y-\hat y)^2
$$

The gradient tells us how the loss changes with respect to prediction:

$$
g_i=
\frac{\partial L(y_i,\hat y_i)}
{\partial \hat y_i}
$$

The new tree tries to approximate the direction that reduces this loss.

That is the **gradient** part.

---

# 4. How XGBoost improves ordinary Gradient Boosting

Traditional Gradient Boosting generally uses the first derivative.

XGBoost uses:

### First derivative

$$
g_i =
\frac{\partial l(y_i,\hat y_i)}
{\partial \hat y_i}
$$

### Second derivative

$$
h_i =
\frac{\partial^2 l(y_i,\hat y_i)}
{\partial \hat y_i^2}
$$

So XGBoost uses both:

$$
\boxed{g_i = \text{gradient}}
$$

$$
\boxed{h_i = \text{Hessian}}
$$

This is one of the most important mathematical differences to understand.

---

# 5. XGBoost objective function

At boosting iteration \(t\), suppose we already have:

$$
\hat y_i^{(t-1)}
$$

We want to add a new tree \(f_t(x)\).

Therefore:

$$
\hat y_i^{(t)}
=
\hat y_i^{(t-1)}
+
f_t(x_i)
$$

The objective becomes:

$$
Obj^{(t)}
=
\sum_{i=1}^{n}
l(y_i,\hat y_i^{(t)})
+
\Omega(f_t)
$$

where:

* \(l\) = loss function
* \(\Omega\) = regularization term
* \(f_t\) = new tree

XGBoost explicitly incorporates regularization into its objective. ([GitHub][2])

---

# 6. Taylor expansion

This is where XGBoost becomes mathematically interesting.

We have:

$$
Obj^{(t)}
=
\sum_i
l(y_i,\hat y_i^{(t-1)}+f_t(x_i))
+
\Omega(f_t)
$$

Instead of directly optimizing the complicated loss, XGBoost uses a **second-order Taylor approximation**.

For a function \(l\):

$$
l(y_i,\hat y_i+f_t(x_i))
\approx
l(y_i,\hat y_i)
+
g_i f_t(x_i)
+
\frac12 h_i f_t(x_i)^2
$$

where:

$$
g_i =
\frac{\partial l(y_i,\hat y_i)}
{\partial\hat y_i}
$$

and

$$
h_i =
\frac{\partial^2 l(y_i,\hat y_i)}
{\partial\hat y_i^2}
$$

Ignoring constants that don't affect optimization:

$$
Obj^{(t)}
\approx
\sum_i
\left[
g_i f_t(x_i)
+
\frac12 h_i f_t(x_i)^2
\right]
+
\Omega(f_t)
$$

This is the core mathematical foundation of XGBoost.

---

# 7. Tree representation

A tree can be represented as:

$$
f(x)=w_{q(x)}
$$

where:

* \(q(x)\) tells us which leaf the sample goes to
* \(w_j\) is the score/value of leaf \(j\)

Suppose the tree has 3 leaves:

```text
             Area < 1500?
              /        \
            Yes         No
            /            \
      Bedroom < 3?       Leaf 3
        /    \
      Leaf1  Leaf2
```

Then:

$$
f(x)=w_{q(x)}
$$

For example:

```text
Leaf 1 → 2.5
Leaf 2 → 5.0
Leaf 3 → 8.0
```

---

# 8. XGBoost regularization

This is another major difference.

XGBoost doesn't just minimize prediction error.

It also penalizes complex trees.

The regularization term is commonly written:

$$
\Omega(f)
=
\gamma T
+
\frac12\lambda\sum_{j=1}^{T}w_j^2
$$

where:

* \(T\) = number of leaves
* \(w_j\) = leaf weight
* \(\gamma\) = penalty for adding leaves
* \(\lambda\) = L2 regularization

XGBoost also supports L1 regularization through `reg_alpha`. ([XGBoost Documentation][3])

---

# 9. Final simplified objective

After substituting the tree representation and grouping samples according to their leaves, the objective becomes:

$$
Obj^{(t)}
=
\sum_{j=1}^{T}
\left[
G_jw_j
+
\frac12(H_j+\lambda)w_j^2
\right]
+
\gamma T
$$

where:

$$
G_j=\sum_{i\in I_j}g_i
$$

and

$$
H_j=\sum_{i\in I_j}h_i
$$

Here:

* \(I_j\) = samples belonging to leaf \(j\)
* \(G_j\) = sum of gradients in leaf \(j\)
* \(H_j\) = sum of Hessians in leaf \(j\)

This equation is extremely important for understanding how XGBoost chooses leaf values and evaluates splits.

---

# 10. Finding the optimal leaf weight

We have:

$$
Obj_j =
G_jw_j+
\frac12(H_j+\lambda)w_j^2
$$

Differentiate with respect to \(w_j\):

$$
\frac{\partial Obj_j}{\partial w_j}
=
G_j+(H_j+\lambda)w_j
$$

Set equal to zero:

$$
G_j+(H_j+\lambda)w_j=0
$$

Therefore:

$$
\boxed{
w_j^*
=
-\frac{G_j}{H_j+\lambda}
}
$$

This is the **optimal leaf weight**.

Notice how \(\lambda\) affects it:

$$
w_j^*
=
-\frac{G_j}{H_j+\lambda}
$$

Increasing \(\lambda\) makes the denominator larger, shrinking the leaf weight.

That's how L2 regularization makes the model more conservative. ([XGBoost Documentation][3])

---

# 11. Optimal value of a leaf

Substituting the optimal \(w_j\) back into the objective gives:

$$
\boxed{
-\frac12
\frac{G_j^2}{H_j+\lambda}
}
$$

Therefore the score of a tree is:

$$
\boxed{
Obj =
-\frac12
\sum_{j=1}^{T}
\frac{G_j^2}{H_j+\lambda}
+
\gamma T
}
$$

This equation helps XGBoost decide whether a tree structure is useful.

---

# 12. How does XGBoost decide a split?

Suppose we have a leaf containing all samples.

We consider splitting it into:

```text
Parent
 /    \
Left  Right
```

Before splitting:

$$
G = G_L+G_R
$$

$$
H = H_L+H_R
$$

The gain from the split is:

$$
\boxed{
Gain =
\frac12
\left[
\frac{G_L^2}{H_L+\lambda}
+
\frac{G_R^2}{H_R+\lambda}
-
\frac{G^2}{H+\lambda}
\right]
-\gamma
}
$$

If this gain is positive, the split can improve the objective.

If the gain isn't large enough, the split isn't worthwhile.

This is closely connected to XGBoost's `gamma` parameter, which specifies the minimum loss reduction required to make a split. ([XGBoost Documentation][3])

---

# 13. Understanding `gamma`

Suppose:

```text
Potential split gain = 3
gamma = 5
```

Then:

$$
3-5=-2
$$

The split isn't worthwhile.

But:

```text
Gain = 10
gamma = 5
```

Then:

$$
10-5=5
$$

The split can be accepted.

So:

> **Higher `gamma` → harder to create new splits → simpler trees → potentially less overfitting.**

---

# 14. Simple intuition behind gradients

Suppose our actual values are:

```text
Actual:
100
200
300
```

Our current predictions are:

```text
Prediction:
80
220
250
```

For squared error:

$$
L=\frac12(y-\hat y)^2
$$

The derivative with respect to prediction is:

$$
g=\hat y-y
$$

Therefore:

```text
Sample 1:
80 - 100 = -20

Sample 2:
220 - 200 = +20

Sample 3:
250 - 300 = -50
```

The next tree learns patterns associated with these errors.

The negative gradients indicate:

> prediction should move upward.

Positive gradients indicate:

> prediction should move downward.

The next tree therefore learns a correction.

---

# 15. Why use the Hessian?

The gradient tells us:

> **Which direction should we move?**

The Hessian tells us:

> **How quickly is the loss changing?**

For squared error:

$$
L=\frac12(y-\hat y)^2
$$

First derivative:

$$
g=\hat y-y
$$

Second derivative:

$$
h=1
$$

For other losses, \(h\) changes depending on the prediction.

Using both allows XGBoost to make a more informed optimization step.

---

# 16. Learning rate

XGBoost doesn't necessarily add the entire tree prediction.

Instead:

$$
\hat y^{(t)}
=
\hat y^{(t-1)}
+
\eta f_t(x)
$$

where:

$$
\eta=\text{learning rate}
$$

In Python:

```python
learning_rate=0.1
```

Suppose:

```text
Current prediction = 50
Tree prediction = 20
learning_rate = 0.1
```

Then:

$$
50+0.1(20)
$$

$$
=52
$$

instead of:

$$
50+20=70
$$

So the learning rate controls how aggressively each tree changes the model.

---

# 17. Learning rate vs number of trees

These two parameters are strongly related:

```text
learning_rate ↓
        ↓
need more trees
```

For example:

```text
learning_rate = 0.3
n_estimators = 100
```

versus:

```text
learning_rate = 0.05
n_estimators = 500
```

The second approach makes smaller updates but more of them.

A common practical strategy is to use a relatively small learning rate and enough boosting rounds, often combined with early stopping. ([XGBoost][4])

---

# 18. Important XGBoost hyperparameters

This is probably the most important practical section.

## A. `n_estimators`

Number of boosting rounds/trees.

```python
XGBClassifier(
    n_estimators=100
)
```

Higher:

```text
more trees
↓
more model capacity
↓
potentially better training fit
↓
potential overfitting + longer training
```

---

# 19. `learning_rate`

Controls contribution of each tree.

```python
learning_rate=0.1
```

Lower:

```text
smaller updates
↓
usually need more trees
```

Higher:

```text
larger updates
↓
fewer trees may be needed
```

---

# 20. `max_depth`

Maximum depth of each tree.

```python
max_depth=6
```

Example:

```text
max_depth = 2

      Root
      /  \
     /    \
    A      B
```

versus:

```text
max_depth = 10

          Root
        /      \
       ...     ...
      many levels
```

Higher depth:

```text
more complex trees
↓
can capture complex relationships
↓
higher overfitting risk
```

Lower depth:

```text
simpler trees
↓
less variance
↓
may underfit
```

---

# 21. `min_child_weight`

Controls the minimum amount of Hessian/second-order information required in a child for a split.

Conceptually:

```text
small min_child_weight
        ↓
easier to create leaves
        ↓
more complex tree
```

```text
large min_child_weight
        ↓
harder to create leaves
        ↓
more conservative tree
```

It is an important regularization parameter.

---

# 22. `gamma`

Minimum loss reduction required for a split.

```python
gamma=0
```

means splitting is easier.

Increasing:

```python
gamma=5
```

makes splitting harder.

Therefore:

$$
\boxed{\gamma\uparrow \Rightarrow \text{simpler trees}}
$$

---

# 23. `subsample`

Fraction of training samples used for each tree.

Example:

```python
subsample=0.8
```

Each boosting round uses approximately 80% of the training observations.

This introduces randomness and can reduce overfitting.

```text
subsample = 1.0
    ↓
all samples

subsample = 0.8
    ↓
80% samples per tree
```

---

# 24. `colsample_bytree`

Fraction of features used for each tree.

Suppose:

```text
100 features
```

and:

```python
colsample_bytree=0.7
```

roughly 70% of features are considered for that tree.

This is similar in spirit to feature subsampling in Random Forest.

XGBoost provides multiple column-sampling parameters, including by-tree and by-node approaches. ([GitHub][5])

---

# 25. `reg_lambda`

L2 regularization.

```python
reg_lambda=1
```

Mathematically:

$$
\frac12\lambda\sum_jw_j^2
$$

Higher `reg_lambda`:

```text
stronger L2 regularization
↓
smaller leaf weights
↓
more conservative model
```

The official parameter documentation describes `lambda`/`reg_lambda` as the L2 regularization term on weights. ([XGBoost Documentation][3])

---

# 26. `reg_alpha`

L1 regularization.

```python
reg_alpha=0
```

Mathematically:

$$
\alpha\sum_j|w_j|
$$

L1 regularization can encourage some weights toward zero.

Useful especially when you want stronger sparsity/regularization.

XGBoost exposes it as `alpha` or `reg_alpha`. ([XGBoost Documentation][3])

---
