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

# 12. How Does XGBoost Build a Tree?

Suppose we have:

```text
Age
 ↓
Income
 ↓
Tenure
 ↓
Complaints
```

XGBoost considers potential splits.

For example:

```text
Age < 30
```

or:

```text
Income < 50,000
```

or:

```text
Complaints < 2
```

It calculates how much each split improves the objective.

Then it chooses a split that provides sufficient gain.

---

# 13. Split Gain

A simplified XGBoost split gain is:

$$
Gain =
\frac{1}{2}
\left[
\frac{G_L^2}{H_L+\lambda}
+
\frac{G_R^2}{H_R+\lambda}
-
\frac{G^2}{H+\lambda}
\right]
-\gamma
$$

where:

* \(G_L\) = sum of gradients in left node
* \(G_R\) = sum of gradients in right node
* \(G\) = total gradient
* \(H_L\) = sum of Hessians in left node
* \(H_R\) = sum of Hessians in right node
* \(H\) = total Hessian
* \(\lambda\) = L2 regularization
* \(\gamma\) = split penalty

The algorithm searches for splits that produce sufficiently large gain.

---

# 14. Leaf Weight

XGBoost calculates an optimal weight for each leaf.

A simplified formula is:

$$
\boxed{
w^* = -\frac{G}{H+\lambda}
}
$$

where:

$$
G=\sum_i g_i
$$

and

$$
H=\sum_i h_i
$$

This tells the tree how much correction should be applied to observations falling into that leaf.

---

# 15. Binary Classification Mathematics

For binary classification, XGBoost commonly uses:

```text
binary:logistic
```

The model first produces a raw score:

$$
F(x)
$$

Then it passes that through the sigmoid function:

$$
p =
\frac{1}{1+e^{-F(x)}}
$$

Therefore:

$$
0 < p < 1
$$

and \(p\) represents the probability of class 1.

---

# 16. Example of Sigmoid

Suppose the model produces:

$$
F(x)=2
$$

Then:

$$
p=\frac{1}{1+e^{-2}}
$$

approximately:

$$
p=0.881
$$

So:

```text
Probability of class 1 = 88.1%
```

With threshold 0.5:

```text
0.881 > 0.5
```

Prediction:

```text
Class 1
```

---

# 17. Gradient for Binary Classification

For logistic loss, the gradient has a particularly convenient form:

$$
\boxed{g_i=p_i-y_i}
$$

where:

* \(p_i\) = predicted probability
* \(y_i\) = actual label

The Hessian is:

$$
\boxed{
h_i=p_i(1-p_i)
}
$$

These two quantities are used by XGBoost when constructing the next tree.

---

# 18. Simple Example

Suppose:

```text
Actual y = 1
Predicted probability = 0.3
```

Then:

$$
g = p-y
$$

$$
g=0.3-1=-0.7
$$

The negative gradient indicates that the model needs to increase its prediction.

Hessian:

$$
h=p(1-p)
$$

$$
h=0.3(0.7)=0.21
$$

So:

```text
Gradient = -0.7
Hessian = 0.21
```

The next tree uses this information to determine the correction.

---

# 19. Boosting Process

Suppose we have:

```text
Actual:
1 0 1 1 0
```

### Step 1 — Initial prediction

The model starts with an initial prediction.

For simplicity:

```text
p = 0.5
```

### Step 2 — Calculate gradients

For:

```text
y = 1
p = 0.5
```

$$
g=0.5-1=-0.5
$$

For:

```text
y = 0
p = 0.5
```

$$
g=0.5-0=0.5
$$

### Step 3 — Build Tree 1

Tree 1 learns patterns associated with these gradients.

### Step 4 — Update predictions

```text
Prediction =
Old prediction
+
learning rate × Tree output
```

### Step 5 — Calculate new gradients

The model now evaluates its new predictions.

### Step 6 — Build Tree 2

Tree 2 learns from the updated gradients.

And so on.

---

# 20. Learning Rate

One important parameter is:

```python
learning_rate
```

It controls how much each new tree contributes.

Suppose:

```python
learning_rate=0.1
```

Then:

$$
F_{new}(x)
=
F_{old}(x)+0.1f_{new}(x)
$$

Instead of allowing the new tree to make a huge correction, we shrink its contribution.

---

# 21. Learning Rate vs Number of Trees

These two parameters are strongly related:

```python
learning_rate
n_estimators
```

For example:

### Model A

```python
learning_rate=0.3
n_estimators=100
```

### Model B

```python
learning_rate=0.05
n_estimators=500
```

Model B takes smaller steps but makes more of them.

General idea:

```text
Higher learning_rate
        ↓
Fewer trees may be needed

Lower learning_rate
        ↓
More trees may be needed
```

But the best combination depends on the dataset.

---

# 22. `n_estimators`

This determines the number of boosting rounds / trees.

```python
n_estimators=100
```

means approximately:

```text
Tree 1
Tree 2
...
Tree 100
```

More trees can increase model capacity.

But too many trees without proper regularization or early stopping can lead to overfitting.

---

# 23. `max_depth`

Controls the maximum depth of each tree.

Example:

```python
max_depth=3
```

means each tree can have depth up to 3.

### Small depth

```text
Simple trees
↓
Less complexity
↓
Potentially less overfitting
```

### Large depth

```text
Complex trees
↓
Can learn complicated patterns
↓
Higher overfitting risk
```

---

# 24. `min_child_weight`

This controls the minimum amount of Hessian weight required in a child node.

Conceptually:

> Don't create a child node unless there is enough information to justify it.

Increasing it generally makes trees more conservative.

---

# 25. `gamma`

`gamma` specifies the minimum loss reduction required for making a split.

For example:

```python
gamma=0
```

allows splits whenever they provide positive gain.

Increasing:

```python
gamma=1
```

makes XGBoost more conservative about splitting.

So:

```text
Higher gamma
      ↓
Fewer splits
      ↓
Simpler trees
      ↓
Potentially less overfitting
```

---

# 26. `subsample`

XGBoost can train each tree using only a portion of the training data.

Example:

```python
subsample=0.8
```

means roughly 80% of the training samples are used for each boosting round.

This introduces randomness and can help reduce overfitting.

---

# 27. `colsample_bytree`

Similar idea, but for features.

```python
colsample_bytree=0.8
```

means each tree can use approximately 80% of the features.

This adds additional randomness.

---

# 28. Regularization Parameters

The major regularization parameters include:

### `reg_lambda`

L2 regularization:

$$
\lambda
$$

### `reg_alpha`

L1 regularization:

$$
\alpha
$$

### `gamma`

Controls whether additional splits are worthwhile.

Together:

```text
reg_alpha
reg_lambda
gamma
max_depth
min_child_weight
subsample
colsample_bytree
```

give XGBoost considerable control over model complexity.

---

# 29. Important XGBoost Classification Parameters

A useful classification parameter group is:

```python
XGBClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=3,
    min_child_weight=1,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0,
    reg_alpha=0,
    reg_lambda=1
)
```

These parameters control different aspects of the model.

| Parameter          | Purpose                      |
| ------------------ | ---------------------------- |
| `n_estimators`     | Number of boosting rounds    |
| `learning_rate`    | Contribution of each tree    |
| `max_depth`        | Maximum tree depth           |
| `min_child_weight` | Minimum child-node weight    |
| `subsample`        | Fraction of training samples |
| `colsample_bytree` | Fraction of features         |
| `gamma`            | Minimum split gain           |
| `reg_alpha`        | L1 regularization            |
| `reg_lambda`       | L2 regularization            |

---

# 30. XGBoost Objectives for Classification

## Binary classification

```python
objective="binary:logistic"
```

Outputs probabilities.

Example:

```text
0.12
0.87
0.43
0.91
```

---

## Binary classification with raw margins

There is also:

```python
objective="binary:logitraw"
```

This produces raw scores rather than probabilities.

---

## Multiclass classification

Two commonly encountered objectives are:

```python
objective="multi:softprob"
```

and

```python
objective="multi:softmax"
```

### `multi:softprob`

Returns probabilities for every class.

Example:

```text
[
 [0.1, 0.7, 0.2],
 [0.8, 0.1, 0.1]
]
```

### `multi:softmax`

Returns the predicted class directly:

```text
1
0
```

For analysis, `softprob` is often useful because you retain the probability information.

---

# 31. XGBoost vs Random Forest

This distinction is important.

### Random Forest

Trees are generally built independently:

```text
Tree 1 ─┐
Tree 2 ─┤
Tree 3 ─┤──→ Voting
Tree 4 ─┤
Tree 5 ─┘
```

### XGBoost

Trees are built sequentially:

```text
Tree 1
  ↓
Tree 2
  ↓
Tree 3
  ↓
Tree 4
  ↓
Final prediction
```

### Core difference

| Random Forest                   | XGBoost                                                              |
| ------------------------------- | -------------------------------------------------------------------- |
| Bagging                         | Boosting                                                             |
| Trees largely independent       | Trees sequential                                                     |
| Focuses on averaging            | Focuses on correcting errors                                         |
| Strong out-of-box model         | Strong but needs tuning                                              |
| Usually easier to tune          | More hyperparameters                                                 |
| Naturally parallel across trees | Sequential boosting dependency, but many operations are parallelized |

---

# 32. XGBoost vs AdaBoost

AdaBoost generally focuses on **reweighting incorrectly classified observations**.

XGBoost instead uses gradients and Hessians from the loss function.

Conceptually:

```text
AdaBoost
    ↓
Increase weight of misclassified samples
```

while:

```text
XGBoost
    ↓
Calculate gradients + Hessians
    ↓
Build tree to reduce objective
```

---

# 33. XGBoost vs Gradient Boosting

XGBoost is based on gradient boosting but adds several important improvements.

| Gradient Boosting                          | XGBoost                                                             |
| ------------------------------------------ | ------------------------------------------------------------------- |
| Gradient-based                             | Gradient + second-order information                                 |
| Regularization available                   | Strong regularization                                               |
| Can be slower                              | Highly optimized                                                    |
| Basic implementation                       | Optimized implementation                                            |
| Less extensive engineering                 | Parallelization, efficient algorithms, missing-value handling, etc. |
| Early stopping depending on implementation | Strong early-stopping workflow                                      |

So:

> **XGBoost is not a completely different boosting concept; it is a highly optimized and regularized gradient-boosted tree framework.**

---

# 34. Handling Missing Values

One useful property of XGBoost is its ability to handle missing values.

Suppose:

```text
Age = 25
Income = NaN
Tenure = 3
```

XGBoost can learn a default direction for missing values during tree construction.

This means you don't necessarily need to manually impute every missing value for XGBoost in the same way you would for algorithms that require complete numeric input.

However, preprocessing can still be appropriate depending on the dataset and pipeline.

---

# 35. Feature Scaling

One advantage of tree-based algorithms is that they generally don't require feature scaling.

For example:

```text
Age      = 25
Income   = 500000
Distance = 3.4
```

You generally don't need:

```python
StandardScaler()
```

just because the scales differ.

This is different from models such as:

* Logistic Regression
* SVM
* KNN
* Neural Networks

where scaling can be much more important.

---

# 36. Categorical Features

XGBoost historically worked primarily with numerical features, so categorical variables commonly required encoding.

For example:

```text
Gender
Male
Female
```

could be encoded.

Depending on the XGBoost version and API, native categorical feature support is also available when configured appropriately.

For a beginner pipeline, you should still understand:

```text
categorical data
      ↓
encoding / categorical handling
      ↓
XGBoost
```

---

# 37. Evaluation Metrics

For classification, you should not rely only on accuracy.

Important metrics include:

### Accuracy

$$
Accuracy =
\frac{TP+TN}{TP+TN+FP+FN}
$$

Useful when classes are reasonably balanced and error costs are similar.

---

### Precision

$$
Precision=
\frac{TP}{TP+FP}
$$

Question:

> Of the samples predicted positive, how many were actually positive?

---

### Recall

$$
Recall=
\frac{TP}{TP+FN}
$$

Question:

> Of all actual positive samples, how many did we detect?

---

### F1 Score

$$
F1 =
2\frac{Precision\times Recall}
{Precision+Recall}
$$

Balances precision and recall.

---

### ROC-AUC

Measures ranking ability across classification thresholds.

---

### Log Loss

Particularly relevant for probabilistic predictions:

$$
-\frac{1}{N}
\sum_i
[y_i\log(p_i)+(1-y_i)\log(1-p_i)]
$$

---

# 38. Confusion Matrix

For binary classification:

```text
                 Actual
               0       1
Predicted 0   TN      FN
Predicted 1   FP      TP
```

Example:

```text
                Actual
              No     Yes
Pred No       90      10
Pred Yes       5      95
```

Then:

```text
TN = 90
FN = 10
FP = 5
TP = 95
```

From this you can calculate:

* Accuracy
* Precision
* Recall
* F1
* Specificity

---

# 39. Probability Threshold

By default, binary classification often uses:

```text
p >= 0.5 → class 1
p < 0.5  → class 0
```

But 0.5 is not always the best threshold for a particular application.

For example:

```text
Probability = 0.42
```

could be classified as:

```text
0
```

using threshold 0.5.

But with:

```text
threshold = 0.30
```

it becomes:

```text
1
```

Changing the threshold changes the precision/recall trade-off.

This is especially important for imbalanced classification.

---

# 40. Class Imbalance

Suppose:

```text
Normal = 99,000
Fraud  = 1,000
```

A model predicting:

```text
Normal
```

for every sample gets:

$$
99\%
$$

accuracy.

But it detects:

```text
0% of fraud
```

So accuracy is misleading.

XGBoost provides:

```python
scale_pos_weight
```

which can be useful for binary class imbalance.

A common starting heuristic is:

$$
scale\_pos\_weight
=
\frac{\text{number of negative samples}}
{\text{number of positive samples}}
$$

But it should be treated as a starting point rather than a universally optimal value.

---

# 41. Early Stopping

One of the very useful XGBoost features is **early stopping**.

Suppose you specify:

```python
n_estimators=1000
```

You don't necessarily want all 1000 trees.

You can monitor validation performance.

For example:

```text
Tree 1      validation loss = 0.60
Tree 50     validation loss = 0.42
Tree 100    validation loss = 0.35
Tree 150    validation loss = 0.32
Tree 200    validation loss = 0.32
Tree 250    validation loss = 0.33
```

If performance stops improving for a specified number of rounds, training can stop.

For example:

```python
early_stopping_rounds=50
```

This helps prevent unnecessary boosting rounds and can reduce overfitting.

---

# 42. Feature Importance

XGBoost can provide feature importance.

Suppose:

```text
Feature             Importance
-----------------------------
Income                 0.35
Age                    0.25
Complaints             0.22
Tenure                 0.12
Gender                 0.06
```

This tells us which features contributed strongly to the model according to the selected importance measure.

But remember:

> Feature importance does not automatically mean causation.

A feature being important to the model does not prove that changing that feature causes the prediction to change.

---
