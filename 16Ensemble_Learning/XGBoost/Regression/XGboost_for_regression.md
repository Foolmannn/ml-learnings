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

# 13. Why regularization matters

Imagine two trees.

### Tree A

```text
3 leaves
```

### Tree B

```text
50 leaves
```

Tree B might fit the training data better.

But it may generalize poorly.

XGBoost asks:

> Is the improvement in prediction error worth the additional complexity?

This is the fundamental idea behind its regularization.

---

# 14. Step 6 — Find the best split

XGBoost evaluates potential splits using a **gain** calculation.

For a node:

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

$$
G = \sum_i g_i
$$

and

$$
H = \sum_i h_i
$$

for the corresponding node.

The split is useful when it produces sufficient improvement.

---

# 15. Understanding split gain intuitively

Suppose:

```text
Parent
  |
  +------ Left child
  |
  +------ Right child
```

XGBoost asks:

> Does separating these observations into two groups improve the objective enough?

If yes:

```text
Split accepted
```

If not:

```text
Split rejected
```

The parameter `gamma` makes this stricter.

Higher `gamma` means:

> Only make a split if it provides enough improvement.

---

# 16. Step 7 — Calculate leaf weights

After constructing the tree structure, XGBoost determines the optimal value for each leaf.

The optimal leaf weight is:

$$
w_j^*
=
-\frac{G_j}{H_j+\lambda}
$$

This is a very important XGBoost equation.

For squared-error regression:

$$
h_i=1
$$

so:

$$
H_j=n_j
$$

where \(n_j\) is the number of samples in that leaf.

Thus the leaf weight becomes approximately:

$$
w_j^*
=
-\frac{G_j}{n_j+\lambda}
$$

---

# 17. What does a leaf weight mean?

Suppose a leaf contains observations for which the current model is systematically predicting too high.

Then:

$$
G_j > 0
$$

Therefore:

$$
w_j^* < 0
$$

The tree contributes a negative correction.

Conversely:

```text
Model predicting too low
        ↓
Positive correction
```

So a tree isn't necessarily predicting the final target directly.

Instead, it is providing a **correction to the current model**.

---

# 18. Step 8 — Learning rate

After calculating the new tree, XGBoost does not necessarily apply the entire tree contribution.

It uses a learning rate:

$$
\hat y_i^{(t)}
=
\hat y_i^{(t-1)}
+
\eta f_t(x_i)
$$

where:

* \(\eta\) = learning rate
* \(f_t(x_i)\) = new tree's prediction

Suppose:

```text
Current prediction = 100
Tree correction = 20
```

If:

$$
\eta=0.1
$$

then:

$$
100 + 0.1(20)=102
$$

Instead of immediately moving from 100 to 120, we move to 102.

---

# 19. Why use a learning rate?

A smaller learning rate generally means:

```text
Small corrections
+
More trees
```

A larger learning rate means:

```text
Large corrections
+
Fewer trees
```

Typical relationship:

```text
Learning rate ↓
        ↓
Number of trees ↑
```

and:

```text
Learning rate ↑
        ↓
Number of trees ↓
```

The two parameters therefore interact strongly:

* `learning_rate`
* `n_estimators`

---

# 20. Complete boosting process

Now we can combine everything.

### Iteration 0

Start with an initial prediction.

```text
Prediction
    ↓
Calculate loss
```

### Iteration 1

```text
Calculate gradients
       ↓
Calculate Hessians
       ↓
Find useful splits
       ↓
Build Tree 1
       ↓
Calculate leaf weights
       ↓
Regularize
       ↓
Apply learning rate
       ↓
Update predictions
```

### Iteration 2

Use the updated predictions:

```text
New gradients
       ↓
New Hessians
       ↓
Tree 2
       ↓
Update predictions
```

And so on.

```text
Tree 1
  ↓
Tree 2
  ↓
Tree 3
  ↓
Tree 4
  ↓
...
  ↓
Tree N
```

The final prediction is approximately:

$$
\hat y
=
f_0(x)
+
\eta f_1(x)
+
\eta f_2(x)
+\cdots+
\eta f_T(x)
$$

---

# 21. Simple numerical intuition

Suppose the actual value is:

$$
y=100
$$

Initial prediction:

$$
\hat y_0=70
$$

The model is underpredicting.

Tree 1 gives:

$$
f_1(x)=40
$$

With:

$$
\eta=0.1
$$

we get:

$$
\hat y_1
=
70+0.1(40)
$$

$$
=74
$$

Still not perfect.

Tree 2:

$$
f_2(x)=30
$$

Then:

$$
\hat y_2
=
74+0.1(30)
$$

$$
=77
$$

Tree 3:

$$
f_3(x)=25
$$

$$
\hat y_3=79.5
$$

And so on.

The model gradually approaches the target.

```text
70
 ↓
74
 ↓
77
 ↓
79.5
 ↓
...
 ↓
~100
```

This is the **additive nature of boosting**.

---

# 22. XGBoost vs Random Forest for Regression

This is an important distinction.

| Random Forest                             | XGBoost                                         |
| ----------------------------------------- | ----------------------------------------------- |
| Bagging                                   | Boosting                                        |
| Trees usually trained independently       | Trees trained sequentially                      |
| Each tree tries to solve original problem | Each tree corrects previous model               |
| Reduces variance                          | Reduces bias and variance                       |
| Random samples/features                   | Uses boosting + regularization/subsampling      |
| Trees can be deep                         | Usually shallow/moderate trees                  |
| Parallel tree construction is natural     | Sequential boosting iterations                  |
| Usually robust with little tuning         | Often requires more tuning                      |
| Less sensitive to hyperparameters         | Hyperparameters can strongly affect performance |

The fundamental difference:

### Random Forest

```text
Data
 ├── Tree 1
 ├── Tree 2
 ├── Tree 3
 ├── Tree 4
 └── Tree 5
      ↓
Average predictions
```

### XGBoost

```text
Data
 ↓
Tree 1
 ↓
Errors
 ↓
Tree 2
 ↓
Remaining errors
 ↓
Tree 3
 ↓
Remaining errors
 ↓
...
```

---

# 23. XGBoost vs ordinary Gradient Boosting

XGBoost is based on gradient boosting but adds several improvements.

| Gradient Boosting                 | XGBoost                       |
| --------------------------------- | ----------------------------- |
| Gradient-based                    | Gradient + Hessian            |
| Less regularization               | Strong regularization         |
| Usually slower                    | Highly optimized              |
| Fewer advanced controls           | Many regularization controls  |
| Basic implementation              | Optimized implementation      |
| Less efficient for large datasets | Designed for efficiency       |
| Basic tree construction           | Advanced tree construction    |
| Limited missing-value handling    | Native missing-value handling |

So:

> **XGBoost is not a completely different algorithm from Gradient Boosting. It is an optimized and regularized gradient-boosting framework.**

---

# 24. Important XGBoost Regression hyperparameters

When implementing XGBoost, these are the parameters you should understand.

---

## 24.1 `n_estimators`

Number of boosting rounds/trees.

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

Increasing it can improve learning, but too many trees can contribute to overfitting if not controlled.

---

# 25. `learning_rate`

Controls how strongly each tree contributes.

```python
learning_rate=0.1
```

Small:

```text
0.01
0.05
0.1
```

Large:

```text
0.2
0.3
```

Usually:

```text
small learning rate
+
more trees
```

is a useful combination to explore.

---

# 26. `max_depth`

Controls the maximum depth of each tree.

Example:

```python
max_depth=3
```

means trees can grow up to depth 3.

### Small depth

```text
Shallow trees
↓
Simpler model
↓
Less overfitting
```

### Large depth

```text
Complex trees
↓
More interactions
↓
Greater overfitting risk
```

---

# 27. `min_child_weight`

Controls the minimum amount of Hessian/instance weight required in a child.

Conceptually:

> How much evidence should be required before creating another child node?

Increasing it generally makes the tree more conservative.

---

# 28. `gamma`

Also called the minimum loss reduction.

It controls whether a split should happen.

Conceptually:

```text
Potential split
      ↓
Does it improve objective enough?
      ↓
YES → split
NO  → don't split
```

Higher `gamma`:

```text
More conservative trees
```

---

# 29. `subsample`

Controls the fraction of training rows used for each boosting iteration.

Example:

```python
subsample=0.8
```

means roughly:

```text
80% of training rows
```

are sampled for each tree.

This introduces randomness and can reduce overfitting.

---

# 30. `colsample_bytree`

Controls how many features are randomly sampled for each tree.

Example:

```python
colsample_bytree=0.8
```

means approximately 80% of features can be considered for a tree.

This is similar in spirit to feature sampling in Random Forest.

---

# 31. `reg_alpha`

L1 regularization.

$$
\alpha\sum |w_j|
$$

It can encourage some leaf weights toward zero.

Higher values make the model more regularized.

---

# 32. `reg_lambda`

L2 regularization.

$$
\lambda\sum w_j^2
$$

It penalizes large leaf weights.

Higher `reg_lambda` generally makes the model more conservative.

---

# 33. `objective`

For regression, XGBoost provides different objective functions.

A common one is:

```python
objective="reg:squarederror"
```

This corresponds to squared-error regression.

Other regression-related objectives include options designed for different target distributions or error behavior.

The objective is important because it determines what the model is actually optimizing.

---

# 34. `eval_metric`

This determines how model performance is evaluated.

Common regression metrics include:

### RMSE

$$
RMSE=
\sqrt{
\frac{1}{n}
\sum_{i=1}^{n}
(y_i-\hat y_i)^2
}
$$

### MAE

$$
MAE=
\frac{1}{n}
\sum_{i=1}^{n}
|y_i-\hat y_i|
$$

### \(R^2\)

$$
R^2=
1-
\frac{
\sum(y_i-\hat y_i)^2
}{
\sum(y_i-\bar y)^2
}
$$

---

# 35. Early stopping

One of the most useful features of XGBoost.

Suppose you train:

```text
1000 trees
```

But validation performance stops improving after:

```text
250 trees
```

Continuing to train may be unnecessary.

With early stopping:

```text
Tree 1      validation improves
Tree 2      improves
...
Tree 250    improves
Tree 251    no improvement
...
Tree N      no improvement
```

The algorithm can stop when the evaluation metric has failed to improve for a specified number of rounds.

Conceptually:

```text
Training performance
       ↓
keeps improving

Validation performance
       ↓
improves
       ↓
reaches best point
       ↓
starts worsening
       ↓
early stopping
```

This helps control overfitting and unnecessary computation.

---

# 36. Why XGBoost can overfit

XGBoost is powerful enough to model very complicated relationships.

But that power can cause overfitting.

For example:

```text
Training error → very low
Validation error → high
```

Possible causes:

* `max_depth` too high
* `n_estimators` too high
* `learning_rate`/tree count combination poorly chosen
* insufficient regularization
* insufficient data
* noisy features
* leakage
* weak validation strategy

---

# 37. How to control overfitting

Important parameters include:

```text
max_depth ↓
min_child_weight ↑
gamma ↑
subsample ↓
colsample_bytree ↓
reg_alpha ↑
reg_lambda ↑
```

And:

```text
learning_rate ↓
n_estimators ↑
+ early stopping
```

can also provide a more gradual fitting process.

But these should not be blindly changed. Validation should determine whether the changes actually help.

---

# 38. XGBoost handles nonlinear relationships

Suppose house price behaves like:

```text
Small area → low price
Medium area → moderate price
Large area → high price
```

But the relationship isn't perfectly linear.

Linear regression assumes:

$$
Price=\beta_0+\beta_1Area
$$

XGBoost doesn't require this form.

Trees can learn rules such as:

```text
IF area < 1000
    prediction = ...

ELSE IF area < 2000
    prediction = ...

ELSE
    prediction = ...
```

Multiple trees allow these simple rules to combine into a sophisticated nonlinear function.

---

# 39. Feature interactions

Another major strength is automatic learning of interactions.

Suppose:

```text
House price depends on:

Area
Location
Bedrooms
Age
```

Maybe:

> Area matters differently depending on location.

A tree can naturally learn:

```text
IF location = Kathmandu
    AND area > 1500
        ...
```

You don't necessarily have to manually create an interaction feature such as:

```text
area × location
```

The tree structure can model such interactions.

---

# 40. Missing values

XGBoost can handle missing feature values in its tree-building process.

For example:

```text
Area    Bedrooms    Age
1000       2         5
1500       3        NaN
2000       4         2
```

XGBoost can learn how missing values should be routed during tree construction.

This doesn't mean:

> "You never need to think about missing data."

Data quality and missingness should still be investigated carefully.

---

# 41. Feature scaling

Unlike many algorithms, tree-based XGBoost regression generally **does not require feature scaling**.

For example:

```text
Age      → 5
Income   → 100000
Area     → 1500
```

You generally don't need:

```python
StandardScaler()
```

just because the features have different scales.

Why?

Because trees make decisions based on thresholds:

```text
Income < 50000
Area < 1500
Age < 10
```

rather than relying on distance or coefficient magnitude in the same way as algorithms such as KNN or linear models.

---
