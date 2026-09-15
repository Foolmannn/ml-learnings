# Gradient Boosting in Machine Learning — In Detail

**Gradient Boosting** is one of the most important **ensemble learning** algorithms in machine learning. It is especially powerful for **tabular/structured data** and is the foundation of algorithms such as **XGBoost, LightGBM, and CatBoost**.

Since you've already studied **Bagging, Random Forest, Voting, and AdaBoost**, the key thing to understand is that Gradient Boosting takes the idea of **sequentially adding weak learners** and uses **gradient descent** to decide what each new learner should fix.

---

# 1. What is Gradient Boosting?

Gradient Boosting is an ensemble technique that builds a model **sequentially**.

Instead of training many independent models like Bagging:

```text
Dataset
   │
   ├── Tree 1
   ├── Tree 2
   ├── Tree 3
   └── Tree 4
```

Gradient Boosting trains trees one after another:

```text
Dataset
   │
   ▼
Tree 1
   │
   ▼
Errors / Residuals
   │
   ▼
Tree 2
   │
   ▼
Remaining Errors
   │
   ▼
Tree 3
   │
   ▼
...
```

Each new tree tries to **correct the mistakes made by the previous ensemble**.

The final prediction is the combination of all the weak learners.

---

# 2. The Main Idea

Suppose we're predicting house prices.

Actual:

```text
100
200
300
400
```

Our first decision tree predicts:

```text
120
180
280
350
```

The errors are:

```text
Actual - Prediction

100 - 120 = -20
200 - 180 = 20
300 - 280 = 20
400 - 350 = 50
```

So:

```text
Residuals:
[-20, 20, 20, 50]
```

Gradient Boosting trains another tree to predict these residuals.

The second tree might predict:

```text
[-15, 15, 25, 40]
```

Now we add this correction to the original prediction.

```text
New prediction =
Old prediction + learning_rate × correction
```

So if:

```text
learning_rate = 0.1
```

then:

```text
120 + 0.1(-15) = 118.5
180 + 0.1(15)  = 181.5
280 + 0.1(25)  = 282.5
350 + 0.1(40)  = 354
```

The model is gradually improving.

---

# 3. Why is it called "Gradient" Boosting?

This is the most important theoretical point.

It isn't limited to simply predicting residuals.

Gradient Boosting minimizes a **loss function** using the idea of **gradient descent**.

Suppose our model is:

$$
F(x)
$$

and our loss function is:

$$
L(y,F(x))
$$

We want to find a model that minimizes:

$$
\sum_{i=1}^{n}L(y_i,F(x_i))
$$

Instead of changing all model parameters at once, Gradient Boosting builds a new function/tree that moves the predictions in the direction that **reduces the loss**.

The new model is approximately:

$$
F_m(x)=F_{m-1}(x)+\eta h_m(x)
$$

where:

* \(F_m(x)\) = new model
* \(F_{m-1}(x)\) = previous model
* \(h_m(x)\) = new weak learner
* \(\eta\) = learning rate

This is the core equation of Gradient Boosting.

---

# 4. What does the gradient mean?

Suppose our loss is Mean Squared Error:

$$
L(y,\hat y)=(y-\hat y)^2
$$

We want to know:

> In which direction should I change my prediction to reduce the loss?

We calculate the derivative of the loss with respect to the prediction:

$$
\frac{\partial L}{\partial \hat y}
$$

The negative gradient tells us the direction in which we should move.

Therefore, Gradient Boosting trains the next weak learner to approximate:

$$
-\frac{\partial L}{\partial F(x)}
$$

These are called **pseudo-residuals**.

---

# 5. Gradient Boosting for Regression

Let's understand the classic example.

Suppose:

|  X | Actual Y |
| -: | -------: |
|  1 |       10 |
|  2 |       20 |
|  3 |       30 |
|  4 |       40 |

### Step 1 — Initial prediction

For squared-error loss, the initial prediction is generally the mean:

$$
F_0(x)=\bar y
$$

Here:

$$
\bar y=25
$$

Therefore:

```text
Initial prediction = 25
```

For every observation:

```text
25
25
25
25
```

---

## Step 2 — Calculate residuals

$$
r_i=y_i-\hat y_i
$$

Therefore:

```text
10 - 25 = -15
20 - 25 = -5
30 - 25 =  5
40 - 25 = 15
```

So:

```text
[-15, -5, 5, 15]
```

---

## Step 3 — Train a tree on residuals

Now we train a small decision tree:

```text
X → residual
```

The tree learns:

```text
X <= 2.5 → negative correction
X > 2.5  → positive correction
```

For example:

```text
Tree prediction:

X=1 → -10
X=2 → -10
X=3 → +10
X=4 → +10
```

---

# 6. Add the correction

Suppose:

$$
\eta=0.1
$$

Then:

$$
F_1(x)=F_0(x)+0.1h_1(x)
$$

For X = 1:

$$
25+0.1(-10)=24
$$

For X = 4:

$$
25+0.1(10)=26
$$

The predictions become:

```text
24
24
26
26
```

They're still not perfect, so another tree is trained.

---

# 7. Second tree

The new residuals are calculated again.

```text
Actual       Prediction
10           24
20           24
30           26
40           26
```

Residual:

```text
-14
 -4
 +4
+14
```

The next tree learns these remaining errors.

Then:

```text
Tree 1
   ↓
Residuals
   ↓
Tree 2
   ↓
Residuals
   ↓
Tree 3
   ↓
...
```

This continues for many iterations.

---

# 8. The Complete Gradient Boosting Algorithm

The general process is:

### Step 1

Initialize the model:

$$
F_0(x)=\arg\min_\gamma\sum_iL(y_i,\gamma)
$$

For squared error, this is the mean.

---

### Step 2

Calculate pseudo-residuals:

$$
r_{im}
=
-\left[
\frac{\partial L(y_i,F(x_i))}
{\partial F(x_i)}
\right]_{F=F_{m-1}}
$$

---

### Step 3

Train a weak learner on:

```text
X → pseudo-residuals
```

Usually:

```text
Decision Tree
```

---

### Step 4

Find the optimal contribution of that tree.

Conceptually:

$$
\gamma_m
=
\arg\min_\gamma
\sum_i
L(y_i,F_{m-1}(x_i)+\gamma h_m(x_i))
$$

---

### Step 5

Update the model:

$$
F_m(x)=F_{m-1}(x)+\eta\gamma_mh_m(x)
$$

---

### Step 6

Repeat.

```text
F0
 ↓
Tree 1
 ↓
F1
 ↓
Tree 2
 ↓
F2
 ↓
Tree 3
 ↓
...
 ↓
Final model
```

---

# 9. Why Decision Trees?

Gradient Boosting usually uses **shallow decision trees**.

For example:

```text
max_depth = 3
```

A tree might look like:

```text
             Age < 30?
             /       \
           Yes        No
           /           \
     Income < 40K?   Income < 70K?
       /    \          /     \
     -5      2        4       10
```

These trees are called **weak learners** because individually they aren't very powerful.

But hundreds of small trees can collectively create a very powerful model.

---

# 10. Why shallow trees?

Suppose we use very deep trees.

They can learn:

```text
Training data extremely well
```

but may memorize noise.

That causes:

```text
Overfitting
```

Gradient Boosting therefore commonly uses:

```text
small trees
+
many iterations
+
small learning rate
```

For example:

```python
GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=3
)
```

---

# 11. Learning Rate

One of the most important hyperparameters is:

```python
learning_rate
```

It controls how much contribution each new tree makes.

The update is:

$$
F_m(x)=F_{m-1}(x)+\eta h_m(x)
$$

where \(\eta\) is the learning rate.

### Large learning rate

```text
learning_rate = 1.0
```

Each tree has a large effect.

Advantages:

* faster learning
* fewer trees required

Disadvantages:

* greater chance of overfitting
* less gradual learning

---

### Small learning rate

```text
learning_rate = 0.01
```

Each tree makes a small correction.

Advantages:

* usually better generalization
* more controlled learning

Disadvantages:

* requires more trees
* training takes longer

---

# 12. Learning Rate vs Number of Estimators

There is an important relationship:

```text
Lower learning rate
        ↓
Need more trees
```

For example:

```text
learning_rate = 0.1
n_estimators = 100
```

versus:

```text
learning_rate = 0.01
n_estimators = 1000
```

The second approach may produce better generalization, although there is no universal best combination.

A common practical strategy is:

```text
Choose relatively small learning_rate
+
increase n_estimators
+
use validation / early stopping
```

---

# 13. n_estimators

```python
n_estimators
```

means:

> Number of boosting stages / trees.

Example:

```python
n_estimators=100
```

means approximately:

```text
Tree 1
Tree 2
Tree 3
...
Tree 100
```

Increasing it generally increases model capacity.

Too few:

```text
Underfitting
```

Too many:

```text
Possible overfitting
```

---

# 14. max_depth

Controls the depth of each decision tree.

Example:

```python
max_depth=1
```

creates decision stumps.

```python
max_depth=3
```

creates moderately complex trees.

```python
max_depth=10
```

creates much more complex trees.

Generally:

```text
Smaller depth → simpler model → less overfitting
Larger depth  → complex model → more overfitting
```

---

# 15. Gradient Boosting Classification

Gradient Boosting can also perform classification.

For binary classification, the model predicts a score that can be converted into a probability.

For example:

```text
Tree ensemble score = 2.3
```

The sigmoid function converts it into probability:

$$
P(y=1)=\frac{1}{1+e^{-F(x)}}
$$

Then:

```text
P > 0.5 → class 1
P < 0.5 → class 0
```

The loss commonly used is **log loss / binary cross-entropy**.

---

# 16. Classification Example

Suppose we want to predict whether an email is spam.

Features:

```text
number of links
number of words
number of capital letters
contains "free"
sender reputation
```

First tree:

```text
Spam probability → 0.55
```

The model makes errors.

Second tree focuses on correcting those errors:

```text
Spam probability → 0.62
```

Third:

```text
Spam probability → 0.71
```

Eventually:

```text
Final probability → 0.94
```

Therefore:

```text
Spam
```

---

# 17. Gradient Boosting vs AdaBoost

You just studied AdaBoost, so this distinction is extremely important.

### AdaBoost

AdaBoost focuses on **misclassified observations**.

Conceptually:

```text
Wrong prediction
      ↓
Increase sample weight
      ↓
Next learner focuses more on it
```

### Gradient Boosting

Gradient Boosting focuses on the **negative gradient of the loss function**.

```text
Current prediction
       ↓
Calculate loss
       ↓
Calculate gradient
       ↓
Train next tree to reduce loss
```

So:

| AdaBoost                         | Gradient Boosting                 |
| -------------------------------- | --------------------------------- |
| Focuses on misclassified samples | Fits negative gradients/residuals |
| Changes sample weights           | Builds correction model           |
| Usually exponential loss         | Can use different loss functions  |
| Sequential                       | Sequential                        |
| Often uses weak trees            | Usually shallow trees             |
| Sensitive to noisy/outlier data  | Also can be sensitive to noise    |

---

# 18. Gradient Boosting vs Random Forest

This is another important comparison.

### Random Forest

Trees are generally trained **independently**.

```text
        Dataset
      /    |    \
   Tree1 Tree2 Tree3
      \    |    /
       Prediction
```

### Gradient Boosting

Trees are trained **sequentially**.

```text
Dataset
   ↓
Tree 1
   ↓
Errors
   ↓
Tree 2
   ↓
Errors
   ↓
Tree 3
```

| Random Forest                     | Gradient Boosting                                  |
| --------------------------------- | -------------------------------------------------- |
| Bagging                           | Boosting                                           |
| Parallel trees                    | Sequential trees                                   |
| Trees independent                 | Trees dependent                                    |
| Usually deep trees                | Usually shallow trees                              |
| Uses bootstrap samples            | Typically fits sequentially to residuals/gradients |
| Less sensitive to hyperparameters | More sensitive                                     |
| Easier to tune                    | More tuning required                               |
| Strong baseline                   | Often higher predictive performance                |

---

# 19. Gradient Boosting vs Bagging

### Bagging

Main goal:

> Reduce variance.

Example:

```text
Random Forest
```

Train multiple models independently and average them.

### Boosting

Main goal:

> Build a strong learner by sequentially reducing errors/bias.

Example:

```text
AdaBoost
Gradient Boosting
XGBoost
LightGBM
CatBoost
```

Simplified:

```text
Bagging:

Model 1 ─┐
Model 2 ─┤
Model 3 ─┼──> Average
Model 4 ─┤
Model 5 ─┘


Boosting:

Model 1
   ↓
Correction
   ↓
Model 2
   ↓
Correction
   ↓
Model 3
   ↓
Final model
```

---

# 20. Important Hyperparameters in sklearn

Let's look at the implementation.

```python
from sklearn.ensemble import GradientBoostingClassifier

gb = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    min_samples_split=2,
    min_samples_leaf=1,
    subsample=1.0,
    max_features=None,
    random_state=42
)
```

Now understand each one.

---

## `n_estimators`

Number of boosting stages.

```python
n_estimators=100
```

More trees → more capacity.

---

## `learning_rate`

Contribution of each tree.

```python
learning_rate=0.1
```

Smaller learning rate usually requires more trees.

---

## `max_depth`

Maximum depth of individual trees.

```python
max_depth=3
```

Controls tree complexity.

---

## `min_samples_split`

Minimum samples required to split an internal node.

```python
min_samples_split=2
```

Increasing it makes trees more conservative.

---

## `min_samples_leaf`

Minimum number of samples that must exist in a leaf.

```python
min_samples_leaf=1
```

Increasing it can reduce overfitting.

---

## `subsample`

Fraction of training samples used for each boosting stage.

```python
subsample=1.0
```

means:

```text
100% of samples
```

while:

```python
subsample=0.8
```

means approximately:

```text
80% of samples
```

Using less than 1.0 introduces randomness and is called **stochastic gradient boosting**.

---

## `max_features`

Number of features considered when finding splits.

Possible values include:

```python
None
"sqrt"
"log2"
```

or a fraction/integer.

This can reduce correlation and overfitting.

---

# 21. Regression Implementation

```python
from sklearn.ensemble import GradientBoostingRegressor

gb = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)

gb.fit(X_train, y_train)

y_pred = gb.predict(X_test)
```

Evaluation:

```python
from sklearn.metrics import mean_squared_error, r2_score

print("MSE:", mean_squared_error(y_test, y_pred))
print("R2:", r2_score(y_test, y_pred))
```

---

# 22. Classification Implementation

```python
from sklearn.ensemble import GradientBoostingClassifier

gb = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)

gb.fit(X_train, y_train)

y_pred = gb.predict(X_test)
```

Then:

```python
from sklearn.metrics import accuracy_score, classification_report

print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

For probabilities:

```python
y_prob = gb.predict_proba(X_test)[:, 1]
```

---

# 23. Does Gradient Boosting require feature scaling?

Usually **no**.

Decision trees split based on conditions such as:

```text
Age <= 30
Income <= 50000
```

Therefore the scale of the features isn't important.

You generally don't need:

```python
StandardScaler()
```

for ordinary tree-based Gradient Boosting.

For example:

```text
Age       = 25
Income    = 50000
Distance  = 2.4
```

can be used directly.

---

# 24. Why Gradient Boosting is powerful

Imagine a single shallow tree can only learn:

```text
Simple relationship
```

But 500 trees can progressively learn:

```text
Tree 1 → broad pattern
Tree 2 → correction
Tree 3 → another correction
...
Tree 500 → fine corrections
```

The combination can approximate very complicated functions.

Conceptually:

$$
F(x)=
f_1(x)+f_2(x)+f_3(x)+\cdots+f_M(x)
$$

where each \(f_i\) is a weak learner.

---

# 25. Bias-Variance Perspective

This connects directly with the **bias-variance tradeoff** you studied.

A shallow tree has:

```text
High bias
Low variance
```

Boosting combines many such weak learners.

As boosting proceeds:

```text
Bias ↓
```

and model complexity increases.

Eventually:

```text
Variance ↑
```

So:

```text
Too few trees
     ↓
High bias
     ↓
Underfitting

Good number of trees
     ↓
Good bias/variance balance

Too many / overly complex trees
     ↓
High variance
     ↓
Overfitting
```

---

# 26. Overfitting in Gradient Boosting

Common causes:

### 1. Very large `n_estimators`

```python
n_estimators=2000
```

can potentially overfit.

### 2. Large `learning_rate`

```python
learning_rate=1
```

can make learning aggressive.

### 3. Deep trees

```python
max_depth=10
```

can make individual learners too powerful.

### 4. Noisy data

Boosting can spend too much effort correcting noise.

---

# 27. How to reduce overfitting

You can:

```text
↓ learning_rate
↓ max_depth
↓ n_estimators (if excessive)
↑ min_samples_leaf
↓ subsample
```

And most importantly:

```text
Use cross-validation
```

or validation-based early stopping.

---

# 28. Early Stopping

Instead of blindly training:

```text
1000 trees
```

we can monitor validation performance.

For example:

```text
Tree 1    validation error = 0.50
Tree 50   validation error = 0.30
Tree 100  validation error = 0.22
Tree 200  validation error = 0.18
Tree 300  validation error = 0.17
Tree 400  validation error = 0.18
```

Performance starts getting worse after around 300.

So we stop around there.

This is **early stopping**.

In sklearn, modern boosting implementations can expose early-stopping-related parameters such as:

```python
n_iter_no_change
validation_fraction
tol
```

---

# 29. Feature Importance

Gradient Boosting can provide feature importance:

```python
gb.feature_importances_
```

For example:

```python
import pandas as pd

importance = pd.Series(
    gb.feature_importances_,
    index=X.columns
)

print(importance.sort_values(ascending=False))
```

You might get:

```text
income       0.35
age          0.25
experience   0.20
education    0.12
distance     0.08
```

Meaning the model relied heavily on:

```text
income
age
experience
```

However, remember that standard tree feature importance has limitations; **permutation importance** or **SHAP** can provide more informative explanations in many cases.

---

# 30. Gradient Boosting Loss Functions

This is one of the reasons Gradient Boosting is more general than simply "fit residuals."

For regression, sklearn's `GradientBoostingRegressor` supports losses such as:

```text
squared_error
absolute_error
huber
quantile
```

### Squared Error

Penalizes large errors strongly.

Good general-purpose choice.

### Absolute Error

More robust to outliers.

### Huber

Combines ideas from squared and absolute error.

```text
small errors → squared behavior
large errors → absolute-like behavior
```

### Quantile

Useful when predicting a particular conditional quantile rather than only the mean.

---

# 31. A very important distinction: residuals vs gradients

You'll often hear:

> "Gradient Boosting fits residuals."

That's **true for squared-error regression**, but it's an oversimplification.

For MSE:

$$
L=(y-F(x))^2
$$

The negative gradient is proportional to:

$$
y-F(x)
$$

which is exactly the residual.

Therefore:

```text
Squared-error regression
        ↓
Negative gradient ≈ residual
```

But for other loss functions:

```text
Negative gradient ≠ ordinary residual
```

That's why the more general description is:

> **Gradient Boosting fits the negative gradient of the loss function.**

This is a very important interview/exam concept.

---

# 32. Stochastic Gradient Boosting

Normally:

```text
subsample = 1.0
```

uses all training samples.

With:

```python
subsample=0.8
```

each tree is trained on a random subset.

Example:

```text
Training data = 10,000 samples

subsample=0.8

≈ 8,000 samples per boosting stage
```

Benefits can include:

* reduced variance
* less overfitting
* faster training

This introduces an element of randomness into boosting.

---

# 33. Advantages

### Excellent predictive performance

Gradient Boosting can produce very strong models.

### Works with nonlinear relationships

It can learn:

```text
X → Y
```

without assuming a linear relationship.

### Handles feature interactions

For example:

```text
Age + Income + Education
```

can interact naturally through tree splits.

### Little preprocessing

Usually no scaling is required.

### Flexible loss functions

Different losses can be optimized.

### Works for classification and regression

Both are supported.

---

# 34. Disadvantages

### Training is sequential

Unlike Random Forest, trees cannot simply all be trained independently.

Therefore:

```text
Training can be slower
```

### Sensitive to hyperparameters

Especially:

```text
learning_rate
n_estimators
max_depth
subsample
```

### Can overfit

Especially with:

```text
deep trees
large number of trees
large learning rate
```

### Sensitive to noisy data

Because the algorithm repeatedly focuses on what the previous ensemble hasn't fitted well.

### Less interpretable

A model containing hundreds of trees is harder to understand than one decision tree.

---

# 35. Gradient Boosting workflow

A practical workflow looks like:

```text
             Dataset
                ↓
        Train/Test Split
                ↓
        Preprocessing
                ↓
   Gradient Boosting Model
                ↓
       Tune Hyperparameters
                ↓
        Cross Validation
                ↓
       Evaluate on Test Set
                ↓
        Feature Importance
                ↓
          Final Model
```

---

# 36. Hyperparameter tuning

Since you've been using `GridSearchCV`, Gradient Boosting is a good place to apply it.

For example:

```python
params = {
    "n_estimators": [100, 200, 300],
    "learning_rate": [0.01, 0.05, 0.1],
    "max_depth": [2, 3, 4],
    "subsample": [0.8, 1.0]
}
```

Then:

```python
from sklearn.model_selection import GridSearchCV

grid = GridSearchCV(
    GradientBoostingClassifier(random_state=42),
    params,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

grid.fit(X_train, y_train)
```

Then:

```python
print(grid.best_params_)
print(grid.best_score_)
```

---

# 37. Important parameter interaction

Don't think about hyperparameters independently.

For example:

```text
learning_rate ↓
        +
n_estimators ↑
```

is a very common combination.

And:

```text
max_depth ↑
        +
n_estimators ↑
```

can make the model much more complex.

Think of model capacity approximately as being influenced by:

```text
Tree complexity
       ×
Number of trees
       ×
Learning rate
```

It's not a literal formula, but it's a useful intuition.

---

# 38. Gradient Boosting family

The basic sklearn Gradient Boosting algorithm led to several extremely important implementations.

```text
Gradient Boosting
       │
       ├── sklearn GradientBoosting
       │
       ├── XGBoost
       │
       ├── LightGBM
       │
       └── CatBoost
```

These aren't simply identical implementations with different names; they introduce different algorithmic and engineering improvements.

---

# 39. XGBoost

**XGBoost** = Extreme Gradient Boosting.

It became extremely popular in:

* Kaggle competitions
* tabular ML
* ranking
* classification
* regression

Important features include:

```text
regularization
parallelization
tree pruning
efficient optimization
handling of missing values
```

A typical model:

```python
from xgboost import XGBClassifier

model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=3
)
```

---

# 40. LightGBM

LightGBM is another highly optimized gradient boosting framework.

It is particularly useful for:

```text
large datasets
many features
fast training
```

It uses a tree-growing strategy that differs from traditional level-wise growth.

---

# 41. CatBoost

CatBoost is particularly well known for handling **categorical features** effectively.

It is often very convenient when your dataset contains many categorical variables.

```text
Gender
Country
Education
Occupation
Category
...
```

without requiring the same kind of manual encoding workflow you'd otherwise use.

---

# 42. The most important conceptual picture

If you remember only one diagram, remember this:

```text
                  Gradient Boosting

                       Dataset
                          │
                          ▼
                    Initial Model
                          │
                          ▼
                    Make Prediction
                          │
                          ▼
                    Calculate Loss
                          │
                          ▼
                Calculate Negative
                     Gradient
                          │
                          ▼
                  Train Small Tree
                          │
                          ▼
                  Add Tree × η
                          │
                          ▼
                 Updated Ensemble
                          │
                          ▼
                 Calculate New Loss
                          │
                          ▼
                  Train Next Tree
                          │
                         ...
                          │
                          ▼
                     Final Model
```

The central idea is:

> **Each new tree is added to reduce the loss left by the current ensemble.**

---

# 43. Gradient Boosting vs AdaBoost vs Random Forest

Since you've now covered all three, this comparison is worth memorizing:

| Property            | Random Forest             | AdaBoost                       | Gradient Boosting           |
| ------------------- | ------------------------- | ------------------------------ | --------------------------- |
| Ensemble type       | Bagging                   | Boosting                       | Boosting                    |
| Trees               | Independent               | Sequential                     | Sequential                  |
| Main idea           | Average many trees        | Focus on mistakes              | Follow loss gradient        |
| Training            | Parallel                  | Sequential                     | Sequential                  |
| Weak learners       | Usually trees             | Usually shallow trees          | Usually shallow trees       |
| Sample weighting    | Bootstrap/random sampling | Explicit sample weights        | Gradient-based correction   |
| Loss optimization   | Not directly              | Exponential loss traditionally | General differentiable loss |
| Overfitting         | Relatively resistant      | Can overfit                    | Can overfit                 |
| Tuning difficulty   | Easier                    | Moderate                       | Higher                      |
| Scaling required    | No                        | No                             | No                          |
| Typical performance | Strong                    | Strong                         | Very strong                 |

---

# 44. Exam/Interview definition

A good definition to memorize:

> **Gradient Boosting is a sequential ensemble learning technique that builds an additive model by training weak learners, typically shallow decision trees, to approximate the negative gradient of a loss function and progressively reduce the model's prediction error.**

And the most important equation:

$$
\boxed{
F_m(x)=F_{m-1}(x)+\eta h_m(x)
}
$$

where:

* \(F_{m-1}\) = existing ensemble
* \(h_m\) = new weak learner
* \(\eta\) = learning rate
* \(F_m\) = updated ensemble

---

# 45. What you should study next

Since you're going through ensemble learning systematically, I'd recommend this order:

```text
Ensemble Learning
       │
       ├── Voting
       │
       ├── Bagging
       │     └── Random Forest
       │
       └── Boosting
             │
             ├── AdaBoost       ← studied
             │
             ├── Gradient Boosting ← now
             │
             ├── XGBoost
             │
             ├── LightGBM
             │
             └── CatBoost
```

**The next important topic should be XGBoost in detail**, because it builds directly on the Gradient Boosting concepts you've just learned and introduces important ideas such as **regularization, second-order gradients, tree pruning, shrinkage, subsampling, and handling overfitting**.
