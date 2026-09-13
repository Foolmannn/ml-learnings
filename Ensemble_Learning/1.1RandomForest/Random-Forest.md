# Random Forest in Detail

Random Forest is one of the most important **ensemble learning algorithms** in machine learning. Since you're studying ensemble methods after Decision Trees, Bagging, and Voting, Random Forest is the natural next step.

---

# 1. What is Random Forest?

**Random Forest = Multiple Decision Trees + Randomness + Aggregation**

Instead of building one Decision Tree, Random Forest builds **many different Decision Trees** and combines their predictions.

For classification:

> **Majority voting** among all trees

For regression:

> **Average** of predictions from all trees

### Simple example

Suppose we have 5 trees predicting whether a movie is good:

| Tree | Prediction |
|---|---|
| Tree 1 | Good |
| Tree 2 | Good |
| Tree 3 | Bad |
| Tree 4 | Good |
| Tree 5 | Bad |

Votes:

- Good → 3
- Bad → 2

Final Random Forest prediction:

**Good**

---

# 2. Why do we need Random Forest?

A single Decision Tree has a major problem:

### High variance

A Decision Tree can become very complex and memorize the training data.

For example:

```text
Training Accuracy = 100%
Testing Accuracy  = 78%
```

This is a typical sign of overfitting.

Random Forest solves this by creating many different trees and combining them.

The important idea is:

> **Individual trees may be noisy, but their combined prediction is usually much more stable.**

---

# 3. Random Forest is based on Bagging

Random Forest is closely related to **Bagging**.

You recently studied Bagging, so think of the relationship like this:

```text
Bagging
   |
   └── Multiple models
          |
          └── Bootstrap samples
                 |
                 └── Train models independently
                        |
                        └── Aggregate predictions
```

Random Forest adds another source of randomness:

```text
Random Forest
     |
     ├── Random rows
     │     └── Bootstrap sampling
     │
     └── Random features
           └── Feature subsampling
```

So the two important sources of randomness are:

1. **Random samples**
2. **Random features**

This is what makes Random Forest different from ordinary Bagging with Decision Trees.

---

# 4. How Random Forest works

Suppose we have a dataset:

```text
1000 samples
10 features
```

We want to create:

```text
100 Decision Trees
```

Random Forest roughly does the following.

### Step 1 — Create bootstrap samples

For every tree, Random Forest randomly samples training observations **with replacement**.

For example:

```text
Original dataset:

1 2 3 4 5 6 7 8 9 10
```

Tree 1 might receive:

```text
1 3 3 5 6 8 9 9 10
```

Tree 2:

```text
2 2 4 5 7 7 8 10 10
```

Tree 3:

```text
1 1 3 4 6 7 8 8 9
```

Each tree sees a different dataset.

---

# 5. Why sampling with replacement?

This is called **bootstrap sampling**.

Suppose we have:

```text
A B C D E
```

We randomly select 5 observations **with replacement**:

```text
A C C E B
```

Notice:

- C appears twice
- D doesn't appear

This creates a slightly different training dataset for each tree.

---

# 6. Step 2 — Randomly select features

This is the part that makes Random Forest particularly powerful.

Suppose we have:

```text
10 features
```

At a particular tree node, Random Forest might consider only:

```text
3 random features
```

For example:

```text
All features:

Age
Income
Education
Experience
Credit Score
Debt
Location
Marital Status
Employment
Loan Amount
```

A particular node may randomly consider:

```text
Income
Debt
Credit Score
```

The algorithm finds the best split **only among those selected features**.

At another node:

```text
Age
Education
Loan Amount
```

might be selected.

Therefore, different trees don't always use the same features.

---

# 7. Why random features?

This is extremely important.

Suppose one feature is extremely powerful:

```text
Income
```

A normal Bagging model may cause almost every tree to use:

```text
Income
```

near the root.

Then the trees become very similar.

For example:

```text
Tree 1 → Income → Debt → Age
Tree 2 → Income → Debt → Experience
Tree 3 → Income → Age → Debt
Tree 4 → Income → Debt → Age
```

These trees are highly correlated.

If all trees make the same mistake, averaging doesn't help much.

Random Forest forces trees to look at different subsets of features:

```text
Tree 1 → Income, Age, Debt
Tree 2 → Education, Location, Debt
Tree 3 → Experience, Age, Credit Score
Tree 4 → Employment, Loan Amount, Education
```

Now the trees are more diverse.

---

# 8. Diversity is extremely important

The power of an ensemble comes from:

> **Strong models + diversity**

Suppose we have 100 trees.

If all 100 trees are identical:

```text
Tree 1 → Wrong
Tree 2 → Wrong
Tree 3 → Wrong
...
Tree 100 → Wrong
```

Majority voting doesn't help.

But if errors are different:

```text
Tree 1 → Correct
Tree 2 → Correct
Tree 3 → Wrong
Tree 4 → Correct
...
```

the ensemble can produce a much better prediction.

So Random Forest tries to make trees:

### Strong enough

but also

### Different enough

---

# 9. Random Forest training process

Let's summarize the complete process.

Suppose:

```text
Dataset = D
Number of trees = 100
```

For each tree:

### Step 1

Create a bootstrap sample:

```text
D₁
```

### Step 2

Train a Decision Tree on `D₁`.

### Step 3

At each split:

```text
Randomly select a subset of features
```

### Step 4

Find the best split among those features.

### Step 5

Continue until the stopping criteria are reached.

Repeat:

```text
Tree 1
Tree 2
Tree 3
...
Tree 100
```

Finally aggregate their predictions.

---

# 10. Random Forest Classification

Suppose there are 5 trees:

```text
Tree 1 → Class A
Tree 2 → Class B
Tree 3 → Class A
Tree 4 → Class A
Tree 5 → Class B
```

Voting:

```text
Class A = 3
Class B = 2
```

Final:

```text
Class A
```

Mathematically:

$$
\hat{y} = \operatorname{mode}(T_1(x),T_2(x),...,T_B(x))
$$

where:

- $B$ = number of trees
- $T_i(x)$ = prediction from tree $i$

---

# 11. Random Forest Regression

For regression, we don't vote.

We average predictions.

Suppose:

```text
Tree 1 → 100
Tree 2 → 110
Tree 3 → 90
Tree 4 → 105
Tree 5 → 95
```

Final:

$$
\frac{100+110+90+105+95}{5}=100
$$

So:

$$
\hat{y} = \frac{1}{B}\sum_{i=1}^{B}T_i(x)
$$

---

# 12. Why does Random Forest reduce overfitting?

This is one of the most important theoretical questions.

A single Decision Tree has high variance.

Suppose:

```text
Tree 1 prediction = 80
Tree 2 prediction = 100
Tree 3 prediction = 120
```

There is significant variation.

When we average many trees, the variance decreases.

For independent models with variance:

$$
\sigma^2
$$

the variance of their average is approximately:

$$
\frac{\sigma^2}{B}
$$

where $B$ is the number of trees.

So increasing the number of trees can reduce variance.

However, Random Forest trees aren't completely independent.

They are correlated.

A more realistic expression is:

$$
Var(\bar{T}) =
\rho\sigma^2+
\frac{1-\rho}{B}\sigma^2
$$

where:

- $\rho$ = correlation between trees
- $B$ = number of trees
- $\sigma^2$ = variance of individual trees

This equation explains something extremely important:

### Increasing trees

reduces the second component.

### Reducing correlation between trees

reduces the first component.

And Random Forest uses **random feature selection** specifically to reduce correlation.

---

# 13. Random Forest vs Decision Tree

| Decision Tree | Random Forest |
|---|---|
| One tree | Many trees |
| High variance | Lower variance |
| More prone to overfitting | Less prone to overfitting |
| Fast training | More computationally expensive |
| Easy to visualize | Hard to visualize |
| Highly interpretable | Less interpretable |
| Can be unstable | More stable |
| Usually weaker generalization | Usually better generalization |

---

# 14. Random Forest vs Bagging

This distinction is very important.

### Bagging

```text
Bootstrap samples
       ↓
Decision Tree
Decision Tree
Decision Tree
Decision Tree
       ↓
Aggregation
```

All features may be considered when determining splits.

### Random Forest

```text
Bootstrap samples
       ↓
Random feature selection
       ↓
Decision Tree
Decision Tree
Decision Tree
Decision Tree
       ↓
Aggregation
```

So:

> **Random Forest = Bagging of Decision Trees + Random feature selection**

That's a useful exam/interview definition.

---

# 15. Random Forest vs Extra Trees

Another algorithm you'll eventually encounter is **Extra Trees (Extremely Randomized Trees)**.

The difference is roughly:

### Random Forest

Random:

- samples
- features

But split thresholds are optimized.

### Extra Trees

Random:

- samples/features depending on configuration
- split thresholds

Instead of searching for the best threshold among all possible thresholds, Extra Trees generates random thresholds and chooses the best among those.

This introduces even more randomness.

---

# 16. Important Random Forest hyperparameters

This is especially important because you've recently been working with `GridSearchCV`.

For:

```python
RandomForestClassifier()
```

some of the most important parameters are:

```python
n_estimators
criterion
max_depth
min_samples_split
min_samples_leaf
max_features
bootstrap
max_samples
class_weight
max_leaf_nodes
```

Let's understand them.

---

# 17. `n_estimators`

Number of trees in the forest.

```python
RandomForestClassifier(
    n_estimators=100
)
```

Example:

```text
n_estimators = 10
→ 10 trees

n_estimators = 500
→ 500 trees
```

Increasing it generally:

- improves stability
- reduces variance
- increases computation
- increases memory usage

Usually, increasing trees doesn't cause the same kind of overfitting problem as increasing tree depth.

For example:

```python
n_estimators=10
```

may be unstable.

```python
n_estimators=500
```

is generally much more stable.

But after some point, performance improvement becomes very small.

---

# 18. `max_depth`

Maximum depth of each tree.

```python
RandomForestClassifier(
    max_depth=10
)
```

Example:

```text
max_depth = 3
```

produces shallow trees.

```text
max_depth = None
```

allows trees to grow until other stopping conditions are reached.

### Small max_depth

- simpler trees
- less variance
- potentially more bias
- may underfit

### Large max_depth

- complex trees
- lower bias
- higher variance
- potentially more overfitting

Random Forest can tolerate fairly deep trees because averaging reduces variance.

---

# 19. `max_features`

This is one of the **most important Random Forest parameters**.

It determines how many features are randomly considered at each split.

For example:

```python
max_features=0.5
```

means approximately 50% of the features are considered at each split.

Suppose:

```text
20 features
```

and:

```python
max_features=0.5
```

then approximately:

```text
10 features
```

are randomly considered at a split.

---

### Common values

You can use:

```python
max_features="sqrt"
```

or

```python
max_features="log2"
```

or:

```python
max_features=0.5
```

or an integer:

```python
max_features=5
```

or:

```python
max_features=None
```

which means all features.

---

# 20. Why `max_features` matters

Suppose:

```text
max_features = all features
```

Trees become more similar.

Therefore:

```text
Correlation ↑
Diversity ↓
```

If:

```text
max_features = small
```

trees become more diverse.

Therefore:

```text
Correlation ↓
Diversity ↑
```

But if you make it **too small**, individual trees may become weak.

So there is a trade-off:

```text
More features
    ↓
Stronger individual trees
    ↓
More correlation


Fewer features
    ↓
More diversity
    ↓
Potentially weaker individual trees
```

---

# 21. `bootstrap`

Determines whether bootstrap samples are used.

```python
bootstrap=True
```

means sampling with replacement.

This is the traditional Random Forest approach.

```python
bootstrap=False
```

means bootstrap sampling isn't used.

This changes the way the training data for each tree is generated.

---

# 22. `max_samples`

This controls how many samples are drawn for each tree when:

```python
bootstrap=True
```

For example:

```python
max_samples=0.7
```

means each tree receives approximately:

```text
70% of training samples
```

with replacement.

Or:

```python
max_samples=500
```

means 500 samples.

This is similar to what you were tuning with Bagging.

---

# 23. `min_samples_split`

Minimum number of samples required to split an internal node.

Example:

```python
min_samples_split=10
```

A node must contain at least 10 samples before it can be split.

Small value:

```text
2
```

allows more splitting.

Larger value:

```text
10, 20, 50
```

produces simpler trees.

---

# 24. `min_samples_leaf`

Minimum number of samples allowed in a leaf.

Example:

```python
min_samples_leaf=5
```

means every leaf must have at least 5 samples.

This can help prevent trees from creating extremely specific rules.

For example, instead of:

```text
Leaf 1 → 1 sample
```

you could require:

```text
Leaf 1 → at least 5 samples
```

This generally makes the model smoother and can reduce overfitting.

---

# 25. `criterion`

The criterion determines how the tree evaluates splits.

For classification, common options include:

```python
criterion="gini"
```

and:

```python
criterion="entropy"
```

or, in current scikit-learn versions, `"log_loss"`.

### Gini impurity

$$
Gini = 1-\sum_{k=1}^{K}p_k^2
$$

where $p_k$ is the proportion of class $k$.

### Entropy

$$
H=-\sum_{k=1}^{K}p_k\log_2(p_k)
$$

The tree tries to find splits that reduce impurity.

For regression, criteria include measures based on squared error and related objectives.

---

# 26. `class_weight`

Useful for imbalanced classification.

Suppose:

```text
Class 0 → 950 samples
Class 1 → 50 samples
```

A model might become biased toward Class 0.

You can use:

```python
class_weight="balanced"
```

which gives more importance to the minority class.

---

# 27. Example with Scikit-Learn

Classification:

```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    max_features="sqrt",
    min_samples_split=2,
    min_samples_leaf=1,
    bootstrap=True,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
```

---

# 28. Evaluate the model

```python
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)
```

You can also use:

```python
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))
```

---

# 29. Random Forest Regression

For regression:

```python
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(
    n_estimators=200,
    max_depth=None,
    max_features=1.0,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
```

Then:

```python
from sklearn.metrics import mean_squared_error, r2_score

print("MSE:", mean_squared_error(y_test, y_pred))
print("R2:", r2_score(y_test, y_pred))
```

---

# 30. What is `random_state`?

Random Forest contains randomness:

- bootstrap sampling
- feature selection

Therefore, different runs can produce slightly different results.

```python
random_state=42
```

makes the random process reproducible.

For example:

```python
rf1 = RandomForestClassifier(random_state=42)
rf2 = RandomForestClassifier(random_state=42)
```

Both should produce the same results when everything else is identical.

The number `42` has no special mathematical meaning.

---

# 31. What is `n_jobs`?

Controls how many CPU cores are used.

```python
n_jobs=-1
```

means:

> Use all available CPU cores.

This is particularly useful because Random Forest trees can be trained independently.

For example:

```text
Tree 1 ── CPU 1
Tree 2 ── CPU 2
Tree 3 ── CPU 3
Tree 4 ── CPU 4
```

This makes Random Forest highly parallelizable.

---

# 32. Out-of-Bag (OOB) Evaluation

This is another important Random Forest concept.

Remember bootstrap sampling.

Suppose we have:

```text
1000 training samples
```

A bootstrap sample of 1000 observations is created **with replacement**.

Because some observations are selected multiple times, some observations aren't selected at all.

These are called:

> **Out-of-Bag samples**

On average, approximately:

$$
36.8\%
$$

of the original observations are not selected for a particular bootstrap sample.

Why?

Probability that one observation is not selected:

$$
\left(1-\frac{1}{n}\right)^n
$$

As $n$ becomes large:

$$
\lim_{n\to\infty}
\left(1-\frac{1}{n}\right)^n
=
e^{-1}
\approx0.368
$$

So roughly 36.8% of observations are OOB for each tree.

---

# 33. OOB score

Those OOB observations can be used to evaluate the tree.

In scikit-learn:

```python
rf = RandomForestClassifier(
    n_estimators=200,
    bootstrap=True,
    oob_score=True,
    random_state=42
)

rf.fit(X_train, y_train)

print(rf.oob_score_)
```

This gives an OOB estimate of generalization performance.

Conceptually:

```text
Bootstrap data
      ↓
Train tree
      ↓
OOB data
      ↓
Evaluate tree
```

Then combine OOB predictions across trees.

---

# 34. OOB vs Cross-Validation

You recently studied cross-validation, so this distinction is useful.

### Cross-validation

```text
Dataset
   ↓
Fold 1
Fold 2
Fold 3
Fold 4
Fold 5
```

Models are trained repeatedly using different train/validation splits.

### OOB

Random Forest naturally creates unused samples through bootstrap sampling.

Those unused samples can be used for evaluation.

Therefore:

> OOB evaluation can provide a convenient validation estimate without explicitly performing K-fold cross-validation.

However, cross-validation remains very useful for **hyperparameter tuning**.

---

# 35. Feature Importance

Random Forest can tell us which features are important.

```python
rf.feature_importances_
```

Example:

```python
import pandas as pd

importance = pd.Series(
    rf.feature_importances_,
    index=X_train.columns
)

print(importance.sort_values(ascending=False))
```

Output might look like:

```text
income          0.32
credit_score    0.25
debt            0.18
age             0.12
education       0.08
location        0.05
```

So:

```text
Income
```

is the most important feature according to this importance measure.

---

# 36. Important warning about feature importance

`feature_importances_` in tree models is typically based on **impurity decrease**.

It can be biased toward:

- continuous variables
- high-cardinality categorical variables

Therefore, for serious analysis, **permutation importance** or other methods such as SHAP can be more informative.

Permutation importance asks:

> "How much does model performance decrease when I randomly shuffle this feature?"

Example:

```python
from sklearn.inspection import permutation_importance

result = permutation_importance(
    rf,
    X_test,
    y_test,
    random_state=42
)
```

---

# 37. Does Random Forest require feature scaling?

Generally:

> **No.**

Random Forest uses decision-tree splits such as:

```text
income <= 50000
```

rather than distance or gradient calculations.

Therefore:

```text
StandardScaler
MinMaxScaler
RobustScaler
```

are generally unnecessary for Random Forest.

For example:

```python
Random Forest
    ↓
Age = 20–80
Income = 20,000–2,000,000
    ↓
No scaling required
```

This is different from algorithms such as:

- KNN
- SVM
- Logistic Regression with regularization
- Neural Networks

where scaling is often important.

---

# 38. Does Random Forest handle nonlinear relationships?

Yes.

This is one of its major strengths.

For example:

```text
Age
 ↓
    ┌── < 25 → Low risk
    │
    └── ≥ 25
          ↓
       Income
          ↓
       ...
```

Trees naturally create nonlinear decision boundaries.

Therefore Random Forest can learn complex relationships without manually creating polynomial features.

---

# 39. Does Random Forest capture feature interactions?

Yes.

For example:

```text
Income
+
Age
+
Credit Score
```

may interact in determining loan default.

A tree can naturally create rules such as:

```text
IF income < X
AND credit_score < Y
AND age < Z
THEN default
```

Random Forest combines many such rules.

---

# 40. Advantages of Random Forest

### 1. Excellent general-purpose algorithm

Works well on many tabular datasets.

### 2. Handles nonlinear relationships

No need to assume linearity.

### 3. Reduces overfitting compared with a single tree

Bagging + feature randomness reduces variance.

### 4. Little preprocessing

Usually no feature scaling required.

### 5. Handles feature interactions

Naturally captures complex interactions.

### 6. Works for classification and regression

```python
RandomForestClassifier
RandomForestRegressor
```

### 7. Robust

Often performs well even when the dataset isn't perfectly clean.

### 8. Parallelizable

Trees can be trained independently.

### 9. Provides feature importance

Useful for understanding which features contribute to predictions.

---

# 41. Disadvantages

### 1. Less interpretable

A single tree:

```text
IF age > 30
   ↓
IF income > 50K
   ↓
Class A
```

is easy to understand.

A forest with 500 trees is much harder to explain.

---

### 2. Computationally expensive

500 trees can require significantly more:

- CPU
- RAM
- training time

than one tree.

---

### 3. Large model size

A forest with hundreds of large trees can consume considerable memory.

---

### 4. Not always the best model

For some datasets:

```text
Gradient Boosting
XGBoost
LightGBM
CatBoost
```

may outperform Random Forest.

---

### 5. Poor extrapolation in regression

This is important.

Suppose the training target values are:

```text
100
200
300
400
```

A Random Forest generally predicts combinations/averages of values supported by the training data.

It doesn't naturally extrapolate to:

```text
1000
```

outside the training range the way some parametric models can.

---

# 42. Random Forest and Missing Values

A common misconception is:

> "Random Forest always handles missing values automatically."

This depends on the implementation and version.

In scikit-learn, support for missing values depends on the estimator/version and data constraints, so you should not blindly assume that arbitrary NaNs are accepted.

A safe general workflow is still:

```text
Missing values
      ↓
Imputation
      ↓
Random Forest
```

For example:

```python
from sklearn.impute import SimpleImputer
```

combined with a Pipeline.

---

# 43. Random Forest and categorical data

Standard scikit-learn Random Forest models generally expect numerical input.

Therefore categorical variables typically need encoding:

```text
Male/Female
      ↓
0/1
```

or:

```text
Kathmandu
Pokhara
Lalitpur
      ↓
One-hot encoding
```

A pipeline is often a clean solution.

---

# 44. Hyperparameter tuning with GridSearchCV

Since you were recently using `GridSearchCV`, Random Forest is a good example.

```python
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    random_state=42,
    n_jobs=-1
)

params = {
    "n_estimators": [100, 200, 500],
    "max_depth": [None, 10, 20, 30],
    "max_features": ["sqrt", "log2"],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}

grid = GridSearchCV(
    estimator=rf,
    param_grid=params,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

grid.fit(X_train, y_train)

print(grid.best_params_)
print(grid.best_score_)
```

---

# 45. How GridSearchCV sees this

Suppose:

```python
n_estimators = [100, 200]
max_depth = [10, 20]
```

Then combinations are:

```text
100, 10
100, 20
200, 10
200, 20
```

That's:

$$
2\times2=4
$$

combinations.

With:

```python
cv=5
```

each combination is evaluated 5 times:

$$
4\times5=20
$$

model fits.

This is why Random Forest + GridSearchCV can become computationally expensive.

---

# 46. RandomizedSearchCV

If the hyperparameter space is large, use:

```python
RandomizedSearchCV
```

instead of exhaustive GridSearchCV.

Example:

```python
from sklearn.model_selection import RandomizedSearchCV

search = RandomizedSearchCV(
    rf,
    param_distributions=params,
    n_iter=30,
    cv=5,
    scoring="accuracy",
    random_state=42,
    n_jobs=-1
)
```

Instead of trying every possible combination, it samples a fixed number.

---

# 47. Random Forest prediction intuition

Imagine you're asking:

> "Will this customer default?"

You have 300 trees.

```text
Tree 1 → No
Tree 2 → Yes
Tree 3 → No
Tree 4 → No
...
Tree 300 → No
```

Suppose:

```text
No  → 240 votes
Yes → 60 votes
```

Probability-like class output:

```text
No = 80%
Yes = 20%
```

In scikit-learn:

```python
rf.predict_proba(X_test)
```

can return class probabilities based on the forest's aggregated tree predictions.

---

# 48. Random Forest complete architecture

The entire algorithm can be visualized as:

```text
                     Original Dataset
                            │
             ┌──────────────┼──────────────┐
             ↓              ↓              ↓
       Bootstrap 1    Bootstrap 2    Bootstrap 3
             │              │              │
             ↓              ↓              ↓
       Random Features Random Features Random Features
             │              │              │
             ↓              ↓              ↓
          Tree 1          Tree 2          Tree 3
             │              │              │
             └──────────────┼──────────────┘
                            ↓
                       Aggregation
                            │
                    ┌───────┴───────┐
                    ↓               ↓
               Classification    Regression
                  Voting            Average
```

---

# 49. The most important intuition

You should remember these four concepts:

### Decision Tree

```text
High variance
```

### Bagging

```text
Many trees
+
Bootstrap samples
=
Lower variance
```

### Random Forest

```text
Many trees
+
Bootstrap samples
+
Random feature selection
=
Lower variance + lower tree correlation
```

### Boosting

Different philosophy:

```text
Trees are built sequentially
+
Each new tree focuses on previous errors
```

So Random Forest and Boosting should **not** be confused.

---

# 50. Random Forest vs Boosting

| Random Forest | Boosting |
|---|---|
| Trees trained independently | Trees trained sequentially |
| Bagging-based | Boosting-based |
| Reduces variance strongly | Can reduce bias strongly |
| Random samples/features | Later trees focus on previous errors |
| Easy to parallelize | Less parallelizable sequentially |
| Usually robust | Can be more sensitive |
| Great baseline for tabular data | Often excellent predictive performance |

Examples of boosting:

```text
AdaBoost
Gradient Boosting
XGBoost
LightGBM
CatBoost
```

---

# 51. When should you use Random Forest?

Random Forest is an excellent choice when:

- Your data is mostly **tabular**
- Relationships are nonlinear
- You have a mixture of useful features
- You don't want extensive feature engineering
- You need a strong baseline
- You want relatively robust performance
- Interpretability isn't as important as a single decision tree

A common practical workflow is:

```text
Dataset
   ↓
Train/Test Split
   ↓
Baseline model
   ↓
Decision Tree
   ↓
Random Forest
   ↓
Tune hyperparameters
   ↓
Compare with Boosting
   ↓
Final model
```

---

# 52. Random Forest cheat sheet

| Parameter | Purpose |
|---|---|
| `n_estimators` | Number of trees |
| `max_depth` | Maximum tree depth |
| `max_features` | Features considered at each split |
| `bootstrap` | Whether bootstrap samples are used |
| `max_samples` | Number/fraction of samples per tree |
| `min_samples_split` | Minimum samples required to split |
| `min_samples_leaf` | Minimum samples in a leaf |
| `criterion` | Split quality measure |
| `class_weight` | Handles class imbalance |
| `max_leaf_nodes` | Maximum number of leaves |
| `random_state` | Reproducibility |
| `n_jobs` | Parallel CPU usage |
| `oob_score` | Out-of-bag evaluation |

---

# 53. The key formula/concept to remember

The core Random Forest idea can be summarized as:

$$
\boxed{
\text{Random Forest}
=
\text{Bagging}
+
\text{Random Feature Selection}
}
$$

And its main objective is:

$$
\boxed{
\text{Reduce variance while maintaining strong individual trees}
}
$$

The **three most important things to understand deeply** are:

1. **Bootstrap sampling** → different trees see different training samples.
2. **Random feature selection** → different trees see different subsets of features.
3. **Aggregation** → combining many trees produces a more stable final prediction.

That gives you the conceptual bridge from **Decision Tree → Bagging → Random Forest → Boosting**, which is the most useful sequence to understand ensemble learning.