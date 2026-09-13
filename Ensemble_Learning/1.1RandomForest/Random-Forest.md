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
