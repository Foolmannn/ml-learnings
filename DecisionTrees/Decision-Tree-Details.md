

# Decision Tree in Machine Learning — Detailed Notes

A **Decision Tree** is one of the most important supervised machine learning algorithms used for both **classification** and **regression**.

It learns a sequence of **if–else decision rules** from the training data and uses those rules to make predictions.

For example, suppose we want to decide whether a person will buy a laptop:

```text
                Age < 30?
               /        \
             Yes         No
             /            \
       Income > 50K?     No
        /      \
      Yes      No
      /         \
    Buy       Don't Buy
```

The tree is essentially learning:

```text
IF Age < 30
    IF Income > 50K
        → Buy
    ELSE
        → Don't Buy
ELSE
    → Don't Buy
```

---

# 1. What is a Decision Tree?

A **Decision Tree** is a tree-structured model where:

- **Root node** → starting decision
- **Internal node** → decision based on a feature
- **Branch** → outcome of that decision
- **Leaf node** → final prediction

Example:

```text
                         Outlook
                    /      |       \
                Sunny   Overcast    Rain
                 /         |          \
             Humidity    Play       Wind
             /    \                 /   \
           High   Normal          Weak  Strong
            |       |              |      |
           No      Yes            Yes     No
```

The model recursively divides the dataset into smaller subsets.

---

# 2. Why is it called a "Tree"?

Because its structure looks like a tree:

```text
                    Root
                     |
              ┌──────┴──────┐
              |             |
           Decision       Decision
              |             |
          ┌───┴───┐     ┌───┴───┐
          |       |     |       |
        Leaf    Leaf   Leaf    Leaf
```

Unlike a biological tree, the **root is at the top** and the leaves are at the bottom.

---

# 3. Decision Tree Terminology

You should understand these terms clearly.

## 3.1 Root Node

The first and most important decision.

```text
             Age < 30?
             /       \
           Yes        No
```

Here:

```text
Age < 30
```

is the root decision.

The root is generally selected because it provides the **best split** according to a splitting criterion.

---

## 3.2 Internal Node

A node where another decision is made.

```text
             Age < 30?
              /     \
            Yes      No
            |
       Income > 50K?
```

`Income > 50K` is an internal node.

---

## 3.3 Branch

A branch represents an outcome of a decision.

For example:

```text
Age < 30?
 /       \
Yes       No
```

There are two branches:

```text
Yes
No
```

---

## 3.4 Leaf Node

The final prediction.

```text
                Age < 30?
                /       \
              Yes        No
               |          |
              Buy      Don't Buy
```

`Buy` and `Don't Buy` are leaf nodes.

---

# 4. How Does a Decision Tree Learn?

This is the most important part.

Suppose we have:

| Age | Income | Student | Buy |
|---:|---:|---|---|
| 25 | 40K | Yes | Yes |
| 30 | 60K | No | No |
| 22 | 35K | Yes | Yes |
| 40 | 80K | No | No |
| 28 | 55K | Yes | Yes |

The tree needs to determine:

> **Which feature should I split on first?**

Possible features:

```text
Age
Income
Student
```

The algorithm evaluates possible splits and chooses the one that produces the **best separation of the target variable**.

This is where concepts such as:

- Entropy
- Information Gain
- Gini Impurity
- Gain Ratio
- Variance Reduction

come into play.

---

# 5. Classification vs Regression Trees

Decision Trees can solve two major types of problems.

## Classification Tree

Target is categorical.

Examples:

```text
Spam / Not Spam
Disease / No Disease
Buy / Don't Buy
Cat / Dog
```

The leaf contains a class prediction.

---

## Regression Tree

Target is numerical.

Examples:

```text
House price
Salary
Temperature
AQI
Sales
```

Example:

```text
              Area < 1500?
              /           \
            Yes            No
            /               \
        Price = 50L      Price = 90L
```

The leaf generally predicts an average/representative target value.

---

# 6. The Main Idea: Splitting

Decision Trees work by repeatedly splitting data.

Suppose:

```text
                Dataset
                   |
            Income > 50K?
              /        \
            No         Yes
           /             \
       Group A         Group B
```

Then each group can be split again:

```text
                Dataset
                   |
            Income > 50K?
              /        \
            No         Yes
           /             \
       Student?        Age < 30?
       /    \           /     \
     Yes    No        Yes      No
```

This process continues recursively.

---

# 7. What Makes a Good Split?

A good split should make the resulting groups as **pure** as possible.

Suppose we have:

```text
Group A:
Yes Yes Yes Yes
```

This is perfectly pure.

But:

```text
Group B:
Yes No Yes No
```

is impure.

A Decision Tree tries to find splits that reduce impurity.

---

# 8. Entropy

Entropy is commonly used to measure impurity.

For binary classification:

$$
H(S)=-p_+\log_2(p_+)-p_-\log_2(p_-)
$$

where:

- $p_+$ = probability of positive class
- $p_-$ = probability of negative class

---

## Example

Suppose:

```text
Yes = 4
No = 0
```

Then:

$$
p(Yes)=1
$$

$$
p(No)=0
$$

Therefore:

$$
H(S)=-(1\log_2 1 + 0\log_2 0)
$$

$$
H(S)=0
$$

So the dataset is perfectly pure.

---

# 9. Maximum Entropy

Suppose:

```text
Yes = 5
No = 5
```

Then:

$$
p(Yes)=0.5
$$

$$
p(No)=0.5
$$

Therefore:

$$
H(S)
=
-(0.5\log_2 0.5+0.5\log_2 0.5)
$$

$$
H(S)=1
$$

For binary classification:

$$
0 \leq H(S) \leq 1
$$

- `0` → completely pure
- `1` → maximum impurity

---

# 10. Information Gain

Information Gain tells us how much a split reduces entropy.

The formula is:

$$
IG(S,A)
=
H(S)
-
\sum_{v\in Values(A)}
\frac{|S_v|}{|S|}
H(S_v)
$$

In simpler terms:

$$
\boxed{
Information\ Gain
=
Parent\ Entropy
-
Weighted\ Child\ Entropy
}
$$

A higher Information Gain is generally better.

---

# 11. Example of Information Gain

Suppose our parent dataset contains:

```text
10 samples

Yes = 5
No = 5
```

Therefore:

$$
H(parent)=1
$$

Suppose a feature splits it into:

```text
Left:
Yes = 4
No = 1

Right:
Yes = 1
No = 4
```

Entropy of left:

$$
H(left)
=
-\frac45\log_2\frac45
-\frac15\log_2\frac15
$$

Approximately:

$$
H(left)=0.722
$$

Similarly:

$$
H(right)=0.722
$$

Weighted entropy:

$$
\frac{5}{10}(0.722)+
\frac{5}{10}(0.722)
=0.722
$$

Therefore:

$$
IG=1-0.722
$$

$$
\boxed{IG=0.278}
$$

The split provides some useful information.

---

# 12. Gini Impurity

Another very important splitting criterion is **Gini Impurity**.

It is widely used in Decision Tree implementations.

Formula:

$$
Gini=1-\sum_{i=1}^{K}p_i^2
$$

For binary classification:

$$
Gini=1-p_1^2-p_2^2
$$

---

## Example

Suppose:

```text
Yes = 5
No = 5
```

Then:

$$
p(Yes)=0.5
$$

$$
p(No)=0.5
$$

Therefore:

$$
Gini=1-(0.5)^2-(0.5)^2
$$

$$
Gini=1-0.25-0.25
$$

$$
\boxed{Gini=0.5}
$$

---

## Pure Node

Suppose:

```text
Yes = 10
No = 0
```

Then:

$$
Gini=1-(1)^2-(0)^2
$$

$$
\boxed{Gini=0}
$$

So:

```text
Gini = 0
```

means perfectly pure.

---

# 13. Entropy vs Gini

| Property | Entropy | Gini |
|---|---|---|
| Measures | Impurity | Impurity |
| Range for binary | 0–1 | 0–0.5 |
| Formula | $-\sum p_i\log_2p_i$ | $1-\sum p_i^2$ |
| Computational cost | Slightly higher | Lower |
| Common use | ID3/C4.5-style trees | CART |
| Pure node | 0 | 0 |

In practice, both often produce similar trees.

---

# 14. CART

**CART = Classification and Regression Trees**

It is one of the most important Decision Tree algorithms.

CART generally creates **binary splits**.

For example:

```text
Age < 30?
 /       \
Yes       No
```

Even if a categorical feature has multiple categories, CART generally represents decisions through binary partitions.

For classification, CART commonly uses:

$$
Gini\ Impurity
$$

For regression, it commonly uses a squared-error/variance reduction criterion.

---

# 15. ID3

ID3 stands for:

> **Iterative Dichotomiser 3**

It primarily uses:

$$
\boxed{Information\ Gain}
$$

to select attributes.

Basic process:

```text
Calculate entropy
       ↓
Calculate information gain
       ↓
Select highest gain
       ↓
Split dataset
       ↓
Repeat recursively
```

---

# 16. C4.5

C4.5 is an extension of ID3.

It introduced improvements such as:

- Gain Ratio
- Handling continuous attributes
- Handling missing values
- Pruning
- Better tree construction

---

# 17. Gain Ratio

Information Gain can sometimes favor features with many unique values.

For example:

```text
Student_ID
```

may have almost entirely unique values.

A tree could split based on:

```text
Student_ID = 101
Student_ID = 102
Student_ID = 103
...
```

This is usually useless for generalization.

C4.5 uses **Gain Ratio** to address this tendency.

$$
GainRatio =
\frac{Information\ Gain}{Split\ Information}
$$

---

# 18. Continuous Features

Decision Trees can work with numerical features.

Suppose:

```text
Age:

18
22
25
30
35
40
```

The algorithm can consider thresholds such as:

```text
Age < 20
Age < 23.5
Age < 27.5
Age < 32.5
Age < 37.5
```

For example:

```text
             Age < 27.5?
              /        \
            Yes         No
```

The algorithm searches for a threshold that gives the best split.

---

# 19. How Does It Choose a Threshold?

Suppose:

```text
Age = [20, 25, 30, 35]
```

Possible thresholds can be between consecutive values:

```text
22.5
27.5
32.5
```

The tree evaluates:

```text
Age < 22.5
Age < 27.5
Age < 32.5
```

and calculates the impurity reduction for each.

The best one is selected.

---

# 20. Recursive Partitioning

The overall Decision Tree algorithm can be understood as:

```text
                 Dataset
                    |
              Find best split
                    |
            ┌───────┴───────┐
            |               |
         Subset 1         Subset 2
            |               |
       Best split       Best split
            |               |
        ┌───┴───┐       ┌───┴───┐
        |       |       |       |
      Leaf    Leaf     Leaf    Leaf
```

This is called **recursive partitioning**.

---

# 21. Stopping Criteria

If we keep splitting indefinitely, the tree can become extremely large.

Therefore, we need stopping conditions.

Common conditions include:

### 1. Maximum depth

```python
max_depth=5
```

Stops tree growth after depth 5.

---

### 2. Minimum samples required for split

```python
min_samples_split=10
```

A node must have at least 10 samples before it can be split.

---

### 3. Minimum samples per leaf

```python
min_samples_leaf=5
```

Every leaf must contain at least 5 samples.

---

### 4. Maximum number of leaf nodes

```python
max_leaf_nodes=20
```

Limits the total number of leaves.

---

### 5. Minimum impurity decrease

```python
min_impurity_decrease=0.01
```

A split must provide enough impurity reduction.

---

# 22. Overfitting in Decision Trees

This is one of the biggest problems with Decision Trees.

Suppose the tree keeps splitting:

```text
                  Root
                    |
               Feature A
              /        \
             /          \
        Feature B      Feature C
        /    \          /    \
       ...   ...       ...   ...
```

Eventually it may memorize the training dataset.

For example:

```text
Training accuracy = 100%
Testing accuracy = 70%
```

This is a sign of **overfitting**.

---

# 23. Why Do Trees Overfit?

A Decision Tree has the ability to create very specific rules.

For example:

```text
Age < 31.5
AND
Income > 52,500
AND
Experience < 3.2
AND
...
```

The model can eventually create rules that describe individual training samples rather than general patterns.

---

# 24. Controlling Overfitting

The most important techniques are:

### Pre-pruning

Stop the tree from becoming too complex.

Important parameters:

```python
max_depth
min_samples_split
min_samples_leaf
max_leaf_nodes
min_impurity_decrease
```

---

### Post-pruning

First grow the tree and then remove unnecessary branches.

In scikit-learn, an important parameter is:

```python
ccp_alpha
```

This is related to **cost-complexity pruning**.

---

# 25. Cost-Complexity Pruning

A useful conceptual objective is:

$$
R_\alpha(T)=R(T)+\alpha |T|
$$

where:

- $R(T)$ = error/impurity of the tree
- $|T|$ = number of terminal nodes
- $\alpha$ = complexity penalty

The idea is:

```text
Model error
     +
Complexity penalty
```

A larger `ccp_alpha` generally encourages a smaller tree.

---

# 26. Decision Tree for Regression

Classification uses impurity such as Gini.

Regression works differently.

Suppose:

```text
X       y
1       10
2       12
3       11
4       30
5       32
6       31
```

A good split could be:

```text
X < 3.5
```

giving:

```text
Left:
10, 12, 11

Right:
30, 32, 31
```

The values within each group are relatively similar.

---

# 27. Regression Tree Prediction

Suppose a leaf contains:

```text
10
12
11
```

The predicted value is generally:

$$
\frac{10+12+11}{3}
$$

$$
=\boxed{11}
$$

So a regression tree often predicts the **mean target value of the samples in the leaf**.

---

# 28. Mean Squared Error in Regression Trees

A common criterion is:

$$
MSE =
\frac{1}{n}
\sum_{i=1}^{n}(y_i-\bar y)^2
$$

where:

$$
\bar y
$$

is the mean target value in that node.

The algorithm searches for splits that reduce the resulting squared error.

---

# 29. Important Difference: Linear Regression vs Decision Tree

Suppose:

```text
X → Y
```

Linear Regression assumes approximately:

$$
Y=\beta_0+\beta_1X
$$

So it creates a straight line.

Decision Tree instead creates regions:

```text
Y
│             ┌────────────
│             │
│      ┌──────┘
│      │
│──────┘
└──────────────────── X
```

The prediction is **piecewise constant**.

This makes Decision Trees capable of learning nonlinear relationships without explicitly adding polynomial features.

---

# 30. Decision Boundary

For classification, Decision Trees create decision boundaries.

For example:

```text
Feature 2
   ↑
   │       Class B
   │       │
   │       │
───┼───────┼────────
   │       │
   │ Class A
   │
   └────────────────→ Feature 1
```

The boundaries are typically **axis-aligned** in standard Decision Trees.

For example:

```text
Age < 30
```

rather than:

```text
Age + Income > 100
```

---

# 31. Feature Scaling

One major advantage of Decision Trees:

> **Feature scaling is generally not required.**

For example:

```text
Age: 20–60
Salary: 20,000–200,000
```

You generally don't need:

```python
StandardScaler()
```

before training a Decision Tree.

Why?

Because trees primarily compare values against thresholds.

Example:

```text
Age < 30
Salary < 50000
```

Scaling doesn't fundamentally change the ordering of values.

---

# 32. Handling Categorical Variables

Conceptually, Decision Trees can naturally work with categorical decisions.

However, **scikit-learn's standard `DecisionTreeClassifier` does not directly accept ordinary string categories**.

You typically encode categories first.

For example:

```text
Color
Red
Blue
Green
```

could be encoded using:

```text
One-hot encoding
```

or another appropriate encoding approach.

Be careful with arbitrary label encoding because assigning:

```text
Red = 0
Blue = 1
Green = 2
```

can introduce an artificial ordering.

---

# 33. Missing Values

Handling missing values depends on the implementation/version.

A common practical approach is to preprocess missing values:

```python
SimpleImputer
```

For example:

```python
from sklearn.impute import SimpleImputer
```

Then:

```text
Missing values
      ↓
Imputation
      ↓
Decision Tree
```

---

# 34. Decision Tree Algorithm — Complete Flow

The overall process is:

```text
                Training Data
                     ↓
             Calculate impurity
                     ↓
       Search possible feature splits
                     ↓
             Find best split
                     ↓
              Create node
                     ↓
             Split the data
                /       \
               /         \
          Left subset   Right subset
               ↓           ↓
          Repeat recursively
               ↓
          Stopping criterion?
             /       \
           No         Yes
           ↓           ↓
       Split again    Leaf
                       ↓
                  Prediction
```

---

# 35. Classification Algorithm

A simplified algorithm:

```text
function build_tree(data):

    if stopping_condition:
        return leaf

    for every feature:
        for every possible threshold:
            calculate split quality

    choose best split

    left_data = samples satisfying split
    right_data = remaining samples

    left_tree = build_tree(left_data)
    right_tree = build_tree(right_data)

    return node(left_tree, right_tree)
```

This is the core recursive idea.

---

# 36. Prediction Process

Suppose we have:

```text
             Age < 30?
             /       \
           Yes        No
           |
       Income > 50K?
        /       \
      Yes       No
      |          |
     Buy       Don't Buy
```

For:

```text
Age = 25
Income = 60K
```

Prediction:

```text
Age < 30 → Yes
        ↓
Income > 50K → Yes
        ↓
Buy
```

The prediction simply follows the path from root to leaf.

---

# 37. Advantages of Decision Trees

### 1. Easy to understand

Rules are human-readable.

```text
IF age < 30
AND income > 50K
THEN buy
```

---

### 2. Little preprocessing

Usually no feature scaling is required.

---

### 3. Handles nonlinear relationships

No need to explicitly create polynomial features.

---

### 4. Works for classification and regression

```text
DecisionTreeClassifier
DecisionTreeRegressor
```

---

### 5. Can model feature interactions

For example:

```text
Age
   ↓
Income
   ↓
Education
```

The effect of one feature can depend on another.

---

### 6. Interpretable

Individual decisions can be traced through the tree.

---

# 38. Disadvantages

### 1. Easily overfits

Unrestricted trees can memorize training data.

---

### 2. High variance

Small changes in training data can produce a very different tree.

---

### 3. Greedy algorithm

Most standard tree-building algorithms choose the best split **locally**, rather than searching all possible complete trees.

So the resulting tree isn't necessarily globally optimal.

---

### 4. Large trees become difficult to interpret

A tree with:

```text
depth = 20
```

can be extremely complicated.

---

### 5. Regression trees extrapolate poorly

Suppose training data contains:

```text
X = 1–100
```

and you ask the tree to predict:

```text
X = 150
```

A regression tree generally cannot extrapolate a new increasing trend like linear regression can. It predicts based on an existing leaf/region.

---

# 39. Important Hyperparameters in Scikit-Learn

For classification:

```python
from sklearn.tree import DecisionTreeClassifier
```

Example:

```python
model = DecisionTreeClassifier(
    criterion="gini",
    max_depth=5,
    min_samples_split=10,
    min_samples_leaf=5,
    max_leaf_nodes=20,
    random_state=42
)
```

---

# 40. `criterion`

Controls how split quality is measured.

Common options include:

```python
criterion="gini"
```

or:

```python
criterion="entropy"
```

and, depending on the scikit-learn version, `log_loss` is also available for classification.

---

# 41. `max_depth`

Maximum depth of the tree.

```python
max_depth=5
```

Small value:

```text
Less complex
↓
Less overfitting
↓
Potential underfitting
```

Large value:

```text
More complex
↓
Potential overfitting
```

---

# 42. `min_samples_split`

Minimum number of samples needed to split an internal node.

```python
min_samples_split=10
```

If a node has fewer than 10 samples, it won't be split.

Increasing this value generally makes the tree simpler.

---

# 43. `min_samples_leaf`

Minimum samples allowed in a leaf.

```python
min_samples_leaf=5
```

This prevents extremely small leaves.

Useful for reducing overfitting.

---

# 44. `max_leaf_nodes`

Limits the maximum number of leaf nodes.

```python
max_leaf_nodes=20
```

Useful when you want direct control over tree complexity.

---

# 45. `max_features`

Controls the number of features considered when searching for a split.

This becomes particularly important in tree ensembles such as:

```text
Random Forest
Extra Trees
```

It can also be used in individual Decision Trees.

---

# 46. `ccp_alpha`

Used for **Minimal Cost-Complexity Pruning**.

```python
ccp_alpha=0.01
```

Higher value:

```text
More pruning
→ smaller tree
→ potentially less overfitting
```

---

# 47. Complete Classification Example

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = DecisionTreeClassifier(
    criterion="gini",
    max_depth=5,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
```

---

# 48. Regression Example

```python
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = DecisionTreeRegressor(
    criterion="squared_error",
    max_depth=5,
    min_samples_leaf=5,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)

print("MSE:", mse)
```

---

# 49. Visualizing a Decision Tree

One of the best things about Decision Trees is that we can visualize them.

```python
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(15, 10))

plot_tree(
    model,
    feature_names=X.columns,
    filled=True
)

plt.show()
```

You can see:

```text
feature
threshold
impurity
samples
value
class
```

at each node.

---

# 50. Understanding a Tree Node

You may see something like:

```text
petal length <= 2.45
gini = 0.667
samples = 150
value = [50, 50, 50]
```

Meaning:

### `petal length <= 2.45`

The decision rule.

### `gini = 0.667`

Impurity of the current node.

### `samples = 150`

Number of training samples reaching that node.

### `value = [50, 50, 50]`

Number of samples belonging to each class.

---

# 51. Feature Importance

Decision Trees can calculate feature importance.

```python
model.feature_importances_
```

Example:

```text
Age       0.25
Income    0.45
Education 0.20
Gender    0.10
```

This means the tree used `Income` more heavily in its impurity-reducing splits.

However, **feature importance should not automatically be interpreted as causal importance**, and impurity-based importance can be biased toward certain types of features.

---

# 52. Probability Prediction

For classification:

```python
model.predict_proba(X_test)
```

Example:

```text
Class 0     Class 1
0.20        0.80
```

The model estimates class probabilities based on the training samples reaching the leaf.

---

# 53. Decision Tree vs KNN

Since you've been studying KNN, this comparison is useful.

| Feature | Decision Tree | KNN |
|---|---|---|
| Learning type | Supervised | Supervised |
| Training | Relatively fast | Very little model fitting |
| Prediction | Usually fast | Can be expensive |
| Scaling | Usually unnecessary | Important |
| Nonlinear patterns | Excellent | Excellent |
| Interpretability | High | Low |
| Overfitting | High risk | Depends on K |
| Handles feature interactions | Yes | Indirectly |
| Model structure | Tree | Stored training points |

---

# 54. Decision Tree vs Logistic Regression

| | Decision Tree | Logistic Regression |
|---|---|---|
| Decision boundary | Nonlinear / piecewise | Linear unless features transformed |
| Scaling | Usually unnecessary | Often useful |
| Interpretability | High | High |
| Overfitting | High risk | Generally lower |
| Nonlinear relationships | Naturally handles | Needs feature engineering |
| Probability output | Yes | Yes |
| Feature interactions | Naturally | Need explicit interaction features |

---

# 55. Decision Tree vs SVM

This is particularly useful because you're currently studying SVM.

| | Decision Tree | SVM |
|---|---|---|
| Scaling | Usually unnecessary | Usually important |
| Nonlinear data | Good | Excellent with kernels |
| Interpretability | High | Lower |
| Training complexity | Usually manageable | Can become expensive |
| Large datasets | Often good | Can be challenging depending on formulation |
| Outliers | Can be sensitive to data splits | Depends strongly on kernel/parameters |
| Feature engineering | Often less | Sometimes important |

---

# 56. Why Decision Trees Are the Foundation of Powerful Algorithms

Decision Trees become extremely powerful when combined into ensembles.

Important algorithms you'll study next are:

```text
Decision Tree
      ↓
Random Forest
      ↓
Gradient Boosting
      ↓
XGBoost
      ↓
LightGBM
      ↓
CatBoost
```

The basic Decision Tree is therefore extremely important for understanding modern tabular ML.

---

# 57. Random Forest Connection

Random Forest creates many Decision Trees.

```text
             Dataset
          /     |      \
        Tree   Tree    Tree
         |      |       |
        Pred   Pred    Pred
          \     |      /
           Majority Vote
                 ↓
            Final Class
```

Instead of relying on one tree, Random Forest combines many trees to reduce variance and improve generalization.

---

# 58. Boosting Connection

Boosting also uses trees, but differently.

Conceptually:

```text
Tree 1 → mistakes
           ↓
Tree 2 focuses more on mistakes
           ↓
Tree 3 focuses on remaining mistakes
           ↓
Tree 4 ...
           ↓
Final prediction
```

This leads to algorithms such as:

- Gradient Boosting
- XGBoost
- LightGBM
- CatBoost

---

# 59. A Very Important Concept: Greedy Splitting

Decision Tree construction is generally **greedy**.

At each node:

```text
Find best feature + threshold
             ↓
Make split
             ↓
Never go back and reconsider the entire tree
```

It doesn't normally search every possible tree structure because that would be computationally impractical.

This is one reason trees can be sensitive to small changes in data.

---

# 60. Bias-Variance Perspective

Decision Trees are typically capable of producing:

```text
Low bias
High variance
```

An unrestricted tree can fit the training data extremely well:

```text
Training error → very low
Test error     → potentially high
```

Pruning and constraints increase bias somewhat but reduce variance.

This is an important connection to:

```text
max_depth
min_samples_leaf
min_samples_split
ccp_alpha
```

---

# 61. Decision Tree Mental Model

You can remember the entire algorithm like this:

```text
                 DATA
                   ↓
          "What is the best question?"
                   ↓
           Feature + Threshold
                   ↓
              Split data
              /          \
             /            \
        Subset A        Subset B
           ↓                ↓
   "Best question?"   "Best question?"
           ↓                ↓
        Split              Split
           ↓                ↓
         Repeat           Repeat
              \            /
               \          /
                 LEAVES
                   ↓
               PREDICTION
```

---

# 62. What You Should Study in Order

Since you're learning ML systematically, I'd recommend studying Decision Trees in this sequence:

### Level 1 — Fundamentals

1. What is Decision Tree?
2. Classification vs Regression Trees
3. Root, node, branch, leaf
4. Recursive partitioning
5. How splitting works
6. Pure vs impure nodes

### Level 2 — Mathematical Foundation

7. Entropy
8. Information Gain
9. Gini Impurity
10. Gain Ratio
11. Variance reduction
12. MSE for regression trees

### Level 3 — Tree Construction

13. Finding the best feature
14. Finding the best threshold
15. Continuous variables
16. Categorical variables
17. Recursive splitting
18. Stopping criteria
19. Greedy splitting

### Level 4 — Overfitting

20. Why Decision Trees overfit
21. Pre-pruning
22. Post-pruning
23. Cost-complexity pruning
24. `ccp_alpha`
25. Bias-variance tradeoff

### Level 5 — Scikit-learn

26. `DecisionTreeClassifier`
27. `DecisionTreeRegressor`
28. `criterion`
29. `max_depth`
30. `min_samples_split`
31. `min_samples_leaf`
32. `max_leaf_nodes`
33. `max_features`
34. `min_impurity_decrease`
35. `ccp_alpha`
36. `random_state`

### Level 6 — Practical ML

37. Train/test split
38. Cross-validation
39. Hyperparameter tuning
40. GridSearchCV
41. RandomizedSearchCV
42. Evaluation metrics
43. Feature importance
44. Tree visualization
45. Handling missing/categorical data

### Level 7 — Advanced

46. CART
47. ID3
48. C4.5
49. CHAID
50. Random Forest
51. Extra Trees
52. Gradient Boosting
53. XGBoost
54. LightGBM
55. CatBoost

---

## ⭐ The most important concepts to master

If you're preparing for ML interviews/exams, don't just memorize the algorithm. Make sure you can **derive and explain**:

$$
\boxed{Entropy}
$$

$$
\boxed{Information\ Gain}
$$

$$
\boxed{Gini\ Impurity}
$$

$$
\boxed{Best\ Split}
$$

$$
\boxed{Recursive\ Partitioning}
$$

$$
\boxed{Overfitting}
$$

$$
\boxed{Pruning}
$$

$$
\boxed{Bias-Variance\ Tradeoff}
$$

and understand why:

> **A Decision Tree tries to recursively divide the feature space into increasingly pure regions, while controlling tree complexity to generalize well to unseen data.**

A particularly good next step is to work through **one complete Decision Tree example by hand**, calculating **Entropy → Information Gain → selecting the root → splitting → calculating the next split**, because that makes the mathematics much easier to understand.