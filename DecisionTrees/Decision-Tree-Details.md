

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
