# Principal Component Analysis (PCA) in Machine Learning (with Scikit-learn)

Principal Component Analysis (PCA) is one of the most important **dimensionality reduction** techniques in Machine Learning.

It transforms a dataset with many features into a smaller set of new features while preserving as much information (variance) as possible.

---

# Why do we need PCA?

Suppose you have a dataset with **100 features**.

Problems:

* Training is slow.
* Some features are highly correlated.
* Visualization is impossible.
* Curse of dimensionality.
* Overfitting becomes more likely.

PCA solves these by creating fewer features called **Principal Components**.

Example:

Instead of

```
Height
Weight
Waist
Chest
Hip
BMI
...
```

PCA may create

```
PC1
PC2
PC3
```

These PCs capture most of the original information.

---

# Intuition

Imagine a dataset like this:

```
        *
      *
    *
  *
*
```

The points lie almost on a straight line.

There are two dimensions (x,y), but almost all information lies along one direction.

PCA finds that direction.

```
        *
      *
    *
  *
*
-------------
Principal Component 1
```

Instead of storing x and y separately, PCA stores only the position along this line.

Result:

```
2 dimensions
↓

1 dimension
```

Almost no information is lost.

---

# Goal of PCA

PCA finds

* Maximum variance direction
* Removes redundancy
* Makes features independent (orthogonal)

---

# What is Variance?

Variance tells us how spread out the data is.

Example

```
2 2 2 2 2
```

Variance = 0

No information.

Example

```
1 5 8 10 15
```

Large variance

More information.

PCA keeps the directions with **highest variance**.

---

# Principal Components

Suppose we have

```
Height
Weight
```

They are highly correlated.

```
Height ↑

170 → 60kg
175 → 65kg
180 → 70kg
185 → 75kg
```

Instead of two axes

```
Height
Weight
```

PCA rotates the axes.

```
      PC2
       |
      /
     /
----/---------
  PC1
```

PC1 captures most variation.

PC2 captures remaining variation.

---

# Properties of Principal Components

They are

* Linear combinations of original features
* Orthogonal (90° apart)
* Uncorrelated
* Ordered by explained variance

```
PC1 > PC2 > PC3 > PC4
```

---

# Mathematical Idea

Suppose

```
X =
[
Height
Weight
Age
]
```

PCA computes

```
PC1 = a₁Height + a₂Weight + a₃Age

PC2 = b₁Height + b₂Weight + b₃Age
```

The coefficients are chosen to maximize variance while keeping components orthogonal.

---

# PCA Step-by-Step

## Step 1: Standardize Data

Very important.

Example

```
Age = 20–60

Salary = 30,000–500,000
```

Without scaling

Salary dominates.

So use

```
StandardScaler
```

---

## Step 2: Compute Covariance Matrix

Covariance tells how features vary together.

Example

```
Height ↑
Weight ↑

Positive covariance
```

Example

```
Temperature ↑
Heating bill ↓

Negative covariance
```

Covariance matrix

```
        H    W    A

H      1   .8   .2
W      .8   1   .3
A      .2  .3    1
```

---

## Step 3: Compute Eigenvalues and Eigenvectors

This is the heart of PCA.

### Eigenvector

Direction of maximum variance.

### Eigenvalue

Amount of variance in that direction.

Example

```
Eigenvalue

10
5
1
0.1
```

Variance explained

```
10
5
1
0.1
```

Largest eigenvalue

↓

PC1

---

## Step 4: Sort Eigenvalues

Largest first

```
PC1

PC2

PC3

...
```

---

## Step 5: Select Top Components

Suppose

```
Feature count = 100
```

Need only

```
10 PCs
```

Now

```
100 → 10
```

---

## Step 6: Transform Data

Original

```
100 features
```

becomes

```
10 PCs
```

---

# Explained Variance

The most important PCA concept.

Suppose

```
Variance

PC1 = 60%

PC2 = 25%

PC3 = 10%

PC4 = 5%
```

Total

```
100%
```

If we keep

```
PC1 + PC2
```

We retain

```
85%
```

of information.

---

# Cumulative Explained Variance

```
PC1

60%

PC2

85%

PC3

95%

PC4

100%
```

Usually we keep enough PCs to explain **90–95%** of the variance.

---

# PCA Example

Original dataset

| Height | Weight |
| ------ | ------ |
| 170    | 60     |
| 172    | 63     |
| 175    | 65     |
| 180    | 70     |
| 185    | 75     |

These variables are correlated.

PCA rotates them into

| PC1 | PC2  |
| --- | ---- |
| 1.5 | 0.2  |
| 2.1 | -0.1 |
| 3.0 | 0.3  |
| 4.7 | -0.2 |
| 6.1 | 0.1  |

Most variation is captured by PC1.

---

# PCA in Scikit-learn

## Step 1: Import

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
```

---

## Step 2: Scale Data

```python
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)
```

---

## Step 3: Apply PCA

```python
pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_scaled)
```

Now

```
Original

100 features

↓

2 principal components
```

---

# Complete Example

```python
import pandas as pd

from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load dataset
wine = load_wine()

X = wine.data

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA
pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_scaled)

print(X_pca.shape)
```

Output

```python
(178, 2)
```

---

# Explained Variance Ratio

```python
pca.explained_variance_ratio_
```

Output

```python
array([0.36, 0.19])
```

Meaning

```
PC1

36%

PC2

19%
```

Total

```
55%
```

---

# Total Explained Variance

```python
pca.explained_variance_ratio_.sum()
```

Output

```
0.55
```

Meaning

```
55% variance retained
```

---

# Automatically Keep 95% Variance

Instead of specifying the number of components, let PCA decide:

```python
pca = PCA(n_components=0.95)

X_pca = pca.fit_transform(X_scaled)
```

PCA chooses the smallest number of components that preserve **95%** of the variance.

---

# Number of Components Chosen

```python
print(pca.n_components_)
```

Example output

```
9
```

Meaning

```
Original

13 features

↓

9 features
```

while preserving about **95%** of the variance.

---

# Eigenvalues

```python
pca.explained_variance_
```

Example

```
[4.73, 2.51, 1.45, ...]
```

These are the eigenvalues corresponding to each principal component.

---

# Principal Component Directions (Loadings)

```python
pca.components_
```

Example output:

```python
array([
    [ 0.14, 0.48, -0.21, ...],
    [-0.30, 0.05,  0.67, ...]
])
```

* Each **row** is a principal component.
* Each **column** corresponds to an original feature.
* Larger absolute values indicate that a feature contributes more to that component.

---

# PCA Pipeline (Recommended)

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=0.95)),
    ("classifier", LogisticRegression())
])

pipeline.fit(X_train, y_train)
```

Using a pipeline prevents data leakage by fitting the scaler and PCA only on the training data during model training.

---

# Choosing the Number of Components

A common approach is to inspect the cumulative explained variance:

```python
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

X_scaled = StandardScaler().fit_transform(X)

pca = PCA()
pca.fit(X_scaled)

cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
print(cumulative_variance)
```

Select the smallest number of components where the cumulative variance reaches your target (e.g., 90%, 95%, or 99%).

---

# Advantages

* Reduces dimensionality.
* Speeds up model training.
* Reduces overfitting by removing noisy or redundant features.
* Removes multicollinearity (principal components are uncorrelated).
* Makes visualization easier (2D or 3D projections).

---

# Disadvantages

* Reduced interpretability: principal components are combinations of original features.
* PCA is a **linear** technique; it may not capture nonlinear relationships.
* Sensitive to feature scaling.
* Some information is lost when components are discarded.

---

# When Should You Use PCA?

Use PCA when:

* You have many numerical features.
* Features are highly correlated.
* Training is slow due to high dimensionality.
* You want to visualize high-dimensional data.
* You need to reduce noise before modeling.

Avoid PCA when:

* Feature interpretability is critical (e.g., explaining why a prediction was made).
* Your dataset has only a few features.
* Important relationships are nonlinear (consider methods like Kernel PCA, t-SNE, or UMAP for visualization).

---

# Interview Questions

1. **Why is StandardScaler important before PCA?**
   PCA is variance-based. Features with larger scales dominate the principal components if data is not standardized.

2. **Can PCA be used on categorical features?**
   No. Standard PCA is designed for continuous numerical features.

3. **Does PCA use the target variable?**
   No. PCA is an **unsupervised** technique and ignores the labels.

4. **Does PCA always improve model accuracy?**
   Not necessarily. It can improve generalization by reducing noise, but it may also discard information useful for prediction.

5. **What is the difference between `fit()`, `transform()`, and `fit_transform()` in PCA?**

   * `fit(X)`: Learns the principal components from the data.
   * `transform(X)`: Projects data onto the learned principal components.
   * `fit_transform(X)`: Performs both steps in one call.

---

# Summary

| Concept                     | Description                                                                   |
| --------------------------- | ----------------------------------------------------------------------------- |
| PCA                         | Dimensionality reduction technique                                            |
| Input                       | Numerical features (preferably standardized)                                  |
| Output                      | Principal components (uncorrelated linear combinations)                       |
| Goal                        | Preserve maximum variance with fewer dimensions                               |
| Core mathematics            | Covariance matrix, eigenvalues, eigenvectors (or SVD internally)              |
| `explained_variance_ratio_` | Fraction of variance explained by each component                              |
| `components_`               | Directions (loadings) of the principal components                             |
| `n_components=0.95`         | Automatically keeps enough components to retain 95% of the variance           |
| Common preprocessing        | `StandardScaler`                                                              |
| Typical use cases           | Feature reduction, visualization, noise reduction, handling multicollinearity |

A key point to remember is that **scikit-learn computes PCA using Singular Value Decomposition (SVD) internally**, which is numerically more stable and efficient than explicitly computing the covariance matrix and its eigenvectors, even though the underlying concepts are mathematically equivalent.
