# K-Means Clustering in Machine Learning — Detailed Guide

K-Means is one of the most important **unsupervised machine learning algorithms**. It is mainly used to divide data into **K groups (clusters)** based on similarity.

Unlike supervised learning, we don't provide target labels such as `0` or `1`. The algorithm discovers groups from the structure of the data itself.

---

# 1. What is K-Means Clustering?

**K-Means clustering** is an unsupervised learning algorithm that partitions a dataset into **K clusters**, where each data point belongs to the cluster whose **centroid is closest** to it.

The word **K** means:

> The number of clusters we want to create.

The word **Means** comes from the fact that each cluster is represented by the **mean (average) of its data points**, called the **centroid**.

### Simple example

Suppose we have customers described by:

* Annual income
* Spending score

We might have data like:

```text
Customer       Income       Spending Score
A              20           80
B              22           75
C              25           85
D              80           20
E              85           25
F              90           15
```

We can ask K-Means:

```python
K = 2
```

It might discover:

```text
Cluster 1 → low income + high spending
Cluster 2 → high income + low spending
```

We didn't tell the algorithm what these groups mean.

That's the key idea behind unsupervised learning.

---

# 2. Why Do We Need Clustering?

Many datasets don't have labels.

For example, imagine a company has 10,000 customers:

```text
Customer
   ↓
Age
Income
Spending
Purchases
Visits
```

There may be no column saying:

```text
Customer Type = Premium
Customer Type = Regular
Customer Type = Inactive
```

We can use clustering to discover naturally occurring groups.

For example:

```text
                 Customers
                     |
          -----------------------
          |          |          |
       Cluster 1  Cluster 2  Cluster 3
          |          |          |
       Premium     Regular    Inactive
```

The names are assigned **after examining the clusters**.

---

# 3. Where is K-Means Used?

K-Means is commonly used for:

### Customer segmentation

Group customers according to:

* income
* spending
* purchase frequency
* age

### Image compression

Group similar colors together.

For example:

```text
Original image
     ↓
Millions of colors
     ↓
K-Means
     ↓
256 representative colors
     ↓
Compressed image
```

### Document clustering

Group documents according to their content.

For example:

```text
Documents
   |
   +--- Sports
   |
   +--- Politics
   |
   +--- Technology
   |
   +--- Entertainment
```

### Anomaly detection

Points that are very far from cluster centers may potentially be unusual.

### Recommendation systems

Group users/items with similar characteristics.

### Market segmentation

Identify groups of customers with similar behavior.

---

# 4. Supervised vs Unsupervised Learning

This distinction is very important.

## Supervised learning

We have:

```text
X → Features
y → Target
```

Example:

```text
Hours studied → Exam score
```

We know the target.

Algorithms include:

* Linear Regression
* Logistic Regression
* Decision Tree
* Random Forest
* XGBoost
* SVM

---

## Unsupervised learning

We only have:

```text
X → Features
```

There is no target.

Example:

```text
Age
Income
Spending
```

We want the algorithm to discover patterns.

Algorithms include:

* K-Means
* Hierarchical Clustering
* DBSCAN
* Gaussian Mixture Models
* PCA

---

# 5. Basic Idea of K-Means

Suppose we have points:

```text
      • •
    • •
              •
             • •
```

We want:

```text
K = 2
```

K-Means tries to find:

```text
Cluster 1          Cluster 2

  • •                •
• •                 • •
```

Each cluster gets a **centroid**.

A centroid is essentially the center of the cluster.

---

# 6. What is a Centroid?

Suppose we have three points:

```text
(2, 2)
(4, 4)
(6, 6)
```

The centroid is:

$$
\left(
\frac{2+4+6}{3},
\frac{2+4+6}{3}
\right)
$$

Therefore:

$$
(4,4)
$$

So the centroid represents the average location of the cluster.

For multiple features:

$$
C_j =
\left(
\frac{1}{n}\sum_{i=1}^{n}x_{i1},
\frac{1}{n}\sum_{i=1}^{n}x_{i2},
...
\frac{1}{n}\sum_{i=1}^{n}x_{id}
\right)
$$

---

# 7. How K-Means Works

K-Means follows an iterative process.

The basic algorithm is:

```text
1. Choose K
       ↓
2. Initialize K centroids
       ↓
3. Assign each point to nearest centroid
       ↓
4. Recalculate centroids
       ↓
5. Repeat assignment + recalculation
       ↓
6. Stop when centroids stop changing significantly
```

Let's understand each step.

---

# 8. Step 1 — Choose K

First, we decide how many clusters we want.

For example:

```python
K = 3
```

means:

```text
Create 3 clusters.
```

This is one of the biggest challenges of K-Means:

> How do we know what K should be?

We'll discuss this later using the **Elbow Method** and **Silhouette Score**.

---

# 9. Step 2 — Initialize Centroids

Suppose:

```text
K = 2
```

K-Means initially chooses two centroids.

For example:

```text
C1 = (2, 3)

C2 = (8, 7)
```

These are initial guesses.

The algorithm will improve them.

---

# 10. Step 3 — Assign Points to Nearest Centroid

For every data point, calculate its distance to each centroid.

Usually K-Means uses **Euclidean distance**.

For two points:

$$
A=(x_1,y_1)
$$

and

$$
B=(x_2,y_2)
$$

the Euclidean distance is:

$$
d(A,B)=
\sqrt{(x_1-x_2)^2+(y_1-y_2)^2}
$$

Example:

```text
Point = (3,4)

Centroid 1 = (2,3)
Centroid 2 = (8,8)
```

Distance to centroid 1:

$$
\sqrt{(3-2)^2+(4-3)^2}
$$

$$
=\sqrt{1+1}
$$

$$
=\sqrt2
$$

Distance to centroid 2:

$$
\sqrt{(3-8)^2+(4-8)^2}
$$

$$
=\sqrt{25+16}
$$

\approx 6.4
]

Therefore:

```text
Point → Cluster 1
```

because:

```text
distance to C1 < distance to C2
```

---

# 11. Step 4 — Recalculate the Centroids

After assigning all points to clusters, K-Means calculates a new centroid for each cluster.

Suppose Cluster 1 contains:

```text
(2,2)
(4,4)
(6,5)
```

The new centroid is:

$$
x = \frac{2+4+6}{3}=4
$$

$$
y = \frac{2+4+5}{3}=3.67
$$

Therefore:

```text
New centroid = (4, 3.67)
```

---

# 12. Step 5 — Repeat

Now we again:

```text
Calculate distances
        ↓
Assign points
        ↓
Calculate new centroids
        ↓
Calculate distances
        ↓
Assign points
        ↓
Calculate new centroids
```

This continues until convergence.

---

# 13. What Does Convergence Mean?

K-Means stops when the clusters don't change significantly anymore.

For example:

### Iteration 1

```text
C1 = (2.1, 3.2)
C2 = (7.5, 8.1)
```

### Iteration 2

```text
C1 = (3.1, 3.8)
C2 = (7.0, 7.4)
```

### Iteration 3

```text
C1 = (3.15, 3.82)
C2 = (7.01, 7.41)
```

### Iteration 4

```text
C1 = (3.15, 3.82)
C2 = (7.01, 7.41)
```

Centroids are no longer changing.

Therefore:

```text
CONVERGED
```

---
