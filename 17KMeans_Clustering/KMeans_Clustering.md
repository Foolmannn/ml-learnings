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

# 14. The Complete K-Means Algorithm

The algorithm can be summarized as:

```text
Input:
    Dataset X
    Number of clusters K

1. Initialize K centroids

2. Repeat:

   a. Assign every data point
      to the nearest centroid

   b. Calculate the mean of
      every cluster

   c. Move each centroid
      to the calculated mean

3. Stop when:
   - assignments don't change
   OR
   - centroid movement is very small
   OR
   - maximum iterations are reached
```

---

# 15. Mathematical Objective of K-Means

The main objective of K-Means is:

> Minimize the distance between each point and the centroid of its assigned cluster.

This is usually expressed using **Within-Cluster Sum of Squares (WCSS)**.

Also called:

* Inertia
* Sum of Squared Errors in some contexts

The objective function is:

$$
J =
\sum_{k=1}^{K}
\sum_{x_i \in C_k}
||x_i-\mu_k||^2
$$

where:

* \(K\) = number of clusters
* \(C_k\) = cluster \(k\)
* \(x_i\) = data point
* \(\mu_k\) = centroid of cluster \(k\)
* \(||x_i-\mu_k||^2\) = squared Euclidean distance

The goal is:

$$
\boxed{\min J}
$$

In simple terms:

> Make points within the same cluster as close to their centroid as possible.

---

# 16. Why Squared Distance?

Instead of simply using:

$$
d
$$

K-Means uses:

$$
d^2
$$

Squaring has useful mathematical properties and makes larger errors contribute more strongly to the objective.

For example:

```text
Distance = 2
Squared distance = 4
```

while:

```text
Distance = 10
Squared distance = 100
```

Therefore, points far away from their centroid contribute significantly to the objective.

---

# 17. K-Means Is an Optimization Algorithm

You can think of K-Means as trying to solve:

$$
\min_{\text{clusters},\text{centroids}}
\sum_{k=1}^{K}
\sum_{x_i\in C_k}
||x_i-\mu_k||^2
$$

It alternates between two optimization steps:

### Assignment step

Keep centroids fixed and find the best cluster for each point.

```text
Point → nearest centroid
```

### Update step

Keep assignments fixed and calculate the best centroid.

```text
Centroid = mean of cluster points
```

Then repeat.

This is sometimes called **Lloyd's algorithm**.

---

# 18. Why Does the Mean Become the Centroid?

Suppose a cluster contains:

```text
2
4
6
```

We want to find a value \(c\) that minimizes:

$$
(2-c)^2+(4-c)^2+(6-c)^2
$$

The value that minimizes the squared errors is the **mean**:

$$
c=\frac{2+4+6}{3}=4
$$

That's why the algorithm is called:

> K-Means

The centroid is the mean of the points assigned to the cluster.

---

# 19. Example of K-Means

Let's take one-dimensional data:

```text
1
2
3
10
11
12
```

Suppose:

$$
K=2
$$

Initially:

```text
C1 = 2
C2 = 10
```

Assignments:

```text
1 → C1
2 → C1
3 → C1

10 → C2
11 → C2
12 → C2
```

New centroids:

$$
C_1=\frac{1+2+3}{3}=2
$$

$$
C_2=\frac{10+11+12}{3}=11
$$

Assignments remain the same.

Therefore:

```text
Cluster 1 = {1,2,3}
Cluster 2 = {10,11,12}
```

---

# 20. Choosing K

This is one of the most important parts of K-Means.

Suppose you have:

```text
K = 2
K = 3
K = 4
K = 5
...
```

Which one should you choose?

Two popular methods are:

1. **Elbow Method**
2. **Silhouette Score**

---

# 21. Elbow Method

The Elbow Method uses **inertia/WCSS**.

For different values of K, calculate:

$$
WCSS =
\sum_{k=1}^{K}
\sum_{x_i \in C_k}
||x_i-\mu_k||^2
$$

Then plot:

```text
K        WCSS
1        5000
2        2500
3        1200
4         900
5         850
6         820
```

You might see something like:

```text
WCSS
 |
 |\
 | \
 |  \
 |   \
 |    \__
 |       \____
 |
 +----------------
    1  2  3  4  5
             K
```

The point where the reduction starts becoming much smaller is called the:

> **Elbow**

If the elbow occurs at:

```text
K = 3
```

we may choose:

```text
K = 3
```

---

# 22. Why Does Inertia Always Decrease as K Increases?

This is extremely important.

Suppose:

```text
K = 1
```

All points belong to one cluster.

There is a lot of error.

Then:

```text
K = 2
```

We have two centroids.

The error decreases.

Then:

```text
K = 3
```

Error decreases further.

Eventually:

```text
K = number of data points
```

Each point can have its own cluster.

Then:

$$
WCSS = 0
$$

Therefore:

> You cannot simply choose the K with the lowest inertia.

Because increasing K will almost always reduce inertia.

That's why we look for the **elbow** rather than the absolute minimum.

---

# 23. Silhouette Score

Another important method is the **Silhouette Score**.

It measures how well each point fits into its assigned cluster compared with other clusters.

The score ranges approximately from:

$$
-1 \text{ to } 1
$$

### Close to +1

The point is:

```text
Very close to its own cluster
Far from other clusters
```

Good separation.

### Around 0

The point may lie near a boundary.

### Negative

The point may be closer to another cluster than its assigned cluster.

---

# 24. Silhouette Formula

For a data point \(i\):

$$
s(i)=
\frac{b(i)-a(i)}
{\max(a(i),b(i))}
$$

where:

### \(a(i)\)

Average distance from point \(i\) to points in its own cluster.

### \(b(i)\)

Smallest average distance from point \(i\) to points in another cluster.

So:

$$
s(i)\rightarrow1
$$

means good clustering.

---

# 25. Elbow vs Silhouette

| Method     | Measures                      | Goal                       |
| ---------- | ----------------------------- | -------------------------- |
| Elbow      | Inertia/WCSS                  | Find diminishing returns   |
| Silhouette | Cluster separation + cohesion | Higher is generally better |

You can use both rather than depending on only one.

---

# 26. Feature Scaling and K-Means

This is **very important**.

K-Means is distance-based.

Suppose we have:

```text
Age:     20 - 60
Income:  20,000 - 200,000
```

Income has a much larger numerical scale.

Euclidean distance may therefore be dominated by income.

For example:

```text
Age difference = 10
Income difference = 50,000
```

The income difference can dominate the distance calculation.

Therefore, we often perform:

```text
StandardScaler
```

before K-Means.

---

# 27. Standardization

StandardScaler transforms:

$$
x'=\frac{x-\mu}{\sigma}
$$

where:

* \(\mu\) = mean
* \(\sigma\) = standard deviation

After standardization, features are approximately:

```text
mean = 0
standard deviation = 1
```

Example:

```text
Income
10000
20000
30000
40000
```

becomes something roughly like:

```text
-1.34
-0.45
 0.45
 1.34
```

Now features are on comparable scales.

---

# 28. Important Exception

Scaling isn't automatically required in every situation.

If your features are already:

```text
0–1
```

or naturally have comparable scales, additional scaling may not be necessary.

But because K-Means is distance-based, **always check feature scales**.

---
