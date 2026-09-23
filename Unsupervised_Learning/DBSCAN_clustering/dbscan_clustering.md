# DBSCAN Clustering in Detail

**DBSCAN (Density-Based Spatial Clustering of Applications with Noise)** is an **unsupervised machine learning clustering algorithm** that groups data points based on their **density** rather than requiring us to specify the number of clusters beforehand.

It is especially useful when:

* Clusters have irregular/non-spherical shapes.
* The dataset contains noise or outliers.
* We don't know the number of clusters in advance.
* We want to explicitly identify outliers.

---

# 1. What is DBSCAN?

DBSCAN stands for:

> **Density-Based Spatial Clustering of Applications with Noise**

The main idea is:

> **Points that are closely packed together form a cluster, while points lying in low-density regions are considered noise or outliers.**

For example, imagine data like this:

```text
        ● ● ●
      ● ● ● ●
       ● ● ●

                    ×

                         ● ●
                       ● ● ●
                        ● ●
```

DBSCAN can identify:

```text
Cluster 1          Noise       Cluster 2

   ● ● ●
 ● ● ● ●             ×           ● ●
  ● ● ●                       ● ● ●
                                  ● ●
```

Notice that DBSCAN does **not** assume that clusters must be circular.

---

# 2. Why Do We Need DBSCAN?

Consider K-Means.

K-Means requires us to specify:

```text
n_clusters = 2
```

It also tends to work best when clusters are roughly spherical/convex.

Suppose our data looks like:

```text
       ● ● ● ●
     ●         ●
    ●           ●
     ●         ●
       ● ● ● ●
```

and another cluster is inside/outside another irregular shape.

K-Means can struggle because it divides the space according to distance from centroids.

DBSCAN instead asks:

> "Where are the data points densely packed?"

This allows it to discover complicated cluster shapes.

---

# 3. Core Idea of DBSCAN

DBSCAN is based on **density**.

It looks at the neighborhood around each point.

For example:

```text
        ●
      ● ● ●
        ●
        P
```

If many points are close to `P`, then `P` belongs to a dense region.

But:

```text
                    ●

             P
```

has very few nearby points, so `P` may be considered noise.

DBSCAN mainly uses **two parameters**:

### 1. `eps`

The maximum distance that defines a point's neighborhood.

### 2. `min_samples`

The minimum number of points required in that neighborhood to consider the region sufficiently dense.

---

# 4. Important Terminology

There are three important types of points in DBSCAN:

1. **Core Point**
2. **Border Point**
3. **Noise Point**

Understanding these three is extremely important for exams and implementation.

---

# 5. Core Point

A point is called a **core point** if it has at least `min_samples` points within distance `eps`.

Suppose:

```text
eps = 1
min_samples = 5
```

Consider:

```text
       ●
    ●  P  ●
       ●
       ●
```

If at least 5 points are within the `eps` radius of `P`, then:

```text
P = Core Point
```

Formally:

$$
|N_{\epsilon}(P)| \geq MinPts
$$

where:

* \(N_{\epsilon}(P)\) = neighborhood of P within epsilon
* \(MinPts\) = minimum number of points

---

# 6. Border Point

A **border point** does not have enough neighbors to be a core point itself, but it lies within the `eps` neighborhood of a core point.

Example:

```text
           ●
        ●  ●  ●
           ●

                  B
```

Suppose the points on the left form a dense region.

`B` might not have enough neighbors to become a core point.

But because:

```text
distance(B, core point) <= eps
```

it can be included in the cluster.

Therefore:

> A border point belongs to a cluster but does not have enough neighbors to be a core point itself.

---

# 7. Noise Point

A point is considered **noise** if:

* It is not a core point.
* It is not reachable from any core point.

Example:

```text
      ● ● ●
    ● ● ● ●

                    ×
```

The `×` has no sufficiently dense neighborhood.

Therefore:

```text
× = Noise
```

Noise is sometimes called an **outlier** in DBSCAN.

---

# 8. Visual Understanding

Consider:

```text
                    N
                    ×


       ● ● ●
     ● ● ● ●
    ● ● ● ● ●
     ● ● ● ●
       ● ●

                     ●
                   ● ● ●
                  ● ● ● ●
                   ● ● ●
                     ●
```

DBSCAN could identify:

```text
Cluster 1
Cluster 2
Noise
```

without us specifying:

```python
n_clusters = 2
```

This is one of DBSCAN's major advantages.

---

# 9. The `eps` Parameter

`eps` means **epsilon**.

It defines the radius of the neighborhood around a point.

Mathematically:

$$
N_{\epsilon}(p)
=
\{q \in D \mid dist(p,q) \leq \epsilon\}
$$

In simple terms:

> Find all points whose distance from `p` is less than or equal to `eps`.

For example:

```text
eps = 1
```

means:

```text
          ●
       ●  P  ●
          ●
```

Only points within that radius are considered neighbors.

---

# 10. Effect of `eps`

Choosing `eps` incorrectly can significantly affect DBSCAN.

### Very small `eps`

```text
●     ●      ●
```

Very few points become neighbors.

Result:

* Many points may become noise.
* Clusters may break into smaller pieces.

---

### Very large `eps`

```text
● ● ● ● ● ● ● ● ●
```

Many points become neighbors.

Result:

* Different clusters may merge.
* Almost everything may become one cluster.
* Noise detection becomes poor.

Therefore:

> Choosing a suitable `eps` is important.

---

# 11. `min_samples`

`min_samples` determines how many points are required in a neighborhood for a point to be considered a core point.

For example:

```python
DBSCAN(eps=0.5, min_samples=5)
```

means:

> A point needs at least 5 points in its epsilon neighborhood to qualify as a core point.

Higher `min_samples` means:

* Higher density requirement
* More conservative clusters
* Potentially more noise

Lower `min_samples` means:

* Easier to form clusters
* More points can become core points
* Potentially more spurious clusters

---

# 12. How DBSCAN Works

The basic process is:

```text
Start
  ↓
Select an unvisited point
  ↓
Find its eps-neighbors
  ↓
Does it have enough neighbors?
  ↓
 ┌───────────────┐
 │               │
No              Yes
 │               │
 ↓               ↓
Noise        Core Point
                 ↓
          Create a cluster
                 ↓
        Expand the cluster
                 ↓
        Repeat for neighbors
                 ↓
       Process remaining points
                 ↓
                End
```

---

# 13. Step-by-Step Algorithm

Suppose:

```text
eps = 0.5
min_samples = 4
```

### Step 1: Select a point

Choose an unvisited point `P`.

---

### Step 2: Find neighbors

Find every point within:

$$
distance(P,Q) \leq eps
$$

---

### Step 3: Count neighbors

Suppose:

```text
Number of neighbors = 6
min_samples = 4
```

Then:

```text
6 >= 4
```

So `P` is a core point.

---

### Step 4: Create a cluster

Start:

```text
Cluster 1
```

---

### Step 5: Expand the cluster

Check every neighbor.

If a neighboring point is also a core point, its neighbors are added to the cluster.

For example:

```text
Core P
 ↓
neighbors
 ↓
Core Q
 ↓
neighbors of Q
 ↓
Core R
 ↓
neighbors of R
```

The cluster keeps expanding.

---

### Step 6: Border points

If a point doesn't have enough neighbors to become a core point but is close to a core point, it is added as a border point.

---

### Step 7: Noise

If a point cannot be associated with any dense region, it becomes noise.

---

### Step 8: Repeat

DBSCAN continues until all points have been processed.

---

# 14. Density Reachability

This is an important theoretical concept in DBSCAN.

Suppose:

```text
A → B → C → D
```

and the points are sufficiently close according to the density conditions.

A point can become part of a cluster through a chain of dense points.

This is called **density-reachability**.

For example:

```text
Core A
   ↓
Core B
   ↓
Core C
   ↓
Border D
```

`D` can belong to the same cluster because it is reachable from a core point.

---

# 15. Directly Density-Reachable

Point `q` is directly density-reachable from point `p` if:

1. `q` is within `eps` of `p`
2. `p` is a core point

Formally:

$$
q \in N_\epsilon(p)
$$

and

$$
|N_\epsilon(p)| \geq MinPts
$$

---

# 16. Density-Reachable

A point `q` is density-reachable from `p` if there exists a sequence:

$$
p=p_1,p_2,\ldots,p_n=q
$$

such that each point is directly density-reachable from the previous point.

For example:

```text
P1 → P2 → P3 → P4
```

If:

```text
P1 → P2
P2 → P3
P3 → P4
```

satisfy the density conditions, then `P4` is density-reachable from `P1`.

---

# 17. Density-Connected

Two points are density-connected if there is some point `o` from which both points are density-reachable.

Conceptually:

```text
        P
       /
      O
       \
        Q
```

Both `P` and `Q` can be reached from the same dense region.

Therefore, they can belong to the same DBSCAN cluster.

---

# 18. Mathematical Definition of a DBSCAN Cluster

A DBSCAN cluster satisfies two important properties.

### Connectivity

Points in the cluster should be density-connected.

### Maximality

If a point is density-reachable from a cluster point, it should also belong to the cluster.

This allows DBSCAN to construct clusters based on density rather than centroid distance.

---

# 19. DBSCAN Example

Consider:

```text
      A   B   C
        D E F
      G   H

                      X


             P Q R
              S T
```

Suppose:

```text
eps = suitable radius
min_samples = suitable density
```

The first group may become:

```text
Cluster 0
```

The second group:

```text
Cluster 1
```

And:

```text
X
```

becomes:

```text
Noise = -1
```

In scikit-learn, DBSCAN uses:

```text
-1
```

to represent noise.

---

# 20. DBSCAN vs K-Means

| Feature                     | DBSCAN               | K-Means               |
| --------------------------- | -------------------- | --------------------- |
| Type                        | Density-based        | Centroid-based        |
| Need number of clusters?    | No                   | Yes                   |
| Handles noise               | Excellent            | Poor                  |
| Irregular clusters          | Excellent            | Poor                  |
| Spherical clusters          | Not required         | Generally preferred   |
| Outlier detection           | Yes                  | No explicit mechanism |
| Main parameters             | `eps`, `min_samples` | `n_clusters`          |
| Cluster center              | No centroid          | Centroid              |
| Sensitive to scale          | Yes                  | Yes                   |
| Works with arbitrary shapes | Yes                  | Limited               |

---

# 21. DBSCAN vs Hierarchical Clustering

| Feature                               | DBSCAN                                     | Hierarchical         |
| ------------------------------------- | ------------------------------------------ | -------------------- |
| Number of clusters required initially | No                                         | No                   |
| Handles noise                         | Very well                                  | Not inherently       |
| Arbitrary shapes                      | Yes                                        | Often yes            |
| Main concept                          | Density                                    | Distance/tree        |
| Main parameter                        | `eps`, `min_samples`                       | Linkage/distance     |
| Output                                | Flat clusters                              | Dendrogram           |
| Large datasets                        | Generally efficient with suitable indexing | Can become expensive |

---

# 22. Advantages of DBSCAN

### 1. No need to specify number of clusters

Unlike K-Means:

```python
KMeans(n_clusters=3)
```

DBSCAN doesn't require:

```text
number of clusters
```

beforehand.

---

### 2. Finds arbitrary-shaped clusters

For example:

```text
      ● ● ●
    ●       ●
   ●         ●
    ●       ●
      ● ● ●
```

DBSCAN can potentially identify the shape as one cluster.

---

### 3. Handles noise

DBSCAN explicitly identifies points as:

```text
-1
```

when they don't belong to any dense region.

---

### 4. No centroid calculation

DBSCAN does not need:

```text
centroid
```

for each cluster.

---

### 5. Useful for spatial data

The algorithm was originally designed for spatial/database applications, making density-based spatial grouping a natural use case.

---

# 23. Disadvantages of DBSCAN

### 1. Sensitive to `eps`

A poor choice of epsilon can produce poor clusters.

---

### 2. Sensitive to `min_samples`

Different values can significantly change the result.

---

### 3. Struggles with varying-density clusters

Suppose:

```text
Cluster A = very dense

Cluster B = relatively sparse
```

Using one global:

```text
eps
```

can be difficult.

A value suitable for Cluster A may not work well for Cluster B.

---

### 4. Sensitive to feature scaling

Suppose we have:

```text
Age       = 20–60
Salary    = 20,000–500,000
```

Salary dominates Euclidean distance.

Therefore, scaling is usually important.

---

### 5. High-dimensional data

Distance becomes less informative in very high-dimensional spaces because of the **curse of dimensionality**.

Dimensionality reduction or appropriate distance measures may be necessary.

---

# 24. Feature Scaling with DBSCAN

Suppose:

```text
Age = 20
Income = 50000
```

Using Euclidean distance directly:

$$
d = \sqrt{(Age_1-Age_2)^2+(Income_1-Income_2)^2}
$$

Income dominates the distance.

Therefore, we commonly use:

```python
from sklearn.preprocessing import StandardScaler
```

and:

```python
X_scaled = StandardScaler().fit_transform(X)
```

Then apply DBSCAN.

---

# 25. DBSCAN Implementation with Scikit-Learn

Let's create a simple example.

```python
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt

X, y = make_moons(
    n_samples=300,
    noise=0.05,
    random_state=42
)

model = DBSCAN(
    eps=0.3,
    min_samples=5
)

labels = model.fit_predict(X)

plt.scatter(
    X[:, 0],
    X[:, 1],
    c=labels
)

plt.show()
```

The important part is:

```python
model = DBSCAN(
    eps=0.3,
    min_samples=5
)
```

---

# 26. Understanding `fit_predict()`

```python
labels = model.fit_predict(X)
```

performs two operations:

### `fit()`

Learns the clusters from the data.

### `predict()`

Returns cluster labels.

For DBSCAN, `fit_predict()` is convenient because DBSCAN is fundamentally designed to identify clusters in the provided dataset rather than learning a centroid-based model for arbitrary future points.

---

# 27. Understanding DBSCAN Labels

Suppose:

```python
print(labels)
```

you might get:

```text
[0 0 0 1 1 1 1 -1 0 1 ...]
```

Here:

```text
0 → Cluster 0
1 → Cluster 1
-1 → Noise
```

For example:

```python
import numpy as np

print(np.unique(labels))
```

might produce:

```text
[-1  0  1]
```

That means:

```text
Cluster 0
Cluster 1
Noise
```

---

# 28. Finding Number of Clusters

Because DBSCAN doesn't require the number of clusters beforehand, we can determine it from the resulting labels.

```python
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

print(n_clusters)
```

Why subtract one?

Because:

```text
-1
```

represents noise, not a real cluster.

---

# 29. Counting Noise Points

You can calculate:

```python
n_noise = list(labels).count(-1)

print("Noise points:", n_noise)
```

Or:

```python
n_noise = np.sum(labels == -1)
```

---

# 30. Complete Example

```python
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# Generate data
X, y = make_moons(
    n_samples=500,
    noise=0.08,
    random_state=42
)

# Scale features
X = StandardScaler().fit_transform(X)

# Create DBSCAN model
dbscan = DBSCAN(
    eps=0.3,
    min_samples=5
)

# Fit and predict
labels = dbscan.fit_predict(X)

# Number of clusters
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

# Number of noise points
n_noise = np.sum(labels == -1)

print("Number of clusters:", n_clusters)
print("Number of noise points:", n_noise)

# Plot
plt.scatter(
    X[:, 0],
    X[:, 1],
    c=labels
)

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("DBSCAN Clustering")

plt.show()
```

---

# 31. How to Choose `eps`

Choosing `eps` is one of the most important parts of DBSCAN.

A common technique is the:

> **k-distance graph**

The basic idea is:

1. Choose a value of `k`, often related to `min_samples`.
2. Calculate the distance to the k-th nearest neighbor for every point.
3. Sort those distances.
4. Plot them.
5. Look for a noticeable **elbow/knee**.
6. Use that region as a candidate for `eps`.

---

# 32. K-Distance Method

Example:

```python
from sklearn.neighbors import NearestNeighbors
import numpy as np
import matplotlib.pyplot as plt

k = 5

neighbors = NearestNeighbors(
    n_neighbors=k
)

neighbors.fit(X)

distances, indices = neighbors.kneighbors(X)

k_distances = distances[:, -1]

k_distances = np.sort(k_distances)

plt.plot(k_distances)

plt.xlabel("Data Points")
plt.ylabel("5th Nearest Neighbor Distance")
plt.title("K-Distance Graph")

plt.show()
```

You look for the point where the curve begins to rise sharply.

That can provide a reasonable candidate for `eps`.

---

# 33. Choosing `min_samples`

There isn't one universally correct value.

A common heuristic is:

$$
MinPts \geq D+1
$$

where:

* \(D\) = number of dimensions.

Another commonly used practical rule is:

$$
MinPts \approx 2D
$$

or higher for noisy datasets.

For example, if:

```text
D = 2
```

you might experiment with:

```text
min_samples = 4, 5, 6, 10
```

The final value should be selected based on the dataset and desired density definition rather than blindly following a formula.

---

# 34. DBSCAN Mathematics

The central mathematical idea is the **epsilon neighborhood**.

For a point \(p\):

$$
N_\epsilon(p)
=
\{q \in D \mid d(p,q) \leq \epsilon\}
$$

where:

* \(D\) = dataset
* \(p\) = current point
* \(q\) = another point
* \(d(p,q)\) = distance
* \(\epsilon\) = neighborhood radius

A point is core if:

$$
|N_\epsilon(p)| \geq MinPts
$$

Depending on the implementation/convention, whether the point itself is counted in `MinPts` matters. **scikit-learn counts the point itself** when determining `min_samples`.

---

# 35. Euclidean Distance

For two points:

$$
P=(x_1,y_1)
$$

and

$$
Q=(x_2,y_2)
$$

Euclidean distance is:

$$
d(P,Q)
=
\sqrt{
(x_1-x_2)^2+(y_1-y_2)^2
}
$$

DBSCAN checks whether:

$$
d(P,Q)\leq\epsilon
$$

If yes, `Q` is within the epsilon neighborhood of `P`.

---

# 36. DBSCAN Pseudocode

A simplified version:

```text
DBSCAN(D, eps, MinPts):

    cluster_id = 0

    for each point P in D:

        if P is visited:
            continue

        mark P as visited

        neighbors = points within eps of P

        if number of neighbors < MinPts:
            mark P as noise

        else:
            cluster_id += 1
            create cluster

            add P to cluster

            for each point Q in neighbors:

                if Q is not visited:
                    mark Q as visited

                    Q_neighbors = points within eps of Q

                    if Q has at least MinPts neighbors:
                        add Q_neighbors to neighbors

                if Q is not yet assigned to a cluster:
                    add Q to current cluster

    return clusters
```

The key operation is:

```text
Find all neighbors within eps
```

and then recursively expand the cluster through core points.

---

# 37. DBSCAN Algorithm Complexity

With a straightforward implementation, neighborhood searches can be expensive.

A common complexity description is approximately:

$$
O(n\log n)
$$

when efficient spatial indexing structures can be used under favorable conditions.

In unfavorable/high-dimensional settings, it can approach:

$$
O(n^2)
$$

because many pairwise distances may need to be examined.

Therefore, DBSCAN can become computationally expensive for very large or high-dimensional datasets depending on the distance metric and indexing method.

---

# 38. Important Difference: DBSCAN Does Not Use Centroids

K-Means:

```text
Data
 ↓
Centroid
 ↓
Distance
 ↓
Cluster
```

DBSCAN:

```text
Data
 ↓
Neighborhood
 ↓
Density
 ↓
Cluster
```

There is no:

```text
mean
centroid
cluster center
```

in the DBSCAN clustering process.

---

# 39. Why DBSCAN Can Find Arbitrary Shapes

Consider two moon-shaped clusters:

```text
       ● ● ● ●
     ●         ●
   ●             ●

             ●
           ● ● ●
         ● ● ● ●
```

A centroid-based algorithm may struggle because the clusters aren't naturally represented by circular regions around a center.

DBSCAN follows dense connected regions:

```text
Dense region → Dense region → Dense region
```

so it can capture such shapes.

This is one of DBSCAN's biggest strengths.

---

# 40. DBSCAN in Real-World Applications

DBSCAN can be used in:

### Geographic/spatial analysis

Finding dense locations:

```text
GPS points → locations → clusters
```

### Anomaly detection

Finding unusual observations:

```text
normal dense region
        ↓
isolated point → anomaly
```

### Customer segmentation

Customers can potentially be grouped according to behavioral features.

### Image processing

Pixels/features can be grouped according to spatial or feature similarity.

### Network analysis

Dense groups of related entities can sometimes be discovered.

### Location-based applications

For example:

```text
GPS taxi locations
       ↓
DBSCAN
       ↓
popular pickup zones
```

---

# 41. DBSCAN for Anomaly Detection

One particularly useful property is:

```python
labels == -1
```

These observations are considered noise.

For example:

```python
outliers = X[labels == -1]
```

This extracts the points identified as noise.

Therefore:

> DBSCAN can simultaneously perform clustering and density-based outlier identification.

However, `-1` means "not assigned to a DBSCAN cluster"; whether every such point should be treated as a true anomaly depends on the application.

---

# 42. DBSCAN and Scaling

Always think about this when using DBSCAN:

```text
Are my features on comparable scales?
```

If not:

```python
from sklearn.preprocessing import StandardScaler

X_scaled = StandardScaler().fit_transform(X)
```

Then:

```python
DBSCAN(...).fit_predict(X_scaled)
```

Other transformations can also be appropriate depending on the data.

---
