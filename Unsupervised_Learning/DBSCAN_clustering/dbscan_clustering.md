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
