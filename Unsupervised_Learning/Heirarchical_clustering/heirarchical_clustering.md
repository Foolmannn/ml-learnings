# Hierarchical Clustering in Machine Learning

**Hierarchical Clustering** is an unsupervised learning algorithm that creates a **hierarchy of clusters**.

Unlike K-Means, you don't necessarily need to decide the number of clusters before running the algorithm. Instead, hierarchical clustering builds a tree-like structure of how observations are grouped, and you can choose the final number of clusters afterward.

The resulting tree is called a **dendrogram**.

---

# 1. Basic Idea

Suppose we have six data points:

```text
A   B   C                 D   E   F
•   •   •                 •   •   •
```

We want to discover groups.

Hierarchical clustering might discover:

```text
          ┌───────────────┐
          │               │
       ┌──┴──┐         ┌──┴──┐
       │     │         │     │
      ┌┴┐    C        ┌┴┐    F
      A B     D        E
```

The hierarchy tells us **which points were grouped together and at what distance**.

---

# 2. Why "Hierarchical"?

Because clusters are created at different levels.

For example:

```text
Level 1:

A   B   C   D   E   F
│   │   │   │   │   │

Level 2:

└─AB─┘       └─DE─┘

Level 3:

└──ABC──┘    └──DEF──┘

Level 4:

└──────── ABCDEF ────────┘
```

So instead of producing only one grouping, hierarchical clustering gives us a **whole hierarchy of possible groupings**.

---

# 3. Two Types of Hierarchical Clustering

There are two major approaches:

### 1. Agglomerative Clustering

**Bottom → Up**

Start with every observation as its own cluster and repeatedly merge clusters.

This is the most commonly used form.

### 2. Divisive Clustering

**Top → Down**

Start with all observations in one cluster and repeatedly split them.

```text
             Hierarchical
              Clustering
                   │
          ┌────────┴────────┐
          ↓                 ↓
    Agglomerative        Divisive
      Bottom-Up           Top-Down
```

Most practical implementations, including `sklearn.cluster.AgglomerativeClustering`, use the agglomerative approach.

---

# 4. Agglomerative Hierarchical Clustering

Let's understand this carefully.

Suppose we have:

```text
A
B
C
D
E
```

Initially:

```text
{A} {B} {C} {D} {E}
```

Each point is its own cluster.

Then:

```text
{A,B} {C} {D} {E}
```

Then:

```text
{A,B} {C,D} {E}
```

Then:

```text
{A,B} {C,D,E}
```

Finally:

```text
{A,B,C,D,E}
```

This is called **agglomeration**, meaning progressively combining smaller groups into larger groups.

---

# 5. The Complete Algorithm

The basic algorithm is:

### Step 1

Treat every observation as an individual cluster.

```text
A B C D E
```

### Step 2

Calculate the distances between clusters.

### Step 3

Find the two closest clusters.

### Step 4

Merge them.

### Step 5

Recalculate the distances between the new cluster and the remaining clusters.

### Step 6

Repeat until everything becomes one cluster.

```text
A B C D E
↓
AB C D E
↓
AB CD E
↓
AB CDE
↓
ABCDE
```

The history of these merges is represented by the dendrogram.

---

# 6. Simple Example

Consider one-dimensional data:

$$
X = [1,2,3,10,11,12]
$$

Visually:

```text
1   2   3                    10  11  12
•   •   •                    •   •   •
```

Clearly, there seem to be two natural groups:

```text
Cluster 1: 1, 2, 3

Cluster 2: 10, 11, 12
```

But hierarchical clustering doesn't start by saying:

> "There are 2 clusters."

It starts with:

```text
{1} {2} {3} {10} {11} {12}
```

Then progressively merges the closest observations.

---

# 7. Distance Between Individual Points

The simplest distance is **Euclidean distance**.

For two points:

$$
x=(x_1,x_2)
$$

and

$$
y=(y_1,y_2)
$$

Euclidean distance is:

$$
d(x,y)=
\sqrt{(x_1-y_1)^2+(x_2-y_2)^2}
$$

For \(n\) dimensions:

$$
d(x,y)=
\sqrt{\sum_{i=1}^{n}(x_i-y_i)^2}
$$

Example:

$$
A=(2,3)
$$

$$
B=(5,7)
$$

Then:

$$
d(A,B)
=
\sqrt{(2-5)^2+(3-7)^2}
$$

$$
=\sqrt{9+16}
$$

$$
=5
$$

But there's an important issue.

Initially we compare **individual points**.

Later, we need to compare **clusters**.

For example:

```text
Cluster A = {A,B}

Cluster B = {C,D,E}
```

So how do we define:

$$
Distance(Cluster A, Cluster B)?
$$

That's where **linkage methods** come in.

---

# 8. Linkage

**Linkage determines how the distance between two clusters is calculated.**

This is one of the most important concepts in hierarchical clustering.

The major linkage methods are:

1. Single linkage
2. Complete linkage
3. Average linkage
4. Ward linkage

---

# 9. Single Linkage

Single linkage defines cluster distance as the distance between the **closest pair of points** from the two clusters.

Suppose:

```text
Cluster A: A1 A2 A3

Cluster B: B1 B2 B3
```

Calculate all cross-cluster distances:

```text
A1 → B1
A1 → B2
A1 → B3

A2 → B1
A2 → B2
A2 → B3

A3 → B1
A3 → B2
A3 → B3
```

Then:

$$
D(A,B)=\min d(a,b)
$$

where:

$$
a\in A,\quad b\in B
$$

So it uses the **closest pair**.

### Intuition

```text
A A A

       B B B
       B B B
```

If one point from A is close to one point from B, the clusters may be considered close.

---

# 10. Problem with Single Linkage

Single linkage can create a **chaining effect**.

Imagine:

```text
A A A

      A

          A

             B

                B B B
```

Individual points can form a chain.

Eventually, the algorithm may combine points that don't really form a compact cluster.

This is called:

> **Chaining effect**

Single linkage is therefore sensitive to this phenomenon.

---

# 11. Complete Linkage

Complete linkage does the opposite.

It uses the **farthest pair of points** between two clusters.

$$
D(A,B)=
\max_{a\in A,b\in B}d(a,b)
$$

So instead of asking:

> What's the closest distance?

it asks:

> What's the largest distance between the two clusters?

This tends to produce more compact clusters.

---

# 12. Single vs Complete Linkage

Imagine:

```text
Cluster A          Cluster B

A A A                 B B B
A A A                 B B B
```

### Single linkage

Looks at:

```text
closest A ↔ closest B
```

### Complete linkage

Looks at:

```text
farthest A ↔ farthest B
```

Therefore:

| Linkage  | Uses             |
| -------- | ---------------- |
| Single   | Minimum distance |
| Complete | Maximum distance |

---

# 13. Average Linkage

Average linkage considers the **average distance between all pairs of points** across the two clusters.

$$
D(A,B)=
\frac{1}{|A||B|}
\sum_{a\in A}
\sum_{b\in B}
d(a,b)
$$

This is a compromise between:

* Single linkage
* Complete linkage

It is often less sensitive to chaining than single linkage.

---

# 14. Ward Linkage

Ward linkage is particularly important.

Instead of directly considering pairwise cluster distances, Ward's method chooses the merge that results in the **smallest increase in within-cluster variance** (equivalently, within-cluster sum of squares under the standard formulation).

The intuition is:

> Merge the two clusters that produce the smallest increase in total within-cluster variation.

This tends to create relatively compact, roughly spherical clusters.

In scikit-learn:

```python
AgglomerativeClustering(
    n_clusters=3,
    linkage="ward"
)
```

Ward linkage requires **Euclidean distance** in scikit-learn's implementation.

---

# 15. Linkage Comparison

| Linkage  | Main idea                     | Typical behavior                |
| -------- | ----------------------------- | ------------------------------- |
| Single   | Closest pair                  | Can create chains               |
| Complete | Farthest pair                 | Compact clusters                |
| Average  | Average pair distance         | Balanced                        |
| Ward     | Minimize increase in variance | Compact, spherical-ish clusters |

---

# 16. Dendrogram

The **dendrogram** is the most recognizable part of hierarchical clustering.

It shows the sequence of merges.

For example:

```text
Distance
   ↑
  10 |             ┌──────────────┐
     |             │              │
   8 |       ┌─────┴─────┐        │
     |       │           │        │
   6 |   ┌───┴───┐       │        │
     |   │       │       │        │
   4 | ┌─┴─┐     │       │        │
     | │   │     │       │        │
   2 | A   B     C       D        E
     +──────────────────────────────→
```

The **vertical height at which two branches merge** represents the linkage distance/merge criterion used by the algorithm.

---

# 17. How to Choose Number of Clusters

This is one of the biggest advantages of hierarchical clustering.

You can inspect the dendrogram and choose where to cut it.

Suppose:

```text
                ┌───────────────┐
                │               │
          ┌─────┴─────┐         │
          │           │         │
       ┌──┴──┐        │         │
       A     B        C         D
```

If we cut here:

```text
------------------------------
```

we might get:

```text
Cluster 1 → A,B
Cluster 2 → C
Cluster 3 → D
```

A different cut gives a different number of clusters.

---

# 18. The Horizontal Cut

This is an important exam/interview concept.

Imagine:

```text
             ┌─────────────┐
             │             │
        ┌────┴────┐        │
        │         │        │
       ┌┴┐       ┌┴┐       D
       A B       C E
```

Draw a horizontal line:

```text
-----------------------------
```

The number of branches that the line intersects gives the number of clusters.

So:

> **Number of clusters = number of dendrogram branches intersected by the horizontal cut.**

---

# 19. How Does the Dendrogram Decide the Merge?

Let's use:

$$
X=[1,2,3,10,11,12]
$$

Initially:

```text
{1} {2} {3} {10} {11} {12}
```

Closest pairs:

$$
1,2
$$

$$
2,3
$$

$$
10,11
$$

$$
11,12
$$

The algorithm starts merging the closest clusters.

Eventually:

```text
{1,2,3}
```

and:

```text
{10,11,12}
```

become two large clusters.

Finally, they merge into:

```text
{1,2,3,10,11,12}
```

That final merge occurs at a much larger distance.

That large jump is often useful when deciding where to cut the dendrogram.

---

# 20. Agglomerative Clustering Algorithm — Pseudocode

```text
Start with every data point as its own cluster

while more than one cluster exists:

    calculate distance between clusters

    find the two closest clusters

    merge them

    update cluster distances

return hierarchy
```

The actual behavior depends heavily on the selected linkage.

---

# 21. Mathematical View

Suppose we have clusters:

$$
C_1,C_2,\ldots,C_k
$$

At each iteration, we select:

$$
(C_i,C_j)
=
\arg\min_{C_a,C_b}D(C_a,C_b)
$$

where \(D\) is the chosen linkage-based cluster distance.

Then:

$$
C_{new}=C_i\cup C_j
$$

We repeat until:

$$
C=\{X_1,X_2,\ldots,X_n\}
$$

becomes one cluster.

---

# 22. Important Difference from K-Means

This is extremely important.

### K-Means

You normally choose:

$$
K
$$

before training.

```text
K = 3

Data → K-Means → 3 clusters
```

### Hierarchical

You can construct the hierarchy first:

```text
Data
 ↓
Hierarchy
 ↓
Dendrogram
 ↓
Choose cut
 ↓
Clusters
```

Therefore, hierarchical clustering gives you more flexibility to inspect different cluster resolutions.

---

# 23. Hierarchical vs K-Means

| Feature                  | K-Means                | Hierarchical            |
| ------------------------ | ---------------------- | ----------------------- |
| Need K initially         | Yes                    | Not necessarily         |
| Output                   | Cluster assignments    | Hierarchy + assignments |
| Dendrogram               | No                     | Yes                     |
| Scalability              | Generally better       | Generally worse         |
| Visualization            | Limited                | Excellent               |
| Cluster shape            | Usually centroid-based | Depends on linkage      |
| Need repeated iterations | Yes                    | Sequential merging      |
| Large datasets           | Usually preferable     | Can become expensive    |
| Hierarchy information    | No                     | Yes                     |

---

# 24. Hierarchical Clustering in Python

Using scikit-learn:

```python
from sklearn.cluster import AgglomerativeClustering

model = AgglomerativeClustering(
    n_clusters=3,
    linkage="ward"
)

labels = model.fit_predict(X)

print(labels)
```

Example output:

```text
[0 0 1 1 2 2]
```

These numbers are cluster IDs.

**Important:** Cluster `0` isn't inherently "better" or "smaller" than cluster `1`. The labels are simply identifiers.

---

# 25. Using Different Linkages

### Single

```python
model = AgglomerativeClustering(
    n_clusters=3,
    linkage="single"
)
```

### Complete

```python
model = AgglomerativeClustering(
    n_clusters=3,
    linkage="complete"
)
```

### Average

```python
model = AgglomerativeClustering(
    n_clusters=3,
    linkage="average"
)
```

### Ward

```python
model = AgglomerativeClustering(
    n_clusters=3,
    linkage="ward"
)
```

---

# 26. Scaling is Important

Suppose you have:

```text
Age       → 18–60
Income    → 20,000–5,000,000
```

Distance calculations will be heavily influenced by income.

Therefore, feature scaling is usually important.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)
```

Then:

```python
model = AgglomerativeClustering(
    n_clusters=3,
    linkage="ward"
)

labels = model.fit_predict(X_scaled)
```

---

# 27. Creating a Dendrogram

For dendrogram visualization, we commonly use SciPy.

```python
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

Z = linkage(X_scaled, method="ward")

dendrogram(Z)

plt.xlabel("Data Points")
plt.ylabel("Distance")
plt.title("Hierarchical Clustering Dendrogram")

plt.show()
```

Here:

```python
linkage(X_scaled, method="ward")
```

builds the hierarchical merge structure.

Then:

```python
dendrogram(Z)
```

visualizes it.

---

# 28. Dendrogram + Number of Clusters

Suppose the dendrogram looks roughly like:

```text
Distance

  10 |          ┌───────────────┐
     |          │               │
   8 |          │               │
     |     ┌────┘               │
   6 |     │                    │
     | ┌───┴───┐                │
   4 | │       │                │
     | │       │                │
   2 | A B C D E F
```

There is a large vertical jump before the final merge.

A horizontal cut before that large jump can produce a useful cluster count.

This is conceptually similar to looking for a large separation in the dendrogram.

---

# 29. Agglomerative Clustering Parameters in Scikit-Learn

The important parameters include:

```python
AgglomerativeClustering(
    n_clusters=3,
    metric="euclidean",
    linkage="ward"
)
```

### `n_clusters`

Number of clusters you want at the end.

```python
n_clusters=3
```

### `metric`

Distance metric.

Examples can include:

```text
euclidean
manhattan
cosine
```

The valid combinations depend on the linkage.

For example, Ward uses Euclidean distance.

### `linkage`

Controls how cluster-to-cluster distance/merge cost is determined.

```text
single
complete
average
ward
```

---

# 30. Important Difference: `metric` vs `linkage`

These are often confused.

### Metric

Answers:

> **How do I measure distance between individual observations?**

Example:

$$
d(A,B)
$$

### Linkage

Answers:

> **How do I determine the distance/merge criterion between two clusters?**

Example:

$$
D(C_1,C_2)
$$

So:

```text
Individual points
      ↓
    Metric
      ↓
Point-to-point distances
      ↓
   Linkage
      ↓
Cluster relationship
```

---

# 31. Advantages

### 1. No need to commit to K at the beginning

You can inspect the hierarchy and choose a cut.

### 2. Dendrogram is interpretable

You can visually see how observations merge.

### 3. Can discover different levels of grouping

For example:

```text
Country
 ├── Region
 │    ├── Group A
 │    └── Group B
 └── Region
      ├── Group C
      └── Group D
```

### 4. No centroid requirement

Unlike K-Means, hierarchical clustering does not require calculating cluster centroids as its fundamental representation.

### 5. Useful for exploratory analysis

Especially when you want to understand relationships between observations.

---

# 32. Disadvantages

### 1. Computationally expensive

Hierarchical clustering can become expensive for large datasets because it repeatedly considers distances between clusters.

### 2. Sensitive to distance metric

Different metrics can produce different hierarchies.

### 3. Sensitive to linkage

Single, complete, average, and Ward can produce substantially different results.

### 4. Difficult to undo merges

In standard agglomerative clustering, once two clusters are merged, that decision isn't normally undone.

### 5. Scaling matters

Features with larger numerical scales can dominate distance calculations.

### 6. Noise/outliers can affect the hierarchy

Outliers may create undesirable merges depending on the linkage.

---

# 33. When Should You Use Hierarchical Clustering?

It is particularly useful when:

* Dataset size is manageable
* You want to understand hierarchical relationships
* You don't know the appropriate cluster count initially
* You want a dendrogram
* Interpretability of the hierarchy matters
* You are doing exploratory data analysis

Examples:

### Biology

Grouping species based on characteristics.

### Document analysis

Grouping documents based on similarity.

### Customer segmentation

Finding different levels of customer similarity.

### Gene expression analysis

Finding groups of genes with similar behavior.

### Image analysis

Grouping similar image representations.

---

# 34. Hierarchical Clustering Example

Suppose we have customer data:

| Customer | Age | Annual Income |
| -------- | --: | ------------: |
| A        |  20 |            25 |
| B        |  22 |            27 |
| C        |  25 |            30 |
| D        |  45 |            80 |
| E        |  47 |            82 |
| F        |  50 |            85 |

After scaling:

```text
A B C                     D E F
• • •                     • • •
```

Hierarchical clustering might discover:

```text
Cluster 1:
A B C

Cluster 2:
D E F
```

But the dendrogram can also show the finer structure:

```text
             ┌───────────────┐
             │               │
        ┌────┴────┐      ┌───┴───┐
        │         │      │       │
       A B        C     D E      F
```

You could choose:

```text
2 clusters
```

or potentially:

```text
3 clusters
```

depending on the desired level of granularity.

---

# 35. A Very Important Concept: Hierarchical Clustering Does Not "Know" the Meaning of Clusters

Suppose the algorithm returns:

```text
Cluster 0
Cluster 1
Cluster 2
```

It doesn't know:

```text
Cluster 0 = poor customers
Cluster 1 = middle-income customers
Cluster 2 = wealthy customers
```

You have to analyze the characteristics of each cluster.

For example:

```python
df["cluster"] = labels

print(
    df.groupby("cluster")[["age", "income", "spending"]].mean()
)
```

You might then discover:

```text
cluster    age    income    spending
0          23     27        85
1          48     82        20
2          35     50        60
```

Now you can interpret the groups.

---

# 36. Hierarchical Clustering Workflow

A practical workflow looks like this:

```text
Raw Data
   ↓
Data Cleaning
   ↓
Select Features
   ↓
Handle Missing Values
   ↓
Feature Scaling
   ↓
Choose Distance Metric
   ↓
Choose Linkage
   ↓
Build Hierarchy
   ↓
Generate Dendrogram
   ↓
Choose Cut / Number of Clusters
   ↓
Fit Clustering
   ↓
Evaluate
   ↓
Interpret Clusters
```

---

# 37. Hierarchical Clustering vs DBSCAN vs K-Means

| Property         | K-Means        | Hierarchical       | DBSCAN               |
| ---------------- | -------------- | ------------------ | -------------------- |
| Type             | Centroid-based | Hierarchical       | Density-based        |
| Need K           | Yes            | Not necessarily    | No                   |
| Dendrogram       | No             | Yes                | No                   |
| Handles noise    | Poorly         | Depends            | Well                 |
| Arbitrary shapes | Poorly         | Depends on linkage | Very well            |
| Large data       | Good           | Less suitable      | Moderate             |
| Hierarchy        | No             | Yes                | No                   |
| Main parameter   | K              | Linkage + cut      | `eps`, `min_samples` |

---

# 38. What You Should Remember

If you're preparing this for ML study/interviews, remember these key points:

### Hierarchical clustering

> Builds a hierarchy of clusters by progressively merging or splitting groups.

### Agglomerative

> Bottom-up: every point starts as its own cluster and clusters are repeatedly merged.

### Divisive

> Top-down: all points start in one cluster and are repeatedly split.

### Dendrogram

> Tree diagram showing the hierarchy of merges.

### Linkage

Determines how the distance/merge criterion between clusters is calculated.

### Single linkage

$$
\boxed{\text{minimum distance}}
$$

### Complete linkage

$$
\boxed{\text{maximum distance}}
$$

### Average linkage

$$
\boxed{\text{average pairwise distance}}
$$

### Ward linkage

$$
\boxed{\text{minimize increase in within-cluster variance}}
$$

### Main advantage

> Gives a hierarchy rather than only one fixed clustering.

### Main disadvantage

> Can be computationally expensive for large datasets.

---

## Recommended next study

Since you're learning unsupervised learning in sequence, after this I would study **Hierarchical Clustering Mathematics**, especially:

1. Euclidean distance
2. Distance matrix
3. Single linkage calculation
4. Complete linkage calculation
5. Average linkage calculation
6. Ward linkage mathematics
7. Step-by-step construction of a dendrogram
8. How a dendrogram determines \(K\)
9. Time/space complexity
10. Full Python implementation from scratch
11. Scikit-learn implementation
12. Silhouette-score evaluation

That mathematical step-by-step calculation is particularly useful because it makes the entire hierarchical clustering algorithm much easier to understand.
