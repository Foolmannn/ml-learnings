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
