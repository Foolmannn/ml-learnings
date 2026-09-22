# Unsupervised Learning in Machine Learning

**Unsupervised Learning** is a branch of Machine Learning where the model learns patterns, structures, or relationships from **data without labeled target/output values**.

In supervised learning, we have:

$$
X \rightarrow y
$$

where \(X\) is input and \(y\) is the known target.

In unsupervised learning, we only have:

$$
X
$$

The algorithm tries to discover useful structure within \(X\).

---

## 1. What is Unsupervised Learning?

Suppose you have customer data:

| Customer | Age | Income | Spending |
| -------- | --: | -----: | -------: |
| A        |  22 |    25K |       80 |
| B        |  24 |    28K |       85 |
| C        |  45 |    90K |       20 |
| D        |  48 |    95K |       15 |
| E        |  25 |    30K |       90 |

There is no column saying:

> Customer A = "High-value customer"

Instead, the algorithm can discover that some customers naturally form groups.

For example:

```text
        Customer Data
              |
              ↓
      Unsupervised Algorithm
              |
       ┌──────┴──────┐
       ↓             ↓
   Group 1        Group 2
 Young/high      Older/high
 spending        income/
                 low spending
```

The groups were **not given beforehand**. The algorithm discovered them.

---

# 2. Why Do We Need Unsupervised Learning?

A huge amount of real-world data is **unlabeled**.

For example:

* Millions of customer transactions
* Website visitor behavior
* Images
* Documents
* Sensor data
* Social media posts
* Purchase histories
* Network traffic
* Medical measurements
* Satellite/environmental data

Manually labeling all of this data can be expensive and time-consuming.

Unsupervised learning allows us to ask:

> **"What structure exists naturally in this data?"**

---

# 3. Supervised vs Unsupervised Learning

| Aspect          | Supervised                            | Unsupervised             |
| --------------- | ------------------------------------- | ------------------------ |
| Training data   | Labeled                               | Unlabeled                |
| Target variable | Required                              | Not required             |
| Main goal       | Predict                               | Discover structure       |
| Output          | Prediction                            | Groups/patterns/features |
| Example         | House price prediction                | Customer segmentation    |
| Algorithms      | Linear Regression, Random Forest, SVM | K-Means, DBSCAN, PCA     |
| Evaluation      | Often straightforward                 | Often more difficult     |

### Supervised

```text
X → Model → y
```

Example:

```text
House features → Model → House price
```

### Unsupervised

```text
X → Model → Structure
```

Example:

```text
Customer data → Model → Customer groups
```

---

# 4. Main Types of Unsupervised Learning

Unsupervised learning contains several important areas.

```text
                 Unsupervised Learning
                         |
       ┌─────────────────┼─────────────────┐
       ↓                 ↓                 ↓
 Clustering       Dimensionality      Association
                    Reduction            Rules
       |                 |                 |
   K-Means             PCA             Apriori
   DBSCAN              t-SNE           FP-Growth
   Hierarchical        LDA
```

There are also other techniques such as anomaly detection and some generative approaches.

---

# 5. Clustering

**Clustering** means dividing data into groups based on similarity.

The important idea is:

> Objects in the same cluster should be more similar to each other than to objects in other clusters.

For example:

```text
       • • •
      • • •          ○ ○
       • •           ○ ○ ○

        Cluster 1     Cluster 2
```

There are no predefined labels.

The algorithm discovers the groups.

---

## 5.1 K-Means Clustering

K-Means is one of the most important unsupervised algorithms.

You specify:

$$
K = \text{number of clusters}
$$

The algorithm then tries to divide the data into \(K\) clusters.

Example:

```text
K = 3

        • •
       • • •        ○ ○
                    ○ ○ ○

                       ▲ ▲
                      ▲ ▲
```

Three clusters are discovered.

### Basic K-Means process

1. Choose \(K\)
2. Initialize centroids
3. Assign points to nearest centroid
4. Recalculate centroids
5. Repeat
6. Stop when centroids stabilize

You have already started studying K-Means, so this is one of the core algorithms to understand deeply.

---

# 6. Hierarchical Clustering

Hierarchical clustering creates a hierarchy of clusters.

It can be represented using a **dendrogram**.

Example:

```text
                 ┌───────────────┐
                 │               │
             ┌───┴───┐       ┌───┴───┐
             │       │       │       │
            A B     C D     E       F
```

There are two major approaches:

### Agglomerative

Bottom → Up

```text
A   B   C   D

↓   ↓   ↓   ↓

AB     CD

      ↓

     ABCD
```

Start with every point as its own cluster and progressively merge them.

### Divisive

Top → Down

Start with one large cluster and repeatedly split it.

---

# 7. DBSCAN

**DBSCAN = Density-Based Spatial Clustering of Applications with Noise**

Instead of asking:

> "How many clusters should I create?"

DBSCAN looks for **dense regions**.

It can also identify noise/outliers.

Example:

```text
       • • •
      • • • •

                   x

             ○ ○ ○
            ○ ○ ○ ○
```

The `x` could be identified as noise.

This is one of the major differences from K-Means.

### DBSCAN is useful when:

* Clusters have irregular shapes
* Noise exists
* You don't know the number of clusters beforehand

---

# 8. K-Means vs DBSCAN

| Feature                     | K-Means           | DBSCAN           |
| --------------------------- | ----------------- | ---------------- |
| Need number of clusters?    | Yes               | No               |
| Handles noise               | Poorly            | Well             |
| Arbitrary shapes            | Poorly            | Well             |
| Based on                    | Centroid/distance | Density          |
| Outlier detection           | No                | Yes              |
| Sensitive to scale          | Yes               | Yes              |
| Good for spherical clusters | Yes               | Not specifically |

---

# 9. Dimensionality Reduction

Another major area of unsupervised learning is **dimensionality reduction**.

Suppose your dataset has:

```text
1000 features
```

Working with 1000 dimensions can cause problems:

* High computational cost
* Visualization becomes difficult
* Noise
* Redundant features
* Curse of dimensionality

Dimensionality reduction attempts to represent the data using fewer dimensions while retaining important information.

```text
100 features
     ↓
Dimensionality Reduction
     ↓
10 features
```

---

# 10. PCA

**PCA = Principal Component Analysis**

PCA is one of the most important dimensionality-reduction techniques.

Suppose:

$$
X \in \mathbb{R}^{1000}
$$

PCA might transform it into:

$$
Z \in \mathbb{R}^{2}
$$

while preserving as much variance as possible.

This can allow us to visualize complicated datasets:

```text
100-dimensional data
        ↓
       PCA
        ↓
    2 dimensions
        ↓
     Plot it
```

PCA is also useful for:

* Noise reduction
* Feature compression
* Visualization
* Removing correlated/redundant information
* Improving computational efficiency

---

# 11. PCA Intuition

Imagine data distributed like this:

```text
             •
          •
       •
    •
 •
```

Although there are two dimensions, most of the variation occurs along one direction.

PCA finds that important direction.

```text
             •
          •
       •
    •
 •
──────────────────→
       PC1
```

The first principal component captures the largest possible variance.

Then:

$$
PC_1 = \text{maximum variance direction}
$$

$$
PC_2 = \text{maximum remaining variance direction}
$$

with:

$$
PC_1 \perp PC_2
$$

---

# 12. t-SNE

**t-SNE = t-distributed Stochastic Neighbor Embedding**

It is primarily used for **visualizing high-dimensional data**.

For example:

```text
784-dimensional image data
          ↓
         t-SNE
          ↓
       2D data
          ↓
       Visualization
```

It tries to preserve local relationships.

If two points are very similar in the original high-dimensional space, t-SNE attempts to keep them close in the visualization.

It is commonly used for:

* Image embeddings
* Word embeddings
* Deep-learning representations
* Exploratory data analysis

---

# 13. UMAP

**UMAP = Uniform Manifold Approximation and Projection**

Like t-SNE, UMAP is commonly used to visualize high-dimensional data.

It can often be faster and preserve more global structure than t-SNE, although behavior depends on the dataset and parameters.

Typical use:

```text
High-dimensional embeddings
          ↓
         UMAP
          ↓
        2D/3D
          ↓
    Visualization
```

---

# 14. Association Rule Learning

Association rule learning discovers relationships between items.

The classic example is:

> Customers who buy X often also buy Y.

For example:

```text
Bread + Butter
      ↓
   Milk
```

A supermarket could discover:

$$
\{Bread, Butter\} \rightarrow \{Milk\}
$$

This is called an **association rule**.

---

# 15. Apriori Algorithm

Apriori is a classic association-rule algorithm.

Example transaction data:

```text
T1 → Bread, Milk
T2 → Bread, Butter, Milk
T3 → Bread, Butter
T4 → Milk, Butter
```

The algorithm looks for frequently occurring item combinations.

Important concepts:

### Support

How frequently an itemset occurs.

$$
Support(X)=
\frac{\text{Transactions containing X}}
{\text{Total transactions}}
$$

### Confidence

How often \(Y\) occurs when \(X\) occurs.

$$
Confidence(X\rightarrow Y)
=
\frac{Support(X\cup Y)}
{Support(X)}
$$

### Lift

Measures how much stronger the association is compared with random co-occurrence.

$$
Lift(X\rightarrow Y)
=
\frac{Confidence(X\rightarrow Y)}
{Support(Y)}
$$

---

# 16. Anomaly Detection

Anomaly detection attempts to identify observations that behave differently from the majority.

Example:

```text
• • • • •
 • • • •
• • • • •

                 X
```

The `X` could be an anomaly.

Applications include:

* Credit-card fraud
* Network intrusion
* Machine failure
* Sensor errors
* Manufacturing defects
* Unusual user behavior

Important algorithms include:

* Isolation Forest
* One-Class SVM
* Local Outlier Factor
* DBSCAN
* Autoencoders

---

# 17. Isolation Forest

Isolation Forest is particularly useful for anomaly detection.

Its intuition is interesting:

> Anomalies are easier to isolate than normal observations.

Suppose:

```text
Normal points:

••••••••
••••••••
••••••••

Anomaly:

                    X
```

The isolated point can usually be separated with fewer random splits.

Therefore:

$$
\text{Shorter isolation path}
\Rightarrow
\text{more likely anomaly}
$$

---

# 18. Autoencoders

Autoencoders are neural networks used for representation learning, dimensionality reduction, and anomaly detection.

Basic structure:

```text
Input
  ↓
Encoder
  ↓
Latent representation
  ↓
Decoder
  ↓
Reconstructed input
```

For example:

```text
100 features
     ↓
   Encoder
     ↓
10 features
     ↓
   Decoder
     ↓
100 features
```

The network learns a compact representation of the input.

For anomaly detection, if an unusual sample reconstructs poorly:

$$
Reconstruction\ Error
=
||X-\hat X||^2
$$

A high reconstruction error can indicate an anomaly.

---

# 19. Generative Learning

Another important direction related to unsupervised/self-supervised representation learning is learning the underlying distribution of data.

Examples include:

* Autoencoders
* Variational Autoencoders (VAEs)
* Generative models

The model attempts to learn characteristics of the data distribution and potentially generate new samples.

For example:

```text
Training images
      ↓
 Generative model
      ↓
Learn distribution
      ↓
Generate new image
```

Modern generative AI often uses **self-supervised learning** or related objectives rather than purely classical unsupervised learning, so it's useful to distinguish these concepts.

---

# 20. What Makes Unsupervised Learning Difficult?

The biggest challenge is:

> **There is usually no correct answer provided during training.**

Suppose K-Means creates:

```text
Cluster A
Cluster B
Cluster C
```

What does that mean?

The algorithm doesn't inherently know:

```text
A = rich customers
B = middle-class customers
C = low-income customers
```

You need to interpret the clusters.

This makes unsupervised learning more exploratory than supervised learning.

---

# 21. How Do We Evaluate Unsupervised Learning?

Because labels are absent, evaluation can be difficult.

For clustering, common metrics include:

### Silhouette Score

Measures how well a point fits its own cluster compared with other clusters.

$$
s =
\frac{b-a}{\max(a,b)}
$$

where:

* \(a\) = average distance from the point to its own cluster
* \(b\) = average distance to the nearest other cluster

Range:

$$
-1 \leq s \leq 1
$$

Generally:

* Near 1 → well-separated clustering
* Around 0 → overlapping clusters
* Negative → potentially poor assignment

Other metrics include:

* Davies-Bouldin Index
* Calinski-Harabasz Index
* Adjusted Rand Index — when ground-truth labels are available for evaluation

---

# 22. Important Unsupervised Learning Algorithms

You should organize your study like this:

```text
UNSUPERVISED LEARNING
│
├── 1. Clustering
│   ├── K-Means
│   ├── Hierarchical Clustering
│   ├── DBSCAN
│   ├── Gaussian Mixture Models
│   └── Mean Shift
│
├── 2. Dimensionality Reduction
│   ├── PCA
│   ├── Kernel PCA
│   ├── t-SNE
│   └── UMAP
│
├── 3. Association Rule Learning
│   ├── Apriori
│   └── FP-Growth
│
├── 4. Anomaly Detection
│   ├── Isolation Forest
│   ├── One-Class SVM
│   └── Local Outlier Factor
│
└── 5. Representation Learning
    ├── Autoencoders
    ├── VAE
    └── Modern self-supervised methods
```

---

# 23. Real-World Applications

### Customer Segmentation

```text
Customer data
     ↓
K-Means
     ↓
Customer groups
```

Used for understanding different behavioral segments.

### Recommendation Systems

User/item behavior can be analyzed to discover similarities and latent structures.

### Fraud Detection

```text
Transaction data
       ↓
Anomaly detection
       ↓
Unusual transactions
```

### Image Analysis

```text
Image features
      ↓
Clustering / PCA
      ↓
Groups / compact representation
```

### Network Security

```text
Network traffic
       ↓
Anomaly detection
       ↓
Suspicious behavior
```

### Environmental Data

For a project such as air-quality analysis, unsupervised learning can help discover:

```text
AQI + PM2.5 + PM10 + NO2 + weather
                    ↓
                 Clustering
                    ↓
      Different pollution regimes
```

For example, the algorithm might reveal groups corresponding to different combinations of pollution/weather conditions without those groups being manually labeled.

---

# 24. The Most Important Concept

Don't think of unsupervised learning simply as:

> "ML without labels."

A better way to think about it is:

> **Unsupervised learning attempts to discover hidden structure, relationships, representations, or unusual patterns in data when explicit target labels are not available.**

---

# 25. Recommended Study Order

Since you're currently studying ML systematically, I would recommend this sequence:

### Phase 1 — Clustering

1. **K-Means**
2. K-Means mathematics
3. K-Means implementation from scratch
4. Choosing \(K\)
5. Elbow Method
6. Silhouette Score
7. Limitations of K-Means
8. Hierarchical Clustering
9. Dendrogram
10. DBSCAN
11. DBSCAN mathematics/intuitions
12. K-Means vs Hierarchical vs DBSCAN
13. Gaussian Mixture Models

### Phase 2 — Dimensionality Reduction

14. Curse of Dimensionality
15. PCA intuition
16. PCA mathematics
17. PCA implementation
18. Explained Variance
19. Choosing number of components
20. Kernel PCA
21. t-SNE
22. UMAP

### Phase 3 — Association Rules

23. Association Rule Learning
24. Support
25. Confidence
26. Lift
27. Apriori
28. FP-Growth

### Phase 4 — Anomaly Detection

29. What is anomaly detection?
30. Isolation Forest
31. One-Class SVM
32. Local Outlier Factor
33. Autoencoder-based anomaly detection

### Phase 5 — Advanced

34. Autoencoders
35. Variational Autoencoders
36. Representation Learning
37. Self-Supervised Learning
38. Contrastive Learning

Since you've just been working through **K-Means**, the natural next step is to go **K-Means → Hierarchical Clustering → DBSCAN → GMM**, then move into **PCA**.
