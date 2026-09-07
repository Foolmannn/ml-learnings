

# K-Nearest Neighbors (KNN) — In Detail

**K-Nearest Neighbors (KNN)** is one of the simplest and most intuitive supervised machine-learning algorithms. It can be used for both:

- **Classification** → predict a class/category
- **Regression** → predict a continuous value

The core idea is:

> **Similar data points tend to have similar outputs.**

For a new data point, KNN looks at the **K closest training examples** and uses them to make a prediction. ([scikit-learn](https://scikit-learn.org/stable/modules/neighbors.html?utm_source=chatgpt.com))

---

# 1. What is KNN?

Suppose you have data about fruits:

| Weight | Sweetness | Fruit |
|---:|---:|---|
| 150 | 8 | Apple |
| 160 | 7 | Apple |
| 170 | 8 | Apple |
| 300 | 4 | Orange |
| 320 | 3 | Orange |
| 310 | 5 | Orange |

Now a new fruit arrives:

```text
Weight = 165
Sweetness = 7.5
```

We don't know whether it is an Apple or Orange.

KNN asks:

> "Which existing fruits are closest to this new fruit?"

If the closest 3 fruits are:

```text
Apple
Apple
Apple
```

then:

```text
Prediction = Apple
```

That's essentially KNN.

---

# 2. Why is it called K-Nearest Neighbors?

The name tells you the algorithm:

### K

Number of neighbors we want to consider.

For example:

```text
K = 3
```

means:

> Look at the 3 closest training points.

### Nearest

We need some way to measure **distance**.

### Neighbors

The closest training examples are the neighbors.

---

# 3. KNN is a Lazy Learning Algorithm

This is an extremely important concept.

Many ML algorithms learn a mathematical model during training.

For example:

### Linear Regression

It learns:

$$
y = \beta_0 + \beta_1x
$$

### Logistic Regression

It learns:

$$
P(y=1|x)=\sigma(\beta_0+\beta_1x)
$$

But KNN doesn't really construct such a predictive model.

It essentially **stores the training data** and waits until prediction time to find neighboring samples. Scikit-learn describes nearest-neighbor methods as "non-generalizing" methods because they remember the training instances rather than learning a conventional parametric model. ([scikit-learn](https://scikit-learn.org/stable/modules/neighbors.html?utm_source=chatgpt.com))

Therefore:

> **KNN has very little training computation but potentially expensive prediction.**

This is called:

**Lazy Learning / Instance-Based Learning / Memory-Based Learning**

---

# 4. How KNN Works

Suppose we have:

```text
       Feature 2
           ↑
       A   A
     A
                  B
             B
          B
           └────────────→ Feature 1
```

Now we receive:

```text
        ?
```

We calculate the distance between `?` and all training points.

Then:

1. Calculate distances
2. Sort the distances
3. Select the K closest points
4. Make prediction

For classification:

> Majority vote

For regression:

> Average of neighbors

Scikit-learn uses majority voting for classification and the mean of neighboring target values for regression. ([scikit-learn](https://scikit-learn.org/stable/modules/neighbors.html?utm_source=chatgpt.com))

---

# 5. Step-by-Step Example

Suppose:

| Point | X | Y | Class |
|---|---:|---:|---|
| A | 1 | 1 | Red |
| B | 2 | 2 | Red |
| C | 3 | 3 | Red |
| D | 6 | 5 | Blue |
| E | 7 | 7 | Blue |

New point:

$$
P=(4,4)
$$

Let's use:

$$
K=3
$$

Calculate distances.

---

# 6. Euclidean Distance

The most commonly used distance is **Euclidean distance**.

For two points:

$$
A=(x_1,y_1)
$$

and

$$
B=(x_2,y_2)
$$

distance is:

$$
d(A,B)=\sqrt{(x_2-x_1)^2+(y_2-y_1)^2}
$$

For multiple features:

$$
d(x,y)=
\sqrt{
\sum_{i=1}^{n}(x_i-y_i)^2
}
$$

For our point:

$$
P=(4,4)
$$

Distance from A:

$$
d(P,A)=
\sqrt{(4-1)^2+(4-1)^2}
$$

$$
=\sqrt{9+9}
$$

$$
=\sqrt{18}
\approx4.24
$$

Distance from B:

$$
d(P,B)=
\sqrt{(4-2)^2+(4-2)^2}
$$

$$
=\sqrt8
\approx2.83
$$

Distance from C:

$$
d(P,C)=
\sqrt{(4-3)^2+(4-3)^2}
$$

$$
=\sqrt2
\approx1.41
$$

Distance from D:

$$
d(P,D)=
\sqrt{(4-6)^2+(4-5)^2}
$$

$$
=\sqrt5
\approx2.24
$$

Distance from E:

$$
d(P,E)=
\sqrt{(4-7)^2+(4-7)^2}
$$

$$
=\sqrt{18}
\approx4.24
$$

So:

| Point | Distance | Class |
|---|---:|---|
| C | 1.41 | Red |
| D | 2.24 | Blue |
| B | 2.83 | Red |
| A | 4.24 | Red |
| E | 4.24 | Blue |

For:

$$
K=3
$$

neighbors are:

```text
C → Red
D → Blue
B → Red
```

Vote:

```text
Red = 2
Blue = 1
```

Therefore:

$$
\boxed{\text{Prediction = Red}}
$$

---

# 7. KNN Classification

For classification, KNN uses the **majority class** among the nearest neighbors.

Suppose:

```text
K = 5
```

Nearest neighbors:

```text
A
A
B
A
B
```

Votes:

```text
A → 3
B → 2
```

Therefore:

$$
\boxed{Prediction=A}
$$

The standard `KNeighborsClassifier` in scikit-learn implements this nearest-neighbor voting approach. ([scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier?utm_source=chatgpt.com))

---

# 8. KNN Regression

KNN can also predict numerical values.

Suppose:

| Neighbor | Distance | House Price |
|---|---:|---:|
| A | 1.0 | 200 |
| B | 1.5 | 220 |
| C | 2.0 | 240 |

For:

$$
K=3
$$

prediction:

$$
\hat y=
\frac{200+220+240}{3}
$$

$$
\boxed{\hat y=220}
$$

So:

### Classification

Use:

$$
\boxed{\text{Majority Vote}}
$$

### Regression

Use:

$$
\boxed{\text{Mean}}
$$

Scikit-learn's `KNeighborsRegressor` follows this mean-based approach by default. ([scikit-learn](https://scikit-learn.org/stable/modules/neighbors.html?utm_source=chatgpt.com))

---

# 9. Choosing K

Choosing `K` is one of the most important decisions in KNN.

Suppose:

```text
K = 1
```

The algorithm only looks at the closest point.

With:

```text
K = 20
```

it considers a much larger neighborhood.

This creates a **bias-variance tradeoff**.

---

# 10. K = 1

Suppose:

```text
K = 1
```

The closest point completely determines the prediction.

### Advantages

- Very flexible
- Can capture complex decision boundaries

### Disadvantages

- Very sensitive to noise
- Can overfit

So:

$$
K=1
$$

usually has:

$$
\boxed{\text{Low Bias + High Variance}}
$$

---

# 11. Large K

Suppose:

```text
K = 100
```

The prediction is based on many points.

This makes the model smoother.

### Advantages

- Less sensitive to noise
- Lower variance

### Disadvantages

- Can oversimplify the problem
- Can underfit

Therefore:

$$
\boxed{\text{Large K → Higher Bias + Lower Variance}}
$$

Scikit-learn similarly notes that larger `k` tends to suppress noise but produces less distinct decision boundaries. ([scikit-learn](https://scikit-learn.org/stable/modules/neighbors.html?utm_source=chatgpt.com))

---

# 12. Underfitting vs Overfitting

A useful mental model:

```text
Small K
   ↓
Very flexible
   ↓
Overfitting
```

while:

```text
Large K
   ↓
Very smooth
   ↓
Underfitting
```

So we want an appropriate middle value.

---

# 13. How Do We Choose the Best K?

Usually using **cross-validation**.

For example:

```text
K = 1  → accuracy = 0.82
K = 3  → accuracy = 0.88
K = 5  → accuracy = 0.91
K = 7  → accuracy = 0.89
K = 9  → accuracy = 0.87
```

Choose:

$$
\boxed{K=5}
$$

because it gives the best validation performance.

In practice, don't simply assume that odd `K` is always best; odd values can reduce ties in binary classification, but the best `K` remains data-dependent.

---

# 14. Why Feature Scaling is Extremely Important

This is one of the **most important things to understand about KNN**.

Suppose we have:

| Age | Salary |
|---:|---:|
| 20 | 30,000 |
| 25 | 40,000 |
| 30 | 50,000 |

Consider:

```text
Age = 30
Salary = 50,000
```

Distance might look like:

$$
d =
\sqrt{
(30-25)^2+
(50000-40000)^2
}
$$

$$
=
\sqrt{
25+100000000
}
$$

The salary dominates the distance.

So KNN might effectively ignore Age.

---

# 15. Solution: Feature Scaling

Usually use:

### Standardization

$$
z=\frac{x-\mu}{\sigma}
$$

using `StandardScaler`.

Or:

### Min-Max Scaling

$$
x'=
\frac{x-x_{min}}
{x_{max}-x_{min}}
$$

using `MinMaxScaler`.

For KNN, scaling is usually important because the algorithm relies directly on distances.

---

# 16. KNN + StandardScaler

Typical workflow:

```python
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(X_train_scaled, y_train)

y_pred = knn.predict(X_test_scaled)
```

Notice:

```python
scaler.fit_transform(X_train)
```

but:

```python
scaler.transform(X_test)
```

Do **not** fit the scaler separately on test data.

---

# 17. Different Distance Metrics

KNN doesn't have to use only Euclidean distance.

Scikit-learn supports multiple distance metrics, with the default being Minkowski distance with `p=2`, which corresponds to Euclidean distance. ([scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html?utm_source=chatgpt.com))

---

## 17.1 Euclidean Distance

$$
d=
\sqrt{
\sum_i(x_i-y_i)^2
}
$$

Example:

```text
metric="euclidean"
```

Good general-purpose choice.

---

# 18. Manhattan Distance

Also called:

$$
L_1
$$

distance.

Formula:

$$
d=
\sum_i|x_i-y_i|
$$

For two dimensions:

$$
d=
|x_1-y_1|+
|x_2-y_2|
$$

Example:

```python
KNeighborsClassifier(
    n_neighbors=5,
    metric="manhattan"
)
```

---

# 19. Minkowski Distance

Minkowski is a generalization.

$$
d=
\left(
\sum_i |x_i-y_i|^p
\right)^{1/p}
$$

When:

$$
p=1
$$

we get Manhattan distance.

When:

$$
p=2
$$

we get Euclidean distance.

In scikit-learn, `p=2` is the default when using the Minkowski metric. ([scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier?utm_source=chatgpt.com))

---

# 20. Distance-Weighted KNN

Normally, all K neighbors have equal influence.

For example:

```text
Neighbor 1 → distance 1
Neighbor 2 → distance 5
Neighbor 3 → distance 10
```

With uniform weighting:

```text
1 → equal importance
5 → equal importance
10 → equal importance
```

But that doesn't always make sense.

The point at distance `1` should arguably matter more.

So we can use:

```python
weights="distance"
```

Scikit-learn implements distance weighting using greater influence for closer points. ([scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier?utm_source=chatgpt.com))

Conceptually:

$$
w_i=\frac{1}{d_i}
$$

Therefore:

```text
closer point
     ↓
higher weight
     ↓
greater influence
```

---

# 21. Uniform vs Distance Weighting

### Uniform

```python
weights="uniform"
```

Every neighbor has equal influence.

### Distance

```python
weights="distance"
```

Closer neighbors have greater influence.

Example:

```python
knn = KNeighborsClassifier(
    n_neighbors=5,
    weights="distance"
)
```

---

# 22. Important KNN Hyperparameters

For `KNeighborsClassifier`, the important parameters include `n_neighbors`, `weights`, `algorithm`, `leaf_size`, `p`, `metric`, and `n_jobs`. ([scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier?utm_source=chatgpt.com))

| Parameter | Meaning |
|---|---|
| `n_neighbors` | Number of neighbors |
| `weights` | How neighbors influence prediction |
| `metric` | Distance metric |
| `p` | Minkowski power |
| `algorithm` | Neighbor search algorithm |
| `leaf_size` | Tree optimization parameter |
| `n_jobs` | Parallel processing |

---

# 23. `n_neighbors`

```python
KNeighborsClassifier(n_neighbors=5)
```

Controls K.

Typical values to test:

```text
1
3
5
7
9
11
15
21
...
```

But don't blindly choose these. Use validation/cross-validation.

---

# 24. `weights`

```python
weights="uniform"
```

or:

```python
weights="distance"
```

### Uniform

All neighbors equal.

### Distance

Closer neighbors have greater influence.

---

# 25. `metric`

Examples:

```python
metric="euclidean"
```

```python
metric="manhattan"
```

```python
metric="minkowski"
```

There are also other metrics available depending on the data and implementation. ([scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html?utm_source=chatgpt.com))

---

# 26. `p`

Used with Minkowski distance.

```python
p=1
```

means Manhattan.

```python
p=2
```

means Euclidean.

Example:

```python
knn = KNeighborsClassifier(
    n_neighbors=5,
    p=2
)
```

---

# 27. `algorithm`

Scikit-learn provides:

```text
auto
ball_tree
kd_tree
brute
```

([scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier?utm_source=chatgpt.com))

### `brute`

Calculate distances against points directly.

### `KDTree`

Uses a spatial tree structure.

### `BallTree`

Another tree-based spatial structure.

### `auto`

Scikit-learn chooses an appropriate approach.

Usually:

```python
algorithm="auto"
```

is a reasonable starting point.

---

# 28. Why Does KNN Have Expensive Prediction?

Imagine:

```text
Training samples = 1,000,000
```

For a new sample, the algorithm may need to find its nearest neighbors among a huge number of training points.

Therefore:

### Training

Very cheap.

### Prediction

Potentially expensive.

This is almost the opposite of many parametric algorithms.

---

# 29. Brute Force KNN

The simplest approach:

```text
New point
    ↓
Calculate distance to point 1
Calculate distance to point 2
Calculate distance to point 3
...
Calculate distance to point N
    ↓
Sort
    ↓
Choose K
```

For large datasets this can become expensive.

---

# 30. KD Tree

A **KD Tree** organizes points into a tree structure to make neighbor searches faster in suitable low-dimensional settings.

Conceptually:

```text
             Root
           /      \
        Region   Region
        /   \     /   \
       ...  ...  ...  ...
```

Instead of checking every point, some regions can be eliminated from consideration.

---

# 31. Ball Tree

Ball Tree organizes data into nested regions, often represented as hyperspheres/balls.

Conceptually:

```text
        Large Ball
       /          \
   Ball A        Ball B
   /   \         /   \
 ...
```

It can be useful for certain metrics and datasets.

---

# 32. Curse of Dimensionality

This is a **major issue with KNN**.

Suppose:

```text
2 features
```

Distance works nicely.

But now imagine:

```text
1000 features
```

Distances become less informative.

As dimensionality increases:

- Data becomes sparse
- Distances become less discriminative
- Nearest and farthest points can become relatively similar
- Computational cost increases

This is known as the:

$$
\boxed{\text{Curse of Dimensionality}}
$$

Scikit-learn specifically notes that neighbor methods become less effective in high-dimensional parameter spaces. ([scikit-learn](https://scikit-learn.org/stable/modules/neighbors.html?utm_source=chatgpt.com))

---

# 33. Why High Dimensions Hurt KNN

Imagine:

```text
2D

● ● ●
  ●
● ● ●
```

There are plenty of nearby points.

Now imagine:

```text
100 dimensions
```

The space becomes enormous.

Your data might look like:

```text
●                       ●

          ●

                         ●

     ●
```

The concept of "nearby" becomes less meaningful.

This is one reason KNN is often better for **low-to-moderate dimensional datasets**.

---

# 34. KNN and Outliers

KNN can be affected by outliers, especially when:

```text
K is small
```

For example:

```text
A A A A A A
       X
```

If `X` is an unusual point close to a query point, it can influence the prediction.

Using a somewhat larger K or distance weighting can sometimes help, but the best solution depends on the dataset.

---

# 35. KNN Decision Boundary

One interesting property of KNN is that it can create **nonlinear decision boundaries**.

For example, suppose the classes look like:

```text
      A A A
   A       A
  A    B    A
   A       A
      A A A
```

A linear classifier may struggle.

KNN can naturally create irregular boundaries because it makes local decisions.

This is one reason nearest-neighbor methods can work well when the decision boundary is highly irregular. ([scikit-learn](https://scikit-learn.org/stable/modules/neighbors.html?utm_source=chatgpt.com))

---

# 36. KNN vs Logistic Regression

| Property | KNN | Logistic Regression |
|---|---|---|
| Learning style | Instance-based | Parametric |
| Training | Very fast | Moderate |
| Prediction | Can be slow | Fast |
| Scaling important | Yes | Usually yes |
| Nonlinear boundary | Naturally possible | Requires transformations |
| Interpretability | Moderate | High |
| High dimensions | Often problematic | Can work better |
| Large datasets | Can struggle | Generally better |

---

# 37. KNN vs Decision Tree

| Property | KNN | Decision Tree |
|---|---|---|
| Scaling | Important | Usually unnecessary |
| Training | Very fast | Fast |
| Prediction | Potentially expensive | Fast |
| Nonlinear | Yes | Yes |
| Interpretability | Moderate | High |
| High dimensionality | Problematic | Often better |
| Outliers | Can affect | Depends |

---

# 38. KNN vs SVM

| Property | KNN | SVM |
|---|---|---|
| Training | Easy | More computational |
| Prediction | Potentially expensive | Usually efficient after training |
| Scaling | Important | Important |
| Nonlinear | Naturally local | With kernels |
| Large datasets | Can struggle | Can also struggle depending on setup |
| Interpretability | Moderate | Low/Moderate |

---

# 39. KNN Classification in Scikit-Learn

Here's the basic implementation:

```python
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(
    n_neighbors=5
)

knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)
```

---

# 40. Complete KNN Pipeline

For real projects, I'd recommend a pipeline:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier(n_neighbors=5))
])

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
```

This is cleaner because scaling and KNN become one workflow.

---

# 41. KNN Regression

```python
from sklearn.neighbors import KNeighborsRegressor

knn = KNeighborsRegressor(
    n_neighbors=5
)

knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)
```

For regression:

```python
weights="uniform"
```

means the neighbors contribute equally, while:

```python
weights="distance"
```

gives greater influence to nearby points. ([scikit-learn](https://scikit-learn.org/stable/modules/neighbors.html?utm_source=chatgpt.com))

---

# 42. KNN with Pipeline for Regression

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor

model = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsRegressor(
        n_neighbors=5,
        weights="distance"
    ))
])

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
```

---

# 43. Finding the Best K

You can use `GridSearchCV`.

```python
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier())
])

param_grid = {
    "knn__n_neighbors": [3, 5, 7, 9, 11, 15, 21],
    "knn__weights": ["uniform", "distance"],
    "knn__metric": ["euclidean", "manhattan"]
}

grid = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring="accuracy"
)

grid.fit(X_train, y_train)

print(grid.best_params_)
print(grid.best_score_)
```

This is much better than manually guessing K.

---

# 44. KNN Probability

KNN can also produce class probabilities.

Example:

```python
knn.predict_proba(X_test)
```

Suppose:

```text
K = 5
```

and neighbors are:

```text
Class A → 4
Class B → 1
```

Then approximately:

$$
P(A)=\frac45=0.8
$$

$$
P(B)=\frac15=0.2
$$

Scikit-learn's `predict_proba()` returns class probability estimates based on the neighbor voting process. ([scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier?utm_source=chatgpt.com))

---

# 45. KNN from Scratch

Understanding the algorithm manually is extremely useful.

```python
import numpy as np

class KNN:

    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):

        predictions = []

        for x in X:

            distances = np.sqrt(
                np.sum(
                    (self.X_train - x) ** 2,
                    axis=1
                )
            )

            k_indices = np.argsort(distances)[:self.k]

            k_labels = self.y_train[k_indices]

            values, counts = np.unique(
                k_labels,
                return_counts=True
            )

            prediction = values[np.argmax(counts)]

            predictions.append(prediction)

        return np.array(predictions)
```

The important part is:

```python
distances = ...
```

then:

```python
np.argsort(distances)
```

then:

```python
[:self.k]
```

then majority voting.

---
