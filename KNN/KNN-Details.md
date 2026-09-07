

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
