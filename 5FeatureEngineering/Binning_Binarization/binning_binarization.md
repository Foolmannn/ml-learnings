# Binning and Binarization in Machine Learning

Binning (also called **discretization**) and **binarization** are important **feature engineering** and **data preprocessing** techniques used in Machine Learning. They transform numerical data into forms that are easier for certain algorithms to understand.

---

# 1. Binning (Discretization)

## Definition

**Binning** is the process of converting continuous numerical values into a finite number of intervals (bins).

Instead of storing the exact value, we store the interval to which the value belongs.

For example

| Age |
| --- |
| 18  |
| 21  |
| 25  |
| 30  |
| 35  |
| 42  |
| 50  |
| 60  |

After binning

| Age | Age Group  |
| --- | ---------- |
| 18  | Young      |
| 21  | Young      |
| 25  | Young      |
| 30  | Adult      |
| 35  | Adult      |
| 42  | Middle Age |
| 50  | Middle Age |
| 60  | Senior     |

So continuous data becomes categorical data.

---

# Why do we use Binning?

Binning helps when

* Reducing noise
* Handling outliers
* Making data easier to interpret
* Improving performance of some algorithms
* Capturing non-linear relationships

Suppose income is

```
32000
33000
32500
32600
32100
5000000
```

The last value is an outlier.

Instead of using raw values, we create bins

```
Low Income
Medium Income
High Income
```

Now the outlier has much less influence.

---

# Advantages

* Reduces overfitting
* Handles outliers
* Makes visualization easier
* Can improve simple models
* Easier interpretation

---

# Disadvantages

* Loss of information
* Choice of bins is difficult
* Different binning methods give different results

---

# Types of Binning

There are mainly two categories.

## 1. Unsupervised Binning

Does not use target labels.

Examples

* Equal Width
* Equal Frequency
* K-Means Binning

---

## 2. Supervised Binning

Uses target labels while creating bins.

Examples

* Decision Tree based discretization
* MDLP
* ChiMerge

These often produce better predictive performance because they consider the relationship with the target.

---

# Equal Width Binning

The simplest method.

The whole range is divided into equal-sized intervals.

Formula

[
\text{Bin Width} = \frac{\text{Maximum}-\text{Minimum}}{\text{Number of Bins}}
]

Example

Suppose

```
Marks

12
20
28
35
41
48
55
63
72
90
```

Minimum

```
12
```

Maximum

```
90
```

Range

```
90 - 12 = 78
```

Suppose we want 3 bins

```
Bin Width = 78 / 3 = 26
```

Bins become

```
12–38
39–64
65–90
```

Result

| Marks | Bin |
| ----- | --- |
| 12    | 1   |
| 20    | 1   |
| 28    | 1   |
| 35    | 1   |
| 41    | 2   |
| 48    | 2   |
| 55    | 2   |
| 63    | 2   |
| 72    | 3   |
| 90    | 3   |

---

### Python Example

```python
import pandas as pd

marks = [12,20,28,35,41,48,55,63,72,90]

df = pd.DataFrame({"Marks": marks})

df["Bin"] = pd.cut(df["Marks"], bins=3)

print(df)
```

Output

```
Marks      Bin

12   (11.9,38]
20   (11.9,38]
28   (11.9,38]
...
```

---

# Equal Frequency Binning

Instead of equal width,

Each bin contains approximately the same number of observations.

Example

```
Marks

10
15
18
25
30
40
45
50
80
100
```

Suppose 5 bins.

Each bin contains

```
10/5 = 2 observations
```

Bins

```
10 15

18 25

30 40

45 50

80 100
```

Notice

Widths are different.

---

### Python

```python
df["Bin"] = pd.qcut(df["Marks"], q=5)
```

Here

```
q = number of quantiles
```

---

# Difference

Equal Width

```
|----|----|----|
```

Equal Frequency

```
|-|-|------|-|----------|
```

Widths vary.

---

# K-Means Binning

Instead of manually creating bins,

K-Means clustering groups similar values.

Example

```
Income

25000
26000
28000
60000
62000
65000
200000
210000
```

K-Means may produce

```
Low

Medium

High
```

Unlike equal-width binning, the boundaries adapt to the data distribution.

---

# Decision Tree Binning

Decision Trees automatically determine the best split points.

Suppose

```
Age

18
20
22
25
28
40
45
50
60
```

Tree might split

```
Age <= 30

Age > 30
```

This is supervised because it uses the target variable.

---

# Using sklearn KBinsDiscretizer

Scikit-Learn provides

```python
from sklearn.preprocessing import KBinsDiscretizer
```

Example

```python
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer

X = np.array([[12],
              [20],
              [28],
              [35],
              [41],
              [48],
              [55],
              [63],
              [72],
              [90]])

kbd = KBinsDiscretizer(
    n_bins=3,
    encode='ordinal',
    strategy='uniform'
)

X_new = kbd.fit_transform(X)

print(X_new)
```

Output

```
[[0.]
 [0.]
 [0.]
 [0.]
 [1.]
 [1.]
 [1.]
 [1.]
 [2.]
 [2.]]
```

---

# Parameters

```python
KBinsDiscretizer(
    n_bins=5,
    encode='ordinal',
    strategy='uniform'
)
```

### n_bins

Number of bins

```
5
```

---

### strategy

Three options

### uniform

Equal Width

---

### quantile

Equal Frequency

---

### kmeans

K-Means Binning

---

### encode

Three options

#### ordinal

```
0
1
2
```

---

#### onehot

```
1 0 0

0 1 0

0 0 1
```

Sparse matrix.

---

#### onehot-dense

Dense matrix.

---

# Visual Example

Original Data

```
5 8 12 18 25 31 45 60 72 90
```

Uniform

```
|-------|-------|-------|

0       1       2
```

Quantile

```
|---|------|-----------|

0   1       2
```

KMeans

```
Clusters are created automatically.
```

---

# When Should You Use Binning?

Use it when

* Data contains noise.
* There are many outliers.
* The relationship with the target is non-linear.
* You want easier interpretation (e.g., age groups).
* A model benefits from categorical inputs (or you plan to one-hot encode the bins).

Avoid it when

* Precise numerical values matter.
* You have little data, because binning discards information.
* You're using models like neural networks that often benefit from continuous values.

---

# 2. Binarization

## Definition

**Binarization** converts numerical values into only **two values**, typically **0 and 1**, based on a threshold.

General rule:

[
x' =
\begin{cases}
1 & \text{if } x > \text{threshold}\
0 & \text{otherwise}
\end{cases}
]

---

# Example

Marks

```
20
35
50
65
80
95
```

Threshold = 60

Result

| Marks | Binary |
| ----- | ------ |
| 20    | 0      |
| 35    | 0      |
| 50    | 0      |
| 65    | 1      |
| 80    | 1      |
| 95    | 1      |

---

# Why Use Binarization?

* Convert a feature into a yes/no indicator.
* Simplify model inputs.
* Handle sparse or count data.
* Create interpretable features.

Examples:

* Purchased? (0/1)
* Passed exam? (0/1)
* Income above ₹50,000? (0/1)
* Clicked ad? (0/1)

---

# Scikit-Learn Binarizer

```python
from sklearn.preprocessing import Binarizer
```

Example

```python
import numpy as np
from sklearn.preprocessing import Binarizer

X = np.array([
    [20],
    [35],
    [50],
    [65],
    [80],
    [95]
])

binarizer = Binarizer(threshold=60)

X_new = binarizer.fit_transform(X)

print(X_new)
```

Output

```
[[0.]
 [0.]
 [0.]
 [1.]
 [1.]
 [1.]]
```

---

# Multiple Features

```python
X = np.array([
    [2, 100],
    [5, 80],
    [7, 150]
])

binarizer = Binarizer(threshold=5)

print(binarizer.fit_transform(X))
```

Output

```
[[0. 1.]
 [0. 1.]
 [1. 1.]]
```

The threshold is applied independently to every value.

---

# Choosing the Threshold

The threshold can come from:

* Domain knowledge (e.g., age ≥ 18 means adult).
* Statistical measures (mean, median, percentile).
* Business rules (credit score > 700).
* Hyperparameter tuning.

Choosing a poor threshold can reduce model performance.

---

# Binarization vs One-Hot Encoding

| Binarization                                      | One-Hot Encoding                                          |
| ------------------------------------------------- | --------------------------------------------------------- |
| Converts numeric values to 0/1 using a threshold. | Converts categorical values into multiple binary columns. |
| Produces one binary feature per original feature. | Produces one binary feature per category.                 |
| Used for continuous or count data.                | Used for categorical data.                                |

Example:

Age → Binarization (threshold = 18)

| Age | Adult |
| --: | ----: |
|  15 |     0 |
|  22 |     1 |

Color → One-Hot Encoding

| Color | Red | Blue | Green |
| ----- | --: | ---: | ----: |
| Red   |   1 |    0 |     0 |
| Blue  |   0 |    1 |     0 |

---

# Binning vs Binarization

| Feature          | Binning                             | Binarization                        |
| ---------------- | ----------------------------------- | ----------------------------------- |
| Output           | Multiple intervals                  | Two values (0/1)                    |
| Data type        | Usually categorical or ordinal      | Binary                              |
| Number of groups | 2 or more                           | Exactly 2                           |
| Typical use      | Group continuous values into ranges | Create yes/no indicators            |
| Example          | Age → Child, Adult, Senior          | Age ≥ 18 → Adult (1), otherwise (0) |

---

# Best Practices

* Use **binning** when ranges are meaningful (e.g., income brackets, age groups) or when you want to reduce the impact of noise and outliers.
* Use **binarization** when a binary decision boundary is meaningful (e.g., passed/failed, active/inactive).
* If you use binning with linear models, consider **one-hot encoding** the bins instead of treating them as ordinal unless the order itself carries meaningful information.
* Fit preprocessing steps (like `KBinsDiscretizer`) on the **training data only**, then apply the learned transformation to validation and test data to avoid data leakage.
* Experiment with different numbers of bins and strategies (`uniform`, `quantile`, `kmeans`) using cross-validation, as the best choice depends on your data and model.

In summary, **binning** divides continuous values into multiple intervals to simplify or stabilize features, while **binarization** reduces a feature to a simple yes/no representation based on a threshold. Both are valuable preprocessing techniques, but they serve different purposes and should be chosen according to the problem and the learning algorithm.
