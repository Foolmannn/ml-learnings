# Outlier Detection and Removal in Machine Learning (Complete Guide)

Outliers are one of the most important aspects of data preprocessing. A few extreme values can significantly affect statistical analysis and machine learning models.

---

# What is an Outlier?

An **outlier** is an observation that is significantly different from the majority of the data.

Example dataset:

```
Age:
21, 22, 23, 24, 25, 26, 27, 28, 150
```

Here,

```
150
```

is clearly an outlier.

Graphically

```
Normal Data

        ●
      ● ● ●
    ● ● ● ● ●
  ● ● ● ● ● ●

Outlier

                      ●
```

---

# Why Outliers Occur

Outliers may occur due to

### 1. Measurement Error

Example

A sensor records

```
Temperature

25
26
27
250   ← sensor error
28
```

---

### 2. Data Entry Error

Salary

```
45000
47000
52000
5000000   ← typing mistake
```

---

### 3. Natural Variation

Suppose you're measuring human heights.

Most people:

```
150–190 cm
```

Someone:

```
220 cm
```

is unusual but valid.

---

### 4. Fraudulent Activities

Example

Credit card transactions

```
Normal:
$10
$25
$100

Fraud:
$50,000
```

---

# Should We Always Remove Outliers?

**No.**

This is one of the biggest misconceptions.

Remove them only if

* measurement error
* impossible value
* data entry mistake

Do **NOT** remove if they are genuine observations.

Example

House prices

```
200k
250k
300k
15 million
```

If the house is actually a mansion, this is **not** an error.

---

# Types of Outliers

## 1. Univariate Outlier

Only one feature.

Example

```
Age
```

---

## 2. Multivariate Outlier

Combination of variables.

Example

```
Age = 12
Salary = $500,000
```

Age alone isn't strange.

Salary alone isn't strange.

Together they are impossible.

---

# Which Models Are Sensitive?

Very Sensitive

* Linear Regression
* Logistic Regression
* KNN
* K-Means
* PCA
* Neural Networks (sometimes)

Moderately Sensitive

* SVM

Less Sensitive

* Decision Trees
* Random Forest
* XGBoost
* LightGBM
* CatBoost

Tree-based models split data rather than relying on distances or means, making them more robust to outliers.

---

# Detecting Outliers

There are many methods.

## Method 1: Box Plot

Most common.

```python
import seaborn as sns

sns.boxplot(x=df["Age"])
```

A boxplot looks like

```
        |
        |
   -----|------
  |            |
  |            |
   -------------
        |
        |

      ●    ← Outlier
```

---

# Understanding Box Plot

```
Minimum

Q1

Median

Q3

Maximum
```

```
Q1 ---------------- Q2 ---------------- Q3
```

Where

```
Q2 = Median
```

---

# IQR (Interquartile Range)

Most famous technique.

```
IQR = Q3 − Q1
```

Suppose

```
Q1 = 20

Q3 = 40
```

Then

```
IQR = 20
```

Lower limit

```
Q1 − 1.5 × IQR

20 − 30

= -10
```

Upper limit

```
Q3 + 1.5 × IQR

40 + 30

= 70
```

Any value outside

```
[-10,70]
```

is an outlier.

---

# Python Implementation

```python
Q1 = df["Age"].quantile(0.25)
Q3 = df["Age"].quantile(0.75)

IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

outliers = df[(df["Age"] < lower) | (df["Age"] > upper)]

print(outliers)
```

---

# Removing Them

```python
df = df[(df["Age"] >= lower) &
        (df["Age"] <= upper)]
```

---

# Capping Instead of Removing

Instead of deleting,

replace with boundary values.

```python
df["Age"] = df["Age"].clip(lower, upper)
```

Example

Original

```
10
20
25
30
200
```

After clipping

```
10
20
25
30
70
```

This technique is called **Winsorization**.

---

# Method 2: Z-Score

Works when data follows a **normal distribution**.

Formula

```
Z = (x − μ) / σ
```

Where

```
μ = mean

σ = standard deviation
```

Interpretation

```
|Z| > 3

→ Outlier
```

---

Example

Mean

```
100
```

Std

```
10
```

Value

```
135
```

```
Z=(135−100)/10

=3.5
```

Outlier.

---

Python

```python
from scipy import stats

z = stats.zscore(df["Age"])

outliers = df[abs(z) > 3]
```

Removing

```python
df = df[abs(z) <= 3]
```

---

# Why ±3?

For a normal distribution:

* 68% of data lies within ±1σ
* 95% within ±2σ
* 99.7% within ±3σ

Anything beyond ±3σ is very rare, so it is often treated as an outlier.

---

# Method 3: Modified Z-Score (Median Absolute Deviation)

Better for skewed data.

Formula

```
Modified Z = 0.6745 × (x − Median) / MAD
```

where

```
MAD = median(|x − median|)
```

Threshold

```
|Modified Z| > 3.5
```

Python

```python
from scipy.stats import median_abs_deviation

median = df["Age"].median()
mad = median_abs_deviation(df["Age"])

modified_z = 0.6745 * (df["Age"] - median) / mad

outliers = df[abs(modified_z) > 3.5]
```

---

# Method 4: Percentile Method

Suppose

```
Top 1%

Bottom 1%
```

are extreme.

```python
lower = df["Age"].quantile(0.01)
upper = df["Age"].quantile(0.99)

df = df[(df["Age"] >= lower) &
        (df["Age"] <= upper)]
```

Common thresholds are 1st/99th or 5th/95th percentiles.

---

# Method 5: Isolation Forest

Used for high-dimensional datasets.

Idea

Normal observations require many random splits to isolate.

Outliers require very few.

```python
from sklearn.ensemble import IsolationForest

iso = IsolationForest(
    contamination=0.05,
    random_state=42
)

pred = iso.fit_predict(df)

# 1  -> normal
# -1 -> outlier

df_clean = df[pred == 1]
```

Key parameters:

* `contamination`: expected proportion of outliers (e.g., 0.05 for 5%)
* `random_state`: for reproducibility

Works well for large datasets and many features.

---

# Method 6: Local Outlier Factor (LOF)

Density-based.

A point is an outlier if its local density is much lower than that of its neighbors.

```python
from sklearn.neighbors import LocalOutlierFactor

lof = LocalOutlierFactor(n_neighbors=20)

pred = lof.fit_predict(df)

df_clean = df[pred == 1]
```

Suitable for datasets where outliers are local rather than global.

---

# Method 7: DBSCAN

A clustering algorithm that labels points in sparse regions as noise.

```python
from sklearn.cluster import DBSCAN

db = DBSCAN(eps=0.5, min_samples=5)

labels = db.fit_predict(df)

# Noise points
outliers = df[labels == -1]
```

---

# Method 8: Elliptic Envelope

Assumes data follows a multivariate Gaussian distribution.

```python
from sklearn.covariance import EllipticEnvelope

ee = EllipticEnvelope(
    contamination=0.05,
    random_state=42
)

pred = ee.fit_predict(df)

df_clean = df[pred == 1]
```

---

# Visualization

## Histogram

```python
import matplotlib.pyplot as plt

plt.hist(df["Age"], bins=30)
plt.show()
```

---

## Boxplot

```python
sns.boxplot(x=df["Age"])
```

---

## Scatter Plot

```python
plt.scatter(df["Height"], df["Weight"])
plt.xlabel("Height")
plt.ylabel("Weight")
plt.show()
```

Great for identifying multivariate outliers.

---

# Pipeline Example

**Important:** Fit any outlier detection thresholds using only the training data to avoid data leakage.

```python
from sklearn.model_selection import train_test_split

X_train, X_test = train_test_split(
    df,
    test_size=0.2,
    random_state=42
)

Q1 = X_train["Age"].quantile(0.25)
Q3 = X_train["Age"].quantile(0.75)
IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

X_train = X_train[
    (X_train["Age"] >= lower) &
    (X_train["Age"] <= upper)
]

# Optionally clip the test set using the same boundaries
X_test["Age"] = X_test["Age"].clip(lower, upper)
```

---

# Advantages and Disadvantages

| Method            | Advantages                                  | Disadvantages                        |
| ----------------- | ------------------------------------------- | ------------------------------------ |
| IQR               | Simple, robust to skewness                  | Univariate only                      |
| Z-Score           | Easy, fast                                  | Assumes normality                    |
| Modified Z        | Robust to outliers and skew                 | Univariate                           |
| Percentile        | Flexible                                    | Threshold choice is subjective       |
| Isolation Forest  | Handles high dimensions, nonlinear patterns | Requires tuning `contamination`      |
| LOF               | Detects local anomalies                     | Slower on very large datasets        |
| DBSCAN            | Finds arbitrary-shaped clusters and noise   | Sensitive to `eps` and `min_samples` |
| Elliptic Envelope | Good for Gaussian data                      | Assumes elliptical distribution      |

---

# Best Practices

1. Visualize the data first (boxplots, histograms, scatter plots).
2. Understand the source of outliers before removing them.
3. Never remove outliers blindly—they may contain valuable information.
4. Compute thresholds only on the training set, then apply them to validation/test data.
5. Consider capping (winsorization) instead of deleting when you want to retain all observations.
6. Use robust models (e.g., tree-based methods) when appropriate, especially if genuine outliers are expected.
7. Choose the detection method based on the data distribution:

   * Approximately normal → Z-Score
   * Skewed → IQR or Modified Z-Score
   * High-dimensional → Isolation Forest
   * Density-based anomalies → LOF or DBSCAN
   * Gaussian multivariate data → Elliptic Envelope

---

# Summary Decision Tree

```text
                 Start
                    │
         Is data approximately normal?
               │             │
             Yes             No
              │               │
        Use Z-Score      Is data skewed?
                              │
                         Yes       No
                          │         │
                 IQR / Modified Z   High-dimensional?
                                        │
                                   Yes       No
                                    │         │
                           Isolation Forest  LOF / DBSCAN
```

Understanding **why** an observation is unusual is just as important as detecting it. The goal is not to eliminate every extreme value but to distinguish erroneous data from rare yet meaningful observations.
