# Mathematical Transformations in Machine Learning (Detailed Notes)

Mathematical transformations are **feature engineering techniques** used to modify numerical features so that machine learning algorithms can learn patterns more effectively.

They are mainly used to:

* Make data more normally distributed.
* Reduce skewness.
* Reduce the effect of outliers.
* Improve linear relationships.
* Stabilize variance.
* Improve model performance.

---

# Why do we need Mathematical Transformations?

Consider the following dataset representing salaries.

| Person | Salary ($) |
| ------ | ---------: |
| A      |     20,000 |
| B      |     25,000 |
| C      |     30,000 |
| D      |     40,000 |
| E      |     50,000 |
| F      |    800,000 |

The last salary is much larger than the others.

Many ML algorithms (especially Linear Regression) struggle because:

* Mean shifts
* Variance increases
* Data becomes highly skewed

Applying a mathematical transformation compresses large values.

Example:

```
Original

20000
25000
30000
40000
50000
800000

↓

Log Transform

9.90
10.12
10.31
10.60
10.82
13.59
```

Notice how the gap becomes much smaller.

---

# When should Mathematical Transformations be used?

They are useful when data has:

* Right skewed distribution
* Left skewed distribution
* Non-linear relationships
* Exponential growth
* Wide range of values
* Unequal variance (heteroscedasticity)

---

# Common Mathematical Transformations

There are several commonly used transformations.

```
Numerical Data
      │
      ├── Log Transformation
      ├── Square Root Transformation
      ├── Cube Root Transformation
      ├── Reciprocal Transformation
      ├── Power Transformation
      │      ├── Box-Cox
      │      └── Yeo-Johnson
      └── Quantile Transformation
```

---

# 1. Log Transformation

Formula

[
x' = \log(x)
]

Usually

[
\log_e(x)
]

or

[
\log_{10}(x)
]

---

## Example

Original values

|    x |
| ---: |
|    1 |
|    2 |
|    5 |
|   10 |
|   50 |
|  100 |
| 1000 |

After log

|    x | log(x) |
| ---: | -----: |
|    1 |      0 |
|    2 |   0.69 |
|    5 |   1.61 |
|   10 |   2.30 |
|   50 |   3.91 |
|  100 |   4.61 |
| 1000 |   6.91 |

Notice:

Large values are compressed.

---

## Visualization

Original

```
|
|                          *
|
|
|                 *
|
|          *
|
|     *
| *
+-----------------------------
```

Log transformed

```
|
|                *
|
|            *
|
|         *
|
|      *
|
|   *
+-----------------------------
```

The data becomes more balanced.

---

## Advantages

✔ Reduces right skew

✔ Compresses large values

✔ Handles exponential growth

✔ Improves linear regression

---

## Disadvantages

Cannot handle

[
x \le 0
]

because

```
log(0) = undefined
log(-5) = undefined
```

---

## Python Example

```python
import numpy as np

df['salary_log'] = np.log(df['salary'])
```

If zeros exist:

```python
df['salary_log'] = np.log1p(df['salary'])
```

`log1p(x)` computes:

[
\log(1+x)
]

which safely handles zero values.

---

# 2. Square Root Transformation

Formula

[
x' = \sqrt{x}
]

---

Example

| Original | √x |
| -------: | -: |
|        1 |  1 |
|        4 |  2 |
|        9 |  3 |
|       16 |  4 |
|       25 |  5 |
|      100 | 10 |

---

Advantages

* Mildly reduces skewness
* Good for count data
* Keeps ordering

---

Python

```python
df['feature'] = np.sqrt(df['feature'])
```

---

# 3. Cube Root Transformation

Formula

[
x' = \sqrt[3]{x}
]

---

Example

| Original | Cube Root |
| -------: | --------: |
|        8 |         2 |
|       27 |         3 |
|       64 |         4 |
|      125 |         5 |

---

Advantages

Works with:

Positive numbers

Negative numbers

Zero

Unlike log transformation.

---

Python

```python
df['feature'] = np.cbrt(df['feature'])
```

---

# 4. Reciprocal Transformation

Formula

[
x'=\frac{1}{x}
]

---

Example

| Original | Reciprocal |
| -------: | ---------: |
|        1 |          1 |
|        2 |        0.5 |
|        5 |        0.2 |
|       10 |        0.1 |
|       20 |       0.05 |

---

Advantages

Very strong reduction of large values.

---

Disadvantages

Cannot use

[
x=0
]

---

Python

```python
df['feature'] = 1 / df['feature']
```

---

# Comparison

Suppose

```
1
10
100
1000
```

Different transformations

| Original |  Log |     √ | Cube Root | Reciprocal |
| -------: | ---: | ----: | --------: | ---------: |
|        1 |    0 |     1 |         1 |          1 |
|       10 | 2.30 |  3.16 |      2.15 |       0.10 |
|      100 | 4.61 |    10 |      4.64 |       0.01 |
|     1000 | 6.91 | 31.62 |        10 |      0.001 |

Notice how each transformation compresses large values differently.

---

# 5. Power Transformation

Instead of manually selecting a transformation, power transformations automatically choose an exponent (λ) that makes the data resemble a normal distribution.

There are two popular methods:

* Box-Cox
* Yeo-Johnson

---

## Box-Cox Transformation

Works **only with positive values**.

The transformation is:

[
x' =
\begin{cases}
\dfrac{x^\lambda - 1}{\lambda}, & \lambda \neq 0 \
\ln(x), & \lambda = 0
\end{cases}
]

The algorithm finds the best value of λ that makes the data as close to normally distributed as possible.

Examples:

* λ = 1 → almost no transformation.
* λ = 0 → log transformation.
* λ = 0.5 → square root-like transformation.
* λ = -1 → reciprocal-like transformation.

### Python

```python
from sklearn.preprocessing import PowerTransformer

pt = PowerTransformer(method='box-cox')

df['feature'] = pt.fit_transform(df[['feature']])
```

---

## Yeo-Johnson Transformation

Yeo-Johnson extends the Box-Cox idea to work with **positive, zero, and negative values**.

Advantages:

* Handles zero values.
* Handles negative values.
* Automatically selects the optimal λ.

### Python

```python
from sklearn.preprocessing import PowerTransformer

pt = PowerTransformer(method='yeo-johnson')

df['feature'] = pt.fit_transform(df[['feature']])
```

---

# 6. Quantile Transformation

A quantile transformation maps the original data to follow a chosen distribution, typically **uniform** or **normal**, by replacing values with their corresponding quantiles (ranks).

Instead of using a mathematical formula directly, it:

1. Sorts the data.
2. Computes the percentile (quantile) of each value.
3. Maps these percentiles to the target distribution.

This transformation is particularly useful when data contains many outliers or has a complex, non-Gaussian distribution.

### Example

Original data:

```
2, 3, 5, 8, 10
```

Each value is assigned a percentile and then transformed to fit a normal or uniform distribution.

### Python

```python
from sklearn.preprocessing import QuantileTransformer

qt = QuantileTransformer(
    output_distribution='normal'
)

df['feature'] = qt.fit_transform(df[['feature']])
```

You can also set:

```python
output_distribution='uniform'
```

---

# Which Transformation Should You Choose?

| Situation                                | Recommended Transformation |
| ---------------------------------------- | -------------------------- |
| Positive values with strong right skew   | Log or Box-Cox             |
| Positive values with moderate right skew | Square Root                |
| Positive, zero, or negative values       | Cube Root or Yeo-Johnson   |
| Very strong influence of large values    | Reciprocal                 |
| Automatic normalization (positive only)  | Box-Cox                    |
| Automatic normalization (any values)     | Yeo-Johnson                |
| Many outliers or unusual distributions   | Quantile Transformer       |

---

# How to Decide Whether a Transformation Helped?

After transforming the data, compare the distribution before and after using:

* Histogram
* KDE (Kernel Density Estimate) plot
* Q-Q plot
* Skewness statistic
* Model performance (cross-validation or test set)

If the transformed feature is closer to a normal distribution (when appropriate) and your model performs better, the transformation was likely beneficial.

---

# Advantages of Mathematical Transformations

* Reduce skewness.
* Stabilize variance.
* Reduce the impact of outliers.
* Improve linear relationships.
* Help satisfy assumptions of models like Linear Regression.
* Often improve optimization and predictive performance.

---

# Limitations

* Some transformations (e.g., log and Box-Cox) cannot handle non-positive values.
* They can make feature values less interpretable.
* Tree-based models (Decision Trees, Random Forests, XGBoost, LightGBM, CatBoost) usually gain little benefit because they do not rely on feature distributions or linear relationships.

---

# Which ML Algorithms Benefit Most?

| Algorithm                     | Benefit from Transformations |
| ----------------------------- | ---------------------------- |
| Linear Regression             | ⭐⭐⭐⭐⭐                        |
| Logistic Regression           | ⭐⭐⭐⭐☆                        |
| Support Vector Machine (SVM)  | ⭐⭐⭐⭐☆                        |
| K-Nearest Neighbors (KNN)     | ⭐⭐⭐⭐☆                        |
| Neural Networks               | ⭐⭐⭐☆☆                        |
| Naive Bayes                   | ⭐⭐⭐☆☆                        |
| Decision Tree                 | ⭐☆☆☆☆                        |
| Random Forest                 | ⭐☆☆☆☆                        |
| XGBoost / LightGBM / CatBoost | ⭐☆☆☆☆                        |

**Key takeaway:** Mathematical transformations are most valuable for algorithms that assume linear relationships, normality, or are sensitive to the scale and distribution of features. Always inspect the data and validate performance before and after transformation rather than applying them blindly.
