# Power Transforms in Scikit-learn (Machine Learning)

Power transformations are a **feature engineering technique** used to make data **more Gaussian (normally distributed)**. Many machine learning algorithms work better when the input features are approximately normally distributed.

Scikit-learn provides the `PowerTransformer` class, which implements two methods:

1. **Box-Cox Transformation**
2. **Yeo-Johnson Transformation**

---

# Why do we need Power Transformations?

Real-world datasets are rarely normally distributed.

They may have:

* Right skew (positive skew)
* Left skew (negative skew)
* Heavy tails
* Unequal variance (heteroscedasticity)

Example of right-skewed data:

```
|
|         *
|       **
|      **
|     **
|    **
|  ***
|***
+---------------------->
```

After power transformation:

```
        *
      *****
    *********
  *************
****************
------------------>
```

The distribution becomes much closer to a bell curve.

---

# Why is Normal Distribution Important?

Some algorithms assume features follow a Gaussian distribution.

Examples include:

* Linear Regression
* Logistic Regression
* LDA
* QDA
* Gaussian Naive Bayes
* PCA

Even when an algorithm doesn't require normality, reducing skewness often improves optimization and model performance.

---

# Problems Solved by Power Transformation

It helps to:

* Reduce skewness
* Stabilize variance
* Make data more Gaussian
* Reduce influence of large values
* Improve linear relationships
* Improve model performance

---

# PowerTransformer in sklearn

```python
from sklearn.preprocessing import PowerTransformer
```

Syntax:

```python
pt = PowerTransformer(
    method='yeo-johnson',
    standardize=True
)
```

Parameters

### method

```
'yeo-johnson'
```

or

```
'box-cox'
```

Default:

```python
method='yeo-johnson'
```

---

### standardize

After transformation, sklearn can automatically perform Standard Scaling.

```
Mean = 0
Std = 1
```

Default:

```python
standardize=True
```

---

# How Power Transformation Works

Instead of simply taking

```
log(x)
```

or

```
sqrt(x)
```

PowerTransformer searches for the **best exponent (λ, lambda)** that makes the data as close to normal as possible.

Instead of manually trying

```
log(x)
sqrt(x)
cube_root(x)
```

it automatically estimates

```
λ = best value
```

using **Maximum Likelihood Estimation (MLE)**.

---

# Types of Power Transformation

```
PowerTransformer
      |
      +----------------+
      |                |
 Box-Cox        Yeo-Johnson
```

---

# 1. Box-Cox Transformation

This is the older method.

Works **only on positive numbers**.

```
✓ 1
✓ 5
✓ 200

✗ 0
✗ -3
```

---

### Formula

For

```
λ ≠ 0
```

[
y = \frac{x^\lambda - 1}{\lambda}
]

If

```
λ = 0
```

then

[
y = \log(x)
]

---

### Example

Original

```
2
4
8
16
32
```

Suppose sklearn finds

```
λ = 0.5
```

Transformation becomes approximately

```
sqrt(x)
```

Result

```
1.41
2
2.82
4
5.65
```

Much less skewed.

---

### Example

```python
from sklearn.preprocessing import PowerTransformer

pt = PowerTransformer(method='box-cox')

X_train_bc = pt.fit_transform(X_train)
```

---

# Limitation of Box-Cox

Cannot handle

```
0
```

or

```
negative numbers
```

Example

```
Income

500
700
900
```

Works.

Example

```
Profit

-200
0
500
```

Fails.

---

# 2. Yeo-Johnson Transformation

Yeo-Johnson is an improved version.

Supports

* Positive values
* Zero
* Negative values

This is why sklearn uses it as the default.

---

Example

```
-100
-20
0
50
120
400
```

Yeo-Johnson works perfectly.

---

Example

```python
from sklearn.preprocessing import PowerTransformer

pt = PowerTransformer(method='yeo-johnson')

X_train_yj = pt.fit_transform(X_train)
```

---

# Box-Cox vs Yeo-Johnson

| Feature                  | Box-Cox | Yeo-Johnson |
| ------------------------ | ------- | ----------- |
| Positive values          | ✅       | ✅           |
| Zero values              | ❌       | ✅           |
| Negative values          | ❌       | ✅           |
| More commonly used today | ❌       | ✅           |
| Default in sklearn       | ❌       | ✅           |

---

# Standardization

If

```python
standardize=True
```

then after transformation,

every feature has approximately

```
Mean = 0
Std = 1
```

Equivalent to doing

```
Power Transform

↓

StandardScaler
```

automatically.

---

# Example

Original

```
Income

500
600
700
15000
30000
```

Mean

```
9360
```

Std

Very large.

After transformation

```
-1.3
-0.8
-0.3
0.9
1.5
```

Now

Mean

```
0
```

Std

```
1
```

---

# Complete Example

```python
from sklearn.preprocessing import PowerTransformer

pt = PowerTransformer(
    method='yeo-johnson',
    standardize=True
)

X_train_transformed = pt.fit_transform(X_train)

X_test_transformed = pt.transform(X_test)
```

Always use `fit_transform()` on the training data and only `transform()` on the test data to avoid data leakage.

---

# Visualizing Before and After

```python
for col in X_train.columns:

    fig, ax = plt.subplots(1, 2, figsize=(14, 4))

    sns.histplot(X_train[col], kde=True, stat="density", ax=ax[0])
    ax[0].set_title("Before")

    sns.histplot(X_train_transformed[col], kde=True, stat="density", ax=ax[1])
    ax[1].set_title("After")

    plt.show()
```

You can also inspect Q-Q plots before and after transformation using `scipy.stats.probplot()` to see how closely the transformed data follows a normal distribution.

---

# When Should You Use Power Transformation?

Use it when:

* Features are highly skewed.
* Features contain large outliers that distort the distribution (though extreme outliers may still need separate treatment).
* You are using models that benefit from approximately normal inputs, such as Linear Regression, Logistic Regression, LDA/QDA, PCA, or Gaussian Naive Bayes.

Avoid it when:

* Your data is already close to normally distributed.
* You're using tree-based models like Decision Trees, Random Forests, XGBoost, LightGBM, or CatBoost, which are generally insensitive to monotonic transformations of features.

---

# PowerTransformer vs Log Transform

| Aspect                  | Log Transform    | PowerTransformer              |
| ----------------------- | ---------------- | ----------------------------- |
| Formula                 | Fixed (`log(x)`) | Learns the best power (`λ`)   |
| Handles negative values | ❌                | Yeo-Johnson: ✅                |
| Handles zero            | ❌                | Yeo-Johnson: ✅                |
| Automatic               | ❌                | ✅                             |
| Standardization         | ❌                | Optional (`standardize=True`) |
| Flexibility             | Low              | High                          |

The log transform is just one specific type of transformation. `PowerTransformer` is more flexible because it automatically estimates the most suitable transformation for each feature.

---

# Best Practices

* Use **Yeo-Johnson** unless you know all values are strictly positive and want to use Box-Cox.
* Fit the transformer only on the training data.
* Use `transform()` on validation/test data.
* Check histograms and Q-Q plots to verify that the transformation improved normality.
* Consider `standardize=False` if you plan to apply a separate scaler afterward, although the default `standardize=True` is convenient in many workflows.
