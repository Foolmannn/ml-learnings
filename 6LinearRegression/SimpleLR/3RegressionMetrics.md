> From: https://chatgpt.com/c/6a70b724-6594-83ee-8335-acc597d24d7e


# Regression Metrics in Machine Learning (Complete Guide)

When building a regression model, one of the biggest questions is:

> **"How good is my model?"**

Unlike classification (where we use accuracy, precision, recall, etc.), regression predicts **continuous values** (price, temperature, salary, stock price, etc.), so we need different evaluation metrics.

The most commonly used regression metrics are:

1. **Mean Absolute Error (MAE)**
2. **Mean Squared Error (MSE)**
3. **Root Mean Squared Error (RMSE)**
4. **R² Score (Coefficient of Determination)**
5. **Adjusted R² Score**
6. **Mean Absolute Percentage Error (MAPE)**
7. **Median Absolute Error**
8. **Explained Variance Score**
9. **Mean Squared Logarithmic Error (MSLE)**
10. **Root Mean Squared Log Error (RMSLE)**

Let's understand each in detail.

---

# Before Learning Metrics

Suppose we have a regression model predicting house prices.

| House | Actual Price (y) | Predicted Price (ŷ) |
|---------|-----------------|---------------------|
| A | 100 | 110 |
| B | 120 | 118 |
| C | 140 | 150 |
| D | 160 | 158 |
| E | 180 | 170 |

Prediction error (Residual)

$$
Error = y - \hat{y}
$$

| House | Actual | Predicted | Error |
|---------|---------|-----------|-------|
|A|100|110|-10|
|B|120|118|2|
|C|140|150|-10|
|D|160|158|2|
|E|180|170|10|

These errors are used to compute every regression metric.

---

# 1. Mean Absolute Error (MAE)

## Definition

MAE measures the **average absolute difference** between actual and predicted values.

Absolute value ignores whether the prediction is above or below the true value.

Formula

$$
MAE=\frac{1}{n}\sum_{i=1}^{n}|y_i-\hat{y_i}|
$$

---

## Example

Errors

-10, 2, -10, 2, 10

Absolute errors

10,2,10,2,10

Average

$$
MAE=\frac{10+2+10+2+10}{5}
=\frac{34}{5}
=6.8
$$

Meaning:

On average, predictions are **6.8 units away** from actual values.

---

## Advantages

- Easy to understand
- Same unit as target
- Less affected by outliers
- Robust

---

## Disadvantages

Since errors aren't squared,

Large mistakes are **not punished heavily**.

Example

Error = 2

Error = 20

Contribution to MAE:

2

20

Only 10 times larger.

---

## Use MAE When

- Outliers exist
- Every error is equally important
- Interpretability matters

---

# 2. Mean Squared Error (MSE)

MSE squares every error before averaging.

Formula

$$
MSE=\frac{1}{n}\sum(y_i-\hat y_i)^2
$$

---

Example

Errors

-10,2,-10,2,10

Squares

100

4

100

4

100

Average

$$
MSE=\frac{100+4+100+4+100}{5}
=\frac{308}{5}
=61.6
$$

---

## Why Square Errors?

Because

Negative errors disappear

Large errors become much larger.

Example

Error

2² = 4

20² = 400

Instead of being 10 times worse,

it becomes

100 times worse.

This heavily penalizes poor predictions.

---

## Advantages

Differentiable

Works well for optimization

Strong penalty for large errors

Used by most regression algorithms

---

## Disadvantages

Not interpretable

Unit becomes squared

(price², meter², etc.)

Very sensitive to outliers.

---

Example

Errors

2,3,4,50

MAE

≈15

MSE

≈630

One outlier dominates.

---

## Use MSE When

Large errors are unacceptable.

Example

Medical diagnosis

Weather prediction

Autonomous driving

---

# 3. Root Mean Squared Error (RMSE)

RMSE is simply the square root of MSE.

Formula

$$
RMSE=\sqrt{MSE}
$$

For our example

$$
RMSE=\sqrt{61.6}=7.85
$$

---

Why take square root?

Because MSE has squared units.

RMSE converts back to original units.

House price example

MSE

Dollar²

RMSE

Dollar

Easy to interpret.

---

Advantages

Same unit as target

Penalizes large errors

Widely used

---

Disadvantages

Still affected by outliers.

---

Use RMSE when

Large mistakes should be penalized,

but interpretation is important.

---

# MAE vs RMSE

| Feature | MAE | RMSE |
|----------|------|------|
|Squares errors?|No|Yes|
|Outlier sensitive?|Low|High|
|Interpretability|Easy|Easy|
|Optimization|Less smooth|Very smooth|
|Large error penalty|Low|High|

---

Example

Errors

2,2,2,20

MAE

6.5

RMSE

10.1

Notice RMSE increased much more.

---

# 4. R² Score (Coefficient of Determination)

Most popular regression metric.

Instead of measuring prediction error,

it measures

> **How much variance is explained by the model?**

Think of it like

How much information in the data your model captures.

---

Variance

Suppose house prices are

100

120

140

160

180

There is variation.

A regression model tries to explain this variation.

---

Formula

$$
R^2 = 1-\frac{SS_{res}}{SS_{tot}}
$$

Where

Residual Sum of Squares

$$
SS_{res}
=
\sum(y-\hat y)^2
$$

Total Sum of Squares

$$
SS_{tot}
=
\sum(y-\bar y)^2
$$

---

## Interactive intuition



---

## Interpretation

### R² = 1

Perfect prediction

Every point lies exactly on regression line.

---

### R² = 0

Model performs no better than predicting the mean.

---

### R² < 0

Model performs worse than simply predicting the average.

---

Example

Actual

10

20

30

40

Mean

25

A bad model predicts

100

100

100

100

Its error exceeds the error from always predicting 25.

R² becomes negative.

---

## Example

Suppose

SSres = 50

SStotal = 200

Then

$$
R^2
=
1-\frac{50}{200}
=
0.75
$$

Meaning

The model explains **75% of the variability** in the target values, while the remaining **25%** is unexplained (noise, missing features, or model limitations).

---

Advantages

Independent of scale

Easy comparison

Popular

---

Disadvantages

Doesn't indicate overfitting

Always increases when features increase

Not reliable alone

---

# 5. Adjusted R² Score

Problem

Adding more features always increases R².

Even useless features.

Example

Predict Salary

Features

Age

Experience

Education

Now add

Random number

R² usually increases slightly.

But the model didn't truly improve.

Adjusted R² fixes this.

Formula

$$
Adjusted\ R^2
=
1-
(1-R^2)
\frac{n-1}{n-p-1}
$$

Where

- n = Number of observations
- p = Number of independent features

---

## Why Adjusted R²?

It rewards useful features and penalizes unnecessary ones.

If a newly added feature does not improve the model enough to justify its complexity, Adjusted R² decreases.

---

## Example

Model A

2 features

R² = 0.85

Adjusted R² = 0.84

Model B

10 features

R² = 0.86

Adjusted R² = 0.80

Although Model B has a higher R², its much lower Adjusted R² suggests many of the added features are not contributing meaningful predictive power.

---

## When to Use

- Multiple Linear Regression
- Comparing models with different numbers of features
- Feature selection

---

# 6. Mean Absolute Percentage Error (MAPE)

Formula

$$
MAPE
=
\frac{100}{n}
\sum
\left|
\frac{y-\hat y}{y}
\right|
$$

Measures average percentage error.

Example

Actual =100

Predicted=90

Error

10%

Very interpretable.

---

Disadvantages

Cannot handle actual value = 0

Explodes for small values

---

Use

Business forecasting

Sales prediction

Demand prediction

---

# 7. Median Absolute Error

Formula

Median of

$$
|y-\hat y|
$$

Instead of averaging,

takes median.

Very robust to outliers.

---
