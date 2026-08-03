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
