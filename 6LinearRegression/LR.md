# Linear Regression in Machine Learning (Complete Guide)

Linear Regression is one of the **most fundamental supervised machine learning algorithms**. It is used to predict **continuous numerical values** by finding the relationship between one or more input variables (features) and an output variable (target).

Examples:

* Predicting house prices
* Predicting salary based on years of experience
* Predicting temperature
* Predicting stock demand
* Predicting sales revenue

---

# 1. What is Linear Regression?

Linear Regression tries to find the **best-fit straight line** through the data points.

For example:

| Experience (Years) | Salary ($) |
| -----------------: | ---------: |
|                  1 |     30,000 |
|                  2 |     35,000 |
|                  3 |     40,000 |
|                  4 |     45,000 |
|                  5 |     50,000 |

If we plot them, they almost form a straight line.

The regression algorithm learns this relationship and predicts future values.

---

# 2. Goal

Given input

[
X
]

predict output

[
Y
]

Example

```
Years Experience -----> Salary
```

---

# 3. Types of Linear Regression

There are two main types.

## A. Simple Linear Regression

One input feature.

Example

```
Experience ---> Salary
```

Only one independent variable.

Equation

[
Y=\beta_0+\beta_1X
]

Where

* β₀ = Intercept
* β₁ = Slope
* X = Feature
* Y = Prediction

---

## B. Multiple Linear Regression

More than one feature.

Example

```
House Price

depends on

Area
Bedrooms
Bathrooms
Location
Age
```

Equation

[
Y=\beta_0+\beta_1X_1+\beta_2X_2+\cdots+\beta_nX_n
]

---

# 4. Terminologies

## Dependent Variable

Target variable

Usually denoted by

[
Y
]

Example

Salary

---

## Independent Variable

Input variable

Usually

[
X
]

Example

Experience

---

## Coefficient

Represents how much Y changes when X changes.

Example

Salary = 20000 + 5000 × Experience

5000 is coefficient.

Meaning:

Every extra year increases salary by $5000.

---

## Intercept

The value of Y when X = 0.

Example

Salary = 20000 + 5000 × Experience

Intercept = 20000

---

# 5. Linear Regression Equation

The model learns

[
\hat y=\beta_0+\beta_1x
]

where:

* (\hat y) = Predicted value
* β₀ = Intercept
* β₁ = Slope
* x = Input feature

To help visualize the fitted line and how the regression adjusts to data:

genui{"inference_regression_ml_learning_block":{"type_id":"LEAST_SQUARE_REGRESSION"}}

---

# Example

Suppose

```
β₀ = 10

β₁ = 3
```

Predict for

```
x = 5
```

Then

```
y = 10 + 3×5

=25
```

Prediction = 25

---

# 6. Why is it called "Linear"?

Because the relationship is assumed to be linear.

```
Y = a + bX
```

Straight line.

Not

```
X²

X³

sin(X)

log(X)
```

---

# 7. How Does Linear Regression Learn?

Suppose we have

| Hours Studied | Marks |
| ------------- | ----- |
| 1             | 40    |
| 2             | 50    |
| 3             | 55    |
| 4             | 65    |
| 5             | 75    |

Initially the algorithm guesses a line.

Maybe

```
Y = 20 + 2X
```

This is poor.

Then it changes

* slope
* intercept

until prediction becomes close to actual values.

---

# 8. Prediction Error

Error

```
Actual − Predicted
```

Example

Actual

```
70
```

Predicted

```
65
```

Error

```
5
```

Every point has an error.

---

# 9. Residual

Residual

[
e=y-\hat y
]

Actual minus prediction.

Residuals tell us how far the model's predictions are from the observed values.

---

# 10. Cost Function

The model tries to minimize error.

We cannot simply sum errors because positive and negative errors cancel out.

Instead, we square them.

## Mean Squared Error (MSE)

[
MSE=\frac1n\sum(y-\hat y)^2
]

Steps

1. Find error.
2. Square it.
3. Sum all errors.
4. Divide by number of observations.

Smaller MSE means better fit.

---

# 11. Why Square Errors?

Suppose

Errors

```
2
-2
```

Sum

```
0
```

Looks perfect but isn't.

After squaring

```
4
4

Total=8
```

No cancellation.

---

# 12. Ordinary Least Squares (OLS)

Linear Regression chooses parameters that minimize

```
Sum of squared errors
```

Therefore

```
Least Squares
```

means

> Find the line whose squared residuals are as small as possible.

---

# 13. Gradient Descent

There are two ways to compute coefficients.

## Method 1

Normal Equation

Closed-form mathematical solution.

---

## Method 2

Gradient Descent

Common for large datasets.

Steps

```
Initialize coefficients

↓

Predict

↓

Compute loss

↓

Compute gradients

↓

Update coefficients

↓

Repeat
```

Update rule

[
\theta=\theta-\alpha\frac{\partial J}{\partial\theta}
]

where:

* θ = parameter
* α = learning rate
* J = cost function

---

# 14. Loss Function vs Cost Function

Loss

Error for one sample.

Cost

Average loss over all samples.

---

# 15. Assumptions of Linear Regression

Linear Regression works well when these assumptions approximately hold.

### 1. Linearity

Relationship should be linear.

```
✓ Straight-line trend
```

---

### 2. Independence

Observations should be independent.

---

### 3. Homoscedasticity

Equal variance of residuals.

Good

```
*******
*******
*******
```

Bad

```
*
**
****
*******
**********
```

---

### 4. Normality of Residuals

Residuals should approximately follow a normal distribution.

---

### 5. No Multicollinearity

Features should not be highly correlated.

Example

```
Height (cm)

Height (meter)
```

Both contain almost identical information.

---

# 16. Evaluation Metrics

## Mean Absolute Error (MAE)

[
MAE=\frac1n\sum|y-\hat y|
]

Easy to interpret.

---

## Mean Squared Error (MSE)

Squares errors.

Sensitive to outliers.

---

## Root Mean Squared Error (RMSE)

[
RMSE=\sqrt{MSE}
]

Units match the target variable.

---

## R² Score

Measures how much variance is explained by the model.

Range

```
0 to 1
```

Example

```
0.90

means

90% variance explained.
```

---

# 17. Advantages

* Simple
* Fast
* Easy to interpret
* Works well on linear relationships
* Requires little tuning
* Good baseline model

---

# 18. Disadvantages

* Only models linear relationships
* Sensitive to outliers
* Can overfit with many irrelevant features
* Assumes independent observations
* Cannot model complex nonlinear patterns

---

# 19. Real-Life Applications

* House price prediction
* Salary prediction
* Sales forecasting
* Demand prediction
* Weather forecasting (simple trends)
* Risk analysis
* Population growth (over limited ranges)
* Marketing analytics

---

# 20. Linear Regression in Scikit-Learn

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("house_price.csv")

# Features and target
X = df[["Area", "Bedrooms"]]
y = df["Price"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create model
model = LinearRegression()

# Train
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluation
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
print("MSE:", mean_squared_error(y_test, y_pred))
print("R²:", r2_score(y_test, y_pred))
```

---

# 21. Important Attributes in Scikit-Learn

After training:

```python
model.coef_
```

Returns feature coefficients.

Example

```python
array([250.5, 5000.2])
```

---

```python
model.intercept_
```

Returns intercept.

---

```python
model.predict(X)
```

Predicts outputs.

---

```python
model.score(X, y)
```

Returns the **R² score**.

---

# 22. Complete Workflow

```text
Collect Data
      │
      ▼
Clean Data
      │
      ▼
Handle Missing Values
      │
      ▼
Feature Engineering
      │
      ▼
Train/Test Split
      │
      ▼
Train Linear Regression
      │
      ▼
Predict
      │
      ▼
Evaluate (MAE, MSE, RMSE, R²)
      │
      ▼
Deploy Model
```

---

# 23. Interview Questions

1. What is Linear Regression?
2. Difference between Simple and Multiple Linear Regression?
3. What is the role of the intercept?
4. Why do we square residuals in MSE?
5. What are residuals?
6. Explain the Least Squares method.
7. What is the Normal Equation?
8. What is Gradient Descent, and when is it preferred?
9. What assumptions does Linear Regression make?
10. What is multicollinearity?
11. Difference between MAE, MSE, and RMSE?
12. What does the R² score indicate?
13. How do outliers affect Linear Regression?
14. How can you detect underfitting and overfitting in a regression model?
15. When would you choose Linear Regression over more complex models?

### Summary

Linear Regression is the foundation of regression analysis in machine learning. It models the relationship between input features and a continuous target by fitting the line that minimizes the **sum of squared residuals (Ordinary Least Squares)**. Understanding concepts such as **coefficients, intercept, residuals, cost functions, gradient descent, evaluation metrics (MAE, MSE, RMSE, R²), and model assumptions** is essential before moving on to more advanced regression techniques like **Ridge, Lasso, Elastic Net, Polynomial Regression,** and **Logistic Regression**.
