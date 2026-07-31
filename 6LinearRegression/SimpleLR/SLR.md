## Simple Linear Regression

Simple Linear Regression is the most basic supervised learning algorithm used to model the relationship between **one input variable** and **one output variable**.

It answers a question like:

* How does house price change with area?
* How does salary change with years of experience?
* How does marks change with study hours?

It tries to fit a **straight line** through the data.

---

## 1) The idea

Suppose we have:

* `X` = independent variable / feature / input
* `y` = dependent variable / target / output

The model assumes a linear relationship:

[
y = mx + b
]

or in ML form:

[
\hat{y} = wX + b
]

Where:

* `\hat{y}` = predicted value
* `w` = slope / coefficient
* `b` = intercept
* `X` = input value

So the goal is to find the **best line** that fits the data.

---

## 2) Why it is called “simple”

It is called **simple** because it uses only **one independent variable**.

If there are multiple input variables, it becomes **Multiple Linear Regression**.

Example:

* Simple Linear Regression: price based on area only
* Multiple Linear Regression: price based on area, bedrooms, age, location, etc.

---

## 3) What the line means

### Slope (`w`)

The slope tells us how much `y` changes when `X` increases by 1 unit.

Example:
If

[
\hat{y} = 2000X + 5000
]

then each 1 unit increase in `X` increases predicted `y` by `2000`.

### Intercept (`b`)

The intercept is the predicted value when `X = 0`.

In the same equation:

[
\hat{y} = 2000X + 5000
]

if `X = 0`, then `\hat{y} = 5000`.

---

## 4) How it works

The model looks at training data and tries to find the best line such that the predicted values are as close as possible to actual values.

Example data:

| Hours studied | Marks |
| ------------- | ----- |
| 1             | 50    |
| 2             | 55    |
| 3             | 65    |
| 4             | 70    |
| 5             | 80    |

The model tries to draw a line that represents this pattern.

---

## 5) Prediction equation

For each input `x`, predicted output is:

[
\hat{y} = wx + b
]

Example:

If `w = 10` and `b = 30`:

* for `x = 2`, prediction = `10(2) + 30 = 50`
* for `x = 5`, prediction = `10(5) + 30 = 80`

---

## 6) Error in regression

The predicted value is usually not exactly equal to the actual value.

So we calculate the **error**:

[
\text{error} = y - \hat{y}
]

Example:

* actual = 70
* predicted = 65
* error = 5

The model tries to reduce these errors.

---

## 7) Cost function

To know how good the line is, we use a cost function.

The most common one is **Mean Squared Error (MSE)**:

[
MSE = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y_i})^2
]

Why square the error?

* positive and negative errors should not cancel out
* larger errors should be penalized more

The best line is the one that gives the **lowest MSE**.

---

## 8) Finding the best line

There are two main ways:

### a) Ordinary Least Squares

This is the mathematical method used to directly compute the best line.

It finds the line that minimizes the squared vertical distances between actual points and the line.

### b) Gradient Descent

This is an iterative optimization method.

It starts with random values of `w` and `b`, then keeps updating them to reduce the cost.

---

## 9) Assumptions of simple linear regression

Simple Linear Regression works well when these assumptions are roughly true:

* There is a linear relationship between `X` and `y`
* Errors are independent
* Errors have constant variance
* Errors are approximately normally distributed
* The feature is not highly noisy

If the relationship is not linear, the model may perform poorly.

---

## 10) Example in real life

### Predict salary from experience

Suppose:

* `X` = years of experience
* `y` = salary

A model might learn:

[
\text{Salary} = 50000 \times \text{Experience} + 300000
]

Meaning:

* every extra year of experience adds about 50,000 salary units
* base salary is 300,000

---

## 11) Training process

The training process usually follows these steps:

1. Collect data
2. Choose `X` and `y`
3. Split into training and test sets
4. Fit the model on training data
5. Predict on test data
6. Evaluate performance

---

## 12) Evaluation metrics

Common ways to evaluate simple linear regression:

### Mean Absolute Error (MAE)

Average absolute difference between actual and predicted values.

### Mean Squared Error (MSE)

Average squared difference.

### Root Mean Squared Error (RMSE)

Square root of MSE.

### R-squared (`R²`)

Shows how much variance in the target is explained by the model.

* `R² = 1` → perfect fit
* `R² = 0` → no explanatory power
* negative values can happen if the model is very bad

---

## 13) Graphical view

In a scatter plot:

* each point is a data sample
* the regression line is the fitted line
* the vertical distance from a point to the line is the error

The line is chosen so that these distances are minimized overall.

---

## 14) Simple Python example with sklearn

```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Example data
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
y = np.array([50, 55, 65, 70, 80])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create model
model = LinearRegression()

# Train
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)
print("Predictions:", y_pred)

# Evaluation
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))
```

---

## 15) Interpreting sklearn output

* `model.coef_` → slope
* `model.intercept_` → intercept

If slope is high, the target changes faster with the feature.

---

## 16) Advantages

* Simple and easy to understand
* Fast to train
* Works well for linearly related data
* Easy to interpret

---

## 17) Limitations

* Only works with one feature
* Assumes linear relationship
* Sensitive to outliers
* Poor performance on complex/nonlinear data

---

## 18) When to use it

Use simple linear regression when:

* you have one input variable
* the relationship looks roughly linear
* interpretability matters
* you want a quick baseline model

---

## 19) One-line summary

Simple Linear Regression is a method that fits a straight line between one input and one output to predict continuous values.

---

If you want, I can also explain this with a **full numerical example by hand** or with a **graph-based visualization**.
