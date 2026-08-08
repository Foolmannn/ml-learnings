
The main assumptions of **linear regression** (often remembered as **LINE**) are:

1. **Linearity**
   - The relationship between the independent variable(s) and the dependent variable is linear.
   - This means the expected value of the response is a straight-line function of the predictors.

2. **Independence of Errors**
   - The residuals (errors) are independent of each other.
   - This assumption is especially important for time-series data, where observations may be correlated.

3. **Normality of Errors**
   - The residuals are approximately normally distributed.
   - This assumption is mainly required for valid hypothesis tests and confidence intervals, not for making predictions.

4. **Equal Variance (Homoscedasticity)**
   - The residuals have constant variance across all levels of the independent variable(s).
   - If the variance changes, it is called **heteroscedasticity**.

5. **No Multicollinearity** (for multiple linear regression)
   - Independent variables should not be highly correlated with each other.
   - High multicollinearity makes it difficult to determine the individual effect of each predictor.

6. **No Significant Outliers or Influential Points**
   - Extreme observations can disproportionately affect the regression line and distort the results.

### Summary Table

| Assumption | Meaning | Common Check |
|------------|---------|--------------|
| Linearity | Relationship is linear | Scatter plot, residual plot |
| Independence | Errors are independent | Durbin–Watson test |
| Normality | Residuals are normally distributed | Q–Q plot, Shapiro–Wilk test |
| Homoscedasticity | Constant error variance | Residuals vs. fitted plot, Breusch–Pagan test |
| No Multicollinearity | Predictors are not highly correlated | Variance Inflation Factor (VIF) |
| No Influential Outliers | No overly influential observations | Cook's distance, leverage plots |

### Easy Mnemonic: **LINE**
- **L** – Linearity
- **I** – Independence
- **N** – Normality of residuals
- **E** – Equal variance (Homoscedasticity)

These assumptions ensure that the regression coefficients, hypothesis tests, confidence intervals, and predictions are reliable.

---


> **What does the assumption mean → why is it needed → what happens when it is violated → how do we detect it → how do we fix it → Python code.**

Below is a practical, detailed guide using `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `scipy`, and `statsmodels`.

---

# Assumptions of Linear Regression — Detailed Guide

Suppose we have:

$$
Y = \beta_0+\beta_1X_1+\beta_2X_2+\cdots+\beta_pX_p+\epsilon
$$

where:

- $Y$ = dependent/target variable
- $X_1,\dots,X_p$ = independent variables
- $\beta_0$ = intercept
- $\beta_1,\dots,\beta_p$ = coefficients
- $\epsilon$ = error/residual

The fitted model gives:

$$
\hat{Y}=\hat{\beta}_0+\hat{\beta}_1X_1+\cdots+\hat{\beta}_pX_p
$$

and the residual is:

$$
e_i=y_i-\hat{y}_i
$$

The important assumptions are:

1. **Linearity**
2. **Independence of observations/errors**
3. **Homoscedasticity**
4. **Normality of residuals**
5. **No severe multicollinearity**
6. **No problematic outliers**
7. **No highly influential observations**
8. **Correct model specification / exogeneity**

Let's go through each.

---

# 1. Linearity

## What does linearity mean?

The most fundamental assumption is that the **expected value of the target is a linear function of the predictors**.

For simple linear regression:

$$
Y=\beta_0+\beta_1X+\epsilon
$$

We assume:

$$
E[Y|X]=\beta_0+\beta_1X
$$

For multiple regression:

$$
E[Y|X_1,X_2,\ldots,X_p]
=
\beta_0+\beta_1X_1+\cdots+\beta_pX_p
$$

### Example

Suppose we're predicting salary based on experience.

A linear relationship might look like:

```text
Salary
  |
  |          *
  |       *
  |    *
  |  *
  | *
  +---------------- Experience
```

But if the actual relationship looks like:

```text
Salary
  |
  |             *
  |          *
  |       *
  |    *
  | *
  +---------------- Experience
```

or a U-shaped relationship:

```text
Y
|
| *             *
|   *         *
|     *     *
|       * *
+---------------- X
```

then a simple linear model may not be appropriate.

---

# How to detect linearity

There are several approaches.

## Method 1: Scatter plot

For simple linear regression:

```python
import matplotlib.pyplot as plt
import seaborn as sns

sns.scatterplot(x=X, y=y)

plt.xlabel("X")
plt.ylabel("Y")
plt.title("X vs Y")
plt.show()
```

You can also add the regression line:

```python
sns.regplot(x=X, y=y)

plt.title("Linearity Check")
plt.show()
```

If the points generally follow a straight-line pattern, linearity is plausible.

---

# Method 2: Residual vs fitted plot

This is one of the **most important diagnostics**.

First train the model:

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_train)

residuals = y_train - y_pred
```

Now plot residuals:

```python
plt.scatter(y_pred, residuals)

plt.axhline(y=0, linestyle="--")

plt.xlabel("Fitted Values")
plt.ylabel("Residuals")
plt.title("Residuals vs Fitted Values")

plt.show()
```

### Good result

You want something approximately like:

```text
Residual
   |
 + |    *   *      *
   | *      *   *
 0 |------------------------
   |   *    *   *     *
 - | *       *     *
   |
   +------------------------- Fitted
```

Randomly scattered around zero.

### Bad result — nonlinear relationship

```text
Residual
   |
 + | *             *
   |   *         *
 0 |------*---*--------------
   |        *   *
 - |          *
   |
   +------------------------- Fitted
```

A curved pattern means your model may be missing a nonlinear relationship.

---

# What can you do if linearity is violated?

### Polynomial features

For example:

$$
Y=\beta_0+\beta_1X+\beta_2X^2+\epsilon
$$

Python:

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

model = make_pipeline(
    PolynomialFeatures(degree=2),
    LinearRegression()
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
```

Notice something important:

> Polynomial regression is still a **linear regression model with respect to its coefficients**.

---

# 2. Independence of Errors

The residuals should be independent.

Mathematically:

$$
Cov(\epsilon_i,\epsilon_j)=0
$$

for:

$$
i\neq j
$$

In simple words:

> The error for one observation should not depend on the error of another observation.

---

## Example

Imagine predicting temperature:

```text
Day 1 → 20°C
Day 2 → 21°C
Day 3 → 22°C
Day 4 → 23°C
```

Temperature measurements from consecutive days are naturally related.

Therefore, their errors might also be correlated.

This violates independence.

---

# Why is this a problem?

The coefficient estimates can still sometimes be unbiased, but:

- standard errors become unreliable
- confidence intervals can become wrong
- hypothesis tests can become misleading

This is particularly important with **time-series data**.

---

# Detection: Residuals over observation order

If observations have a natural sequence:

```python
plt.plot(residuals)

plt.axhline(y=0, linestyle="--")

plt.xlabel("Observation Order")
plt.ylabel("Residual")
plt.title("Residuals Over Observation Order")

plt.show()
```

You don't want patterns such as:

```text
Residual
   |
 + |       /\
   |      /  \
 0 |-----/----\---------
   |    /      \
 - |___/        \____
   |
   +-------------------- Time
```

This suggests autocorrelation.

---

# Durbin-Watson Test

A common statistical test is the **Durbin-Watson test**.

Using `statsmodels`:

```python
from statsmodels.stats.stattools import durbin_watson

dw = durbin_watson(residuals)

print("Durbin-Watson:", dw)
```

The statistic ranges approximately from:

$$
0 \rightarrow 4
$$

Interpretation:

| DW | Interpretation |
|---:|---|
| ~2 | No strong autocorrelation |
| < 2 | Positive autocorrelation |
| > 2 | Negative autocorrelation |
| ~0 | Strong positive autocorrelation |
| ~4 | Strong negative autocorrelation |

For example:

```text
DW = 1.98
```

is generally good.

But:

```text
DW = 0.45
```

suggests strong positive autocorrelation.

---

# Important limitation

Durbin-Watson is most meaningful when your observations have a meaningful ordering, especially time series.

For ordinary randomly sampled independent observations, this isn't usually the primary concern.

---

# 3. Homoscedasticity

This is one of the most important assumptions.

### Homoscedasticity means:

The variance of residuals should remain approximately constant across the range of predictions.

Mathematically:

$$
Var(\epsilon|X)=\sigma^2
$$

where $\sigma^2$ is constant.

---

## Good example

```text
Residual
  |
  |   *  *    * *  *
  | *   *  * *   * *
0 |---------------------
  | * *   *   * *  *
  |   * *   *   *  *
  +---------------------- Fitted
```

The spread is roughly constant.

---

# Heteroscedasticity

Suppose the residuals look like:

```text
Residual
  |
  | *
  | * *
  |  *  *
  |   *   *
0 |---------------------
  |      *   *
  |         *    *
  |             *    *
  +---------------------- Fitted
```

The spread increases as predictions increase.

This is **heteroscedasticity**.

---

# Why is heteroscedasticity a problem?

The coefficients may still be reasonable, but:

- standard errors become unreliable
- confidence intervals can be wrong
- p-values can be misleading
- statistical inference becomes less reliable

---

# Detection 1: Residual vs fitted plot

```python
plt.scatter(y_pred, residuals)

plt.axhline(0, linestyle="--")

plt.xlabel("Fitted Values")
plt.ylabel("Residuals")
plt.title("Homoscedasticity Check")

plt.show()
```

Look for a constant spread.

---

# Detection 2: Breusch-Pagan Test

This is a very useful statistical test.

```python
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan

X_train_sm = sm.add_constant(X_train)

ols_model = sm.OLS(y_train, X_train_sm).fit()

bp_test = het_breuschpagan(
    ols_model.resid,
    ols_model.model.exog
)

labels = [
    "LM Statistic",
    "LM-Test p-value",
    "F-Statistic",
    "F-Test p-value"
]

for label, value in zip(labels, bp_test):
    print(label, ":", value)
```

The important value is the **p-value**.

### Hypotheses

$$
H_0:
\text{Homoscedasticity}
$$

$$
H_1:
\text{Heteroscedasticity}
$$

If:

```text
p-value > 0.05
```

we generally don't have sufficient evidence to reject homoscedasticity.

If:

```text
p-value < 0.05
```

there is evidence of heteroscedasticity.

---

# What can you do about heteroscedasticity?

Possible solutions include:

### 1. Transform the target

For example:

$$
Y'=\log(Y)
$$

```python
import numpy as np

y_log = np.log1p(y)
```

Then train using `y_log`.

---

### 2. Transform predictors

For example:

```python
X["income_log"] = np.log1p(X["income"])
```

---

### 3. Weighted Least Squares

Instead of ordinary least squares:

$$
\min \sum e_i^2
$$

you use:

$$
\min \sum w_i e_i^2
$$

---

### 4. Robust standard errors

If your primary goal is inference rather than prediction, robust standard errors can help.

---

# 4. Normality of Residuals

The assumption is:

$$
\epsilon \sim N(0,\sigma^2)
$$

approximately.

Notice that it is the **residuals**, not necessarily the original target variable, that should be approximately normal.

---

# Very important clarification

Many beginners incorrectly think:

> "The target variable must be normally distributed."

That's not the assumption.

You should examine:

```text
Residuals
```

rather than simply:

```text
y
```

---

# Why do we need normal residuals?

Normality is mainly important for:

- hypothesis testing
- confidence intervals
- t-tests on coefficients
- F-tests

For pure prediction, slight departures from normality aren't necessarily disastrous, especially with large datasets.

---

# Detection 1: Histogram

```python
sns.histplot(residuals, kde=True)

plt.xlabel("Residual")
plt.title("Distribution of Residuals")

plt.show()
```

Ideally, it should look approximately bell-shaped.

---

# Detection 2: Q-Q plot

This is generally more informative.

```python
import statsmodels.api as sm

sm.qqplot(
    residuals,
    line="45"
)

plt.title("Q-Q Plot of Residuals")
plt.show()
```

### Good Q-Q plot

The points approximately follow the diagonal line.

```text
Observed
   |
   |             *
   |          *
   |       *
   |    *
   | *
   +---------------- Expected
```

### Bad Q-Q plot

Strong deviation from the line, especially at the tails, suggests non-normality.

---

# Detection 3: Shapiro-Wilk test

```python
from scipy.stats import shapiro

stat, p_value = shapiro(residuals)

print("Statistic:", stat)
print("p-value:", p_value)
```

Hypotheses:

$$
H_0:
\text{Residuals are normally distributed}
$$

$$
H_1:
\text{Residuals are not normally distributed}
$$

If:

```text
p > 0.05
```

we don't have sufficient evidence to reject normality.

If:

```text
p < 0.05
```

there is evidence against normality.

---

# Important problem with Shapiro-Wilk

With a **very large dataset**, Shapiro-Wilk can detect extremely tiny deviations from normality.

So don't blindly say:

```python
p < 0.05
```

→ "Linear regression is useless."

Instead combine:

- Q-Q plot
- histogram
- sample size
- domain knowledge

---

# 5. No Multicollinearity

This applies primarily to **multiple linear regression**.

Suppose:

$$
Y=\beta_0+\beta_1X_1+\beta_2X_2+\epsilon
$$

and:

$$
X_1 \approx X_2
$$

Then the predictors contain almost the same information.

This is **multicollinearity**.

---

# Example

Suppose you're predicting house price using:

```text
house_size
house_area_sqft
```

These might essentially represent the same information.

Or:

```text
age_in_years
year_built
```

which are strongly related.

---

# Why is multicollinearity a problem?

It can cause:

- unstable coefficients
- large standard errors
- coefficients changing dramatically
- unexpected coefficient signs
- difficulty interpreting individual predictors

For example:

Without multicollinearity:

```text
Size coefficient = +5000
```

With strong multicollinearity:

```text
Size coefficient = -12000
```

even though you expected a positive relationship.

The model may still predict well, but interpreting individual coefficients becomes difficult.

---

# Detection 1: Correlation matrix

```python
corr = X.corr()

plt.figure(figsize=(10, 8))

sns.heatmap(
    corr,
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Matrix")

plt.show()
```

If you see:

```text
X1 ↔ X2 = 0.95
```

there may be a multicollinearity problem.

But:

> Correlation alone is not enough.

Why?

Because multicollinearity can involve combinations of **multiple variables**, not just pairs.

---

# Detection 2: VIF

The most common approach is **Variance Inflation Factor**.

For predictor $X_j$:

$$
VIF_j=\frac{1}{1-R_j^2}
$$

where $R_j^2$ is obtained by regressing $X_j$ against all the other predictors.

---

## Python

```python
from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd

X_vif = X_train.copy()

X_vif = sm.add_constant(X_vif)

vif = pd.DataFrame()

vif["Feature"] = X_vif.columns

vif["VIF"] = [
    variance_inflation_factor(
        X_vif.values,
        i
    )
    for i in range(X_vif.shape[1])
]

print(vif)
```

Example:

```text
Feature          VIF
---------------------
const            2.1
age              1.8
income           3.2
house_size       12.5
rooms            11.8
```

---

# VIF interpretation

There isn't one universally accepted cutoff, but a practical rule is:

| VIF | Interpretation |
|---:|---|
| 1 | No multicollinearity |
| 1–5 | Usually acceptable |
| 5–10 | Potential concern |
| >10 | Serious concern |

Some practitioners use **VIF > 5** as the warning threshold.

Don't treat these as absolute mathematical rules.

---

# How to fix multicollinearity

### Remove redundant variables

```python
X = X.drop(columns=["rooms"])
```

### Combine variables

For example:

```text
total_rooms
bedrooms
bathrooms
```

could potentially be transformed into meaningful composite features.

### Use regularization

Ridge regression is especially useful:

$$
Loss =
RSS+\lambda\sum\beta_j^2
$$

```python
from sklearn.linear_model import Ridge

model = Ridge(alpha=1.0)

model.fit(X_train, y_train)
```

---

# 6. Outliers

An outlier is an observation that is unusually far from the general pattern.

Example:

```text
Salary

30k
32k
35k
31k
34k
33k
500k  ← outlier
```

That 500k observation could significantly affect the regression line.

---

# Why are outliers dangerous?

OLS minimizes:

$$
\sum_{i=1}^{n}(y_i-\hat y_i)^2
$$

Because the error is **squared**, large errors receive disproportionately large weight.

For example:

$$
2^2=4
$$

but:

$$
20^2=400
$$

Therefore, a single extreme observation can have a large effect.

---

# Detection 1: Box plot

```python
sns.boxplot(x=y)

plt.title("Target Variable Outliers")

plt.show()
```

For individual predictors:

```python
sns.boxplot(x=X["feature"])

plt.show()
```

---

# Detection 2: IQR method

The interquartile range is:

$$
IQR=Q_3-Q_1
$$

Lower boundary:

$$
Q_1-1.5(IQR)
$$

Upper boundary:

$$
Q_3+1.5(IQR)
$$

Python:

```python
Q1 = X["feature"].quantile(0.25)
Q3 = X["feature"].quantile(0.75)

IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

outliers = X[
    (X["feature"] < lower) |
    (X["feature"] > upper)
]

print(outliers)
```

---

# But there is an important distinction

A point can be:

- an outlier in **X**
- an outlier in **Y**
- an outlier in **residuals**
- influential because it has high leverage

These aren't exactly the same thing.

---

# 7. Leverage

Leverage asks:

> Is this observation unusual in terms of its predictor values?

Suppose most houses are:

```text
500 - 3000 sqft
```

but one house is:

```text
20,000 sqft
```

That observation has high leverage because its $X$ values are far from the rest of the data.

---

# High leverage does NOT automatically mean bad.

A high-leverage point can be completely valid.

The important question is whether it **strongly influences the fitted model**.

---

# Detection using leverage

With `statsmodels`:

```python
influence = ols_model.get_influence()

leverage = influence.hat_matrix_diag

print(leverage)
```

You can visualize it:

```python
plt.stem(leverage)

plt.xlabel("Observation")
plt.ylabel("Leverage")
plt.title("Leverage Values")

plt.show()
```

A common rough threshold is:

$$
h_i > \frac{2(p+1)}{n}
$$

where:

- $p$ = number of predictors
- $n$ = number of observations

For a more conservative rule, some use:

$$
h_i > \frac{3(p+1)}{n}
$$

---

# 8. Influential Observations

An observation can simultaneously have:

- unusual $X$
- unusual $Y$
- large residual

and therefore have a large effect on the regression coefficients.

This is an **influential observation**.

---

# Cook's Distance

One of the most common diagnostics is **Cook's distance**.

It essentially asks:

> How much would the fitted regression model change if this observation were removed?

---

## Python

```python
influence = ols_model.get_influence()

cooks_distance = influence.cooks_distance[0]

print(cooks_distance)
```

Plot:

```python
plt.stem(cooks_distance)

plt.xlabel("Observation")
plt.ylabel("Cook's Distance")
plt.title("Cook's Distance")

plt.show()
```

---

# Rule of thumb

A common rough threshold is:

$$
D_i>\frac{4}{n}
$$

where $n$ is the number of observations.

```python
threshold = 4 / len(X_train)

influential_points = np.where(
    cooks_distance > threshold
)[0]

print("Potential influential observations:")
print(influential_points)
```

Another commonly used rule is:

$$
D_i > 1
$$

but this is a rough heuristic, not a universal law.

---

# 9. Correct Model Specification

This is a deeper assumption.

Your model should include the **important variables and appropriate functional relationships**.

Suppose the true relationship is:

$$
Y=5+2X+3X^2+\epsilon
$$

but you fit:

$$
Y=\beta_0+\beta_1X+\epsilon
$$

You're missing an important nonlinear component.

Similarly, suppose:

```text
Salary = f(
    experience,
    education,
    location,
    industry
)
```

but you only use:

```text
Salary = f(experience)
```

You may have omitted important predictors.

---

# Detecting model specification problems

One useful test is the **Ramsey RESET test**.

```python
from statsmodels.stats.diagnostic import linear_reset

reset_result = linear_reset(
    ols_model,
    power=2,
    use_f=True
)

print(reset_result)
```

The RESET test can indicate that nonlinear combinations or omitted terms may be important.

Again, a significant result doesn't tell you exactly what variable or transformation is missing—it tells you that the current specification deserves investigation.

---

# 10. Exogeneity / Zero Conditional Mean

This is one of the most important theoretical assumptions.

We assume:

$$
E[\epsilon|X]=0
$$

In simple language:

> The error should not systematically depend on the predictors.

This means there should be no systematic relationship between the unexplained part of $Y$ and $X$.

---

# Example: omitted variable bias

Suppose you're predicting salary:

$$
Salary=\beta_0+\beta_1Experience+\epsilon
$$

But education also affects salary:

$$
Salary=
\beta_0+
\beta_1Experience+
\beta_2Education+
\epsilon
$$

If education is correlated with experience, and you omit education, then part of education's effect can get absorbed into the experience coefficient.

This creates **omitted variable bias**.

---

# Can we detect this perfectly?

No.

This is important.

You generally **cannot prove exogeneity from the dataset alone**.

You need:

- domain knowledge
- experimental design
- causal reasoning
- appropriate controls
- sometimes instrumental variables
- sometimes randomized experiments

This is why linear regression assumptions aren't all things you can simply "run a test" for.

---

# Complete Practical Diagnostic Workflow

Suppose you have:

```python
X = df.drop(columns=["target"])
y = df["target"]
```

Split your data:

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

Fit the model:

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_train)

residuals = y_train - y_pred
```

---

# Diagnostic 1 — Residual vs Fitted

```python
plt.figure(figsize=(8, 5))

plt.scatter(y_pred, residuals)

plt.axhline(
    y=0,
    linestyle="--"
)

plt.xlabel("Fitted Values")
plt.ylabel("Residuals")
plt.title("Residuals vs Fitted")

plt.show()
```

Look for:

- curvature → possible nonlinearity
- funnel → heteroscedasticity
- clusters → missing structure
- extreme points → possible outliers

---

# Diagnostic 2 — Q-Q Plot

```python
import statsmodels.api as sm

sm.qqplot(
    residuals,
    line="45"
)

plt.title("Q-Q Plot")

plt.show()
```

Look for:

> Points approximately following the diagonal line.

---

# Diagnostic 3 — Histogram

```python
sns.histplot(
    residuals,
    kde=True
)

plt.title("Residual Distribution")

plt.xlabel("Residual")

plt.show()
```

---

# Diagnostic 4 — Durbin-Watson

```python
from statsmodels.stats.stattools import durbin_watson

dw = durbin_watson(residuals)

print("Durbin-Watson:", dw)
```

Approximately:

```text
2 → good
<2 → positive autocorrelation
>2 → negative autocorrelation
```

---

# Diagnostic 5 — Breusch-Pagan

For this, it's easiest to fit the model with `statsmodels`.

```python
X_train_sm = sm.add_constant(X_train)

ols_model = sm.OLS(
    y_train,
    X_train_sm
).fit()
```

Then:

```python
from statsmodels.stats.diagnostic import het_breuschpagan

bp = het_breuschpagan(
    ols_model.resid,
    ols_model.model.exog
)

print("LM Statistic:", bp[0])
print("LM p-value:", bp[1])
print("F Statistic:", bp[2])
print("F p-value:", bp[3])
```

Focus primarily on the p-value.

---

# Diagnostic 6 — VIF

```python
from statsmodels.stats.outliers_influence import variance_inflation_factor

X_vif = sm.add_constant(X_train)

vif_df = pd.DataFrame()

vif_df["Feature"] = X_vif.columns

vif_df["VIF"] = [
    variance_inflation_factor(
        X_vif.values,
        i
    )
    for i in range(X_vif.shape[1])
]

print(vif_df)
```

---

# Diagnostic 7 — Cook's Distance

```python
influence = ols_model.get_influence()

cooks_d = influence.cooks_distance[0]

threshold = 4 / len(X_train)

print("Threshold:", threshold)

for i, value in enumerate(cooks_d):
    if value > threshold:
        print(
            f"Observation {i}: Cook's Distance = {value:.4f}"
        )
```

---

# Diagnostic 8 — Leverage

```python
leverage = influence.hat_matrix_diag

p = X_train.shape[1]
n = X_train.shape[0]

threshold = 2 * (p + 1) / n

print("Leverage threshold:", threshold)

high_leverage = np.where(
    leverage > threshold
)[0]

print("High leverage observations:")
print(high_leverage)
```

---

# A Complete Diagnostic Code

You can put many of these together:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from scipy.stats import shapiro

from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import (
    variance_inflation_factor
)


# --------------------------------
# 1. Split data
# --------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# --------------------------------
# 2. Train model
# --------------------------------

model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_train)

residuals = y_train - y_pred


# --------------------------------
# 3. Statsmodels model
# --------------------------------

X_train_sm = sm.add_constant(X_train)

ols_model = sm.OLS(
    y_train,
    X_train_sm
).fit()


# --------------------------------
# 4. Residual vs fitted
# --------------------------------

plt.figure(figsize=(8, 5))

plt.scatter(
    y_pred,
    residuals
)

plt.axhline(
    y=0,
    linestyle="--"
)

plt.xlabel("Fitted Values")
plt.ylabel("Residuals")
plt.title("Residuals vs Fitted")

plt.show()


# --------------------------------
# 5. Q-Q Plot
# --------------------------------

sm.qqplot(
    residuals,
    line="45"
)

plt.title("Q-Q Plot")

plt.show()


# --------------------------------
# 6. Histogram
# --------------------------------

sns.histplot(
    residuals,
    kde=True
)

plt.title("Residual Distribution")

plt.show()


# --------------------------------
# 7. Normality
# --------------------------------

stat, p = shapiro(residuals)

print("\nNormality Test")
print("----------------")
print("Statistic:", stat)
print("p-value:", p)


# --------------------------------
# 8. Independence
# --------------------------------

dw = durbin_watson(residuals)

print("\nDurbin-Watson")
print("----------------")
print("DW:", dw)


# --------------------------------
# 9. Homoscedasticity
# --------------------------------

bp = het_breuschpagan(
    ols_model.resid,
    ols_model.model.exog
)

print("\nBreusch-Pagan Test")
print("----------------")
print("LM Statistic:", bp[0])
print("LM p-value:", bp[1])
print("F Statistic:", bp[2])
print("F p-value:", bp[3])


# --------------------------------
# 10. Multicollinearity
# --------------------------------

X_vif = sm.add_constant(X_train)

vif_df = pd.DataFrame()

vif_df["Feature"] = X_vif.columns

vif_df["VIF"] = [
    variance_inflation_factor(
        X_vif.values,
        i
    )
    for i in range(X_vif.shape[1])
]

print("\nVIF")
print("----------------")
print(vif_df)


# --------------------------------
# 11. Cook's Distance
# --------------------------------

influence = ols_model.get_influence()

cooks_d = influence.cooks_distance[0]

threshold = 4 / len(X_train)

print("\nCook's Distance")
print("----------------")
print("Threshold:", threshold)

print(
    "Influential observations:",
    np.where(cooks_d > threshold)[0]
)


# --------------------------------
# 12. Leverage
# --------------------------------

leverage = influence.hat_matrix_diag

p = X_train.shape[1]
n = X_train.shape[0]

leverage_threshold = 2 * (p + 1) / n

print("\nLeverage")
print("----------------")
print("Threshold:", leverage_threshold)

print(
    "High leverage observations:",
    np.where(leverage > leverage_threshold)[0]
)
```

---

# How to Interpret Everything Together

Don't look at only one test.

Suppose you obtain:

```text
Shapiro p-value       = 0.21
Durbin-Watson         = 1.96
Breusch-Pagan p-value = 0.42
Maximum VIF           = 3.1
```

You could conclude:

```text
✓ Residual normality: reasonable
✓ Independence: reasonable
✓ Homoscedasticity: reasonable
✓ Multicollinearity: not concerning
```

But suppose:

```text
Shapiro p-value       = 0.001
Durbin-Watson         = 0.72
Breusch-Pagan p-value = 0.003
Maximum VIF           = 14.2
```

Now you have several concerns:

```text
✗ Non-normal residuals
✗ Positive autocorrelation
✗ Heteroscedasticity
✗ Severe multicollinearity
```

You would investigate the data/model before trusting coefficient inference.

---

# The Most Important Visuals

If you're doing linear regression in a real ML project, I would make these **four plots first**:

### 1. Actual vs Predicted

```python
plt.scatter(y_train, y_pred)

plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted")

plt.show()
```

Ideally:

```text
Predicted
   |
   |       *
   |     *
   |   *
   | *
   +---------------- Actual
```

---

### 2. Residuals vs Fitted

```python
plt.scatter(y_pred, residuals)

plt.axhline(0, linestyle="--")

plt.xlabel("Fitted")
plt.ylabel("Residual")
plt.title("Residuals vs Fitted")

plt.show()
```

Look for random scatter.

---

### 3. Q-Q Plot

```python
sm.qqplot(
    residuals,
    line="45"
)

plt.show()
```

Look for points approximately following the diagonal.

---

### 4. Cook's Distance

```python
plt.stem(cooks_d)

plt.xlabel("Observation")
plt.ylabel("Cook's Distance")

plt.show()
```

Look for unusually large observations.

---

# One Important Distinction for ML

There is a difference between **assumptions required for prediction** and **assumptions required for statistical inference**.

For example:

### Normal residuals

Not necessarily critical for:

```text
Prediction
```

But much more important for:

```text
Confidence intervals
Hypothesis tests
p-values
Coefficient inference
```

Similarly, heteroscedasticity doesn't necessarily mean:

> "Linear regression cannot predict."

It means that **ordinary least-squares inference can become unreliable**.

---

# Final Mental Model

Think of your regression diagnostics like this:

```text
                    LINEAR REGRESSION
                           │
          ┌────────────────┼────────────────┐
          │                │                │
     Relationship       Errors          Predictors
          │                │                │
       Linearity      ┌────┼────┐       Multicollinearity
                      │    │    │
                Independence  Normality
                      │
                Homoscedasticity
          │
     Data Problems
          │
    ┌─────┴─────────┐
    │               │
 Outliers       Influential
                Observations
```

And the practical diagnostic toolkit is:

| Problem | Main detection |
|---|---|
| **Non-linearity** | Residual vs fitted plot |
| **Autocorrelation** | Residual sequence + Durbin-Watson |
| **Heteroscedasticity** | Residual plot + Breusch-Pagan |
| **Non-normal residuals** | Q-Q plot + histogram + Shapiro |
| **Multicollinearity** | Correlation matrix + VIF |
| **Outliers** | Boxplot + standardized residuals |
| **High leverage** | Leverage / hat values |
| **Influential observations** | Cook's distance |
| **Wrong specification** | Residual patterns + RESET/domain knowledge |
| **Endogeneity** | Domain/causal reasoning; cannot generally be diagnosed by a simple residual test |

**The key idea:** don't just run statistical tests and blindly follow the p-values. **Start with diagnostic plots, then use statistical tests to support what you see, and finally use domain knowledge to decide what to do.**

---
