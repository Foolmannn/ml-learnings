

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

