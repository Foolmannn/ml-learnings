
# Logistic Regression in Detail

Logistic Regression is one of the most important **supervised machine-learning algorithms for classification**.

Despite its name, **Logistic Regression is primarily a classification algorithm**, not a regression algorithm.

It is especially useful when the target variable is categorical, such as:

- Spam / Not Spam
- Disease / No Disease
- Fraud / Not Fraud
- Pass / Fail
- Customer Churn / No Churn
- 0 / 1

---

# 1. What is Logistic Regression?

Suppose we want to predict whether a student passes an exam based on study hours.

Our target is:

$$
y =
\begin{cases}
0 & \text{Fail}\\
1 & \text{Pass}
\end{cases}
$$

A first idea might be to use Linear Regression:

$$
\hat y = b_0+b_1x
$$

But this creates a problem.

For example, Linear Regression might produce:

$$
\hat y=-0.4
$$

or

$$
\hat y=1.7
$$

Those values don't make sense as probabilities.

We need an output between **0 and 1**.

That's where Logistic Regression comes in.

---

# 2. Main idea

Logistic Regression works in two major steps:

### Step 1 — Calculate a linear combination

$$
z=b_0+b_1x_1+b_2x_2+\cdots+b_nx_n
$$

This is exactly like linear regression.

### Step 2 — Pass that value through the sigmoid function

$$
P(y=1|X)=\sigma(z)
$$

where

$$
\sigma(z)=\frac{1}{1+e^{-z}}
$$

So the complete equation is:

$$
\boxed{
P(y=1|X)=
\frac{1}
{1+e^{-(b_0+b_1x_1+\cdots+b_nx_n)}}
}
$$

The sigmoid converts any value from:

$$
-\infty \rightarrow +\infty
$$

into:

$$
0\rightarrow1
$$

---

# 3. The Sigmoid Function

The sigmoid function is the heart of Logistic Regression.

$$
\sigma(z)=\frac{1}{1+e^{-z}}
$$

Its shape looks like an **S**.

Important values:

| $z$ | Sigmoid |
|---:|---:|
| -10 | ≈ 0 |
| -5 | ≈ 0.007 |
| -2 | ≈ 0.119 |
| -1 | ≈ 0.269 |
| 0 | 0.5 |
| 1 | ≈ 0.731 |
| 2 | ≈ 0.881 |
| 5 | ≈ 0.993 |
| 10 | ≈ 1 |

The important point is:

$$
z=0 \Rightarrow P(y=1)=0.5
$$

So the model transforms its raw score into a probability.

---

# 4. Probability vs Prediction

This distinction is extremely important.

Suppose Logistic Regression produces:

$$
P(y=1)=0.82
$$

This means the model estimates an **82% probability** of class 1.

It does not automatically mean the final prediction is 1.

We need a **classification threshold**.

The default threshold is usually:

$$
0.5
$$

Therefore:

$$
\hat y =
\begin{cases}
1 & P(y=1)\geq0.5\\
0 & P(y=1)<0.5
\end{cases}
$$

You can experiment with the threshold here:



For example:

```text
Probability = 0.82
Threshold   = 0.50
Prediction  = 1
```

But:

```text
Probability = 0.42
Threshold   = 0.50
Prediction  = 0
```

---

# 5. Why not Linear Regression?

This is a very common interview question.

Suppose we use Linear Regression for binary classification:

$$
y=b_0+b_1x
$$

It could predict:

```text
-0.4
 0.2
 0.8
 1.4
```

The values outside `[0,1]` are not valid probabilities.

Logistic Regression solves this using sigmoid:

$$
z \rightarrow \sigma(z)
$$

For example:

$$
z=3
$$

becomes:

$$
\sigma(3)\approx0.953
$$

And:

$$
z=-3
$$

becomes:

$$
\sigma(-3)\approx0.047
$$

Therefore:

$$
0\leq P(y=1)\leq1
$$

---

# 6. A Simple Example

Suppose:

$$
z=-4+1.2x
$$

where $x$ represents study hours.

For 2 hours:

$$
z=-4+(1.2)(2)
$$

$$
z=-1.6
$$

Now apply sigmoid:

$$
P(y=1)=
\frac{1}{1+e^{1.6}}
$$

Approximately:

$$
P(y=1)=0.168
$$

So there is approximately a **16.8% predicted probability of passing**.

For 5 hours:

$$
z=-4+(1.2)(5)
$$

$$
z=2
$$

Then:

$$
P(y=1)=0.881
$$

So:

$$
P(\text{pass})\approx88.1\%
$$

With threshold 0.5:

```text
2 hours → 0.168 → Class 0
5 hours → 0.881 → Class 1
```

---

# 7. What does the model actually learn?

Suppose we have:

$$
z=b_0+b_1x_1+b_2x_2
$$

The model learns:

- $b_0$ → intercept
- $b_1,b_2$ → coefficients

For example:

$$
z=-3+0.8x_1+1.5x_2
$$

Suppose:

- $x_1$ = study hours
- $x_2$ = attendance ratio

Then:

$$
z=-3+0.8(\text{study hours})+1.5(\text{attendance})
$$

The coefficients determine how each feature changes the model's output.

---

# 8. Interpretation of Coefficients

This part is particularly important.

A Logistic Regression coefficient does **not directly represent the change in probability**.

Instead, it represents a change in **log-odds**.

The fundamental equation is:

$$
\log\left(\frac{p}{1-p}\right)
=
b_0+b_1x_1+\cdots+b_nx_n
$$

This is called the **logit function**.

---

# 9. Odds

Before understanding coefficients, we need to understand odds.

If:

$$
P(\text{success})=0.8
$$

then:

$$
\text{odds}=
\frac{p}{1-p}
$$

Therefore:

$$
\text{odds}=\frac{0.8}{0.2}=4
$$

Meaning success is 4 times as likely as failure.

Another example:

$$
p=0.2
$$

Then:

$$
\text{odds}=\frac{0.2}{0.8}=0.25
$$

---

# 10. Log-Odds

Logistic Regression models:

$$
\log(\text{odds})
$$

Therefore:

$$
\log\left(\frac{p}{1-p}\right)
=
b_0+b_1x_1+\cdots+b_nx_n
$$

This is the core mathematical relationship behind Logistic Regression.

---

# 11. Deriving the Sigmoid from Log-Odds

Start with:

$$
\log\left(\frac{p}{1-p}\right)=z
$$

Exponentiate both sides:

$$
\frac{p}{1-p}=e^z
$$

Multiply:

$$
p=e^z(1-p)
$$

$$
p=e^z-e^zp
$$

Move terms:

$$
p+e^zp=e^z
$$

$$
p(1+e^z)=e^z
$$

Therefore:

$$
p=\frac{e^z}{1+e^z}
$$

Divide numerator and denominator by $e^z$:

$$
\boxed{
p=\frac{1}{1+e^{-z}}
}
$$

That's the sigmoid function.

---

# 12. Meaning of a Positive Coefficient

Suppose:

$$
b_1=2
$$

A positive coefficient means increasing $x_1$ increases the **log-odds** of class 1.

Since:

$$
e^{b_1}
$$

is the **odds ratio**, we have:

$$
e^2\approx7.39
$$

So a one-unit increase in $x_1$ multiplies the odds by approximately 7.39, assuming other features remain constant.

---

# 13. Meaning of a Negative Coefficient

Suppose:

$$
b_1=-2
$$

Then:

$$
e^{-2}\approx0.135
$$

A one-unit increase in the feature multiplies the odds by approximately 0.135.

So the odds decrease.

General interpretation:

```text
Coefficient > 0 → odds increase
Coefficient < 0 → odds decrease
Coefficient = 0 → no effect on log-odds
```

---

# 14. Decision Boundary

The classifier needs to determine where class 0 changes to class 1.

With threshold 0.5:

$$
P(y=1)=0.5
$$

Since sigmoid(0) = 0.5:

$$
z=0
$$

Therefore:

$$
b_0+b_1x_1+\cdots+b_nx_n=0
$$

This equation represents the **decision boundary**.

For two features:

$$
b_0+b_1x_1+b_2x_2=0
$$

This produces a straight line.

For three features, it produces a plane.

For many dimensions, it is called a **hyperplane**.

---

# 15. Important Insight

Although the output is nonlinear because of the sigmoid function, the **decision boundary is linear**.

For example:

$$
z=b_0+b_1x_1+b_2x_2
$$

The boundary occurs when:

$$
z=0
$$

which gives:

$$
x_2=
-\frac{b_0+b_1x_1}{b_2}
$$

That's a straight line.

This is why standard Logistic Regression is called a **linear classifier**.

---

# 16. How Logistic Regression is Trained

This is where Logistic Regression differs significantly from Linear Regression.

Linear Regression commonly minimizes:

$$
MSE
$$

Logistic Regression commonly uses **Log Loss / Binary Cross Entropy**.

The loss for one training example is:

$$
L(y,p)
=
-\left[y\log(p)+(1-y)\log(1-p)\right]
$$

where:

- $y$ = actual class
- $p$ = predicted probability

---

# 17. Understanding Binary Cross Entropy

Suppose actual value:

$$
y=1
$$

Then:

$$
L=-\log(p)
$$

If model predicts:

$$
p=0.9
$$

then:

$$
L=-\log(0.9)\approx0.105
$$

Very small loss.

But if:

$$
p=0.1
$$

then:

$$
L=-\log(0.1)\approx2.303
$$

Large loss.

So the model strongly penalizes **confident incorrect predictions**.

---

# 18. For Class 0

If:

$$
y=0
$$

then:

$$
L=-\log(1-p)
$$

Suppose:

$$
p=0.1
$$

Then:

$$
L=-\log(0.9)\approx0.105
$$

Good prediction.

But if:

$$
p=0.9
$$

then:

$$
L=-\log(0.1)\approx2.303
$$

Bad prediction.

---

# 19. Dataset Loss

For $m$ training examples:

$$
J(\theta)
=
-\frac{1}{m}
\sum_{i=1}^{m}
\left[
y_i\log(p_i)
+
(1-y_i)\log(1-p_i)
\right]
$$

The training algorithm tries to find parameters that minimize this loss.

---

# 20. Gradient Descent

Logistic Regression can be trained using gradient-based optimization.

The general update rule is:

$$
\theta_j
=
\theta_j
-
\alpha
\frac{\partial J}{\partial\theta_j}
$$

where:

- $\theta_j$ = parameter
- $\alpha$ = learning rate
- $J$ = cost function

For Logistic Regression:

$$
\frac{\partial J}{\partial\theta}
=
\frac{1}{m}X^T(p-y)
$$

Therefore:

$$
\boxed{
\theta
\leftarrow
\theta-
\alpha
\frac{1}{m}X^T(p-y)
}
$$

This is conceptually similar to gradient descent in linear regression, but the prediction $p$ comes from the sigmoid function.

---

# 21. Why MSE is Generally Not Preferred

You might wonder:

> Why not use Mean Squared Error with sigmoid?

It can be used in some settings, but binary cross-entropy is preferred because it gives a better-behaved optimization objective for probabilistic classification.

Binary cross-entropy comes naturally from the **maximum likelihood estimation** of Bernoulli-distributed outcomes.

---

# 22. Maximum Likelihood Estimation

Another way to understand Logistic Regression is through **Maximum Likelihood Estimation (MLE)**.

Suppose:

$$
y_i\in\{0,1\}
$$

and:

$$
P(y_i=1|x_i)=p_i
$$

For one observation:

$$
P(y_i|x_i)
=
p_i^{y_i}(1-p_i)^{1-y_i}
$$

For all observations:

$$
L(\theta)
=
\prod_{i=1}^{m}
p_i^{y_i}(1-p_i)^{1-y_i}
$$

We want parameters that maximize this likelihood.

Taking logarithm:

$$
\log L(\theta)
=
\sum_i
[
y_i\log p_i+
(1-y_i)\log(1-p_i)
]
$$

Maximizing log-likelihood is equivalent to minimizing:

$$
-\log L(\theta)
$$

which gives binary cross-entropy.

So:

```text
Maximum Likelihood
        ↓
Log Likelihood
        ↓
Negative Log Likelihood
        ↓
Binary Cross Entropy
```

---

# 23. Logistic Regression with Multiple Features

Suppose we have:

```text
Age
Salary
Credit Score
Years of Employment
```

The model might be:

$$
z=
b_0+
b_1(\text{Age})+
b_2(\text{Salary})+
b_3(\text{Credit Score})+
b_4(\text{Employment Years})
$$

Then:

$$
p=\sigma(z)
$$

Finally:

$$
\hat y=
\begin{cases}
1 & p\ge0.5\\
0 & p<0.5
\end{cases}
$$

---

# 24. Logistic Regression Workflow

A typical workflow is:

```text
Dataset
   ↓
Data Cleaning
   ↓
Feature Selection / Engineering
   ↓
Train-Test Split
   ↓
Feature Scaling (when appropriate)
   ↓
Train Logistic Regression
   ↓
Predict Probability
   ↓
Choose Threshold
   ↓
Convert Probability → Class
   ↓
Evaluate Model
```

---

# 25. Do We Need Feature Scaling?

Logistic Regression does **not mathematically require** scaling.

However, scaling is usually beneficial when:

- features have very different magnitudes
- gradient-based optimization is used
- regularization is used
- you want faster/more stable optimization

Example:

```text
Age              → 20–60
Salary           → 20,000–500,000
Credit Score     → 300–850
```

Scaling can make optimization easier.

A common choice:

```python
from sklearn.preprocessing import StandardScaler
```

---

# 26. Logistic Regression in Scikit-Learn

Simple binary classification:

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
```

For probabilities:

```python
y_prob = model.predict_proba(X_test)
```

For a binary classifier:

```python
y_prob[:, 1]
```

contains the probability of class 1.

---

# 27. Complete Example

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

Notice the important rule:

```python
scaler.fit_transform(X_train)
scaler.transform(X_test)
```

Do **not** fit the scaler separately on the test set.

---

# 28. `predict()` vs `predict_proba()`

This distinction is very important.

### `predict()`

Returns the final class:

```python
model.predict(X_test)
```

Example:

```text
[0, 1, 1, 0, 1]
```

### `predict_proba()`

Returns probabilities:

```python
model.predict_proba(X_test)
```

Example:

```text
[[0.80, 0.20],
 [0.10, 0.90],
 [0.25, 0.75]]
```

Each row is:

```text
[P(class 0), P(class 1)]
```

---

# 29. Changing the Classification Threshold

The default threshold is typically:

$$
0.5
$$

But you can choose another threshold.

```python
y_prob = model.predict_proba(X_test)[:, 1]

y_pred = (y_prob >= 0.7).astype(int)
```

Now:

$$
P(y=1)\ge0.7
$$

is required to classify an observation as class 1.

This can be extremely useful for:

- fraud detection
- medical screening
- spam detection
- anomaly detection

because the costs of false positives and false negatives may differ.

---

# 30. Confusion Matrix

For binary classification:

| | Predicted 0 | Predicted 1 |
|---|---:|---:|
| Actual 0 | TN | FP |
| Actual 1 | FN | TP |

Where:

- **TP** = True Positive
- **TN** = True Negative
- **FP** = False Positive
- **FN** = False Negative

---

# 31. Accuracy

$$
Accuracy=
\frac{TP+TN}
{TP+TN+FP+FN}
$$

Accuracy answers:

> How many predictions were correct overall?

But accuracy can be misleading with imbalanced data.

Example:

```text
1000 transactions
990 legitimate
10 fraud
```

A model predicting everything as legitimate gets:

$$
99\%
$$

accuracy.

Yet it detects:

$$
0
$$

fraud cases.

---

# 32. Precision

$$
Precision=
\frac{TP}{TP+FP}
$$

It answers:

> Of everything predicted positive, how much was actually positive?

High precision means fewer false positives.

Useful when false alarms are expensive.

---

# 33. Recall

$$
Recall=
\frac{TP}{TP+FN}
$$

It answers:

> Of all actual positive cases, how many did we find?

High recall means fewer false negatives.

Important in things like disease screening and fraud detection.

---

# 34. F1 Score

$$
F1=
2\frac{Precision\cdot Recall}
{Precision+Recall}
$$

It balances precision and recall.

F1 is particularly useful when:

- classes are imbalanced
- both FP and FN matter

---

# 35. ROC Curve

The ROC curve plots:

$$
TPR
$$

against:

$$
FPR
$$

where:

$$
TPR=\frac{TP}{TP+FN}
$$

and:

$$
FPR=\frac{FP}{FP+TN}
$$

Different probability thresholds produce different points on the curve.

---

# 36. AUC

AUC means **Area Under the ROC Curve**.

Generally:

```text
AUC ≈ 1.0 → excellent discrimination
AUC ≈ 0.5 → random-like discrimination
AUC < 0.5 → worse than random
```

AUC evaluates how well the model **ranks positive examples above negative examples across thresholds**.

---

# 37. Regularization

Logistic Regression often uses regularization to prevent overfitting.

Two major types are:

### L1 Regularization

Also called **Lasso**.

$$
Loss + \lambda\sum_j|w_j|
$$

It can make some coefficients exactly zero.

Therefore:

```text
Feature A → 2.4
Feature B → 0
Feature C → -1.7
Feature D → 0
```

This can perform feature selection.

---

### L2 Regularization

Also called **Ridge**.

$$
Loss+\lambda\sum_jw_j^2
$$

It shrinks coefficients toward zero but typically does not make them exactly zero.

---

# 38. Logistic Regression Regularization in sklearn

This:

```python
model = LogisticRegression()
```

uses regularization by default in standard scikit-learn usage.

The `C` parameter controls inverse regularization strength:

$$
C=\frac{1}{\lambda}
$$

Therefore:

```text
Small C → stronger regularization
Large C → weaker regularization
```

Example:

```python
LogisticRegression(C=0.1)
```

has stronger regularization than:

```python
LogisticRegression(C=10)
```

---

# 39. Multiclass Logistic Regression

Logistic Regression is not limited to:

```text
0 vs 1
```

It can also classify:

```text
Cat
Dog
Horse
```

There are common approaches:

### One-vs-Rest (OvR)

Train:

```text
Cat vs all
Dog vs all
Horse vs all
```

Then choose the strongest score.

### Multinomial Logistic Regression

The model directly estimates probabilities across all classes using the **softmax function**.

For class $k$:

$$
P(y=k|X)
=
\frac{e^{z_k}}
{\sum_j e^{z_j}}
$$

Unlike sigmoid, softmax produces probabilities across multiple mutually exclusive classes whose sum is 1.

---

# 40. Logistic Regression vs Linear Regression

| Property | Linear Regression | Logistic Regression |
|---|---|---|
| Main task | Regression | Classification |
| Output | Continuous | Probability |
| Activation | None | Sigmoid |
| Typical target | Numeric | Categorical |
| Loss | MSE commonly | Log Loss |
| Boundary | N/A | Linear |
| Output range | $-\infty,\infty$ | 0–1 |
| Example | House price | Spam detection |

---

# 41. Logistic Regression vs Decision Tree

| Property | Logistic Regression | Decision Tree |
|---|---|---|
| Type | Linear model | Tree model |
| Boundary | Linear | Nonlinear |
| Interpretability | High | High |
| Feature scaling | Often useful | Usually unnecessary |
| Handles nonlinear relationships | Poorly without feature engineering | Well |
| Probability output | Yes | Yes |
| Overfitting control | Regularization | Depth/pruning/etc. |

---

# 42. Logistic Regression vs Random Forest

Logistic Regression:

```text
simple
fast
interpretable
linear boundary
```

Random Forest:

```text
nonlinear
more flexible
handles feature interactions naturally
usually less interpretable
```

A good practical approach is often to start with Logistic Regression as a **strong baseline**.

---

# 43. Assumptions of Logistic Regression

Logistic Regression has several important assumptions.

### 1. Binary/multiclass target

The target should represent appropriate categorical outcomes.

### 2. Independent observations

Observations should generally be independent.

### 3. Linear relationship with log-odds

This is very important.

Logistic Regression does **not** assume that:

$$
X \rightarrow P(Y=1)
$$

is linear.

It assumes:

$$
X \rightarrow \log\left(\frac{p}{1-p}\right)
$$

is linear.

### 4. No severe multicollinearity

Highly correlated predictors can make coefficient estimates unstable.

### 5. Sufficient sample information

Very small or sparse datasets can create unstable estimates.

---

# 44. Multicollinearity

Suppose:

```text
X1 = Age
X2 = Years of experience
```

These may be highly correlated.

Then it becomes difficult for Logistic Regression to determine the individual contribution of each feature.

Consequences can include:

- unstable coefficients
- large standard errors
- confusing coefficient interpretation

Regularization can help.

---

# 45. Outliers

Logistic Regression can be affected by influential observations.

Because the model estimates coefficients based on the training data, extreme observations can significantly alter the fitted decision boundary.

This is one reason to inspect:

- distributions
- leverage/influence
- feature scaling
- data quality

---

# 46. Decision Boundary Example

Suppose:

$$
z=-5+2x_1+x_2
$$

The boundary is:

$$
-5+2x_1+x_2=0
$$

Therefore:

$$
x_2=5-2x_1
$$

Points on one side are classified as one class, and points on the other side as the other class.

This is why Logistic Regression works best when classes are reasonably separable by a linear boundary.

---

# 47. Feature Engineering Can Make Logistic Regression More Powerful

Suppose the actual relationship is curved.

You can introduce polynomial features:

$$
x^2
$$

or interaction terms:

$$
x_1x_2
$$

Then Logistic Regression can model:

$$
z=b_0+b_1x+b_2x^2
$$

The boundary in the **original feature space** can now become nonlinear.

So although Logistic Regression itself is linear in its parameters/features, feature engineering can allow much richer decision boundaries.

---

# 48. Why Logistic Regression is Popular

It has several advantages:

- simple
- fast
- interpretable
- probability-based
- strong baseline
- works well with high-dimensional sparse data
- regularization is straightforward
- easy to implement
- often effective for text classification

For many datasets, a well-regularized Logistic Regression model can be surprisingly competitive.

---

# 49. Limitations

Logistic Regression struggles when:

### Strong nonlinear relationships

For example, if the classes form circles:

```text
       00000
    000111000
   0011111000
    000111000
       00000
```

A simple straight decision boundary won't work well.

### Complex feature interactions

It cannot automatically discover complex interactions the way tree-based models can.

### Strongly overlapping classes

If the classes are inherently difficult to separate, changing the algorithm may not solve the fundamental problem.

---

# 50. Complete Mental Model

You can remember Logistic Regression as:

```text
Features
   ↓
Weighted Sum
   ↓
z = b0 + b1x1 + b2x2 + ...
   ↓
Sigmoid
   ↓
Probability
   ↓
Threshold
   ↓
Class
```

Mathematically:

$$
X
\rightarrow
z=Xw+b
\rightarrow
\sigma(z)
\rightarrow
P(y=1)
\rightarrow
\hat y
$$

---

# 51. The Most Important Equations

### Linear score

$$
\boxed{z=Xw+b}
$$

### Sigmoid

$$
\boxed{\sigma(z)=\frac{1}{1+e^{-z}}}
$$

### Probability

$$
\boxed{P(y=1|X)=\sigma(Xw+b)}
$$

### Logit

$$
\boxed{
\log\left(\frac{p}{1-p}\right)=Xw+b
}
$$

### Binary cross entropy

$$
\boxed{
J=
-\frac{1}{m}
\sum
[y\log p+(1-y)\log(1-p)]
}
$$

### Decision boundary

$$
\boxed{Xw+b=0}
$$

### Odds ratio

$$
\boxed{OR=e^{w_j}}
$$

---

# 52. A Practical Scikit-Learn Example

Here's a more realistic example:

```python
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)

# Data
X = df.drop("target", axis=1)
y = df["target"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model
model = LogisticRegression(
    C=1.0,
    max_iter=1000
)

# Training
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Probability
y_prob = model.predict_proba(X_test)[:, 1]

# Metrics
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1:", f1_score(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_prob))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
```

---

# 53. Understanding the Entire Algorithm from Scratch

Suppose we have:

```text
Study Hours → Pass

1 → 0
2 → 0
3 → 0
4 → 1
5 → 1
6 → 1
```

The model starts with parameters such as:

$$
w=0,\quad b=0
$$

For each sample:

### Linear computation

$$
z=wx+b
$$

### Sigmoid

$$
p=\frac{1}{1+e^{-z}}
$$

### Calculate loss

$$
L=
-[y\log p+(1-y)\log(1-p)]
$$

### Calculate gradients

Determine how each parameter contributes to the loss.

### Update parameters

$$
w\leftarrow w-\alpha\frac{\partial J}{\partial w}
$$

$$
b\leftarrow b-\alpha\frac{\partial J}{\partial b}
$$

Repeat many times.

Eventually the model learns a boundary such as:

```text
Study hours < 3.7  → Fail
Study hours > 3.7  → Pass
```

with probabilities rather than merely hard labels.

---

# 54. One Very Important Distinction

There are **three different things** that people often mix up:

### Score

$$
z=Xw+b
$$

Can be any real number.

### Probability

$$
p=\sigma(z)
$$

Always between 0 and 1.

### Class

$$
\hat y=
\begin{cases}
1&p\ge threshold\\
0&p<threshold
\end{cases}
$$

So:

```text
Raw score → Probability → Class
```

For example:

```text
z = 1.5
    ↓
sigmoid
    ↓
p = 0.818
    ↓
threshold = 0.5
    ↓
class = 1
```

That distinction will help you understand almost every classification algorithm afterward.

---

# 55. Interview Questions You Should Know

**Why is Logistic Regression called regression?**

Because it models a linear combination of features and estimates the log-odds; historically it is also framed as a regression of the logit, despite being used for classification.

**Why sigmoid?**

To map the linear score into a value between 0 and 1 that can be interpreted as a probability.

**What loss function does Logistic Regression use?**

Binary cross-entropy / log loss, commonly derived from Bernoulli maximum likelihood.

**What does a coefficient mean?**

It represents the change in log-odds for a one-unit increase in the feature, holding other features constant.

**What is $e^{w_j}$?**

The odds ratio associated with a one-unit increase in feature $j$.

**Why is 0.5 used as the threshold?**

It is the natural default threshold corresponding to $z=0$, but it is not universally optimal.

**Can Logistic Regression handle multiclass classification?**

Yes, using approaches such as OvR or multinomial logistic regression.

**Does Logistic Regression require feature scaling?**

Not mathematically, but scaling is often useful, especially with regularization and gradient-based optimization.

**Is Logistic Regression a linear or nonlinear model?**

It is a **linear classifier** in the input features because its decision boundary satisfies $Xw+b=0$.

---

## Final Mental Picture

Think of Logistic Regression as a pipeline:

$$
\boxed{
\text{Features}
\rightarrow
\text{Linear Score}
\rightarrow
\text{Sigmoid}
\rightarrow
\text{Probability}
\rightarrow
\text{Threshold}
\rightarrow
\text{Class}
}
$$

And the training process as:

$$
\boxed{
\text{Predict probability}
\rightarrow
\text{Calculate log loss}
\rightarrow
\text{Calculate gradient}
\rightarrow
\text{Update weights}
\rightarrow
\text{Repeat}
}
$$

The **three concepts I would make absolutely sure you understand** are:

1. **Sigmoid and why it converts the linear score into probability**
2. **Log-odds and what Logistic Regression coefficients actually mean**
3. **Log loss + gradient descent and how the weights are learned**

These three form the foundation for understanding Logistic Regression mathematically rather than just knowing how to call `LogisticRegression()` in scikit-learn.