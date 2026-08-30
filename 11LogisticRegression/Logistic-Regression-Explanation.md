
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
