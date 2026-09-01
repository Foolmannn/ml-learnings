
# Loss Function, Maximum Likelihood & Binary Cross-Entropy in Logistic Regression

These three concepts are **closely connected**:

> **Logistic Regression → Probability → Likelihood → Maximum Likelihood → Negative Log-Likelihood → Binary Cross-Entropy Loss**

Understanding this chain makes logistic regression much easier to understand.

---

# 1. Why do we need a Loss Function?

Suppose we have a binary classification problem:

- `0` → Not spam
- `1` → Spam

Our model should predict the probability that an observation belongs to class `1`.

For example:

| Actual $y$ | Predicted probability $P(y=1)$ |
|---:|---:|
| 1 | 0.90 |
| 1 | 0.70 |
| 0 | 0.20 |
| 0 | 0.05 |

The model is doing reasonably well.

But how do we **measure how good or bad** these predictions are?

That's the job of a **loss function**.

A loss function gives us a numerical value representing prediction error.

$$
\boxed{\text{Loss} = \text{How bad was the prediction?}}
$$

Generally:

- Small loss → good prediction
- Large loss → bad prediction

---

# 2. Why not use Mean Squared Error?

You might already know MSE from Linear Regression:

$$
MSE = \frac{1}{n}\sum_{i=1}^{n}(y_i-\hat y_i)^2
$$

It might seem reasonable to use it for logistic regression.

But logistic regression is fundamentally based on **probability estimation**, and MSE combined with the sigmoid function gives a less convenient optimization problem.

Instead, we use **Log Loss / Binary Cross-Entropy**.

The reason for BCE becomes much clearer when we derive it from **Maximum Likelihood Estimation (MLE)**.

---

# 3. First understand Logistic Regression

Logistic regression starts with a linear equation:

$$
z = w_1x_1+w_2x_2+\cdots+w_nx_n+b
$$

or:

$$
\boxed{z=w^Tx+b}
$$

But $z$ can be anything:

$$
-\infty < z < +\infty
$$

We need a probability between 0 and 1.

So we apply the **sigmoid function**.

$$
\sigma(z)=\frac{1}{1+e^{-z}}
$$

Therefore:

$$
\boxed{\hat y = \sigma(w^Tx+b)}
$$

where:

$$
0<\hat y<1
$$

We interpret:

$$
\hat y=P(y=1|x)
$$

and therefore:

$$
1-\hat y=P(y=0|x)
$$

---

# 4. What does the sigmoid output mean?

Suppose:

$$
z=2
$$

Then:

$$
\sigma(2)=\frac{1}{1+e^{-2}}\approx0.881
$$

So:

$$
P(y=1|x)=0.881
$$

and:

$$
P(y=0|x)=1-0.881=0.119
$$

The model is saying:

> There is approximately an 88.1% probability that this observation belongs to class 1.

---

# 5. Where does Maximum Likelihood come from?

Now comes the important part.

Suppose we have training data:

$$
(x_1,y_1),(x_2,y_2),\ldots,(x_n,y_n)
$$

For each observation:

$$
y_i\in\{0,1\}
$$

Our model predicts:

$$
p_i=P(y_i=1|x_i)
$$

where:

$$
p_i=\sigma(w^Tx_i+b)
$$

We want to find the values of $w$ and $b$ that make our observed training data **most probable**.

This is called:

# Maximum Likelihood Estimation

---

# 6. What is Likelihood?

Likelihood asks:

> **Given our model parameters, how probable is the observed data?**

For one observation, suppose:

$$
y=1
$$

and the model predicts:

$$
p=0.8
$$

Then the probability of observing $y=1$ is:

$$
P(y=1)=0.8
$$

Now suppose:

$$
y=0
$$

The probability of observing $y=0$ is:

$$
P(y=0)=1-p
$$

So:

$$
P(y=0)=0.2
$$

---

# 7. Combining both cases

We can elegantly represent both cases using:

$$
\boxed{
P(y|x)=p^y(1-p)^{1-y}
}
$$

This is extremely important.

Let's see why.

### Case 1: $y=1$

$$
p^1(1-p)^0
$$

Since:

$$
(1-p)^0=1
$$

we get:

$$
P(y|x)=p
$$

Exactly what we want.

---

### Case 2: $y=0$

$$
p^0(1-p)^1
$$

Since:

$$
p^0=1
$$

we get:

$$
P(y|x)=1-p
$$

Again, exactly what we want.

Therefore:

$$
\boxed{
P(y|x)=p^y(1-p)^{1-y}
}
$$

---

# 8. Likelihood for the entire dataset

Assuming the training examples are independent, the probability of observing the entire dataset is the product of the individual probabilities.

Therefore:

$$
L(w,b)
=
\prod_{i=1}^{n}
p_i^{y_i}(1-p_i)^{1-y_i}
$$

where:

$$
p_i=\sigma(w^Tx_i+b)
$$

So:

$$
\boxed{
L(w,b)=
\prod_{i=1}^{n}
p_i^{y_i}(1-p_i)^{1-y_i}
}
$$

This is the **likelihood function**.

---

# 9. What does Maximum Likelihood mean?

We want to choose $w$ and $b$ such that:

$$
L(w,b)
$$

is as large as possible.

Therefore:

$$
\boxed{
\max_{w,b} L(w,b)
}
$$

In words:

> Find model parameters that make the observed training data as likely as possible.

---

# 10. Example of Maximum Likelihood

Suppose the true labels are:

$$
y=[1,0,1]
$$

And our model predicts:

$$
p=[0.9,0.8,0.7]
$$

The likelihood is:

$$
L=(0.9)(1-0.8)(0.7)
$$

$$
L=0.9\times0.2\times0.7
$$

$$
L=0.126
$$

Now suppose another model predicts:

$$
p=[0.8,0.1,0.9]
$$

Then:

$$
L=(0.8)(1-0.1)(0.9)
$$

$$
L=0.8\times0.9\times0.9
$$

$$
L=0.648
$$

The second model has a much larger likelihood.

Therefore, according to Maximum Likelihood:

$$
\boxed{\text{Model 2 is better}}
$$

---

# 11. Problem with multiplying probabilities

For a large dataset:

$$
L=\prod_{i=1}^{n}p_i^{y_i}(1-p_i)^{1-y_i}
$$

There might be thousands or millions of probabilities.

Multiplying many numbers between 0 and 1 can result in extremely small numbers.

For example:

$$
0.9^{1000}
$$

is extremely small.

So instead of maximizing the likelihood directly, we take the **logarithm**.

---

# 12. Log-Likelihood

Take logarithm:

$$
\log L
=
\log
\left[
\prod_{i=1}^{n}
p_i^{y_i}(1-p_i)^{1-y_i}
\right]
$$

Using:

$$
\log(ab)=\log a+\log b
$$

we get:

$$
\log L
=
\sum_{i=1}^{n}
\log
\left[
p_i^{y_i}(1-p_i)^{1-y_i}
\right]
$$

Using:

$$
\log(a^b)=b\log(a)
$$

we get:

$$
\boxed{
\log L
=
\sum_{i=1}^{n}
[
y_i\log(p_i)
+
(1-y_i)\log(1-p_i)
]
}
$$

This is the **log-likelihood**.

We want to maximize it:

$$
\boxed{
\max \log L
}
$$

---

# 13. But Machine Learning usually minimizes loss

Most ML optimization algorithms are formulated as:

$$
\boxed{\min Loss}
$$

But Maximum Likelihood says:

$$
\max \log L
$$

So we simply multiply by $-1$:

$$
\boxed{
\min -\log L
}
$$

Therefore:

$$
-\log L
=
-\sum_{i=1}^{n}
[
y_i\log(p_i)
+
(1-y_i)\log(1-p_i)
]
$$

If we divide by $n$, we get:

$$
\boxed{
J(w,b)
=
-\frac{1}{n}
\sum_{i=1}^{n}
[
y_i\log(p_i)
+
(1-y_i)\log(1-p_i)
]
}
$$

And this is:

# Binary Cross-Entropy Loss

Also called:

- Binary Log Loss
- Log Loss
- Negative Log-Likelihood (for Bernoulli logistic regression)

---

# 14. The Binary Cross-Entropy Formula

The fundamental formula is:

$$
\boxed{
BCE
=
-\frac{1}{n}
\sum_{i=1}^{n}
[
y_i\log(\hat y_i)
+
(1-y_i)\log(1-\hat y_i)
]
}
$$

where:

- $y_i$ = actual label
- $\hat y_i$ = predicted probability
- $n$ = number of observations

---

# 15. Understand BCE for one observation

For a single observation:

$$
\boxed{
Loss=
-[y\log(\hat y)+(1-y)\log(1-\hat y)]
}
$$

There are two cases.

---

## Case 1: Actual class is 1

Suppose:

$$
y=1
$$

Then:

$$
Loss
=
-[1\log(\hat y)+0\log(1-\hat y)]
$$

Therefore:

$$
\boxed{
Loss=-\log(\hat y)
}
$$

So if the actual answer is 1, we want $\hat y$ to be close to 1.

For example:

$$
\hat y=0.9
$$

$$
Loss=-\log(0.9)\approx0.105
$$

Very small loss.

But:

$$
\hat y=0.1
$$

$$
Loss=-\log(0.1)\approx2.303
$$

Large loss.

And if:

$$
\hat y=0.001
$$

then:

$$
Loss=-\log(0.001)\approx6.908
$$

Huge loss.

---

# 16. Case 2: Actual class is 0

Suppose:

$$
y=0
$$

Then:

$$
Loss
=
-[0\log(\hat y)+1\log(1-\hat y)]
$$

Therefore:

$$
\boxed{
Loss=-\log(1-\hat y)
}
$$

We want $\hat y$ close to 0.

For example:

$$
\hat y=0.1
$$

$$
Loss=-\log(0.9)
$$

$$
Loss\approx0.105
$$

Good.

But:

$$
\hat y=0.9
$$

$$
Loss=-\log(0.1)
$$

$$
Loss\approx2.303
$$

Bad.

---

# 17. Why does BCE punish confident wrong predictions so heavily?

This is one of the most important properties of BCE.

Suppose actual:

$$
y=1
$$

Consider:

| Prediction | Loss |
|---:|---:|
| 0.99 | 0.010 |
| 0.90 | 0.105 |
| 0.70 | 0.357 |
| 0.50 | 0.693 |
| 0.30 | 1.204 |
| 0.10 | 2.303 |
| 0.01 | 4.605 |

Notice what happens.

If the model confidently predicts:

$$
P(y=1)=0.01
$$

while the actual answer is 1, the model gets a **very large penalty**.

This is desirable because:

> Being confidently wrong is much worse than being uncertain.

---

# 18. Visual intuition

For $y=1$:

$$
Loss=-\log(\hat y)
$$

As $\hat y$ approaches 1:

$$
Loss\rightarrow0
$$

As $\hat y$ approaches 0:

$$
Loss\rightarrow\infty
$$

For $y=0$:

$$
Loss=-\log(1-\hat y)
$$

As $\hat y\rightarrow0$:

$$
Loss\rightarrow0
$$

As $\hat y\rightarrow1$:

$$
Loss\rightarrow\infty
$$

---

# 19. Why is it called "Cross-Entropy"?

This comes from information theory.

For a binary distribution, the true distribution can be represented as:

$$
[y,1-y]
$$

The predicted distribution is:

$$
[\hat y,1-\hat y]
$$

Cross-entropy between them is:

$$
H(P,Q)
=
-\sum_x P(x)\log Q(x)
$$

For binary classification:

$$
H(P,Q)
=
-
[
y\log(\hat y)
+
(1-y)\log(1-\hat y)
]
$$

Which is exactly our BCE loss.

So:

$$
\boxed{
\text{Binary Cross-Entropy}
=
\text{Cross-Entropy between true and predicted distributions}
}
$$

---

# 20. The complete chain

This is the most important thing to remember:

### Step 1 — Linear model

$$
z=w^Tx+b
$$

### Step 2 — Sigmoid

$$
\hat y=\sigma(z)
$$

$$
\hat y=\frac{1}{1+e^{-z}}
$$

### Step 3 — Interpret as probability

$$
\hat y=P(y=1|x)
$$

### Step 4 — Bernoulli probability

$$
P(y|x)
=
\hat y^y(1-\hat y)^{1-y}
$$

### Step 5 — Likelihood

$$
L=
\prod_i
\hat y_i^{y_i}
(1-\hat y_i)^{1-y_i}
$$

### Step 6 — Log likelihood

$$
\log L
=
\sum_i
[
y_i\log\hat y_i+
(1-y_i)\log(1-\hat y_i)
]
$$

### Step 7 — Negative log likelihood

$$
-\log L
$$

### Step 8 — Average

$$
\boxed{
BCE=
-\frac1n
\sum_i
[
y_i\log\hat y_i+
(1-y_i)\log(1-\hat y_i)
]
}
$$

Therefore:

$$
\boxed{
\text{MLE}
\Longleftrightarrow
\text{Maximize Likelihood}
\Longleftrightarrow
\text{Minimize Negative Log-Likelihood}
\Longleftrightarrow
\text{Minimize BCE}
}
$$

---

# 21. Why BCE is particularly suitable for Logistic Regression

There is a beautiful mathematical reason.

We have:

$$
\hat y=\sigma(z)
$$

where:

$$
z=w^Tx+b
$$

and BCE:

$$
L=
-[y\log(\hat y)+(1-y)\log(1-\hat y)]
$$

The derivative with respect to $z$ simplifies to:

$$
\boxed{
\frac{\partial L}{\partial z}=\hat y-y
}
$$

That's extremely convenient.

Since:

$$
z=w^Tx+b
$$

we obtain:

$$
\frac{\partial L}{\partial w}
=
(\hat y-y)x
$$

and:

$$
\boxed{
\frac{\partial L}{\partial b}=\hat y-y
}
$$

For the entire dataset:

$$
\boxed{
\frac{\partial J}{\partial w}
=
\frac1nX^T(\hat y-y)
}
$$

This makes gradient descent straightforward.

---

# 22. Example of one gradient calculation

Suppose:

$$
x=2
$$

Actual:

$$
y=1
$$

Prediction:

$$
\hat y=0.3
$$

Then:

$$
\frac{\partial L}{\partial z}
=
\hat y-y
$$

$$
=0.3-1
$$

$$
=-0.7
$$

For weight:

$$
\frac{\partial L}{\partial w}
=
(\hat y-y)x
$$

$$
=(-0.7)(2)
$$

$$
=-1.4
$$

The gradient is negative.

Gradient descent performs:

$$
w_{new}
=
w-\alpha\frac{\partial L}{\partial w}
$$

So:

$$
w_{new}
=
w-\alpha(-1.4)
$$

$$
w_{new}=w+1.4\alpha
$$

The weight increases, which tends to increase $z$, which increases the sigmoid probability.

That makes sense because:

> Actual = 1, but prediction = 0.3 → model needs to push its prediction upward.

---

# 23. What happens when prediction is already good?

Suppose:

$$
y=1
$$

and:

$$
\hat y=0.99
$$

Then:

$$
\frac{\partial L}{\partial z}
=
0.99-1
=
-0.01
$$

Very small gradient.

Therefore, the model makes only a small adjustment.

This is intuitive:

> The model is already predicting correctly with high confidence.

---

# 24. What happens when the model is confidently wrong?

Suppose:

$$
y=1
$$

but:

$$
\hat y=0.01
$$

Then:

$$
\frac{\partial L}{\partial z}
=
0.01-1
=
-0.99
$$

Large gradient.

The model gets a strong signal:

> "You are very wrong. Change your parameters."

---

# 25. BCE vs MSE

This distinction is useful for understanding why logistic regression normally uses BCE.

### MSE

$$
MSE=
\frac1n\sum(y-\hat y)^2
$$

### BCE

$$
BCE=
-\frac1n\sum
[
y\log\hat y+
(1-y)\log(1-\hat y)
]
$$

| Property | MSE | BCE |
|---|---|---|
| Common use | Regression | Binary classification |
| Output | Continuous value | Probability |
| Based on | Squared error | Likelihood / entropy |
| Logistic regression | Not standard | Standard |
| Probabilistic interpretation | Weaker here | Strong |
| Punishes confident wrong predictions | Less appropriately | Strongly |

---

# 26. Important distinction: Loss vs Cost vs Objective

You'll see these terms used somewhat interchangeably, but conceptually:

### Loss

Loss for **one training example**:

$$
\boxed{
L_i=
-[y_i\log\hat y_i+(1-y_i)\log(1-\hat y_i)]
}
$$

### Cost

Average loss over the dataset:

$$
\boxed{
J=
\frac1n\sum_iL_i
}
$$

Therefore:

$$
J=
-\frac1n\sum_i
[
y_i\log\hat y_i+
(1-y_i)\log(1-\hat y_i)
]
$$

### Objective

The function we are trying to optimize.

For logistic regression:

$$
\boxed{\min_{w,b}J(w,b)}
$$

---

# 27. Classification threshold is separate from BCE

This is another important concept.

The model outputs a probability:

$$
\hat y=0.73
$$

You might classify it as:

$$
\hat y\geq0.5\Rightarrow1
$$

and:

$$
\hat y<0.5\Rightarrow0
$$

But **BCE uses the probability itself**, not merely the final class.

For example:

$$
0.51
$$

and:

$$
0.99
$$

would both be classified as 1 using a 0.5 threshold.

But BCE considers them very different predictions.

That's one reason probability-based loss is useful.

---

# 28. A simple intuition

Imagine you're answering a yes/no question.

The model says:

> "I'm 99% sure the answer is YES."

If the answer is YES:

✅ Excellent prediction.

If the answer is NO:

❌ Extremely bad prediction.

BCE captures this.

But if the model says:

> "I'm only 51% sure it's YES."

and the answer is NO:

That's not nearly as bad.

Therefore BCE rewards:

**correct + confident**

and heavily penalizes:

**wrong + confident**

---

# 29. One complete numerical example

Suppose we have:

$$
y=[1,0,1,1]
$$

and predictions:

$$
\hat y=[0.9,0.2,0.8,0.4]
$$

Calculate each loss.

### Example 1

$$
y=1,\hat y=0.9
$$

$$
L_1=-\log(0.9)=0.105
$$

### Example 2

$$
y=0,\hat y=0.2
$$

$$
L_2=-\log(0.8)=0.223
$$

### Example 3

$$
y=1,\hat y=0.8
$$

$$
L_3=-\log(0.8)=0.223
$$

### Example 4

$$
y=1,\hat y=0.4
$$

$$
L_4=-\log(0.4)=0.916
$$

Average:

$$
BCE=
\frac{0.105+0.223+0.223+0.916}{4}
$$

$$
\boxed{BCE\approx0.367}
$$

That is the cost that the optimization algorithm tries to minimize.

---

# 30. The relationship with Bernoulli Distribution

Binary classification has two possible outcomes:

$$
y\in\{0,1\}
$$

This is modeled using a **Bernoulli distribution**:

$$
P(y|x)
=
p^y(1-p)^{1-y}
$$

Logistic regression predicts:

$$
p=\sigma(w^Tx+b)
$$

Then Maximum Likelihood estimates $w,b$.

So conceptually:

$$
\boxed{
\text{Logistic Regression}
=
\text{Bernoulli model}
+
\text{Sigmoid}
+
\text{Maximum Likelihood}
}
$$

And the resulting optimization objective is:

$$
\boxed{\text{Binary Cross-Entropy}}
$$

---

# 31. The big picture

You can visualize the entire process like this:

```text
                 INPUT
                   │
                   ▼
             x₁, x₂, ..., xₙ
                   │
                   ▼
             Linear Equation
             z = wᵀx + b
                   │
                   ▼
                Sigmoid
          p = 1 / (1 + e⁻ᶻ)
                   │
                   ▼
          Probability of class 1
                   │
                   ▼
       ┌─────────────────────────┐
       │                         │
     y = 1                     y = 0
       │                         │
       ▼                         ▼
   Probability p          Probability 1-p
       │                         │
       └────────────┬────────────┘
                    ▼
              Likelihood
                    │
                    ▼
             Log Likelihood
                    │
                    ▼
          Negative Log Likelihood
                    │
                    ▼
        Binary Cross-Entropy Loss
                    │
                    ▼
             Gradient Descent
                    │
                    ▼
             Update w and b
                    │
                    └──────► Repeat
```

---

# 32. The most important formulas to remember

### Logistic regression

$$
\boxed{
\hat y=\sigma(w^Tx+b)
}
$$

### Sigmoid

$$
\boxed{
\sigma(z)=\frac1{1+e^{-z}}
}
$$

### Bernoulli probability

$$
\boxed{
P(y|x)=
\hat y^y(1-\hat y)^{1-y}
}
$$

### Likelihood

$$
\boxed{
L=
\prod_i
\hat y_i^{y_i}(1-\hat y_i)^{1-y_i}
}
$$

### Log-likelihood

$$
\boxed{
\log L=
\sum_i[
y_i\log\hat y_i+
(1-y_i)\log(1-\hat y_i)
]
}
$$

### Binary Cross-Entropy

$$
\boxed{
BCE=
-\frac1n\sum_i[
y_i\log\hat y_i+
(1-y_i)\log(1-\hat y_i)
]
}
$$

### Gradient

$$
\boxed{
\frac{\partial L}{\partial z}
=
\hat y-y
}
$$

### Weight gradient

$$
\boxed{
\frac{\partial L}{\partial w}
=
(\hat y-y)x
}
$$

---

# 33. One-line mental model

If you want to remember the entire topic for exams/interviews:

> **Logistic regression predicts a probability using sigmoid. We choose the parameters that make the observed training labels most likely (Maximum Likelihood). Taking the negative log of that likelihood gives the Binary Cross-Entropy loss, which we minimize using gradient-based optimization.**

Or even shorter:

$$
\boxed{
\text{Sigmoid}
\rightarrow
\text{Probability}
\rightarrow
\text{Likelihood}
\rightarrow
\text{Log}
\rightarrow
\text{Negative}
\rightarrow
\text{BCE}
}
$$

This connection is **the core mathematical foundation of logistic regression**.