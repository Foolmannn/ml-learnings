
# Stochastic Gradient Descent (SGD) in Machine Learning

Stochastic Gradient Descent is one of the most important optimization algorithms used to train machine-learning models, especially when the dataset is large.

It is a variation of **Gradient Descent** where the model parameters are updated using **one training example at a time**, rather than using the entire dataset.

---

## 1. Why do we need Gradient Descent?

Suppose we have a linear regression model:

$$
\hat y = wx+b
$$

where:

- $w$ = weight
- $b$ = bias
- $x$ = input
- $\hat y$ = predicted value

We want to find values of $w$ and $b$ that minimize the error between actual and predicted values.

For example:

$$
y = [2,4,6,8]
$$

and our model should learn something approximately like:

$$
y = 2x
$$

We define a **loss function** to measure how wrong our predictions are.

For linear regression, a common loss is Mean Squared Error:

$$
J(w,b)=\frac{1}{2m}\sum_{i=1}^{m}(\hat y_i-y_i)^2
$$

Gradient Descent finds the minimum of this function.

---

# 2. What is Stochastic Gradient Descent?

In **Batch Gradient Descent**, we calculate the gradient using **all training examples**:

$$
w := w-\eta\frac{\partial J}{\partial w}
$$

In **Stochastic Gradient Descent**, we randomly/selectively take **one training example**:

$$
w := w-\eta\frac{\partial J_i}{\partial w}
$$

and similarly:

$$
b := b-\eta\frac{\partial J_i}{\partial b}
$$

where:

- $\eta$ = learning rate
- $J_i$ = loss for the $i^{th}$ training example

So the key difference is:

> **SGD updates the parameters after every individual training example.**

---

# 3. Batch GD vs SGD

Suppose you have:

$$
1,000,000
$$

training examples.

### Batch Gradient Descent

One update requires processing:

$$
1,000,000
$$

examples.

So:

```text
Dataset
   ↓
Process all 1,000,000 examples
   ↓
Calculate gradient
   ↓
Update parameters
```
