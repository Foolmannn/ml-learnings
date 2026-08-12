
# Batch Gradient Descent in Machine Learning

**Batch Gradient Descent (BGD)** is an optimization algorithm used to find the parameters of a machine-learning model that minimize a **loss/cost function**.

It is especially important for understanding how models such as **Linear Regression, Logistic Regression, and neural networks** learn their parameters.

---

## 1. The basic idea

Suppose we have a linear regression model:

$$
\hat y = wx+b
$$

where:

- $x$ = input feature
- $y$ = actual target
- $w$ = weight
- $b$ = bias
- $\hat y$ = predicted value

Initially, $w$ and $b$ are usually initialized to some values.

The model makes predictions:

$$
\hat y_i = wx_i+b
$$

Then we calculate how wrong those predictions are using a **cost function**.

For linear regression, a common choice is Mean Squared Error:

$$
J(w,b)=\frac{1}{n}\sum_{i=1}^{n}(\hat y_i-y_i)^2
$$

Substituting the prediction:

$$
J(w,b)=
\frac{1}{n}
\sum_{i=1}^{n}(wx_i+b-y_i)^2
$$

The goal is:

$$
\boxed{\min_{w,b}J(w,b)}
$$

Gradient descent is the algorithm that helps us find the values of $w$ and $b$ that minimize this cost.

---

# 2. Why is it called "gradient descent"?

There are two important words:

### Gradient

The **gradient** tells us the direction in which the cost function increases most rapidly.

For a function:

$$
J(w,b)
$$

the gradient is:

$$
\nabla J=
\begin{bmatrix}
\frac{\partial J}{\partial w}\\
\frac{\partial J}{\partial b}
\end{bmatrix}
$$

### Descent

We want to go in the **opposite direction** of the gradient because the gradient points toward increasing cost.

Therefore:

$$
\boxed{
\theta_{new}=
\theta_{old}-\eta\nabla J
}
$$

where:

- $\theta$ = model parameters
- $\eta$ = learning rate
- $\nabla J$ = gradient of cost

The `-` sign is what makes it **descent**.

---

# 3. What makes it "Batch"?

This is the most important part.

Suppose our dataset contains:

$$
1000
$$

training examples.

In **Batch Gradient Descent**, we use **all 1000 examples** to calculate the gradient before making **one parameter update**.

The process is:

```text
1000 training examples
        ↓
Make 1000 predictions
        ↓
Calculate errors for all 1000
        ↓
Calculate total gradient
        ↓
Update parameters ONCE
```

So:

> **One gradient update uses the entire training dataset.**

That's why it is called **Batch** Gradient Descent.

---

# 4. Batch vs Stochastic vs Mini-Batch

This distinction is extremely important.

| Algorithm | Data used for one update |
|---|---:|
| Batch Gradient Descent | Entire dataset |
| Stochastic Gradient Descent | 1 sample |
| Mini-Batch Gradient Descent | Small batch |

Suppose:

$$
n=1000
$$

training examples.

### Batch GD

```text
1000 samples → update
1000 samples → update
1000 samples → update
```

### SGD

```text
1 sample → update
1 sample → update
1 sample → update
...
```

### Mini-Batch GD

For batch size 32:

```text
32 samples → update
32 samples → update
32 samples → update
...
```

---

# 5. Mathematical derivation for Linear Regression

Let's derive Batch Gradient Descent properly.

Consider:

$$
\hat y_i=wx_i+b
$$

and the cost function:

$$
J(w,b)=
\frac{1}{2n}
\sum_{i=1}^{n}
(\hat y_i-y_i)^2
$$

We use $\frac{1}{2n}$ instead of $\frac{1}{n}$ because the $2$ makes the derivative cleaner.

Substitute $\hat y_i$:

$$
J(w,b)=
\frac{1}{2n}
\sum_{i=1}^{n}
(wx_i+b-y_i)^2
$$

---

# 6. Derivative with respect to $w$

We want:

$$
\frac{\partial J}{\partial w}
$$

Starting with:

$$
J=
\frac{1}{2n}
\sum_{i=1}^{n}
(wx_i+b-y_i)^2
$$

Using the chain rule:

$$
\frac{\partial J}{\partial w}
=
\frac{1}{2n}
\sum_{i=1}^{n}
2(wx_i+b-y_i)x_i
$$

The $2$'s cancel:

$$
\boxed{
\frac{\partial J}{\partial w}
=
\frac{1}{n}
\sum_{i=1}^{n}
(wx_i+b-y_i)x_i
}
$$

---

# 7. Derivative with respect to $b$

Similarly:

$$
\frac{\partial J}{\partial b}
=
\frac{1}{2n}
\sum_{i=1}^{n}
2(wx_i+b-y_i)
$$

Therefore:

$$
\boxed{
\frac{\partial J}{\partial b}
=
\frac{1}{n}
\sum_{i=1}^{n}
(wx_i+b-y_i)
}
$$

These two derivatives are our gradients.

---

# 8. Parameter update

Now we apply gradient descent.

For $w$:

$$
\boxed{
w=w-\eta
\frac{1}{n}
\sum_{i=1}^{n}
(wx_i+b-y_i)x_i
}
$$

For $b$:

$$
\boxed{
b=b-\eta
\frac{1}{n}
\sum_{i=1}^{n}
(wx_i+b-y_i)
}
$$

This is **Batch Gradient Descent for linear regression**.

Notice something important:

The summation goes from:

$$
i=1
$$

to:

$$
i=n
$$

That means **every training example contributes to the update**.

---

# 9. Complete Batch Gradient Descent algorithm

Suppose:

$$
X=\{x_1,x_2,\ldots,x_n\}
$$

and

$$
Y=\{y_1,y_2,\ldots,y_n\}
$$

The algorithm is:

### Step 1 — Initialize parameters

For example:

$$
w=0
$$

$$
b=0
$$

### Step 2 — Make predictions

$$
\hat y_i=wx_i+b
$$

for every training example.

### Step 3 — Calculate errors

$$
e_i=\hat y_i-y_i
$$

### Step 4 — Calculate gradients

$$
\frac{\partial J}{\partial w}
=
\frac{1}{n}
\sum e_ix_i
$$

$$
\frac{\partial J}{\partial b}
=
\frac{1}{n}
\sum e_i
$$

### Step 5 — Update parameters

$$
w\leftarrow
w-\eta\frac{\partial J}{\partial w}
$$

$$
b\leftarrow
b-\eta\frac{\partial J}{\partial b}
$$

### Step 6 — Repeat

Repeat until the cost stops decreasing significantly or a maximum number of iterations is reached.

---

# 10. Small numerical example

Consider:

| $x$ | $y$ |
|---:|---:|
| 1 | 3 |
| 2 | 5 |
| 3 | 7 |

Clearly, the underlying relationship is:

$$
y=2x+1
$$

Let's see how gradient descent can discover this.

Initially:

$$
w=0,\qquad b=0
$$

Therefore predictions are:

$$
\hat y=[0,0,0]
$$

Actual values:

$$
y=[3,5,7]
$$

Errors:

$$
e=[-3,-5,-7]
$$

---

## Calculate gradient for $w$

$$
\frac{\partial J}{\partial w}
=
\frac{1}{3}
[(-3)(1)+(-5)(2)+(-7)(3)]
$$

$$
=
\frac{-3-10-21}{3}
$$

$$
=-\frac{34}{3}
$$

$$
\boxed{
\frac{\partial J}{\partial w}\approx-11.33
}
$$

---

## Calculate gradient for $b$

$$
\frac{\partial J}{\partial b}
=
\frac{-3-5-7}{3}
$$

$$
\boxed{
\frac{\partial J}{\partial b}=-5
}
$$

---

## Update parameters

Suppose:

$$
\eta=0.01
$$

Then:

$$
w_{new}
=
0-(0.01)(-11.33)
$$

$$
w_{new}\approx0.1133
$$

And:

$$
b_{new}
=
0-(0.01)(-5)
$$

$$
b_{new}=0.05
$$

So after **one iteration**:

$$
\boxed{w\approx0.1133}
$$

$$
\boxed{b=0.05}
$$

The model has moved from:

$$
y=0x+0
$$

toward:

$$
y=0.1133x+0.05
$$

Further iterations continue moving the parameters toward:

$$
y=2x+1
$$

---

# 11. Visual intuition

Imagine the cost function as a valley.

```text
Cost
 ^
 |\
 | \
 |  \
 |   \       ● Starting point
 |    \     /
 |     \   /
 |      \ /
 |       ●  Minimum
 +----------------------> parameter
```

The goal is to reach the bottom of the valley.

If the current parameter is on the left side:

```text
       ●
      /
     /
    /
   /
  ●
```

The gradient tells us which direction cost increases.

Gradient descent moves in the opposite direction.

---

# 12. Cost function and gradient relationship

For a simple one-parameter model, imagine:

$$
J(w)
$$

as a bowl-shaped function.



The derivative:

$$
\frac{dJ}{dw}
$$

tells us the slope at the current position.

If:

$$
\frac{dJ}{dw}>0
$$

then the cost increases as $w$ increases, so gradient descent decreases $w$.

If:

$$
\frac{dJ}{dw}<0
$$

then gradient descent increases $w$.

If:

$$
\frac{dJ}{dw}=0
$$

we have reached a stationary point.

For convex linear regression, this stationary point corresponds to the global minimum.

---

# 13. Learning rate

The learning rate is one of the most important hyperparameters.

We denote it by:

$$
\eta
$$

It controls **how large each parameter update is**.

---

## Small learning rate

Example:

$$
\eta=0.0001
$$

Updates are tiny.

```text
Start
 ↓
 ↓
 ↓
 ↓
 ↓
Minimum
```

Advantages:

- stable
- less likely to overshoot

Disadvantages:

- training can be extremely slow

---

## Large learning rate

Example:

$$
\eta=1
$$

Updates can become very large.

```text
       ●
      ↙ ↘
     ↙   ↘
    ●     ●
     ↘   ↙
      ↘ ↙
```

The algorithm may overshoot the minimum.

---

## Extremely large learning rate

The cost may actually increase:

```text
● →       → ●
   →     →
      → ←
        ●
```

It may diverge instead of converging.

---

# 14. What does one epoch mean in Batch GD?

This is another important concept.

Suppose you have:

$$
1000
$$

training examples.

In Batch Gradient Descent:

```text
Epoch 1:
    Process all 1000 examples
    Calculate gradient
    Update parameters once

Epoch 2:
    Process all 1000 examples
    Calculate gradient
    Update parameters once

Epoch 3:
    Process all 1000 examples
    Calculate gradient
    Update parameters once
```

Therefore:

$$
\boxed{\text{1 epoch = 1 parameter update}}
$$

for pure Batch Gradient Descent.

This differs from mini-batch training.

For example, with:

$$
1000\text{ samples}
$$

and:

$$
\text{batch size}=100
$$

there are:

$$
1000/100=10
$$

updates per epoch.

---

# 15. Batch Gradient Descent using vectors

The scalar equations become much cleaner using linear algebra.

Suppose:

$$
X=
\begin{bmatrix}
x_1 & 1\\
x_2 & 1\\
\vdots & \vdots\\
x_n & 1
\end{bmatrix}
$$

and:

$$
\theta=
\begin{bmatrix}
w\\
b
\end{bmatrix}
$$

Then:

$$
\hat y=X\theta
$$

The error vector is:

$$
e=X\theta-y
$$

The cost is:

$$
J(\theta)
=
\frac{1}{2n}
(X\theta-y)^T(X\theta-y)
$$

The gradient becomes:

$$
\boxed{
\nabla J(\theta)
=
\frac{1}{n}X^T(X\theta-y)
}
$$

And the update is:

$$
\boxed{
\theta
\leftarrow
\theta-\eta
\frac{1}{n}X^T(X\theta-y)
}
$$

This is the form you will commonly see in ML implementations.

---

# 16. Python implementation from scratch

Let's implement Batch Gradient Descent for linear regression without using scikit-learn.

```python
import numpy as np

# Training data
X = np.array([1, 2, 3, 4, 5], dtype=float)
y = np.array([3, 5, 7, 9, 11], dtype=float)

# Parameters
w = 0.0
b = 0.0

# Hyperparameters
learning_rate = 0.01
epochs = 1000

n = len(X)

for epoch in range(epochs):

    # Predictions
    y_pred = w * X + b

    # Errors
    errors = y_pred - y

    # Gradients
    dw = (1 / n) * np.sum(errors * X)
    db = (1 / n) * np.sum(errors)

    # Parameter update
    w = w - learning_rate * dw
    b = b - learning_rate * db

    # Cost
    cost = (1 / (2 * n)) * np.sum(errors ** 2)

    if epoch % 100 == 0:
        print(
            f"Epoch {epoch}, "
            f"Cost: {cost:.4f}, "
            f"w: {w:.4f}, "
            f"b: {b:.4f}"
        )

print("Final weight:", w)
print("Final bias:", b)
```

The result should approach:

```text
w ≈ 2
b ≈ 1
```

So the learned model becomes approximately:

$$
\boxed{\hat y=2x+1}
$$

---

# 17. Vectorized implementation

For larger datasets, vectorization is much better than explicitly looping through samples.

```python
import numpy as np

X = np.array([
    [1],
    [2],
    [3],
    [4],
    [5]
], dtype=float)

y = np.array([
    [3],
    [5],
    [7],
    [9],
    [11]
], dtype=float)

# Add bias column
X_b = np.c_[np.ones((len(X), 1)), X]

# Parameters
theta = np.zeros((2, 1))

learning_rate = 0.01
epochs = 1000

n = len(X)

for epoch in range(epochs):

    # Predictions
    predictions = X_b @ theta

    # Error
    errors = predictions - y

    # Gradient
    gradients = (1 / n) * X_b.T @ errors

    # Update
    theta -= learning_rate * gradients

print(theta)
```

You should get something close to:

```text
[[1.]
 [2.]]
```

Remember that we put the bias first:

$$
\theta=
\begin{bmatrix}
b\\
w
\end{bmatrix}
$$

Therefore:

$$
\hat y=b+wx
$$

---

# 18. Why vectorization is important

Consider a dataset with:

$$
1,000,000
$$

samples.

Doing Python-level loops can be relatively slow.

NumPy can perform operations such as:

```python
predictions = X @ theta
```

using optimized numerical operations underneath.

So instead of:

```python
for sample in dataset:
    ...
```

we often prefer:

```python
predictions = X @ theta
```

This becomes particularly important when implementing gradient descent yourself.

---

# 19. Batch Gradient Descent with multiple features

Suppose we have:

$$
x_1=\text{area}
$$

$$
x_2=\text{bedrooms}
$$

$$
x_3=\text{age}
$$

Our model becomes:

$$
\hat y=
w_1x_1+w_2x_2+w_3x_3+b
$$

For $m$ features:

$$
\hat y=
w_1x_1+w_2x_2+\cdots+w_mx_m+b
$$

The parameter vector is:

$$
\theta=
\begin{bmatrix}
b\\
w_1\\
w_2\\
\vdots\\
w_m
\end{bmatrix}
$$

The update remains:

$$
\boxed{
\theta
\leftarrow
\theta-
\eta
\frac{1}{n}
X^T(X\theta-y)
}
$$

The basic algorithm doesn't change.

Only the number of parameters increases.

---

# 20. Why feature scaling matters

Suppose we have:

```text
Age:       20 - 60
Salary:    20,000 - 200,000
Experience: 0 - 30
```

The features have very different scales.

This can make gradient descent take an inefficient path.

Without scaling, the cost surface can look elongated:

```text
          ______
       __/
    __/
 __/
```

Gradient descent may bounce around instead of heading directly toward the minimum.

After scaling:

```text
       /\
      /  \
     /    \
    /  ●   \
   /        \
```

The optimization can become much faster.

Common techniques include:

### Standardization

$$
x'=\frac{x-\mu}{\sigma}
$$

### Min-Max scaling

$$
x'=
\frac{x-x_{min}}
{x_{max}-x_{min}}
$$

For gradient-based algorithms, feature scaling is often extremely useful.

---

# 21. Batch Gradient Descent and convex functions

For ordinary linear regression with MSE:

$$
J(\theta)
=
\frac{1}{2n}
\|X\theta-y\|^2
$$

the cost function is **convex**.

Conceptually:

```text
       Cost
        ^
        |
       / \
      /   \
     /     \
    /   ●   \
   /_________\____> θ
        minimum
```

There is one global minimum.

Therefore, with an appropriate learning rate, Batch Gradient Descent can converge to the global optimum.

This is different from neural networks, where the optimization landscape can be much more complicated.

---

# 22. Batch Gradient Descent vs Normal Equation

For linear regression, there are two different ways to find parameters.

### Normal Equation

The analytical solution is:

$$
\boxed{
\theta=(X^TX)^{-1}X^Ty
}
$$

### Gradient Descent

Iteratively:

$$
\boxed{
\theta\leftarrow
\theta-\eta
\frac{1}{n}X^T(X\theta-y)
}
$$

Comparison:

| Feature | Normal Equation | Batch GD |
|---|---|---|
| Iterative | No | Yes |
| Learning rate | No | Yes |
| Feature scaling | Usually not necessary | Often useful |
| Very large features | Can become expensive | More practical |
| Sparse data | Less attractive | Can be useful |
| Implementation | Direct formula | Iterative |
| Convex linear regression | Exact solution | Approximates/converges to solution |

---

# 23. Advantages of Batch Gradient Descent

### 1. Stable gradient

Because every training sample contributes to the gradient, the update is relatively smooth.

### 2. Deterministic

For fixed data, initialization, and learning rate, each update is predictable.

### 3. Good for convex optimization

For linear regression with MSE, it can converge toward the global minimum.

### 4. Efficient vectorization

The entire dataset can be processed with matrix operations.

---

# 24. Disadvantages

### 1. Expensive for huge datasets

If you have:

$$
10,000,000
$$

examples, every update requires processing all of them.

### 2. Requires memory

Depending on implementation, the complete dataset needs to be available for each update.

### 3. One update can take a long time

With enormous datasets, waiting for the complete batch before updating can be inefficient.

### 4. Less suitable for online learning

If new data arrives continuously, pure Batch GD isn't ideal because it normally trains using the whole dataset repeatedly.

---

# 25. Batch vs SGD vs Mini-Batch in detail

Suppose:

$$
n=10,000
$$

samples.

### Batch GD

Batch size:

$$
10,000
$$

Updates per epoch:

$$
1
$$

```text
████████████████████ → update
```

### SGD

Batch size:

$$
1
$$

Updates per epoch:

$$
10,000
$$

```text
█ → update
█ → update
█ → update
...
```

### Mini-Batch

Suppose:

$$
batch\ size=32
$$

Approximately:

$$
\frac{10,000}{32}\approx313
$$

updates per epoch.

```text
████ → update
████ → update
████ → update
...
```

---

# 26. The key trade-off

Think of it this way:

```text
Batch GD
    ↓
More accurate gradient
    ↓
Stable updates
    ↓
But expensive updates
```

versus:

```text
SGD
    ↓
Very cheap updates
    ↓
Very noisy gradient
    ↓
Can move toward minimum quickly
```

and:

```text
Mini-Batch GD
    ↓
Moderate-sized batches
    ↓
Good computational efficiency
    ↓
Less noisy than SGD
```

This is why **mini-batch gradient descent is extremely common in modern deep learning**.

---

# 27. Common problems in Batch Gradient Descent

## Problem 1: Learning rate too small

```text
Cost
 |
 |\
 | \
 |  \
 |   \
 |    \
 |     \
 +----------> iterations
```

Cost decreases very slowly.

Solution:

Increase learning rate carefully.

---

## Problem 2: Learning rate too large

Cost can oscillate:

```text
Cost
 |
 | ●   ●
 |   ●   ●
 | ●       ●
 |    ●
 +--------------> iterations
```

Or diverge:

```text
Cost
 |
 |          ●
 |      ●
 |   ●
 | ●
 |________________> iterations
```

Solution:

Reduce learning rate.

---

# 28. How do we know gradient descent has converged?

Several approaches are possible.

### Method 1: Fixed number of epochs

```python
epochs = 1000
```

Simply train for a predefined number of iterations.

### Method 2: Cost improvement

Stop when:

$$
|J_t-J_{t-1}|<\epsilon
$$

For example:

```python
if abs(previous_cost - cost) < tolerance:
    break
```

### Method 3: Gradient magnitude

Stop when:

$$
\|\nabla J\|<\epsilon
$$

This means the gradient is very close to zero.

---

# 29. Important distinction: epoch, iteration, batch

These terms are often confused.

### Batch

The number of samples used to calculate **one gradient update**.

### Iteration

One parameter update.

### Epoch

One complete pass through the training dataset.

For pure Batch GD:

```text
1 epoch
   ↓
Entire dataset
   ↓
1 iteration/update
```

For mini-batch GD:

```text
1 epoch
   ↓
Multiple batches
   ↓
Multiple iterations
```

---

# 30. Batch Gradient Descent in neural networks

The same basic idea applies to neural networks.

Suppose:

```text
Input
  ↓
Neural Network
  ↓
Prediction
  ↓
Loss
```

We calculate:

$$
L(\theta)
$$

Then backpropagation calculates:

$$
\nabla_\theta L
$$

and gradient descent updates:

$$
\boxed{
\theta\leftarrow\theta-\eta\nabla_\theta L
}
$$

With Batch GD, the gradient is calculated using the **entire training dataset**.

In modern neural-network training, however, **mini-batch gradient descent** is generally much more practical.

---

# 31. A complete mental model

You can remember Batch Gradient Descent as:

```text
                DATASET
                   │
                   ▼
             Make predictions
                   │
                   ▼
              Calculate loss
                   │
                   ▼
             Calculate gradient
             using ALL samples
                   │
                   ▼
             Update parameters
                   │
                   ▼
          Repeat for next epoch
                   │
                   ▼
             Minimum reached
```

Mathematically:

$$
\boxed{
\theta_{new}
=
\theta_{old}
-
\eta
\frac{1}{n}
X^T(X\theta-y)
}
$$

For linear regression, this single equation captures the essence of Batch Gradient Descent.

---

# 32. What you should remember for ML

The most important concepts are:

1. **Gradient Descent is an optimization algorithm.**
2. Its purpose is to minimize a **loss/cost function**.
3. The gradient tells us the direction of **steepest increase**.
4. We move in the **opposite direction**.
5. The learning rate controls the **step size**.
6. **Batch GD uses the entire training dataset for one update.**
7. One epoch in pure Batch GD corresponds to one update.
8. Feature scaling often improves convergence.
9. Too-small learning rate → slow training.
10. Too-large learning rate → oscillation/divergence.
11. Linear regression with MSE has a convex cost function.
12. Batch GD differs from SGD and Mini-Batch GD mainly by **how many samples are used for each update**.
13. Modern deep learning generally favors **mini-batches** rather than pure Batch GD.

### The core formula

For any differentiable model:

$$
\boxed{
\theta\leftarrow\theta-\eta\nabla J(\theta)
}
$$

For linear regression:

$$
\boxed{
\nabla J(\theta)
=
\frac{1}{n}X^T(X\theta-y)
}
$$

Therefore:

$$
\boxed{
\theta
\leftarrow
\theta-
\frac{\eta}{n}
X^T(X\theta-y)
}
$$

That is the fundamental mathematical mechanism behind **Batch Gradient Descent**.

---
