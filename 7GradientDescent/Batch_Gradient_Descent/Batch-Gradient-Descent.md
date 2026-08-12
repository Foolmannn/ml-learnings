
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
