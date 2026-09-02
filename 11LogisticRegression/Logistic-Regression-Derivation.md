

# Logistic Regression with Gradient Descent — Mathematical Derivation

## 1. What problem does Logistic Regression solve?

Logistic Regression is primarily used for **binary classification**.

Suppose:

$$
y \in \{0,1\}
$$

For example:

| Hours studied | Passed |
|---:|---:|
| 1 | 0 |
| 2 | 0 |
| 3 | 0 |
| 5 | 1 |
| 6 | 1 |
| 8 | 1 |

We want the model to estimate:

$$
P(y=1\mid x)
$$

That means:

> Given input $x$, what is the probability that the output belongs to class 1?

---

# 2. Start with Linear Regression

In linear regression, we have:

$$
z = w_0+w_1x
$$

or more generally:

$$
z = \mathbf{w}^T\mathbf{x}+b
$$

The problem is that $z$ can be any number:

$$
-\infty < z < +\infty
$$

But probability must satisfy:

$$
0\leq P\leq1
$$

So we cannot directly use the linear output as a probability.

We need a function that converts any real number into the range $(0,1)$.

That function is the **sigmoid function**.

---

# 3. Sigmoid Function

The sigmoid function is:

$$
\sigma(z)=\frac{1}{1+e^{-z}}
$$

Therefore logistic regression first calculates:

$$
z=\mathbf{w}^T\mathbf{x}+b
$$

and then:

$$
\hat y=\sigma(z)
$$

So:

$$
\boxed{
\hat y=\frac{1}{1+e^{-(\mathbf{w}^T\mathbf{x}+b)}}
}
$$

Here:

- $x$ = input features
- $w$ = weights
- $b$ = bias
- $z$ = linear combination
- $\hat y$ = predicted probability

For example:

$$
z=2
$$

Then:

$$
\hat y=\frac{1}{1+e^{-2}}
$$

$$
\hat y\approx0.881
$$

So the model predicts:

$$
P(y=1|x)\approx88.1\%
$$

---

# 4. Why sigmoid?

The sigmoid has useful properties:

$$
\sigma(z)\rightarrow1 \quad\text{as }z\rightarrow+\infty
$$

and

$$
\sigma(z)\rightarrow0 \quad\text{as }z\rightarrow-\infty
$$

and:

$$
\sigma(0)=0.5
$$

Therefore:

```text
       probability
1.0 |                    ______
    |                 __/
0.5 |-------------___/
    |          __/
0.0 |_________/
    +------------------------> z
             0
```

The usual classification rule is:

$$
\hat y=
\begin{cases}
1 & \text{if }P(y=1|x)\geq0.5\\
0 & \text{if }P(y=1|x)<0.5
\end{cases}
$$



---

# 5. The important question: How do we find $w$ and $b$?

Initially, weights are usually initialized randomly or to zero.

For example:

$$
w=0.2,\qquad b=-0.5
$$

These values won't necessarily produce good predictions.

We therefore need to **optimize** $w$ and $b$.

This requires three things:

1. A prediction function
2. A loss function
3. An optimization algorithm

For logistic regression:

$$
\boxed{\text{Prediction}+\text{Binary Cross Entropy}+\text{Gradient Descent}}
$$

---

# 6. Why not Mean Squared Error?

You might think we can use:

$$
MSE=\frac1n\sum_{i=1}^{n}(y_i-\hat y_i)^2
$$

Technically, it can be used in some settings, but it gives logistic regression an inconvenient optimization problem.

Instead, logistic regression uses **Log Loss / Binary Cross Entropy**:

$$
\boxed{
L(y,\hat y)
=
-\left[y\log(\hat y)+(1-y)\log(1-\hat y)\right]
}
$$

Let's understand why.

---

# 7. Binary Cross Entropy

For a single training example:

$$
L=-[y\log(\hat y)+(1-y)\log(1-\hat y)]
$$

There are two possibilities.

## Case 1: $y=1$

Then:

$$
L=-[1\log(\hat y)+0\log(1-\hat y)]
$$

Therefore:

$$
\boxed{L=-\log(\hat y)}
$$

If:

$$
\hat y=0.9
$$

then:

$$
L=-\log(0.9)\approx0.105
$$

Small loss.

But if:

$$
\hat y=0.1
$$

then:

$$
L=-\log(0.1)\approx2.303
$$

Large loss.

So the loss strongly punishes confident incorrect predictions.

---

## Case 2: $y=0$

Now:

$$
L=-[0\log(\hat y)+(1)\log(1-\hat y)]
$$

Therefore:

$$
\boxed{L=-\log(1-\hat y)}
$$

If:

$$
\hat y=0.1
$$

then:

$$
L=-\log(0.9)\approx0.105
$$

Good.

But if:

$$
\hat y=0.9
$$

then:

$$
L=-\log(0.1)\approx2.303
$$

Bad.

---

# 8. Cost Function for the Entire Dataset

For $n$ training examples:

$$
\boxed{
J(w,b)
=
-\frac1n
\sum_{i=1}^{n}
\left[
y_i\log(\hat y_i)
+
(1-y_i)\log(1-\hat y_i)
\right]
}
$$

where:

$$
\hat y_i=\sigma(z_i)
$$

and:

$$
z_i=w^Tx_i+b
$$

Our goal is:

$$
\boxed{\min_{w,b}J(w,b)}
$$

Now comes the most important part:

# 9. Deriving the Gradient

We need:

$$
\frac{\partial J}{\partial w}
$$

and:

$$
\frac{\partial J}{\partial b}
$$

This tells us how the cost changes when we change the parameters.

Let's derive it carefully.

---

# 10. Start with one training example

Consider:

$$
L=-[y\log(\hat y)+(1-y)\log(1-\hat y)]
$$

We know:

$$
\hat y=\sigma(z)
$$

and:

$$
z=w^Tx+b
$$

We want:

$$
\frac{\partial L}{\partial w}
$$

Because $L$ depends on $w$ indirectly through $\hat y$ and $z$, we'll use the **chain rule**.

The dependency is:

$$
w
\rightarrow z
\rightarrow \hat y
\rightarrow L
$$

Therefore:

$$
\boxed{
\frac{\partial L}{\partial w}
=
\frac{\partial L}{\partial\hat y}
\frac{\partial\hat y}{\partial z}
\frac{\partial z}{\partial w}
}
$$

Now derive each component.

---

# 11. Derivative of Binary Cross Entropy

We have:

$$
L=-[y\log(\hat y)+(1-y)\log(1-\hat y)]
$$

Differentiate with respect to $\hat y$:

$$
\frac{\partial L}{\partial\hat y}
=
-\left[
\frac{y}{\hat y}
+
(1-y)\frac{-1}{1-\hat y}
\right]
$$

Therefore:

$$
\frac{\partial L}{\partial\hat y}
=
-\frac{y}{\hat y}
+
\frac{1-y}{1-\hat y}
$$

Take common denominator:

$$
=
\frac{-y(1-\hat y)+(1-y)\hat y}
{\hat y(1-\hat y)}
$$

Expand:

$$
=
\frac{-y+y\hat y+\hat y-y\hat y}
{\hat y(1-\hat y)}
$$

The $y\hat y$ terms cancel:

$$
\boxed{
\frac{\partial L}{\partial\hat y}
=
\frac{\hat y-y}{\hat y(1-\hat y)}
}
$$

---

# 12. Derivative of Sigmoid

This is extremely important.

We have:

$$
\hat y=\sigma(z)=\frac1{1+e^{-z}}
$$

Rewrite:

$$
\hat y=(1+e^{-z})^{-1}
$$

Differentiate:

$$
\frac{d\hat y}{dz}
=
-(1+e^{-z})^{-2}(-e^{-z})
$$

Therefore:

$$
\frac{d\hat y}{dz}
=
\frac{e^{-z}}{(1+e^{-z})^2}
$$

This can be rewritten as:

$$
\boxed{
\frac{d\hat y}{dz}
=
\hat y(1-\hat y)
}
$$

This beautiful property of sigmoid is one of the reasons it works nicely with BCE.

---

# 13. Derivative of $z$ with respect to $w$

Recall:

$$
z=w^Tx+b
$$

For one feature:

$$
z=wx+b
$$

Therefore:

$$
\boxed{
\frac{\partial z}{\partial w}=x
}
$$

And:

$$
\boxed{
\frac{\partial z}{\partial b}=1
}
$$

---

# 14. Put everything together

We had:

$$
\frac{\partial L}{\partial w}
=
\frac{\partial L}{\partial\hat y}
\frac{\partial\hat y}{\partial z}
\frac{\partial z}{\partial w}
$$

Substitute:

$$
\frac{\partial L}{\partial\hat y}
=
\frac{\hat y-y}{\hat y(1-\hat y)}
$$

$$
\frac{\partial\hat y}{\partial z}
=
\hat y(1-\hat y)
$$

and:

$$
\frac{\partial z}{\partial w}=x
$$

Therefore:

$$
\frac{\partial L}{\partial w}
=
\frac{\hat y-y}{\hat y(1-\hat y)}
\cdot
\hat y(1-\hat y)
\cdot x
$$

The terms cancel:

$$
\boxed{
\frac{\partial L}{\partial w}
=
(\hat y-y)x
}
$$

This is the key result.

Similarly:

$$
\frac{\partial L}{\partial b}
=
\frac{\partial L}{\partial\hat y}
\frac{\partial\hat y}{\partial z}
\frac{\partial z}{\partial b}
$$

Therefore:

$$
\boxed{
\frac{\partial L}{\partial b}
=
\hat y-y
}
$$

---

# 15. Gradient for the Entire Dataset

For $n$ examples:

$$
J=
-\frac1n
\sum_{i=1}^{n}
[y_i\log(\hat y_i)
+(1-y_i)\log(1-\hat y_i)]
$$

The gradient with respect to $w$ is:

$$
\boxed{
\frac{\partial J}{\partial w}
=
\frac1n
\sum_{i=1}^{n}
(\hat y_i-y_i)x_i
}
$$

And:

$$
\boxed{
\frac{\partial J}{\partial b}
=
\frac1n
\sum_{i=1}^{n}
(\hat y_i-y_i)
}
$$

These are the gradients that Gradient Descent uses.

---

# 16. Gradient Descent

Now we have the gradient.

The basic Gradient Descent rule is:

$$
\boxed{
\theta_{\text{new}}
=
\theta_{\text{old}}
-
\eta
\frac{\partial J}{\partial\theta}
}
$$

where:

- $\theta$ = parameter
- $\eta$ = learning rate
- gradient = direction of increasing loss

Therefore, for $w$:

$$
\boxed{
w_{\text{new}}
=
w_{\text{old}}
-
\eta
\frac{\partial J}{\partial w}
}
$$

Substitute the gradient:

$$
\boxed{
w_{\text{new}}
=
w_{\text{old}}
-
\eta
\frac1n
\sum_{i=1}^{n}
(\hat y_i-y_i)x_i
}
$$

For bias:

$$
\boxed{
b_{\text{new}}
=
b_{\text{old}}
-
\eta
\frac1n
\sum_{i=1}^{n}
(\hat y_i-y_i)
}
$$

---

# 17. Why do we subtract the gradient?

Imagine the loss function looks like:

```text
Loss
 ^
 |\
 | \
 |  \
 |   \        /
 |    \______/ 
 |           \
 +----------------> w
             minimum
```

The gradient tells us the direction in which loss is increasing.

We want to move in the opposite direction.

Therefore:

$$
\boxed{\text{new parameter}=\text{old parameter}-\text{learning rate}\times\text{gradient}}
$$

---

# 18. Complete Logistic Regression Algorithm

The complete algorithm is:

### Step 1 — Initialize

$$
w=0,\qquad b=0
$$

### Step 2 — Calculate linear output

For each sample:

$$
z_i=w^Tx_i+b
$$

### Step 3 — Apply sigmoid

$$
\hat y_i=\frac1{1+e^{-z_i}}
$$

### Step 4 — Calculate loss

$$
J=
-\frac1n
\sum
[y_i\log(\hat y_i)
+
(1-y_i)\log(1-\hat y_i)]
$$

### Step 5 — Calculate gradients

$$
dw=
\frac1nX^T(\hat y-y)
$$

$$
db=
\frac1n\sum(\hat y-y)
$$

### Step 6 — Update parameters

$$
w=w-\eta dw
$$

$$
b=b-\eta db
$$

### Step 7 — Repeat

Repeat for many epochs until the loss converges.

---

# 19. Matrix Form

For practical implementation, the matrix formulation is very useful.

Suppose:

$$
X=
\begin{bmatrix}
x_{11}&x_{12}&\cdots&x_{1m}\\
x_{21}&x_{22}&\cdots&x_{2m}\\
\vdots&\vdots&&\vdots\\
x_{n1}&x_{n2}&\cdots&x_{nm}
\end{bmatrix}
$$

and:

$$
w=
\begin{bmatrix}
w_1\\
w_2\\
\vdots\\
w_m
\end{bmatrix}
$$

Then:

$$
\boxed{
z=Xw+b
}
$$

Apply sigmoid:

$$
\boxed{
\hat y=\sigma(Xw+b)
}
$$

Then:

$$
\boxed{
dw=\frac1nX^T(\hat y-y)
}
$$

and:

$$
\boxed{
db=\frac1n\sum(\hat y-y)
}
$$

Finally:

$$
\boxed{
w:=w-\eta dw
}
$$

$$
\boxed{
b:=b-\eta db
}
$$

This is essentially what your NumPy implementation will do.

---

# 20. A Numerical Example

Let's take a very simple dataset:

$$
x=
\begin{bmatrix}
1\\
2\\
3
\end{bmatrix}
$$

and:

$$
y=
\begin{bmatrix}
0\\
0\\
1
\end{bmatrix}
$$

Initialize:

$$
w=0,\qquad b=0
$$

Learning rate:

$$
\eta=0.1
$$

---

## Step 1: Calculate $z$

$$
z=wx+b
$$

Since:

$$
w=0,\quad b=0
$$

we get:

$$
z=
\begin{bmatrix}
0\\0\\0
\end{bmatrix}
$$

---

## Step 2: Sigmoid

Since:

$$
\sigma(0)=0.5
$$

we get:

$$
\hat y=
\begin{bmatrix}
0.5\\
0.5\\
0.5
\end{bmatrix}
$$

---

## Step 3: Calculate errors

$$
\hat y-y
$$

Therefore:

$$
\begin{bmatrix}
0.5\\
0.5\\
0.5
\end{bmatrix}
-
\begin{bmatrix}
0\\
0\\
1
\end{bmatrix}
=
\begin{bmatrix}
0.5\\
0.5\\
-0.5
\end{bmatrix}
$$

---

## Step 4: Calculate $dw$

$$
dw=
\frac13
X^T(\hat y-y)
$$

Here:

$$
X^T=
\begin{bmatrix}
1&2&3
\end{bmatrix}
$$

So:

$$
dw=
\frac13
[
1(0.5)+2(0.5)+3(-0.5)
]
$$

$$
=
\frac13(0.5+1-1.5)
$$

$$
=\frac13(0)
$$

Therefore:

$$
\boxed{dw=0}
$$

---

## Step 5: Calculate $db$

$$
db=
\frac13
(0.5+0.5-0.5)
$$

$$
db=\frac{0.5}{3}
$$

$$
\boxed{db\approx0.1667}
$$

---

## Step 6: Update

$$
w_{\text{new}}=0-0.1(0)=0
$$

and:

$$
b_{\text{new}}
=
0-0.1(0.1667)
$$

$$
\boxed{b_{\text{new}}\approx-0.01667}
$$

Then the next iteration begins.

Notice something interesting: **the weight didn't change in this particular first iteration, but the bias did.** That's perfectly possible.

---

# 21. Why does the gradient contain $(\hat y-y)$?

This is one of the most important intuitions.

The gradient is:

$$
\boxed{
dw=\frac1n\sum(\hat y-y)x
}
$$

The term:

$$
\hat y-y
$$

is essentially the prediction error.

Suppose:

$$
y=1
$$

but:

$$
\hat y=0.2
$$

Then:

$$
\hat y-y=0.2-1=-0.8
$$

The model predicted too low.

The gradient becomes negative (depending on $x$), and subtracting a negative gradient increases the corresponding weight.

So the model moves toward predicting a larger probability.

---

If:

$$
y=0
$$

but:

$$
\hat y=0.8
$$

then:

$$
\hat y-y=0.8
$$

The model predicted too high.

Gradient descent pushes the weights in the opposite direction.

So:

$$
\boxed{\hat y-y}
$$

provides the fundamental correction signal.

---

# 22. Connection to the Perceptron

Since you've also been studying the **Perceptron**, this connection is important.

### Perceptron

First calculates:

$$
z=w^Tx+b
$$

Then uses a hard threshold:

$$
\hat y=
\begin{cases}
1 & z\geq0\\
0 & z<0
\end{cases}
$$

### Logistic Regression

Uses:

$$
z=w^Tx+b
$$

then:

$$
\hat y=\sigma(z)
$$

So:

```text
                Linear Score
                    z
                    │
           ┌────────┴────────┐
           ↓                 ↓
      Perceptron       Logistic Regression
           │                 │
      Step Function        Sigmoid
           │                 │
        0 or 1          probability
```

The major advantage of logistic regression is that sigmoid is differentiable, which makes gradient-based optimization straightforward.

---

# 23. Decision Boundary

The decision boundary occurs when:

$$
\hat y=0.5
$$

Since:

$$
\hat y=\sigma(z)
$$

and:

$$
\sigma(0)=0.5
$$

we have:

$$
z=0
$$

Therefore:

$$
w^Tx+b=0
$$

For two features:

$$
w_1x_1+w_2x_2+b=0
$$

This is a straight line.

For three features:

$$
w_1x_1+w_2x_2+w_3x_3+b=0
$$

This is a plane.

For higher dimensions, it is called a **hyperplane**.

So logistic regression produces a **linear decision boundary**.

---

# 24. The Entire Mathematical Pipeline

You can remember logistic regression as this chain:

$$
\boxed{
X
\rightarrow
z
\rightarrow
\hat y
\rightarrow
Loss
\rightarrow
Gradient
\rightarrow
Parameter\ Update
}
$$

Mathematically:

### Linear model

$$
z=Xw+b
$$

↓

### Sigmoid

$$
\hat y=\sigma(z)
$$

↓

### Binary Cross Entropy

$$
J=
-\frac1n
\sum
[y\log\hat y+(1-y)\log(1-\hat y)]
$$

↓

### Gradient

$$
\boxed{
dw=\frac1nX^T(\hat y-y)
}
$$

$$
\boxed{
db=\frac1n\sum(\hat y-y)
}
$$

↓

### Gradient descent

$$
\boxed{
w=w-\eta dw
}
$$

$$
\boxed{
b=b-\eta db
}
$$

↓

### Repeat

$$
\boxed{\text{until the loss converges}}
$$

---

# 25. The Most Important Derivation to Remember

If you're studying this for ML fundamentals/interviews, focus heavily on this sequence:

$$
\hat y=\sigma(z)
$$

$$
z=w^Tx+b
$$

$$
L=-[y\log\hat y+(1-y)\log(1-\hat y)]
$$

Then chain rule:

$$
\frac{\partial L}{\partial w}
=
\frac{\partial L}{\partial\hat y}
\frac{\partial\hat y}{\partial z}
\frac{\partial z}{\partial w}
$$

where:

$$
\frac{\partial L}{\partial\hat y}
=
\frac{\hat y-y}{\hat y(1-\hat y)}
$$

$$
\frac{\partial\hat y}{\partial z}
=
\hat y(1-\hat y)
$$

$$
\frac{\partial z}{\partial w}=x
$$

Therefore:

$$
\boxed{
\frac{\partial L}{\partial w}
=(\hat y-y)x
}
$$

and:

$$
\boxed{
\frac{\partial L}{\partial b}
=\hat y-y
}
$$

For the whole dataset:

$$
\boxed{
dw=\frac1nX^T(\hat y-y)
}
$$

$$
\boxed{
db=\frac1n\sum(\hat y-y)
}
$$

and finally:

$$
\boxed{
w:=w-\eta dw
}
$$

$$
\boxed{
b:=b-\eta db
}
$$

That is the mathematical heart of **Logistic Regression + Gradient Descent**.