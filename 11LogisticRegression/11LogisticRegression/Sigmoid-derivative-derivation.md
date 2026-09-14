
The **derivative of the sigmoid function** is one of the most important results in logistic regression and neural networks.

### 1. Sigmoid function

The sigmoid function is:

$$
\sigma(z)=\frac{1}{1+e^{-z}}
$$

We want:

$$
\frac{d\sigma(z)}{dz}
$$

---

### 2. Differentiate step-by-step

Rewrite:

$$
\sigma(z)=(1+e^{-z})^{-1}
$$

Using the chain rule:

$$
\frac{d\sigma}{dz}
=-1(1+e^{-z})^{-2}\frac{d}{dz}(1+e^{-z})
$$

Since:

$$
\frac{d}{dz}e^{-z}=-e^{-z}
$$

we get:

$$
\frac{d\sigma}{dz}
=\frac{e^{-z}}{(1+e^{-z})^2}
$$

So:

$$
\boxed{\sigma'(z)=\frac{e^{-z}}{(1+e^{-z})^2}}
$$

---

### 3. The important simplification

We can express the derivative **in terms of the sigmoid itself**.

Recall:

$$
\sigma(z)=\frac{1}{1+e^{-z}}
$$

Also:

$$
1-\sigma(z)
=1-\frac{1}{1+e^{-z}}
$$

$$
=\frac{e^{-z}}{1+e^{-z}}
$$

Therefore:

$$
\sigma(z)(1-\sigma(z))
=
\frac{1}{1+e^{-z}}
\frac{e^{-z}}{1+e^{-z}}
$$

$$
=\frac{e^{-z}}{(1+e^{-z})^2}
$$

which is exactly our derivative.

Hence:

$$
\boxed{\sigma'(z)=\sigma(z)(1-\sigma(z))}
$$

This is the **key sigmoid derivative formula**.

---

### 4. Example

Suppose:

$$
z=0
$$

Then:

$$
\sigma(0)=\frac{1}{1+e^0}=\frac12
$$

Therefore:

$$
\sigma'(0)
=
\sigma(0)(1-\sigma(0))
$$

$$
=\frac12\left(1-\frac12\right)
$$

$$
=\frac14
$$

So:

$$
\boxed{\sigma'(0)=0.25}
$$

The sigmoid has its **maximum slope at $z=0$**.

---

### 5. Why this matters in Logistic Regression

In logistic regression:

$$
z=w^Tx+b
$$

and:

$$
\hat y=\sigma(z)
$$

Therefore, using the chain rule:

$$
\frac{\partial \hat y}{\partial w}
=
\frac{\partial \hat y}{\partial z}
\frac{\partial z}{\partial w}
$$

Since:

$$
\frac{\partial \hat y}{\partial z}
=
\hat y(1-\hat y)
$$

and:

$$
\frac{\partial z}{\partial w}=x
$$

we get:

$$
\boxed{
\frac{\partial \hat y}{\partial w}
=
\hat y(1-\hat y)x
}
$$

This is one of the steps that eventually leads to the **gradient of Binary Cross-Entropy loss**:

$$
\boxed{
\frac{\partial L}{\partial w}
=
(\hat y-y)x
}
$$

The beautiful part is that the sigmoid derivative and the BCE derivative **simplify/cancel in a very convenient way**.