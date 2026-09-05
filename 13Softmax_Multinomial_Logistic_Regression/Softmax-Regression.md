
# Softmax Regression / Multinomial Logistic Regression — In Detail

**Softmax Regression**, also called **Multinomial Logistic Regression**, is the multiclass extension of logistic regression.

If ordinary logistic regression answers:

> **“Is this either class 0 or class 1?”**

Softmax regression answers:

> **“Which one of K classes does this example belong to?”**

For example:

- Class 0 → Cat
- Class 1 → Dog
- Class 2 → Horse

The model produces a **probability for every class**, and the class with the highest probability becomes the prediction.

---

# 1. Why do we need Softmax Regression?

Recall binary logistic regression.

For binary classification:

$$
z = w^Tx+b
$$

Then we use the sigmoid function:

$$
\sigma(z)=\frac{1}{1+e^{-z}}
$$

which gives:

$$
P(y=1|x)=\sigma(z)
$$

and

$$
P(y=0|x)=1-\sigma(z)
$$

So we get two probabilities:

$$
P(y=0|x)+P(y=1|x)=1
$$

But suppose we have **3 classes**:

$$
y\in\{0,1,2\}
$$

A sigmoid isn't naturally designed to produce three mutually exclusive probabilities.

We need something like:

$$
P(y=0|x)=0.10
$$

$$
P(y=1|x)=0.70
$$

$$
P(y=2|x)=0.20
$$

Notice:

$$
0.10+0.70+0.20=1
$$

That's exactly what the **softmax function** provides.

---

# 2. Basic idea

Suppose we have $K$ classes.

For every class, the model calculates a score:

$$
z_1,z_2,\ldots,z_K
$$

These scores are called **logits**.

Softmax converts these logits into probabilities.

$$
\boxed{
P(y=k|x)=
\frac{e^{z_k}}
{\sum_{j=1}^{K}e^{z_j}}
}
$$

where:

- $K$ = number of classes
- $z_k$ = score/logit for class $k$
- $e^{z_k}$ = exponential of that score
- denominator = sum of exponentials of all class scores

---

# 3. Architecture

Suppose our input contains $n$ features:

$$
x=
\begin{bmatrix}
x_1\\
x_2\\
\vdots\\
x_n
\end{bmatrix}
$$

and we have $K$ classes.

Instead of one weight vector, we have **one weight vector per class**.

For example, with 3 classes:

$$
w_0,w_1,w_2
$$

Each class gets its own linear equation:

$$
z_0=w_0^Tx+b_0
$$

$$
z_1=w_1^Tx+b_1
$$

$$
z_2=w_2^Tx+b_2
$$

Then:

$$
[z_0,z_1,z_2]
$$

goes through softmax.

---

# 4. Matrix representation

This is the more useful representation for implementation.

Let:

$$
X\in\mathbb{R}^{m\times n}
$$

where:

- $m$ = number of training examples
- $n$ = number of features

Weights:

$$
W\in\mathbb{R}^{n\times K}
$$

Bias:

$$
b\in\mathbb{R}^{K}
$$

Then:

$$
\boxed{Z=XW+b}
$$

where:

$$
Z\in\mathbb{R}^{m\times K}
$$

Each row of $Z$ contains the logits for one training example.

---

# 5. Example

Suppose:

$$
z=[2,1,0]
$$

We calculate:

$$
e^2=7.389
$$

$$
e^1=2.718
$$

$$
e^0=1
$$

Sum:

$$
7.389+2.718+1=11.107
$$

Therefore:

$$
P(y=0)=\frac{7.389}{11.107}\approx0.665
$$

$$
P(y=1)=\frac{2.718}{11.107}\approx0.245
$$

$$
P(y=2)=\frac{1}{11.107}\approx0.090
$$

So:

$$
P(y|x)=[0.665,0.245,0.090]
$$

The prediction is:

$$
\boxed{\hat y=0}
$$

because class 0 has the highest probability.

---

# 6. Why exponential?

The exponential function has useful properties.

If:

$$
z_1>z_2
$$

then:

$$
e^{z_1}>e^{z_2}
$$

So larger logits produce larger probabilities.

Also:

$$
e^z>0
$$

Therefore every softmax probability is positive.

Finally, because we divide by the sum:

$$
\sum_{k=1}^{K}P(y=k|x)=1
$$

So the outputs form a valid probability distribution.

---

# 7. The complete Softmax Regression pipeline

The entire model can be thought of as:

$$
X
$$

↓

### Linear transformation

$$
Z=XW+b
$$

↓

### Softmax

$$
P_k=
\frac{e^{z_k}}
{\sum_j e^{z_j}}
$$

↓

### Probability distribution

$$
P=[P_1,P_2,\ldots,P_K]
$$

↓

### Argmax

$$
\boxed{\hat y=\arg\max_k P_k}
$$

---

# 8. One-hot encoding

For multiclass classification, labels are often represented using **one-hot encoding**.

Suppose:

$$
y=2
$$

and there are three classes.

Then:

$$
y=
[0,0,1]
$$

For class 0:

$$
[1,0,0]
$$

For class 1:

$$
[0,1,0]
$$

For class 2:

$$
[0,0,1]
$$

This becomes important when deriving the loss function.

---

# 9. Cross-Entropy Loss

Softmax regression typically uses **categorical cross-entropy**.

For one training example:

$$
\boxed{
L=-\sum_{k=1}^{K}y_k\log(\hat y_k)
}
$$

where:

- $y_k$ = true one-hot label
- $\hat y_k$ = predicted probability

Because only one $y_k$ is equal to 1, this simplifies to:

$$
\boxed{
L=-\log(\hat y_{\text{true}})
}
$$

---

# 10. Example of cross-entropy

Suppose the actual class is:

$$
y=[0,1,0]
$$

and model predicts:

$$
\hat y=[0.2,0.7,0.1]
$$

The correct class is class 1.

Therefore:

$$
L=-\log(0.7)
$$

$$
L\approx0.357
$$

Good prediction → low loss.

---

Now suppose:

$$
\hat y=[0.8,0.1,0.1]
$$

Then:

$$
L=-\log(0.1)
$$

$$
L\approx2.303
$$

Bad prediction → high loss.

---

# 11. Why cross-entropy works so well

Consider:

$$
L=-\log(p)
$$

where $p$ is the probability assigned to the correct class.

| Correct-class probability | Loss |
|---:|---:|
| 0.99 | 0.010 |
| 0.90 | 0.105 |
| 0.70 | 0.357 |
| 0.50 | 0.693 |
| 0.20 | 1.609 |
| 0.01 | 4.605 |

The model is heavily penalized when it is **confidently wrong**.

---

# 12. Derivation of Softmax + Cross-Entropy Gradient

This is one of the most important derivations.

We have:

$$
z_k=w_k^Tx+b_k
$$

Softmax:

$$
p_k=\frac{e^{z_k}}{\sum_j e^{z_j}}
$$

Loss:

$$
L=-\sum_k y_k\log p_k
$$

We want:

$$
\frac{\partial L}{\partial z_k}
$$

The beautiful result is:

$$
\boxed{
\frac{\partial L}{\partial z_k}
=
p_k-y_k
}
$$

This is extremely important in neural networks and machine learning.

---
