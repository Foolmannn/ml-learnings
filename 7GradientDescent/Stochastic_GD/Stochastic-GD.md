
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

### Stochastic Gradient Descent

One update requires processing only:

```text
Example 1 → Update
Example 2 → Update
Example 3 → Update
Example 4 → Update
...
```

Therefore, SGD can start updating the model much earlier.

---

# 4. Mathematical Derivation of SGD

Consider a linear regression model:

$$
\hat y_i = wx_i+b
$$

For one training example, define the squared-error loss as:

$$
J_i(w,b)=\frac{1}{2}(\hat y_i-y_i)^2
$$

Substitute the prediction:

$$
J_i(w,b)
=
\frac{1}{2}(wx_i+b-y_i)^2
$$

We need the gradients.

## Gradient with respect to $w$

$$
\frac{\partial J_i}{\partial w}
=
(wx_i+b-y_i)x_i
$$

Since:

$$
\hat y_i=wx_i+b
$$

we get:

$$
\boxed{
\frac{\partial J_i}{\partial w}
=
(\hat y_i-y_i)x_i
}
$$

---

## Gradient with respect to $b$

$$
\frac{\partial J_i}{\partial b}
=
\hat y_i-y_i
$$

Therefore:

$$
\boxed{
\frac{\partial J_i}{\partial b}
=
\hat y_i-y_i
}
$$

---

# 5. SGD Update Equations

The general gradient descent update is:

$$
\theta := \theta-\eta\nabla J
$$

Therefore:

$$
w := w-\eta(\hat y_i-y_i)x_i
$$

and:

$$
b := b-\eta(\hat y_i-y_i)
$$

These updates happen **after every training example**.

---

# 6. Simple Example

Suppose:

$$
x=2
$$

$$
y=10
$$

Initial parameters:

$$
w=2
$$

$$
b=1
$$

Learning rate:

$$
\eta=0.01
$$

### Step 1: Prediction

$$
\hat y=wx+b
$$

$$
=2(2)+1
$$

$$
=5
$$

Actual value:

$$
y=10
$$

Error:

$$
\hat y-y=5-10=-5
$$

---

### Step 2: Calculate gradient

For $w$:

$$
\frac{\partial J}{\partial w}
=
-5(2)
=
-10
$$

For $b$:

$$
\frac{\partial J}{\partial b}
=
-5
$$

---

### Step 3: Update $w$

$$
w_{new}
=
2-0.01(-10)
$$

$$
w_{new}=2.1
$$

---

### Step 4: Update $b$

$$
b_{new}
=
1-0.01(-5)
$$

$$
b_{new}=1.05
$$

So after **one training example**:

$$
w=2.1
$$

$$
b=1.05
$$

The model has moved toward a better solution.

---

# 7. How SGD Works

A typical SGD training process looks like:

```text
Initialize parameters
       ↓
Shuffle training dataset
       ↓
Take one training example
       ↓
Calculate prediction
       ↓
Calculate loss
       ↓
Calculate gradient
       ↓
Update parameters
       ↓
Take next example
       ↓
Calculate prediction
       ↓
Calculate gradient
       ↓
Update parameters
       ↓
...
       ↓
Complete one epoch
       ↓
Shuffle again
       ↓
Next epoch
```

---

# 8. What is an Epoch?

An **epoch** means that the model has processed the entire training dataset once.

Suppose:

```text
Dataset = 1000 samples
```

With pure SGD:

```text
1 sample → 1 update
```

Therefore:

```text
1000 samples
=
1000 parameter updates
=
1 epoch
```

If we train for 10 epochs:

$$
1000\times10=10,000
$$

parameter updates.

This is a major difference from batch gradient descent.

---

# 9. Why Do We Shuffle the Data?

Shuffling is extremely important in SGD.

Suppose the dataset is ordered like:

```text
Class A
Class A
Class A
Class A
...
Class B
Class B
Class B
...
```

The model could receive many similar examples consecutively.

This can produce poor or unstable updates.

Instead:

```text
Before:

A A A A A B B B B B

After shuffle:

A B A B B A A B A B
```

This gives SGD a more representative sequence of examples.

In practice:

```python
indices = np.random.permutation(m)
```

is commonly used to shuffle the training data.

---

# 10. Why is SGD "Stochastic"?

The word **stochastic** means involving randomness/probability.

For every update, SGD uses a sample:

$$
(x_i,y_i)
$$

instead of calculating the exact gradient over the complete dataset.

Because the individual sample gives only an approximation to the overall gradient, the updates contain some randomness/noise.

For example:

```text
Batch GD:

Gradient:
→ → → → → → → → →


SGD:

Gradient:
↗ → ↘ → ↗ ↘ → ↗
```

The SGD path tends to be noisy.

---

# 11. SGD Path vs Batch GD Path

Imagine the loss function looks like a valley:

```text
Loss
 ↑
 |       \        /
 |        \      /
 |         \____/
 |              \
 +--------------------→ Parameters
```

Batch Gradient Descent generally moves smoothly toward the minimum:

```text
Start
  ↓
  ↓
  ↘
    ↘
      ↘
       Minimum
```

SGD may behave like:

```text
Start
  ↓
   ↘
     ↙
       ↘
      ↙
        ↘
          ↙
        Minimum region
```

It fluctuates because every sample produces a slightly different gradient.

---

# 12. SGD Does Not Usually Converge Smoothly

This is one of the most important concepts.

Batch GD:

```text
Loss
│\
│ \
│  \
│   \
│    \____
│
└──────────→
```

SGD:

```text
Loss
│\
│ \  ↗
│  \/ \↘
│    ↗ \
│      \↘
│       ~~~
└────────────→
```

SGD may continue jumping around the minimum.

Therefore, the loss might not decrease smoothly.

This is **normal**.

---

# 13. Advantages of SGD

## 13.1 Much faster updates

SGD doesn't have to process the entire dataset before updating the model.

---

## 13.2 Memory efficient

You only need to process one sample at a time.

This can be useful for extremely large datasets.

---

## 13.3 Works well with huge datasets

Suppose you have:

$$
10^8
$$

training examples.

Batch GD may become expensive because every update requires processing all $10^8$ samples.

SGD can update continuously.

---

## 13.4 Can escape shallow local minima

The noise introduced by individual examples can sometimes help SGD move out of problematic regions.

This is particularly useful for complex non-convex optimization problems such as neural networks.

---

## 13.5 Can work with streaming data

SGD is naturally suited to situations where data arrives continuously.

For example:

```text
New data
   ↓
Model update
   ↓
New data
   ↓
Model update
```

---
