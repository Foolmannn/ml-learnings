
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

# 14. Disadvantages of SGD

## 14.1 Noisy convergence

The loss can fluctuate significantly.

---

## 14.2 Learning rate becomes very important

If:

$$
\eta
$$

is too large:

```text
Minimum
  ↑
 / \
/   \

Updates:
↗       ↘
      ↗
            ↘
```

The algorithm may overshoot the minimum.

If it is too small:

```text
Very slow convergence
```

---

## 14.3 Less computationally efficient per update

Although each update is cheap, SGD performs many updates.

---

## 14.4 May never settle exactly at the minimum

Because each individual sample generates a slightly different gradient, SGD can keep bouncing around the minimum.

---

# 15. Learning Rate in SGD

The learning rate controls how large each update is.

$$
\theta_{new}
=
\theta_{old}
-
\eta\nabla J
$$

### Large learning rate

```text
        ↗
      ↙
    ↗
  ↙
```

Can cause instability.

### Small learning rate

```text
→ → → → → → → → → → 
```

Stable but slow.

### Appropriate learning rate

```text
→
  ↘
    ↘
      ↘
        ●
```

Good convergence.

---

# 16. Learning Rate Scheduling

Instead of keeping:

$$
\eta
$$

constant, we can reduce it during training.

For example:

$$
\eta_t=\frac{\eta_0}{1+kt}
$$

where:

- $\eta_0$ = initial learning rate
- $k$ = decay parameter
- $t$ = iteration

Initially:

```text
Large steps
```

Later:

```text
Smaller steps
```

This can help SGD stabilize around the minimum.

---

# 17. SGD From Scratch in Python

Let's implement SGD for linear regression.

```python
import numpy as np

# Training data
X = np.array([1, 2, 3, 4, 5], dtype=float)
y = np.array([2, 4, 6, 8, 10], dtype=float)

# Parameters
w = 0.0
b = 0.0

learning_rate = 0.01
epochs = 20

n = len(X)

for epoch in range(epochs):

    # Shuffle indices
    indices = np.random.permutation(n)

    for i in indices:

        x_i = X[i]
        y_i = y[i]

        # Prediction
        y_pred = w * x_i + b

        # Error
        error = y_pred - y_i

        # Gradients
        dw = error * x_i
        db = error

        # Update parameters
        w -= learning_rate * dw
        b -= learning_rate * db

    print(
        f"Epoch {epoch + 1}: "
        f"w={w:.4f}, b={b:.4f}"
    )
```

Eventually you'll see $w$ move toward approximately:

$$
w\approx2
$$

and:

$$
b\approx0
$$

because the underlying relationship is:

$$
y=2x
$$

---

# 18. SGD Using Scikit-Learn

For linear models, scikit-learn provides `SGDRegressor`.

```python
import numpy as np
from sklearn.linear_model import SGDRegressor

X = np.array([
    [1],
    [2],
    [3],
    [4],
    [5]
])

y = np.array([2, 4, 6, 8, 10])

model = SGDRegressor(
    learning_rate="constant",
    eta0=0.01,
    max_iter=1000,
    random_state=42
)

model.fit(X, y)

print("Weight:", model.coef_)
print("Bias:", model.intercept_)
```

Prediction:

```python
prediction = model.predict([[6]])

print(prediction)
```

The result should be close to:

$$
12
$$

---

# 19. SGD and Feature Scaling

Feature scaling is particularly important for gradient-based algorithms.

Suppose:

```text
Feature 1: 1 → 10
Feature 2: 10,000 → 100,000
```

The optimization landscape can become badly shaped.

Without scaling:

```text
        ______
      /        \
     /          \
    /            \
   \              /
    \____________/
```

The optimizer may zig-zag.

After scaling:

```text
       _____
     /       \
    |    ●    |
     \_______/
```

Optimization becomes easier.

Common techniques:

### Standardization

$$
x'=\frac{x-\mu}{\sigma}
$$

Using:

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)
```

---

# 20. Batch GD vs SGD vs Mini-Batch GD

There are three important approaches.

| Method | Samples per update | Updates per epoch | Noise |
|---|---:|---:|---|
| Batch GD | Entire dataset | 1 | Low |
| SGD | 1 | $m$ | High |
| Mini-Batch GD | Batch of $B$ | $m/B$ | Medium |

Suppose:

$$
m=1000
$$

### Batch GD

```text
1000 samples → Update
```

Updates per epoch:

$$
1
$$

### SGD

```text
1 sample → Update
```

Updates per epoch:

$$
1000
$$

### Mini-Batch

Suppose:

$$
B=32
$$

Then:

```text
32 samples → Update
32 samples → Update
32 samples → Update
...
```

Approximately:

$$
\frac{1000}{32}\approx31.25
$$

updates per epoch.

---

# 21. Why Mini-Batch GD Became So Popular

In modern deep learning, we generally don't use pure SGD in the strict one-example-at-a-time sense.

Instead, we commonly use **mini-batches**.

For example:

```text
Dataset
   │
   ├── Batch 1: 32 samples
   ├── Batch 2: 32 samples
   ├── Batch 3: 32 samples
   └── ...
```

This provides a good compromise:

```text
Batch GD
   │
   │ low noise
   │
Mini-Batch GD
   │
   │ balance
   │
SGD
   │
   │ high noise
```

Mini-batch training also works very well with GPU hardware.

---

# 22. Important Terminology

There is a subtle terminology issue.

People often say:

> "SGD" when they actually mean mini-batch stochastic gradient descent.

For example, in deep learning you may see:

```python
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.01
)
```

The optimizer is called `SGD`, but the model may actually be trained using batches such as:

```text
batch_size = 32
```

Therefore, don't assume that the term `SGD` always means exactly one sample per update in practical deep-learning code.

---

# 23. SGD in Neural Networks

SGD is especially important in neural-network training.

Consider:

```text
Input
  ↓
Hidden Layer
  ↓
Hidden Layer
  ↓
Output
```

For each batch:

```text
Forward Pass
     ↓
Calculate Loss
     ↓
Backpropagation
     ↓
Calculate Gradients
     ↓
SGD Update
```

The update is conceptually:

$$
W:=W-\eta\frac{\partial L}{\partial W}
$$

and:

$$
b:=b-\eta\frac{\partial L}{\partial b}
$$

This happens repeatedly during training.

---

# 24. SGD With Momentum

Pure SGD can be noisy.

Momentum helps smooth the updates.

Instead of:

$$
w:=w-\eta\nabla J
$$

we maintain a velocity:

$$
v_t=\beta v_{t-1}+\nabla J
$$

Then:

$$
w_t=w_{t-1}-\eta v_t
$$

where:

$$
\beta
$$

is usually close to $0.9$.

Conceptually:

```text
SGD:

↗ ↘ ↗ ↘ ↗ ↘


Momentum SGD:

↗
  ↗
    ↗
      ↗
        ↓
```

Momentum remembers the previous direction.

---

# 25. SGD vs Adam

Another extremely important distinction:

**SGD is an optimization algorithm.**

Adam is also an optimization algorithm.

Adam combines ideas related to:

- momentum
- adaptive learning rates

Conceptually:

```text
SGD
 │
 ├── Basic SGD
 │
 └── SGD + Momentum


Adaptive optimizers
 │
 ├── AdaGrad
 ├── RMSProp
 └── Adam
```

Adam often converges quickly and is widely used for neural networks.

However, SGD with momentum can sometimes produce better generalization in certain deep-learning settings.

So:

> Faster optimization does not automatically mean better final model performance.

---

# 26. A Practical Training Example

Suppose you are training a neural network to classify images.

Dataset:

$$
50,000
$$

images.

Batch size:

$$
32
$$

Then approximately:

$$
\frac{50,000}{32}=1562.5
$$

So there are roughly:

$$
1563
$$

updates per epoch.

For 20 epochs:

$$
1563\times20\approx31,260
$$

optimization updates.

The process looks like:

```text
Epoch 1
 ├── Batch 1 → Forward → Loss → Backprop → Update
 ├── Batch 2 → Forward → Loss → Backprop → Update
 ├── Batch 3 → Forward → Loss → Backprop → Update
 └── ...

Epoch 2
 ├── Batch 1
 ├── Batch 2
 └── ...

...
```

---

# 27. SGD and Convex vs Non-Convex Problems

For simple linear regression with MSE, the loss function is convex.

There is theoretically a single global minimum.

For neural networks, the loss landscape is generally **non-convex**.

There may be:

- local minima
- saddle points
- flat regions
- complex valleys

SGD's noisy updates can be useful in navigating these landscapes.

---

# 28. Why the Noise Can Actually Help

Imagine a very shallow local region:

```text
          ______
        /        \
_______/          \_______
```

A deterministic optimizer might settle into a problematic region.

SGD's noisy gradients can provide enough movement to continue exploring.

This doesn't mean:

> "SGD always escapes every local minimum."

Rather:

> The stochasticity can sometimes help optimization explore the loss landscape.

---

# 29. Common Mistakes With SGD

### Mistake 1: Using an excessively large learning rate

```python
learning_rate = 10
```

can cause divergence.

---

### Mistake 2: Not scaling features

Gradient-based algorithms can perform poorly when feature scales differ dramatically.

---

### Mistake 3: Not shuffling training data

Especially problematic when the dataset is ordered.

---

### Mistake 4: Assuming loss must decrease every iteration

With SGD:

```text
Loss:

1.20
0.91
1.03
0.72
0.81
0.60
0.66
0.51
```

This is completely possible.

Look at the overall trend rather than expecting monotonic improvement.

---

### Mistake 5: Confusing epoch and iteration

If:

```text
1000 samples
batch_size = 1
```

then approximately:

```text
1000 iterations = 1 epoch
```

If:

```text
batch_size = 100
```

then:

```text
10 iterations = 1 epoch
```

---

# 30. The Core Formula to Remember

For a general parameter $\theta$:

$$
\boxed{
\theta_{t+1}
=
\theta_t
-
\eta
\nabla_\theta J_i(\theta_t)
}
$$

where:

- $\theta$ = model parameters
- $\eta$ = learning rate
- $J_i$ = loss for one training sample
- $\nabla_\theta J_i$ = gradient of that loss

For linear regression:

$$
\boxed{
w_{t+1}
=
w_t-\eta(\hat y_i-y_i)x_i
}
$$

$$
\boxed{
b_{t+1}
=
b_t-\eta(\hat y_i-y_i)
}
$$

These two equations are the heart of SGD for simple linear regression.

---

# 31. Big Picture

You can remember the three versions like this:

```text
                 Gradient Descent
                       │
          ┌────────────┼────────────┐
          │            │            │
       Batch GD      SGD       Mini-Batch GD
          │            │            │
     All samples    1 sample    Small batch
          │            │            │
       Stable        Noisy       Balanced
          │            │            │
       Slow/large    Fast/cheap   Most practical
       computation   updates      in deep learning
```

### The key intuition

**Batch GD:**

> "Look at everything, then make one decision."

**SGD:**

> "Look at one example, make a decision, then immediately update."

**Mini-Batch GD:**

> "Look at a small group of examples, make a decision, then update."

For your ML progression, the next concepts that fit naturally after SGD are **Mini-Batch Gradient Descent → SGD with Momentum → AdaGrad → RMSProp → Adam**, because these show how modern optimizers solve the instability and learning-rate problems of basic SGD.

---
