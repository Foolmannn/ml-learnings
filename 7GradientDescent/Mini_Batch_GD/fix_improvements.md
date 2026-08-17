

Yes — there is a **significant issue in your implementation**, and it can absolutely cause a very poor $R^2$ score.

The main problem is this line:

```python
coef_der = -2 * np.dot((y_train[idx] - y_hat), X_train[idx]) / X_train.shape[0]
```

You're calculating the gradient using a **mini-batch**, but dividing by the **entire training-set size**.

It should be divided by the **batch size**.

---

# 1. The Main Bug

You have:

```python
coef_der = -2 * np.dot(
    (y_train[idx] - y_hat),
    X_train[idx]
) / X_train.shape[0]
```

Suppose:

```text
X_train.shape[0] = 1000
batch_size = 32
```

Your gradient becomes:

$$
\frac{-2}{1000}\sum_{i=1}^{32}(y_i-\hat y_i)x_i
$$

But for mini-batch GD, you want:

$$
\boxed{
\frac{-2}{32}\sum_{i=1}^{32}(y_i-\hat y_i)x_i
}
$$

So the correct code is:

```python
coef_der = -2 * np.dot(
    (y_train[idx] - y_hat),
    X_train[idx]
) / self.batch_size
```

Or, even better:

```python
coef_der = -2 * np.mean(
    (y_train[idx] - y_hat)[:, np.newaxis] * X_train[idx],
    axis=0
)
```

---

# 2. Why This Makes Such a Big Difference

Imagine:

```text
training samples = 1000
batch size       = 32
```

Your gradient is approximately scaled by:

$$
\frac{32}{1000}=0.032
$$

of what it should be.

So your weight updates are roughly **31 times smaller** than intended.

Your optimizer is essentially saying:

> "I'll learn, but extremely slowly." 😄

Even after 100 epochs, the coefficients may not have moved enough toward the optimum.

---

# 3. Your Intercept Gradient Is Correct

You have:

```python
intercept_der = -2 * np.mean(y_train[idx] - y_hat)
```

This is correct.

Because:

$$
MSE =
\frac{1}{B}
\sum_{i=1}^{B}(y_i-\hat y_i)^2
$$

and:

$$
\frac{\partial MSE}{\partial b}
=
-\frac{2}{B}
\sum_{i=1}^{B}(y_i-\hat y_i)
$$

which is equivalent to:

```python
-2 * np.mean(y_batch - y_hat)
```

So your intercept calculation is fine.

---

# 4. Corrected Version of Your Code

I'd make a few improvements while keeping your implementation conceptually the same:

```python
import numpy as np
import random


class MBGDRegressor:

    def __init__(self, batch_size, learning_rate=0.01, epochs=100):

        self.coef_ = None
        self.intercept_ = None

        self.batch_size = batch_size
        self.lr = learning_rate
        self.epochs = epochs

    def fit(self, X_train, y_train):

        # Initialize parameters
        self.intercept_ = 0
        self.coef_ = np.ones(X_train.shape[1])

        n_samples = X_train.shape[0]

        for epoch in range(self.epochs):

            for j in range(n_samples // self.batch_size):

                # Randomly select a mini-batch
                idx = random.sample(
                    range(n_samples),
                    self.batch_size
                )

                X_batch = X_train[idx]
                y_batch = y_train[idx]

                # Forward pass
                y_hat = np.dot(X_batch, self.coef_) + self.intercept_

                # Gradient of intercept
                intercept_der = -2 * np.mean(y_batch - y_hat)

                # Gradient of coefficients
                coef_der = (
                    -2
                    * np.dot(
                        (y_batch - y_hat),
                        X_batch
                    )
                    / self.batch_size
                )

                # Update parameters
                self.intercept_ -= self.lr * intercept_der
                self.coef_ -= self.lr * coef_der

        print("Intercept:", self.intercept_)
        print("Coefficients:", self.coef_)

    def predict(self, X_test):

        return np.dot(X_test, self.coef_) + self.intercept_
```

The most important change is:

```python
/ self.batch_size
```

instead of:

```python
/ X_train.shape[0]
```

---

# 5. But There Is Another Thing I'd Change

You're doing:

```python
for j in range(int(X_train.shape[0] / self.batch_size)):

    idx = random.sample(
        range(X_train.shape[0]),
        self.batch_size
    )
```

This means that **every mini-batch randomly samples from the entire dataset again**.

For example:

```text
Dataset = 100 samples
Batch size = 10
```

You might get:

```text
Batch 1 → [1, 5, 8, 20, ...]
Batch 2 → [3, 7, 8, 15, ...]
Batch 3 → [1, 9, 20, 33, ...]
```

Some samples can appear multiple times during the same epoch, while some samples may not appear at all.

That's not necessarily mathematically wrong — random mini-batch sampling can be used — but the more conventional implementation is:

### Shuffle once per epoch

```text
Epoch
 ↓
Shuffle entire dataset
 ↓
Batch 1
Batch 2
Batch 3
...
```

---

# 6. Better Implementation

I'd recommend implementing mini-batches like this:

```python
import numpy as np


class MBGDRegressor:

    def __init__(self, batch_size=32, learning_rate=0.01, epochs=100):

        self.coef_ = None
        self.intercept_ = None

        self.batch_size = batch_size
        self.lr = learning_rate
        self.epochs = epochs

    def fit(self, X_train, y_train):

        n_samples = X_train.shape[0]

        # Initialize parameters
        self.intercept_ = 0
        self.coef_ = np.ones(X_train.shape[1])

        for epoch in range(self.epochs):

            # Shuffle dataset
            indices = np.random.permutation(n_samples)

            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]

            # Mini-batches
            for start in range(0, n_samples, self.batch_size):

                end = start + self.batch_size

                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                # Prediction
                y_hat = np.dot(X_batch, self.coef_) + self.intercept_

                # Error
                error = y_batch - y_hat

                # Gradients
                intercept_der = -2 * np.mean(error)

                coef_der = (
                    -2
                    * np.dot(error, X_batch)
                    / len(X_batch)
                )

                # Update
                self.intercept_ -= self.lr * intercept_der
                self.coef_ -= self.lr * coef_der

        return self

    def predict(self, X):

        return np.dot(X, self.coef_) + self.intercept_
```

Notice this:

```python
/ len(X_batch)
```

instead of:

```python
/ self.batch_size
```

I actually prefer `len(X_batch)` because of the **last batch**.

---

# 7. Why `len(X_batch)` Is Better

Suppose:

```text
n_samples = 103
batch_size = 32
```

Your batches are:

```text
32
32
32
7
```

For the final batch:

```python
len(X_batch)
```

is:

```text
7
```

whereas:

```python
self.batch_size
```

is:

```text
32
```

Therefore:

```python
/ len(X_batch)
```

always calculates the actual mini-batch mean correctly.

---

# 8. Your Gradient Derivation

Let's connect this directly to your code.

Your model is:

$$
\hat y = Xw+b
$$

Your MSE is:

$$
J =
\frac{1}{B}
\sum_{i=1}^{B}
(y_i-\hat y_i)^2
$$

### Intercept

$$
\frac{\partial J}{\partial b}
=
-\frac{2}{B}
\sum_{i=1}^{B}(y_i-\hat y_i)
$$

Your code:

```python
intercept_der = -2 * np.mean(y_batch - y_hat)
```

Correct.

---

### Coefficients

For each coefficient:

$$
\frac{\partial J}{\partial w_j}
=
-\frac{2}{B}
\sum_{i=1}^{B}
(y_i-\hat y_i)x_{ij}
$$

Vectorized:

$$
\boxed{
\nabla_w J =
-\frac{2}{B}X_B^T(y_B-\hat y)
}
$$

Your code should therefore be:

```python
coef_der = -2 * np.dot(
    (y_batch - y_hat),
    X_batch
) / len(X_batch)
```

Exactly.

---

# 9. One More Important Issue: Feature Scaling

Even after fixing the denominator, if you're getting a poor $R^2$, **check whether your features are scaled**.

This is especially important for Gradient Descent.

Suppose your features are:

```text
Age       → 18 to 70
Salary    → 20,000 to 500,000
Experience → 0 to 30
```

The scales are very different.

Gradient descent can struggle because the loss surface becomes elongated:

```text
       w2
       ↑
       │       ______
       │     /        \
       │    /          \
       │    \          /
       │     \________/
       │
       └────────────────→ w1
```

Scaling makes optimization much easier.

Try:

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

Then:

```python
model = MBGDRegressor(
    batch_size=32,
    learning_rate=0.01,
    epochs=100
)

model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
```

Then:

```python
from sklearn.metrics import r2_score

print(r2_score(y_test, y_pred))
```

---

# 10. Learning Rate Also Matters

After fixing the gradient, your previous:

```python
learning_rate=0.01
```

might be too large or too small depending on your dataset.

Try something like:

```python
learning_rates = [
    0.0001,
    0.001,
    0.01,
    0.05,
    0.1
]
```

But **don't blindly increase the learning rate** before scaling the features.

A good order is:

```text
1. Fix gradient calculation
          ↓
2. Scale X
          ↓
3. Try different learning rates
          ↓
4. Try different batch sizes
          ↓
5. Increase epochs if necessary
```

---

# 11. One Subtle Point About Your Original Code

You calculate:

```python
y_hat = ...
```

Then:

```python
intercept_der = ...
self.intercept_ = self.intercept_ - self.lr * intercept_der

coef_der = ...
self.coef_ = self.coef_ - self.lr * coef_der
```

Some people might suggest calculating the coefficient gradient **before** updating the intercept.

For clarity, I would do:

```python
y_hat = ...

intercept_der = ...
coef_der = ...

self.intercept_ -= self.lr * intercept_der
self.coef_ -= self.lr * coef_der
```

This is cleaner because both gradients correspond explicitly to the **same parameter state**.

So:

```python
# Forward pass
y_hat = ...

# Calculate ALL gradients
intercept_der = ...
coef_der = ...

# Update ALL parameters
self.intercept_ -= self.lr * intercept_der
self.coef_ -= self.lr * coef_der
```

That's the pattern you should remember.

---

# 12. Final Diagnosis of Your Code

Your original implementation:

```python
intercept_der = -2 * np.mean(y_train[idx] - y_hat)
```

✅ Correct

```python
self.intercept_ = self.intercept_ - (self.lr * intercept_der)
```

✅ Correct

```python
coef_der = -2 * np.dot(
    (y_train[idx] - y_hat),
    X_train[idx]
) / X_train.shape[0]
```

❌ **Main problem**

Should be:

```python
coef_der = -2 * np.dot(
    (y_train[idx] - y_hat),
    X_train[idx]
) / len(idx)
```

or:

```python
coef_der = -2 * np.dot(
    (y_train[idx] - y_hat),
    X_train[idx]
) / len(X_train[idx])
```

And:

```python
self.coef_ = self.coef_ - (self.lr * coef_der)
```

✅ Correct.

The other major thing to check is **feature scaling**.

So if your $R^2$ is very low, I'd first change:

```python
/ X_train.shape[0]
```

to:

```python
/ len(X_train[idx])
```

and then make sure you're doing:

```text
StandardScaler
      ↓
Mini-Batch GD
      ↓
Prediction
      ↓
R²
```

That should make a substantial difference.

---