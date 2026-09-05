# Logistic Regression Using the Perceptron Trick

The **perceptron trick** is a very intuitive way to understand how a linear classifier learns a decision boundary. Logistic Regression takes the same basic linear idea but improves it by producing **probabilities** and using a differentiable loss function.

So, the relationship is:

> **Perceptron → linear decision boundary → sigmoid → probability → logistic loss → gradient descent → Logistic Regression**

---

## 1. Start with the Perceptron

Suppose we have two features:

* \(x_1\) = hours studied
* \(x_2\) = hours slept

and want to predict:

* \(y=1\): Pass
* \(y=0\): Fail

A perceptron calculates:

$$
z = w_1x_1+w_2x_2+b
$$

or in vector form:

$$
z=w^Tx+b
$$

Then it applies a **step function**:

$$
\hat y =
\begin{cases}
1 & z\geq0\\
0 & z<0
\end{cases}
$$

The decision boundary is therefore:

$$
w^Tx+b=0
$$

For two features:

$$
w_1x_1+w_2x_2+b=0
$$

This is a straight line.

---

# 2. The Perceptron Trick

The basic perceptron makes a hard decision:

```text
z = w₁x₁ + w₂x₂ + b

             z >= 0
                ↓
             predict 1

             z < 0
                ↓
             predict 0
```

The problem is that the step function isn't smooth.

For example:

$$
z=10
$$

and

$$
z=0.1
$$

both produce:

$$
\hat y=1
$$

The model doesn't tell us **how confident** it is.

That's where logistic regression comes in.

---

# 3. Replace the Step Function with Sigmoid

Instead of:

$$
\hat y=
\begin{cases}
1 & z\geq0\\
0 & z<0
\end{cases}
$$

we use the **sigmoid function**:

$$
\sigma(z)=\frac{1}{1+e^{-z}}
$$

Therefore:

$$
\boxed{\hat y=\sigma(w^Tx+b)}
$$

The output is always between 0 and 1.

For example:

| \(z\) | Sigmoid |
| ----: | ------: |
|    -5 |  0.0067 |
|    -2 |   0.119 |
|    -1 |   0.269 |
|     0 |   0.500 |
|     1 |   0.731 |
|     2 |   0.881 |
|     5 |   0.993 |

So logistic regression can say:

> "I'm 99.3% confident this is class 1."

rather than simply saying:

> "1."

---

# 4. Why Is This Called Logistic Regression?

Although the name says **regression**, we're usually using it for **classification**.

The model first performs a linear calculation:

$$
z=w^Tx+b
$$

This is essentially linear regression's equation.

Then we transform the result using sigmoid:

$$
P(y=1|x)=\frac{1}{1+e^{-(w^Tx+b)}}
$$

So:

$$
\boxed{
P(y=1|x)=\sigma(w^Tx+b)
}
$$

This is the fundamental equation of logistic regression.

---

# 5. Where the Perceptron Trick Comes In

Think of the perceptron as:

$$
\text{Features}
\rightarrow
\text{Weighted Sum}
\rightarrow
\text{Step Function}
\rightarrow
\text{Class}
$$

Logistic regression changes this to:

$$
\text{Features}
\rightarrow
\text{Weighted Sum}
\rightarrow
\text{Sigmoid}
\rightarrow
\text{Probability}
\rightarrow
\text{Class}
$$

The **linear part is exactly the same idea**:

$$
z=w^Tx+b
$$

The major improvement is how we train the model.

---

# 6. Perceptron Learning Rule

The classic perceptron updates weights when it makes an incorrect prediction.

One common representation is:

$$
w_j \leftarrow w_j+\eta(y-\hat y)x_j
$$

and:

$$
b\leftarrow b+\eta(y-\hat y)
$$

where:

* \(w_j\) = weight
* \(\eta\) = learning rate
* \(y\) = actual value
* \(\hat y\) = predicted class
* \(x_j\) = feature

Suppose:

$$
y=1
$$

but:

$$
\hat y=0
$$

Then:

$$
y-\hat y=1
$$

so the weights move in the direction of the input.

If:

$$
y=0
$$

but:

$$
\hat y=1
$$

then:

$$
y-\hat y=-1
$$

and the weights move in the opposite direction.

This is the **perceptron trick**:

> If the point is classified incorrectly, modify the weights so that the point moves toward the correct side of the decision boundary.

---

# 7. Why Logistic Regression Needs a Different Trick

Here's an important distinction.

The perceptron uses:

$$
\hat y=\text{step}(z)
$$

The step function is not differentiable at the decision boundary and has zero gradient almost everywhere.

Therefore, ordinary gradient descent doesn't work nicely with it.

Logistic regression replaces the step function with sigmoid:

$$
\hat y=\sigma(z)
$$

Sigmoid is differentiable:

$$
\frac{d\sigma(z)}{dz}
=
\sigma(z)(1-\sigma(z))
$$

This gives us a smooth way to adjust the weights.

---

# 8. Logistic Regression Training

The training process is:

### Step 1 — Calculate weighted sum

$$
z=w^Tx+b
$$

### Step 2 — Apply sigmoid

$$
\hat y=\frac{1}{1+e^{-z}}
$$

### Step 3 — Calculate error using Log Loss

$$
L=
-\left[
y\log(\hat y)+(1-y)\log(1-\hat y)
\right]
$$

### Step 4 — Calculate gradients

For one training example:

$$
\frac{\partial L}{\partial w}
=
(\hat y-y)x
$$

and:

$$
\frac{\partial L}{\partial b}
=
\hat y-y
$$

### Step 5 — Update weights

$$
w\leftarrow w-\eta(\hat y-y)x
$$

$$
b\leftarrow b-\eta(\hat y-y)
$$

And repeat.

---

# 9. Notice Something Interesting

Compare the perceptron update:

$$
w\leftarrow w+\eta(y-\hat y)x
$$

with logistic regression:

$$
w\leftarrow w-\eta(\hat y-y)x
$$

These are algebraically the same **form**:

$$
w\leftarrow w+\eta(y-\hat y)x
$$

The huge difference is what \(\hat y\) represents.

### Perceptron

$$
\hat y\in\{0,1\}
$$

### Logistic Regression

$$
\hat y\in(0,1)
$$

This makes logistic regression's update **continuous**.

That's one of the most useful ways to connect perceptron and logistic regression.

---

# 10. Numerical Example

Suppose:

$$
x=2
$$

Actual class:

$$
y=1
$$

Initial parameters:

$$
w=0
$$

$$
b=0
$$

Learning rate:

$$
\eta=0.1
$$

---

## Step 1: Weighted sum

$$
z=wx+b
$$

$$
z=(0)(2)+0
$$

$$
z=0
$$

---

## Step 2: Sigmoid

$$
\hat y=\frac{1}{1+e^{-0}}
$$

$$
\hat y=0.5
$$

So the model predicts:

$$
P(y=1)=0.5
$$

---

## Step 3: Calculate error

$$
\hat y-y=0.5-1
$$

$$
=-0.5
$$

---

## Step 4: Calculate gradient

$$
\frac{\partial L}{\partial w}
=
(\hat y-y)x
$$

$$
=(-0.5)(2)
$$

$$
=-1
$$

Bias gradient:

$$
\frac{\partial L}{\partial b}=-0.5
$$

---

## Step 5: Update

$$
w_{\text{new}}
=
w-\eta\frac{\partial L}{\partial w}
$$

$$
=0-(0.1)(-1)
$$

$$
\boxed{w=0.1}
$$

Bias:

$$
b_{\text{new}}
=
0-(0.1)(-0.5)
$$

$$
\boxed{b=0.05}
$$

So after one update:

$$
w=0.1,\qquad b=0.05
$$

---

# 11. Next Prediction

Now:

$$
z=wx+b
$$

$$
z=(0.1)(2)+0.05
$$

$$
z=0.25
$$

Sigmoid:

$$
\hat y=
\frac{1}{1+e^{-0.25}}
$$

Approximately:

$$
\boxed{\hat y=0.562}
$$

Initially:

$$
P(y=1)=0.5
$$

After one update:

$$
P(y=1)=0.562
$$

The model has moved toward the correct answer.

That's the key idea behind learning.

---

# 12. Why Does the Weight Move in the Correct Direction?

Suppose:

$$
y=1
$$

and the model predicts:

$$
\hat y=0.2
$$

Then:

$$
\hat y-y=-0.8
$$

Therefore:

$$
w_{\text{new}}
=
w-\eta(-0.8)x
$$

which means:

$$
w_{\text{new}}
=
w+0.8\eta x
$$

The weight moves **toward increasing the prediction**.

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

and:

$$
w_{\text{new}}
=
w-0.8\eta x
$$

The weight moves in the opposite direction.

So the model automatically learns:

> **Prediction too low for class 1 → increase the score.**

> **Prediction too high for class 0 → decrease the score.**

---

# 13. Decision Boundary

Even though logistic regression produces probabilities, the actual decision boundary is still linear.

Usually:

$$
\hat y\geq0.5\Rightarrow class\ 1
$$

and:

$$
\hat y<0.5\Rightarrow class\ 0
$$

Because:

$$
\sigma(0)=0.5
$$

therefore:

$$
w^Tx+b=0
$$

is the decision boundary.

For two features:

$$
w_1x_1+w_2x_2+b=0
$$

So logistic regression is still a **linear classifier**.

---

# 14. Geometric Interpretation

Imagine data points:

```text
Class 1       ● ● ●
             ● ●

--------------------------  ← decision boundary

        ○ ○
      ○ ○ ○
Class 0
```

The model learns:

$$
w_1x_1+w_2x_2+b=0
$$

The perceptron tries to place the line so that:

* Class 1 is on one side
* Class 0 is on the other side

Logistic regression does the same thing, but instead of just asking:

> "Which side?"

it asks:

> "How strongly does this point belong to class 1?"

---

# 15. Score vs Probability

This distinction is extremely important.

The linear model produces a **score**:

$$
z=w^Tx+b
$$

For example:

$$
z=3
$$

This isn't a probability.

After sigmoid:

$$
P(y=1|x)=\sigma(3)
$$

$$
\approx0.953
$$

Now we have a probability.

So:

```text
x
 ↓
wᵀx + b
 ↓
score z
 ↓
sigmoid
 ↓
probability
 ↓
threshold
 ↓
class
```

---

# 16. Why Not Use Mean Squared Error?

You might wonder why we don't simply use:

$$
MSE=(y-\hat y)^2
$$

as in linear regression.

We can mathematically construct such a model, but it doesn't give the nicest optimization properties for logistic regression.

Instead, we use **Binary Cross Entropy / Log Loss**:

$$
\boxed{
L=
-\left[y\log(\hat y)+(1-y)\log(1-\hat y)\right]
}
$$

This loss strongly penalizes confident incorrect predictions.

For example, if:

$$
y=1
$$

and:

$$
\hat y=0.99
$$

the loss is tiny.

But if:

$$
y=1
$$

and:

$$
\hat y=0.01
$$

the loss is very large.

---

# 17. Why the Gradient Becomes So Simple

This is one of the beautiful parts of logistic regression.

We have:

$$
L=
-\left[
y\log(\hat y)+(1-y)\log(1-\hat y)
\right]
$$

and:

$$
\hat y=\sigma(z)
$$

After applying the chain rule:

$$
\boxed{
\frac{\partial L}{\partial z}
=
\hat y-y
}
$$

Then because:

$$
z=w^Tx+b
$$

we get:

$$
\boxed{
\frac{\partial L}{\partial w}
=
(\hat y-y)x
}
$$

and:

$$
\boxed{
\frac{\partial L}{\partial b}
=
\hat y-y
}
$$

That's why the logistic regression update resembles the perceptron update so closely.

---

# 18. Perceptron vs Logistic Regression

| Feature           | Perceptron                       | Logistic Regression   |
| ----------------- | -------------------------------- | --------------------- |
| Linear score      | \(w^Tx+b\)                       | \(w^Tx+b\)            |
| Activation        | Step                             | Sigmoid               |
| Output            | 0 or 1                           | 0 to 1                |
| Probability       | ❌                                | ✅                     |
| Loss              | Perceptron loss / mistake-driven | Log loss              |
| Gradient-based    | Not naturally                    | ✅                     |
| Decision boundary | Linear                           | Linear                |
| Differentiable    | Step isn't                       | Sigmoid is            |
| Confidence        | Poor                             | Good                  |
| Common use        | Basic linear classifier          | Binary classification |

---

# 19. The Most Important Connection

You can remember the evolution like this:

### Perceptron

$$
\boxed{
w^Tx+b
\rightarrow
Step
\rightarrow
Class
}
$$

### Logistic Regression

$$
\boxed{
w^Tx+b
\rightarrow
Sigmoid
\rightarrow
Probability
\rightarrow
Class
}
$$

And the learning update becomes:

### Perceptron

$$
\boxed{
w\leftarrow w+\eta(y-\hat y)x
}
$$

where prediction is hard:

$$
\hat y\in\{0,1\}
$$

### Logistic Regression

$$
\boxed{
w\leftarrow w+\eta(y-\hat y)x
}
$$

where prediction is soft:

$$
\hat y\in(0,1)
$$

This is the **perceptron intuition behind logistic regression**.

---

# 20. Complete Logistic Regression Algorithm

For every epoch:

```text
Initialize w and b

Repeat for each epoch:

    1. Calculate linear score
       z = wᵀx + b

    2. Calculate probability
       ŷ = sigmoid(z)

    3. Calculate error
       error = ŷ - y

    4. Calculate gradients
       dw = error × x
       db = error

    5. Update parameters
       w = w - learning_rate × dw
       b = b - learning_rate × db

After training:

    probability = sigmoid(wᵀx + b)

    if probability >= 0.5:
        class = 1
    else:
        class = 0
```

For a dataset with \(n\) samples, the vectorized gradients are:

$$
\boxed{
dw=\frac{1}{n}X^T(\hat y-y)
}
$$

$$
\boxed{
db=\frac{1}{n}\sum_{i=1}^{n}(\hat y_i-y_i)
}
$$

Then:

$$
w\leftarrow w-\eta dw
$$

$$
b\leftarrow b-\eta db
$$

---

# 21. One Mental Model to Remember

Think of the whole process as a **smart perceptron**:

```text
             INPUT FEATURES
                  │
                  ▼
          Weighted Sum
          z = wᵀx + b
                  │
                  ▼
              Sigmoid
                  │
                  ▼
          Probability 0–1
                  │
                  ▼
          ┌───────────────┐
          │ Threshold 0.5 │
          └───────────────┘
             │         │
             ▼         ▼
            0           1
                  │
                  ▼
              Compare
              with y
                  │
                  ▼
             Error = ŷ-y
                  │
                  ▼
          Gradient Descent
                  │
                  ▼
          Update w and b
                  │
                  └──────► Repeat
```

**In one sentence:** Logistic regression can be understood as a perceptron-like linear classifier where the hard step activation is replaced by a sigmoid, and the mistake-driven update is turned into a smooth gradient-based update using log loss.
