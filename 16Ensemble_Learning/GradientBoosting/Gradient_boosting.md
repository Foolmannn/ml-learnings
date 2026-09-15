# Gradient Boosting in Machine Learning — In Detail

**Gradient Boosting** is one of the most important **ensemble learning** algorithms in machine learning. It is especially powerful for **tabular/structured data** and is the foundation of algorithms such as **XGBoost, LightGBM, and CatBoost**.

Since you've already studied **Bagging, Random Forest, Voting, and AdaBoost**, the key thing to understand is that Gradient Boosting takes the idea of **sequentially adding weak learners** and uses **gradient descent** to decide what each new learner should fix.

---

# 1. What is Gradient Boosting?

Gradient Boosting is an ensemble technique that builds a model **sequentially**.

Instead of training many independent models like Bagging:

```text
Dataset
   │
   ├── Tree 1
   ├── Tree 2
   ├── Tree 3
   └── Tree 4
```

Gradient Boosting trains trees one after another:

```text
Dataset
   │
   ▼
Tree 1
   │
   ▼
Errors / Residuals
   │
   ▼
Tree 2
   │
   ▼
Remaining Errors
   │
   ▼
Tree 3
   │
   ▼
...
```

Each new tree tries to **correct the mistakes made by the previous ensemble**.

The final prediction is the combination of all the weak learners.

---

# 2. The Main Idea

Suppose we're predicting house prices.

Actual:

```text
100
200
300
400
```

Our first decision tree predicts:

```text
120
180
280
350
```

The errors are:

```text
Actual - Prediction

100 - 120 = -20
200 - 180 = 20
300 - 280 = 20
400 - 350 = 50
```

So:

```text
Residuals:
[-20, 20, 20, 50]
```

Gradient Boosting trains another tree to predict these residuals.

The second tree might predict:

```text
[-15, 15, 25, 40]
```

Now we add this correction to the original prediction.

```text
New prediction =
Old prediction + learning_rate × correction
```

So if:

```text
learning_rate = 0.1
```

then:

```text
120 + 0.1(-15) = 118.5
180 + 0.1(15)  = 181.5
280 + 0.1(25)  = 282.5
350 + 0.1(40)  = 354
```

The model is gradually improving.

---

# 3. Why is it called "Gradient" Boosting?

This is the most important theoretical point.

It isn't limited to simply predicting residuals.

Gradient Boosting minimizes a **loss function** using the idea of **gradient descent**.

Suppose our model is:

$$
F(x)
$$

and our loss function is:

$$
L(y,F(x))
$$

We want to find a model that minimizes:

$$
\sum_{i=1}^{n}L(y_i,F(x_i))
$$

Instead of changing all model parameters at once, Gradient Boosting builds a new function/tree that moves the predictions in the direction that **reduces the loss**.

The new model is approximately:

$$
F_m(x)=F_{m-1}(x)+\eta h_m(x)
$$

where:

* \(F_m(x)\) = new model
* \(F_{m-1}(x)\) = previous model
* \(h_m(x)\) = new weak learner
* \(\eta\) = learning rate

This is the core equation of Gradient Boosting.

---

# 4. What does the gradient mean?

Suppose our loss is Mean Squared Error:

$$
L(y,\hat y)=(y-\hat y)^2
$$

We want to know:

> In which direction should I change my prediction to reduce the loss?

We calculate the derivative of the loss with respect to the prediction:

$$
\frac{\partial L}{\partial \hat y}
$$

The negative gradient tells us the direction in which we should move.

Therefore, Gradient Boosting trains the next weak learner to approximate:

$$
-\frac{\partial L}{\partial F(x)}
$$

These are called **pseudo-residuals**.

---

# 5. Gradient Boosting for Regression

Let's understand the classic example.

Suppose:

|  X | Actual Y |
| -: | -------: |
|  1 |       10 |
|  2 |       20 |
|  3 |       30 |
|  4 |       40 |

### Step 1 — Initial prediction

For squared-error loss, the initial prediction is generally the mean:

$$
F_0(x)=\bar y
$$

Here:

$$
\bar y=25
$$

Therefore:

```text
Initial prediction = 25
```

For every observation:

```text
25
25
25
25
```

---

## Step 2 — Calculate residuals

$$
r_i=y_i-\hat y_i
$$

Therefore:

```text
10 - 25 = -15
20 - 25 = -5
30 - 25 =  5
40 - 25 = 15
```

So:

```text
[-15, -5, 5, 15]
```

---

## Step 3 — Train a tree on residuals

Now we train a small decision tree:

```text
X → residual
```

The tree learns:

```text
X <= 2.5 → negative correction
X > 2.5  → positive correction
```

For example:

```text
Tree prediction:

X=1 → -10
X=2 → -10
X=3 → +10
X=4 → +10
```

---

# 6. Add the correction

Suppose:

$$
\eta=0.1
$$

Then:

$$
F_1(x)=F_0(x)+0.1h_1(x)
$$

For X = 1:

$$
25+0.1(-10)=24
$$

For X = 4:

$$
25+0.1(10)=26
$$

The predictions become:

```text
24
24
26
26
```

They're still not perfect, so another tree is trained.

---

# 7. Second tree

The new residuals are calculated again.

```text
Actual       Prediction
10           24
20           24
30           26
40           26
```

Residual:

```text
-14
 -4
 +4
+14
```

The next tree learns these remaining errors.

Then:

```text
Tree 1
   ↓
Residuals
   ↓
Tree 2
   ↓
Residuals
   ↓
Tree 3
   ↓
...
```

This continues for many iterations.

---

# 8. The Complete Gradient Boosting Algorithm

The general process is:

### Step 1

Initialize the model:

$$
F_0(x)=\arg\min_\gamma\sum_iL(y_i,\gamma)
$$

For squared error, this is the mean.

---

### Step 2

Calculate pseudo-residuals:

$$
r_{im}
=
-\left[
\frac{\partial L(y_i,F(x_i))}
{\partial F(x_i)}
\right]_{F=F_{m-1}}
$$

---

### Step 3

Train a weak learner on:

```text
X → pseudo-residuals
```

Usually:

```text
Decision Tree
```

---

### Step 4

Find the optimal contribution of that tree.

Conceptually:

$$
\gamma_m
=
\arg\min_\gamma
\sum_i
L(y_i,F_{m-1}(x_i)+\gamma h_m(x_i))
$$

---

### Step 5

Update the model:

$$
F_m(x)=F_{m-1}(x)+\eta\gamma_mh_m(x)
$$

---

### Step 6

Repeat.

```text
F0
 ↓
Tree 1
 ↓
F1
 ↓
Tree 2
 ↓
F2
 ↓
Tree 3
 ↓
...
 ↓
Final model
```

---

# 9. Why Decision Trees?

Gradient Boosting usually uses **shallow decision trees**.

For example:

```text
max_depth = 3
```

A tree might look like:

```text
             Age < 30?
             /       \
           Yes        No
           /           \
     Income < 40K?   Income < 70K?
       /    \          /     \
     -5      2        4       10
```

These trees are called **weak learners** because individually they aren't very powerful.

But hundreds of small trees can collectively create a very powerful model.

---
