# Mathematics of XGBoost — In Detail

Since you've already studied the mathematics of **Gradient Boosting**, the key to understanding XGBoost is to see exactly what it changes:

> **Gradient Boosting uses the gradient of the loss to build the next tree. XGBoost uses a second-order Taylor approximation, meaning it uses both the gradient and the Hessian, and it explicitly regularizes the tree.**

The most important mathematical pieces are:

$$
\boxed{
\text{Loss}
\rightarrow
\text{Gradient + Hessian}
\rightarrow
\text{Tree Objective}
\rightarrow
\text{Optimal Leaf Weight}
\rightarrow
\text{Split Gain}
\rightarrow
\text{New Tree}
}
$$

Let's derive this from the beginning.

---

# 1. Start with the XGBoost model

Suppose our dataset contains:

$$
D=\{(x_i,y_i)\}_{i=1}^{n}
$$

where:

* \(x_i\) = features of sample \(i\)
* \(y_i\) = actual target
* \(n\) = number of training samples

XGBoost builds an ensemble of \(K\) trees.

The prediction is:

$$
\boxed{
\hat y_i=\sum_{k=1}^{K}f_k(x_i)
}
$$

where:

$$
f_k \in \mathcal F
$$

and \(\mathcal F\) is the set of possible decision trees.

So instead of:

```text
One Decision Tree
       ↓
Prediction
```

we have:

```text
Tree 1
   +
Tree 2
   +
Tree 3
   +
...
   +
Tree K
   ↓
Final Prediction
```

---

# 2. Additive learning

XGBoost doesn't build all trees simultaneously.

At iteration \(t\), we already have:

$$
\hat y_i^{(t-1)}
$$

Then we add a new tree \(f_t\):

$$
\boxed{
\hat y_i^{(t)}
=
\hat y_i^{(t-1)}
+
f_t(x_i)
}
$$

If learning rate \(\eta\) is included:

$$
\boxed{
\hat y_i^{(t)}
=
\hat y_i^{(t-1)}
+
\eta f_t(x_i)
}
$$

This is the basic additive boosting structure.

---

# 3. Objective function

XGBoost wants to minimize two things:

1. Prediction error
2. Tree complexity

Therefore:

$$
\boxed{
Obj=
\text{Training Loss}
+
\text{Regularization}
}
$$

At iteration \(t\):

$$
Obj^{(t)}
=
\sum_{i=1}^{n}
l(y_i,\hat y_i^{(t)})
+
\Omega(f_t)
$$

Substitute:

$$
\hat y_i^{(t)}
=
\hat y_i^{(t-1)}+f_t(x_i)
$$

Therefore:

$$
Obj^{(t)}
=
\sum_{i=1}^{n}
l
\left(
y_i,
\hat y_i^{(t-1)}+f_t(x_i)
\right)
+
\Omega(f_t)
$$

This is the fundamental XGBoost objective.

---

# 4. What is the regularization term?

XGBoost doesn't only ask:

> "Does this tree improve predictions?"

It also asks:

> "Is this tree unnecessarily complicated?"

The standard tree regularization term is:

$$
\boxed{
\Omega(f_t)
=
\gamma T
+
\frac12\lambda\sum_{j=1}^{T}w_j^2
}
$$

where:

* \(T\) = number of leaves
* \(w_j\) = weight/value of leaf \(j\)
* \(\gamma\) = penalty for each leaf
* \(\lambda\) = L2 regularization parameter

Therefore:

$$
\boxed{
Obj=
\sum_i l(y_i,\hat y_i)
+
\gamma T
+
\frac12\lambda\sum_jw_j^2
}
$$

This is one of the major reasons XGBoost can control overfitting.

---

# 5. Why do we need Taylor expansion?

The objective contains:

$$
l(y_i,\hat y_i+f_t(x_i))
$$

This can be complicated depending on the loss function.

XGBoost therefore approximates the loss using a **second-order Taylor expansion**.

For a function \(f(x)\):

$$
f(x+\Delta x)
\approx
f(x)
+
f'(x)\Delta x
+
\frac12f''(x)(\Delta x)^2
$$

For XGBoost:

$$
\Delta x=f_t(x_i)
$$

Therefore:

$$
l(y_i,\hat y_i+f_t(x_i))
$$

is approximated by:

$$
\boxed{
l(y_i,\hat y_i)
+
g_i f_t(x_i)
+
\frac12h_i f_t(x_i)^2
}
$$

where:

$$
\boxed{
g_i=
\frac{\partial l(y_i,\hat y_i)}
{\partial\hat y_i}
}
$$

and:

$$
\boxed{
h_i=
\frac{\partial^2l(y_i,\hat y_i)}
{\partial\hat y_i^2}
}
$$

So:

```text
g = First derivative = Gradient
h = Second derivative = Hessian
```

---

# 6. Gradient and Hessian intuition

Think of the loss curve:

```text
Loss
 ^
 |              *
 |           *     *
 |        *           *
 |      *
 |    *
 |___*____________________> Prediction
```

The **gradient** tells us the direction in which the loss is changing.

The **Hessian** tells us about the curvature.

Very roughly:

```text
Gradient
   ↓
Which direction?

Hessian
   ↓
How quickly does it change?
```

This gives XGBoost more information than using only the gradient.

---

# 7. Substitute Taylor expansion into the objective

We had:

$$
Obj^{(t)}
=
\sum_i
l(y_i,\hat y_i+f_t(x_i))
+
\Omega(f_t)
$$

Using Taylor approximation:

$$
Obj^{(t)}
\approx
\sum_i
[
l(y_i,\hat y_i)
+
g_if_t(x_i)
+
\frac12h_if_t(x_i)^2
]
+
\Omega(f_t)
$$

The first term:

$$
\sum_i l(y_i,\hat y_i)
$$

doesn't depend on the new tree \(f_t\).

Therefore, we can ignore it while optimizing.

So:

$$
\boxed{
Obj^{(t)}
\approx
\sum_i
\left[
g_if_t(x_i)
+
\frac12h_if_t(x_i)^2
\right]
+
\Omega(f_t)
}
$$

This is the key XGBoost optimization equation.

---

# 8. Representing a tree mathematically

A decision tree can be represented as:

$$
\boxed{
f(x)=w_{q(x)}
}
$$

where:

* \(q(x)\) tells us which leaf the sample reaches
* \(w_j\) is the weight of leaf \(j\)

Suppose:

```text
                 Root
                /    \
               /      \
          Leaf 1      Leaf 2
```

Then:

$$
q(x)=
\begin{cases}
1 & x\text{ reaches Leaf 1}\\
2 & x\text{ reaches Leaf 2}
\end{cases}
$$

and:

$$
f(x)=w_{q(x)}
$$

---

# 9. Group observations by leaf

Suppose a tree has \(T\) leaves.

For leaf \(j\), define:

$$
I_j=\{i:q(x_i)=j\}
$$

That means:

> \(I_j\) contains all training samples that end up in leaf \(j\).

For all those samples:

$$
f_t(x_i)=w_j
$$

Therefore:

$$
\sum_{i\in I_j}
\left[
g_iw_j+
\frac12h_iw_j^2
\right]
$$

Since \(w_j\) is constant for that leaf:

$$
=
w_j\sum_{i\in I_j}g_i
+
\frac12w_j^2\sum_{i\in I_j}h_i
$$

Define:

$$
\boxed{
G_j=\sum_{i\in I_j}g_i
}
$$

and:

$$
\boxed{
H_j=\sum_{i\in I_j}h_i
}
$$

Then the leaf contribution becomes:

$$
G_jw_j+\frac12H_jw_j^2
$$

---

# 10. The simplified tree objective

Including regularization:

$$
\Omega(f_t)
=
\gamma T
+
\frac12\lambda\sum_jw_j^2
$$

Therefore:

$$
Obj^{(t)}
=
\sum_{j=1}^{T}
\left[
G_jw_j+
\frac12H_jw_j^2
\right]
+
\gamma T
+
\frac12\lambda\sum_jw_j^2
$$

Combine the \(w_j^2\) terms:

$$
\boxed{
Obj^{(t)}
=
\sum_{j=1}^{T}
\left[
G_jw_j
+
\frac12(H_j+\lambda)w_j^2
\right]
+
\gamma T
}
$$

This equation is extremely important.

---

# 11. Finding the optimal leaf weight

Now we want to find the value of \(w_j\) that minimizes:

$$
G_jw_j+
\frac12(H_j+\lambda)w_j^2
$$

Take derivative with respect to \(w_j\):

$$
\frac{\partial Obj}{\partial w_j}
=
G_j+(H_j+\lambda)w_j
$$

Set derivative equal to zero:

$$
G_j+(H_j+\lambda)w_j=0
$$

Therefore:

$$
(H_j+\lambda)w_j=-G_j
$$

and:

$$
\boxed{
w_j^*
=
-\frac{G_j}{H_j+\lambda}
}
$$

This is one of the most important equations in XGBoost.

---

# 12. What does this equation mean?

$$
w_j^*
=
-\frac{G_j}{H_j+\lambda}
$$

Think of it as:

```text
Leaf Weight
     =
   - Gradient Sum
   ───────────────
   Hessian Sum + Regularization
```

The numerator:

$$
G_j
$$

tells us the direction and magnitude of correction needed.

The denominator:

$$
H_j+\lambda
$$

controls how aggressively we make that correction.

Increasing \(\lambda\):

$$
\lambda\uparrow
$$

makes:

$$
|w_j^*|\downarrow
$$

So regularization shrinks leaf weights.

---

# 13. Optimal objective value

We can substitute:

$$
w_j^*=-\frac{G_j}{H_j+\lambda}
$$

into:

$$
G_jw_j+
\frac12(H_j+\lambda)w_j^2
$$

The resulting minimum is:

$$
\boxed{
-\frac12
\frac{G_j^2}{H_j+\lambda}
}
$$

Therefore, for the entire tree:

$$
\boxed{
Obj^*
=
-\frac12
\sum_{j=1}^{T}
\frac{G_j^2}{H_j+\lambda}
+
\gamma T
}
$$

This allows XGBoost to evaluate different possible tree structures.

---

# 14. Now comes the most important part: split finding

Suppose we currently have a leaf.

We are considering splitting it:

```text
                 Parent
                /      \
               /        \
          Left Leaf    Right Leaf
```

Before splitting:

$$
G=G_L+G_R
$$

$$
H=H_L+H_R
$$

---

# 15. Score before splitting

The parent leaf's score is:

$$
-\frac12
\frac{G^2}{H+\lambda}
$$

---

# 16. Score after splitting

The two new leaves have score:

$$
-\frac12
\frac{G_L^2}{H_L+\lambda}
$$

and:

$$
-\frac12
\frac{G_R^2}{H_R+\lambda}
$$

There is also a complexity penalty because we've added a leaf.

The split gain can therefore be written:

$$
\boxed{
Gain=
\frac12
\left(
\frac{G_L^2}{H_L+\lambda}
+
\frac{G_R^2}{H_R+\lambda}
-
\frac{G^2}{H+\lambda}
\right)
-\gamma
}
$$

This is the **XGBoost split gain formula**.

---

# 17. Why does XGBoost calculate split gain?

Suppose we have:

```text
Feature: Age
```

Potential splits:

```text
Age < 20
Age < 25
Age < 30
Age < 35
Age < 40
...
```

XGBoost evaluates possible splits and asks:

> Which split gives the greatest improvement in the objective?

Conceptually:

```text
Candidate split 1 → Gain = 2.4
Candidate split 2 → Gain = 5.7
Candidate split 3 → Gain = 1.9
Candidate split 4 → Gain = 8.2
```

The split with the highest useful gain becomes a candidate for the tree.

---

# 18. What does `gamma` do mathematically?

Recall:

$$
Gain=
\frac12(...)
-\gamma
$$

Therefore:

```text
Higher gamma
      ↓
Higher cost for creating leaves
      ↓
Need larger improvement
      ↓
Fewer splits
      ↓
Simpler trees
```

If:

$$
Gain\leq0
$$

then the split doesn't improve the regularized objective.

---

# 19. Numerical example

Let's make a simple example.

Suppose a parent node has:

$$
G=10
$$

$$
H=8
$$

and:

$$
\lambda=2
$$

Suppose we consider a split:

### Left child

$$
G_L=6
$$

$$
H_L=5
$$

### Right child

$$
G_R=4
$$

$$
H_R=3
$$

Notice:

$$
G_L+G_R=6+4=10
$$

and:

$$
H_L+H_R=5+3=8
$$

Good.

Assume:

$$
\gamma=0
$$

The gain is:

$$
Gain=
\frac12
\left[
\frac{6^2}{5+2}
+
\frac{4^2}{3+2}
-
\frac{10^2}{8+2}
\right]
$$

Calculate:

$$
=
\frac12
\left[
\frac{36}{7}
+
\frac{16}{5}
-
\frac{100}{10}
\right]
$$

Approximately:

$$
=
\frac12
(5.143+3.2-10)
$$

$$
=
\frac12(-1.657)
$$

$$
\approx-0.829
$$

The gain is negative.

Therefore:

> **This split isn't useful according to the regularized objective.**

---

# 20. A better split

Suppose instead:

$$
G_L=9,\quad H_L=5
$$

$$
G_R=1,\quad H_R=3
$$

Then:

$$
G=10,\quad H=8
$$

Gain:

$$
Gain=
\frac12
\left[
\frac{81}{7}
+
\frac{1}{5}
-
10
\right]
$$

$$
=
\frac12(11.571+0.2-10)
$$

$$
=
\frac12(1.771)
$$

$$
\approx0.886
$$

Now:

$$
Gain>0
$$

So this split improves the objective.

---

# 21. Where does the tree actually get its values?

For the left leaf:

$$
w_L
=
-\frac{G_L}{H_L+\lambda}
$$

Using:

$$
G_L=9,\quad H_L=5,\quad\lambda=2
$$

we get:

$$
w_L=-\frac9{7}
$$

$$
\boxed{w_L\approx-1.286}
$$

For the right leaf:

$$
w_R=-\frac1{3+2}
$$

$$
\boxed{w_R=-0.2}
$$

Those are the optimal leaf weights according to the simplified XGBoost objective.

---

# 22. Where does the "boosting" happen?

After building the tree, XGBoost updates predictions.

Without learning rate:

$$
\hat y_i^{(t)}
=
\hat y_i^{(t-1)}
+
f_t(x_i)
$$

With learning rate:

$$
\boxed{
\hat y_i^{(t)}
=
\hat y_i^{(t-1)}
+
\eta f_t(x_i)
}
$$

Suppose:

$$
\eta=0.1
$$

and the new tree gives:

$$
f_t(x_i)=-1.286
$$

Then its contribution is:

$$
0.1(-1.286)
$$

$$
=-0.1286
$$

So:

$$
\hat y_{\text{new}}
=
\hat y_{\text{old}}-0.1286
$$

Then XGBoost calculates new gradients and Hessians based on the updated predictions.

And the process repeats.

---

# 23. The complete mathematical training loop

Now put everything together.

### Initially:

$$
\hat y_i^{(0)}
$$

Then for each boosting round \(t\):

### Step 1 — Current prediction

$$
\hat y_i^{(t-1)}
$$

### Step 2 — Gradient

$$
\boxed{
g_i=
\frac{\partial l(y_i,\hat y_i)}
{\partial\hat y_i}
}
$$

### Step 3 — Hessian

$$
\boxed{
h_i=
\frac{\partial^2l(y_i,\hat y_i)}
{\partial\hat y_i^2}
}
$$

### Step 4 — Candidate tree structures

Consider possible feature splits.

### Step 5 — Aggregate gradients and Hessians

For each leaf:

$$
G_j=\sum_{i\in I_j}g_i
$$

$$
H_j=\sum_{i\in I_j}h_i
$$

### Step 6 — Calculate leaf weight

$$
\boxed{
w_j=-\frac{G_j}{H_j+\lambda}
}
$$

### Step 7 — Calculate split gain

$$
\boxed{
Gain=
\frac12
\left[
\frac{G_L^2}{H_L+\lambda}
+
\frac{G_R^2}{H_R+\lambda}
-
\frac{G^2}{H+\lambda}
\right]
-\gamma
}
$$

### Step 8 — Select useful splits

Build the tree.

### Step 9 — Update predictions

$$
\boxed{
\hat y_i^{(t)}
=
\hat y_i^{(t-1)}
+
\eta f_t(x_i)
}
$$

### Step 10 — Repeat

```text
New predictions
      ↓
New gradients
      ↓
New Hessians
      ↓
New tree
      ↓
New predictions
      ↓
...
```

---

# 24. Important example: squared-error regression

Let's connect this to a familiar loss.

For regression, suppose:

$$
l(y,\hat y)
=
\frac12(y-\hat y)^2
$$

Equivalent to:

$$
l(y,\hat y)
=
\frac12(\hat y-y)^2
$$

---

## Gradient

Differentiate:

$$
g=
\frac{\partial l}{\partial\hat y}
$$

Therefore:

$$
\boxed{
g=\hat y-y
}
$$

---

## Hessian

Differentiate again:

$$
h=
\frac{\partial^2l}{\partial\hat y^2}
$$

Therefore:

$$
\boxed{
h=1
}
$$

So for squared-error regression:

$$
g_i=\hat y_i-y_i
$$

$$
h_i=1
$$

This makes the XGBoost calculations particularly simple.

---

# 25. Connection to ordinary Gradient Boosting

This is a very important conceptual connection.

In ordinary gradient boosting, the next tree is related to the negative gradient:

$$
-r_i
$$

or, more generally:

$$
-\frac{\partial L}{\partial\hat y_i}
$$

XGBoost goes further.

Instead of only considering:

$$
g_i
$$

it considers:

$$
\boxed{
g_i+h_i
}
$$

in the sense that both first- and second-order information determine the optimization.

More precisely, the local objective is:

$$
g_if_t(x_i)
+
\frac12h_if_t(x_i)^2
$$

This second-order approximation is one of the defining mathematical features of XGBoost.

---

# 26. Classification: binary logistic loss

Now let's look at classification.

For binary classification, XGBoost commonly models a raw score:

$$
z_i
$$

and converts it into probability using the sigmoid function:

$$
\boxed{
p_i=\sigma(z_i)
=
\frac1{1+e^{-z_i}}
}
$$

The binary cross-entropy loss is:

$$
l(y_i,p_i)
=
-y_i\log p_i
-
(1-y_i)\log(1-p_i)
$$

When expressed with respect to the raw score \(z_i\), the derivatives become:

$$
\boxed{
g_i=p_i-y_i
}
$$

and:

$$
\boxed{
h_i=p_i(1-p_i)
}
$$

So for binary logistic classification:

```text
Gradient:
g = p - y

Hessian:
h = p(1-p)
```

This is a very important result.

---

# 27. Example of classification gradients

Suppose:

$$
y=1
$$

and the current model predicts:

$$
p=0.8
$$

Then:

$$
g=0.8-1
$$

$$
g=-0.2
$$

Hessian:

$$
h=0.8(1-0.8)
$$

$$
h=0.16
$$

Another sample:

$$
y=0
$$

and:

$$
p=0.8
$$

Then:

$$
g=0.8-0
$$

$$
g=0.8
$$

$$
h=0.16
$$

Notice:

* Correct confident prediction → relatively small gradient
* Wrong prediction → larger gradient

Therefore the next tree can focus on observations where the current model needs stronger correction.

---

# 28. Why does XGBoost use regularization?

Without regularization, boosting can keep adding increasingly complex trees.

For example:

```text
Tree 1
Tree 2
Tree 3
...
Tree 500
```

Eventually the model may start fitting noise.

XGBoost controls this using:

### L2 regularization

$$
\frac12\lambda\sum_jw_j^2
$$

### Leaf penalty

$$
\gamma T
$$

And practical tree-complexity controls such as:

* `max_depth`
* `min_child_weight`
* `gamma`

So the optimization becomes:

> Improve predictions **while keeping the tree sufficiently simple**.

---

# 29. L1 regularization

XGBoost can also use L1 regularization.

The general regularization can include:

$$
\Omega(f)
=
\gamma T
+
\lambda\frac12\sum_jw_j^2
+
\alpha\sum_j|w_j|
$$

where:

$$
\alpha=\text{L1 regularization}
$$

and:

$$
\lambda=\text{L2 regularization}
$$

Conceptually:

```text
L1
 ↓
encourages sparsity

L2
 ↓
shrinks weights
```

The exact optimal leaf-weight calculation becomes slightly different when L1 is included because the absolute-value term changes the optimization.

---

# 30. Why `min_child_weight` matters mathematically

Remember:

$$
H_j=\sum_{i\in I_j}h_i
$$

`min_child_weight` places a constraint on the amount of Hessian information that a child node must have before a split is accepted.

Conceptually:

```text
Small min_child_weight
       ↓
Small leaves allowed
       ↓
More complex trees
```

versus:

```text
Large min_child_weight
       ↓
Tiny/specialized leaves discouraged
       ↓
More conservative trees
```

For squared-error regression where \(h_i=1\), the sum of Hessians is closely related to the number of observations in the leaf.

For logistic classification, the Hessians depend on the predicted probabilities.

---

# 31. Why subsampling works

XGBoost can use only a fraction of observations for each boosting round.

Suppose:

$$
subsample=0.8
$$

Then approximately 80% of observations are used for that boosting round.

Similarly, feature subsampling can be used:

$$
colsample\_bytree=0.8
$$

This introduces randomness and can help reduce overfitting.

Conceptually:

```text
All training samples
       ↓
Random subset
       ↓
Build tree
```

rather than always using the entire dataset for every tree.

---

# 32. Why learning rate works

Suppose a new tree predicts:

$$
f_t(x)=10
$$

If:

$$
\eta=1
$$

then:

$$
\hat y_{new}=\hat y_{old}+10
$$

If:

$$
\eta=0.1
$$

then:

$$
\hat y_{new}=\hat y_{old}+1
$$

So:

$$
\boxed{
\eta\downarrow
\Rightarrow
\text{smaller updates}
}
$$

Usually, smaller learning rates require more boosting rounds.

---

# 33. Why XGBoost can overfit

Suppose:

```text
max_depth ↑
n_estimators ↑
learning_rate ↑
gamma ↓
min_child_weight ↓
```

The model becomes capable of making increasingly detailed corrections.

Eventually:

```text
Training error
      ↓
very low

Validation error
      ↑
may start increasing
```

That is overfitting.

XGBoost therefore gives you several ways to control complexity.

---

# 34. The role of each major parameter mathematically

| Parameter          | Mathematical/conceptual role             |
| ------------------ | ---------------------------------------- |
| `learning_rate`    | Scales contribution of each tree         |
| `n_estimators`     | Number of boosting rounds                |
| `max_depth`        | Controls tree depth                      |
| `gamma`            | Penalizes additional leaves/splits       |
| `reg_lambda`       | L2 penalty on leaf weights               |
| `reg_alpha`        | L1 penalty on leaf weights               |
| `min_child_weight` | Minimum Hessian-based child requirement  |
| `subsample`        | Fraction of rows used per boosting round |
| `colsample_bytree` | Fraction of features used per tree       |

---

# 35. The deepest intuition

You can understand XGBoost mathematically through four quantities:

### 1. Loss

$$
L
$$

Answers:

> How wrong is the current model?

### 2. Gradient

$$
g
$$

Answers:

> In which direction should the prediction move?

### 3. Hessian

$$
h
$$

Answers:

> What is the curvature/second-order behavior of the loss?

### 4. Regularization

$$
\Omega
$$

Answers:

> How complex should the tree be allowed to become?

Therefore:

$$
\boxed{
\text{XGBoost}
=
\text{Loss minimization}
+
\text{Gradient}
+
\text{Hessian}
+
\text{Tree optimization}
+
\text{Regularization}
}
$$

---

# 36. The entire mathematics in one chain

This is the chain I'd recommend memorizing:

```text
Current Model
     ↓
ŷᵢ
     ↓
Calculate Loss
     ↓
Calculate Gradient
gᵢ = ∂L/∂ŷᵢ
     ↓
Calculate Hessian
hᵢ = ∂²L/∂ŷᵢ²
     ↓
Group samples into candidate leaves
     ↓
Gⱼ = Σgᵢ
Hⱼ = Σhᵢ
     ↓
Optimal Leaf Weight
wⱼ* = -Gⱼ/(Hⱼ + λ)
     ↓
Evaluate Candidate Splits
     ↓
Gain =
½[G_L²/(H_L+λ)
 + G_R²/(H_R+λ)
 - G²/(H+λ)] - γ
     ↓
Choose useful splits
     ↓
Build Tree
     ↓
Update Prediction
ŷᵢ ← ŷᵢ + ηfₜ(xᵢ)
     ↓
Repeat
```

---

# 37. The five formulas you absolutely need to know

For your ML notes, these are the core equations:

### ① Model

$$
\boxed{
\hat y_i=\sum_{k=1}^{K}f_k(x_i)
}
$$

### ② Gradient

$$
\boxed{
g_i=
\frac{\partial l(y_i,\hat y_i)}
{\partial\hat y_i}
}
$$

### ③ Hessian

$$
\boxed{
h_i=
\frac{\partial^2l(y_i,\hat y_i)}
{\partial\hat y_i^2}
}
$$

### ④ Optimal leaf weight

$$
\boxed{
w_j^*=-\frac{G_j}{H_j+\lambda}
}
$$

### ⑤ Split gain

$$
\boxed{
Gain=
\frac12
\left[
\frac{G_L^2}{H_L+\lambda}
+
\frac{G_R^2}{H_R+\lambda}
-
\frac{G^2}{H+\lambda}
\right]
-\gamma
}
$$

where:

$$
G_j=\sum_{i\in I_j}g_i
$$

and:

$$
H_j=\sum_{i\in I_j}h_i
$$

---

## Final mental picture

The mathematics of XGBoost is essentially answering **three questions repeatedly**:

**1. How should the model change?**

$$
\rightarrow g_i,\ h_i
$$

**2. What should each new leaf predict?**

$$
\rightarrow
w_j^*=-\frac{G_j}{H_j+\lambda}
$$

**3. Is a particular split worth making?**

$$
\rightarrow \text{Split Gain}
$$

Then:

$$
\boxed{
\text{Find gradients/Hessians}
\rightarrow
\text{build optimal tree}
\rightarrow
\text{add small part of tree}
\rightarrow
\text{repeat}
}
$$

That is the mathematical engine behind XGBoost.
