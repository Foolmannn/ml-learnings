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
