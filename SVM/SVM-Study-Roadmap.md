# Support Vector Machine (SVM) — Detailed Notes

Support Vector Machine is one of the most important classical machine-learning algorithms, especially for **classification**, but it can also be used for **regression** and **anomaly detection**.

The best way to learn SVM is to connect the intuition with the mathematics:

> **Hyperplane → Margin → Support Vectors → Optimization → Soft Margin → Hinge Loss → Kernel Trick → SVC/SVR**

---

# 1. What is SVM?

**Support Vector Machine (SVM)** is a supervised machine-learning algorithm that finds a decision boundary that separates different classes while trying to **maximize the margin** between the classes.

For a binary classification problem:

$$
y_i \in \{-1,+1\}
$$

SVM tries to find:

$$
\boxed{w^Tx+b=0}
$$

such that the two classes are separated as well as possible.

The most important idea is:

> **SVM doesn't simply look for a separating line. It looks for the separating boundary with the maximum possible margin.**

---

# 2. Simple Intuition

Suppose we have two classes:

```text
Class +                 Class -

   ● ● ●
     ● ●

--------------------     ← possible boundary

                 × ×
               × × ×
```

There can be many boundaries that separate the two classes.

SVM asks:

> Which boundary gives us the safest separation?

It chooses the boundary that leaves the **largest possible gap** between the classes.

That gap is called the **margin**.

---

# 3. Why Maximum Margin?

Suppose we have two possible decision boundaries.

### Boundary A

```text
● ● ● | × × ×
      |
```

Very close to some training points.

### Boundary B

```text
● ● ●       |       × × ×
            |
```

There is more space between the boundary and the closest points.

SVM prefers **Boundary B**.

Why?

Because a larger margin generally means the model is less sensitive to small changes/noise in the data and can generalize better to unseen examples.

---

# 4. Hyperplane

The decision boundary of SVM is called a **hyperplane**.

For two features:

$$
w_1x_1+w_2x_2+b=0
$$

For multiple features:

$$
\boxed{w^Tx+b=0}
$$

where:

- $w$ = weight vector
- $x$ = input feature vector
- $b$ = bias/intercept

For example:

$$
2x_1+3x_2-6=0
$$

is a line in 2D.

In 3D:

$$
w_1x_1+w_2x_2+w_3x_3+b=0
$$

is a plane.

In higher dimensions, it is called a hyperplane.

---

# 5. Decision Function

The SVM decision function is:

$$
f(x)=w^Tx+b
$$

For binary classification:

$$
\hat y =
\begin{cases}
+1 & f(x)>0\\
-1 & f(x)<0
\end{cases}
$$

Therefore:

$$
\boxed{\hat y=\operatorname{sign}(w^Tx+b)}
$$

The decision boundary occurs when:

$$
w^Tx+b=0
$$

---

# 6. Understanding the Weight Vector

Consider:

$$
w^Tx+b=0
$$

The vector $w$ is perpendicular to the decision boundary.

For example:

$$
2x_1+3x_2-6=0
$$

Here:

$$
w=
\begin{bmatrix}
2\\
3
\end{bmatrix}
$$

The direction of $w$ tells us the orientation of the boundary.

This becomes important when calculating the distance from a point to the hyperplane.

---

# 7. Distance Between a Point and Hyperplane

Suppose:

$$
w^Tx+b=0
$$

For a point $x$, the perpendicular distance from the point to the hyperplane is:

$$
\boxed{
\frac{|w^Tx+b|}{||w||}
}
$$

where:

$$
||w||=\sqrt{w_1^2+w_2^2+\cdots+w_n^2}
$$

This formula is extremely important in SVM.

---

# 8. Functional Margin

For a training example:

$$
(x_i,y_i)
$$

where:

$$
y_i\in\{-1,+1\}
$$

define:

$$
f(x_i)=w^Tx_i+b
$$

The **functional margin** is:

$$
\boxed{
y_i(w^Tx_i+b)
}
$$

Why multiply by $y_i$?

Because it allows both classes to be treated using the same expression.

### Correctly classified positive point

$$
y_i=+1
$$

and:

$$
w^Tx_i+b>0
$$

Therefore:

$$
y_i(w^Tx_i+b)>0
$$

### Correctly classified negative point

$$
y_i=-1
$$

and:

$$
w^Tx_i+b<0
$$

Again:

$$
y_i(w^Tx_i+b)>0
$$

So:

$$
\boxed{y_i(w^Tx_i+b)>0}
$$

means correctly classified.

---

# 9. Geometric Margin

Functional margin depends on the scale of $w$ and $b$.

The **geometric margin** is:

$$
\boxed{
\frac{y_i(w^Tx_i+b)}{||w||}
}
$$

The distance of the closest training point from the decision boundary determines the margin.

SVM wants to maximize this geometric margin.

---

# 10. The Three Important Hyperplanes

SVM uses three parallel hyperplanes:

### Decision boundary

$$
\boxed{w^Tx+b=0}
$$

### Positive margin boundary

$$
\boxed{w^Tx+b=1}
$$

### Negative margin boundary

$$
\boxed{w^Tx+b=-1}
$$

Visually:

```text
Class -        Margin       Decision       Margin       Class +

   × × ×
   × ×
      \          -1             0              +1
       \           |             |              |
        \          |             |              |
---------\---------|-------------|--------------|---------
          \        |             |              |
           \       |             |              |
             ● ● ●
```

The closest points lie on the margin boundaries.

Those points are the **support vectors**.

---

# 11. Deriving the Margin

Consider:

$$
w^Tx+b=1
$$

and:

$$
w^Tx+b=-1
$$

The distance between two parallel hyperplanes:

$$
w^Tx+b=c_1
$$

and

$$
w^Tx+b=c_2
$$

is:

$$
\frac{|c_1-c_2|}{||w||}
$$

Therefore:

$$
\text{margin}
=
\frac{|1-(-1)|}{||w||}
$$

$$
=\frac{2}{||w||}
$$

So:

$$
\boxed{
\text{Margin}=\frac{2}{||w||}
}
$$

Therefore maximizing margin means:

$$
\max\frac{2}{||w||}
$$

which is equivalent to:

$$
\boxed{\min ||w||}
$$

and, for mathematical convenience:

$$
\boxed{
\min\frac12||w||^2
}
$$

---

# 12. Support Vectors

The training points closest to the decision boundary are called **support vectors**.

They typically satisfy:

$$
\boxed{
y_i(w^Tx_i+b)=1
}
$$

for the hard-margin case.

These points are extremely important because they determine the optimal boundary.

Imagine:

```text
● ● ● ●        × × × ×

       ●       ×
         \     /
          \   /
           \ /
            |
            |
```

The points nearest the boundary are the support vectors.

---

# 13. Why Are They Called Support Vectors?

Think of the decision boundary as being "supported" by the closest points.

If you move the support vectors, the optimal boundary can change.

But if you move a point far away from the boundary, the optimal boundary may not change at all.

That's why SVM focuses on the critical boundary points.

---

# 14. Hard-Margin SVM

Hard-margin SVM assumes that the data is:

> **Perfectly linearly separable.**

We want:

$$
y_i(w^Tx_i+b)\geq1
$$

for every training point.

The optimization problem becomes:

$$
\boxed{
\min_{w,b}\frac12||w||^2
}
$$

subject to:

$$
\boxed{
y_i(w^Tx_i+b)\geq1
}
$$

This is the fundamental mathematical formulation of hard-margin SVM.

---

# 15. Why the Constraint is $ \geq 1 $

You might ask:

> Why not simply use $y_i(w^Tx_i+b)>0$?

Because $w$ and $b$ can be multiplied by any positive constant without changing the decision boundary.

For example:

$$
w^Tx+b=0
$$

and:

$$
10w^Tx+10b=0
$$

represent the same boundary.

So SVM fixes the scale using:

$$
y_i(w^Tx_i+b)\geq1
$$

This gives us standardized margin boundaries:

$$
+1,\quad 0,\quad -1
$$

---

# 16. Problem With Hard-Margin SVM

Real-world data is rarely perfectly separable.

Example:

```text
● ● ●
  ●
     ×
       ×
    ●
        × ×
```

There may be:

- noise
- outliers
- overlapping classes
- mislabeled observations

A hard-margin SVM cannot tolerate these points.

So we introduce **soft-margin SVM**.

---

# 17. Soft-Margin SVM

Soft-margin SVM allows some observations to violate the margin.

We introduce a new variable:

$$
\boxed{\xi_i}
$$

called the **slack variable**.

The optimization problem becomes:

$$
\boxed{
\min_{w,b,\xi}
\frac12||w||^2+
C\sum_i\xi_i
}
$$

subject to:

$$
\boxed{
y_i(w^Tx_i+b)\geq1-\xi_i
}
$$

and:

$$
\xi_i\geq0
$$

---

# 18. Understanding Slack Variable

The value of $\xi_i$ tells us how much a point violates the margin.

### Case 1

$$
\xi_i=0
$$

The point satisfies the margin.

---

### Case 2

$$
0<\xi_i<1
$$

The point is correctly classified but lies inside the margin.

---

### Case 3

$$
\xi_i=1
$$

The point lies on the decision boundary.

---

### Case 4

$$
\xi_i>1
$$

The point is misclassified.

This is a very useful way to understand soft-margin SVM.

---

# 19. The C Parameter

Now we introduce one of the most important SVM hyperparameters:

$$
\boxed{C}
$$

The objective is:

$$
\frac12||w||^2+C\sum_i\xi_i
$$

There are two competing goals:

### Goal 1

Minimize:

$$
\frac12||w||^2
$$

→ maximize margin.

### Goal 2

Minimize:

$$
\sum_i\xi_i
$$

→ reduce margin violations.

$C$ controls the trade-off.

---

# 20. Large C

Suppose:

$$
C=1000
$$

Violations become very expensive.

The model tries hard to classify training examples correctly.

Usually:

- fewer training errors
- narrower margin
- more complex boundary
- higher risk of overfitting

Conceptually:

```text
Large C
   ↓
Error is expensive
   ↓
Fit training data aggressively
   ↓
Potential overfitting
```

---

# 21. Small C

Suppose:

$$
C=0.01
$$

Violations are less expensive.

The model is more willing to tolerate misclassified or margin-violating points in exchange for a larger margin.

Usually:

- wider margin
- stronger regularization
- smoother model
- potentially more training errors
- potential underfitting

Conceptually:

```text
Small C
   ↓
Errors are less expensive
   ↓
Prefer wider margin
   ↓
More regularization
```

---

# 22. Hinge Loss

Soft-margin SVM can also be understood through **hinge loss**.

For a point:

$$
(x_i,y_i)
$$

define:

$$
f(x_i)=w^Tx_i+b
$$

The hinge loss is:

$$
\boxed{
L_i=\max(0,1-y_if(x_i))
}
$$

or:

$$
\boxed{
L_i=\max(0,1-y_i(w^Tx_i+b))
}
$$

---

# 23. Understanding Hinge Loss

Let's define:

$$
z=y_if(x_i)
$$

Then:

$$
L=\max(0,1-z)
$$

### Case 1: $z>1$

$$
L=0
$$

The point is correctly classified and outside the margin.

---

### Case 2: $0<z<1$

$$
L=1-z
$$

The point is correctly classified but inside the margin.

---

### Case 3: $z<0$

The point is misclassified.

Loss becomes:

$$
1-z>1
$$

So misclassified points are heavily penalized.

---

# 24. SVM Objective Using Hinge Loss

The primal SVM objective can be written conceptually as:

$$
\boxed{
\frac12||w||^2+
C\sum_i
\max(0,1-y_i(w^Tx_i+b))
}
$$

This equation is extremely important.

It combines:

### Regularization

$$
\frac12||w||^2
$$

with:

### Hinge loss

$$
\sum_i\max(0,1-y_if(x_i))
$$

So SVM is balancing:

> **Large margin vs classification/margin violations.**

---

# 25. SVM and Regularization

There is an important connection:

$$
\boxed{C \text{ controls regularization strength}}
$$

But be careful with the terminology.

In `sklearn`, **larger `C` means weaker regularization**, because the model puts greater emphasis on minimizing training violations.

Therefore:

```text
C ↑
→ regularization ↓
→ training fit ↑
→ overfitting risk ↑
```

and:

```text
C ↓
→ regularization ↑
→ training fit ↓
→ smoother model
```

---

# 26. From Primal to Dual

Now we enter the mathematical heart of SVM.

The primal problem is:

$$
\min_{w,b}\frac12||w||^2
$$

subject to:

$$
y_i(w^Tx_i+b)\geq1
$$

We introduce **Lagrange multipliers**:

$$
\alpha_i\geq0
$$

The Lagrangian becomes:

$$
L(w,b,\alpha)
=
\frac12||w||^2
-
\sum_i\alpha_i
[y_i(w^Tx_i+b)-1]
$$

We then minimize with respect to $w$ and $b$.

---

# 27. Derivative With Respect to $w$

Take:

$$
\frac{\partial L}{\partial w}=0
$$

This gives:

$$
w-\sum_i\alpha_i y_i x_i=0
$$

Therefore:

$$
\boxed{
w=\sum_i\alpha_i y_i x_i
}
$$

This is a major result.

It tells us:

> The optimal weight vector is a weighted combination of training points.

But there is something even more interesting.

For most points:

$$
\alpha_i=0
$$

Only important points near the boundary have non-zero $\alpha_i$.

Those are the **support vectors**.

---

# 28. Why Support Vectors Matter Mathematically

We have:

$$
w=\sum_i\alpha_i y_i x_i
$$

If:

$$
\alpha_i=0
$$

then that point contributes nothing to $w$.

If:

$$
\alpha_i>0
$$

then that point contributes to the decision boundary.

Therefore:

$$
\boxed{
\alpha_i>0 \Rightarrow \text{support vector}
}
$$

This connects the intuition of support vectors directly to the mathematics.

---

# 29. Dual Formulation

After eliminating $w$ and $b$, the hard-margin dual problem becomes:

$$
\boxed{
\max_{\alpha}
\sum_i\alpha_i
-
\frac12
\sum_i\sum_j
\alpha_i\alpha_jy_iy_jx_i^Tx_j
}
$$

subject to:

$$
\alpha_i\geq0
$$

and:

$$
\boxed{
\sum_i\alpha_i y_i=0
}
$$

Notice something important:

The data appears through:

$$
x_i^Tx_j
$$

—the **dot product between data points**.

This leads to one of the most powerful ideas in SVM:

# The Kernel Trick

---
