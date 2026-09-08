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

# 30. Why Do We Need Kernels?

Suppose the data looks like this:

```text
       ● ● ●
    ●         ●

       × ×
      ×   ×
```

A straight line cannot separate the classes.

We need a nonlinear decision boundary.

One approach is to transform the features into a higher-dimensional space.

Suppose:

$$
x=(x_1,x_2)
$$

We could create:

$$
\phi(x)=
(x_1,x_2,x_1^2,x_2^2,x_1x_2)
$$

The data may become linearly separable in that higher-dimensional space.

But explicitly computing all these features can be expensive.

---

# 31. Kernel Trick

Instead of explicitly computing:

$$
\phi(x_i)
$$

we calculate:

$$
\boxed{
K(x_i,x_j)=\phi(x_i)^T\phi(x_j)
}
$$

This is the **kernel trick**.

It allows SVM to operate as though the data had been transformed into a higher-dimensional feature space without explicitly constructing that space.

---

# 32. Common Kernels

The most important kernels are:

### Linear

$$
\boxed{
K(x_i,x_j)=x_i^Tx_j
}
$$

### Polynomial

$$
\boxed{
K(x_i,x_j)
=
(\gamma x_i^Tx_j+r)^d
}
$$

### RBF / Gaussian ⭐

$$
\boxed{
K(x_i,x_j)
=
e^{-\gamma||x_i-x_j||^2}
}
$$

### Sigmoid

$$
\boxed{
K(x_i,x_j)
=
\tanh(\gamma x_i^Tx_j+r)
}
$$

---

# 33. Linear Kernel

The linear kernel is:

$$
K(x_i,x_j)=x_i^Tx_j
$$

Use it when the relationship is approximately linear.

In scikit-learn:

```python
SVC(kernel="linear")
```

For very large datasets, `LinearSVC` is often more appropriate than kernelized `SVC`.

---

# 34. Polynomial Kernel

Polynomial kernel:

$$
K(x_i,x_j)
=
(\gamma x_i^Tx_j+r)^d
$$

where:

- $\gamma$ controls scale
- $r$ is `coef0`
- $d$ is polynomial degree

Example:

```python
SVC(
    kernel="poly",
    degree=3
)
```

Higher degree allows more complex relationships but can increase overfitting.

---

# 35. RBF Kernel ⭐

The **Radial Basis Function (RBF)** kernel is probably the most important nonlinear SVM kernel in practical ML.

Formula:

$$
\boxed{
K(x_i,x_j)
=
\exp(-\gamma||x_i-x_j||^2)
}
$$

It measures how similar two points are based on their distance.

If two points are close:

$$
||x_i-x_j||^2\approx0
$$

then:

$$
K(x_i,x_j)\approx1
$$

If they are far apart:

$$
||x_i-x_j||^2\rightarrow\infty
$$

then:

$$
K(x_i,x_j)\rightarrow0
$$

---

# 36. Gamma

For RBF:

$$
K(x_i,x_j)
=
e^{-\gamma||x_i-x_j||^2}
$$

$\gamma$ controls how quickly the influence of a training point decreases with distance.

### Small gamma

Each point has a broad influence.

Decision boundary tends to be smoother.

```text
Low γ
→ broad influence
→ smooth boundary
→ possible underfitting
```

### Large gamma

Each point has a very local influence.

Decision boundary can become highly complex.

```text
High γ
→ local influence
→ complex boundary
→ possible overfitting
```

---

# 37. C and Gamma Together

This is one of the most important practical concepts.

For RBF SVM:

$$
\boxed{C+\gamma}
$$

are usually the two parameters you tune most carefully.

Think of them as:

### $C$

> How much do I care about training violations?

### $\gamma$

> How locally should each training point influence the boundary?

A rough conceptual grid:

| C | Gamma | Typical behavior |
|---|---|---|
| Low | Low | Very smooth / possibly underfit |
| High | Low | Larger-scale boundary, stronger fitting |
| Low | High | Local effects but errors tolerated |
| High | High | Very complex, overfitting risk |

---

# 38. Feature Scaling in SVM

Feature scaling is **very important** for SVM, especially with RBF and other distance-based kernels.

Suppose:

```text
Age       = 18–70
Salary    = 20,000–500,000
```

The salary feature has a much larger numerical scale.

For RBF:

$$
||x_i-x_j||^2
$$

is affected heavily by feature magnitude.

Therefore, standardize your features.

Typical approach:

```python
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

model = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(kernel="rbf"))
])
```

---

# 39. Why Pipeline Is Better

Instead of:

```python
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

you can use:

```python
Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC())
])
```

This helps prevent **data leakage** during cross-validation because scaling is fitted separately within each training fold.

---

# 40. SVM Classification in Scikit-Learn

Basic example:

```python
from sklearn.svm import SVC

model = SVC(
    kernel="rbf",
    C=1.0,
    gamma="scale"
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
```

---

# 41. Understanding `SVC`

Important parameters:

```python
SVC(
    C=1.0,
    kernel="rbf",
    degree=3,
    gamma="scale",
    coef0=0.0,
    probability=False,
    class_weight=None
)
```

### `C`

Controls regularization/error penalty.

### `kernel`

Controls the type of kernel.

Possible values include:

```text
linear
poly
rbf
sigmoid
```

### `degree`

Polynomial degree.

### `gamma`

Controls kernel influence.

### `coef0`

Independent term for polynomial/sigmoid kernels.

### `class_weight`

Useful for imbalanced classification.

### `probability`

Enables probability estimates.

---

# 42. Decision Function

SVM naturally produces a **decision score**.

```python
model.decision_function(X_test)
```

For a binary linear SVM:

$$
f(x)=w^Tx+b
$$

The sign determines the class.

For example:

```text
+2.8 → Class +1
+0.4 → Class +1
-0.2 → Class -1
-3.1 → Class -1
```

The magnitude indicates how far the point is from the decision boundary in the model's decision-function scale.

---

# 43. Probability Prediction

SVM doesn't inherently produce probabilities like logistic regression.

You can enable probability estimation:

```python
model = SVC(
    kernel="rbf",
    probability=True
)
```

Then:

```python
model.predict_proba(X_test)
```

This adds computational cost because probability calibration is performed.

So don't enable it unless you actually need probability estimates.

---

# 44. Multiclass SVM

SVM is fundamentally a binary classifier.

But real datasets may have:

```text
Class A
Class B
Class C
Class D
```

Scikit-learn's `SVC` handles multiclass classification using a **one-vs-one** strategy.

For $K$ classes:

$$
\frac{K(K-1)}{2}
$$

binary classifiers are trained.

For example, 4 classes:

$$
\frac{4(3)}2=6
$$

classifiers.

---

# 45. SVM for Regression — SVR

SVM can also perform regression.

This is called:

$$
\boxed{\text{Support Vector Regression (SVR)}}
$$

Instead of trying to separate classes, SVR tries to fit a function while allowing errors within an $\epsilon$-wide tube.

Conceptually:

```text
       upper ε boundary
  -------------------------
         prediction
  -------------------------
       lower ε boundary
```

Errors inside the tube are ignored.

---

# 46. Epsilon in SVR

The parameter:

$$
\boxed{\epsilon}
$$

controls the width of the tube.

If:

$$
|y_i-f(x_i)|\leq\epsilon
$$

the error doesn't contribute to the epsilon-insensitive loss.

In scikit-learn:

```python
from sklearn.svm import SVR

model = SVR(
    kernel="rbf",
    C=1.0,
    epsilon=0.1
)
```

---

# 47. Important SVR Hyperparameters

For SVR, understand:

```text
C
epsilon
kernel
gamma
degree
coef0
```

The most important ones for RBF SVR are usually:

$$
\boxed{C,\gamma,\epsilon}
$$

---

# 48. One-Class SVM

SVM can also be used for **anomaly detection**.

Instead of separating:

```text
Class A vs Class B
```

One-Class SVM learns the region containing normal observations.

Then unusual observations can be identified as anomalies.

Example applications:

- Fraud detection
- Network intrusion detection
- Equipment monitoring
- Unusual behavior detection

---

# 49. SVM vs Logistic Regression

This is an important comparison.

| SVM | Logistic Regression |
|---|---|
| Maximizes margin | Models class probability |
| Uses hinge loss | Uses log loss |
| Support vectors are important | All observations influence likelihood |
| Kernel trick allows nonlinear boundaries | Usually linear unless features are transformed |
| Excellent for medium-sized datasets | Excellent baseline and highly interpretable |
| Probability isn't native | Probability is natural |

Both can produce a linear decision boundary, but they optimize different objectives.

---

# 50. SVM vs KNN

### SVM

- Learns a decision boundary
- Training can be expensive
- Prediction can be relatively efficient
- Works well in high-dimensional spaces
- Requires scaling

### KNN

- Doesn't explicitly learn a global boundary
- Prediction can be expensive
- Highly dependent on distance
- Requires scaling
- Simple and intuitive

---

# 51. SVM vs Decision Tree

### SVM

- Margin-based
- Usually requires scaling
- Kernel can model nonlinear boundaries
- Strong mathematical optimization

### Decision Tree

- Rule-based splits
- Scaling generally unnecessary
- Easy to interpret
- Handles nonlinear relationships naturally

---

# 52. Advantages of SVM

SVM is particularly good when:

- Dataset is small/medium-sized
- Feature space is high-dimensional
- Classes have a meaningful separating boundary
- Nonlinear relationships exist
- You want strong classical ML performance

It is especially famous for applications involving:

- Text classification
- Image classification
- Bioinformatics
- Pattern recognition

---

# 53. Disadvantages of SVM

SVM can struggle when:

- Dataset has millions of observations
- Kernel computation becomes expensive
- Hyperparameter tuning is difficult
- Interpretability is important
- Dataset contains many noisy observations

Kernel SVMs can become computationally expensive as the number of training samples grows.

---

# 54. SVM Hyperparameter Tuning

For an RBF SVM, a common starting grid is:

```python
param_grid = {
    "svm__C": [0.1, 1, 10, 100],
    "svm__gamma": ["scale", 0.01, 0.1, 1]
}
```

Using a pipeline:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(kernel="rbf"))
])

grid = GridSearchCV(
    pipe,
    param_grid,
    cv=5,
    scoring="accuracy"
)

grid.fit(X_train, y_train)
```

Then:

```python
grid.best_params_
```

and:

```python
grid.best_score_
```

---

# 55. SVM Evaluation

For classification, don't rely only on accuracy.

Study:

### Confusion Matrix

$$
TP,\ TN,\ FP,\ FN
$$

### Precision

$$
\boxed{
Precision=\frac{TP}{TP+FP}
}
$$

### Recall

$$
\boxed{
Recall=\frac{TP}{TP+FN}
}
$$

### F1 Score

$$
\boxed{
F1=
2\frac{Precision\times Recall}
{Precision+Recall}
}
$$

Also understand:

- ROC-AUC
- Precision-Recall curve
- Cross-validation

---

# 56. Complete Practical SVM Workflow

A good real-world workflow is:

```text
                 Dataset
                    ↓
                 EDA
                    ↓
            Train/Test Split
                    ↓
             Feature Scaling
                    ↓
          ┌─────────┴─────────┐
          ↓                   ↓
      Linear SVM           RBF SVM
          ↓                   ↓
      Baseline          Tune C & γ
          │                   │
          └─────────┬─────────┘
                    ↓
             Cross Validation
                    ↓
                Evaluation
                    ↓
             Final Model
```

---

# 57. The Most Important Mathematical Chain

If you're preparing for ML seriously, memorize the **logic**, not just formulas:

### Step 1 — Decision boundary

$$
w^Tx+b=0
$$

↓

### Step 2 — Margin boundaries

$$
w^Tx+b=\pm1
$$

↓

### Step 3 — Margin

$$
\frac{2}{||w||}
$$

↓

### Step 4 — Maximize margin

$$
\max\frac{2}{||w||}
$$

↓

### Step 5 — Equivalent optimization

$$
\min\frac12||w||^2
$$

↓

### Step 6 — Add constraints

$$
y_i(w^Tx_i+b)\geq1
$$

↓

### Step 7 — Real-world data isn't perfectly separable

Introduce:

$$
\xi_i
$$

↓

### Step 8 — Soft-margin objective

$$
\min
\frac12||w||^2+C\sum_i\xi_i
$$

↓

### Step 9 — Hinge loss interpretation

$$
\max(0,1-y_if(x_i))
$$

↓

### Step 10 — Lagrangian/dual formulation

$$
w=\sum_i\alpha_i y_ix_i
$$

↓

### Step 11 — Support vectors

$$
\alpha_i>0
$$

↓

### Step 12 — Dot products appear

$$
x_i^Tx_j
$$

↓

### Step 13 — Kernel trick

$$
K(x_i,x_j)=\phi(x_i)^T\phi(x_j)
$$

↓

### Step 14 — Nonlinear SVM

Especially:

$$
K(x_i,x_j)
=
e^{-\gamma||x_i-x_j||^2}
$$

↓

### Step 15 — Tune

$$
\boxed{C,\gamma}
$$

This is the **core story of SVM**.

---

# 58. What You Should Be Able to Explain After Studying SVM

Before considering yourself comfortable with SVM, make sure you can answer these questions:

### Conceptual

- What is SVM?
- Why does SVM maximize the margin?
- What is a hyperplane?
- What is a support vector?
- Why are support vectors important?
- What is hard-margin SVM?
- Why do we need soft-margin SVM?
- What is a slack variable?

### Mathematical

- How do we derive the margin?
- Why is margin $2/||w||$?
- Why minimize $||w||^2$?
- What does $C$ represent?
- What is hinge loss?
- How does the primal formulation work?
- Why introduce Lagrange multipliers?
- What is the dual formulation?
- Why do only support vectors matter?

### Kernel

- Why do we need kernels?
- What is the kernel trick?
- What is feature mapping?
- How does the RBF kernel work?
- What does gamma do?
- How do $C$ and gamma interact?

### Implementation

- Why should SVM features be scaled?
- Difference between `SVC` and `LinearSVC`
- How to use `Pipeline`
- How to tune `C` and `gamma`
- How to use cross-validation
- How to evaluate SVM
- How to implement SVR

---

## 🧠 The one-sentence mental model

If you remember only one thing:

> **SVM finds a decision boundary that maximizes the distance to the closest training points, allows controlled violations using soft margins and $C$, and can create nonlinear boundaries using the kernel trick.**

For your ML learning sequence, I would next study **SVM mathematically from scratch**, especially the **derivation of hard-margin → soft-margin → hinge loss → Lagrangian → dual form → kernel trick**. That's the part that turns SVM from a memorized algorithm into something you actually understand.