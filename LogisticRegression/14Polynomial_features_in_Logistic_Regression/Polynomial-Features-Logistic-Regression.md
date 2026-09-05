
# Polynomial Features in Logistic Regression
## Non-Linear Logistic Regression — Detailed Explanation

A standard **Logistic Regression** model creates a **linear decision boundary**. But many real-world classification problems are not linearly separable.

For example, suppose we want to classify points as:

- 🟢 Class 0: inside a circle
- 🔴 Class 1: outside a circle

A straight line cannot separate them effectively.

The key idea is:

> **Keep Logistic Regression, but transform the input features into polynomial features.**

This gives us a **non-linear decision boundary** while the model is still **linear in its parameters**.

---

# 1. First: What does ordinary Logistic Regression do?

Suppose we have two features:

$$
x_1,\;x_2
$$

Logistic Regression first calculates:

$$
z = w_1x_1+w_2x_2+b
$$

Then applies the sigmoid:

$$
\hat y = \sigma(z)
$$

where

$$
\sigma(z)=\frac{1}{1+e^{-z}}
$$

Therefore:

$$
P(y=1|x)=
\frac{1}{1+e^{-(w_1x_1+w_2x_2+b)}}
$$

The classification rule is usually:

$$
\hat y =
\begin{cases}
1 & P(y=1|x)\geq0.5\\
0 & P(y=1|x)<0.5
\end{cases}
$$

---

# 2. Why is the decision boundary linear?

The decision boundary occurs when:

$$
P(y=1|x)=0.5
$$

Since:

$$
\sigma(z)=0.5
$$

when:

$$
z=0
$$

we get:

$$
w_1x_1+w_2x_2+b=0
$$

This is the equation of a **straight line** in two dimensions.

For example:

$$
2x_1+3x_2-6=0
$$

is a straight line.



So ordinary Logistic Regression can only learn a linear boundary in the original feature space.

---

# 3. What if the data is non-linear?

Consider this type of classification:

```text
        Class 0

       ○ ○ ○
     ○       ○
    ○    🔴   ○
     ○       ○
       ○ ○ ○

        Class 1
```

Suppose:

$$
x_1^2+x_2^2 < 4
$$

represents Class 1, while points outside represent Class 0.

The required decision boundary is:

$$
x_1^2+x_2^2=4
$$

which is a **circle**, not a line.

Ordinary Logistic Regression cannot directly learn this boundary because it only has:

$$
w_1x_1+w_2x_2+b
$$

---

# 4. The solution: Polynomial Features

We can create additional features from the original features.

Suppose we have:

$$
x_1,x_2
$$

We can create:

$$
x_1^2
$$

$$
x_2^2
$$

$$
x_1x_2
$$

Now our feature vector becomes:

$$
X'=
[x_1,x_2,x_1^2,x_2^2,x_1x_2]
$$

Logistic Regression can then learn:

$$
z=
w_1x_1+
w_2x_2+
w_3x_1^2+
w_4x_2^2+
w_5x_1x_2+b
$$

and:

$$
P(y=1|x)=\sigma(z)
$$

Notice something important:

### The model is still linear in the weights.

It is linear with respect to:

$$
w_1,w_2,w_3,w_4,w_5,b
$$

but **non-linear with respect to the original input features**.

This distinction is extremely important.

---

# 5. Example: Learning a circular boundary

Suppose our model learns:

$$
z=x_1^2+x_2^2-4
$$

Then:

$$
P(y=1)=
\sigma(x_1^2+x_2^2-4)
$$

The decision boundary is:

$$
x_1^2+x_2^2-4=0
$$

Therefore:

$$
x_1^2+x_2^2=4
$$

That's a circle.



So Logistic Regression has learned a **non-linear decision boundary**.

---

# 6. Why do we call this "Non-Linear Logistic Regression"?

This terminology can be slightly confusing.

We are still using:

$$
\hat y=\sigma(z)
$$

and the model is still linear in its coefficients.

But:

$$
z=w_1x_1+w_2x_2+w_3x_1^2+w_4x_2^2+w_5x_1x_2+b
$$

is nonlinear in the original $x$'s.

Therefore, the **decision boundary in the original feature space is nonlinear**.

So:

> Polynomial Logistic Regression = Logistic Regression + nonlinear feature transformation.

---

# 7. Polynomial Feature Expansion

Let's understand exactly what `PolynomialFeatures` does.

Suppose:

$$
X=[x_1,x_2]
$$

### Degree 1

We get:

$$
1,x_1,x_2
$$

Usually the constant $1$ is represented by the intercept separately.

---

### Degree 2

We get:

$$
1,x_1,x_2,x_1^2,x_1x_2,x_2^2
$$

So:

$$
X'=
[x_1,x_2,x_1^2,x_1x_2,x_2^2]
$$

---

### Degree 3

We get:

$$
1,
x_1,
x_2,
x_1^2,
x_1x_2,
x_2^2,
x_1^3,
x_1^2x_2,
x_1x_2^2,
x_2^3
$$

Now Logistic Regression can learn:

$$
z=
w_1x_1+
w_2x_2+
w_3x_1^2+
w_4x_1x_2+
w_5x_2^2+
w_6x_1^3+
w_7x_1^2x_2+
w_8x_1x_2^2+
w_9x_2^3+b
$$

This can produce much more complicated decision boundaries.

---

# 8. Degree controls the complexity

This is one of the most important concepts.

### Degree 1

$$
x_1,x_2
$$

→ Linear boundary.

### Degree 2

$$
x_1,x_2,x_1^2,x_1x_2,x_2^2
$$

→ Quadratic/nonlinear boundary.

### Degree 3

Adds cubic terms.

→ More flexible boundary.

### Degree 4+

Even more flexibility.

But higher degree isn't automatically better.

---

# 9. Mathematical form

Suppose polynomial transformation produces:

$$
\phi(x)
$$

Then Logistic Regression becomes:

$$
P(y=1|x)=\sigma(w^T\phi(x)+b)
$$

where:

$$
\phi(x)
$$

is the polynomial feature transformation.

For example:

$$
\phi(x_1,x_2)
=
\begin{bmatrix}
x_1\\
x_2\\
x_1^2\\
x_1x_2\\
x_2^2
\end{bmatrix}
$$

Then:

$$
z=w^T\phi(x)+b
$$

and:

$$
P(y=1|x)=\frac{1}{1+e^{-w^T\phi(x)-b}}
$$

The boundary is:

$$
w^T\phi(x)+b=0
$$

which is nonlinear in $x$.

---

# 10. Example with actual coefficients

Suppose Logistic Regression learns:

$$
z=
2x_1+3x_2
-1.5x_1^2
-2x_2^2
+0.5x_1x_2
-1
$$

The probability is:

$$
P(y=1)=
\sigma(
2x_1+3x_2
-1.5x_1^2
-2x_2^2
+0.5x_1x_2
-1
)
$$

The decision boundary occurs when:

$$
z=0
$$

Therefore:

$$
2x_1+3x_2
-1.5x_1^2
-2x_2^2
+0.5x_1x_2
-1=0
$$

That's no longer a straight line.

---

# 11. Important distinction: nonlinear features vs nonlinear model

This is a common interview/exam question.

Consider:

$$
y=w_1x+w_2x^2+b
$$

Is the model nonlinear?

### With respect to $x$:

Yes.

Because it contains:

$$
x^2
$$

### With respect to parameters:

No.

It is linear in:

$$
w_1,w_2,b
$$

That's why ordinary optimization techniques for Logistic Regression still work.

We simply changed the feature representation.

---

# 12. Implementation with Scikit-Learn

Suppose we have:

```python
X
y
```

We can use:

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
```

Create polynomial features:

```python
poly = PolynomialFeatures(degree=2)

X_poly = poly.fit_transform(X)
```

Then train Logistic Regression:

```python
model = LogisticRegression()

model.fit(X_poly, y)
```

Prediction:

```python
X_test_poly = poly.transform(X_test)

y_pred = model.predict(X_test_poly)
```

Probability:

```python
y_prob = model.predict_proba(X_test_poly)
```

---

# 13. Better approach: Pipeline

In practice, use a pipeline.

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression

model = Pipeline([
    ("poly", PolynomialFeatures(degree=2)),
    ("logistic", LogisticRegression())
])

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
```

This is better because the transformation and model are treated as one object.

---

# 14. Add Standardization

Polynomial features can have very different scales.

For example:

$$
x=100
$$

gives:

$$
x^2=10000
$$

and:

$$
x^3=1,000,000
$$

This can cause optimization problems.

Therefore, a common pipeline is:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LogisticRegression

model = Pipeline([
    ("poly", PolynomialFeatures(degree=2)),
    ("scaler", StandardScaler()),
    ("logistic", LogisticRegression())
])

model.fit(X_train, y_train)
```

The workflow becomes:

$$
X
\rightarrow
\text{Polynomial Features}
\rightarrow
\text{Scaling}
\rightarrow
\text{Logistic Regression}
$$

---

# 15. Why polynomial features can solve non-linear problems

Imagine the original feature space:

$$
(x_1,x_2)
$$

Maybe the classes cannot be separated by a line.

But after transformation:

$$
(x_1,x_2)
\rightarrow
(x_1,x_2,x_1^2,x_1x_2,x_2^2)
$$

the same data is represented in a higher-dimensional space.

A **linear hyperplane** can now separate the transformed data.

This is the key idea:

> **Nonlinear transformation of the features + linear classifier = nonlinear decision boundary in the original space.**

---

# 16. Feature-space perspective

This is extremely important for understanding ML.

Original space:

$$
(x_1,x_2)
$$

Suppose the boundary cannot be represented as:

$$
w_1x_1+w_2x_2+b=0
$$

We transform:

$$
\phi(x_1,x_2)
=
(x_1,x_2,x_1^2,x_1x_2,x_2^2)
$$

Then Logistic Regression learns:

$$
w^T\phi(x)+b=0
$$

which is linear in the transformed feature space.

But when mapped back to the original space, the boundary becomes nonlinear.

---

# 17. Polynomial Logistic Regression vs Linear Logistic Regression

| Property | Linear Logistic Regression | Polynomial Logistic Regression |
|---|---|---|
| Original features | $x_1,x_2$ | $x_1,x_2,\dots$ |
| Feature transformation | None | Polynomial |
| Decision boundary | Linear | Nonlinear |
| Sigmoid | Yes | Yes |
| Cross-entropy loss | Yes | Yes |
| Gradient descent | Yes | Yes |
| Parameters | Linear | Linear |
| Can model curves | ❌ | ✅ |
| Overfitting risk | Lower | Higher |

---

# 18. Polynomial degree and overfitting

This is where things become important in real ML.

Suppose:

```text
Degree 1
```

The model is too simple.

It may **underfit**.

```text
Degree 2
```

Can capture moderate nonlinear patterns.

```text
Degree 10
```

The model can become extremely flexible.

It may start fitting:

- noise
- outliers
- random fluctuations

This is **overfitting**.

Conceptually:

$$
\text{Degree} \uparrow
\Rightarrow
\text{Model Complexity} \uparrow
$$

and often:

$$
\text{Overfitting Risk} \uparrow
$$

---

# 19. Number of features grows rapidly

Suppose we have:

$$
n
$$

original features and polynomial degree:

$$
d
$$

The number of polynomial features grows approximately as:

$$
\binom{n+d}{d}
$$

when including interaction terms and the bias term.

For example, with:

$$
n=10
$$

and:

$$
d=5
$$

we get:

$$
\binom{15}{5}=3003
$$

features.

That's a huge increase from only 10 original features.

This is one reason high-degree polynomial expansion can become computationally expensive.

---

# 20. Interaction terms

Polynomial features don't only create powers.

They also create interactions.

For two features:

$$
x_1,x_2
$$

degree 2 gives:

$$
x_1^2,\quad x_1x_2,\quad x_2^2
$$

The term:

$$
x_1x_2
$$

is an **interaction feature**.

It means:

> The effect of $x_1$ may depend on the value of $x_2$.

For example, perhaps:

$$
\text{risk}
$$

depends not only on temperature and humidity individually, but also on their interaction:

$$
temperature \times humidity
$$

---

# 21. Polynomial features and regularization

Polynomial expansion can create many features.

Therefore regularization becomes particularly important.

Logistic Regression commonly uses:

### L2 regularization

$$
J(w)
=
-\frac{1}{m}
\sum_{i=1}^{m}
[
y_i\log(\hat y_i)
+
(1-y_i)\log(1-\hat y_i)
]
+
\lambda\sum_jw_j^2
$$

L2 discourages very large coefficients.

---

### L1 regularization

$$
J(w)
=
Loss+
\lambda\sum_j|w_j|
$$

L1 can force some coefficients to exactly zero.

Therefore polynomial Logistic Regression often benefits from regularization.

---

# 22. Complete Scikit-Learn example

```python
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = Pipeline([
    ("poly", PolynomialFeatures(degree=2)),
    ("scaler", StandardScaler()),
    ("logistic", LogisticRegression(
        C=1.0,
        max_iter=1000
    ))
])

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)
```

---

# 23. What does `C` mean?

In Scikit-Learn Logistic Regression:

$$
C=\frac{1}{\lambda}
$$

approximately, depending on the formulation.

So:

### Small $C$

Strong regularization.

$$
C\downarrow
\Rightarrow
\text{Regularization}\uparrow
$$

### Large $C$

Weak regularization.

$$
C\uparrow
\Rightarrow
\text{Regularization}\downarrow
$$

With polynomial features, tuning `C` is particularly important.

---

# 24. Choosing the polynomial degree

Don't simply choose degree 10 because it produces a better training accuracy.

Instead:

```python
from sklearn.model_selection import GridSearchCV

params = {
    "poly__degree": [1, 2, 3, 4],
    "logistic__C": [0.01, 0.1, 1, 10, 100]
}

grid = GridSearchCV(
    model,
    params,
    cv=5,
    scoring="accuracy"
)

grid.fit(X_train, y_train)

print(grid.best_params_)
```

Now you're selecting:

- polynomial degree
- regularization strength

using cross-validation.

---

# 25. Binary vs Multiclass

Polynomial features work with both.

### Binary classification

$$
y\in\{0,1\}
$$

Use:

```python
LogisticRegression()
```

### Multiclass classification

For:

$$
y\in\{0,1,2,\dots,K-1\}
$$

Logistic Regression can use multinomial classification internally.

Polynomial features can still be applied first:

$$
X
\rightarrow
PolynomialFeatures
\rightarrow
LogisticRegression
$$

The polynomial transformation is independent of whether the final classifier is binary or multiclass.

---

# 26. Polynomial Logistic Regression vs Polynomial Regression

Don't confuse these.

### Polynomial Regression

Predicts a continuous value:

$$
y=w_0+w_1x+w_2x^2
$$

Output:

$$
y\in\mathbb R
$$

Example:

$$
house\ price
$$

---

### Polynomial Logistic Regression

Predicts probability:

$$
P(y=1|x)
=
\sigma(w_0+w_1x+w_2x^2)
$$

Output:

$$
0\leq P\leq1
$$

Example:

$$
spam/not\ spam
$$

So polynomial features don't change the **classification nature** of Logistic Regression.

---

# 27. The complete conceptual pipeline

The most useful way to remember it is:

$$
\boxed{
X
\rightarrow
\phi(X)
\rightarrow
\text{Logistic Regression}
\rightarrow
P(y=1)
\rightarrow
\text{Threshold}
\rightarrow
\hat y
}
$$

For example:

$$
(x_1,x_2)
$$

becomes:

$$
(x_1,x_2,x_1^2,x_1x_2,x_2^2)
$$

then:

$$
z=w^T\phi(X)+b
$$

then:

$$
P(y=1)=\sigma(z)
$$

then:

$$
\hat y=
\begin{cases}
1 & P\geq0.5\\
0 & P<0.5
\end{cases}
$$

---

# 28. The most important insight

There is a subtle but fundamental point:

### Logistic Regression itself hasn't become fundamentally nonlinear.

We're doing:

$$
\boxed{\text{Nonlinear Feature Transformation}+\text{Linear Logistic Model}}
$$

The model is:

$$
w^T\phi(X)+b
$$

linear in the transformed features.

But because:

$$
\phi(X)
$$

contains things such as:

$$
x^2,\quad x^3,\quad x_1x_2
$$

the resulting boundary in the **original feature space** is nonlinear.

---

## Final mental model

Think of it this way:

```text
Original data
     │
     ▼
┌──────────────────────┐
│ Polynomial Expansion  │
│                      │
│ x₁, x₂               │
│ x₁², x₂²             │
│ x₁x₂                 │
│ x₁³, ...             │
└──────────┬───────────┘
           │
           ▼
   Higher-dimensional
      feature space
           │
           ▼
┌──────────────────────┐
│ Logistic Regression  │
│                      │
│ z = wᵀX' + b         │
└──────────┬───────────┘
           │
           ▼
       Sigmoid
           │
           ▼
      Probability
           │
           ▼
      Classification
```

### In one sentence:

> **Polynomial Logistic Regression transforms the original features into polynomial and interaction features, then applies ordinary Logistic Regression to learn a nonlinear decision boundary in the original feature space.**

This is a very useful bridge between **linear models and nonlinear ML models**, and the next natural topic is understanding **how the gradient descent and decision-boundary equation change after polynomial expansion**.