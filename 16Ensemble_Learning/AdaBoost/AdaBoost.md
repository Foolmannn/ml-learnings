# AdaBoost (Adaptive Boosting) in Machine Learning 

**AdaBoost (Adaptive Boosting)** is an **ensemble learning algorithm** that combines many weak learners, usually **small decision trees (Decision Stumps)**, to create a strong classifier.

The key idea is:

> **Train models sequentially, and after each model, give more importance to the samples that previous models classified incorrectly.**

So unlike **Bagging**, where models are trained independently, **AdaBoost learns from previous mistakes**.

---

# 1. Why AdaBoost?

Suppose we have a classification dataset:

```text
Sample     Actual     Model 1
A             0          ✓
B             1          ✓
C             0          ✗
D             1          ✓
E             0          ✗
```

Model 1 made mistakes on **C and E**.

AdaBoost says:

> "The next model should pay more attention to C and E."

So we increase their importance.

The second model focuses more on difficult samples.

Then:

```text
Model 1 → finds mistakes
             ↓
       increase weight
             ↓
Model 2 → focuses on mistakes
             ↓
       increase weight
             ↓
Model 3 → focuses on remaining mistakes
             ↓
        final ensemble
```

---

# 2. What does "Adaptive" mean?

The word **Adaptive** is important.

AdaBoost **adapts the training process after every weak learner**.

If a sample is classified correctly:

```text
weight ↓
```

If a sample is classified incorrectly:

```text
weight ↑
```

Therefore, difficult observations progressively receive more attention.

---

# 3. What is a Weak Learner?

A **weak learner** is a model that performs only slightly better than random guessing.

For binary classification:

```text
Random guessing ≈ 50%
```

A weak learner might achieve:

```text
55%
60%
65%
```

accuracy.

AdaBoost combines many such weak learners.

For example:

```text
Weak learner 1 → 60%
Weak learner 2 → 62%
Weak learner 3 → 58%
Weak learner 4 → 65%
        ↓
AdaBoost ensemble → potentially much stronger
```

The most common weak learner is:

## Decision Stump

A decision stump is a decision tree with:

```text
max_depth = 1
```

Example:

```text
             Age > 30?
              /      \
            No        Yes
            ↓          ↓
          Class 0    Class 1
```

It is extremely simple, but AdaBoost combines many of them.

---

# 4. AdaBoost vs Bagging

This is very important because you recently studied Bagging.

| Bagging                                | AdaBoost                                  |
| -------------------------------------- | ----------------------------------------- |
| Models trained independently           | Models trained sequentially               |
| Focuses on reducing variance           | Often reduces bias and variance           |
| Bootstrap samples                      | Uses sample weights                       |
| Models don't depend on previous models | Later models depend on previous models    |
| Random Forest is an example            | AdaBoost is an example                    |
| Parallel training possible             | Sequential training                       |
| Usually robust to noise                | Can be sensitive to noisy/outlier samples |

### Bagging

```text
Dataset
  ↓
 ┌──────┬──────┬──────┐
 ↓      ↓      ↓
Model1 Model2 Model3
 ↓      ↓      ↓
 └──────┼──────┘
        ↓
      Voting
```

### AdaBoost

```text
Dataset
   ↓
Model 1
   ↓
Find mistakes
   ↓
Increase weights
   ↓
Model 2
   ↓
Find mistakes
   ↓
Increase weights
   ↓
Model 3
   ↓
      Final weighted voting
```

---

# 5. Core Idea of AdaBoost

Initially, every training sample receives equal weight.

Suppose there are 5 samples:

```text
A B C D E
```

Initial weights:

```text
A = 0.2
B = 0.2
C = 0.2
D = 0.2
E = 0.2
```

Train the first weak learner.

Suppose:

```text
A → correct
B → correct
C → wrong
D → correct
E → wrong
```

AdaBoost increases the weights of:

```text
C
E
```

Now the next learner sees:

```text
A → low importance
B → low importance
C → high importance
D → low importance
E → high importance
```

The next model therefore tries harder to correctly classify C and E.

---

# 6. AdaBoost Algorithm

For binary classification, the basic AdaBoost procedure is:

### Step 1 — Initialize weights

For `N` training samples:

$$
w_i = \frac{1}{N}
$$

Every sample initially has equal importance.

For 5 samples:

$$
w_i = \frac{1}{5}=0.2
$$

---

# 7. Step 2 — Train a Weak Learner

Train a weak learner using the current sample weights.

For example:

```python
DecisionTreeClassifier(max_depth=1)
```

The model predicts:

```text
Actual:     1  1  0  0  1
Prediction: 1  0  0  1  1
```

Errors:

```text
Sample 1 → correct
Sample 2 → wrong
Sample 3 → correct
Sample 4 → wrong
Sample 5 → correct
```

---

# 8. Step 3 — Calculate Weighted Error

AdaBoost does not simply calculate normal error.

It calculates **weighted error**.

$$
\epsilon_t =
\sum_{i=1}^{N} w_i I(y_i \neq h_t(x_i))
$$

where:

* \(\epsilon_t\) = error of weak learner
* \(w_i\) = weight of sample \(i\)
* \(h_t(x_i)\) = prediction
* \(y_i\) = actual label
* \(I\) = 1 if prediction is wrong, otherwise 0

---

## Example

Suppose:

| Sample | Weight | Correct? |
| ------ | -----: | -------- |
| A      |   0.20 | ✓        |
| B      |   0.20 | ✗        |
| C      |   0.20 | ✓        |
| D      |   0.20 | ✗        |
| E      |   0.20 | ✓        |

Weighted error:

$$
\epsilon =
0.20+0.20
$$

$$
\boxed{\epsilon=0.40}
$$

So the weak learner has a weighted error of 40%.

---

# 9. Step 4 — Calculate Learner Weight

This is one of the most important equations in AdaBoost.

The weak learner receives a weight:

$$
\boxed{
\alpha_t =
\frac{1}{2}
\ln
\left(
\frac{1-\epsilon_t}{\epsilon_t}
\right)
}
$$

where:

$$
\epsilon_t = \text{weighted error}
$$

---

## Example

Suppose:

$$
\epsilon=0.2
$$

Then:

$$
\alpha =
\frac{1}{2}
\ln
\left(
\frac{1-0.2}{0.2}
\right)
$$

$$
=
\frac{1}{2}\ln(4)
$$

$$
\approx0.693
$$

So:

$$
\boxed{\alpha\approx0.693}
$$

This learner receives relatively high importance.

---

# 10. What Does Alpha Mean?

\(\alpha\) tells us **how much influence the weak learner gets in the final prediction**.

### Low error

$$
\epsilon \downarrow
$$

means:

$$
\alpha \uparrow
$$

So the learner gets more voting power.

### High error

$$
\epsilon \uparrow
$$

means:

$$
\alpha \downarrow
$$

So the learner gets less voting power.

---

# 11. Important Cases

### Perfect learner

If:

$$
\epsilon=0
$$

then:

$$
\alpha \rightarrow \infty
$$

The learner is perfect, so theoretically it gets extremely high influence.

---

### Random learner

For binary classification:

$$
\epsilon=0.5
$$

Then:

$$
\alpha =
\frac12\ln\left(\frac{0.5}{0.5}\right)
$$

$$
=\frac12\ln(1)
$$

$$
=0
$$

So:

$$
\boxed{\alpha=0}
$$

A random classifier gets no influence.

---

### Worse than random

If:

$$
\epsilon>0.5
$$

then:

$$
\alpha<0
$$

The learner is worse than random guessing.

In standard AdaBoost, such a learner generally isn't useful; implementations may stop or otherwise handle the situation.

---

# 12. Step 5 — Update Sample Weights

Now comes the key part.

AdaBoost increases the weights of incorrectly classified samples.

A common binary AdaBoost update is:

$$
\boxed{
w_i^{(t+1)}
=
w_i^{(t)}
\exp(-\alpha_t y_i h_t(x_i))
}
$$

where labels are encoded as:

$$
y_i \in \{-1,+1\}
$$

and predictions are:

$$
h_t(x_i)\in\{-1,+1\}
$$

---

## If prediction is correct

Then:

$$
y_i h_t(x_i)=+1
$$

Therefore:

$$
w_i^{new}
=
w_i e^{-\alpha}
$$

Since:

$$
e^{-\alpha}<1
$$

the weight decreases.

---

## If prediction is wrong

Then:

$$
y_i h_t(x_i)=-1
$$

Therefore:

$$
w_i^{new}
=
w_i e^{+\alpha}
$$

Since:

$$
e^{+\alpha}>1
$$

the weight increases.

Therefore:

```text
Correct prediction
      ↓
weight decreases

Wrong prediction
      ↓
weight increases
```

---

# 13. Numerical Example

Suppose:

$$
\alpha=0.693
$$

and every sample initially has:

$$
w=0.2
$$

### Correct sample

$$
w_{new}=0.2e^{-0.693}
$$

Since:

$$
e^{-0.693}\approx0.5
$$

we get:

$$
w_{new}=0.1
$$

### Incorrect sample

$$
w_{new}=0.2e^{0.693}
$$

$$
\approx0.2(2)
$$

$$
=0.4
$$

So:

```text
Correct sample:
0.20 → 0.10

Wrong sample:
0.20 → 0.40
```

Then all weights are **normalized** so that:

$$
\sum_i w_i=1
$$

---

# 14. Why Normalize the Weights?

After updating, the weights may no longer sum to 1.

For example:

```text
0.1
0.4
0.1
0.4
0.1
```

Sum:

$$
1.1
$$

We normalize:

$$
w_i =
\frac{w_i}{\sum_jw_j}
$$

So:

```text
0.1 / 1.1 = 0.091
0.4 / 1.1 = 0.364
...
```

Now:

$$
\sum_i w_i=1
$$

These become the sample weights for the next learner.

---

# 15. Complete AdaBoost Process

The complete process looks like this:

```text
              Training Data
                    ↓
        Initialize equal weights
                    ↓
            Train weak learner
                    ↓
          Calculate weighted error
                    ↓
             Calculate α
                    ↓
        Update sample weights
                    ↓
              Normalize
                    ↓
        Train next weak learner
                    ↓
          Again calculate error
                    ↓
             Calculate α
                    ↓
        Update sample weights
                    ↓
                  ...
                    ↓
          Combine all learners
                    ↓
             Final prediction
```

---

# 16. Final Prediction

After training \(T\) weak learners, AdaBoost combines them using their \(\alpha\) values.

For binary classification:

$$
\boxed{
H(x)=
sign
\left(
\sum_{t=1}^{T}
\alpha_t h_t(x)
\right)
}
$$

So every weak learner gets a vote weighted by its importance.

---

## Example

Suppose we have three learners:

```text
Learner 1 → α = 0.7
Learner 2 → α = 0.4
Learner 3 → α = 0.9
```

For a particular sample:

```text
Learner 1 → +1
Learner 2 → -1
Learner 3 → +1
```

Then:

$$
0.7(+1)+0.4(-1)+0.9(+1)
$$

$$
=0.7-0.4+0.9
$$

$$
=1.2
$$

Since:

$$
1.2>0
$$

final prediction:

$$
\boxed{+1}
$$

---

# 17. AdaBoost Is Sequential

This is a critical characteristic.

You **cannot generally train AdaBoost's learners independently**.

Why?

Because:

```text
Model 2 depends on Model 1
Model 3 depends on Model 2
Model 4 depends on Model 3
```

because the sample weights change after every learner.

Therefore:

### Bagging

```text
Model 1 ─┐
Model 2 ─┼──→ Ensemble
Model 3 ─┤
Model 4 ─┘
```

Can be parallelized.

### AdaBoost

```text
Model 1
   ↓
Model 2
   ↓
Model 3
   ↓
Model 4
```

Sequential.

---

# 18. What Does AdaBoost Actually Learn?

There are two things being learned:

### 1. Weak learners

$$
h_1,h_2,h_3,\ldots,h_T
$$

### 2. Learner weights

$$
\alpha_1,\alpha_2,\alpha_3,\ldots,\alpha_T
$$

Final model:

$$
F(x)=
\sum_{t=1}^{T}\alpha_t h_t(x)
$$

Classification:

$$
H(x)=sign(F(x))
$$

---

# 19. AdaBoost and Decision Trees

The most common combination is:

```python
DecisionTreeClassifier(max_depth=1)
```

Why shallow trees?

Because AdaBoost wants:

> **Many simple learners that make different mistakes.**

A deep tree can become too powerful individually and may overfit.

For example:

```text
Tree 1 → simple rule
Tree 2 → another simple rule
Tree 3 → focuses on difficult samples
Tree 4 → another correction
...
```

Together they can represent a complex decision boundary.

---

# 20. AdaBoost in Scikit-Learn

For classification:

```python
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

base_model = DecisionTreeClassifier(
    max_depth=1,
    random_state=42
)

model = AdaBoostClassifier(
    estimator=base_model,
    n_estimators=100,
    learning_rate=1.0,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
```

---

# 21. Important Hyperparameters

You should know these well.

## 1. `n_estimators`

Number of weak learners.

```python
AdaBoostClassifier(
    n_estimators=100
)
```

Means:

```text
100 weak learners
```

Example:

```text
n_estimators=10
n_estimators=50
n_estimators=100
n_estimators=500
```

Generally, increasing it allows a more expressive ensemble, but can increase training time and sometimes overfitting.

---

# 22. `learning_rate`

Controls the contribution of each weak learner.

```python
learning_rate=1.0
```

Final learner contribution is effectively scaled by the learning rate.

A smaller learning rate usually requires more estimators.

Typical combinations:

```text
learning_rate = 1.0
n_estimators = 100
```

or:

```text
learning_rate = 0.1
n_estimators = 500
```

There is a trade-off:

$$
\boxed{
\text{learning rate} \downarrow
\Rightarrow
\text{usually need more estimators}
}
$$

---

# 23. `estimator`

Defines the base learner.

Example:

```python
DecisionTreeClassifier(max_depth=1)
```

You can use a different weak learner, subject to AdaBoost's estimator requirements.

For example:

```python
base_model = DecisionTreeClassifier(
    max_depth=2
)

model = AdaBoostClassifier(
    estimator=base_model,
    n_estimators=100
)
```

---

# 24. `max_depth` of the Base Tree

This isn't an AdaBoost hyperparameter itself; it's a parameter of the base estimator.

Example:

```python
DecisionTreeClassifier(max_depth=1)
```

Decision stump.

Or:

```python
DecisionTreeClassifier(max_depth=2)
```

Slightly more complex learner.

Generally:

```text
Depth 1 → very weak learner
Depth 2 → stronger learner
Depth 3 → even stronger
```

But increasing tree complexity can increase overfitting.

---

# 25. `random_state`

Used for reproducibility.

```python
AdaBoostClassifier(
    n_estimators=100,
    random_state=42
)
```

Without a fixed random state, results involving randomized components may vary between runs.

---

# 26. AdaBoost and Learning Rate

This relationship is particularly important.

Suppose:

```text
Model A:
n_estimators = 100
learning_rate = 1.0
```

versus:

```text
Model B:
n_estimators = 500
learning_rate = 0.1
```

Model B takes smaller steps but has more stages.

Conceptually:

```text
High learning rate
    ↓
larger contribution per learner
    ↓
fewer learners may be needed
```

while:

```text
Low learning rate
    ↓
smaller contribution per learner
    ↓
more learners may be needed
```

This is similar to the learning-rate concept you saw in gradient boosting.

---

# 27. AdaBoost Loss Function

One of the most important theoretical concepts is that classical AdaBoost can be interpreted as minimizing **exponential loss**.

For binary labels:

$$
y_i\in\{-1,+1\}
$$

the exponential loss is:

$$
\boxed{
L(y,F(x))=e^{-yF(x)}
}
$$

where:

$$
F(x)=\sum_t\alpha_t h_t(x)
$$

AdaBoost builds the ensemble stage by stage to reduce this loss.

---
