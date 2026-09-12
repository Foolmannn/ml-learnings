
# Bagging in Machine Learning — In Detail

**Bagging** stands for **Bootstrap Aggregating**. It is an **ensemble learning technique** where multiple models are trained independently on different randomly generated samples of the training data, and their predictions are combined.

The main goal of Bagging is to **reduce variance and make the model more stable and less prone to overfitting**.

---

# 1. Why do we need Bagging?

Suppose we train a single Decision Tree:

```text
Training Data
      ↓
Decision Tree
      ↓
Prediction
```

Decision Trees can have **high variance**.

A small change in the training data can produce a very different tree:

```text
Dataset A → Tree A → Prediction A
Dataset B → Tree B → Prediction B
```

This happens because decision trees can strongly adapt to the particular training samples they receive.

Bagging tries to solve this:

```text
                  ┌── Model 1 ──┐
                  │             │
Training Data ────┼── Model 2 ──┼──→ Combine Predictions
                  │             │
                  ├── Model 3 ──┤
                  │             │
                  └── Model N ──┘
```

Instead of trusting one model, we train **many models** and combine them.

---

# 2. What does Bootstrap mean?

The most important concept behind Bagging is **Bootstrap Sampling**.

Suppose we have:

```text
Original Dataset

A B C D E
```

We randomly select samples **with replacement**.

For example:

```text
Bootstrap Sample 1:
A C C D E

Bootstrap Sample 2:
B B C D A

Bootstrap Sample 3:
E D D A C
```

Notice that:

- Some samples appear multiple times.
- Some samples aren't selected.
- Each bootstrap dataset usually has the **same size as the original dataset**.

### What does "with replacement" mean?

Suppose we select `C`.

After selecting C, we put it back into the population.

So C can be selected again:

```text
A B C D E
    ↑
   C selected

A B C D E
    ↑
   C can be selected again
```

Therefore:

```text
A C C D E
```

is possible.

---

# 3. Complete Bagging Process

Suppose our dataset contains:

```text
X → Features
y → Target
```

We create multiple bootstrap datasets:

```text
                 Original Dataset
                       │
             ┌─────────┼─────────┐
             ↓         ↓         ↓
         Bootstrap  Bootstrap  Bootstrap
            1          2          3
             ↓         ↓         ↓
          Model 1   Model 2   Model 3
             ↓         ↓         ↓
          Pred 1    Pred 2    Pred 3
             └─────────┼─────────┘
                       ↓
                 Aggregation
                       ↓
                  Final Prediction
```

The models are generally trained **independently**, so Bagging is naturally parallelizable.

---

# 4. Bagging for Classification

For classification, Bagging usually combines predictions using **majority voting**.

Suppose we have five models:

```text
Model 1 → Cat
Model 2 → Dog
Model 3 → Dog
Model 4 → Dog
Model 5 → Cat
```

Voting:

```text
Cat → 2 votes
Dog → 3 votes
```

Therefore:

```text
Final Prediction = Dog
```

Mathematically:

$$
\hat y = \operatorname{mode}
\{h_1(x),h_2(x),...,h_B(x)\}
$$

where:

- $B$ = number of models
- $h_i(x)$ = prediction from model $i$
- `mode` = most frequently occurring prediction

---

# 5. Bagging for Regression

For regression, we generally take the **average** of predictions.

Suppose:

```text
Model 1 → 100
Model 2 → 110
Model 3 → 105
Model 4 → 95
Model 5 → 90
```

Final prediction:

$$
\hat y =
\frac{100+110+105+95+90}{5}
$$

$$
\hat y = 100
$$

So:

```text
Final Prediction = Average of predictions
```

Mathematically:

$$
\hat y = \frac{1}{B}\sum_{b=1}^{B}h_b(x)
$$

---

# 6. Why does Bagging work?

This is the most important theoretical idea.

Individual models may have high variance.

For example:

```text
Model 1 → 80
Model 2 → 120
Model 3 → 90
Model 4 → 110
Model 5 → 100
```

Individual predictions vary significantly.

But their average is:

```text
Average = 100
```

The random errors tend to partially cancel each other.

Therefore:

> **Bagging primarily reduces variance.**

---

# 7. Bias vs Variance

Recall:

$$
Total\ Error \approx Bias^2 + Variance + Irreducible\ Error
$$

Bagging mainly attacks:

$$
\boxed{Variance}
$$

It generally does **not** dramatically reduce bias.

For example:

```text
High variance model
        ↓
      Bagging
        ↓
Lower variance
        ↓
More stable model
```

This is why Bagging is particularly useful with **high-variance, low-bias models**, such as fully grown Decision Trees.

---

# 8. Statistical Explanation

Suppose we have $B$ models.

Each model has variance:

$$
\sigma^2
$$

If the models were completely independent, the variance of their average would be:

$$
Var(\bar X)=\frac{\sigma^2}{B}
$$

So increasing the number of models decreases variance.

For example:

### One model

$$
Var = \sigma^2
$$

### 10 independent models

$$
Var = \frac{\sigma^2}{10}
$$

### 100 independent models

$$
Var = \frac{\sigma^2}{100}
$$

But there's an important problem.

The models aren't completely independent because they are trained from the same original dataset.

Therefore, **correlation between models matters**.

---

# 9. Effect of Correlation

A useful formula for the variance of an ensemble is:

$$
Var(\bar X)
=
\rho\sigma^2+
\frac{1-\rho}{B}\sigma^2
$$

where:

- $B$ = number of models
- $\sigma^2$ = variance of each model
- $\rho$ = correlation between models

This gives us an important insight.

### If models are highly correlated

$$
\rho \approx 1
$$

Then:

$$
Var(\bar X) \approx \sigma^2
$$

Bagging won't help much.

### If models are less correlated

$$
\rho \approx 0
$$

Then:

$$
Var(\bar X)\approx\frac{\sigma^2}{B}
$$

Bagging becomes very effective.

Therefore:

> **Good ensembles need accurate models that are also sufficiently diverse.**

Bootstrap sampling creates this diversity.

---

# 10. Why Sampling With Replacement?

Suppose we have:

```text
A B C D E
```

We could divide the data into:

```text
A B → Model 1
C D → Model 2
E   → Model 3
```

But this isn't ideal because each model gets a very different amount of data.

Instead, Bagging creates:

```text
A C C D E
B B C D E
A A D E E
```

Each model gets approximately the same number of training observations, but the exact composition differs.

This creates **diversity between models**.

---

# 11. Example

Suppose we have 10 observations:

```text
1 2 3 4 5 6 7 8 9 10
```

We want 3 bootstrap samples.

### Bootstrap sample 1

```text
1 4 4 7 8 9 10 2 2 6
```

### Bootstrap sample 2

```text
3 3 5 6 7 7 8 1 9 10
```

### Bootstrap sample 3

```text
2 5 5 5 6 8 9 9 10 1
```

Then:

```text
Sample 1 → Decision Tree 1
Sample 2 → Decision Tree 2
Sample 3 → Decision Tree 3
```

For a new input:

```text
Xnew
```

Predictions:

```text
Tree 1 → 1
Tree 2 → 0
Tree 3 → 1
```

Final:

```text
1
```

because 1 receives two votes.

---

# 12. Bagging vs Single Decision Tree

### Single Decision Tree

```text
Dataset
   ↓
One Tree
   ↓
Prediction
```

Potential problem:

```text
High variance
Overfitting
Sensitive to training data
```

### Bagging

```text
             ┌→ Tree 1 ─┐
             ├→ Tree 2 ─┤
Dataset ─────┼→ Tree 3 ─┼→ Voting/Average
             ├→ Tree 4 ─┤
             └→ Tree 5 ─┘
```

Advantages:

```text
Lower variance
More stable
Better generalization
Less sensitive to individual observations
```

---

# 13. BaggingClassifier in Scikit-Learn

Scikit-learn provides:

```python
from sklearn.ensemble import BaggingClassifier
```

Example:

```python
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

model = BaggingClassifier(
    estimator=DecisionTreeClassifier(),
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
```

---

# 14. Important Parameters of BaggingClassifier

### `estimator`

Specifies the base model.

```python
estimator=DecisionTreeClassifier()
```

You could theoretically use different estimators that support the required interface.

---

### `n_estimators`

Number of base models.

```python
n_estimators=100
```

For example:

```text
n_estimators = 10
→ 10 models

n_estimators = 100
→ 100 models

n_estimators = 500
→ 500 models
```

Increasing this generally reduces variance up to a point, but increases computation.

---

### `max_samples`

Controls how many samples each bootstrap dataset receives.

Example:

```python
max_samples=0.8
```

Each model receives:

```text
80% of the training samples
```

It can also be an integer:

```python
max_samples=500
```

meaning each base estimator receives 500 sampled observations.

---

### `max_features`

Controls the number/fraction of features used by each base estimator.

Example:

```python
max_features=0.8
```

Each estimator uses 80% of the available features.

This can introduce additional diversity.

---

### `bootstrap`

Controls whether samples are drawn **with replacement**.

```python
bootstrap=True
```

This is the standard Bagging approach.

If:

```python
bootstrap=False
```

sampling is done without replacement.

---

### `bootstrap_features`

Controls whether features are sampled with replacement.

```python
bootstrap_features=True
```

This creates additional randomness among models.

---

### `oob_score`

OOB means **Out-of-Bag**.

```python
oob_score=True
```

This allows us to estimate generalization performance using observations that weren't selected for a particular bootstrap sample.

---

### `random_state`

Controls reproducibility.

```python
random_state=42
```

---

### `n_jobs`

Controls parallel processing.

```python
n_jobs=-1
```

means use all available CPU cores.

---

# 15. BaggingClassifier Example

```python
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

bagging = BaggingClassifier(
    estimator=DecisionTreeClassifier(
        max_depth=None
    ),
    n_estimators=100,
    max_samples=1.0,
    max_features=1.0,
    bootstrap=True,
    oob_score=True,
    n_jobs=-1,
    random_state=42
)

bagging.fit(X_train, y_train)

print("Training Score:", bagging.score(X_train, y_train))
print("Testing Score:", bagging.score(X_test, y_test))
print("OOB Score:", bagging.oob_score_)
```

---

# 16. BaggingRegressor

For regression:

```python
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor

model = BaggingRegressor(
    estimator=DecisionTreeRegressor(),
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
```

Instead of voting:

```text
Tree 1 → 100
Tree 2 → 110
Tree 3 → 90
```

we average:

```text
Final = 100
```

---

# 17. Out-of-Bag Samples

This is an **important Bagging concept**.

Suppose our original dataset has:

```text
A B C D E
```

Bootstrap sample:

```text
A C C D E
```

Which observations were not selected?

```text
B
```

So B is called an:

> **Out-of-Bag (OOB) observation for that model.**

---

# 18. Why are OOB Samples Useful?

Normally, we need:

```text
Training Data
        ↓
Train model
        ↓
Validation/Test Data
```

But with Bagging, some observations aren't used to train each individual model.

Those observations can be used to evaluate that model.

For example:

```text
Bootstrap 1
A C C D E
        ↓
Model 1

B = OOB
```

Then:

```text
B → Model 1 → Prediction
```

Repeat this across many models.

We can aggregate those OOB predictions and calculate an **OOB score**.

This provides an internal estimate of generalization performance without needing a separate validation set for that purpose.

---

# 19. Why does OOB work?

For a bootstrap sample of size $n$, each observation has probability:

$$
1-\frac{1}{n}
$$

of not being selected on any particular draw.

After $n$ draws:

$$
P(\text{not selected})
=
\left(1-\frac{1}{n}\right)^n
$$

As $n\rightarrow\infty$:

$$
\left(1-\frac{1}{n}\right)^n
\rightarrow e^{-1}
$$

Therefore:

$$
e^{-1}\approx0.368
$$

So approximately:

$$
\boxed{36.8\%}
$$

of the unique observations are OOB for a given bootstrap model, while about:

$$
\boxed{63.2\%}
$$

are unique observations included in the bootstrap sample.

This is why OOB evaluation is quite useful in Bagging.

---

# 20. Bagging and Feature Sampling

Bagging can introduce randomness in **two dimensions**:

### 1. Sample randomness

```text
Different rows
```

### 2. Feature randomness

```text
Different columns
```

For example:

```text
             Features
          A B C D E F

Model 1 → A B C
Model 2 → B D E
Model 3 → A C F
```

This makes models even less correlated.

---

# 21. Bagging vs Random Forest

This is extremely important because these two are closely related.

### Bagging with Decision Trees

```text
Bootstrap samples
       ↓
Decision Trees
       ↓
Voting/Average
```

### Random Forest

```text
Bootstrap samples
       ↓
Decision Trees
       ↓
Random feature selection
       ↓
Voting/Average
```

So Random Forest adds **random feature selection** at each tree split.

Therefore:

> **Random Forest can be viewed as a more randomized tree-based ensemble than basic Bagging.**

---

# 22. Bagging vs Random Forest

| Feature | Bagging | Random Forest |
|---|---|---|
| Bootstrap samples | Yes | Yes |
| Multiple models | Yes | Yes |
| Voting/Average | Yes | Yes |
| Base model | Usually same chosen estimator | Decision Trees |
| Random feature selection | Optional | Yes |
| Main goal | Reduce variance | Reduce variance + decorrelate trees |
| Parallelizable | Yes | Yes |
| OOB evaluation | Available | Available |

---

# 23. Bagging vs Boosting

Another very important distinction.

### Bagging

Models are trained:

```text
Independently
```

Example:

```text
Dataset
 ↓
Model 1 ──┐
Model 2 ──┤
Model 3 ──┼→ Combine
Model 4 ──┤
Model 5 ──┘
```

### Boosting

Models are trained:

```text
Sequentially
```

Example:

```text
Dataset
   ↓
Model 1
   ↓
Focus on errors
   ↓
Model 2
   ↓
Focus on remaining errors
   ↓
Model 3
   ↓
Combine
```

---

# 24. Bagging vs Boosting — Core Difference

| Bagging | Boosting |
|---|---|
| Parallel | Sequential |
| Independent models | Dependent models |
| Bootstrap sampling commonly used | Reweighting/residual correction depending on algorithm |
| Primarily reduces variance | Can reduce bias and variance |
| Less prone to overfitting in many settings | Can overfit depending on algorithm/tuning |
| Randomness is important | Sequential correction is important |
| Example: BaggingClassifier | AdaBoost, Gradient Boosting, XGBoost |

---

# 25. When Should You Use Bagging?

Bagging is particularly useful when your base model has **high variance**.

Good example:

```text
Decision Tree
```

Suppose:

```text
Training accuracy = 100%
Testing accuracy = 78%
```

This suggests potential overfitting/high variance.

Bagging can help:

```text
Single Tree
     ↓
High variance

Bagging
     ↓
Many trees
     ↓
Average/Vote
     ↓
Reduced variance
```

---

# 26. When Bagging Is Less Useful

If your base model already has very low variance, Bagging may provide limited benefit.

For example, imagine a very stable model:

```text
Model 1 ≈ Model 2 ≈ Model 3
```

Their predictions are highly correlated.

Averaging them doesn't add much.

Remember:

$$
\boxed{\text{Diversity + Accuracy} \rightarrow \text{Useful Ensemble}}
$$

---

# 27. Advantages of Bagging

### 1. Reduces variance

This is its primary advantage.

### 2. Helps reduce overfitting

Especially for high-variance models.

### 3. More stable predictions

A single unusual observation is less likely to dominate the final result.

### 4. Parallelizable

Models can be trained independently.

### 5. Works for classification and regression

```text
BaggingClassifier
BaggingRegressor
```

### 6. Can provide OOB evaluation

You can use observations not included in individual bootstrap samples.

---

# 28. Disadvantages

### 1. More computationally expensive

Instead of:

```text
1 model
```

you train:

```text
100+ models
```

### 2. More memory usage

Multiple models need to be stored.

### 3. Doesn't necessarily reduce bias

Bagging primarily targets variance.

### 4. Less interpretable

A single Decision Tree can be visualized relatively easily.

A 500-model ensemble is much harder to interpret.

---

# 29. Important Intuition

Imagine asking one person:

> "Should I invest in this company?"

They might say:

```text
YES
```

But their answer could be based on a particular bias.

Now ask 100 independent analysts:

```text
Analyst 1 → YES
Analyst 2 → NO
Analyst 3 → YES
...
Analyst 100 → YES
```

If most say:

```text
YES
```

you have a more stable decision.

This is the basic intuition behind ensemble learning.

Bagging essentially says:

> **Don't depend too much on one unstable learner; train many versions and aggregate their decisions.**

---

# 30. Full Bagging Pipeline

You should remember this diagram:

```text
                  Original Dataset
                         │
              ┌──────────┼──────────┐
              ↓          ↓          ↓
        Bootstrap 1  Bootstrap 2  Bootstrap 3
              │          │          │
              ↓          ↓          ↓
           Model 1     Model 2     Model 3
              │          │          │
              └──────────┼──────────┘
                         ↓
                  Aggregate Results
                    /           \
                   /             \
          Classification       Regression
                ↓                   ↓
         Majority Voting         Average
                ↓                   ↓
          Final Prediction    Final Prediction
```

---

# 31. Bagging in One Example

Imagine predicting whether a person will buy a movie subscription.

Dataset:

```text
Age
Income
Watch Hours
Previous Purchases
```

Create 100 bootstrap datasets:

```text
Dataset
  ↓
100 Bootstrap Samples
  ↓
100 Decision Trees
```

For a new customer:

```text
Tree 1 → Buy
Tree 2 → Don't Buy
Tree 3 → Buy
Tree 4 → Buy
...
Tree 100 → Buy
```

Suppose:

```text
Buy       = 72
Don't Buy = 28
```

Final prediction:

```text
Buy
```

The ensemble is less dependent on the quirks of any one tree.

---

# 32. Bagging Does NOT Mean Simply Training Many Models

This distinction is important.

If you simply do:

```python
model1.fit(X, y)
model2.fit(X, y)
model3.fit(X, y)
```

using exactly the same training data and algorithm with no meaningful randomness, the models may be nearly identical.

That's not the key idea of Bagging.

Bagging specifically involves:

$$
\boxed{\text{Bootstrap Sampling} + \text{Aggregation}}
$$

Usually:

$$
\boxed{\text{Bootstrap Samples} + \text{Multiple Base Learners} + \text{Voting/Averaging}}
$$

---

# 33. Bagging's Core Formula

### Classification

$$
\boxed{
\hat y =
\operatorname{mode}
\left(
h_1(x),h_2(x),...,h_B(x)
\right)
}
$$

### Regression

$$
\boxed{
\hat y =
\frac{1}{B}
\sum_{b=1}^{B}h_b(x)
}
$$

### Main statistical objective

$$
\boxed{\text{Reduce Variance}}
$$

---

# 34. Important Parameters to Remember

For practical ML, focus on these:

```python
BaggingClassifier(
    estimator=...,
    n_estimators=100,
    max_samples=1.0,
    max_features=1.0,
    bootstrap=True,
    bootstrap_features=False,
    oob_score=True,
    n_jobs=-1,
    random_state=42
)
```

The most important ones are:

| Parameter | Purpose |
|---|---|
| `estimator` | Base learner |
| `n_estimators` | Number of models |
| `max_samples` | Samples given to each model |
| `max_features` | Features given to each model |
| `bootstrap` | Sample with replacement |
| `oob_score` | OOB evaluation |
| `n_jobs` | Parallel processing |
| `random_state` | Reproducibility |

---

# 35. Bagging — Exam Definition

> **Bagging (Bootstrap Aggregating) is an ensemble learning technique that trains multiple base learners on different bootstrap samples of the training dataset and combines their predictions using majority voting for classification or averaging for regression, primarily to reduce variance and improve generalization.**

---

# 36. What You Should Remember for ML

The entire concept can be compressed into:

```text
                 BAGGING
                    │
        ┌───────────┴───────────┐
        ↓                       ↓
 Bootstrap Sampling       Multiple Models
        │                       │
        └───────────┬───────────┘
                    ↓
              Predictions
                    ↓
          ┌─────────┴─────────┐
          ↓                   ↓
     Classification       Regression
          ↓                   ↓
    Majority Vote          Average
          ↓                   ↓
       Final Result
```

And the **three most important points** are:

1. **Bootstrap sampling** → create different training datasets.
2. **Train multiple models independently** → create an ensemble.
3. **Aggregate predictions** → reduce variance and improve stability.

The natural next topic after Bagging is **Random Forest**, because Random Forest builds directly on these ideas by adding **random feature selection**, which reduces correlation between the trees even further.