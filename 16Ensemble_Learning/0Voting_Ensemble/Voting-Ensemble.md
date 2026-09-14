
# Voting Ensemble in Machine Learning — In Detail

**Voting Ensemble** is an **ensemble learning technique** where multiple different machine-learning models are combined to make one final prediction.

The core idea is simple:

> **Instead of trusting one model, ask several models and combine their answers.**

For example, suppose we want to classify whether an email is spam:

| Model | Prediction |
|---|---|
| Logistic Regression | Spam |
| Decision Tree | Not Spam |
| KNN | Spam |
| SVM | Spam |
| Naive Bayes | Spam |

The majority says **Spam**, so the Voting Ensemble predicts:

**→ Spam**

This often gives better generalization than using only one of the individual models.

---

# 1. Why do we need Voting Ensemble?

Different ML algorithms learn patterns differently.

For example:

- Logistic Regression → learns a linear decision boundary
- Decision Tree → learns rule-based splits
- KNN → looks at nearby examples
- SVM → finds a maximum-margin boundary
- Naive Bayes → uses probability assumptions

A dataset may contain different types of patterns.

One model might make an incorrect prediction because of its particular assumptions, while another model gets it right.

By combining them:

$$
\boxed{\text{Multiple Models} \rightarrow \text{Combine Predictions} \rightarrow \text{Final Prediction}}
$$

we can often reduce the effect of individual model errors.

---

# 2. Basic Architecture

Suppose we have three classifiers:

```text
                    Training Data
                         │
          ┌──────────────┼──────────────┐
          ↓              ↓              ↓
   Logistic Regression  Decision Tree    SVM
          │              │              │
          ↓              ↓              ↓
       Prediction      Prediction      Prediction
          │              │              │
          └──────────────┼──────────────┘
                         ↓
                  Voting Mechanism
                         ↓
                  Final Prediction
```

The individual models are called **base estimators**.

---

# 3. Types of Voting

There are two major types:

1. **Hard Voting**
2. **Soft Voting**

The difference is extremely important.

---

# 4. Hard Voting

Hard voting uses the **predicted class labels**.

Suppose we have 5 classifiers:

```text
Model 1 → Class A
Model 2 → Class B
Model 3 → Class A
Model 4 → Class A
Model 5 → Class B
```

Count the votes:

```text
Class A → 3 votes
Class B → 2 votes
```

Therefore:

$$
\boxed{\text{Final Prediction} = A}
$$

This is essentially **majority voting**.

---

# 5. Mathematical Formulation of Hard Voting

Suppose there are $M$ classifiers:

$$
h_1(x),h_2(x),...,h_M(x)
$$

Each classifier predicts a class.

The final prediction is the class receiving the maximum number of votes:

$$
\hat y =
\arg\max_k
\sum_{i=1}^{M} I(h_i(x)=k)
$$

where:

$$
I(\text{condition})
=
\begin{cases}
1 & \text{if condition is true}\\
0 & \text{otherwise}
\end{cases}
$$

### Example

Three models predict:

$$
h_1(x)=1
$$

$$
h_2(x)=0
$$

$$
h_3(x)=1
$$

Votes:

$$
1:2
$$

$$
0:1
$$

Therefore:

$$
\boxed{\hat y=1}
$$

---

# 6. Hard Voting Example

Imagine a medical classification problem.

Models:

- Logistic Regression
- Decision Tree
- SVM
- KNN
- Random Forest

Prediction:

| Model | Prediction |
|---|---|
| Logistic Regression | Disease |
| Decision Tree | Disease |
| SVM | No Disease |
| KNN | Disease |
| Random Forest | No Disease |

Votes:

$$
Disease=3
$$

$$
NoDisease=2
$$

Final:

$$
\boxed{Disease}
$$

---

# 7. Soft Voting

Soft voting is more interesting.

Instead of only considering the predicted class, we consider the **probabilities predicted by each model**.

Suppose there are two classes:

- Class 0
- Class 1

Three models produce:

| Model | P(Class 0) | P(Class 1) |
|---|---:|---:|
| Logistic Regression | 0.40 | 0.60 |
| Decision Tree | 0.30 | 0.70 |
| SVM | 0.20 | 0.80 |

Average probabilities:

$$
P(Class0)=\frac{0.40+0.30+0.20}{3}
$$

$$
=0.30
$$

and

$$
P(Class1)=\frac{0.60+0.70+0.80}{3}
$$

$$
=0.70
$$

Therefore:

$$
\boxed{Class1}
$$

---

# 8. Soft Voting Formula

Suppose we have $M$ classifiers.

Each classifier produces:

$$
P_i(y=k|x)
$$

The average probability for class $k$ is:

$$
P(y=k|x)
=
\frac{1}{M}
\sum_{i=1}^{M}P_i(y=k|x)
$$

Then choose the class with the highest probability:

$$
\boxed{
\hat y =
\arg\max_k
\frac{1}{M}
\sum_{i=1}^{M}P_i(y=k|x)
}
$$

---

# 9. Hard Voting vs Soft Voting

| Feature | Hard Voting | Soft Voting |
|---|---|---|
| Uses | Class labels | Probabilities |
| Requires `predict_proba()` | No | Yes |
| Considers confidence | ❌ | ✅ |
| Usually more informative | ❌ | ✅ |
| Sensitive to probability quality | Low | High |
| Common use | Simple ensembles | Probability-aware ensembles |

---

# 10. Why can Soft Voting be better?

Consider this example.

Three models predict:

```text
Model 1 → Class A with probability 0.51
Model 2 → Class A with probability 0.51
Model 3 → Class B with probability 0.99
```

Hard voting:

```text
A → 2 votes
B → 1 vote

Final → A
```

But look at the confidence.

Soft voting:

$$
P(A)=\frac{0.51+0.51+0.01}{3}
$$

$$
=0.343
$$

$$
P(B)=\frac{0.49+0.49+0.99}{3}
$$

$$
=0.657
$$

So:

$$
\boxed{B}
$$

Soft voting recognizes that Model 3 is **much more confident**.

---

# 11. Weighted Voting

Not all models are equally good.

Suppose:

- Logistic Regression → accuracy 85%
- Decision Tree → accuracy 70%
- SVM → accuracy 92%

Giving all models equal importance may not be ideal.

We can assign weights.

For example:

```text
Logistic Regression → weight 1
Decision Tree       → weight 0.5
SVM                 → weight 2
```

The stronger model gets more influence.

---

# 12. Weighted Hard Voting

Suppose:

```text
Model A → Class 1 → weight 2
Model B → Class 0 → weight 1
Model C → Class 1 → weight 3
```

Votes become:

```text
Class 1 → 2 + 3 = 5
Class 0 → 1
```

Therefore:

$$
\boxed{Class1}
$$

---

# 13. Weighted Soft Voting

Suppose:

| Model | Weight | P(Class 1) |
|---|---:|---:|
| LR | 1 | 0.60 |
| DT | 1 | 0.70 |
| SVM | 2 | 0.90 |

Weighted probability:

$$
P(Class1)
=
\frac{
1(0.60)+1(0.70)+2(0.90)
}{
1+1+2
}
$$

$$
=
\frac{0.60+0.70+1.80}{4}
$$

$$
=0.775
$$

So the weighted ensemble gives:

$$
\boxed{P(Class1)=0.775}
$$

---

# 14. Voting Ensemble in Scikit-Learn

Scikit-learn provides:

```python
VotingClassifier
```

Import:

```python
from sklearn.ensemble import VotingClassifier
```

Let's build one.

---

# 15. Dataset

We'll use the Iris dataset.

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

X, y = load_iris(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

---

# 16. Create Base Models

```python
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
```

Create models:

```python
lr = LogisticRegression(max_iter=1000)

dt = DecisionTreeClassifier(
    max_depth=5,
    random_state=42
)

svm = SVC(
    probability=True,
    random_state=42
)
```

Notice:

```python
probability=True
```

For soft voting, the estimator needs to provide probability estimates.

---

# 17. Hard Voting Implementation

```python
from sklearn.ensemble import VotingClassifier

voting_clf = VotingClassifier(
    estimators=[
        ("lr", lr),
        ("dt", dt),
        ("svm", svm)
    ],
    voting="hard"
)
```

Train:

```python
voting_clf.fit(X_train, y_train)
```

Predict:

```python
y_pred = voting_clf.predict(X_test)
```

Evaluate:

```python
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)

print(accuracy)
```

---

# 18. Soft Voting Implementation

Change:

```python
voting="hard"
```

to:

```python
voting="soft"
```

Complete:

```python
voting_clf = VotingClassifier(
    estimators=[
        ("lr", lr),
        ("dt", dt),
        ("svm", svm)
    ],
    voting="soft"
)

voting_clf.fit(X_train, y_train)

y_pred = voting_clf.predict(X_test)

print(accuracy_score(y_test, y_pred))
```

---

# 19. Weighted Voting in Scikit-Learn

We can specify:

```python
weights=[1, 1, 2]
```

For example:

```python
voting_clf = VotingClassifier(
    estimators=[
        ("lr", lr),
        ("dt", dt),
        ("svm", svm)
    ],
    voting="soft",
    weights=[1, 1, 2]
)
```

Here:

```text
Logistic Regression → 1
Decision Tree       → 1
SVM                 → 2
```

SVM therefore gets twice the influence.

---

# 20. Important `VotingClassifier` Parameters

### `estimators`

Defines the models.

```python
estimators=[
    ("lr", lr),
    ("dt", dt),
    ("svm", svm)
]
```

Each element is:

```text
(name, estimator)
```

---

### `voting`

Two options:

```python
voting="hard"
```

or:

```python
voting="soft"
```

Default:

```python
"hard"
```

---

### `weights`

Controls the importance of each model.

```python
weights=[1, 2, 3]
```

The order corresponds to the estimator order.

---

### `n_jobs`

Controls parallel processing where supported.

```python
VotingClassifier(
    estimators=[...],
    voting="soft",
    n_jobs=-1
)
```

`-1` generally means use all available CPU cores.

---

### `flatten_transform`

Relevant when using soft voting and `transform()`.

It controls the shape of the transformed probability output.

Usually you won't need to change it for normal `fit()` / `predict()` usage.

---

# 21. Comparing Individual Models vs Voting

A very important experiment is:

```python
models = {
    "Logistic Regression": lr,
    "Decision Tree": dt,
    "SVM": svm,
    "Voting": voting_clf
}
```

Then:

```python
for name, model in models.items():

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    score = accuracy_score(y_test, y_pred)

    print(name, score)
```

You might get something like:

```text
Logistic Regression    0.967
Decision Tree          0.933
SVM                    1.000
Voting                 0.967
```

The voting ensemble **does not have to outperform every individual model**.

This is an important point.

---

# 22. Very Important: Diversity

One of the most important concepts in ensemble learning is:

> **Good ensembles need diverse models.**

Suppose we have:

```text
Model 1 → Logistic Regression
Model 2 → Logistic Regression
Model 3 → Logistic Regression
```

They are likely to make very similar errors.

Combining them doesn't provide much benefit.

But:

```text
Logistic Regression
Decision Tree
KNN
SVM
Naive Bayes
```

learn patterns differently.

Their errors may be less correlated.

That makes voting more useful.

---

# 23. Accuracy Alone Isn't Enough

Suppose:

```text
Model A → 95% accuracy
Model B → 94% accuracy
Model C → 93% accuracy
```

You might think these are excellent ensemble candidates.

But if all three models make **exactly the same mistakes**, the ensemble gains little.

Instead, you might prefer:

```text
Model A → 95%
Model B → 92%
Model C → 90%
```

if their errors are substantially different.

Therefore:

$$
\boxed{\text{Model Quality + Model Diversity}}
$$

are both important.

---

# 24. Voting Ensemble for Regression

Voting isn't only for classification.

Scikit-learn provides:

```python
VotingRegressor
```

Instead of voting on classes, regression models' predictions are **averaged**.

Suppose:

```text
Linear Regression → 100
Decision Tree     → 110
Random Forest     → 105
```

Then:

$$
\hat y =
\frac{100+110+105}{3}
$$

$$
=105
$$

Final prediction:

$$
\boxed{105}
$$

---

# 25. Voting Regressor Example

```python
from sklearn.ensemble import VotingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

lr = LinearRegression()

dt = DecisionTreeRegressor(
    max_depth=5,
    random_state=42
)

rf = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)
```

Create ensemble:

```python
voting_reg = VotingRegressor(
    estimators=[
        ("lr", lr),
        ("dt", dt),
        ("rf", rf)
    ]
)
```

Train:

```python
voting_reg.fit(X_train, y_train)
```

Predict:

```python
y_pred = voting_reg.predict(X_test)
```

---

# 26. Weighted Voting Regressor

You can also assign weights.

```python
voting_reg = VotingRegressor(
    estimators=[
        ("lr", lr),
        ("dt", dt),
        ("rf", rf)
    ],
    weights=[1, 1, 2]
)
```

Random Forest gets twice the influence.

Conceptually:

$$
\hat y =
\frac{
1y_{LR}+1y_{DT}+2y_{RF}
}{
1+1+2
}
$$

---

# 27. Voting vs Bagging

These are different ensemble techniques.

### Voting

Train **different models**:

```text
LR
DT
SVM
KNN
```

and combine their predictions.

### Bagging

Train **multiple versions of the same type of model** on different bootstrap samples.

Example:

```text
Dataset
   ↓
Bootstrap Sample 1 → Decision Tree
Bootstrap Sample 2 → Decision Tree
Bootstrap Sample 3 → Decision Tree
Bootstrap Sample 4 → Decision Tree
          ↓
       Combine
```

Random Forest is a famous bagging-based algorithm.

---

# 28. Voting vs Random Forest

### Voting

```text
LR
DT
SVM
KNN
 ↓
Voting
```

Different algorithms.

### Random Forest

```text
Tree 1
Tree 2
Tree 3
Tree 4
Tree 5
...
 ↓
Majority Voting
```

Primarily many decision trees.

So:

$$
\boxed{\text{Voting = heterogeneous ensemble}}
$$

while Random Forest is primarily:

$$
\boxed{\text{homogeneous tree ensemble}}
$$

---
 
# 29. Voting vs Boosting

### Voting

Models are generally trained **independently**.

```text
Model 1 ──┐
Model 2 ──┼──> Voting
Model 3 ──┘
```

### Boosting

Models are trained **sequentially**, with later models focusing on errors made by previous models.

```text
Model 1
   ↓
Errors
   ↓
Model 2
   ↓
Errors
   ↓
Model 3
   ↓
Final Model
```

Examples:

- AdaBoost
- Gradient Boosting
- XGBoost
- LightGBM
- CatBoost

---

# 30. Voting vs Stacking

This is another very important distinction.

### Voting

Predictions are combined using a predefined rule:

```text
Predictions
    ↓
Majority / Average
    ↓
Final prediction
```

### Stacking

Predictions from base models become **features for another model**.

```text
             Input
               │
       ┌───────┼────────┐
       ↓       ↓        ↓
      LR       DT       SVM
       ↓       ↓        ↓
      P1       P2       P3
       └───────┼────────┘
               ↓
          Meta Model
               ↓
        Final Prediction
```

The meta-model learns **how to combine the base models**.

---

# 31. Voting Ensemble Example with Real Intuition

Suppose you're predicting whether a student will pass.

Features:

```text
study_hours
attendance
assignment_score
previous_marks
```

Models:

### Logistic Regression

Predicts:

```text
Pass
```

### Decision Tree

Predicts:

```text
Fail
```

### KNN

Predicts:

```text
Pass
```

### SVM

Predicts:

```text
Pass
```

### Naive Bayes

Predicts:

```text
Fail
```

Hard voting:

```text
Pass → 3
Fail → 2
```

Final:

```text
PASS
```

---

# 32. Soft Voting Example

Now suppose their probability predictions are:

| Model | Pass | Fail |
|---|---:|---:|
| LR | 0.80 | 0.20 |
| DT | 0.40 | 0.60 |
| KNN | 0.70 | 0.30 |
| SVM | 0.90 | 0.10 |
| NB | 0.45 | 0.55 |

Average:

$$
P(Pass)=
\frac{0.80+0.40+0.70+0.90+0.45}{5}
$$

$$
=0.65
$$

Therefore:

$$
P(Fail)=0.35
$$

Final:

$$
\boxed{Pass}
$$

---

# 33. When Should You Use Voting?

Voting is particularly useful when:

### 1. You have several good models

For example:

```text
LR → good
SVM → good
Random Forest → good
KNN → good
```

Combining them may improve robustness.

### 2. Models make different errors

This is probably the most important reason.

### 3. You want a simple ensemble

Voting is relatively easy to implement.

### 4. You don't want to train a meta-model

Unlike stacking, voting doesn't require learning a second-level model.

---

# 34. When Should You NOT Use Voting?

Voting isn't automatically better.

Avoid blindly adding models.

For example:

```text
Weak Model 1
Weak Model 2
Weak Model 3
Weak Model 4
Weak Model 5
```

Voting them together doesn't magically create a strong model.

Also, if all models are highly correlated:

```text
Model A ─┐
Model B ─┼─ same errors
Model C ─┘
```

the benefit can be small.

---

# 35. Probability Calibration and Soft Voting

This is an advanced but important concept.

Soft voting assumes that probability outputs are meaningful enough to average.

For example:

```text
Model A → 0.90
Model B → 0.60
Model C → 0.80
```

But a model's:

```text
0.90
```

doesn't necessarily mean that the event happens 90% of the time.

Some models produce poorly calibrated probabilities.

Therefore, soft voting can benefit from **probability calibration**.

Scikit-learn provides:

```python
CalibratedClassifierCV
```

This becomes especially important when:

- models produce very different probability scales
- probability estimates are important
- you are using weighted soft voting

---

# 36. A Good Practical Voting Ensemble

For a classification problem, you could start with:

```python
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
```

Then:

```python
lr = LogisticRegression(max_iter=1000)

dt = DecisionTreeClassifier(
    max_depth=5,
    random_state=42
)

rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

svm = SVC(
    probability=True,
    random_state=42
)

knn = KNeighborsClassifier(
    n_neighbors=5
)
```

Create ensemble:

```python
voting = VotingClassifier(
    estimators=[
        ("lr", lr),
        ("dt", dt),
        ("rf", rf),
        ("svm", svm),
        ("knn", knn)
    ],
    voting="soft",
    weights=[2, 1, 2, 2, 1],
    n_jobs=-1
)
```

Then:

```python
voting.fit(X_train, y_train)

y_pred = voting.predict(X_test)
```

Evaluate:

```python
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))
```

---

# 37. Important Practical Problem: Feature Scaling

Different models have different scaling requirements.

For example:

### Logistic Regression

Usually benefits from scaling.

### SVM

Usually **strongly benefits from scaling**.

### KNN

Usually **strongly benefits from scaling**.

### Decision Tree

Doesn't generally require scaling.

So if you combine:

```text
LR + SVM + KNN + DT
```

you should think carefully about preprocessing.

A good approach is to use separate pipelines:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

lr_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000))
])
```

Similarly for SVM and KNN.

This is often better than blindly scaling the entire dataset for every model.

---

# 38. Voting with Pipelines

Example:

```python
lr_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000))
])

svm_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model", SVC(probability=True))
])

knn_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model", KNeighborsClassifier())
])
```

Then:

```python
voting = VotingClassifier(
    estimators=[
        ("lr", lr_pipe),
        ("svm", svm_pipe),
        ("knn", knn_pipe),
        ("dt", dt)
    ],
    voting="soft"
)
```

This is a much more realistic production-style implementation.

---

# 39. Advantages of Voting Ensemble

### 1. Simple

Easy to understand and implement.

### 2. Combines different algorithms

You can combine:

```text
LR + DT + SVM + KNN
```

### 3. Can improve generalization

If models make complementary errors.

### 4. Reduces dependence on one model

One model's bad prediction may be overridden by others.

### 5. Flexible

Supports:

- hard voting
- soft voting
- weighted voting

---

# 40. Disadvantages

### 1. Computationally expensive

You have to train multiple models.

### 2. More memory

Multiple models need to be stored.

### 3. Doesn't guarantee improvement

The ensemble can perform worse than the best individual model.

### 4. Soft voting depends on probability quality

Bad probability estimates can hurt performance.

### 5. More complexity

Deployment and maintenance become more complicated.

---

# 41. The Core Idea to Remember

Think about Voting Ensemble like a group decision:

```text
              ┌── Logistic Regression ──┐
              │                         │
Input ────────┼── Decision Tree ────────┤
              │                         ├──> Voting → Final
              ├── SVM ──────────────────┤
              │                         │
              └── KNN ──────────────────┘
```

### Hard voting:

> **Which class did most models choose?**

### Soft voting:

> **Which class has the highest average probability?**

### Weighted voting:

> **Which class has the highest weighted vote/probability?**

---

# 42. Voting Ensemble vs Other Ensemble Methods

| Method | Base Models | Training | Combination |
|---|---|---|---|
| **Voting** | Different models | Parallel/independent | Vote/average |
| **Bagging** | Usually same model | Parallel | Average/vote |
| **Random Forest** | Decision Trees | Parallel | Majority vote |
| **Boosting** | Usually weak learners | Sequential | Weighted combination |
| **Stacking** | Different models | Base + meta-model | Meta-model |
| **Blending** | Different models | Base + holdout/meta-model | Meta-model |

A useful mental map is:

```text
                 Ensemble Learning
                       │
        ┌──────────────┼───────────────┐
        ↓              ↓               ↓
     Bagging         Boosting       Combining
        │              │               │
 Random Forest     AdaBoost          Voting
                  Gradient Boosting   Stacking
                  XGBoost             Blending
```

---

# 43. What You Should Study Next

Since you're going through **Ensemble Learning**, I'd study it in this order:

```text
1. Ensemble Learning
        ↓
2. Voting Ensemble          ← You are here
        ↓
3. Bagging
        ↓
4. Random Forest
        ↓
5. Boosting
        ↓
6. AdaBoost
        ↓
7. Gradient Boosting
        ↓
8. XGBoost
        ↓
9. LightGBM
        ↓
10. CatBoost
        ↓
11. Stacking
        ↓
12. Blending
        ↓
13. Bias-Variance + Ensemble connection
```

The **most important conceptual distinction** to have clear before moving on is:

$$
\boxed{
\text{Voting = combine predictions from independently trained models}
}
$$

$$
\boxed{
\text{Bagging = train many models on different bootstrap samples}
}
$$

$$
\boxed{
\text{Boosting = sequentially improve weak learners}
}
$$

$$
\boxed{
\text{Stacking = learn how to combine models using a meta-model}
}
$$

These four ideas form the foundation for understanding almost all practical ensemble methods.