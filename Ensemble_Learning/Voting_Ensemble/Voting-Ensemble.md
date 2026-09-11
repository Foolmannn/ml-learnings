
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
