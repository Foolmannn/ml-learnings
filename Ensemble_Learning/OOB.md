# OOB Score (Out-of-Bag Score) in Random Forest

**OOB score** is a built-in validation technique in Random Forest that lets us estimate how well the model will perform on unseen data **without explicitly creating a validation set or using K-Fold Cross-Validation**.

The key idea is:

> Every Random Forest tree is trained on a bootstrap sample, and the samples not selected for that tree are called **Out-of-Bag (OOB) samples**. Those OOB samples can be used to test that tree.

---

## 1. First understand Bootstrap Sampling

Suppose our training dataset contains:

```text
A B C D E F G H I J
```

We want to build one tree.

Random Forest samples **with replacement**:

```text
A C C E F F H I J
```

Notice:

* `C` appears twice
* `F` appears twice
* `B`, `D`, `G` are not selected

Therefore:

```text
Selected for training:
A C C E F F H I J

OOB samples:
B D G
```

`B`, `D`, and `G` are **Out-of-Bag samples for this tree**.

---

# 2. Why do OOB samples exist?

Because bootstrap sampling happens **with replacement**.

Suppose we have \(n\) samples and randomly draw \(n\) times.

An individual sample has probability:

$$
\frac{1}{n}
$$

of being selected on each draw.

Therefore, probability that it is **not** selected in one draw:

$$
1-\frac{1}{n}
$$

After \(n\) draws:

$$
P(\text{not selected})
=
\left(1-\frac{1}{n}\right)^n
$$

As \(n\) becomes large:

$$
\left(1-\frac{1}{n}\right)^n
\rightarrow e^{-1}
$$

and:

$$
e^{-1}\approx0.368
$$

So approximately:

$$
\boxed{36.8\%}
$$

of the samples are OOB for each tree.

---

# 3. How OOB evaluation works

Suppose we build 5 trees.

```text
              Dataset
                 |
      ┌──────────┼──────────┐
      ↓          ↓          ↓
   Tree 1      Tree 2     Tree 3
      ↓          ↓          ↓
   OOB data    OOB data   OOB data
```

Consider sample `A`.

It might be:

```text
Tree 1 → A was used for training
Tree 2 → A was NOT used
Tree 3 → A was used
Tree 4 → A was NOT used
Tree 5 → A was NOT used
```

So only Trees 2, 4, and 5 can make an OOB prediction for `A`.

Suppose:

```text
Tree 2 → Class 1
Tree 4 → Class 1
Tree 5 → Class 0
```

Majority vote:

```text
Class 1 → 2 votes
Class 0 → 1 vote
```

Therefore:

```text
OOB prediction for A = Class 1
```

We compare this with the actual class of `A`.

This process is repeated for all training samples.

Finally:

$$
OOB\ Score =
\frac{\text{correct OOB predictions}}
{\text{total OOB predictions}}
$$

for classification.

---

# 4. Simple example

Suppose we have 10 training samples:

```text
A B C D E F G H I J
```

After training the forest, suppose the OOB predictions are:

| Sample | Actual | OOB Prediction |
| ------ | -----: | -------------: |
| A      |      1 |              1 |
| B      |      0 |              0 |
| C      |      1 |              0 |
| D      |      1 |              1 |
| E      |      0 |              0 |
| F      |      1 |              1 |
| G      |      0 |              1 |
| H      |      1 |              1 |
| I      |      0 |              0 |
| J      |      1 |              1 |

Correct:

```text
A ✓
B ✓
C ✗
D ✓
E ✓
F ✓
G ✗
H ✓
I ✓
J ✓
```

8 out of 10 are correct.

Therefore:

$$
OOB\ Score = \frac{8}{10}=0.80
$$

So:

```text
OOB Score = 80%
```

---

# 5. OOB Score in Scikit-Learn

For a classifier:

```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=200,
    bootstrap=True,
    oob_score=True,
    random_state=42
)

rf.fit(X_train, y_train)

print(rf.oob_score_)
```

Example output:

```text
0.91
```

Meaning approximately:

```text
OOB accuracy = 91%
```

For regression, `oob_score_` uses a regression scoring measure rather than classification accuracy; by default, this is typically \(R^2\).

---

# 6. Why `bootstrap=True` is important

OOB evaluation depends on having samples that were **not selected** for each tree.

Therefore you need:

```python
bootstrap=True
```

For example:

```python
rf = RandomForestClassifier(
    n_estimators=200,
    bootstrap=True,
    oob_score=True,
    random_state=42
)
```

If you don't use bootstrap sampling, the traditional OOB mechanism isn't available.

---

# 7. OOB Score vs Train Accuracy

This distinction is very important.

Suppose:

```text
Training Accuracy = 100%
OOB Score         = 91%
```

The 100% training accuracy isn't surprising because Random Forest trees can become very complex.

The OOB score is more useful for estimating generalization performance.

Think:

```text
Training Accuracy
       ↓
Performance on data used to train trees

OOB Score
       ↓
Performance on samples that were not used
for the corresponding trees
```

Therefore OOB score is usually much more informative than simply looking at training accuracy.

---

# 8. OOB Score vs Test Score

Suppose:

```text
OOB Score  = 0.91
Test Score = 0.90
```

That's reassuring.

The OOB estimate and final test performance are fairly close.

But don't confuse them:

### OOB score

Uses the training dataset's unused bootstrap observations.

### Test score

Uses a completely held-out dataset that wasn't used during model training.

A proper final evaluation should still use a **separate test set**.

---

# 9. OOB vs Cross-Validation

Since you've been studying `cross_val_score()` and `GridSearchCV`, this comparison is important.

### K-Fold Cross-Validation

Suppose:

```text
5-fold CV
```

The dataset is divided into 5 parts:

```text
Fold 1 → validation
Fold 2 → validation
Fold 3 → validation
Fold 4 → validation
Fold 5 → validation
```

The model is trained repeatedly.

---

### OOB

Random Forest naturally creates its own validation-like samples:

```text
Bootstrap sample
      ↓
Train Tree
      ↓
Unused samples
      ↓
OOB evaluation
```

So:

```text
Cross-validation
→ Explicitly creates folds

OOB
→ Validation samples naturally arise
   from bootstrap sampling
```

---

# 10. Why OOB is useful

Imagine you have:

```text
100,000 samples
```

and you're training a Random Forest.

Doing 5-fold cross-validation means training many models repeatedly.

OOB evaluation comes almost "for free" as part of the Random Forest's bootstrap process.

So it can be:

* convenient
* computationally efficient
* useful for estimating generalization performance

---

# 11. Does more trees improve OOB score?

Usually, as you increase:

```python
n_estimators
```

the OOB estimate becomes more stable.

For example:

```text
10 trees  → OOB = 0.87
50 trees  → OOB = 0.90
100 trees → OOB = 0.91
500 trees → OOB = 0.912
```

Eventually, adding more trees gives diminishing returns.

This is one reason we generally don't need to worry about adding more trees causing the same kind of overfitting associated with making individual trees deeper.

---

# 12. OOB Decision Function

For classification, you can also inspect:

```python
rf.oob_decision_function_
```

Example:

```python
print(rf.oob_decision_function_)
```

You might get:

```text
[[0.10 0.90]
 [0.80 0.20]
 [0.25 0.75]
 ...]
```

For a binary classifier:

```text
Column 0 → probability of Class 0
Column 1 → probability of Class 1
```

These probabilities are generated from the trees for which each sample was OOB.

---

# 13. OOB Score with Random Forest Regression

The same concept works for regression.

Suppose:

```text
Actual = 100
OOB predictions:
Tree 1 → 95
Tree 2 → 103
Tree 3 → 98
Tree 4 → 101
```

Average:

$$
\frac{95+103+98+101}{4}
=99.25
$$

That becomes the OOB prediction for that sample.

After generating OOB predictions for all samples, the model calculates a regression score.

For example:

```python
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(
    n_estimators=200,
    bootstrap=True,
    oob_score=True,
    random_state=42
)

rf.fit(X_train, y_train)

print(rf.oob_score_)
```

---

# 14. Important limitation

OOB score is **not a replacement for a final test set**.

A good workflow is:

```text
Complete Dataset
       |
       ↓
Train/Test Split
       |
       ├───────────────┐
       ↓               ↓
   X_train          X_test
       ↓               ↓
Random Forest       Final evaluation
       ↓
   OOB Score
```

Use OOB during model development, but keep your test set untouched until you're ready for final evaluation.

---

# 15. OOB for hyperparameter tuning

You can even use OOB score while experimenting with Random Forest.

For example:

```python
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=15,
    max_features="sqrt",
    bootstrap=True,
    oob_score=True,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)

print("OOB Score:", rf.oob_score_)
```

However, if you're systematically tuning many hyperparameters, `GridSearchCV` or `RandomizedSearchCV` is generally the more appropriate tool for the search.

---

# 16. One important distinction

Don't think:

> "OOB means 36.8% of the entire dataset is my validation set."

That's not exactly what happens.

For **each tree**, approximately 36.8% of observations are OOB.

But different trees have different OOB observations.

For example:

```text
Tree 1:
OOB → A B D F

Tree 2:
OOB → B C E H

Tree 3:
OOB → A D E G
```

A particular sample gets predictions from **the trees for which it was OOB**.

Those predictions are then aggregated.

---

# 17. The complete picture

```text
                 Training Dataset
                        |
             Bootstrap Sampling
                        |
        ┌───────────────┼───────────────┐
        ↓               ↓               ↓
     Tree 1          Tree 2          Tree 3
        ↓               ↓               ↓
   OOB samples     OOB samples     OOB samples
        ↓               ↓               ↓
   Prediction      Prediction      Prediction
        └───────────────┼───────────────┘
                        ↓
                  OOB Prediction
                        ↓
               Compare with Actual
                        ↓
                   OOB Score
```

---

## ⭐ What you should remember

For your ML notes, remember this:

> **OOB (Out-of-Bag) score is an internal validation estimate used by bagging-based Random Forest models. Because each tree is trained on a bootstrap sample, the observations not selected for that tree are OOB observations. The forest uses those observations to make predictions and compares them with their actual values.**

The important relationship is:

$$
\boxed{
\text{Bootstrap Sampling}
\rightarrow
\text{OOB Samples}
\rightarrow
\text{OOB Predictions}
\rightarrow
\text{OOB Score}
}
$$

And the approximate bootstrap fact:

$$
\boxed{\text{OOB samples per tree} \approx 36.8\%}
$$

The **most important difference** to keep in your head is:

```text
Train score → data used for training
OOB score   → bootstrap-excluded training samples
CV score    → explicitly created validation folds
Test score  → completely held-out final evaluation
```
