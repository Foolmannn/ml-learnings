# Handling Imbalanced Data in Machine Learning — In Detail

**Imbalanced data** occurs when the classes in a classification dataset are not represented equally.

For example, suppose we are building a fraud-detection model:

| Class  | Transactions |
| ------ | -----------: |
| Normal |        9,900 |
| Fraud  |          100 |

Here:

* Normal = **99%**
* Fraud = **1%**

This is a **highly imbalanced dataset**.

The major problem is that a model can achieve very high accuracy while performing terribly on the minority class.

---

# 1. What is Imbalanced Data?

In a classification problem, if one class has significantly more observations than another class, the dataset is called **imbalanced**.

### Balanced dataset

Suppose we have:

```text
Class 0 → 500 samples
Class 1 → 500 samples
```

Ratio:

```text
1 : 1
```

This is balanced.

### Imbalanced dataset

```text
Class 0 → 950 samples
Class 1 → 50 samples
```

Ratio:

```text
19 : 1
```

This is imbalanced.

Another example:

```text
Class 0 → 9,900
Class 1 → 100
```

Ratio:

```text
99 : 1
```

This is **severely imbalanced**.

---

# 2. Why Does Imbalanced Data Matter?

The biggest issue is that the model may become **biased toward the majority class**.

Consider:

```text
10,000 transactions

9,900 → Normal
100   → Fraud
```

Imagine a model that simply predicts:

```text
Every transaction → Normal
```

Its accuracy would be:

$$
Accuracy = \frac{9900}{10000}=99\%
$$

It looks excellent.

But:

```text
Actual fraud     = 100
Predicted fraud  = 0
```

The model detects **none of the fraud cases**.

So accuracy is misleading.

---

# 3. Example: Disease Detection

Suppose:

```text
10,000 people

9,900 → Healthy
100   → Disease
```

A model predicts:

```text
Everyone → Healthy
```

Accuracy:

$$
\frac{9900}{10000}=99\%
$$

But the model identifies:

```text
Disease cases detected = 0
```

For a disease-detection system, this is obviously a serious problem.

This is why **handling imbalance is not simply about improving accuracy**.

---

# 4. Types of Imbalanced Data

## 4.1 Binary Imbalance

There are two classes.

Example:

```text
Class 0 → 9,000
Class 1 → 1,000
```

Applications:

* Fraud detection
* Spam detection
* Disease detection
* Customer churn
* Loan default

---

## 4.2 Multiclass Imbalance

There are more than two classes.

Example:

```text
Class A → 8,000
Class B → 1,500
Class C → 400
Class D → 100
```

Here, multiple minority classes exist.

---

## 4.3 Extreme Imbalance

Sometimes the minority class is extremely rare.

Example:

```text
Normal → 999,000
Fraud  → 1,000
```

Ratio:

$$
999:1
$$

This is common in:

* Cybersecurity
* Financial fraud
* Manufacturing defects
* Rare diseases

---

# 5. First Step: Detect Class Imbalance

Before handling imbalance, determine the class distribution.

## Using Pandas

```python
import pandas as pd

df["target"].value_counts()
```

Example:

```text
0    9500
1     500
Name: target
```

Percentage:

```python
df["target"].value_counts(normalize=True) * 100
```

Output:

```text
0    95%
1     5%
```

You can also visualize it:

```python
import matplotlib.pyplot as plt

df["target"].value_counts().plot(kind="bar")

plt.xlabel("Class")
plt.ylabel("Number of Samples")
plt.title("Class Distribution")
plt.show()
```

---

# 6. Important Metrics for Imbalanced Data

Before discussing techniques, you need to understand the evaluation metrics.

For binary classification, we have:

```text
                 Predicted
                0       1
Actual  0      TN      FP
        1      FN      TP
```

Where:

### True Positive — TP

Actual positive and predicted positive.

### True Negative — TN

Actual negative and predicted negative.

### False Positive — FP

Actual negative but predicted positive.

### False Negative — FN

Actual positive but predicted negative.

---

# 7. Accuracy

$$
Accuracy = \frac{TP+TN}{TP+TN+FP+FN}
$$

Accuracy works well when classes are reasonably balanced.

But with imbalanced data, it can be misleading.

Example:

```text
TN = 9900
TP = 0
FP = 0
FN = 100
```

Accuracy:

$$
\frac{9900}{10000}=99\%
$$

But the model completely fails at detecting the minority class.

---

# 8. Precision

Precision answers:

> Of everything the model predicted as positive, how many were actually positive?

$$
Precision=\frac{TP}{TP+FP}
$$

High precision means **few false positives**.

Example:

```text
Model predicts 100 transactions as fraud.

80 are actually fraud.
20 are normal.
```

Then:

$$
Precision=\frac{80}{80+20}=80\%
$$

Precision is important when false positives are expensive.

---

# 9. Recall

Recall answers:

> Of all actual positive cases, how many did the model find?

$$
Recall=\frac{TP}{TP+FN}
$$

High recall means **few false negatives**.

Example:

```text
100 actual fraud cases
Model detects 90
```

Then:

$$
Recall=\frac{90}{90+10}=90\%
$$

Recall is particularly important when missing a positive case is costly.

Examples:

* Disease detection
* Fraud detection
* Safety failure detection

---

# 10. F1 Score

F1 combines precision and recall.

$$
F1=2\times\frac{Precision\times Recall}
{Precision+Recall}
$$

It is the **harmonic mean** of precision and recall.

For example:

```text
Precision = 0.80
Recall    = 0.90
```

Then:

$$
F1=2\times\frac{0.8\times0.9}{0.8+0.9}
$$

$$
F1=0.847
$$

F1 is useful when you need a balance between precision and recall.

---

# 11. Specificity

Specificity measures how well the model identifies negative cases.

$$
Specificity=\frac{TN}{TN+FP}
$$

It is also called:

$$
True\ Negative\ Rate
$$

Recall is sometimes called:

$$
Sensitivity = \frac{TP}{TP+FN}
$$

Therefore:

```text
Sensitivity = Recall
Specificity = True Negative Rate
```

---

# 12. ROC-AUC

ROC curve plots:

$$
TPR \text{ vs } FPR
$$

where:

$$
TPR = \frac{TP}{TP+FN}
$$

and:

$$
FPR = \frac{FP}{FP+TN}
$$

ROC-AUC measures how well the model separates positive and negative classes across thresholds.

Generally:

```text
AUC = 1.0 → perfect separation
AUC = 0.5 → random-like ranking
```

However, with **very rare positive classes**, ROC-AUC can sometimes look reassuring even when the practical number of false positives is problematic.

---

# 13. Precision-Recall Curve

For highly imbalanced classification, the **Precision-Recall curve** is often particularly informative.

It examines the trade-off between:

```text
Precision
     ↓
Recall
```

This is especially useful when the positive class is rare.

Example applications:

```text
Fraud detection
Disease detection
Defect detection
Spam detection
```

---

# 14. Main Techniques for Handling Imbalanced Data

There are several approaches:

```text
                    Handling Imbalance
                           │
          ┌────────────────┼────────────────┐
          ↓                ↓                ↓
      Resampling       Algorithmic       Threshold
          │             methods            tuning
      ┌───┴───┐             │                │
      ↓       ↓             ↓                ↓
   Oversample Undersample Class weights   Change
      │          │          │              threshold
      ↓          ↓          ↓
    SMOTE      Random      Balanced
              sampling     models
```

Let's examine them in detail.

---

# 15. Method 1 — Random Undersampling

**Undersampling** reduces the number of majority-class samples.

Suppose:

```text
Majority → 9,000
Minority → 1,000
```

We could randomly remove majority samples:

```text
Majority → 1,000
Minority → 1,000
```

Now:

```text
1 : 1
```

---

## Implementation

Using `imbalanced-learn`:

```python
from imblearn.under_sampling import RandomUnderSampler

rus = RandomUnderSampler(random_state=42)

X_resampled, y_resampled = rus.fit_resample(X_train, y_train)
```

Check:

```python
print(y_resampled.value_counts())
```

---

## Advantages

* Simple
* Fast
* Reduces dataset size
* Can help models focus on minority class

---

## Disadvantages

The major problem is:

> You are throwing away information.

Suppose:

```text
9,000 majority samples
```

and you keep only:

```text
1,000
```

You discarded 8,000 potentially useful examples.

---

# 16. Method 2 — Random Oversampling

Instead of removing majority samples, we increase the minority class.

Original:

```text
Majority → 9,000
Minority → 1,000
```

After oversampling:

```text
Majority → 9,000
Minority → 9,000
```

The simplest approach duplicates minority samples.

---

## Implementation

```python
from imblearn.over_sampling import RandomOverSampler

ros = RandomOverSampler(random_state=42)

X_resampled, y_resampled = ros.fit_resample(X_train, y_train)
```

Check:

```python
print(y_resampled.value_counts())
```

---

## Advantage

You don't lose majority-class information.

---

## Disadvantage

Because minority examples are duplicated, the model may **overfit** to them.

For example:

```text
Original minority:

A
B
C
D
```

Oversampling might produce:

```text
A
B
C
D
A
B
C
D
A
B
C
D
```

The model sees the same observations repeatedly.

---

# 17. Method 3 — SMOTE

SMOTE is one of the most important techniques for imbalanced data.

**SMOTE = Synthetic Minority Over-sampling Technique**

Instead of simply copying minority samples, SMOTE generates **synthetic minority observations**.

---

# 18. How SMOTE Works

Suppose we have minority observations:

```text
A
B
C
```

SMOTE finds neighboring minority samples.

It then creates a new point somewhere between them.

Conceptually:

```text
A -------- New Sample -------- B
```

The synthetic sample is generated using:

$$
x_{new}=x_i+\lambda(x_{neighbor}-x_i)
$$

where:

$$
0\leq\lambda\leq1
$$

For example:

```text
x_i       = 10
x_neighbor = 20
λ         = 0.5
```

Then:

$$
x_{new}=10+0.5(20-10)
$$

$$
x_{new}=15
$$

So instead of copying 10 or 20, SMOTE creates a new synthetic point.

---

# 19. SMOTE Example

Original:

```text
Majority → 900
Minority → 100
```

After SMOTE:

```text
Majority → 900
Minority → 900
```

The additional 800 minority observations are synthetically generated.

---

# 20. SMOTE Implementation

Install:

```bash
pip install imbalanced-learn
```

Then:

```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)

X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
```

Check:

```python
print(y_resampled.value_counts())
```

---

# 21. Why SMOTE Can Be Better Than Random Oversampling

Random oversampling:

```text
A
B
C
A
B
C
A
B
C
```

SMOTE:

```text
A
B
C
A'
B'
C'
D'
...
```

where:

```text
A', B', C' ...
```

are synthetic samples.

Therefore, SMOTE can reduce the direct duplication of minority examples.

However, synthetic samples are not automatically realistic. SMOTE can perform poorly when minority samples overlap heavily with the majority class or when the minority distribution has unusual structure.

---

# 22. Important SMOTE Rule: Apply It Only to Training Data

This is extremely important.

### Wrong

```python
X_resampled, y_resampled = smote.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(
    X_resampled,
    y_resampled
)
```

This can cause **data leakage**.

Why?

Because information derived from the full dataset, including what should have been held out for testing, influences the synthetic data.

---

## Correct

First split:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

Then apply SMOTE:

```python
smote = SMOTE(random_state=42)

X_train_resampled, y_train_resampled = smote.fit_resample(
    X_train,
    y_train
)
```

Keep the test set untouched.

---

# 23. Why `stratify=y`?

When splitting imbalanced data, use:

```python
stratify=y
```

This attempts to preserve the original class proportions in train and test sets.

For example:

Original:

```text
Class 0 → 90%
Class 1 → 10%
```

With stratification:

```text
Training:
Class 0 → ~90%
Class 1 → ~10%

Testing:
Class 0 → ~90%
Class 1 → ~10%
```

---

# 24. Method 4 — Borderline-SMOTE

Regular SMOTE generates synthetic samples around minority observations.

But not all minority observations are equally difficult.

**Borderline-SMOTE** focuses on minority observations near the **decision boundary**.

Conceptually:

```text
Majority        Majority
   ○ ○ ○ ○ ○
      ○  ●
    ● ● ●
      ●
   Minority
```

The minority points near the boundary are more difficult to classify.

Borderline-SMOTE generates more synthetic samples around these difficult areas.

---

# 25. Method 5 — SMOTE + Tomek Links

Another approach is:

```text
SMOTE
   ↓
Generate minority samples
   ↓
Tomek Links
   ↓
Remove problematic overlapping samples
```

Tomek links identify pairs of observations from different classes that are very close to each other.

This can help clean class overlap after oversampling.

Implementation:

```python
from imblearn.combine import SMOTETomek

smote_tomek = SMOTETomek(random_state=42)

X_resampled, y_resampled = smote_tomek.fit_resample(
    X_train,
    y_train
)
```

---

# 26. Method 6 — SMOTEENN

Another combination is:

```text
SMOTE + ENN
```

ENN stands for:

**Edited Nearest Neighbours**

It can remove noisy or ambiguous observations after oversampling.

```python
from imblearn.combine import SMOTEENN

smote_enn = SMOTEENN(random_state=42)

X_resampled, y_resampled = smote_enn.fit_resample(
    X_train,
    y_train
)
```

---
