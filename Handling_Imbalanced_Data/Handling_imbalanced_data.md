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
