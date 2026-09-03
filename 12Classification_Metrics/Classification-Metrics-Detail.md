

# Classification Metrics in Machine Learning — Detailed Guide

Classification metrics are used to evaluate **how well a classification model predicts categories/classes**.

For example:

- Spam vs Not Spam → Binary classification
- Disease vs No Disease → Binary classification
- Cat / Dog / Horse → Multiclass classification
- Multiple labels can be true simultaneously → Multilabel classification

The most important classification metrics are:

1. **Confusion Matrix**
2. **Accuracy**
3. **Precision**
4. **Recall / Sensitivity**
5. **Specificity**
6. **F1 Score**
7. **Fβ Score**
8. **ROC-AUC**
9. **PR-AUC**
10. **Log Loss / Cross-Entropy**
11. **Balanced Accuracy**
12. **MCC**
13. **Cohen's Kappa**

Let's build these from the ground up.

---

# 1. Start with the Confusion Matrix

Almost every classification metric starts with four quantities:

| | Actual Positive | Actual Negative |
|---|---:|---:|
| **Predicted Positive** | TP | FP |
| **Predicted Negative** | FN | TN |

Where:

### True Positive — TP

Model predicts **Positive**, and actual class is **Positive**.

Example:

> Model predicts a patient has a disease, and the patient actually has it.

---

### True Negative — TN

Model predicts **Negative**, and actual class is **Negative**.

> Model predicts the patient doesn't have the disease, and they actually don't.

---

### False Positive — FP

Model predicts **Positive**, but actual class is **Negative**.

This is also called a:

> **Type I Error**

Example:

> Model says disease exists, but patient is healthy.

---

### False Negative — FN

Model predicts **Negative**, but actual class is **Positive**.

This is:

> **Type II Error**

Example:

> Model says patient is healthy, but patient actually has the disease.

---

# 2. Example

Suppose we have 100 patients.

Our model produces:

- TP = 40
- TN = 50
- FP = 5
- FN = 5

So:

```text
                 Actual
              Positive Negative
Pred Positive     40       5
Pred Negative      5      50
```

Total:

$$
40+50+5+5=100
$$

Everything else follows from these four values.

---

# 3. Accuracy

Accuracy tells us:

> **Out of all predictions, how many were correct?**

Formula:

$$
Accuracy =
\frac{TP+TN}{TP+TN+FP+FN}
$$

For our example:

$$
Accuracy =
\frac{40+50}{100}
$$

$$
=0.90
$$

So:

$$
\boxed{Accuracy=90\%}
$$

### When is accuracy useful?

Accuracy works well when:

- Classes are reasonably balanced
- FP and FN have roughly similar importance

For example:

```text
Class A = 50%
Class B = 50%
```

Accuracy can be a good metric.

---

# 4. The Problem with Accuracy

Suppose we have 10,000 transactions:

```text
Fraud     = 100
Non-fraud = 9900
```

Imagine our model simply predicts:

```text
Every transaction = Non-fraud
```

Then:

$$
Accuracy=\frac{9900}{10000}=99\%
$$

It looks fantastic!

But the model detected:

$$
0
$$

fraudulent transactions.

So **99% accuracy is completely misleading**.

This is the classic:

> **Class imbalance problem**

That's why we need Precision, Recall, F1, etc.

---

# 5. Precision

Precision answers:

> **Of everything the model predicted as Positive, how many were actually Positive?**

Formula:

$$
Precision=
\frac{TP}{TP+FP}
$$

Notice the denominator:

$$
TP+FP
$$

which means:

> All predicted positives.

---

### Example

Suppose:

```text
TP = 80
FP = 20
```

Then:

$$
Precision=\frac{80}{80+20}
$$

$$
=0.8
$$

$$
\boxed{Precision=80\%}
$$

Interpretation:

> When the model predicts Positive, it is correct 80% of the time.

---

# 6. When Precision Matters

Precision matters when **False Positives are expensive**.

For example:

### Spam detection

Suppose your email classifier labels emails as spam.

A False Positive means:

> Important email → incorrectly marked as spam.

You don't want that.

Therefore, you want **high precision**.

---

### Search/recommendation systems

Suppose a search engine returns 10 results.

You want most returned results to actually be relevant.

Again:

$$
\boxed{\text{High Precision}}
$$

is desirable.

---

# 7. Recall

Recall answers:

> **Of all the actual Positive cases, how many did the model successfully detect?**

Formula:

$$
Recall=
\frac{TP}{TP+FN}
$$

The denominator:

$$
TP+FN
$$

represents:

> All actual positives.

---

### Example

Suppose:

```text
TP = 90
FN = 10
```

Then:

$$
Recall=
\frac{90}{90+10}
$$

$$
=0.9
$$

Therefore:

$$
\boxed{Recall=90\%}
$$

Interpretation:

> The model detected 90% of all actual positive cases.

---

# 8. When Recall Matters

Recall is extremely important when **False Negatives are expensive**.

For example:

### Disease detection

Suppose:

```text
100 patients actually have disease
```

Your model detects only:

```text
70
```

Then:

$$
Recall=70\%
$$

30 sick patients were missed.

That's potentially much worse than falsely alarming some healthy patients.

Therefore:

$$
\boxed{\text{Disease detection → prioritize Recall}}
$$

---

# 9. Precision vs Recall

This is one of the most important concepts in ML.

Imagine a model detecting terrorists, fraud, disease, spam, etc.

There is often a trade-off.

### Precision

> "When I say Positive, am I usually right?"

### Recall

> "Did I find most of the actual Positives?"

---

## Simple memory trick

### Precision

**Predicted Positive → How many correct?**

$$
Precision=\frac{TP}{TP+FP}
$$

Think:

> **Prediction quality**

---

### Recall

**Actual Positive → How many found?**

$$
Recall=\frac{TP}{TP+FN}
$$

Think:

> **Detection completeness**

---

# 10. Specificity

Specificity measures:

> **How well does the model identify actual Negative cases?**

Formula:

$$
Specificity=
\frac{TN}{TN+FP}
$$

The denominator:

$$
TN+FP
$$

represents:

> All actual negatives.

---

### Example

Suppose:

```text
TN = 90
FP = 10
```

Then:

$$
Specificity=
\frac{90}{90+10}
$$

$$
=90\%
$$

---

# 11. Relationship Between Specificity and FPR

False Positive Rate:

$$
FPR=
\frac{FP}{FP+TN}
$$

Since:

$$
Specificity=
\frac{TN}{TN+FP}
$$

we get:

$$
\boxed{FPR=1-Specificity}
$$

For example:

$$
Specificity=0.90
$$

then:

$$
FPR=1-0.90=0.10
$$

---

# 12. Sensitivity

Sensitivity is another name for:

$$
\boxed{Recall}
$$

So:

$$
Sensitivity=Recall=
\frac{TP}{TP+FN}
$$

You'll frequently see:

```text
Sensitivity = Recall = True Positive Rate
```

---

# 13. F1 Score

Precision and Recall can conflict.

F1 provides a single metric that balances both.

Formula:

$$
F1=
2\frac{Precision\times Recall}
{Precision+Recall}
$$

Or:

$$
\boxed{
F1=\frac{2TP}{2TP+FP+FN}
}
$$

---

### Example

Suppose:

$$
Precision=0.8
$$

and:

$$
Recall=0.6
$$

Then:

$$
F1=
2\frac{0.8\times0.6}{0.8+0.6}
$$

$$
=\frac{0.96}{1.4}
$$

$$
=0.686
$$

So:

$$
\boxed{F1\approx68.6\%}
$$

---

# 14. Why Harmonic Mean?

You might wonder why F1 doesn't use the normal arithmetic mean:

$$
\frac{Precision+Recall}{2}
$$

Instead it uses the harmonic mean.

The harmonic mean strongly penalizes imbalance.

For example:

```text
Precision = 1.0
Recall    = 0.1
```

Arithmetic mean:

$$
\frac{1+0.1}{2}=0.55
$$

But F1:

$$
F1=\frac{2(1)(0.1)}{1+0.1}
$$

$$
=0.182
$$

So F1 correctly tells us:

> Having excellent precision but terrible recall is not a balanced classifier.

---

# 15. Fβ Score

F1 gives equal importance to Precision and Recall.

But sometimes we want one to matter more.

That's where **Fβ** comes in.

$$
F_\beta =
(1+\beta^2)
\frac{Precision\times Recall}
{\beta^2 Precision+Recall}
$$

### β = 1

$$
F_1
$$

Equal importance.

---

### β > 1

Recall gets more importance.

For example:

$$
F_2
$$

prioritizes Recall.

Useful for:

> Disease detection, safety systems, fraud detection, etc.

---

### β < 1

Precision gets more importance.

For example:

$$
F_{0.5}
$$

prioritizes Precision.

---

# 16. ROC Curve

ROC stands for:

> **Receiver Operating Characteristic**

The ROC curve plots:

$$
TPR
$$

against:

$$
FPR
$$

where:

$$
TPR=\frac{TP}{TP+FN}
$$

and:

$$
FPR=\frac{FP}{FP+TN}
$$

So:

```text
Y-axis → True Positive Rate

1.0 |               ****
    |           ****
    |        ***
    |      **
    |    **
    |  **
0.0 |____________________
    0                  1
          False Positive Rate
```

---

# 17. Why Does ROC Use Thresholds?

Most classification models don't directly produce:

```text
Positive
Negative
```

They often produce a probability.

For example:

```text
Patient A → 0.91
Patient B → 0.72
Patient C → 0.43
Patient D → 0.21
```

Suppose threshold:

$$
0.5
$$

Then:

```text
0.91 → Positive
0.72 → Positive
0.43 → Negative
0.21 → Negative
```

But we could change threshold:

$$
0.8
$$

Then:

```text
0.91 → Positive
0.72 → Negative
0.43 → Negative
0.21 → Negative
```

Changing threshold changes:

- TP
- FP
- TN
- FN

Therefore it changes:

- Recall
- Precision
- FPR
- etc.

ROC examines model performance across many thresholds.

---

# 18. ROC-AUC

AUC means:

> **Area Under the Curve**

ROC-AUC measures how well the model separates the two classes across thresholds.

Range:

$$
0\leq AUC\leq1
$$

Typical interpretation:

| AUC | Interpretation |
|---:|---|
| 1.0 | Perfect |
| 0.9–1.0 | Excellent |
| 0.8–0.9 | Good |
| 0.7–0.8 | Fair |
| 0.5–0.7 | Weak |
| 0.5 | Random |
| < 0.5 | Worse than random |

The most important concept:

> **ROC-AUC measures ranking/separation ability rather than performance at one particular threshold.**

---

# 19. PR Curve

PR means:

> **Precision-Recall**

The PR curve plots:

$$
Precision
$$

against:

$$
Recall
$$

PR curves are particularly useful for:

> **Highly imbalanced datasets**

For example:

```text
Fraud     = 0.1%
Normal    = 99.9%
```

ROC-AUC can sometimes look deceptively good in such situations.

PR-AUC focuses more directly on the positive class.

---

# 20. PR-AUC

PR-AUC is the:

> Area under the Precision-Recall curve.

High PR-AUC generally means the model maintains good precision while achieving high recall across thresholds.

For heavily imbalanced classification:

$$
\boxed{\text{PR-AUC can be more informative than ROC-AUC}}
$$

---

# 21. Log Loss

So far, we've mostly evaluated hard predictions.

But classification models often produce probabilities.

Example:

```text
Actual = 1

Model A → 0.90
Model B → 0.60
Model C → 0.01
```

All three could potentially be classified as positive depending on threshold.

But clearly:

```text
0.90
```

is much better than:

```text
0.01
```

Log loss captures this.

For binary classification:

$$
Loss=
-\frac{1}{N}
\sum_{i=1}^{N}
[y_i\log(p_i)+(1-y_i)\log(1-p_i)]
$$

where:

- $y_i$ = actual label
- $p_i$ = predicted probability of positive class

---

# 22. Why Log Loss Is Powerful

Suppose actual:

$$
y=1
$$

Prediction A:

$$
p=0.9
$$

Loss:

$$
-\log(0.9)\approx0.105
$$

Prediction B:

$$
p=0.6
$$

Loss:

$$
-\log(0.6)\approx0.511
$$

Prediction C:

$$
p=0.01
$$

Loss:

$$
-\log(0.01)\approx4.605
$$

So log loss heavily penalizes:

> **Confident but wrong predictions.**

This is why probability quality matters.

---
