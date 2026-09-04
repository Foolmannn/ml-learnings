
# ROC Curve and ROC-AUC in Machine Learning

ROC Curve and ROC-AUC are especially important for **binary classification** problems such as:

- Spam vs. Not Spam
- Disease vs. Healthy
- Fraud vs. Legitimate
- Churn vs. No Churn
- Landslide vs. No Landslide

The key idea is:

> **ROC tells us how a classifier's True Positive Rate changes as we vary the classification threshold, while ROC-AUC summarizes this performance into a single number.**

---

# 1. Why do we need ROC?

Suppose we train a logistic regression model to predict whether a patient has a disease.

The model doesn't necessarily output:

```text
Disease
```

or

```text
No Disease
```

Instead, it might output a probability:

```text
P(Disease) = 0.87
```

For example:

| Patient | Actual | Predicted probability |
|---|---:|---:|
| A | 1 | 0.95 |
| B | 1 | 0.82 |
| C | 0 | 0.71 |
| D | 0 | 0.40 |
| E | 1 | 0.35 |
| F | 0 | 0.10 |

We need a **threshold** to convert probabilities into classes.

Usually:

$$
threshold=0.5
$$

So:

$$
P(y=1)\geq0.5 \Rightarrow \hat y=1
$$

and

$$
P(y=1)<0.5 \Rightarrow \hat y=0
$$

But why specifically 0.5?

There is nothing magical about 0.5.

Depending on the application, we might use:

$$
0.2,\;0.3,\;0.5,\;0.7,\;0.9
$$

ROC analysis examines what happens when we change this threshold.

---

# 2. Confusion Matrix

Before understanding ROC, we need the confusion matrix.

For binary classification:

| | Actual Positive | Actual Negative |
|---|---:|---:|
| **Predicted Positive** | TP | FP |
| **Predicted Negative** | FN | TN |

Where:

### True Positive (TP)

Actual = Positive

Prediction = Positive

Example:

> Patient has disease → model predicts disease.

---

### True Negative (TN)

Actual = Negative

Prediction = Negative

Example:

> Patient is healthy → model predicts healthy.

---

### False Positive (FP)

Actual = Negative

Prediction = Positive.

Example:

> Patient is healthy → model predicts disease.

Also called:

**Type I error**

---

### False Negative (FN)

Actual = Positive

Prediction = Negative.

Example:

> Patient has disease → model predicts healthy.

Also called:

**Type II error**

---

# 3. True Positive Rate — TPR

The first important ROC quantity is:

$$
TPR=\frac{TP}{TP+FN}
$$

It is also called:

- Recall
- Sensitivity
- True Positive Rate

It answers:

> **Of all actual positive examples, how many did the model correctly identify?**

For example:

Suppose:

$$
TP=80
$$

and

$$
FN=20
$$

Then:

$$
TPR=\frac{80}{80+20}
$$

$$
TPR=0.8
$$

So:

$$
TPR=80\%
$$

The model detected 80% of the actual positives.

---

# 4. False Positive Rate — FPR

The second important quantity is:

$$
FPR=\frac{FP}{FP+TN}
$$

It answers:

> **Of all actual negative examples, how many were incorrectly classified as positive?**

Suppose:

$$
FP=10
$$

and

$$
TN=90
$$

Then:

$$
FPR=\frac{10}{10+90}
$$

$$
FPR=0.1
$$

Therefore:

$$
FPR=10\%
$$

---

# 5. ROC Curve

The ROC curve plots:

$$
\boxed{TPR\;vs.\;FPR}
$$

Specifically:

$$
\boxed{x=FPR}
$$

and

$$
\boxed{y=TPR}
$$

So:

> **X-axis = False Positive Rate**

> **Y-axis = True Positive Rate**



The important question is:

> How do we obtain multiple points on the ROC curve?

By changing the **classification threshold**.

---

# 6. Threshold and ROC

Suppose our model produces:

| Sample | Actual | Probability |
|---|---:|---:|
| A | 1 | 0.95 |
| B | 1 | 0.85 |
| C | 0 | 0.75 |
| D | 1 | 0.65 |
| E | 0 | 0.55 |
| F | 0 | 0.30 |
| G | 1 | 0.20 |
| H | 0 | 0.10 |

Let's change the threshold.

---

## Threshold = 0.9

Only A is predicted positive.

```text
A → Positive
B → Negative
C → Negative
D → Negative
...
```

Therefore:

$$
TP=1
$$

$$
FN=3
$$

$$
FP=0
$$

$$
TN=4
$$

Thus:

$$
TPR=\frac{1}{4}=0.25
$$

$$
FPR=\frac{0}{4}=0
$$

ROC point:

$$
(0,0.25)
$$

---

## Threshold = 0.7

A, B and C become positive.

Actual:

```text
A = 1
B = 1
C = 0
```

Therefore:

$$
TP=2
$$

$$
FP=1
$$

$$
FN=2
$$

$$
TN=3
$$

So:

$$
TPR=\frac{2}{4}=0.5
$$

and

$$
FPR=\frac{1}{4}=0.25
$$

ROC point:

$$
(0.25,0.5)
$$

---

## Threshold = 0.5

A, B, C, D and E become positive.

Then:

$$
TP=3
$$

$$
FP=2
$$

$$
FN=1
$$

$$
TN=2
$$

Therefore:

$$
TPR=\frac{3}{4}=0.75
$$

and:

$$
FPR=\frac{2}{4}=0.5
$$

ROC point:

$$
(0.5,0.75)
$$

---

# 7. What happens when threshold decreases?

This is extremely important.

Imagine:

$$
threshold=0.9
$$

Only very confident predictions are positive.

So there are:

- Few predicted positives
- Few TP
- Few FP

Therefore:

$$
TPR \downarrow
$$

and

$$
FPR \downarrow
$$

Now lower the threshold:

$$
0.9\rightarrow0.7\rightarrow0.5\rightarrow0.3\rightarrow0.1
$$

More examples become positive.

Consequently:

$$
TPR\uparrow
$$

but generally:

$$
FPR\uparrow
$$

This creates the **trade-off** that the ROC curve represents.

---

# 8. Why is the ROC curve called a curve?

We calculate:

$$
TPR
$$

and

$$
FPR
$$

at many different thresholds.

For example:

| Threshold | TPR | FPR |
|---:|---:|---:|
| 1.0 | 0.00 | 0.00 |
| 0.9 | 0.25 | 0.00 |
| 0.7 | 0.50 | 0.25 |
| 0.5 | 0.75 | 0.50 |
| 0.3 | 1.00 | 0.75 |
| 0.0 | 1.00 | 1.00 |

Plot these points:

$$
(FPR,TPR)
$$

and connect them.

That's the ROC curve.

---

# 9. Perfect Classifier

Consider a perfect classifier.

It can identify every positive and negative correctly.

At an appropriate threshold:

$$
TPR=1
$$

and:

$$
FPR=0
$$

Therefore the ROC curve reaches:

$$
(0,1)
$$

This is the ideal point.

```text
TPR
 1 |     ●────────
   |    /
   |   /
   |  /
   | /
 0 |●────────────
   +---------------- FPR
   0               1
```

A perfect classifier has:

$$
\boxed{AUC=1}
$$

---

# 10. Random Classifier

Suppose the model is essentially guessing randomly.

Then:

$$
TPR\approx FPR
$$

So the ROC curve follows approximately:

$$
y=x
$$

This is called the:

**No-skill line**

or

**random classifier line**.

Its AUC is:

$$
\boxed{0.5}
$$

---

# 11. Bad Classifier

A classifier performing worse than random can have:

$$
AUC<0.5
$$

For example:

$$
AUC=0.3
$$

Interestingly, if the predictions are simply reversed, the model could potentially perform well.

For example:

```text
Model says Positive → treat as Negative
Model says Negative → treat as Positive
```

This is why AUC below 0.5 often indicates that the model's ranking direction may be reversed.

---

# 12. ROC-AUC

Now we arrive at the second major concept.

**ROC-AUC** means:

> **Area Under the Receiver Operating Characteristic Curve**

Mathematically:

$$
AUC=\int_0^1 TPR(FPR)\,d(FPR)
$$

You don't normally calculate this integral manually.

Libraries calculate it from the model's predictions.

---

# 13. What does AUC actually mean?

This is one of the most important interpretations.

Suppose:

$$
AUC=0.85
$$

A useful interpretation is:

> There is approximately an **85% probability that the model ranks a randomly selected positive example higher than a randomly selected negative example**.

This is a very powerful interpretation.

For example:

```text
Random positive:
Disease patient → probability = 0.83

Random negative:
Healthy patient → probability = 0.31
```

The model correctly ranks:

$$
0.83>0.31
$$

That's good.

AUC measures this kind of **ranking ability** across thresholds.

---

# 14. AUC Interpretation

A common rough interpretation:

| AUC | Interpretation |
|---:|---|
| 1.00 | Perfect |
| 0.90–1.00 | Excellent |
| 0.80–0.90 | Good |
| 0.70–0.80 | Fair |
| 0.60–0.70 | Poor |
| 0.50 | Random |
| < 0.50 | Worse than random |

These boundaries are **rules of thumb**, not universal laws.

Context matters.

---

# 15. ROC-AUC vs Accuracy

This distinction is very important.

Suppose we have:

```text
1000 patients
990 healthy
10 diseased
```

A stupid model predicts:

```text
Everyone = Healthy
```

Then:

$$
Accuracy=\frac{990}{1000}=99\%
$$

Looks excellent!

But:

$$
TP=0
$$

because the model detected no diseased patients.

Therefore:

$$
Recall=0
$$

This demonstrates why accuracy can be misleading with imbalanced datasets.

ROC-AUC gives us a threshold-independent view of the model's ability to distinguish the two classes.

---

# 16. ROC-AUC is threshold-independent

Suppose your model outputs:

```text
0.91
0.82
0.77
0.61
0.54
0.42
0.21
```

ROC considers many possible thresholds.

For example:

```text
Threshold = 0.9
Threshold = 0.8
Threshold = 0.7
Threshold = 0.6
Threshold = 0.5
Threshold = 0.4
...
```

Therefore ROC-AUC evaluates the model over its range of possible operating thresholds rather than only at:

$$
threshold=0.5
$$

This is one of its major advantages.

---

# 17. ROC Curve vs Confusion Matrix

The confusion matrix represents **one threshold**.

For example:

```text
threshold = 0.5
```

gives one:

$$
TP,FP,TN,FN
$$

and therefore one:

$$
TPR,FPR
$$

ROC does something different.

It evaluates:

```text
threshold = 0.99
       ↓
threshold = 0.90
       ↓
threshold = 0.80
       ↓
threshold = 0.70
       ↓
...
       ↓
threshold = 0.01
```

and produces many:

$$
(TPR,FPR)
$$

points.

Therefore:

> **Confusion matrix = one operating point**

> **ROC curve = many operating points**

---

# 18. ROC Curve and Precision-Recall Curve

These two are often confused.

### ROC

Plots:

$$
TPR\;vs.\;FPR
$$

where:

$$
TPR=\frac{TP}{TP+FN}
$$

and:

$$
FPR=\frac{FP}{FP+TN}
$$

---

### Precision-Recall

Plots:

$$
Precision\;vs.\;Recall
$$

where:

$$
Precision=\frac{TP}{TP+FP}
$$

and:

$$
Recall=\frac{TP}{TP+FN}
$$

The major difference is that ROC uses:

$$
FPR
$$

while PR uses:

$$
Precision
$$

---

# 19. When should we use ROC-AUC?

ROC-AUC is particularly useful when:

### 1. You care about ranking

You want positive examples to receive higher scores than negative examples.

### 2. You don't know the final threshold yet

ROC helps examine performance over many thresholds.

### 3. You want to compare classifiers

For example:

```text
Logistic Regression → AUC = 0.81
Random Forest       → AUC = 0.87
XGBoost             → AUC = 0.91
```

Generally, XGBoost has the strongest discrimination among these models.

---

# 20. Important limitation with imbalanced datasets

ROC-AUC can sometimes look deceptively good when the positive class is extremely rare.

Consider:

```text
1000 samples
990 negatives
10 positives
```

Suppose:

$$
TP=9
$$

$$
FN=1
$$

Then:

$$
TPR=\frac{9}{10}=0.9
$$

Looks excellent.

But suppose:

$$
FP=90
$$

and:

$$
TN=900
$$

Then:

$$
FPR=\frac{90}{990}\approx0.091
$$

ROC might still look quite good.

But precision is:

$$
Precision=\frac{9}{9+90}
$$

$$
Precision\approx0.091
$$

Only about 9.1% of predicted positives are actually positive.

Therefore, with **highly imbalanced data**, you should often examine:

$$
\boxed{\text{Precision-Recall Curve + PR-AUC}}
$$

alongside ROC-AUC.

---

# 21. ROC-AUC does NOT tell you everything

AUC is useful, but it doesn't tell you:

- The best threshold
- Precision at your chosen threshold
- Recall at your chosen threshold
- Calibration of probabilities
- Business cost of FP vs FN

For example:

```text
Model A: AUC = 0.90
Model B: AUC = 0.88
```

You cannot automatically conclude Model A is better for every real-world application.

Suppose the application strongly penalizes false negatives.

You might care about:

$$
Recall
$$

at a particular:

$$
FPR
$$

rather than overall AUC.

---

# 22. Choosing a threshold using ROC

Suppose your ROC curve contains:

| Threshold | TPR | FPR |
|---:|---:|---:|
| 0.9 | 0.55 | 0.02 |
| 0.7 | 0.75 | 0.08 |
| 0.5 | 0.86 | 0.15 |
| 0.3 | 0.94 | 0.30 |

Which one should you choose?

It depends on the problem.

### Fraud detection

You might want:

```text
Low FPR
```

because investigating thousands of legitimate transactions is expensive.

### Disease detection

You might prioritize:

```text
High TPR
```

because missing a disease can be much worse than a false alarm.

So the threshold is a **business/domain decision**, not simply a machine-learning decision.

---
