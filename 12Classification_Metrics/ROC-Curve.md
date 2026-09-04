
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
