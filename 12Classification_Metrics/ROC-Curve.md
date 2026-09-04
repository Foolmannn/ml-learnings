
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

# 23. The Youden's J statistic

One common method for selecting a threshold is **Youden's J statistic**:

$$
J=TPR-FPR
$$

or equivalently:

$$
J=Sensitivity+Specificity-1
$$

Since:

$$
Specificity=1-FPR
$$

we get:

$$
J=TPR-(1-Specificity)
$$

$$
J=TPR+Specificity-1
$$

You can select the threshold that maximizes:

$$
\boxed{J}
$$

This finds a point that balances sensitivity and specificity.

But again, it isn't always the optimal business threshold.

---

# 24. Specificity and ROC

Specificity is:

$$
Specificity=\frac{TN}{TN+FP}
$$

Notice:

$$
FPR=\frac{FP}{FP+TN}
$$

Therefore:

$$
\boxed{FPR=1-Specificity}
$$

This means ROC can also be understood as:

$$
\boxed{Sensitivity\;vs.\;1-Specificity}
$$

or:

$$
\boxed{TPR\;vs.\;FPR}
$$

These are exactly the same thing.

---

# 25. Geometric interpretation of ROC

The ROC space has:

```text
TPR
 1 |             ● Perfect
   |           /
   |         /
   |       /
   |     /
   |   /
   | /
 0 |●-----------------------●
   0                       1
            FPR
```

The important regions are:

### Upper-left

Ideal.

$$
FPR\rightarrow0
$$

$$
TPR\rightarrow1
$$

### Diagonal

Random classifier.

$$
TPR=FPR
$$

### Lower-right

Poor classifier.

---

# 26. Dominance between ROC curves

Suppose we have two models:

```text
Model A
Model B
```

If Model A's ROC curve is consistently above Model B's curve, then Model A generally has better discrimination.

For example:

```text
TPR
1 |        A
  |      A
  |    A     B
  |   A     B
  |  A    B
0 |----------------
  0              1
       FPR
```

Model A dominates Model B across those operating regions.

---

# 27. ROC-AUC in scikit-learn

Suppose we train Logistic Regression:

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score
```

Train the model:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = LogisticRegression()

model.fit(X_train, y_train)
```

Now get **probabilities**:

```python
y_prob = model.predict_proba(X_test)[:, 1]
```

This is extremely important.

Do **not** use:

```python
y_pred = model.predict(X_test)
```

for ROC-AUC when you want the full ROC curve.

Instead use the continuous probability/score:

```python
y_prob
```

---

# 28. Calculate ROC-AUC

```python
auc = roc_auc_score(y_test, y_prob)

print("ROC-AUC:", auc)
```

For example:

```text
ROC-AUC: 0.87
```

means the model has reasonably strong ability to distinguish positive examples from negative examples.

---

# 29. Calculate the ROC curve

```python
fpr, tpr, thresholds = roc_curve(
    y_test,
    y_prob
)
```

You now have three arrays:

```python
fpr
tpr
thresholds
```

For example:

```text
threshold     FPR     TPR
--------------------------------
inf           0.00    0.00
0.91          0.01    0.20
0.82          0.04    0.45
0.65          0.10    0.70
0.42          0.20    0.85
0.15          0.50    0.95
...
```

These points create the ROC curve.

---

# 30. Plot ROC curve

```python
import matplotlib.pyplot as plt

plt.plot(fpr, tpr, label=f"ROC-AUC = {auc:.2f}")

plt.plot([0, 1], [0, 1], linestyle="--")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title("ROC Curve")

plt.legend()

plt.show()
```

---

# 31. Very important: `predict()` vs `predict_proba()`

This is a common beginner mistake.

### `predict()`

Returns:

```text
0
1
0
1
```

These are final class predictions.

---

### `predict_proba()`

Returns something like:

```text
0.91
0.72
0.13
0.45
```

These are probability estimates.

ROC needs to see how the model ranks examples as the threshold moves.

Therefore:

```python
roc_curve(y_test, y_prob)
```

where:

```python
y_prob = model.predict_proba(X_test)[:, 1]
```

is the usual approach.

---

# 32. ROC-AUC with multiple models

You can compare models:

```python
models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(),
}
```

For each model:

```python
for name, model in models.items():

    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_prob)

    print(name, auc)
```

You might get:

```text
Logistic Regression    0.84
Random Forest          0.91
```

Random Forest has better overall discrimination according to ROC-AUC.

---

# 33. ROC-AUC and logistic regression

Since you've been studying logistic regression, there's a useful connection.

Logistic regression calculates:

$$
z=w^Tx+b
$$

and then:

$$
p=\sigma(z)
$$

where:

$$
\sigma(z)=\frac{1}{1+e^{-z}}
$$

The model produces:

$$
p=P(y=1|x)
$$

Then we select a threshold:

$$
p\geq t
$$

to predict class 1.

ROC essentially asks:

> What happens to TPR and FPR as $t$ moves from 1 toward 0?

So the pipeline is:

$$
X
$$

↓

$$
w^TX+b
$$

↓

$$
\sigma(w^TX+b)
$$

↓

**Probability**

↓

**Change threshold**

↓

$$
TPR,FPR
$$

↓

**ROC Curve**

↓

**Area**

↓

$$
\boxed{ROC-AUC}
$$

---

# 34. AUC and ranking

Here's another excellent way to understand AUC.

Suppose we have:

### Positive examples

```text
P1 = 0.90
P2 = 0.80
P3 = 0.70
```

### Negative examples

```text
N1 = 0.40
N2 = 0.30
N3 = 0.20
```

Every positive score is greater than every negative score.

Therefore:

$$
AUC=1
$$

Now suppose:

```text
Positive:
0.90
0.80
0.30

Negative:
0.70
0.40
0.20
```

Some positive/negative pairs are incorrectly ranked.

Therefore:

$$
AUC<1
$$

This demonstrates that AUC measures **ranking quality**.

---

# 35. Pairwise interpretation of AUC

Suppose:

$$
N_+=\text{number of positive samples}
$$

and:

$$
N_-=\text{number of negative samples}
$$

There are:

$$
N_+N_-
$$

possible positive-negative pairs.

AUC can approximately be interpreted as:

$$
AUC=
\frac{\text{correctly ranked positive-negative pairs}}
{\text{total positive-negative pairs}}
$$

with appropriate handling of ties.

For example:

```text
100 positive samples
100 negative samples
```

Number of pairs:

$$
100\times100=10,000
$$

If the model correctly ranks 8,500 pairs:

$$
AUC\approx\frac{8500}{10000}
$$

$$
\boxed{AUC=0.85}
$$

This is one of the best mathematical intuitions for AUC.

---

# 36. ROC-AUC vs F1 Score

They answer different questions.

### ROC-AUC

Measures:

> How well does the model discriminate/rank positives versus negatives across thresholds?

### F1

Measures:

> How well does the model balance precision and recall at a particular threshold?

F1:

$$
F1=2\frac{Precision\times Recall}
{Precision+Recall}
$$

Therefore:

```text
ROC-AUC
    ↓
Threshold-independent discrimination

F1
    ↓
Performance at a chosen threshold
```

---

# 37. ROC-AUC vs Accuracy vs F1

| Metric | Main purpose |
|---|---|
| Accuracy | Overall correct predictions |
| Precision | How many predicted positives are actually positive |
| Recall/TPR | How many actual positives were detected |
| F1 | Balance between precision and recall |
| ROC-AUC | Ranking/discrimination across thresholds |
| PR-AUC | Precision-recall performance, especially useful for rare positives |

---

# 38. A practical ML evaluation workflow

For a binary classification problem, don't rely on only one metric.

A good evaluation might be:

```text
                 Classification Model
                         |
          +--------------+--------------+
          |              |              |
       Accuracy       Precision       Recall
          |              |              |
          +--------------+--------------+
                         |
                        F1
                         |
                  ROC-AUC / PR-AUC
```

And inspect the:

```text
Confusion Matrix
ROC Curve
Precision-Recall Curve
```

---

# 39. Example: Landslide detection

For the mountain/river landslide-detection system you've been considering, imagine:

```text
Input:
Satellite image + rainfall + river flow + terrain features
                         ↓
                    ML model
                         ↓
              P(landslide)=0.83
```

Suppose:

$$
P(landslide)=0.83
$$

If threshold is:

$$
0.5
$$

then:

```text
0.83 > 0.5
```

so:

```text
Landslide = YES
```

But suppose you want to minimize missed landslides.

You could lower the threshold:

$$
0.5\rightarrow0.3
$$

Now more possible landslides will be detected.

Therefore:

$$
TPR\uparrow
$$

but potentially:

$$
FPR\uparrow
$$

ROC analysis helps you understand this trade-off.

For a safety-oriented system, though, you would **not choose the threshold solely from AUC**. You would consider the real cost of false alarms versus missed events and likely combine ROC/PR analysis with domain-specific alert criteria.

---

# 40. The biggest concepts to remember

If you're preparing for ML interviews/exams, remember these relationships:

### Confusion matrix

$$
TP,\;TN,\;FP,\;FN
$$

↓

### True Positive Rate

$$
\boxed{TPR=\frac{TP}{TP+FN}}
$$

↓

### False Positive Rate

$$
\boxed{FPR=\frac{FP}{FP+TN}}
$$

↓

### ROC

$$
\boxed{ROC=TPR\;vs.\;FPR}
$$

↓

### ROC-AUC

$$
\boxed{\text{Area under ROC curve}}
$$

↓

### Interpretation

$$
\boxed{\text{Probability that a random positive is ranked above a random negative}}
$$

---

# 41. The complete mental model

Think of a classifier as producing a **score**, not immediately producing a class:

```text
                    Model
                      ↓
              Continuous score
                      ↓
        ┌─────────────┼─────────────┐
        ↓             ↓             ↓
     threshold      threshold     threshold
       0.8            0.5           0.2
        ↓             ↓             ↓
     TPR/FPR        TPR/FPR       TPR/FPR
        └─────────────┼─────────────┘
                      ↓
                 ROC Curve
                      ↓
                 Area under it
                      ↓
                   ROC-AUC
```

The most important distinction is:

> **ROC curve = behavior across thresholds.**

> **ROC-AUC = one-number summary of that curve.**

And:

> **AUC evaluates discrimination/ranking, not whether the predicted probabilities are perfectly calibrated.**

---

## Quick interview questions

**Q: What does ROC stand for?**

Receiver Operating Characteristic.

**Q: What does ROC plot?**

$$
TPR\text{ against }FPR
$$

**Q: What is TPR?**

$$
TPR=\frac{TP}{TP+FN}
$$

**Q: What is FPR?**

$$
FPR=\frac{FP}{FP+TN}
$$

**Q: What does AUC = 0.5 mean?**

Approximately random discrimination.

**Q: What does AUC = 1 mean?**

Perfect discrimination.

**Q: Is ROC-AUC threshold-dependent?**

No. It summarizes performance across thresholds.

**Q: Is accuracy threshold-dependent?**

Yes.

**Q: For severe class imbalance, should you only look at ROC-AUC?**

No. Also examine the **Precision-Recall curve / PR-AUC**, precision, recall, and the confusion matrix.

**Q: Should ROC use `predict()` or probability scores?**

Usually probability/decision scores:

```python
model.predict_proba(X_test)[:, 1]
```

rather than hard `0/1` predictions.

### One-line summary

$$
\boxed{
\text{ROC Curve}=\text{TPR vs FPR at different thresholds}
}
$$

$$
\boxed{
\text{ROC-AUC}=\text{Area under that curve}
}
$$

$$
\boxed{
AUC\approx P(\text{random positive gets higher score than random negative})
}
$$