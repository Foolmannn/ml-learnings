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

# 27. Method 7 — Class Weighting

Instead of modifying the dataset, we can tell the model:

> Minority-class mistakes are more expensive.

For example:

```text
Majority class weight = 1
Minority class weight = 10
```

A wrong minority prediction contributes more to the training objective.

This is called **cost-sensitive learning**.

---

# 28. Logistic Regression with Class Weights

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(
    class_weight="balanced",
    random_state=42
)

model.fit(X_train, y_train)
```

`balanced` automatically calculates class weights based on class frequencies.

Conceptually:

$$
w_j=\frac{n}{k n_j}
$$

where:

* \(n\) = total number of samples
* \(k\) = number of classes
* \(n_j\) = number of samples in class \(j\)

Therefore, rare classes receive larger weights.

---

# 29. Decision Tree with Class Weight

```python
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(
    class_weight="balanced",
    random_state=42
)

model.fit(X_train, y_train)
```

---

# 30. Random Forest with Class Weight

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train, y_train)
```

Another option is:

```python
class_weight="balanced_subsample"
```

which calculates weights separately for each bootstrap sample.

---

# 31. XGBoost and Imbalanced Data

For binary classification, XGBoost provides:

```python
scale_pos_weight
```

Example:

```python
from xgboost import XGBClassifier

model = XGBClassifier(
    scale_pos_weight=9
)

model.fit(X_train, y_train)
```

A common starting point is:

$$
scale\_pos\_weight=
\frac{\text{number of negative samples}}
{\text{number of positive samples}}
$$

Suppose:

```text
Negative = 9,000
Positive = 1,000
```

Then:

$$
scale\_pos\_weight=\frac{9000}{1000}=9
$$

---

# 32. Method 8 — Threshold Tuning

Many classification models produce probabilities.

Example:

```python
model.predict_proba(X_test)
```

Output:

```text
[0.12, 0.87]
[0.70, 0.30]
[0.45, 0.55]
```

Normally, a threshold of:

```text
0.5
```

is used.

For example:

```python
probability >= 0.5 → Class 1
probability < 0.5  → Class 0
```

But we can change this threshold.

---

# 33. Example of Threshold Tuning

Suppose:

```text
Probability = 0.35
```

Default:

```text
0.35 < 0.5
→ Class 0
```

But if we use:

```text
threshold = 0.30
```

then:

```text
0.35 >= 0.30
→ Class 1
```

Lowering the threshold generally makes the classifier predict the positive class more often.

This can increase recall, but it can also increase false positives and reduce precision.

---

# 34. Why Threshold Tuning Is Powerful

Suppose you are detecting disease.

You may prefer:

```text
Recall = very high
```

because missing a sick patient is costly.

You could lower the threshold:

```text
0.50 → 0.40 → 0.30
```

This may identify more positive cases.

For a system where false positives are expensive, you might instead choose a higher threshold.

The correct threshold depends on the application's costs and objectives.

---

# 35. Method 9 — Balanced Ensemble Methods

Some ensemble methods are designed specifically for imbalanced data.

Examples include:

* Balanced Random Forest
* EasyEnsemble
* RUSBoost

---

## Balanced Random Forest

Instead of allowing every tree to see the original class distribution, balanced sampling can be used for each tree.

Conceptually:

```text
Original dataset

Majority: 9000
Minority: 1000

       ↓

Tree 1:
Majority: 1000
Minority: 1000

Tree 2:
Majority: 1000
Minority: 1000

Tree 3:
Majority: 1000
Minority: 1000
```

This helps individual trees learn from minority observations.

---

# 36. Method 10 — Collect More Minority-Class Data

Sometimes the best solution is not an algorithmic trick.

If possible, collect more real examples.

For example:

```text
Current:
Normal = 90,000
Fraud = 1,000
```

Instead of artificially generating fraud cases, collect more actual fraud examples.

Real data is generally preferable to synthetic data when it can be obtained reliably.

---

# 37. Method 11 — Data Augmentation

For some domains, particularly images, audio, and text, we can create variations of minority examples.

For example, image classification:

```text
Original minority image
       ↓
Rotation
       ↓
Crop
       ↓
Flip
       ↓
Brightness variation
```

This creates additional training examples.

For image datasets, augmentation can be very useful when the minority class has relatively few examples.

---

# 38. Method 12 — Anomaly Detection

Sometimes the minority class is **extremely rare**.

For example:

```text
Normal transactions → 999,900
Fraud → 100
```

In such cases, treating the problem as ordinary binary classification may not always be ideal.

You can consider anomaly/outlier detection methods such as:

* Isolation Forest
* One-Class SVM
* Autoencoders

The choice depends on whether you have reliable labels for the rare class.

---

# 39. Comparing the Main Techniques

| Technique            | What it does                         | Main advantage                               | Main problem                          |
| -------------------- | ------------------------------------ | -------------------------------------------- | ------------------------------------- |
| Random Undersampling | Removes majority samples             | Simple and fast                              | Loses information                     |
| Random Oversampling  | Duplicates minority samples          | Simple                                       | Overfitting                           |
| SMOTE                | Generates synthetic minority samples | Less direct duplication                      | Synthetic samples may be unrealistic  |
| Borderline-SMOTE     | Focuses on boundary samples          | Focuses on difficult cases                   | Can amplify noisy boundaries          |
| SMOTE-Tomek          | Oversampling + cleaning              | Handles overlap                              | More complex                          |
| SMOTE-ENN            | Oversampling + cleaning              | Strong cleaning                              | Can remove useful samples             |
| Class Weight         | Penalizes minority mistakes          | No resampling                                | May not solve all distributions       |
| Threshold Tuning     | Changes decision threshold           | Directly controls precision/recall trade-off | Doesn't change learned representation |
| Balanced Ensembles   | Balanced sampling within ensembles   | Strong practical approach                    | More computation                      |
| More Data            | Adds real minority examples          | Best when feasible                           | Often expensive                       |

---

# 40. Resampling vs Class Weighting

A very important comparison.

### Resampling

Changes the training dataset:

```text
Original

Majority = 9000
Minority = 1000

        ↓

Resampling

Majority = 9000
Minority = 9000
```

### Class weighting

Keeps the data unchanged:

```text
Majority = 9000
Minority = 1000
```

but changes the training objective:

```text
Majority mistake → low cost
Minority mistake → high cost
```

---

# 41. Which Technique Should You Use?

There is no universal solution.

A practical approach is:

### Case 1: Moderate imbalance

Try:

```text
Class weighting
```

first.

---

### Case 2: Severe imbalance

Consider:

```text
Class weighting
+
Threshold tuning
+
Precision/Recall evaluation
```

and potentially resampling.

---

### Case 3: Small minority dataset

Try:

```text
SMOTE
```

or another carefully chosen oversampling method.

---

### Case 4: Very large dataset

Undersampling can become attractive because training on millions of majority examples may be expensive.

---

### Case 5: Extremely rare events

Consider:

```text
Anomaly detection
```

in addition to supervised classification, depending on the availability and quality of labels.

---

# 42. A Complete Practical Example

Let's create an imbalanced classification dataset.

```python
from sklearn.datasets import make_classification

X, y = make_classification(
    n_samples=10000,
    n_features=10,
    n_informative=5,
    n_redundant=2,
    weights=[0.95, 0.05],
    random_state=42
)
```

Here:

```text
Class 0 ≈ 95%
Class 1 ≈ 5%
```

---

# 43. Split the Dataset

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)
```

---

# 44. Baseline Model

Let's first train a model without handling imbalance.

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
```

Evaluate:

```python
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))
```

Pay attention to:

```text
precision
recall
f1-score
support
```

especially for the minority class.

---

# 45. Using Class Weight

```python
model = LogisticRegression(
    class_weight="balanced",
    max_iter=1000
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))
```

Notice that the minority-class recall may increase, although precision may change.

---

# 46. Using SMOTE

```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train,
    y_train
)
```

Then:

```python
model = LogisticRegression(max_iter=1000)

model.fit(X_train_smote, y_train_smote)

y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))
```

Notice that:

```text
SMOTE → training data changed
Test data → unchanged
```

This is exactly what we want.

---

# 47. Using a Pipeline

For real ML projects, a pipeline is often safer.

```python
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("smote", SMOTE(random_state=42)),
    ("model", LogisticRegression(max_iter=1000))
])

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
```

Then:

```python
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))
```

The pipeline ensures that SMOTE is applied as part of the training workflow rather than manually contaminating the test set.

---

# 48. Cross-Validation with Imbalanced Data

This is another area where mistakes are common.

Suppose you do:

```python
SMOTE → entire dataset → cross-validation
```

This can cause leakage.

Instead, put SMOTE **inside the pipeline**:

```python
from imblearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE

pipeline = Pipeline([
    ("smote", SMOTE(random_state=42)),
    ("model", LogisticRegression(max_iter=1000))
])

scores = cross_val_score(
    pipeline,
    X,
    y,
    cv=5,
    scoring="f1"
)

print(scores)
print(scores.mean())
```

Now SMOTE is performed separately within each training fold.

---

# 49. Why Accuracy Should Not Be Your Main Metric

Suppose:

```text
Actual:

9500 negative
500 positive
```

A model predicts:

```text
9700 negative
300 positive
```

You should not look only at:

```text
Accuracy = ?
```

You should ask:

```text
How many of the 500 positive cases were detected?
How many false positives occurred?
What is precision?
What is recall?
What is F1?
What does the confusion matrix look like?
```

---

# 50. Confusion Matrix

Use:

```python
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print(cm)
```

You can visualize it:

```python
from sklearn.metrics import ConfusionMatrixDisplay

ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred
)
```

For imbalanced classification, always inspect the confusion matrix alongside aggregate metrics.

---

# 51. Macro vs Weighted F1

This becomes particularly important in multiclass imbalance.

Suppose:

```text
Class A → 9000
Class B → 900
Class C → 100
```

### Macro F1

Calculate F1 separately for each class and then take the simple average:

$$
F1_{macro}
=
\frac{F1_A+F1_B+F1_C}{3}
$$

Every class gets equal importance.

---

### Weighted F1

Weights each class according to its number of samples.

Therefore, the majority class has much more influence.

So:

```text
Macro F1
→ "How well am I doing across classes equally?"

Weighted F1
→ "How well am I doing while accounting for class frequency?"
```

For imbalanced multiclass problems, **macro metrics can expose poor minority-class performance that weighted metrics may hide**.

---

# 52. Balanced Accuracy

Another useful metric is:

$$
Balanced\ Accuracy=
\frac{Sensitivity+Specificity}{2}
$$

Unlike ordinary accuracy, it gives equal importance to the two classes.

Example:

```text
Sensitivity = 90%
Specificity = 80%
```

Then:

$$
Balanced\ Accuracy
=
\frac{0.90+0.80}{2}
=0.85
$$

So:

```text
Balanced Accuracy = 85%
```

In scikit-learn:

```python
from sklearn.metrics import balanced_accuracy_score

score = balanced_accuracy_score(y_test, y_pred)

print(score)
```

---

# 53. Important: Scaling and SMOTE

SMOTE uses distances between observations.

Therefore, feature scaling can matter considerably when features have very different scales.

For example:

```text
Age       → 18–80
Income    → 20,000–2,000,000
```

Distance calculations could be dominated by income.

A common pipeline is:

```text
Scaling
   ↓
SMOTE
   ↓
Model
```

For example:

```python
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("smote", SMOTE(random_state=42)),
    ("model", LogisticRegression(max_iter=1000))
])
```

The exact ordering can depend on the preprocessing and model, but scaling before distance-based SMOTE is commonly appropriate for numeric features.

---

# 54. Common Mistakes

## Mistake 1: Using Accuracy Only

```python
accuracy_score(y_test, y_pred)
```

is not enough.

Use:

```text
Precision
Recall
F1
Confusion Matrix
PR-AUC
ROC-AUC
Balanced Accuracy
```

depending on the problem.

---

## Mistake 2: Applying SMOTE Before Train-Test Split

Wrong:

```python
SMOTE → Split
```

Correct:

```text
Split
 ↓
SMOTE only on training
 ↓
Train
 ↓
Evaluate on untouched test set
```

---

## Mistake 3: Resampling the Test Set

Don't do:

```python
X_test, y_test = smote.fit_resample(X_test, y_test)
```

The test set should represent the distribution you actually want to evaluate against.

---

## Mistake 4: Assuming 50/50 Is Always Necessary

You don't necessarily need:

```text
50% / 50%
```

The objective is not:

> Make the dataset perfectly balanced at all costs.

The objective is:

> Build a model that performs appropriately on the classes that matter, under the real-world cost of errors.

Sometimes:

```text
70/30
```

is perfectly reasonable.

---

# 55. Important Concept: Class Distribution vs Class Importance

These are different concepts.

Suppose:

```text
Normal = 99%
Fraud = 1%
```

Fraud is rare.

But that does **not** mean fraud is unimportant.

Therefore:

```text
Frequency ≠ Importance
```

This is the fundamental reason imbalance needs special treatment.

---

# 56. Cost-Sensitive Classification

Sometimes the real objective is not:

```text
maximize accuracy
```

but:

```text
minimize business cost
```

Suppose:

```text
False Positive cost = $5
False Negative cost = $500
```

Then:

```text
FN is 100× more expensive.
```

You would likely want a model and threshold that reflect this asymmetry.

This is why class weights and threshold tuning are often more meaningful than simply forcing a 50/50 dataset.

---

# 57. A Good Practical Workflow

For a real-world imbalanced classification problem, I recommend this workflow:

```text
             Raw Dataset
                  ↓
          Check target distribution
                  ↓
            Train/Test Split
                  ↓
             stratify = y
                  ↓
        ┌─────────┴─────────┐
        ↓                   ↓
   Baseline Model      Class-weighted Model
        │                   │
        └─────────┬─────────┘
                  ↓
          Compare Metrics
                  ↓
       Precision / Recall / F1
                  ↓
       Confusion Matrix + PR Curve
                  ↓
       Try Resampling if useful
                  ↓
      SMOTE / Undersampling /
      SMOTE-Tomek / SMOTE-ENN
                  ↓
          Cross-validation
                  ↓
       Threshold Optimization
                  ↓
       Final Model Selection
                  ↓
     Evaluate ONCE on Test Set
```

---

# 58. What I Would Check in a Real Project

When you encounter an imbalanced dataset, don't immediately apply SMOTE.

First ask:

### 1. How imbalanced is it?

```python
y.value_counts(normalize=True)
```

### 2. What is the cost of FP vs FN?

This determines whether precision or recall deserves more emphasis.

### 3. Is the dataset small or huge?

* Small → oversampling may help.
* Huge → undersampling may be practical.

### 4. Is the minority class noisy?

If yes, blindly applying SMOTE can create more problematic samples.

### 5. Are the features numeric?

SMOTE's standard version is designed around numeric feature spaces. For mixed categorical/numeric data, methods such as **SMOTENC** may be more appropriate.

### 6. Is the positive class extremely rare?

Consider:

```text
Class weighting
Threshold tuning
Anomaly detection
Specialized evaluation
```

---

# 59. Interview/Exam Definition

> **Imbalanced data refers to a classification dataset in which the distribution of samples among classes is significantly unequal. This can cause a machine-learning model to become biased toward the majority class, making accuracy misleading and causing poor minority-class detection. Imbalanced data can be handled using techniques such as random undersampling, random oversampling, SMOTE, Borderline-SMOTE, SMOTE-Tomek, SMOTE-ENN, class weighting, threshold tuning, balanced ensemble methods, and anomaly detection. Appropriate evaluation metrics such as precision, recall, F1-score, balanced accuracy, ROC-AUC, PR-AUC, and confusion matrix should be used instead of relying only on accuracy.**

---

# 60. The Most Important Things to Remember

If you're studying this for **ML implementation + exams**, remember these points:

```text
Imbalanced Data
      ↓
Majority class ≫ Minority class
      ↓
Accuracy can become misleading
      ↓
Use:
    Precision
    Recall
    F1
    PR-AUC
    ROC-AUC
    Balanced Accuracy
      ↓
Solutions:
    ├── Undersampling
    ├── Oversampling
    ├── SMOTE
    ├── Borderline-SMOTE
    ├── SMOTE-Tomek
    ├── SMOTE-ENN
    ├── Class Weighting
    ├── Threshold Tuning
    ├── Balanced Ensembles
    └── Anomaly Detection
```

### The golden rule:

**Never apply resampling before splitting your data.**

Use:

```text
Original Data
     ↓
Train/Test Split
     ↓
Resample TRAIN only
     ↓
Train Model
     ↓
Evaluate on ORIGINAL TEST SET
```

And for cross-validation:

```text
Pipeline:
Scaling → Resampling → Model
```

so resampling happens independently inside each training fold.
