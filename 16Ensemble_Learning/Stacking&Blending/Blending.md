# Blending Ensemble Learning — In Detail

**Blending** is an ensemble-learning technique where we train multiple **base models**, use their predictions on a separate **validation set**, and then train a **meta-model** (or use a simple combination rule) to make the final prediction.

The biggest difference from stacking is:

> **Blending uses a holdout validation set for the meta-model, while stacking typically uses out-of-fold (OOF) predictions generated through cross-validation.**

This distinction is extremely important.

---

# 1. Basic Idea

Suppose you're solving a classification problem.

You have:

```text
Logistic Regression
Random Forest
SVM
Decision Tree
```

Instead of selecting only one model, blending combines their predictions.

The architecture is:

```text
                    Dataset
                       │
                Train / Validation
                       │
        ┌──────────────┴──────────────┐
        │                             │
    Training Set                 Validation Set
        │                             │
        ▼                             │
 ┌──────┼──────┬──────┐              │
 ▼      ▼      ▼      ▼              │
 LR     RF     SVM    DT             │
 │      │      │      │              │
 └──────┴──────┴──────┘              │
        │                             │
        └──── Predictions ────────────┘
                       │
                       ▼
                Meta Features
                       │
                       ▼
                  Meta-model
                       │
                       ▼
                Final Prediction
```

The validation set is **not used to train the base models**.

It is reserved to generate predictions that train the meta-model.

---

# 2. Why Is It Called Blending?

Imagine several models making predictions:

```text
Model 1 → 0.70
Model 2 → 0.85
Model 3 → 0.60
```

These predictions are "blended" together to produce the final prediction.

The blending can be done by:

### Simple average

$$
P = \frac{P_1+P_2+P_3}{3}
$$

or by a learned meta-model:

$$
P_{final}=g(P_1,P_2,P_3)
$$

where \(g\) is the meta-model.

---

# 3. The Most Important Difference: Blending vs Stacking

This is the concept you should remember from this topic.

## Stacking

Uses cross-validation to create OOF predictions.

```text
Training data
      │
      ▼
  5-fold CV
      │
      ▼
OOF predictions
      │
      ▼
Meta-model
```

## Blending

Splits training data into:

```text
Training set
Validation set
```

Then:

```text
Training set
      │
      ▼
Base models
      │
      ▼
Validation predictions
      │
      ▼
Meta-model
```

So:

| Feature            | Stacking                 | Blending            |
| ------------------ | ------------------------ | ------------------- |
| Base models        | Multiple                 | Multiple            |
| Meta-model         | Yes                      | Usually yes         |
| Meta training data | OOF predictions          | Holdout predictions |
| Cross-validation   | Usually used             | Not required        |
| Validation set     | Not necessarily separate | Required            |
| Computational cost | Higher                   | Lower               |
| Data efficiency    | Higher                   | Lower               |
| Implementation     | More complex             | Simpler             |

---

# 4. Simple Example

Suppose we have:

```text
1000 samples
```

We split them:

```text
800 → Base-model training
200 → Validation for meta-model
```

So:

```text
1000 samples
      │
      ├── 800 Training
      │
      └── 200 Validation
```

Train three base models on the 800:

```text
Logistic Regression
Random Forest
SVM
```

Then predict the 200 validation samples.

For one validation sample:

```text
Logistic Regression → 0.70
Random Forest       → 0.85
SVM                 → 0.60
```

Create a meta-feature row:

```text
[0.70, 0.85, 0.60]
```

Repeat for all 200 validation samples:

```text
LR       RF       SVM       y
0.70     0.85     0.60      1
0.20     0.10     0.30      0
0.90     0.80     0.75      1
...
```

Now train the meta-model:

```text
[LR prediction, RF prediction, SVM prediction]
                         ↓
                    Meta-model
                         ↓
                    Final output
```

---

# 5. Why Can't We Use the Training Predictions?

Suppose:

```python
model.fit(X_train, y_train)

pred = model.predict(X_train)
```

and then:

```text
pred → Meta-model
```

This is problematic.

Why?

Because the base model already saw those samples.

Therefore:

```text
X_train
   ↓
Base model
   ↓
Predictions on X_train
```

may be overly optimistic.

The meta-model would learn from predictions that don't accurately represent how the base models perform on unseen data.

Blending avoids this by using:

```text
X_validation
```

which the base models **never saw during training**.

---

# 6. Blending Data Split

A typical setup is:

```text
Original training data
          │
          ▼
    ┌─────────────┐
    │ Train 80%   │
    │ Validation  │
    │ 20%         │
    └─────────────┘
```

Then:

### Training set

Used for:

```text
Base model training
```

### Validation set

Used for:

```text
Generating base predictions
Training meta-model
```

### Test set

Used only at the very end:

```text
Final evaluation
```

This gives us:

```text
Training
   ↓
Base models

Validation
   ↓
Meta-model

Test
   ↓
Final evaluation
```

---

# 7. Three Important Datasets

A proper blending workflow often has **three datasets**:

```text
Dataset
   │
   ├── Base Training Set
   │
   ├── Meta/Validation Set
   │
   └── Test Set
```

For example:

```text
100%
 │
 ├── 60% → Base training
 ├── 20% → Meta training
 └── 20% → Final testing
```

The exact percentages aren't fixed.

You might use:

```text
70 / 15 / 15
```

or:

```text
75 / 15 / 10
```

depending on dataset size.

---

# 8. Complete Blending Workflow

Let's walk through it carefully.

Suppose:

```text
X, y
```

is our dataset.

### Step 1 — Split test set

```text
X, y
 ↓
Train + Test
```

### Step 2 — Split training portion

```text
Train
 ↓
Base Train + Meta Validation
```

So:

```text
Original data
     │
     ▼
Train/Test
     │
     ├──────────→ Test
     │
     ▼
Base Train / Meta Validation
```

---

# 9. Train Base Models

Suppose:

```text
Model A = Logistic Regression
Model B = Random Forest
Model C = SVM
```

Train each on the **base training set**:

```text
Base Training
     │
 ┌───┼────┐
 ▼   ▼    ▼
LR   RF   SVM
```

---

# 10. Generate Validation Predictions

Now use the models on the meta-validation set.

```text
Meta Validation
       │
 ┌─────┼─────┐
 ▼     ▼     ▼
 LR    RF    SVM
 │     │      │
 ▼     ▼      ▼
P1     P2     P3
```

Suppose:

```text
P1 = 0.72
P2 = 0.88
P3 = 0.65
```

Then:

```text
Meta features:

[0.72, 0.88, 0.65]
```

---

# 11. Train the Meta-Model

The meta-model receives:

```text
P1 P2 P3
```

and target:

```text
y
```

So:

$$
Z =
\begin{bmatrix}
P_1 & P_2 & P_3
\end{bmatrix}
$$

Then:

$$
MetaModel.fit(Z,y)
$$

For classification:

```text
Logistic Regression
```

could be used.

For regression:

```text
Linear Regression
```

could be used.

---

# 12. What Happens to the Test Set?

Now we have:

```text
Base models
+
Meta-model
```

We need predictions for completely unseen test data.

For every test sample:

```text
Test sample
     │
 ┌───┼────┐
 ▼   ▼    ▼
LR  RF    SVM
 │   │     │
 ▼   ▼     ▼
P1  P2     P3
 └───┼─────┘
     ▼
Meta-model
     ▼
Final prediction
```

---

# 13. Important Subtlety: Retraining Base Models

After training the meta-model, we have two common approaches.

### Approach 1

Keep base models trained on the base-training portion.

Simple and conceptually straightforward.

### Approach 2

Retrain the base models on the combined:

```text
Base Training + Meta Validation
```

Then use them to generate test predictions.

This gives the base models more training data.

But the meta-model should still be trained using predictions generated from data that the corresponding base models did not train on.

This is one reason stacking with OOF predictions is often more data-efficient.

---

# 14. Simple Blending Without a Meta-Model

Blending doesn't necessarily require a machine-learning meta-model.

You can directly combine predictions.

Suppose:

```text
Model A → 0.70
Model B → 0.85
Model C → 0.60
```

Simple average:

$$
P_{final}
=
\frac{0.70+0.85+0.60}{3}
$$

$$
P_{final}=0.7167
$$

So:

```text
Final probability ≈ 0.717
```

For binary classification:

```text
0.717 > 0.5
```

therefore:

```text
Class = 1
```

This is sometimes called **probability averaging** or **soft blending**.

---

# 15. Weighted Blending

Instead of giving every model equal importance, we can assign weights.

Suppose:

```text
Model A → 0.70
Model B → 0.85
Model C → 0.60
```

and:

```text
Weight A = 0.2
Weight B = 0.5
Weight C = 0.3
```

Then:

$$
P_{final}
=
0.2P_A+
0.5P_B+
0.3P_C
$$

Therefore:

$$
P_{final}
=
0.2(0.70)+0.5(0.85)+0.3(0.60)
$$

$$
=0.14+0.425+0.18
$$

$$
=0.745
$$

Final:

```text
0.745
```

---

# 16. Where Do the Weights Come From?

You could manually choose them:

```text
Model A = 0.2
Model B = 0.5
Model C = 0.3
```

based on validation performance.

Or you can **learn the weights** using a meta-model.

That's where blending becomes closely related to stacking.

For example:

```text
Validation predictions
        ↓
Linear Regression / Logistic Regression
        ↓
Learned combination
```

---

# 17. Blending With a Meta-Model

Suppose our validation predictions are:

```text
P_LR
P_RF
P_SVM
```

We construct:

$$
Z =
[P_{LR},P_{RF},P_{SVM}]
$$

Then:

$$
MetaModel.fit(Z,y_{validation})
$$

For a regression meta-model:

$$
\hat{y}
=
w_1P_{LR}
+
w_2P_{RF}
+
w_3P_{SVM}
+
b
$$

The model learns:

$$
w_1,w_2,w_3,b
$$

automatically.

---

# 18. Classification Example

Suppose we are predicting:

```text
Customer will buy?
```

Base models:

```text
Logistic Regression
Random Forest
SVM
```

Validation predictions:

| Customer |   LR |   RF |  SVM | Actual |
| -------- | ---: | ---: | ---: | -----: |
| A        | 0.70 | 0.85 | 0.60 |      1 |
| B        | 0.20 | 0.10 | 0.30 |      0 |
| C        | 0.90 | 0.80 | 0.75 |      1 |
| D        | 0.30 | 0.45 | 0.40 |      0 |

The meta-model learns:

```text
LR + RF + SVM predictions
             ↓
       Purchase probability
```

---

# 19. Regression Example

Suppose you're predicting house prices.

Base models:

```text
Linear Regression
Random Forest
XGBoost
SVR
```

For one validation house:

```text
Linear Regression → 250,000
Random Forest     → 275,000
XGBoost            → 280,000
SVR                → 260,000
```

Meta-model receives:

```text
[250000, 275000, 280000, 260000]
```

and perhaps predicts:

```text
273000
```

---

# 20. Blending vs Simple Averaging

These aren't necessarily the same.

### Simple averaging

```text
Model predictions
       ↓
Average
       ↓
Final prediction
```

Example:

$$
P=\frac{P_1+P_2+P_3}{3}
$$

No learning occurs.

---

### Blending with meta-model

```text
Model predictions
       ↓
Meta-model
       ↓
Final prediction
```

The meta-model learns how to combine them.

---

# 21. Blending vs Weighted Average

Weighted average:

$$
P=w_1P_1+w_2P_2+w_3P_3
$$

The weights may be manually selected.

Blending with a linear meta-model can learn something similar automatically.

However, a meta-model can potentially learn more complex relationships than a simple fixed weighted average.

---

# 22. Blending vs Stacking — Deep Comparison

Let's make this very clear.

## Stacking

Suppose:

```text
1000 training samples
```

Using 5-fold CV:

```text
Fold 1 → predict fold 1
Fold 2 → predict fold 2
Fold 3 → predict fold 3
Fold 4 → predict fold 4
Fold 5 → predict fold 5
```

Every training sample receives an OOF prediction.

Therefore:

```text
1000 samples
     ↓
1000 OOF predictions
     ↓
Meta-model
```

---

## Blending

Suppose:

```text
1000 training samples
```

Split:

```text
800 → Base training
200 → Meta validation
```

Only the 200 validation samples generate predictions for the meta-model.

Therefore:

```text
200 samples
     ↓
Validation predictions
     ↓
Meta-model
```

The 800 samples aren't directly used as meta-training examples.

---

# 23. Why Blending Is Faster

Suppose you have:

```text
5 base models
5-fold CV
```

Stacking requires repeated training of the base models to generate OOF predictions.

Blending can simply do:

```text
5 models
×
1 training
```

and then predict the validation set.

Therefore:

> **Blending is generally simpler and computationally cheaper than full cross-validated stacking.**

---

# 24. But There Is a Cost

You lose some training data for the base models.

Suppose:

```text
1000 samples
```

and:

```text
80% → Base training
20% → Meta training
```

Base models only train on:

```text
800 samples
```

rather than:

```text
1000 samples
```

This can matter significantly when the dataset is small.

---

# 25. Why Stacking Can Be More Data Efficient

With 5-fold stacking:

```text
For each fold:

80% → train
20% → predict
```

Every sample eventually gets an OOF prediction.

And each base-model training run uses 80% of the training data.

After generating OOF predictions, base models can be refitted using all available training data.

So stacking makes better use of the available data, at the cost of additional computation.

---

# 26. Data Leakage in Blending

A very common mistake:

```text
X_train
   ↓
Base model
   ↓
Prediction
   ↓
Meta-model
```

This leaks information.

Correct:

```text
Base train
    ↓
Base model
    ↓
Meta validation
    ↓
Prediction
    ↓
Meta-model
```

The meta-validation samples must not have been used to train the base models.

---

# 27. Test Set Leakage

Another mistake is using the test set to train the meta-model.

Incorrect:

```text
Base models
    ↓
Test set
    ↓
Meta-model training
```

Never do this.

The test set should remain untouched until final evaluation.

Correct:

```text
Base train
    ↓
Base models
    ↓
Validation
    ↓
Meta-model

Test
 ↓
Final evaluation
```

---

# 28. Scikit-Learn Doesn't Have a Dedicated `BlendingClassifier`

Unlike:

```python
from sklearn.ensemble import StackingClassifier
```

scikit-learn doesn't provide a direct:

```python
BlendingClassifier
```

class in the same way.

You can implement blending manually.

This is actually useful for understanding what's happening.

---

# 29. Manual Blending Implementation

Let's build a simple binary classification example.

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score
```

Load data:

```python
X, y = load_breast_cancer(return_X_y=True)
```

First create the final test set:

```python
X_temp, X_test, y_temp, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
```

Now split the remaining data into:

```text
Base training
Meta validation
```

```python
X_base, X_meta, y_base, y_meta = train_test_split(
    X_temp,
    y_temp,
    test_size=0.25,
    random_state=42,
    stratify=y_temp
)
```

This gives approximately:

```text
60% → Base training
20% → Meta validation
20% → Test
```

---

# 30. Create Base Models

```python
models = [
    LogisticRegression(max_iter=1000),

    RandomForestClassifier(
        n_estimators=200,
        random_state=42
    ),

    SVC(
        probability=True,
        random_state=42
    )
]
```

---

# 31. Train Base Models

```python
for model in models:
    model.fit(X_base, y_base)
```

Now all models have only seen:

```text
X_base
```

They have not seen:

```text
X_meta
X_test
```

---

# 32. Generate Meta Features

For classification, we'll use probabilities:

```python
meta_features = []

for model in models:
    pred = model.predict_proba(X_meta)[:, 1]
    meta_features.append(pred)
```

Now convert to a matrix:

```python
import numpy as np

meta_features = np.column_stack(meta_features)
```

Conceptually:

```text
meta_features

[
    [0.71, 0.82, 0.68],
    [0.10, 0.15, 0.20],
    [0.91, 0.88, 0.94],
    ...
]
```

Each column corresponds to one base model.

---

# 33. Train Meta-Model

```python
meta_model = LogisticRegression()

meta_model.fit(
    meta_features,
    y_meta
)
```

Now the meta-model has learned how to combine:

```text
LR prediction
RF prediction
SVM prediction
```

---

# 34. Make Test Predictions

First generate predictions from the base models:

```python
test_features = []

for model in models:
    pred = model.predict_proba(X_test)[:, 1]
    test_features.append(pred)
```

Convert:

```python
test_features = np.column_stack(test_features)
```

Now:

```text
Test sample
    ↓
LR → 0.72
RF → 0.81
SVM → 0.68
    ↓
[0.72, 0.81, 0.68]
```

Pass that to the meta-model:

```python
final_pred = meta_model.predict(test_features)
```

Evaluate:

```python
accuracy = accuracy_score(
    y_test,
    final_pred
)

print(accuracy)
```

That is a complete manually implemented blending classifier.

---

# 35. Blending With Regression

The exact same idea works for regression.

Suppose:

```python
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
```

Base models:

```python
models = [
    LinearRegression(),

    RandomForestRegressor(
        n_estimators=200,
        random_state=42
    ),

    SVR()
]
```

Train:

```python
for model in models:
    model.fit(X_base, y_base)
```

Generate meta features:

```python
meta_features = []

for model in models:
    pred = model.predict(X_meta)
    meta_features.append(pred)
```

Combine:

```python
meta_features = np.column_stack(meta_features)
```

Train:

```python
meta_model = LinearRegression()

meta_model.fit(
    meta_features,
    y_meta
)
```

Then generate test features and predict through the meta-model.

---

# 36. Classification: Why `predict_proba()`?

Suppose three models produce:

```text
Model A → class 1
Model B → class 1
Model C → class 0
```

We've lost information.

But probabilities give:

```text
Model A → 0.51
Model B → 0.99
Model C → 0.49
```

These contain much more information.

The meta-model can distinguish:

```text
0.51
```

from:

```text
0.99
```

even though both might predict class 1.

Therefore, for many classification blending approaches:

```python
predict_proba()
```

is useful.

---

# 37. Multiclass Blending

Suppose:

```text
3 classes
```

Each model produces:

```text
Model A:
[0.70, 0.20, 0.10]

Model B:
[0.60, 0.30, 0.10]

Model C:
[0.80, 0.10, 0.10]
```

Instead of three meta features, you can have:

```text
9 meta features
```

because each model contributes three probabilities.

```text
A_class1
A_class2
A_class3
B_class1
B_class2
B_class3
C_class1
C_class2
C_class3
```

The meta-model learns from these.

---

# 38. Choosing Base Models

As with stacking, don't blindly add models.

Suppose:

```text
Logistic Regression
Random Forest
XGBoost
SVM
KNN
Decision Tree
Naive Bayes
```

You don't necessarily want all of them.

Consider:

### Performance

Does the model perform reasonably well?

### Diversity

Does it make different errors?

### Computational cost

Is training it expensive?

### Prediction quality

Are its probabilities meaningful and useful?

---

# 39. Correlated Models

Suppose:

```text
Random Forest
Extra Trees
Random Forest with slightly different parameters
```

These models may make highly similar predictions.

Adding all of them may not provide much new information.

A more diverse ensemble might contain:

```text
Logistic Regression
Random Forest
SVM
Gradient Boosting
```

because they learn in substantially different ways.

---

# 40. Advantages of Blending

### 1. Simple

The concept is easier than cross-validated stacking.

### 2. Faster

No need to generate OOF predictions through multiple folds.

### 3. Flexible

You can use completely different models.

### 4. Can improve generalization

If the base models make complementary errors, combining them can improve predictions.

### 5. Easy to experiment with

You can quickly compare:

```text
LR + RF
LR + RF + SVM
LR + RF + XGBoost
RF + SVM + XGBoost
```

---

# 41. Disadvantages

### 1. Uses a holdout set

Some training data is sacrificed for meta-model training.

### 2. Meta-model may have limited data

If you reserve only 10% of the data, the meta-model has only that portion to learn from.

### 3. Sensitive to the split

A different validation split may produce different meta-model results.

### 4. Can overfit

Especially if:

```text
small meta-validation set
+
many base models
+
complex meta-model
```

### 5. More complex than a single model

Deployment and debugging become more complicated.

---

# 42. Blending vs Voting vs Stacking

This is probably the most useful comparison for your ensemble-learning notes.

| Technique       | Base Models       | Combination                      |
| --------------- | ----------------- | -------------------------------- |
| Voting          | Multiple          | Vote/average                     |
| Weighted Voting | Multiple          | Weighted vote                    |
| Blending        | Multiple          | Holdout predictions → meta-model |
| Stacking        | Multiple          | OOF predictions → meta-model     |
| Bagging         | Multiple          | Aggregate predictions            |
| Boosting        | Sequential models | Correct previous errors          |

The key distinction:

```text
Voting
   ↓
Direct combination

Blending
   ↓
Holdout predictions
   ↓
Meta-model

Stacking
   ↓
OOF predictions
   ↓
Meta-model
```

---

# 43. Blending vs Stacking — Visual

### Blending

```text
                  DATA
                   │
          ┌────────┴────────┐
          │                 │
       Base Train       Meta Validation
          │                 │
          ▼                 │
     ┌────┼────┐            │
     ▼    ▼    ▼            │
    LR    RF   SVM           │
     │    │    │            │
     └────┼────┘            │
          │                  │
          └──── Predictions ─┘
                   │
                   ▼
              Meta-model
                   │
                   ▼
              Final output
```

### Stacking

```text
                  DATA
                   │
                   ▼
             Cross Validation
                   │
          ┌────────┼────────┐
          ▼        ▼        ▼
        Fold 1   Fold 2   Fold 3 ...
          │        │        │
          └────────┼────────┘
                   ▼
             OOF Predictions
                   │
                   ▼
               Meta-model
                   │
                   ▼
              Final output
```

---

# 44. When Would You Choose Blending?

Blending can be attractive when:

* you want a simple ensemble;
* training speed matters;
* the dataset is reasonably large;
* you can afford a dedicated validation set;
* you want to experiment quickly with multiple models.

For example:

```text
Dataset = 100,000 rows
```

Using:

```text
70% base training
15% meta validation
15% test
```

leaves plenty of data for each stage.

---

# 45. When Is Stacking More Attractive?

If your dataset is relatively small:

```text
1000 rows
```

giving away:

```text
200 rows
```

for the meta-validation set may be costly.

Stacking with OOF predictions can make better use of the training data.

So conceptually:

```text
Large dataset
    ↓
Blending can be convenient

Smaller dataset
    ↓
Stacking may use data more efficiently
```

This isn't an absolute rule; model behavior and validation design still matter.

---

# 46. A Real-World Example

Imagine you're building a house-price prediction system.

Base models:

```text
Linear Regression
Random Forest
XGBoost
SVR
```

For one house:

```text
Linear Regression → 210k
Random Forest     → 230k
XGBoost           → 225k
SVR               → 218k
```

A simple average:

$$
\frac{210+230+225+218}{4}=220.75k
$$

So:

```text
Simple blending → $220,750
```

A meta-model might learn:

```text
Linear Regression → weight 0.10
Random Forest     → weight 0.30
XGBoost           → weight 0.45
SVR               → weight 0.15
```

Then it can produce a different final prediction based on learned relationships.

The weights here are illustrative; the actual meta-model would learn them from data.

---

# 47. The Core Mathematical Idea

Let there be \(K\) base models:

$$
f_1(X),f_2(X),...,f_K(X)
$$

For the validation set, calculate:

$$
Z =
[
f_1(X_{val}),
f_2(X_{val}),
...,
f_K(X_{val})
]
$$

Then train:

$$
g(Z,y_{val})
$$

where \(g\) is the meta-model.

For a new sample \(X^*\):

$$
Z^*=
[
f_1(X^*),
f_2(X^*),
...,
f_K(X^*)
]
$$

Final prediction:

$$
\boxed{
\hat{y}=g(Z^*)
}
$$

That's blending mathematically.

---

# 48. The Most Important Concepts to Remember

If you're preparing your ML notes, remember these **8 points**:

### 1. Blending is an ensemble technique

Multiple models contribute to one final prediction.

### 2. It has two levels

```text
Base models
     ↓
Meta-model
```

### 3. Base models train on one subset

```text
Base training set
```

### 4. Meta-model learns from a separate holdout set

```text
Meta validation set
```

### 5. Avoid leakage

Never generate meta-training predictions from samples the base model trained on.

### 6. Classification usually uses probabilities

```python
predict_proba()
```

### 7. Stacking and blending are closely related

The major difference is:

```text
Stacking → OOF predictions
Blending → Holdout predictions
```

### 8. Blending sacrifices some training data

The holdout set is unavailable for base-model training.

---

# 49. Your Ensemble Learning Map

Since you've now covered **Voting → Bagging → Random Forest → Boosting → XGBoost → Stacking → Blending**, the overall picture is:

```text
                         ENSEMBLE LEARNING
                                │
             ┌──────────────────┼──────────────────┐
             │                  │                  │
          BAGGING            BOOSTING          META-ENSEMBLES
             │                  │                  │
             │                  │           ┌──────┴──────┐
             │                  │           │             │
       Random Forest        AdaBoost     Stacking      Blending
             │             Gradient
             │             Boosting
             │                │
             │              XGBoost
             │
             └─────────────────────────────────────
```

And another useful way:

```text
Voting
  │
  └── Directly combine predictions

Bagging
  │
  └── Multiple models + aggregation

Boosting
  │
  └── Sequential models improving previous errors

Stacking
  │
  └── OOF predictions → Meta-model

Blending
  │
  └── Holdout predictions → Meta-model
```

**The single most important distinction for your next study session is `Stacking vs Blending`:**

> **Stacking uses cross-validation to generate out-of-fold predictions for the meta-model; blending holds out a validation set, trains the base models on the remaining data, and uses their predictions on that holdout set to train the meta-model.**
