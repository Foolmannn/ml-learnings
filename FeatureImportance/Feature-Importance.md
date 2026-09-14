# Feature Importance in Machine Learning — In Detail

**Feature importance** tells us **which input features contribute the most to a machine learning model's predictions**.

Suppose you're predicting house prices:

| Feature | Example |
|---|---:|
| Area | 1800 sq.ft |
| Bedrooms | 3 |
| Bathrooms | 2 |
| Location | Kathmandu |
| Age | 8 years |

Feature importance helps answer:

> **Which of these features is most influential in predicting the house price?**

For example:

```text
Area        → 0.52
Location    → 0.25
Bathrooms   → 0.12
Bedrooms    → 0.07
Age         → 0.04
```

Here, **Area** is the most important feature according to that particular importance method/model.

---

# 1. Why Feature Importance Is Important

Feature importance is useful for several reasons.

### 1. Understanding the model

Instead of treating the model as a black box, we can understand what it is using.

For example:

```text
Model prediction:
House price = $250,000

Important features:
Area      → very important
Location  → important
Bedrooms  → moderately important
Age       → less important
```

---

### 2. Feature selection

If you have 100 features and only 20 are useful, you may be able to remove some irrelevant features.

```text
100 features
     ↓
Feature importance
     ↓
Select important features
     ↓
20 features
     ↓
Train model
```

This can reduce:

- training time
- memory usage
- model complexity
- noise

---

### 3. Detecting irrelevant features

Suppose you're predicting whether a customer will purchase something:

```text
age             → 0.32
income          → 0.28
previous_buying → 0.21
location        → 0.12
customer_id     → 0.00
```

`customer_id` probably isn't useful for prediction.

---

### 4. Model debugging

Feature importance can reveal suspicious behavior.

Imagine a medical model predicts disease using:

```text
age              → 0.10
blood_pressure   → 0.20
cholesterol      → 0.15
patient_id       → 0.55
```

That's a warning sign.

The model may be learning something related to the dataset rather than the actual disease.

---

# 2. Important Concept: There Is No Single Feature Importance

This is extremely important.

**Feature importance is not one universal algorithm.**

Different models and techniques calculate importance differently.

Major approaches include:

```text
Feature Importance
│
├── 1. Tree-based Feature Importance
│      ├── Decision Tree
│      ├── Random Forest
│      ├── Extra Trees
│      └── Gradient Boosting
│
├── 2. Permutation Importance
│
├── 3. Coefficient-based Importance
│      ├── Linear Regression
│      ├── Logistic Regression
│      └── Linear SVM
│
└── 4. SHAP
       └── Model explanations
```

Let's understand each.

---

# 3. Tree-Based Feature Importance

You have recently studied **Decision Trees, Bagging and Random Forest**, so this is particularly important.

Decision trees make decisions using features.

For example:

```text
             Age > 30?
             /       \
           Yes        No
           /           \
      Income > 50k?    No
```

The tree chooses features that provide useful splits.

A feature used for a very useful split will generally receive higher importance.

---

# 4. How Decision Tree Calculates Feature Importance

Decision trees commonly use **impurity reduction**.

For classification, one common impurity measure is **Gini impurity**.

### Gini impurity

$$
Gini = 1-\sum_{k=1}^{K}p_k^2
$$

For example, suppose a node contains:

```text
10 samples

6 → Class A
4 → Class B
```

Then:

$$
Gini = 1-(0.6^2+0.4^2)
$$

$$
=1-(0.36+0.16)
$$

$$
=0.48
$$

A good split reduces impurity.

---

## Impurity reduction

Suppose:

```text
Parent
Gini = 0.50

       Feature X
       /       \
      /         \
Child 1        Child 2
Gini=.10       Gini=.20
```

The tree measures how much the split improved the purity.

A simplified form is:

$$
\Delta I =
I(parent)
-
\left(
\frac{n_L}{n}I(L)
+
\frac{n_R}{n}I(R)
\right)
$$

where:

- $I(parent)$ = impurity before split
- $I(L)$ = left-child impurity
- $I(R)$ = right-child impurity
- $n_L$ = samples in left child
- $n_R$ = samples in right child
- $n$ = total samples

The more impurity reduction a feature produces, the more important it becomes.

---

# 5. Getting Feature Importance from Decision Tree

Example:

```python
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(random_state=42)

model.fit(X_train, y_train)

print(model.feature_importances_)
```

Example output:

```text
[0.10, 0.55, 0.20, 0.15]
```

Suppose:

```python
X.columns
```

gives:

```text
['age', 'income', 'education', 'experience']
```

Then:

| Feature | Importance |
|---|---:|
| age | 0.10 |
| income | 0.55 |
| education | 0.20 |
| experience | 0.15 |

Therefore:

```text
income > education > experience > age
```

---

# 6. Important Property of `feature_importances_`

For tree-based models:

```python
model.feature_importances_
```

usually sums to:

$$
1
$$

Example:

```text
0.10 + 0.55 + 0.20 + 0.15 = 1.0
```

These values are relative importance scores.

**They are not probabilities.**

If:

```text
income = 0.55
```

you should NOT say:

> Income causes 55% of the prediction.

Instead say:

> Income accounts for 55% of the model's calculated feature-importance score under this importance method.

That's an important distinction.

---

# 7. Random Forest Feature Importance

Now let's connect this to **Random Forest**.

A Random Forest contains many decision trees:

```text
             Random Forest
                  │
      ┌───────────┼───────────┐
      ↓           ↓           ↓
    Tree 1      Tree 2      Tree 3
      ↓           ↓           ↓
    ...         ...         ...
      └───────────┼───────────┘
                  ↓
              Prediction
```

Every tree calculates impurity reductions.

The Random Forest aggregates these across trees.

So:

```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf.fit(X_train, y_train)

importance = rf.feature_importances_

print(importance)
```

---

# 8. Visualizing Random Forest Feature Importance

A very useful approach:

```python
import pandas as pd
import matplotlib.pyplot as plt

importance_df = pd.DataFrame({
    "feature": X_train.columns,
    "importance": rf.feature_importances_
})

importance_df = importance_df.sort_values(
    "importance",
    ascending=False
)

print(importance_df)
```

Plot:

```python
plt.barh(
    importance_df["feature"],
    importance_df["importance"]
)

plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Feature Importance")

plt.gca().invert_yaxis()

plt.show()
```

You might get:

```text
income       ████████████████████
age          ███████████
education    ███████
experience   █████
location     ██
```

---

# 9. Problem With Tree-Based Feature Importance

This is one of the most important things to understand.

`feature_importances_` uses **Mean Decrease in Impurity (MDI)**.

It can be biased toward certain types of features.

For example, it can favor:

### High-cardinality features

Suppose you have:

```text
age
income
gender
customer_id
```

A feature like `customer_id` has many unique values.

A tree can potentially find many possible splits.

This can cause impurity-based importance to overestimate its usefulness.

---

# 10. Correlated Features Problem

Another major issue.

Suppose:

```text
Feature A = height in centimeters

Feature B = height in meters
```

They're almost perfectly correlated.

Both contain essentially the same information.

A tree may choose:

```text
Feature A → 0.80
Feature B → 0.05
```

even though both are predictive.

Why?

Because once the tree uses Feature A, Feature B doesn't provide much additional impurity reduction.

So:

> **Low feature importance does not necessarily mean the feature contains no useful information.**

It may simply mean another correlated feature got credit for that information.

---

# 11. Permutation Importance

This is one of the most useful general-purpose methods.

The basic idea is extremely intuitive.

Suppose your trained model has:

```text
Accuracy = 90%
```

Now randomly shuffle one feature.

For example:

```text
Original income:

50k
70k
40k
90k
60k
```

Shuffle:

```text
90k
40k
60k
50k
70k
```

Everything else stays unchanged.

Then evaluate the model again.

Suppose:

```text
Original accuracy = 90%

After shuffling income:
Accuracy = 65%
```

The performance dropped significantly.

Therefore:

> Income was important to the model.

---

# 12. Permutation Importance Formula

A common definition is:

$$
Importance_j =
Score_{original}
-
Score_{permuted,j}
$$

For example:

```text
Original accuracy = 0.90

After permutation:
income = 0.65
age = 0.87
gender = 0.89
```

Then:

```text
income → 0.25
age    → 0.03
gender → 0.01
```

Therefore:

```text
income is much more important
```

---

# 13. Implementing Permutation Importance

Scikit-learn provides:

```python
from sklearn.inspection import permutation_importance
```

Example:

```python
result = permutation_importance(
    rf,
    X_test,
    y_test,
    n_repeats=10,
    random_state=42
)
```

Then:

```python
importance_df = pd.DataFrame({
    "feature": X_test.columns,
    "importance": result.importances_mean
})

importance_df = importance_df.sort_values(
    "importance",
    ascending=False
)

print(importance_df)
```

You can also inspect variation:

```python
result.importances_std
```

So:

```text
Feature       Mean       Std
income        0.25       0.02
age           0.04       0.01
education     0.02       0.01
gender        0.00       0.01
```

---

# 14. Why `n_repeats` Is Important

When you shuffle a feature once, you might get a lucky/unlucky permutation.

Therefore we repeat it.

```python
n_repeats=10
```

means approximately:

```text
income
 ↓
shuffle → score
shuffle → score
shuffle → score
...
10 times
 ↓
average
```

Increasing `n_repeats` generally gives a more stable estimate but costs more computation.

For serious analysis, something like:

```python
n_repeats=20
```

or more can be useful.

---

# 15. Negative Permutation Importance

You may encounter:

```text
income       0.25
age          0.05
education   -0.01
```

What does negative importance mean?

After shuffling `education`, the model performed **slightly better**.

This can happen because of:

- random variation
- noise
- overfitting
- correlated features
- small datasets

It doesn't automatically mean:

> "Education is harmful."

It means the feature didn't provide reliable predictive value under that permutation experiment, and the observed change happened to be slightly positive.

---

# 16. Coefficient-Based Feature Importance

For models such as:

- Linear Regression
- Logistic Regression
- Linear SVM

we can examine coefficients.

Suppose:

$$
y = \beta_0+\beta_1x_1+\beta_2x_2+\beta_3x_3
$$

The coefficients tell us how the features influence the prediction.

Example:

```text
age          = 0.20
income       = 1.50
experience   = 0.80
```

Larger absolute coefficients generally indicate stronger influence **if the features are on comparable scales**.

---

# 17. Why Scaling Matters for Coefficient Importance

This is extremely important.

Suppose:

```text
age:
20–60

income:
20,000–200,000
```

If you compare raw coefficients directly, the comparison may be misleading because the features have different scales.

For example:

```text
age coefficient    = 500
income coefficient = 0.02
```

You cannot conclude:

```text
age is more important than income
```

just because:

$$
500 > 0.02
$$

The units are different.

---

# 18. Standardization for Coefficient Comparison

Use:

```python
from sklearn.preprocessing import StandardScaler
```

Then:

```python
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)
```

Train the model:

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

model.fit(X_scaled, y)
```

Then:

```python
importance = abs(model.coef_[0])
```

Example:

```text
income       1.80
age          0.90
experience   0.40
```

Now comparing absolute coefficients is much more meaningful because the features are standardized.

---

# 19. Positive and Negative Coefficients

Coefficient-based models provide something tree importance usually doesn't:

**direction.**

Suppose logistic regression gives:

```text
income       +1.5
age          -0.8
debt         +2.1
```

The sign tells us the direction of the relationship in the model:

```text
income → positive contribution
age    → negative contribution
debt   → positive contribution
```

But be careful:

> A coefficient's sign and magnitude describe the model's fitted relationship, not necessarily causation.

---

# 20. SHAP Feature Importance

SHAP is a more advanced and powerful explanation technique.

SHAP stands for:

**SHapley Additive exPlanations**

It is based on concepts from **cooperative game theory**.

The basic idea:

> Treat features like players cooperating to produce a prediction.

Suppose the model predicts:

```text
House price = $300,000
```

SHAP might explain it as:

```text
Base prediction             $200,000
Area contribution           +$70,000
Location contribution       +$40,000
Age contribution            -$10,000
Bedrooms contribution       +$0
----------------------------------
Final prediction            $300,000
```

This is much more detailed than simply saying:

```text
Area importance = 0.45
```

---

# 21. Global vs Local Feature Importance

This distinction is extremely important.

## Global importance

Answers:

> Which features are generally important for the model?

Example:

```text
income       0.40
age          0.25
experience   0.20
education    0.15
```

This describes the model overall.

---

## Local importance

Answers:

> Why did the model make this particular prediction?

Example:

For one customer:

```text
Base prediction       0.50

income                +0.25
age                   +0.10
previous_purchase     +0.20
location              -0.05
--------------------------------
Prediction             1.00
```

SHAP is particularly useful for this type of explanation.

---

# 22. SHAP Global Importance

Suppose we calculate SHAP values for many samples.

We can calculate something like:

$$
Mean(|SHAP_j|)
$$

for each feature.

Example:

```text
Feature         Mean |SHAP|
income             0.42
age                0.25
education          0.15
experience         0.08
```

This gives a global view of how much each feature contributes to predictions.

---

# 23. Feature Importance vs Feature Selection

Don't confuse these two.

### Feature importance

Answers:

> Which features are important?

### Feature selection

Answers:

> Which features should I keep?

For example:

```text
100 features
     ↓
calculate importance
     ↓
top 20 features
     ↓
train model using those 20
```

Feature importance can therefore be used as part of feature selection.

But you should be careful not to select features using the entire dataset before cross-validation, because that can cause **data leakage**.

---

# 24. Feature Importance and Data Leakage

Suppose you do:

```python
importance = model.fit(X, y)
```

using the entire dataset.

Then choose the top features.

Then perform cross-validation.

This can leak information from validation folds into feature selection.

A better approach is to perform feature selection **inside each training fold**, commonly using a pipeline.

Conceptually:

```text
Fold 1
Training data
   ↓
Feature selection
   ↓
Model
   ↓
Validation

Fold 2
Training data
   ↓
Feature selection
   ↓
Model
   ↓
Validation
```

This gives a more honest estimate.

---

# 25. Feature Importance and Correlation

Feature importance should not be confused with correlation.

Suppose:

```text
Feature A ↔ Target
Correlation = 0.80
```

This doesn't automatically mean:

```text
Feature importance = 0.80
```

Correlation measures a specific statistical relationship.

Feature importance measures how useful the feature is **to a particular predictive model/importance method**.

A feature can have:

```text
low linear correlation
```

but still be highly important to a nonlinear model.

For example:

$$
y=x^2
$$

If $x$ is symmetrically distributed around zero, its linear correlation with $y$ can be weak even though $x$ completely determines $y$.

---

# 26. Feature Importance Does Not Mean Causality

This is probably the most important conceptual warning.

Suppose:

```text
Ice cream sales → important
Temperature → important
Drowning → important
```

You cannot conclude:

> Ice cream causes drowning.

There may be a third variable:

```text
Hot weather
     │
 ┌───┴────┐
 ↓        ↓
Ice cream  Swimming
             ↓
          Drowning
```

Feature importance tells you:

> **The model found this feature useful for prediction.**

It does NOT tell you:

> **This feature causes the target.**

---

# 27. Comparing the Main Methods

| Method | Works with | Main idea | Main weakness |
|---|---|---|---|
| Tree `feature_importances_` | Tree models | Impurity reduction | Can be biased |
| Permutation Importance | Almost any model | Shuffle feature and measure performance drop | Can struggle with correlated features |
| Coefficients | Linear models | Magnitude of coefficients | Scale-sensitive |
| SHAP | Many model types | Contribution of each feature | More computationally expensive |

---

# 28. Example: Random Forest + Three Methods

Let's say we have:

```python
X_train
y_train
X_test
y_test
```

Train:

```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

rf.fit(X_train, y_train)
```

### Method 1 — Built-in importance

```python
rf.feature_importances_
```

---

### Method 2 — Permutation

```python
from sklearn.inspection import permutation_importance

result = permutation_importance(
    rf,
    X_test,
    y_test,
    n_repeats=10,
    random_state=42
)

result.importances_mean
```

---

### Method 3 — SHAP

For supported models, you can use SHAP:

```python
import shap

explainer = shap.TreeExplainer(rf)

shap_values = explainer.shap_values(X_test)
```

The exact SHAP output structure can vary by SHAP version and model type, so inspect the returned object before plotting.

---

# 29. Why Different Methods Can Give Different Results

Suppose:

```text
Feature        Tree MDI    Permutation
income           0.45         0.30
age              0.25         0.35
education        0.20         0.20
experience       0.10         0.15
```

Which one is correct?

There isn't necessarily a single "correct" answer.

They answer slightly different questions.

### MDI asks:

> How much did this feature reduce impurity across the trees?

### Permutation asks:

> How much does model performance suffer when this feature's information is destroyed?

### SHAP asks:

> How much does this feature contribute to individual predictions, aggregated across observations?

That's why **understanding the method is more important than blindly looking at an importance number.**

---

# 30. Feature Importance with Random Forest and OOB

Since you recently studied **OOB score**, there's an interesting connection.

Random Forest can use:

**Out-of-Bag samples**

to estimate model performance.

Permutation importance can also be calculated using OOB predictions in certain implementations/workflows.

Conceptually:

```text
Bootstrap sample
      ↓
Train tree
      ↓
Some samples left out
      ↓
OOB samples
      ↓
Evaluate
      ↓
Permute feature
      ↓
Evaluate again
      ↓
Performance decrease
      ↓
Feature importance
```

This avoids needing a separate validation set for that importance calculation.

---

# 31. Feature Importance in Regression

Everything we've discussed applies to regression too.

Suppose:

```python
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

rf.fit(X_train, y_train)

importance = rf.feature_importances_
```

The concept is the same, but tree impurity for regression commonly uses **variance/MSE reduction** rather than Gini impurity.

---

# 32. Feature Importance with XGBoost / Gradient Boosting

Tree-based boosting models also provide feature importance.

For example, models from libraries such as:

- XGBoost
- LightGBM
- CatBoost

can provide different importance measures.

For example, XGBoost can report concepts such as:

```text
gain
weight
cover
```

These don't all mean the same thing.

### Gain

How much a feature improves the objective when used for splitting.

### Weight

How frequently a feature is used for splitting.

### Cover

How many observations are affected by splits involving the feature.

So even within one model family:

> **"Feature importance" can have multiple definitions.**

---

# 33. Feature Importance in Neural Networks

Neural networks don't normally have a simple:

```python
model.feature_importances_
```

like Random Forest.

Why?

Because information flows through many layers:

```text
Input
 ↓
Dense
 ↓
Dense
 ↓
Dense
 ↓
Output
```

A feature's influence isn't represented by a single straightforward coefficient.

Methods such as:

- permutation importance
- SHAP
- Integrated Gradients
- saliency methods

can be used instead.

---

# 34. Feature Importance Workflow

A practical workflow looks like this:

```text
             Dataset
                ↓
          Train/test split
                ↓
          Train model
                ↓
       ┌────────┴────────┐
       ↓                 ↓
 Tree importance   Permutation
       ↓                 ↓
       └────────┬────────┘
                ↓
          Compare results
                ↓
      Investigate important
            features
                ↓
     Feature selection if needed
                ↓
       Retrain + evaluate
                ↓
          Final model
```

---

# 35. Important Mistakes to Avoid

### ❌ Mistake 1

Assuming:

```text
importance = causality
```

Wrong.

---

### ❌ Mistake 2

Assuming low importance means the feature is useless.

Correlated features can hide importance.

---

### ❌ Mistake 3

Comparing raw linear coefficients without scaling.

```text
coef(age) = 500
coef(income) = 0.02
```

doesn't mean age is more important.

---

### ❌ Mistake 4

Using only one importance method.

For important projects, compare:

```text
MDI
Permutation
SHAP
```

when appropriate.

---

### ❌ Mistake 5

Doing feature selection before cross-validation.

This can cause data leakage.

---

### ❌ Mistake 6

Assuming feature importance is stable.

If you retrain the model on another sample, importance values can change.

You should examine stability when feature selection or interpretation matters.

---

# 36. A Good Practical Example

Suppose you're building a customer churn model:

```text
Features:

age
monthly_charge
contract_length
tenure
support_calls
payment_method
```

Random Forest gives:

```text
contract_length    0.31
tenure             0.25
monthly_charge     0.20
support_calls      0.14
age                0.06
payment_method     0.04
```

You might conclude:

```text
contract_length
       ↓
   strongest
       ↓
tenure
       ↓
monthly_charge
       ↓
support_calls
       ↓
age
       ↓
payment_method
```

Then use permutation importance:

```text
contract_length    0.21
tenure             0.18
monthly_charge     0.08
support_calls      0.07
age                0.01
payment_method    -0.01
```

Now you notice something interesting:

```text
MDI says:
monthly_charge = 0.20

Permutation says:
monthly_charge = 0.08
```

This is a signal to investigate the relationship between these features, model behavior, and possibly correlated variables.

---

# 37. What You Should Learn for ML

Since you're currently studying **ensemble learning → Bagging → Random Forest**, I'd learn feature importance in this order:

### Level 1 — Must know

```text
1. What feature importance means
2. Decision Tree feature importance
3. Random Forest feature importance
4. feature_importances_
5. MDI / impurity reduction
```

### Level 2 — Very important

```text
6. Permutation importance
7. n_repeats
8. mean importance
9. standard deviation
10. negative permutation importance
```

### Level 3 — Important for interpretation

```text
11. Correlated features
12. High-cardinality bias
13. Feature importance ≠ causality
14. Feature importance ≠ correlation
15. Feature selection
16. Data leakage
```

### Level 4 — Advanced

```text
17. SHAP
18. Global explanations
19. Local explanations
20. SHAP values
21. Model-specific vs model-agnostic explanations
```

---

# 38. The Most Important Mental Model

Remember this:

```text
Feature Importance
       │
       ├── "How useful is this feature
       │    to this model?"
       │
       └── NOT
            │
            ├── "Does this feature cause the target?"
            │
            ├── "Is this feature correlated with target?"
            │
            └── "Is this feature universally important?"
```

And for your current **Random Forest/Ensemble Learning** studies, remember the three most important techniques:

```text
                    Feature Importance
                           │
          ┌────────────────┼────────────────┐
          ↓                ↓                ↓
      MDI / Built-in   Permutation         SHAP
      importance       importance
          │                │                │
    Tree impurity      Performance       Prediction
       reduction        decrease         contribution
```

**If you're working with Random Forest, don't stop at `rf.feature_importances_`.** Understanding **Permutation Importance + correlated-feature behavior + SHAP** will give you a much stronger understanding of model interpretability.