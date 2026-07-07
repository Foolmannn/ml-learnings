# Encoding Categorical Data in Machine Learning

Many machine learning algorithms **only work with numerical data**. Categorical variables (such as gender, city, color, education level, etc.) must therefore be converted into numbers before training a model. This process is called **categorical encoding**.

Example dataset:

| Name    | Gender | City   | Purchased |
| ------- | ------ | ------ | --------- |
| Alice   | Female | London | Yes       |
| Bob     | Male   | Paris  | No        |
| Charlie | Male   | London | Yes       |
| David   | Female | Tokyo  | No        |

Here,

* **Gender** and **City** are categorical features.
* **Purchased** is the target.

---

# Types of Categorical Variables

Before choosing an encoding method, determine the type of categorical variable.

## 1. Nominal Data

Categories have **no natural order**.

Examples:

* Color: Red, Blue, Green
* City: London, Paris, Tokyo
* Country
* Blood Group

```
Red ≠ Blue ≠ Green
```

No category is greater or smaller than another.

---

## 2. Ordinal Data

Categories have a **meaningful order**.

Examples:

Education:

```
High School < Bachelor's < Master's < PhD
```

Customer Satisfaction:

```
Poor < Average < Good < Excellent
```

T-Shirt Size:

```
Small < Medium < Large < XL
```

The distance between categories is **not necessarily equal**.

---

# Why Not Simply Use Numbers?

Suppose we encode:

| Color | Number |
| ----- | ------ |
| Red   | 0      |
| Blue  | 1      |
| Green | 2      |

A model may incorrectly assume

```
Green > Blue > Red
```

or

```
Green - Red = 2
```

which is meaningless.

Therefore, choosing the correct encoding is important.

---

# Common Encoding Techniques

## 1. Label Encoding

Each unique category gets an integer.

Example:

| City   |
| ------ |
| London |
| Paris  |
| Tokyo  |

becomes

| City |
| ---- |
| 0    |
| 1    |
| 2    |

### Using Scikit-Learn

```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

df["City"] = le.fit_transform(df["City"])
```

Output:

```
London -> 0
Paris  -> 1
Tokyo  -> 2
```

---

### Advantages

* Very simple
* Fast
* Memory efficient

### Disadvantages

Creates an artificial ordering.

The model may assume

```
Tokyo > Paris > London
```

which is incorrect.

---

### When to Use

Good for:

* Ordinal features
* Target labels in classification

Example:

```
Poor -> 0
Average -> 1
Good -> 2
Excellent -> 3
```

Not recommended for nominal input features.

---

# 2. One-Hot Encoding (Dummy Encoding)

Most common encoding method.

Instead of assigning numbers, create one binary column for each category.

Original:

| City   |
| ------ |
| London |
| Paris  |
| Tokyo  |

Encoded:

| London | Paris | Tokyo |
| ------ | ----- | ----- |
| 1      | 0     | 0     |
| 0      | 1     | 0     |
| 0      | 0     | 1     |

---

### Scikit-Learn

```python
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(sparse_output=False)

encoded = encoder.fit_transform(df[["City"]])
```

---

### Pandas

```python
pd.get_dummies(df["City"])
```

Output

```
   London  Paris  Tokyo
0       1      0      0
1       0      1      0
2       1      0      0
3       0      0      1
```

---

### Advantages

* No false ordering
* Works well for nominal data
* Most widely used

### Disadvantages

Suppose

```
Country
```

has

```
250 categories
```

One-hot encoding creates

```
250 columns
```

This leads to:

* High memory usage
* Slower training
* Sparse matrices

This is called the **curse of dimensionality**.

---

### Best For

* Nominal variables
* Low-cardinality features (few unique categories)

Examples:

* Gender
* Color
* Day of week

---

# 3. Ordinal Encoding

Designed specifically for ordinal data.

Example

```
Small
Medium
Large
XL
```

Encode as

```
Small  -> 0
Medium -> 1
Large  -> 2
XL     -> 3
```

Scikit-Learn:

```python
from sklearn.preprocessing import OrdinalEncoder

encoder = OrdinalEncoder(
    categories=[["Small", "Medium", "Large", "XL"]]
)

df["Size"] = encoder.fit_transform(df[["Size"]])
```

---

### Advantages

* Preserves order
* Uses one column only

### Disadvantages

Should **not** be used for nominal features.

---

# 4. Frequency Encoding

Replace each category by its frequency.

Original

| City   |
| ------ |
| London |
| Paris  |
| London |
| Tokyo  |
| London |

Frequency:

```
London = 3
Paris = 1
Tokyo = 1
```

Encoded:

| City |
| ---- |
| 3    |
| 1    |
| 3    |
| 1    |
| 3    |

Python

```python
freq = df["City"].value_counts()

df["City"] = df["City"].map(freq)
```

---

### Advantages

* Handles many categories
* Uses one column
* Fast

### Disadvantages

Different categories with the same frequency become identical.

Example

```
Paris = 12
Tokyo = 12
```

Both become

```
12
```

The model cannot distinguish them.

---

# 5. Count Encoding

Very similar to frequency encoding.

Instead of percentages, use counts.

Example

```
Apple  -> 120
Orange -> 85
Banana -> 50
```

---

# 6. Target Encoding (Mean Encoding)

Replace each category with the average target value for that category.

Example

| City   | Purchased |
| ------ | --------- |
| London | 1         |
| London | 0         |
| Paris  | 1         |
| Tokyo  | 0         |

Means

```
London = 0.5
Paris  = 1
Tokyo  = 0
```

Encoded

| City |
| ---- |
| 0.5  |
| 0.5  |
| 1    |
| 0    |

Python

```python
means = df.groupby("City")["Purchased"].mean()

df["City"] = df["City"].map(means)
```

---

### Advantages

* Excellent for high-cardinality features
* Often improves predictive performance

### Disadvantages

* Can cause **data leakage** if computed using the entire dataset before splitting or without cross-validation.
* Needs smoothing or cross-validation to avoid overfitting.

---

# 7. Binary Encoding

Useful for features with many categories.

Steps:

Assign integers:

```
A -> 1
B -> 2
C -> 3
D -> 4
```

Convert integers to binary.

```
1 -> 001
2 -> 010
3 -> 011
4 -> 100
```

Produces fewer columns than one-hot encoding.

Useful for:

* Thousands of categories

Libraries:

```python
import category_encoders as ce

encoder = ce.BinaryEncoder(cols=["City"])

df = encoder.fit_transform(df)
```

---

# 8. Hash Encoding

Uses a hash function to map categories into a fixed number of columns.

Advantages:

* Memory efficient
* Works with huge datasets
* No need to store category mappings

Disadvantages:

* Different categories can map to the same column (hash collisions), which may lose some information.

---

# 9. Leave-One-Out Encoding

Similar to target encoding.

Instead of using the overall target mean for a category, each row is encoded using the category mean **excluding that row**.

This helps reduce overfitting compared to basic target encoding.

---

# 10. Embedding Encoding (Deep Learning)

Neural networks can learn dense vector representations (embeddings) for categories.

Example

```
London → [0.18, -0.42, 0.75]
Paris  → [0.03, 0.61, -0.10]
Tokyo  → [-0.55, 0.14, 0.87]
```

Advantages:

* Captures relationships between categories
* Very effective for high-cardinality features
* Common in recommendation systems and NLP

---

# Dummy Variable Trap

Suppose we have

| Gender |
| ------ |
| Male   |
| Female |

One-hot encoding gives

| Male | Female |
| ---- | ------ |
| 1    | 0      |
| 0    | 1      |

Since:

```
Female = 1 - Male
```

the columns are perfectly correlated. This is called the **dummy variable trap** (perfect multicollinearity), which can be problematic for linear models.

To avoid it, drop one column:

```python
pd.get_dummies(df, drop_first=True)
```

Output

| Male |
| ---- |
| 1    |
| 0    |

or

```python
encoder = OneHotEncoder(drop="first")
```

Tree-based models (e.g., Decision Trees, Random Forests, Gradient Boosting) are generally not affected by this issue, but dropping one level can still reduce dimensionality.

---

# Which Encoding Should You Use?

| Situation                              | Recommended Encoding                                                                        |
| -------------------------------------- | ------------------------------------------------------------------------------------------- |
| Ordinal categories                     | Ordinal Encoding                                                                            |
| Nominal with few categories            | One-Hot Encoding                                                                            |
| High-cardinality nominal               | Target, Binary, or Hash Encoding                                                            |
| Deep Learning                          | Embedding Encoding                                                                          |
| Classification target labels           | Label Encoding                                                                              |
| Tree-based models with many categories | Target, Frequency, Binary, or Hash Encoding (depending on leakage risk and library support) |

---

# Comparison

| Encoding  | Ordered?                     | Increases Columns? | Good for High Cardinality? |
| --------- | ---------------------------- | ------------------ | -------------------------- |
| Label     | Yes (artificial for nominal) | No                 | Yes                        |
| Ordinal   | Yes (true order)             | No                 | Yes                        |
| One-Hot   | No                           | Yes                | No                         |
| Frequency | No                           | No                 | Yes                        |
| Count     | No                           | No                 | Yes                        |
| Target    | Uses target information      | No                 | Yes                        |
| Binary    | No                           | Slightly           | Yes                        |
| Hash      | No                           | Fixed              | Yes                        |
| Embedding | Learned                      | No                 | Excellent                  |

---

# Practical Guidelines

1. **Identify the feature type first**:

   * **Ordinal** → Use Ordinal Encoding.
   * **Nominal** → Consider One-Hot Encoding for low-cardinality features.

2. **Consider cardinality**:

   * Low cardinality (e.g., < 10–20 unique values): One-Hot Encoding is usually a good choice.
   * High cardinality: Prefer Target, Binary, Frequency, or Hash Encoding.

3. **Match the encoding to the model**:

   * Linear models generally benefit from One-Hot Encoding for nominal features.
   * Tree-based models can often work well with Ordinal Encoding for ordered features and may also perform well with target or frequency-based encodings for high-cardinality variables.

4. **Avoid data leakage**:

   * Compute target-based encodings using only the training data, ideally within cross-validation folds.

5. **Handle unseen categories**:

   * Configure encoders to deal with categories that appear only in the test set. For example:

     ```python
     from sklearn.preprocessing import OneHotEncoder

     encoder = OneHotEncoder(handle_unknown="ignore")
     ```
   * This prevents errors during inference.

Choosing the right encoding depends on the **type of categorical feature**, the **number of unique categories**, the **machine learning algorithm**, and whether you need to balance predictive performance with memory and computational efficiency.
