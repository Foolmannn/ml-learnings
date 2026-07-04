`pandas-profiling` (now maintained as **`ydata-profiling`**) is one of the best tools for **data understanding** during the initial stage of a Machine Learning project. Instead of manually writing dozens of Pandas commands, it generates a complete exploratory data analysis (EDA) report with a single command.

---

# Why use Pandas Profiling?

Suppose you have a dataset:

```python
import pandas as pd

df = pd.read_csv("titanic.csv")
```

Normally, you would inspect it using many commands:

```python
df.head()
df.info()
df.describe()
df.isnull().sum()
df.duplicated().sum()
df.corr()
df.nunique()
```

Pandas Profiling combines all of this into one interactive HTML report.

---

# Installation

The old package

```bash
pip install pandas-profiling
```

is no longer actively maintained.

Use the newer package instead:

```bash
pip install ydata-profiling
```

or

```bash
conda install -c conda-forge ydata-profiling
```

---

# Basic Usage

```python
import pandas as pd
from ydata_profiling import ProfileReport

df = pd.read_csv("titanic.csv")

profile = ProfileReport(df)

profile.to_notebook_iframe()
```

In Jupyter Notebook, it displays an interactive report.

---

To save it as HTML:

```python
profile.to_file("titanic_report.html")
```

Open the HTML file in any browser.

---

# Example

```python
import pandas as pd
from ydata_profiling import ProfileReport

df = pd.read_csv("train.csv")

profile = ProfileReport(
    df,
    title="Titanic Dataset Report",
    explorative=True
)

profile.to_notebook_iframe()
```

---

# What does the report contain?

## 1. Overview

At the top, you'll see summary statistics such as:

```
Rows: 891
Columns: 12
Missing cells: 866
Duplicate rows: 0
Memory usage: 84 KB
```

This gives a quick overview of the dataset.

---

## 2. Alerts

One of the most useful sections.

Example:

```
⚠ Age has 177 missing values

⚠ Cabin has 687 missing values

⚠ Fare is highly skewed

⚠ PassengerId is unique

⚠ SibSp highly correlated with Parch
```

This instantly highlights issues that may need attention.

---

## 3. Variable Types

The report categorizes columns into types:

```
Numeric
Categorical
Boolean
Date
Text
Unsupported
```

For example:

```
Age → Numeric

Sex → Categorical

Embarked → Categorical

Ticket → Text
```

---

## 4. Individual Column Analysis

Clicking a variable opens detailed statistics.

### Numeric Column

Example: `Age`

You'll see:

* Mean
* Median
* Standard deviation
* Variance
* Min
* Max
* Quartiles
* Missing percentage
* Zeros
* Infinite values

It also includes:

* Histogram
* Distribution plot
* Box plot
* Quantiles

---

### Categorical Column

Example: `Sex`

Displays:

```
Male: 577

Female: 314
```

Also shows:

* Frequency table
* Missing values
* Number of unique values
* Bar chart

---

## 5. Missing Values

Shows:

* Count
* Percentage
* Matrix
* Heatmap
* Bar chart

Example:

```
Cabin

Missing:
77%

Present:
23%
```

Useful for deciding whether to drop or impute missing values.

---

## 6. Correlation Analysis

Computes correlations using methods like:

* Pearson
* Spearman
* Kendall
* Phi-k
* Cramér's V (for categorical variables)

Example:

```
Age ↔ Fare = 0.18

SibSp ↔ Parch = 0.41
```

This helps identify multicollinearity.

---

## 7. Interactions

Displays scatter plots between pairs of variables.

Example:

```
Age vs Fare

Fare vs Pclass

Age vs Pclass
```

Useful for spotting relationships and clusters.

---

## 8. Duplicate Rows

Shows:

```
Duplicates: 15
```

You can inspect which rows are duplicates.

---

## 9. Sample Data

Shows:

* First rows
* Last rows
* Random rows

This is similar to using:

```python
df.head()
df.tail()
df.sample()
```

---

## 10. Histograms

Each numeric variable includes a histogram.

Example:

```
Age

▁▂▃▄▅▆▇
```

This helps identify:

* Normal distributions
* Right-skewed distributions
* Left-skewed distributions
* Outliers

---

## 11. Boxplots

Automatically generated for numeric columns to visualize:

* Median
* Quartiles
* Outliers

---

## 12. Value Counts

For categorical variables:

```
Embarked

S 644

C 168

Q 77

Missing 2
```

---

## 13. Extreme Values

Lists:

* Largest values
* Smallest values

Useful for detecting potential outliers.

---

## 14. Memory Usage

Reports memory consumption for each column.

Example:

```
Object columns consume most memory.
```

This can guide optimizations, such as converting object columns to the `category` dtype.

---

# Advanced Options

## Minimal Report

For large datasets:

```python
profile = ProfileReport(
    df,
    minimal=True
)
```

This skips some computationally expensive analyses.

---

## Explorative Mode

```python
profile = ProfileReport(
    df,
    explorative=True
)
```

Adds more detailed statistics and visualizations.

---

## Dark Theme

```python
profile = ProfileReport(
    df,
    theme="flatly"
)
```

Several themes are available to customize the report's appearance.

---

# Working with Large Datasets

If your dataset contains millions of rows:

```python
profile = ProfileReport(
    df,
    minimal=True,
    samples=None,
    correlations=None
)
```

This reduces computation time by skipping samples and correlation calculations.

---

# Advantages

* Generates a comprehensive report with one command.
* Automatically detects missing values, duplicates, and outliers.
* Visualizes distributions and relationships.
* Provides correlation analysis.
* Highlights data quality issues through alerts.
* Saves significant time during exploratory data analysis.

---

# Limitations

* Can be slow on very large datasets.
* High memory usage for wide or large tables.
* Not ideal for datasets with millions of rows unless using `minimal=True`.
* Automatically generated insights still require interpretation; it does not replace thoughtful analysis.

---

# Typical Data Understanding Workflow

A common workflow at the start of an ML project is:

```python
import pandas as pd
from ydata_profiling import ProfileReport

# Load data
df = pd.read_csv("data.csv")

# Generate profile report
profile = ProfileReport(
    df,
    title="Dataset Profiling Report",
    explorative=True
)

# Display in Jupyter
profile.to_notebook_iframe()

# Save as HTML
profile.to_file("data_profile.html")
```

This report helps answer key questions before modeling:

* How many rows and columns are there?
* Which columns have missing values?
* Are there duplicate records?
* What are the data types of each feature?
* Are there outliers or highly skewed distributions?
* Which features are highly correlated?
* Which variables are categorical versus numerical?

Using `ydata-profiling` at the beginning of a project provides a fast, comprehensive understanding of your dataset and helps guide data cleaning, feature engineering, and model selection.
