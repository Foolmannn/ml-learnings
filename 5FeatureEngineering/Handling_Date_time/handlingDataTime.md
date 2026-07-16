# Handling Date and Time Variables in Machine Learning (Scikit-learn)

Date and time features are among the most valuable variables in machine learning because they often contain hidden patterns such as:

* Seasonality
* Trends
* Holidays
* Weekends
* Business hours
* Time differences
* Cyclic patterns (hour, month, weekday)

However, machine learning models **cannot directly understand dates**. Therefore, we convert datetime variables into meaningful numerical features.

---

# Why Can't Models Use Date Directly?

Suppose we have

| Order Date | Sales |
| ---------- | ----- |
| 2024-01-01 | 500   |
| 2024-01-02 | 620   |
| 2024-01-03 | 480   |

The model sees

```text
2024-01-01
```

as a string/object.

Algorithms like Linear Regression, Random Forest, Logistic Regression, and SVM require numeric inputs.

So we perform **feature engineering**.

---

# Step 1: Convert to Datetime

Always convert object/string columns into datetime format.

```python
import pandas as pd

df['OrderDate'] = pd.to_datetime(df['OrderDate'])
```

Example

Before

```python
print(df.dtypes)
```

```
OrderDate    object
Sales         int64
```

After

```
OrderDate    datetime64[ns]
Sales         int64
```

---

# Step 2: Extract Useful Features

Suppose

```python
df['OrderDate']
```

contains

```
2024-05-12
2024-08-25
2025-01-03
```

Pandas provides many useful datetime attributes through `.dt`.

---

## Year

```python
df['Year'] = df['OrderDate'].dt.year
```

Result

| OrderDate  | Year |
| ---------- | ---- |
| 2024-05-12 | 2024 |
| 2024-08-25 | 2024 |
| 2025-01-03 | 2025 |

---

## Month

```python
df['Month'] = df['OrderDate'].dt.month
```

Output

```
5
8
1
```

---

## Day

```python
df['Day'] = df['OrderDate'].dt.day
```

---

## Day of Week

```python
df['Weekday'] = df['OrderDate'].dt.dayofweek
```

Output

```
Monday = 0
Tuesday = 1
Wednesday = 2
Thursday = 3
Friday = 4
Saturday = 5
Sunday = 6
```

---

## Day Name

```python
df['DayName'] = df['OrderDate'].dt.day_name()
```

Output

```
Sunday
Monday
Friday
```

---

## Month Name

```python
df['MonthName'] = df['OrderDate'].dt.month_name()
```

Output

```
May
August
January
```

---

## Quarter

```python
df['Quarter'] = df['OrderDate'].dt.quarter
```

Output

```
Q1 → 1
Q2 → 2
Q3 → 3
Q4 → 4
```

---

## Week Number

```python
df['Week'] = df['OrderDate'].dt.isocalendar().week
```

---

## Day of Year

```python
df['DayOfYear'] = df['OrderDate'].dt.dayofyear
```

Output

```
132
240
3
```

---

## Is Weekend

```python
df['IsWeekend'] = df['OrderDate'].dt.dayofweek >= 5
```

Convert to integer if needed:

```python
df['IsWeekend'] = (df['OrderDate'].dt.dayofweek >= 5).astype(int)
```

Output

```
0
1
0
```

---

## Leap Year

```python
df['LeapYear'] = df['OrderDate'].dt.is_leap_year
```

---

# Working with Time

Suppose

```
2024-08-05 13:45:30
```

---

## Hour

```python
df['Hour'] = df['OrderDate'].dt.hour
```

---

## Minute

```python
df['Minute'] = df['OrderDate'].dt.minute
```

---

## Second

```python
df['Second'] = df['OrderDate'].dt.second
```

---

## Time of Day

```python
def get_period(hour):
    if hour < 12:
        return "Morning"
    elif hour < 17:
        return "Afternoon"
    elif hour < 21:
        return "Evening"
    return "Night"

df['Period'] = df['Hour'].apply(get_period)
```

---

# Time Difference Features

These are extremely useful.

Suppose

| Signup     | Purchase   |
| ---------- | ---------- |
| 2024-01-01 | 2024-01-08 |

```python
df['DaysTaken'] = (
    df['PurchaseDate'] - df['SignupDate']
).dt.days
```

Output

```
7
```

---

## Age of Data

Suppose today's date is

```
2026-07-16
```

```python
today = pd.Timestamp.today()

df['Age'] = (today - df['OrderDate']).dt.days
```

---

# Time Since Last Event

Example

```
Customer Purchases

Jan 1
Jan 5
Jan 20
```

```python
df['Gap'] = df['PurchaseDate'].diff().dt.days
```

Output

```
NaN
4
15
```

---

# Cyclical Features

Months, hours, weekdays, and minutes are **cyclic**, meaning the last value wraps around to the first. For example:

```
Hour

23
0
```

Although 23 and 0 are only one hour apart, a model may interpret them as far apart if encoded as plain integers.

To capture this circular nature, use sine and cosine transformations.

### Hours

```python
import numpy as np

df['Hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
df['Hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
```

### Month

```python
df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
```

### Weekday

```python
df['Weekday_sin'] = np.sin(2 * np.pi * df['Weekday'] / 7)
df['Weekday_cos'] = np.cos(2 * np.pi * df['Weekday'] / 7)
```

These encodings preserve the cyclic relationship.

---

# Holiday Features

Holidays often influence sales, traffic, and demand.

Example:

```python
import holidays

np_holidays = holidays.Nepal()

df['IsHoliday'] = df['OrderDate'].apply(
    lambda x: x in np_holidays
)
```

---

# Lag Features (Time Series)

For forecasting:

```python
df['Sales_Yesterday'] = df['Sales'].shift(1)

df['Sales_LastWeek'] = df['Sales'].shift(7)
```

---

# Rolling Statistics

```python
df['RollingMean'] = df['Sales'].rolling(7).mean()

df['RollingStd'] = df['Sales'].rolling(7).std()
```

Useful in forecasting and anomaly detection.

---

# Date Encoding for Machine Learning

A common preprocessing workflow is:

```python
df['Date'] = pd.to_datetime(df['Date'])

df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
df['Weekday'] = df['Date'].dt.dayofweek
df['IsWeekend'] = (df['Weekday'] >= 5).astype(int)

df = df.drop(columns=['Date'])
```

Now all remaining features are numeric and can be fed into scikit-learn models.

---

# Using a Custom Transformer in Scikit-learn

You can automate datetime feature extraction with a custom transformer.

```python
from sklearn.base import BaseEstimator, TransformerMixin

class DateTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X['Date'] = pd.to_datetime(X['Date'])
        X['Year'] = X['Date'].dt.year
        X['Month'] = X['Date'].dt.month
        X['Day'] = X['Date'].dt.day
        X['Weekday'] = X['Date'].dt.dayofweek
        X['IsWeekend'] = (X['Weekday'] >= 5).astype(int)
        return X.drop(columns=['Date'])
```

You can then include this transformer in a `Pipeline` before scaling or encoding other variables.

---

# Complete Workflow

```text
Raw Dataset
      │
      ▼
Convert to datetime
      │
      ▼
Extract features
(Year, Month, Day, Hour, Weekday, Quarter, etc.)
      │
      ▼
Create time differences
(DaysSince, Lag, Rolling Mean)
      │
      ▼
Encode cyclic features
(Sine/Cosine)
      │
      ▼
Drop original datetime column
      │
      ▼
ColumnTransformer
(Numeric + Categorical preprocessing)
      │
      ▼
Machine Learning Model
```

---

# Best Practices

* Convert date columns to `datetime64` using `pd.to_datetime()` as early as possible.
* Extract only features that are relevant to your problem; avoid creating unnecessary columns.
* Encode cyclic variables (hour, weekday, month) with sine/cosine transformations when their circular nature matters.
* Use time differences (e.g., days since signup, time until delivery) instead of raw dates whenever possible.
* Drop the original datetime column after feature extraction unless your model or pipeline can process it directly.
* For time-series problems, avoid random train-test splits; preserve chronological order and consider lag and rolling-window features.

By engineering meaningful date and time features, you can often improve model performance significantly without changing the learning algorithm itself.
