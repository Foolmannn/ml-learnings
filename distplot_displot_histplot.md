The reason is that **`distplot()` and `displot()` are fundamentally different functions**.

* **`distplot()` (deprecated)** is an **Axes-level** function. It plots directly on the current `plt.subplot()`.
* **`displot()` (modern)** is a **Figure-level** function. It creates its **own Figure** and **cannot be placed inside `plt.subplot()`**.

That's why your code works with `distplot()` but not with `displot()`.

### Why it doesn't work

```python
plt.subplot(121)
sns.displot(X_train[col])   # Creates a NEW figure
```

The `plt.subplot(121)` becomes useless because `displot()` ignores the current axes.

---

## Solution 1 (Recommended): Use `histplot()`

If your goal is exactly what `distplot()` used to do (histogram + KDE inside subplots), the modern replacement is **`histplot()`**, not `displot()`.

```python
for col in X_train.columns:
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    sns.histplot(
        X_train[col],
        kde=True,
        ax=axes[0]
    )
    axes[0].set_title(col)

    stats.probplot(X_train[col], dist="norm", plot=axes[1])
    axes[1].set_title(col)

    plt.tight_layout()
    plt.show()
```

This is the direct modern replacement for `distplot()`.

---

## Solution 2: If you really want `displot()`

Then you cannot combine it with subplots.

```python
for col in X_train.columns:

    sns.displot(
        X_train,
        x=col,
        kde=True,
        height=4,
        aspect=2
    )

    plt.show()

    stats.probplot(X_train[col], dist="norm", plot=plt)
    plt.title(col)
    plt.show()
```

This creates **two separate figures**.

---

## Solution 3: Use `kdeplot()`

If you only need the density curve:

```python
fig, axes = plt.subplots(1,2, figsize=(14,4))

sns.kdeplot(X_train[col], fill=True, ax=axes[0])

stats.probplot(X_train[col], dist="norm", plot=axes[1])
```

---

## Difference between `histplot()` and `displot()`

| Function                    | Axes-level | Figure-level | Can use `ax=` | Can use `subplot()` |
| --------------------------- | ---------- | ------------ | ------------- | ------------------- |
| `distplot()` *(deprecated)* | ✅          | ❌            | ✅             | ✅                   |
| `histplot()`                | ✅          | ❌            | ✅             | ✅                   |
| `kdeplot()`                 | ✅          | ❌            | ✅             | ✅                   |
| `displot()`                 | ❌          | ✅            | ❌             | ❌                   |

### Rule of thumb

* **Use `histplot()`** when you're working with `plt.subplots()`.
* **Use `displot()`** when you want Seaborn to create an entire figure by itself (especially for faceting with `row=`, `col=`, or `hue=`).

For your code, **`histplot(kde=True)` is the correct modern replacement** for the deprecated `distplot()`.
