
# Probability Fundamentals: Independent Events, Mutually Exclusive Events & Bayes' Theorem

These three concepts are **very important for understanding Naive Bayes**, especially the meaning of the word **"independent."**

---

# 1. Events in Probability

An **event** is an outcome or collection of outcomes that we are interested in.

For example, when tossing a coin:

$$
S=\{H,T\}
$$

Possible events:

- $A=$ getting Heads
- $B=$ getting Tails

For a dice:

$$
S=\{1,2,3,4,5,6\}
$$

Examples:

$$
A=\text{getting an even number}=\{2,4,6\}
$$

$$
B=\text{getting a number greater than 4}=\{5,6\}
$$

---

# 2. Independent Events

Two events $A$ and $B$ are **independent** if the occurrence of one event does **not affect the probability of the other**.

In simple words:

> Knowing that A happened gives us no additional information about whether B will happen.

The mathematical definition is:

$$
\boxed{P(A|B)=P(A)}
$$

and equivalently:

$$
\boxed{P(B|A)=P(B)}
$$

Another extremely important formula is:



$$
\boxed{P(A\cap B)=P(A)P(B)}
$$

Here:

$$
A\cap B
$$

means:

> A **AND** B occur.

---

## Example: Tossing Two Coins

Suppose we toss two coins.

Let:

$$
A=\text{first coin is Heads}
$$

$$
B=\text{second coin is Heads}
$$

We know:

$$
P(A)=\frac12
$$

and:

$$
P(B)=\frac12
$$

The first coin does not affect the second coin.

Therefore:

$$
P(A\cap B)
=
P(A)P(B)
$$

$$
=\frac12\times\frac12
$$

$$
=\frac14
$$

So:

$$
\boxed{P(\text{both heads})=25\%}
$$

---

# 3. Another Example: Dice

Suppose we roll a die twice.

Let:

$$
A=\text{first roll is 6}
$$

$$
B=\text{second roll is 6}
$$

Then:

$$
P(A)=\frac16
$$

$$
P(B)=\frac16
$$

The first roll doesn't affect the second roll.

Therefore:

$$
P(A\cap B)
=
\frac16\times\frac16
$$

$$
=\frac1{36}
$$

So:

$$
\boxed{P(\text{6 on both rolls})=\frac1{36}}
$$

---

# 4. Conditional Probability and Independence

This is where things become particularly important.

Recall:

$$
P(A|B)=
\frac{P(A\cap B)}
{P(B)}
$$

If A and B are independent:

$$
P(A\cap B)=P(A)P(B)
$$

Therefore:

$$
P(A|B)
=
\frac{P(A)P(B)}
{P(B)}
$$

Cancel $P(B)$:

$$
\boxed{P(A|B)=P(A)}
$$

That's exactly what independence means.

---

# 5. Important: Independent ≠ Mutually Exclusive

This is one of the most common probability mistakes.

**Independent events** and **mutually exclusive events** are very different.

Let's understand mutually exclusive first.

---

# 6. Mutually Exclusive Events

Two events are **mutually exclusive** if they **cannot happen at the same time**.

In mathematical notation:

$$
\boxed{A\cap B=\emptyset}
$$

Therefore:

$$
\boxed{P(A\cap B)=0}
$$

In simple words:

> If A happens, B cannot happen.

---

## Example: One Dice Roll

Suppose:

$$
A=\text{roll a 2}
$$

$$
B=\text{roll a 5}
$$

Can a single die show both 2 and 5?

No.

Therefore:

$$
P(A\cap B)=0
$$

They are mutually exclusive.

---

# 7. Another Example

A person cannot simultaneously be:

- 20 years old
- 30 years old

assuming we're talking about exact age at the same instant.

Thus these events are mutually exclusive.

---

# 8. Addition Rule for Mutually Exclusive Events

For any two events:

$$
P(A\cup B)
=
P(A)+P(B)-P(A\cap B)
$$

If A and B are mutually exclusive:

$$
P(A\cap B)=0
$$

Therefore:

$$
\boxed{
P(A\cup B)=P(A)+P(B)
}
$$

where:

$$
A\cup B
$$

means:

> A **OR** B.

---

## Example

Roll a die.

Let:

$$
A=\text{getting 1}
$$

$$
B=\text{getting 6}
$$

Then:

$$
P(A)=\frac16
$$

$$
P(B)=\frac16
$$

Since they are mutually exclusive:

$$
P(A\cup B)
=
\frac16+\frac16
$$

$$
=\frac26
$$

$$
=\frac13
$$

Therefore:

$$
\boxed{P(1\text{ or }6)=\frac13}
$$

---

# 9. Independent vs Mutually Exclusive

This table is worth remembering:

| Independent Events | Mutually Exclusive Events |
|---|---|
| One does not affect the other | They cannot happen together |
| $P(A|B)=P(A)$ | $P(A\cap B)=0$ |
| $P(A\cap B)=P(A)P(B)$ | $P(A\cup B)=P(A)+P(B)$ |
| Can happen together | Cannot happen together |
| Example: two coin tosses | Example: one die roll being 2 and 5 |

---

# 10. Can Events Be Both Independent and Mutually Exclusive?

Generally, **no**, unless one of the events has probability zero.

Suppose A and B are mutually exclusive:

$$
P(A\cap B)=0
$$

If they were also independent:

$$
P(A\cap B)=P(A)P(B)
$$

Therefore:

$$
P(A)P(B)=0
$$

So at least one must have probability zero.

For ordinary non-zero probability events:

$$
\boxed{\text{Mutually exclusive events are NOT independent.}}
$$

---

# 11. Visual Intuition

Think of two events as regions.

### Independent

They **can overlap**:

```text
       A           B
    _______     _______
   /       \___/       \
  /         X           \
  \                     /
   \_______/ \_________/
```

The intersection:

$$
A\cap B
$$

can have non-zero probability.

---

### Mutually Exclusive

They don't overlap:

```text
      A              B
   _______        _______
  /       \      /       \
 /         \    /         \
 \         /    \         /
  \_______/      \_______/
```

Therefore:

$$
A\cap B=\emptyset
$$

---

# 12. Why Independence Matters in Naive Bayes

This connects directly to your previous question.

Suppose we have features:

$$
X_1,X_2,X_3
$$

and class:

$$
C
$$

Naive Bayes assumes:

$$
X_1,X_2,X_3
$$

are **conditionally independent given C**.

That means:

$$
\boxed{
P(X_1,X_2,X_3|C)
=
P(X_1|C)
P(X_2|C)
P(X_3|C)
}
$$

Notice something very important:

Naive Bayes does **not necessarily assume that the features are completely independent**.

It assumes they are independent **conditional on the class**.

This distinction is extremely important.

---

# 13. Example of Conditional Independence

Suppose we're classifying emails:

$$
C=\text{Spam}
$$

Features:

- $X_1=$ contains "free"
- $X_2=$ contains "offer"

Naive Bayes assumes:

$$
P(Free,Offer|Spam)
$$

can be approximated by:

$$
P(Free|Spam)P(Offer|Spam)
$$

This is the **naive assumption**.

---

# 14. Bayes' Theorem

Now we reach the third major concept.

Bayes' theorem tells us how to **update our belief about an event after observing evidence**.

The formula is:



$$
\boxed{
P(A|B)
=
\frac{P(B|A)P(A)}
{P(B)}
}
$$

Let's understand each term.

### $P(A|B)$

**Posterior probability**

> Probability of A after observing B.

---

### $P(B|A)$

**Likelihood**

> Probability of observing B if A is true.

---

### $P(A)$

**Prior probability**

> Probability of A before observing B.

---

### $P(B)$

**Evidence**

> Overall probability of observing B.

So:

$$
\boxed{
Posterior
=
\frac{Likelihood\times Prior}
{Evidence}
}
$$

---

# 15. Simple Bayes Example

Suppose there is a disease.

Only 1% of people have it:

$$
P(D)=0.01
$$

A medical test is positive 90% of the time when the disease is present:

$$
P(+|D)=0.90
$$

But 5% of healthy people also get a positive result:

$$
P(+|\neg D)=0.05
$$

We want:

$$
P(D|+)
$$

That means:

> Probability of disease given that the test is positive.

---

## Step 1: Bayes theorem

$$
P(D|+)
=
\frac{P(+|D)P(D)}
{P(+)}
$$

---

## Step 2: Calculate evidence

A positive test can happen in two ways:

1. Person has disease and test is positive.
2. Person doesn't have disease and test is falsely positive.

Therefore:

$$
P(+)
=
P(+|D)P(D)
+
P(+|\neg D)P(\neg D)
$$

Since:

$$
P(\neg D)=1-P(D)=0.99
$$

we get:

$$
P(+)
=
(0.90)(0.01)+(0.05)(0.99)
$$

$$
=0.009+0.0495
$$

$$
=0.0585
$$

---

## Step 3: Calculate posterior

$$
P(D|+)
=
\frac{(0.90)(0.01)}
{0.0585}
$$

$$
\approx0.1538
$$

Therefore:

$$
\boxed{P(D|+)\approx15.38\%}
$$

This demonstrates why **prior probability matters**.

---

# 16. Bayes Theorem and Naive Bayes

Now connect everything together.

Bayes theorem:

$$
P(C|X)
=
\frac{P(X|C)P(C)}
{P(X)}
$$

Suppose:

$$
X=(X_1,X_2,\ldots,X_n)
$$

Naive Bayes makes the conditional independence assumption:

$$
P(X|C)
=
\prod_{i=1}^{n}P(X_i|C)
$$

Therefore:

$$
P(C|X)
=
\frac{
P(C)\prod_{i=1}^{n}P(X_i|C)
}
{P(X)}
$$

Since $P(X)$ is the same for all classes when comparing them:

$$
\boxed{
\hat C
=
\arg\max_C
P(C)
\prod_{i=1}^{n}P(X_i|C)
}
$$

That is the fundamental Naive Bayes classifier.

---

# 17. The Three Concepts Together

You can understand the whole topic through this chain:

### Independent events

$$
P(A\cap B)=P(A)P(B)
$$

means:

> A doesn't affect B.

### Mutually exclusive events

$$
P(A\cap B)=0
$$

means:

> A and B cannot happen together.

### Bayes theorem

$$
P(A|B)=
\frac{P(B|A)P(A)}
{P(B)}
$$

means:

> Update the probability of A after observing B.

### Naive Bayes

$$
P(C|X)
\propto
P(C)\prod_iP(X_i|C)
$$

means:

> Use Bayes theorem and assume the features are conditionally independent given the class.

---

# 18. Very Important Exam/Interview Differences

### Independent

> **Can happen together, but don't influence each other.**

Example:

Two separate coin tosses.

$$
P(A\cap B)=P(A)P(B)
$$

---

### Mutually exclusive

> **Cannot happen together.**

Example:

Getting 2 and 5 on the **same single die roll**.

$$
P(A\cap B)=0
$$

---

### Conditional probability

> Probability of A when B is already known.

$$
P(A|B)
$$

---

### Bayes theorem

> Reverses conditional probability and updates belief using evidence.

$$
P(A|B)
\leftarrow
P(B|A)
$$

---

### Naive Bayes

> Applies Bayes theorem to classification while assuming conditional independence among features.

$$
\boxed{
P(Class|Features)
\propto
P(Class)
\times
\prod P(Feature|Class)
}
$$

**The key conceptual distinction to memorize is:**

$$
\boxed{
\text{Independent} \neq \text{Mutually Exclusive}
}
$$

and for Naive Bayes:

$$
\boxed{
\text{Conditional Independence} \neq \text{Mutual Exclusivity}
}
$$