
# Probability Basis of the Naive Bayes Classifier

Naive Bayes is a **probabilistic classification algorithm** based mainly on **Bayes' Theorem**.

The important idea is:

> Given some observed features, calculate the probability that the sample belongs to each class, and choose the class with the highest probability.

For example, suppose we want to classify an email as:

- Spam
- Not Spam

using features such as:

- contains `"free"`
- contains `"offer"`
- contains `"money"`

Naive Bayes asks:

> Given these words/features, what is the probability that this email is Spam?

---

# 1. Probability Fundamentals

Before Naive Bayes, we need a few probability concepts.

## 1.1 Probability

Probability measures how likely an event is.

$$
0 \leq P(A) \leq 1
$$

Examples:

$$
P(\text{Heads}) = 0.5
$$

$$
P(\text{Rain}) = 0.3
$$

A probability of:

- `0` → impossible
- `1` → certain
- `0.5` → 50% chance

---

# 2. Conditional Probability

Conditional probability means:

> Probability of event A happening **given that B has already happened**.

It is written as:

$$
P(A|B)
$$

and read:

> Probability of A given B.

The formula is:

$$
\boxed{
P(A|B)=\frac{P(A\cap B)}{P(B)}
}
$$

where:

- $P(A|B)$ = probability of A given B
- $P(A\cap B)$ = probability that A and B both occur
- $P(B)$ = probability of B

---

## Example

Suppose:

- 60% of students are male
- 40% are female
- 30% of all students play football
- 20% of all students are male and play football

We want:

$$
P(\text{Football}|\text{Male})
$$

Using:

$$
P(F|M)=\frac{P(F\cap M)}{P(M)}
$$

we get:

$$
P(F|M)=\frac{0.20}{0.60}
$$

$$
P(F|M)=0.333
$$

So:

$$
\boxed{P(Football|Male)=33.3\%}
$$

---

# 3. Joint Probability

Joint probability is the probability that **two events occur together**.

$$
P(A,B)
$$

or:

$$
P(A\cap B)
$$

For example:

$$
P(\text{Rain},\text{Traffic})
$$

means:

> Probability that it rains AND there is traffic.

The relationship between conditional and joint probability is:

$$
\boxed{
P(A,B)=P(A|B)P(B)
}
$$

Similarly:

$$
P(A,B)=P(B|A)P(A)
$$

Therefore:

$$
\boxed{
P(A|B)P(B)=P(B|A)P(A)
}
$$

This leads directly to Bayes' theorem.

---

# 4. Bayes' Theorem



Bayes' theorem is:

$$
\boxed{
P(A|B)=
\frac{P(B|A)P(A)}
{P(B)}
}
$$

This is the mathematical foundation of **Naive Bayes**.

Let's understand every term.

### $P(A|B)$ — Posterior

Probability of A after observing B.

Also called the **posterior probability**.

---

### $P(B|A)$ — Likelihood

Probability of observing B assuming A is true.

---

### $P(A)$ — Prior

Probability of A before seeing B.

---

### $P(B)$ — Evidence

Overall probability of observing B.

---

So we can remember:

$$
\boxed{
Posterior=
\frac{Likelihood\times Prior}{Evidence}
}
$$

---

# 5. A Simple Bayes Example

Suppose we want to determine whether a person has a disease based on a positive test.

Suppose:

$$
P(Disease)=0.01
$$

So only 1% of people have the disease.

The test has:

$$
P(Positive|Disease)=0.90
$$

and:

$$
P(Positive|NoDisease)=0.05
$$

We observe:

> The test is positive.

We want:

$$
P(Disease|Positive)
$$

Bayes' theorem:

$$
P(D|+)
=
\frac{P(+|D)P(D)}
{P(+)}
$$

We first calculate $P(+)$.

$$
P(+)=P(+|D)P(D)+P(+|\neg D)P(\neg D)
$$

Therefore:

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

Now:

$$
P(D|+)
=
\frac{(0.90)(0.01)}
{0.0585}
$$

$$
=0.1538
$$

Therefore:

$$
\boxed{P(Disease|Positive)\approx15.38\%}
$$

This is a very important example.

Even though the test is 90% accurate for people with the disease, a positive result doesn't mean there is a 90% chance of having the disease.

Why?

Because the **prior probability** of the disease is very low.

This is the basic power of Bayesian reasoning.

---

# 6. From Bayes Theorem to Naive Bayes

Now let's move toward machine learning.

Suppose we have:

$$
X=(x_1,x_2,x_3,\ldots,x_n)
$$

where:

- $X$ = input sample
- $x_1,x_2,\ldots,x_n$ = features

And:

$$
C
$$

is the class.

We want to calculate:

$$
P(C|X)
$$

Using Bayes theorem:

$$
P(C|X)
=
\frac{P(X|C)P(C)}
{P(X)}
$$

Therefore:

$$
\boxed{
P(C|X)
=
\frac{P(X|C)P(C)}
{P(X)}
}
$$

Since $P(X)$ is the same for every class, when comparing classes we can ignore it.

So:

$$
P(C|X)\propto P(X|C)P(C)
$$

---

# 7. Where Does "Naive" Come From?

Suppose we have three features:

$$
X=(x_1,x_2,x_3)
$$

Then:

$$
P(X|C)
=
P(x_1,x_2,x_3|C)
$$

Calculating this joint probability can be difficult.

Naive Bayes makes a **strong assumption**:

> Features are conditionally independent given the class.

Therefore:

$$
\boxed{
P(x_1,x_2,\ldots,x_n|C)
=
\prod_{i=1}^{n}P(x_i|C)
}
$$

So:

$$
P(C|X)
\propto
P(C)
\prod_{i=1}^{n}P(x_i|C)
$$

This is the fundamental Naive Bayes equation:

$$
\boxed{
P(C|x_1,\ldots,x_n)
\propto
P(C)
\prod_{i=1}^{n}P(x_i|C)
}
$$

---

# 8. Why the Independence Assumption?

Consider an email:

> "Congratulations! You won a free lottery."

Features could be:

$$
x_1=\text{"free"}
$$

$$
x_2=\text{"lottery"}
$$

$$
x_3=\text{"won"}
$$

We want:

$$
P(Spam|free,lottery,won)
$$

Bayes:

$$
P(Spam|free,lottery,won)
\propto
P(free,lottery,won|Spam)P(Spam)
$$

The joint probability can be difficult to estimate.

Naive Bayes approximates:

$$
P(free,lottery,won|Spam)
$$

as:

$$
P(free|Spam)
P(lottery|Spam)
P(won|Spam)
$$

Hence:

$$
\boxed{
P(Spam|free,lottery,won)
\propto
P(Spam)
P(free|Spam)
P(lottery|Spam)
P(won|Spam)
}
$$

That's the **naive assumption**.

---
