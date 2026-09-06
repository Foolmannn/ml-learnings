
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

# 9. Classification Rule

Suppose we have classes:

$$
C_1,C_2,\ldots,C_k
$$

For a new sample $X$, calculate:

$$
P(C_1|X)
$$

$$
P(C_2|X)
$$

$$
\vdots
$$

$$
P(C_k|X)
$$

Then choose the class with the highest probability.

Mathematically:

$$
\boxed{
\hat{C}
=
\arg\max_C
P(C)
\prod_{i=1}^{n}P(x_i|C)
}
$$

This is the **Naive Bayes decision rule**.

---

# 10. Complete Numerical Example

Let's classify an email as:

- Spam
- Not Spam

Suppose our feature is:

> Email contains the word "free".

Training data gives us:

$$
P(Spam)=0.4
$$

$$
P(NotSpam)=0.6
$$

And:

$$
P(Free|Spam)=0.8
$$

$$
P(Free|NotSpam)=0.1
$$

A new email contains `"free"`.

We calculate both scores.

### Spam

$$
P(Spam)P(Free|Spam)
$$

$$
=0.4\times0.8
$$

$$
=0.32
$$

### Not Spam

$$
P(NotSpam)P(Free|NotSpam)
$$

$$
=0.6\times0.1
$$

$$
=0.06
$$

Since:

$$
0.32>0.06
$$

we classify it as:

$$
\boxed{Spam}
$$

---

# 11. Getting the Actual Posterior Probability

The values above are technically **unnormalized probabilities**.

To get actual posterior probabilities, divide by their sum.

Total:

$$
0.32+0.06=0.38
$$

Therefore:

$$
P(Spam|Free)
=
\frac{0.32}{0.38}
$$

$$
=0.8421
$$

And:

$$
P(NotSpam|Free)
=
\frac{0.06}{0.38}
$$

$$
=0.1579
$$

Therefore:

$$
\boxed{
P(Spam|Free)=84.21\%
}
$$

$$
\boxed{
P(NotSpam|Free)=15.79\%
}
$$

---

# 12. Multiple Features

Now suppose the email contains three features:

- `free`
- `offer`
- `money`

Suppose:

| Probability | Spam | Not Spam |
|---|---:|---:|
| Prior | 0.4 | 0.6 |
| P(free \| class) | 0.8 | 0.1 |
| P(offer \| class) | 0.7 | 0.2 |
| P(money \| class) | 0.6 | 0.1 |

For Spam:

$$
Score(Spam)
=
P(Spam)
P(free|Spam)
P(offer|Spam)
P(money|Spam)
$$

$$
=0.4\times0.8\times0.7\times0.6
$$

$$
=0.1344
$$

For Not Spam:

$$
Score(NotSpam)
=
0.6\times0.1\times0.2\times0.1
$$

$$
=0.0012
$$

Clearly:

$$
0.1344>0.0012
$$

Therefore:

$$
\boxed{Spam}
$$

---

# 13. Why Multiplication?

This is one of the most important things to understand.

Suppose:

$$
P(free|Spam)=0.8
$$

$$
P(offer|Spam)=0.7
$$

$$
P(money|Spam)=0.6
$$

Under the naive independence assumption:

$$
P(free,offer,money|Spam)
$$

becomes:

$$
0.8\times0.7\times0.6
$$

So:

$$
P(free,offer,money|Spam)=0.336
$$

We multiply because we are calculating the probability of observing **all the features together**, under the conditional-independence assumption.

---

# 14. The Denominator $P(X)$

The complete equation is:

$$
P(C|X)
=
\frac{P(X|C)P(C)}
{P(X)}
$$

For multiple classes:

$$
P(X)
=
\sum_{j=1}^{K}
P(X|C_j)P(C_j)
$$

Therefore:

$$
\boxed{
P(C_k|X)=
\frac{
P(C_k)\prod_iP(x_i|C_k)
}{
\sum_j
P(C_j)\prod_iP(x_i|C_j)
}
}
$$

This gives the normalized posterior probability.

However, for simply **choosing the class**, we don't need to calculate the denominator because it is identical for every class.

---

# 15. Prior Probability in Naive Bayes

The prior is:

$$
P(C)
$$

It represents how frequently a class occurs in the training dataset.

For example, suppose:

- 700 emails are Spam
- 300 emails are Not Spam

Total:

$$
1000
$$

Then:

$$
P(Spam)=\frac{700}{1000}=0.7
$$

and:

$$
P(NotSpam)=\frac{300}{1000}=0.3
$$

So the model already knows:

> Spam is more common in this dataset.

This prior information influences classification.

---

# 16. Likelihood

Likelihood is:

$$
P(x_i|C)
$$

It answers:

> How likely is this feature if the sample belongs to this class?

For example:

$$
P(free|Spam)=0.8
$$

means:

> Among Spam emails, 80% contain the word "free."

Whereas:

$$
P(free|NotSpam)=0.1
$$

means:

> Among Not Spam emails, only 10% contain "free."

This makes `"free"` a useful feature for distinguishing Spam.

---

# 17. Posterior

Posterior is:

$$
P(C|X)
$$

It answers:

> After seeing the features, how likely is the sample to belong to this class?

The entire Bayesian process can be remembered as:

$$
\boxed{
Prior + Evidence
\rightarrow
Posterior
}
$$

More mathematically:

$$
\boxed{
Posterior
\propto
Likelihood\times Prior
}
$$

---

# 18. Zero Probability Problem

There is an important problem with Naive Bayes.

Suppose:

$$
P(x_1|C)=0
$$

Then:

$$
P(C)\prod_iP(x_i|C)=0
$$

because:

$$
0\times anything=0
$$

So the entire class probability becomes zero.

### Example

Suppose a word never appeared in Spam training examples.

Then:

$$
P(word|Spam)=0
$$

For an email containing that word:

$$
P(Spam|X)=0
$$

even though the word might simply have been absent from the limited training dataset.

This is called the:

$$
\boxed{\text{Zero-frequency problem}}
$$

---

# 19. Laplace Smoothing

Naive Bayes commonly solves this using **Laplace smoothing**.

For categorical/count-based probabilities:

$$
\boxed{
P(x_i|C)
=
\frac{count(x_i,C)+\alpha}
{count(C)+\alpha K}
}
$$

where:

- $\alpha$ = smoothing parameter
- $K$ = number of possible feature values/classes depending on the formulation

For standard Multinomial Naive Bayes, for a word $w$:

$$
\boxed{
P(w|C)
=
\frac{N_{w,C}+\alpha}
{N_C+\alpha V}
}
$$

where:

- $N_{w,C}$ = count of word $w$ in class $C$
- $N_C$ = total word count in class $C$
- $V$ = vocabulary size
- $\alpha$ = smoothing parameter

With:

$$
\alpha=1
$$

this is called **Laplace smoothing**.

---
