# Mathematics of Gradient Boosting

Gradient Boosting is easiest to understand mathematically as **iteratively fitting new weak learners to the errors made by the current model**.

The key idea is:

> Instead of training every tree independently, each new tree is trained to reduce the loss of the combined model built so far.

---

## 1. Start with the prediction problem

Suppose we have a training dataset:

$$
D=\{(x_1,y_1),(x_2,y_2),\ldots,(x_n,y_n)\}
$$

where:

* \(x_i\) = features of observation \(i\)
* \(y_i\) = actual target
* \(F(x)\) = our overall prediction function

We want to minimize a loss function:

$$
\boxed{
\mathcal{L}(F)=\sum_{i=1}^{n}L(y_i,F(x_i))
}
$$

For example, in regression with Mean Squared Error:

$$
L(y_i,F(x_i))
=
\frac{1}{2}(y_i-F(x_i))^2
$$

So the objective becomes:

$$
\mathcal{L}(F)
=
\sum_{i=1}^{n}
\frac{1}{2}(y_i-F(x_i))^2
$$

---

# 2. The Gradient Boosting model

Instead of trying to learn one complicated function directly, Gradient Boosting builds it gradually:

$$
\boxed{
F_M(x)=F_0(x)+\sum_{m=1}^{M}\eta h_m(x)
}
$$

where:

* \(F_0(x)\) = initial prediction
* \(h_m(x)\) = weak learner, usually a decision tree
* \(M\) = number of boosting iterations
* \(\eta\) = learning rate

For example:

$$
F_3(x)
=
F_0(x)+
\eta h_1(x)+
\eta h_2(x)+
\eta h_3(x)
$$

Each tree contributes only a small correction.

---

# 3. Why is it called "Gradient" Boosting?

This is the most important mathematical part.

We want to minimize:

$$
\mathcal{L}(F)
=
\sum_{i=1}^{n}L(y_i,F(x_i))
$$

In ordinary gradient descent, we update parameters in the direction opposite to the gradient:

$$
\theta_{\text{new}}
=
\theta_{\text{old}}
-
\eta\nabla_\theta L
$$

Gradient Boosting does something conceptually similar.

But instead of updating numerical parameters, we update the **function itself**:

$$
\boxed{
F_m(x)
=
F_{m-1}(x)+\eta h_m(x)
}
$$

The new tree \(h_m(x)\) is chosen to approximate the **negative gradient of the loss**.

---

# 4. Negative gradient

For every training example \(i\), calculate:

$$
\boxed{
r_{im}
=
-
\left[
\frac{\partial L(y_i,F(x_i))}
{\partial F(x_i)}
\right]_{F=F_{m-1}}
}
$$

These \(r_{im}\) are called **pseudo-residuals**.

The new decision tree is trained to predict these pseudo-residuals.

So:

$$
\boxed{
h_m(x_i)\approx r_{im}
}
$$

This is the mathematical heart of Gradient Boosting.

---

# 5. Why does this become ordinary residuals in regression?

Let's use squared error.

Define:

$$
L(y_i,F(x_i))
=
\frac{1}{2}(y_i-F(x_i))^2
$$

Take the derivative with respect to \(F(x_i)\):

$$
\frac{\partial L}{\partial F(x_i)}
=
-(y_i-F(x_i))
$$

Therefore:

$$
-\frac{\partial L}{\partial F(x_i)}
=
y_i-F(x_i)
$$

So:

$$
\boxed{
r_i=y_i-F(x_i)
}
$$

That's exactly the ordinary residual.

Therefore, for squared-error regression:

> **Gradient Boosting trains each new tree on the residuals of the previous model.**

---

# 6. Complete mathematical algorithm

Suppose we have:

$$
(x_1,y_1),\ldots,(x_n,y_n)
$$

### Step 1 — Initialize the model

We first find a constant prediction that minimizes the loss:

$$
\boxed{
F_0(x)=
\arg\min_\gamma
\sum_{i=1}^{n}L(y_i,\gamma)
}
$$

For squared-error regression:

$$
F_0(x)=\bar y
$$

because the mean minimizes squared error.

---

# 7. Calculate residuals / negative gradients

At iteration \(m\):

$$
\boxed{
r_{im}
=
-
\left[
\frac{\partial L(y_i,F(x_i))}
{\partial F(x_i)}
\right]_{F=F_{m-1}}
}
$$

For MSE:

$$
r_{im}=y_i-F_{m-1}(x_i)
$$

---

# 8. Train a weak learner

Train a decision tree on:

$$
X \rightarrow r_{im}
$$

So the tree learns:

$$
\boxed{
h_m(x)\approx r_m
}
$$

The tree is therefore learning:

> "What correction should I make to the current model?"

---

# 9. Find the optimal contribution of the tree

This is an important part that is often skipped in beginner explanations.

We don't necessarily add the entire tree prediction.

We find a multiplier \(\gamma_m\):

$$
\boxed{
\gamma_m
=
\arg\min_\gamma
\sum_{i=1}^{n}
L\left(
y_i,
F_{m-1}(x_i)+\gamma h_m(x_i)
\right)
}
$$

Then update:

$$
\boxed{
F_m(x)
=
F_{m-1}(x)
+
\eta\gamma_mh_m(x)
}
$$

Some simplified explanations combine \(\gamma_m\) into the tree's leaf values.

---

# 10. For squared-error regression

Let's derive \(\gamma_m\).

We have:

$$
L(y_i,F_m(x_i))
=
\frac12
\left[
y_i-
(F_{m-1}(x_i)+\gamma h_m(x_i))
\right]^2
$$

We want:

$$
\gamma_m=
\arg\min_\gamma
\sum_i
\frac12
[y_i-F_{m-1}(x_i)-\gamma h_m(x_i)]^2
$$

Let:

$$
r_i=y_i-F_{m-1}(x_i)
$$

Then:

$$
\gamma_m=
\arg\min_\gamma
\sum_i
\frac12
(r_i-\gamma h_m(x_i))^2
$$

Differentiate:

$$
\frac{d}{d\gamma}
\sum_i
\frac12(r_i-\gamma h_i)^2
=0
$$

Therefore:

$$
-\sum_i h_i(r_i-\gamma h_i)=0
$$

$$
\sum_i h_ir_i
=
\gamma\sum_i h_i^2
$$

Thus:

$$
\boxed{
\gamma_m
=
\frac{\sum_i h_m(x_i)r_{im}}
{\sum_i h_m(x_i)^2}
}
$$

Then:

$$
\boxed{
F_m(x)
=
F_{m-1}(x)
+
\eta\gamma_mh_m(x)
}
$$

---

# 11. A small numerical example

Suppose our target values are:

| \(x\) | Actual \(y\) |
| ----: | -----------: |
|     1 |           10 |
|     2 |           20 |
|     3 |           30 |
|     4 |           40 |

For squared-error regression, our initial model is the mean:

$$
F_0(x)=\frac{10+20+30+40}{4}=25
$$

Therefore:

| \(x\) | \(y\) | \(F_0(x)\) | Residual |
| ----: | ----: | ---------: | -------: |
|     1 |    10 |         25 |      -15 |
|     2 |    20 |         25 |       -5 |
|     3 |    30 |         25 |        5 |
|     4 |    40 |         25 |       15 |

The first tree tries to learn:

$$
X\rightarrow[-15,-5,5,15]
$$

Suppose the tree produces:

$$
h_1(x)
=
[-12,-4,4,12]
$$

Then with learning rate:

$$
\eta=0.1
$$

the updated predictions are:

$$
F_1(x)
=
F_0(x)+0.1h_1(x)
$$

Therefore:

| \(x\) | \(F_0\) | Tree correction | \(F_1\) |
| ----: | ------: | --------------: | ------: |
|     1 |      25 |            -1.2 |    23.8 |
|     2 |      25 |            -0.4 |    24.6 |
|     3 |      25 |            +0.4 |    25.4 |
|     4 |      25 |            +1.2 |    26.2 |

Now calculate the new residuals:

$$
y-F_1
$$

giving:

$$
[-13.8,-4.6,4.6,13.8]
$$

The second tree tries to learn these **new residuals**.

Then:

$$
F_2(x)=F_1(x)+\eta h_2(x)
$$

And the process continues.

---

# 12. The intuition behind the trees

Imagine the current model predicts:

$$
\hat y=60
$$

while:

$$
y=75
$$

The model is:

$$
15
$$

too low.

So the next tree should learn approximately:

$$
+15
$$

If another observation has:

$$
y=40,\qquad\hat y=50
$$

then its residual is:

$$
-10
$$

The next tree should learn:

$$
-10
$$

So each tree is effectively learning:

$$
\boxed{\text{Correction to the current prediction}}
$$

---

# 13. Learning rate

The learning rate controls how much each tree contributes.

The update is:

$$
F_m(x)
=
F_{m-1}(x)
+
\eta\gamma_mh_m(x)
$$

Suppose:

$$
h_m(x)=20
$$

and:

$$
\eta=0.1
$$

Then the model only moves:

$$
0.1(20)=2
$$

rather than adding the full 20.

### Small learning rate

$$
\eta=0.01
$$

Small updates → generally requires more trees.

### Large learning rate

$$
\eta=0.5
$$

Large updates → generally requires fewer trees.

This gives the important relationship:

$$
\boxed{
\text{Lower learning rate}
\Longleftrightarrow
\text{usually more trees}
}
$$

---

# 14. Why does Gradient Boosting use trees?

Suppose the negative gradient is:

$$
r_i=
-\frac{\partial L_i}{\partial F(x_i)}
$$

We need a function that can approximate:

$$
x_i\rightarrow r_i
$$

Decision trees are convenient because they can approximate nonlinear relationships.

For example:

$$
h_1(x)=
\begin{cases}
-10 & x<5\\
+8 & x\ge5
\end{cases}
$$

Another tree can refine the approximation:

$$
h_2(x)=
\begin{cases}
-3 & x<2\\
+4 & x\ge2
\end{cases}
$$

Together:

$$
F(x)=F_0(x)+\eta h_1(x)+\eta h_2(x)+\cdots
$$

can create a complex nonlinear prediction function.

---

# 15. Gradient Boosting as functional gradient descent

This is the deeper mathematical interpretation.

Ordinary gradient descent:

$$
\theta_m
=
\theta_{m-1}
-
\eta
\nabla_\theta J(\theta)
$$

Gradient Boosting:

$$
\boxed{
F_m(x)
=
F_{m-1}(x)
-
\eta
\frac{\delta\mathcal L}{\delta F(x)}
}
$$

where:

$$
\frac{\delta\mathcal L}{\delta F(x)}
$$

is a **functional gradient**.

We can't simply write the exact gradient function as a normal parameter vector, so we approximate it using a weak learner:

$$
h_m(x)
\approx
-\frac{\delta\mathcal L}{\delta F(x)}
$$

Thus:

$$
\boxed{
\text{Gradient Boosting}
=
\text{Gradient Descent in Function Space}
}
$$

This is the central mathematical idea.

---

# 16. Different loss functions → different "residuals"

This is another very important concept.

The new tree isn't always trained on ordinary:

$$
y-\hat y
$$

Instead, it is trained on the **negative gradient of whatever loss function we choose**.

### Squared error

$$
L(y,F)=\frac12(y-F)^2
$$

Negative gradient:

$$
\boxed{r=y-F}
$$

So we get ordinary residuals.

---

### Absolute error

$$
L(y,F)=|y-F|
$$

The derivative involves the sign:

$$
\frac{\partial L}{\partial F}
=
-\operatorname{sign}(y-F)
$$

Therefore the negative gradient is approximately:

$$
\boxed{
r=\operatorname{sign}(y-F)
}
$$

So the new tree focuses on whether predictions are too high or too low.

---

### Logistic loss for binary classification

For binary classification, let:

$$
p_i=P(y_i=1|x_i)
$$

and:

$$
p_i=\sigma(F(x_i))
$$

where:

$$
\sigma(z)=\frac{1}{1+e^{-z}}
$$

Using binary cross-entropy:

$$
L_i
=
-y_i\log p_i
-
(1-y_i)\log(1-p_i)
$$

The gradient with respect to \(F(x_i)\) becomes:

$$
\boxed{
\frac{\partial L_i}{\partial F(x_i)}
=
p_i-y_i
}
$$

Therefore the negative gradient is:

$$
\boxed{
r_i=y_i-p_i
}
$$

So for classification, the tree is essentially learning corrections based on:

$$
\boxed{y_i-p_i}
$$

rather than simply \(y_i-\hat y_i\).

---

# 17. The entire algorithm mathematically

You can remember Gradient Boosting using this sequence:

### Initialization

$$
\boxed{
F_0(x)=
\arg\min_\gamma
\sum_iL(y_i,\gamma)
}
$$

### For \(m=1,\ldots,M\)

Calculate negative gradient:

$$
\boxed{
r_{im}
=
-
\left[
\frac{\partial L(y_i,F(x_i))}
{\partial F(x_i)}
\right]_{F=F_{m-1}}
}
$$

Fit tree:

$$
\boxed{
h_m(x_i)\approx r_{im}
}
$$

Find optimal step:

$$
\boxed{
\gamma_m=
\arg\min_\gamma
\sum_i
L(y_i,F_{m-1}(x_i)+\gamma h_m(x_i))
}
$$

Update:

$$
\boxed{
F_m(x)
=
F_{m-1}(x)+
\eta\gamma_mh_m(x)
}
$$

Finally:

$$
\boxed{
F_M(x)
=
F_0(x)+
\eta\sum_{m=1}^{M}\gamma_mh_m(x)
}
$$

---

# 18. One picture to remember the mathematics

```text
                 Training Data
                       │
                       ▼
              Initial Prediction
                    F₀(x)
                       │
                       ▼
              Calculate Gradient
                       │
                       ▼
          Negative Gradient / Residual
                       │
                       ▼
              Train Decision Tree
                    h₁(x)
                       │
                       ▼
              Find optimal γ₁
                       │
                       ▼
       F₁(x) = F₀(x) + ηγ₁h₁(x)
                       │
                       ▼
              Calculate New Gradient
                       │
                       ▼
              Train Tree 2
                    h₂(x)
                       │
                       ▼
       F₂(x) = F₁(x) + ηγ₂h₂(x)
                       │
                       ▼
                     ...
                       │
                       ▼
       Fₘ(x) = Fₘ₋₁(x) + ηγₘhₘ(x)
```

---

## 19. The most important formulas for your ML notes

If you're studying Gradient Boosting mathematically, focus on these **five formulas**:

### ① Objective

$$
\boxed{
\mathcal L(F)=\sum_iL(y_i,F(x_i))
}
$$

### ② Negative gradient

$$
\boxed{
r_{im}
=
-\frac{\partial L(y_i,F(x_i))}
{\partial F(x_i)}
}
$$

### ③ Weak learner

$$
\boxed{
h_m(x_i)\approx r_{im}
}
$$

### ④ Model update

$$
\boxed{
F_m(x)=F_{m-1}(x)+\eta\gamma_mh_m(x)
}
$$

### ⑤ Final model

$$
\boxed{
F_M(x)=F_0(x)+
\eta\sum_{m=1}^{M}\gamma_mh_m(x)
}
$$

### In one sentence:

$$
\boxed{
\text{Gradient Boosting}
=
\text{Sequentially fit trees to negative gradients and add their corrections}
}
$$

The **big conceptual distinction from Bagging** is that Bagging trains trees independently and averages them, while Gradient Boosting trains each new tree based on the **current model's loss gradient**.
