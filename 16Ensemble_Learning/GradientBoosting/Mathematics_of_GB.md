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
