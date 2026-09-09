# Machine Learning From Scratch --- Notes

These notes are based on the algorithms I implemented from scratch using
Python and NumPy. The main goal is to understand what happens inside the
algorithm instead of directly depending on Scikit-learn implementations.

------------------------------------------------------------------------

## 1. Machine Learning Basics

### What is Machine Learning?

Machine Learning is a way of making a computer learn patterns from data
and use those patterns to make predictions.

A simple flow is:

**Data → Algorithm → Learning → Model → Prediction**

### Basic terms

-   **X** → input features
-   **y** → actual/target output
-   **Model** → mathematical function that learns the relationship
    between X and y
-   **Training** → process of learning the model parameters
-   **Prediction** → using the trained model on new input

------------------------------------------------------------------------

# 2. K-Nearest Neighbors (KNN)

## What is KNN?

K-Nearest Neighbors is a simple supervised learning algorithm.

It does not learn a complicated mathematical equation during training.
Instead, it keeps the training data and uses nearby data points when
making a prediction.

KNN can be used for:

-   Classification
-   Regression

My implementation focuses on **KNN classification**.

## Basic idea

Suppose we have two classes of points.

For a new point:

1.  Calculate the distance from the new point to every training point.
2.  Find the nearest `k` points.
3.  Look at their classes.
4.  Choose the majority class.
5.  Return that class as the prediction.

## Euclidean Distance

For two points:

`x = (x1, x2)`

`p = (p1, p2)`

the Euclidean distance is:

`d = sqrt((x1 - p1)^2 + (x2 - p2)^2)`

For more features:

`d = sqrt(sum((xi - pi)^2))`

## Step-by-step KNN

### Step 1 --- Choose K

Example:

`k = 3`

This means we will consider the 3 closest training points.

### Step 2 --- Calculate distances

For the new point, calculate its distance from every training point.

### Step 3 --- Sort the distances

Arrange the distances from smallest to largest.

### Step 4 --- Select K nearest points

Take the first `k` points.

### Step 5 --- Majority voting

If the nearest classes are:

`[0, 0, 1]`

then class `0` wins.

### Step 6 --- Predict

Return the majority class.

## Important point about K

-   Very small K → sensitive to noise
-   Very large K → can ignore local patterns
-   A suitable K gives a better balance

## Does KNN really have training?

KNN has very little training computation.

The main work happens during prediction because distances must be
calculated against the stored training data.

This is why KNN is often called a **lazy learning** or
**instance-based** method.

## Feature Scaling

KNN is distance-based, so feature scale matters.

Example:

-   Age: 20--60
-   Salary: 20,000--200,000

Salary can dominate the distance because its numerical range is much
larger.

So scaling features can be important for KNN.

## When to use KNN

KNN can be useful when:

-   The dataset is relatively small.
-   Similar data points tend to have similar labels.
-   The decision boundary is not easy to describe with a simple
    equation.
-   Interpretability through nearest examples is useful.

## Limitations

-   Prediction can be slow for large datasets.
-   It can require storing the training data.
-   Sensitive to feature scaling.
-   Sensitive to the choice of K.
-   Irrelevant features can affect distance.

------------------------------------------------------------------------

# 3. Linear Regression

## What is Linear Regression?

Linear Regression is used mainly for predicting a **continuous numerical
value**.

Examples:

-   House price
-   Salary
-   Sales
-   Temperature
-   Demand

The basic equation is:

`y_hat = theta0 + theta1*x1 + theta2*x2 + ... + thetan*xn`

Where:

-   `y_hat` → predicted value
-   `theta0` → bias/intercept
-   `theta1 ... thetan` → model weights
-   `x1 ... xn` → input features

For one feature:

`y_hat = b + wx`

------------------------------------------------------------------------

# 4. Linear Regression Using Gradient Descent

This is the version I implemented to understand how the model learns its
weights step by step.

## Main idea

Initially, the model does not know the correct weights.

We:

1.  Initialize the weights.
2.  Make predictions.
3.  Calculate the error.
4.  Calculate the gradient.
5.  Update the weights.
6.  Repeat the process.

The goal is to reduce the error.

## Cost Function --- MSE

A common cost function is Mean Squared Error:

`MSE = (1/m) * sum((y - y_hat)^2)`

Where:

-   `m` → number of training examples
-   `y` → actual value
-   `y_hat` → predicted value

The smaller the MSE, the better the fit.

## Gradient Descent intuition

Imagine standing on a hill and trying to reach the lowest point.

The gradient tells us the direction of the slope.

We move in the opposite direction of the gradient.

For a parameter `theta`:

`theta = theta - learning_rate * gradient`

## Step-by-step

### Step 1 --- Initialize parameters

Start with values such as:

`weights = 0`

`bias = 0`

### Step 2 --- Predict

Calculate:

`y_hat = Xw + b`

### Step 3 --- Calculate error

`error = y_hat - y`

### Step 4 --- Calculate gradients

For weights:

`dw = (1/m) * X.T * (y_hat - y)`

For bias:

`db = (1/m) * sum(y_hat - y)`

### Step 5 --- Update parameters

`w = w - learning_rate * dw`

`b = b - learning_rate * db`

### Step 6 --- Repeat

Repeat these steps for a number of iterations.

Eventually, the parameters move toward values that produce smaller
error.

## Learning Rate

The learning rate controls how large each update is.

### Too small

Learning becomes very slow.

### Too large

The algorithm may jump around the minimum or fail to converge.

### Suitable learning rate

The model gradually moves toward a minimum.

## Why implement Gradient Descent from scratch?

Using a library gives us the final model, but implementing it ourselves
helps us understand:

-   Where the weights come from
-   Why the weights change
-   What the gradient represents
-   How the learning rate affects learning
-   How predictions improve over iterations

------------------------------------------------------------------------

# 5. Linear Regression Using OLS / Normal Equation

Gradient Descent is not the only way to find Linear Regression
parameters.

Another approach is the **Ordinary Least Squares (OLS)** solution, also
called the **Normal Equation**.

## Formula

`theta = (X.T X)^(-1) X.T y`

Here, `theta` contains the model parameters.

## Why add a column of ones?

For the intercept/bias term, we add a column of `1`s to X.

For example:

``` text
X =
[1  x1]
[1  x2]
[1  x3]
```

The first column represents the bias term.

Then:

`y_hat = X theta`

## Step-by-step

1.  Add a column of ones to the feature matrix.
2.  Calculate `X.T`.
3.  Calculate `X.T X`.
4.  Find the inverse of `X.T X`.
5.  Multiply by `X.T`.
6.  Multiply by `y`.
7.  Obtain `theta`.
8.  Use `theta` to make predictions.

## Gradient Descent vs OLS

### Gradient Descent

-   Iterative method
-   Starts with initial weights
-   Updates weights repeatedly
-   Needs a learning rate
-   Can work well for large datasets
-   Can be useful when a direct closed-form solution is expensive

### OLS / Normal Equation

-   Direct mathematical solution
-   Does not require repeated iterations
-   Does not require a learning rate
-   Can be convenient for smaller datasets
-   Requires matrix operations and inversion/pseudo-inverse

## When to use which?

A simple way to remember:

**Gradient Descent → learn by repeated updates**

**OLS → solve the parameters directly using linear algebra**

------------------------------------------------------------------------

# 6. Logistic Regression

## What is Logistic Regression?

Despite its name, Logistic Regression is mainly used for
**classification**.

For binary classification, the output is commonly:

-   `0`
-   `1`

Examples:

-   Spam / Not Spam
-   Pass / Fail
-   Disease / No Disease
-   Churn / No Churn

The important idea is that Logistic Regression produces a
**probability**.

------------------------------------------------------------------------

# 7. Why not use Linear Regression for Classification?

Linear Regression can produce any numerical value:

`-5, 0.4, 2, 10, ...`

But for binary classification, we want a probability between:

`0 and 1`

So Logistic Regression applies the **sigmoid function**.

------------------------------------------------------------------------

# 8. Sigmoid Function

The sigmoid function is:

`sigmoid(z) = 1 / (1 + e^(-z))`

where:

`z = theta0 + theta1*x1 + theta2*x2 + ...`

The output is always between 0 and 1.

Examples:

-   `sigmoid(z) ≈ 0.95` → high probability of class 1
-   `sigmoid(z) ≈ 0.10` → low probability of class 1

## Sigmoid intuition

The sigmoid converts the raw linear score into a probability-like value.

``` text
Linear score
     ↓
    z
     ↓
 Sigmoid
     ↓
Probability
     ↓
Class
```

------------------------------------------------------------------------

# 9. Logistic Regression Training

The overall process is:

### Step 1 --- Initialize weights

Start with initial weights and bias.

### Step 2 --- Calculate linear score

`z = Xw + b`

### Step 3 --- Apply sigmoid

`y_hat = sigmoid(z)`

Now `y_hat` represents the predicted probability.

### Step 4 --- Calculate the gradient

The model calculates how each parameter contributed to the error.

### Step 5 --- Update weights

`w = w - learning_rate * dw`

`b = b - learning_rate * db`

### Step 6 --- Repeat

Repeat for a number of iterations.

------------------------------------------------------------------------

# 10. Logistic Regression Prediction

After training, we calculate:

`probability = sigmoid(Xw + b)`

Then use a threshold.

A common threshold is:

`0.5`

So:

``` text
if probability >= 0.5:
    class = 1
else:
    class = 0
```

For example:

-   Probability = `0.82` → class `1`
-   Probability = `0.21` → class `0`

The important distinction is:

**Probability is not the same thing as the final class prediction.**

------------------------------------------------------------------------

# 11. Logistic Regression Loss

Logistic Regression commonly uses **Log Loss / Binary Cross-Entropy**
rather than MSE.

For one example:

`Loss = -[y log(y_hat) + (1-y) log(1-y_hat)]`

The loss strongly penalizes confident wrong predictions.

Example:

If the true class is `1`:

-   Prediction `0.99` → small loss
-   Prediction `0.60` → larger loss
-   Prediction `0.01` → very large loss

------------------------------------------------------------------------

# 12. Linear Regression vs Logistic Regression

  Point          Linear Regression   Logistic Regression
  -------------- ------------------- ---------------------------
  Main task      Regression          Classification
  Output         Continuous value    Probability / class
  Function       Linear equation     Linear equation + sigmoid
  Example        House price         Spam detection
  Common loss    MSE                 Log Loss
  Final output   Numerical value     Probability, then class

### Easy way to remember

**Linear Regression → "How much?"**

**Logistic Regression → "Which class?"**

------------------------------------------------------------------------

# 13. KNN vs Linear Regression vs Logistic Regression

  -----------------------------------------------------------------------
  Algorithm               Main use                How it works
  ----------------------- ----------------------- -----------------------
  KNN                     Classification /        Looks at nearby data
                          Regression              points

  Linear Regression       Regression              Learns a linear
                                                  relationship

  Logistic Regression     Classification          Learns a linear
                                                  boundary and converts
                                                  score to probability
  -----------------------------------------------------------------------

### KNN

**"Which training examples are closest to this new point?"**

### Linear Regression

**"What numerical value should I predict?"**

### Logistic Regression

**"What is the probability of this class?"**

------------------------------------------------------------------------

# 14. KNN vs Logistic Regression

Both can be used for classification, but they work differently.

### KNN

-   Distance-based
-   Does not learn explicit weights in the usual sense
-   Uses nearby training examples
-   Can model more irregular/local decision boundaries
-   Prediction can be expensive

### Logistic Regression

-   Parameter-based
-   Learns weights during training
-   Uses a linear decision boundary in feature space
-   Faster prediction after training
-   Works well when the classes can be separated reasonably by a linear
    boundary

------------------------------------------------------------------------

# 15. How to Decide Which Algorithm to Use

Start with the target.

## If the target is continuous

Examples:

-   Price
-   Salary
-   Sales

Start by considering:

**Linear Regression**

## If the target is a class

Examples:

-   Yes / No
-   Spam / Not Spam
-   0 / 1

Consider:

**Logistic Regression**

or

**KNN**

Then ask:

### Do similar points tend to have similar labels?

If yes, KNN may be useful.

### Is there a reasonably linear relationship/boundary?

If yes, Linear Regression or Logistic Regression may be appropriate
depending on the target.

------------------------------------------------------------------------

# 16. Training vs Prediction

This difference is important.

## Training

The model learns from known data.

For Linear Regression:

`X + y → learn weights`

For Logistic Regression:

`X + y → learn weights`

For KNN:

`store training examples`

## Prediction

The trained model receives new X.

For Linear Regression:

`X → numerical prediction`

For Logistic Regression:

`X → probability → class`

For KNN:

`X → distances → nearest points → majority class`

------------------------------------------------------------------------

# 17. Parameters vs Hyperparameters

## Parameters

These are values learned by the model.

Examples:

-   Linear Regression weights
-   Linear Regression bias
-   Logistic Regression weights
-   Logistic Regression bias

## Hyperparameters

These are values we choose before/during training.

Examples:

-   Learning rate
-   Number of iterations
-   K in KNN

### Simple rule

**Parameters → model learns them**

**Hyperparameters → we choose them**

------------------------------------------------------------------------

# 18. What I Learned by Implementing These Algorithms

## KNN

I understood:

-   How distance is calculated
-   Why nearest points matter
-   How K affects prediction
-   How majority voting works
-   Why feature scaling matters

## Linear Regression --- Gradient Descent

I understood:

-   How predictions are calculated
-   How error is measured
-   What a gradient means
-   How weights are updated
-   How learning rate affects the updates
-   How repeated iterations reduce error

## Linear Regression --- OLS

I understood:

-   How the Normal Equation is formed
-   Why a bias column is added
-   How matrix operations are used to calculate parameters
-   The difference between a direct solution and iterative optimization

## Logistic Regression

I understood:

-   Why classification needs a different output
-   How the sigmoid converts a score into a probability
-   How weights are updated using gradients
-   How probability is converted into a class
-   Why Log Loss is used

------------------------------------------------------------------------

# 19. Common Interview Questions

### KNN

**Q: Why is KNN called a lazy learner?**

Because it does not perform much parameter learning during training and
performs most of its computation when making predictions.

**Q: Why does K matter?**

Because K controls how many neighboring points influence the prediction.

**Q: Why is scaling important in KNN?**

Because KNN uses distance, and features with larger numerical ranges can
dominate the distance.

------------------------------------------------------------------------

### Linear Regression

**Q: What is the goal of Linear Regression?**

To model the relationship between input features and a continuous
target.

**Q: What is MSE?**

Mean Squared Error measures the average squared difference between
actual and predicted values.

**Q: What is Gradient Descent?**

An iterative optimization method that updates parameters in the
direction that reduces the cost.

**Q: What happens if the learning rate is too high?**

The updates can become too large and the algorithm may fail to converge.

------------------------------------------------------------------------

### OLS

**Q: What is the Normal Equation?**

It is a closed-form mathematical solution for the parameters of Linear
Regression:

`theta = (X.T X)^(-1) X.T y`

**Q: Does OLS need a learning rate?**

No.

------------------------------------------------------------------------

### Logistic Regression

**Q: Why is sigmoid used?**

To convert the linear score into a value between 0 and 1.

**Q: Is Logistic Regression actually regression?**

It is a classification algorithm even though its name contains
"Regression".

**Q: What is the purpose of the threshold?**

To convert the predicted probability into a class label.

------------------------------------------------------------------------

# 20. Final Mental Model

The easiest way to remember all four algorithms:

``` text
KNN
↓
Look at nearby examples
↓
Vote
↓
Prediction


Linear Regression
↓
Calculate linear value
↓
Compare with actual value
↓
Reduce error
↓
Predict a number


Linear Regression + Gradient Descent
↓
Predict
↓
Calculate error
↓
Calculate gradient
↓
Update weights
↓
Repeat


Linear Regression + OLS
↓
Build matrix equation
↓
Solve parameters directly
↓
Predict


Logistic Regression
↓
Calculate linear score
↓
Apply sigmoid
↓
Get probability
↓
Apply threshold
↓
Predict class
```

## The main idea behind learning these from scratch

The purpose of these implementations is not simply to reproduce
Scikit-learn.

The purpose is to understand **what is happening behind the function
calls**:

-   How inputs are processed
-   How predictions are calculated
-   How error is measured
-   How parameters are obtained
-   How parameters are updated
-   Why different algorithms are used for different problems

That understanding makes it easier to use ML libraries correctly later,
because the library is no longer a black box.
