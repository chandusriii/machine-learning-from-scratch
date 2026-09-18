# Decision Trees — Supervised Learning

A practical and mathematical study of Decision Trees, combining lecture notes, formulas, hands-on notebooks, and a real classification assignment.

## Learning Map

Decision Tree
- Classification
  - Entropy
  - Gini Impurity
  - Information Gain
  - Pure vs Impure nodes
  - Pre-pruning
  - Post-pruning
  - Common pruning parameters
- Regression
  - Variance
  - Variance Reduction
- Practical implementation
  - scikit-learn
  - preprocessing
  - pipelines
  - evaluation
  - hyperparameter tuning

## 1. What is a Decision Tree?

A Decision Tree predicts an outcome by asking a sequence of questions about input features.

A tree is made of:
- Root node — starting decision
- Internal nodes — feature-based questions
- Branches — outcomes of those questions
- Leaf nodes — final prediction

The central idea is to repeatedly split data into groups that are increasingly pure (classification) or have lower variance (regression).

## 2. Classification

A Decision Tree Classifier predicts a class such as:
- Yes / No
- Spam / Not Spam
- Survived / Died

### Pure and Impure Nodes

A pure node contains observations belonging to one class.

An impure node contains observations from multiple classes.

The splitting process tries to produce purer child nodes.

## 3. Entropy

Entropy measures impurity, disorder, or uncertainty in a dataset.

### Formula

$$
H(S) = -\sum_i p_i \log_2(p_i)
$$

For binary classification:

$$
H(S) = -p_{yes}\log_2(p_{yes}) - p_{no}\log_2(p_{no})
$$

Important cases:
- Entropy = 0 → completely pure
- Maximum binary entropy = 1 → 50% / 50% class distribution

## 4. Gini Impurity

Gini Impurity measures the likelihood of misclassifying a randomly selected observation if it were labelled according to the class distribution of the node.

### Formula

$$
Gini = 1 - \sum_i p_i^2
$$

For binary classification:

$$
Gini = 1 - (p_{yes}^2 + p_{no}^2)
$$

Important cases:
- Gini = 0 → completely pure
- Maximum binary Gini = 0.5 → 50% / 50% distribution

## 5. Entropy vs Gini

Both are impurity measures used to evaluate classification splits.

| Aspect | Entropy | Gini Impurity |
|---|---|---|
| Main idea | Uncertainty | Impurity |
| Formula | Uses logarithms | Uses squared probabilities |
| Binary maximum | 1 | 0.5 |
| Practical role | Split selection | Split selection |
| Typical computation | Slightly more involved | Simpler |

## 6. Information Gain

Information Gain measures the reduction in entropy produced by a split.

$$
IG(S,A) = H(S) - \sum_{v\in A}\frac{|S_v|}{|S|}H(S_v)
$$

In simple terms:

**Information Gain = Parent Entropy − Weighted Child Entropy**

A split with greater information gain provides a larger reduction in entropy.

## 7. Decision Tree Algorithms

The lecture notes cover:
- CART
- ID3
- C4.5
- CHAID

These algorithms differ mainly in how they evaluate/select splits and construct trees.

## 8. Pruning

A decision tree can become too complex and overfit the training data.

Pruning removes or prevents unnecessary complexity so the model can generalize better.

### Pre-Pruning

Stop or restrict tree growth while the tree is being built.

Common controls:
- max_depth
- min_samples_split
- min_samples_leaf
- max_leaf_nodes
- min_impurity_decrease

### Post-Pruning

First allow the tree to grow and then remove unnecessary branches.

In scikit-learn, Cost-Complexity Pruning uses:

ccp_alpha

Larger pruning strength generally produces a simpler tree.

## 9. Regression

A Decision Tree Regressor predicts a continuous numerical value.

Examples:
- house price
- sales
- temperature
- demand

Instead of class purity, regression trees use the spread of target values to determine useful splits.

## 10. Variance

Variance measures how far observations are spread around their mean.

$$
Var(Y)=\frac{1}{n}\sum_{i=1}^{n}(y_i-\bar y)^2
$$

where:
- y_i = target value
- ȳ = mean target value
- n = number of observations

Lower variance means the values in the node are more closely grouped.

## 11. Variance Reduction

A regression tree looks for splits that reduce the weighted variance of the child nodes.

Conceptually:

$$
Variance\ Reduction =
Variance_{parent} - Weighted\ Variance_{children}
$$

A useful split creates child nodes whose target values are more similar to one another.

## 12. Practical scikit-learn Mapping

### Classifier

~~~python
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(
    max_depth=6,
    min_samples_leaf=30,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
~~~

### Regressor

~~~python
from sklearn.tree import DecisionTreeRegressor

model = DecisionTreeRegressor(
    max_depth=7,
    min_samples_leaf=20
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
~~~

### Evaluation

For classification, the notebooks use accuracy for the Titanic exercise and F1 score for the imbalanced ShopSmart assignment.

For regression, the notebook uses:
- Mean Squared Error (MSE)
- R² score

## 13. ShopSmart Assignment

The assignment asks for a Decision Tree based classification model that predicts whether an e-commerce visitor makes a purchase.

The provided problem statement describes 12,330 individual user sessions and asks for:
- Exploratory Data Analysis
- feature preprocessing and transformations
- Decision Tree classification
- F1-score evaluation because the data is imbalanced
- pruning to improve the classifier
- a benchmark F1 value of 0.55

Target:
Revenue

The practical notebook builds a preprocessing + Decision Tree pipeline, handles numerical/categorical features, uses class balancing, evaluates F1, and applies hyperparameter tuning.

## 14. Learning Takeaways

The most important connections from this study are:

**Classification → impurity → Entropy / Gini → Information Gain → splitting**

**Regression → variance → variance reduction → splitting**

**Tree becomes too complex → overfitting → pruning**

This folder is intended as a study reference for students and ML learners who want both the mathematics and the practical implementation of Decision Trees in one place.
