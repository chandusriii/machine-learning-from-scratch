# Decision Trees — Learning Journey

## What I Learned

Today I focused on Decision Trees as part of Supervised Learning and connected the mathematical concepts from my handwritten notes with practical model implementation.

The biggest improvement in my understanding was seeing the same algorithm from three perspectives:

1. **Concept** — how a Decision Tree asks feature-based questions to reach a prediction.
2. **Mathematics** — how impurity, information gain, and variance help the tree choose splits.
3. **Implementation** — how those ideas become preprocessing pipelines, model parameters, evaluation metrics, pruning, and hyperparameter tuning.

## Core Concepts

### Classification

A Decision Tree Classifier predicts a categorical outcome.

Key concepts:
- Pure vs Impure nodes
- Entropy
- Gini Impurity
- Entropy vs Gini
- Information Gain
- Feature selection for splits

### Entropy

$$
H(S)=-\sum_i p_i\log_2(p_i)
$$

Entropy represents impurity, disorder, or uncertainty.

### Gini Impurity

$$
Gini=1-\sum_i p_i^2
$$

Gini measures impurity in a node.

### Information Gain

$$
IG(S,A)=H(S)-\sum_v\frac{|S_v|}{|S|}H(S_v)
$$

Information Gain represents the reduction in entropy after a split.

### Regression

A Decision Tree Regressor predicts a continuous numerical value.

The key concept covered in the lecture is variance.

$$
Variance=\frac{1}{n}\sum_i(y_i-\bar y)^2
$$

A regression tree looks for splits that reduce weighted child variance.

$$
Variance\ Reduction
=
Parent\ Variance
-
Weighted\ Child\ Variance
$$

## Pruning

I also learned why an unrestricted Decision Tree can become unnecessarily complex and overfit.

### Pre-Pruning

Control tree growth while building the tree using parameters such as:
- max_depth
- min_samples_split
- min_samples_leaf
- max_leaf_nodes
- min_impurity_decrease

### Post-Pruning

Grow the tree and then remove unnecessary branches.

The practical classifier notebook demonstrates Cost-Complexity Pruning with ccp_alpha.

## Practical Work

### Decision Tree Classifier — Titanic

The classifier notebook covers:
- missing-value handling
- label encoding
- train/test split
- baseline Decision Tree
- pre-pruning with max_depth and min_samples_split
- post-pruning with ccp_alpha
- tree visualization

### Decision Tree Regressor — Diabetes

The regressor notebook covers:
- train/test split
- DecisionTreeRegressor
- max_depth and min_samples_leaf
- MSE
- R²
- tree visualization

### ShopSmart — Assignment 4

The ShopSmart exercise applies Decision Trees to e-commerce purchase prediction.

The assignment uses 12,330 sessions and requires EDA, preprocessing, Decision Tree classification, F1-score evaluation for the imbalanced target, and pruning.

The practical pipeline connects:
- numerical preprocessing
- categorical encoding
- ColumnTransformer
- Pipeline
- DecisionTreeClassifier
- class weighting
- F1 score
- classification report
- confusion matrix
- GridSearchCV

The current notebook run recorded a test F1 score of approximately **0.628** for the configured pipeline, with a 5-fold cross-validation best F1 of approximately **0.634** for max_depth=4 and min_samples_leaf=50.

## Final Mental Model

Classification:
**Impurity → Entropy / Gini → Information Gain → Split**

Regression:
**Variance → Variance Reduction → Split**

Complex tree:
**Overfitting risk → Pruning**

This repository section is meant to be useful not only as a record of my learning, but also as a readable reference for students and ML learners studying Decision Trees.
