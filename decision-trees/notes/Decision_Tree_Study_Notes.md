# Decision Tree — Study Notes

These notes reorganize the handwritten lecture material into a readable reference while preserving the topics covered in the original notes.

## Decision Tree

A Decision Tree predicts an outcome by asking a sequence of questions about input features.

### Node terminology

- Pure node
- Impure node
- Root
- Decision/internal node
- Branch
- Leaf

## Important Keywords

- Entropy
- Gini Impurity
- Information Gain
- Variance
- Variance Reduction
- Pre-Pruning
- Post-Pruning

## Classification

### Entropy

$$
H(S) = -\sum_i p_i\log_2(p_i)
$$

For binary classification:

$$
H(S) = -p_{yes}\log_2(p_{yes}) - p_{no}\log_2(p_{no})
$$

Entropy measures impurity, disorder, or uncertainty.

- Pure node → entropy 0
- Binary maximum → entropy 1 at a 50/50 split

### Gini Impurity

$$
Gini = 1 - \sum_i p_i^2
$$

For binary classification:

$$
Gini = 1 - (p_{yes}^2 + p_{no}^2)
$$

- Pure node → Gini 0
- Binary maximum → Gini 0.5 at a 50/50 distribution

### Information Gain

$$
IG(S,A) = H(S) - \sum_v \frac{|S_v|}{|S|}H(S_v)
$$

It is the reduction in entropy after a split.

## Entropy vs Gini

The lecture notes compare them using:
1. Training speed
2. Split behaviour
3. Dataset size
4. Sensitivity to distribution
5. Common usage

The key conceptual point is that both are impurity metrics used for classification split selection.

## Decision Tree Algorithms

- CART
- ID3
- C4.5
- CHAID

## Pruning

Pruning is used to reduce unnecessary tree complexity and control overfitting.

### Pre-Pruning

Restrict tree growth while training.

Common parameters:
- Maximum depth
- Minimum samples required to split
- Minimum samples required in a leaf
- Maximum number of leaf nodes
- Minimum impurity decrease

### Post-Pruning

Grow the tree and then remove unnecessary branches.

The practical notebook demonstrates Cost-Complexity Pruning using ccp_alpha.

## Regression

A Decision Tree Regressor predicts a numerical target.

### Variance

$$
Variance = \frac{1}{n}\sum_i(y_i-\bar y)^2
$$

Variance describes how spread out target values are around their mean.

### Variance Reduction

The regression tree evaluates candidate splits by looking for a reduction in weighted child variance.

$$
Variance\ Reduction
=
Parent\ Variance
-
Weighted\ Child\ Variance
$$

## Practical Learning

The lecture was connected to practical implementation through scikit-learn:
- DecisionTreeClassifier
- DecisionTreeRegressor
- Pipeline
- ColumnTransformer
- StandardScaler
- OneHotEncoder
- GridSearchCV
- classification metrics
- regression metrics
- tree visualization

The practical work demonstrates how the mathematical ideas become model-building decisions.
