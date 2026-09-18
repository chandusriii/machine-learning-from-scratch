# LinkedIn Post — Decision Trees | Supervised Learning

Today, I spent my learning session going deeper into **Decision Trees** as part of Supervised Learning.

What changed for me today was not just learning the algorithm, but connecting the **mathematics, intuition, and practical implementation** behind it.

### What I explored

🌳 **Decision Tree Classification**
- Pure vs. Impure nodes
- Entropy
- Gini Impurity
- Entropy vs. Gini
- Information Gain
- Feature selection for splitting

📉 **Pruning**
- Pre-Pruning
- Post-Pruning
- max_depth
- min_samples_split
- min_samples_leaf
- Cost-Complexity Pruning with ccp_alpha

📈 **Decision Tree Regression**
- Variance
- Variance Reduction
- MSE
- R²

### Practical implementation

I also worked with multiple notebooks and connected the concepts to real model-building workflows using scikit-learn.

I practiced:
- Decision Tree Classifier on the Titanic dataset
- Decision Tree Regressor on the Diabetes dataset
- A ShopSmart e-commerce classification assignment with preprocessing, Pipeline, ColumnTransformer, class weighting, F1-score evaluation, and GridSearchCV

One of the most useful takeaways was understanding that a Decision Tree is not simply about calling model.fit(). The important part is understanding **why a split is selected, how impurity is reduced, how variance guides regression splits, and how pruning controls model complexity.**

Today's learning can be summarized as:

**Classification → Entropy / Gini → Information Gain → Splitting**

**Regression → Variance → Variance Reduction → Splitting**

**Complexity → Overfitting → Pruning**

I’ve documented the formulas, concepts, handwritten study notes, practical notebooks, and ShopSmart assignment in my GitHub repository so that the learning process is easy to follow and useful for others preparing for Machine Learning.

🔗 GitHub: https://github.com/chandusriii/machine-learning-from-scratch

#MachineLearning #DecisionTrees #SupervisedLearning #Python #ScikitLearn #DataScience #AI #MachineLearningJourney #LearningInPublic
