# ShopSmart — Assignment 4

## Problem

ShopSmart wants to predict whether an e-commerce visitor is likely to complete a purchase based on session behaviour.

The assignment describes **12,330 individual user sessions** collected over one year and a mixture of numerical and categorical features.

## Required Work

1. Exploratory Data Analysis (EDA)
2. Feature preprocessing and transformations
3. Decision Tree based classification model
4. Evaluation using F1 score because the dataset is imbalanced
5. Pruning to improve Decision Tree performance

**Benchmark F1:** 0.55

## Target

Revenue — whether the visitor made a purchase.

## Features

- Administrative
- Administrative_Duration
- Informational
- Informational_Duration
- ProductRelated
- ProductRelated_Duration
- BounceRates
- ExitRates
- PageValues
- SpecialDay
- Month
- OperatingSystems
- Browser
- Region
- TrafficType
- VisitorType
- Weekend

## Practical Pipeline

The accompanying ShopSmart notebook demonstrates:

~~~text
Raw data
   ↓
Separate features and target
   ↓
Train / test split
   ↓
Numerical + categorical feature identification
   ↓
ColumnTransformer
   ├── Numerical → StandardScaler
   └── Categorical → OneHotEncoder
   ↓
DecisionTreeClassifier
   ↓
F1 score + classification report + confusion matrix
   ↓
GridSearchCV
~~~

The model configuration in the notebook uses tree-growth controls and class weighting to address model complexity and class imbalance.

## Learning Connection

This assignment turns the lecture theory into an applied workflow:

- Decision Tree → classification
- Impurity → split decisions
- Pre-pruning → max_depth, min_samples_leaf
- Imbalanced target → F1 score and class weighting
- Hyperparameter tuning → GridSearchCV
