# Early Fault Detection on Distribution Transformers

## Project Overview

This project develops a binary classification system for early fault detection
in distribution transformers.

The model combines transformer registry information with historical oil/thermal
readings and load measurements to predict whether a transformer is likely to
be in a Fault or Normal condition.

## Dataset

The project uses five datasets:

- Transformer registry
- Oil and thermal readings
- Load logs
- Labeled transformer fault status
- Unlabeled transformer records for prediction

Sensor and load readings were aggregated at transformer level before being merged
with the transformer registry data.

## Methodology

1. Raw dataset inspection
2. Data cleaning and aggregation
3. Exploratory Data Analysis
4. Feature engineering
5. Categorical encoding
6. Numerical feature scaling
7. Model training and evaluation
8. Stratified 5-fold cross-validation
9. Logistic Regression tuning
10. Final prediction of previously unlabeled transformers

## Models Evaluated

- Logistic Regression
- Random Forest
- Support Vector Machine (SVM)

Class imbalance was handled using balanced class weights.

## Final Model

The final model was a tuned Logistic Regression model with:

- `class_weight="balanced"`
- `C=10`

The model was selected based primarily on Fault detection performance,
with particular attention to Fault recall and Fault F1-score.

## Results

The final tuned Logistic Regression achieved:

- Cross-validation Accuracy: 0.676
- Fault Precision: 0.409
- Fault Recall: 0.596
- Fault F1-score: 0.479

The final model was retrained using all 198 labeled transformers and used to
predict 34 previously unlabeled transformers.

## Project Structure

```text
classification/
├── data/
├── notebooks/
└── src/