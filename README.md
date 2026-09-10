# Early Fault Detection on Distribution Transformers

A machine learning classification project for identifying whether a distribution transformer is likely to be in a **Fault** or **Normal** condition using transformer information, oil/thermal measurements, and load data.

## Project Overview

Distribution transformer faults can lead to equipment damage, service interruptions, and increased maintenance costs. Detecting potential faults early can help maintenance teams identify transformers that may require further inspection before a failure occurs.

This project develops a binary classification model to predict the condition of a transformer as **Fault** or **Normal**. Transformer registry information is combined with historical oil/thermal readings and load measurements.

The final model is intended as a decision-support tool for prioritizing transformers for further inspection, rather than as a replacement for physical inspection or engineering diagnosis.

## Business Problem

The objective is to identify potentially faulty distribution transformers early using available historical transformer, thermal/oil, and load information.

An effective early-warning system can help maintenance teams prioritize transformers that may require additional inspection or preventive maintenance.

Because missing a faulty transformer can be more costly than incorrectly flagging a normal transformer, this project gives particular attention to **Fault Recall** and **Fault F1-score**, rather than relying only on overall accuracy.

## Dataset

The project uses five datasets provided for the classification task:

| Dataset | Description | Records |
|---|---|---:|
| `transformer_registry.csv` | Transformer installation and equipment information | 232 |
| `oil_thermal_readings.csv` | Historical oil and thermal measurements | 1,602 |
| `load_logs.csv` | Historical transformer load measurements | 1,978 |
| `fault_status_labeled.csv` | Labeled transformer condition (`Fault` / `Normal`) | 198 |
| `fault_status_to_predict.csv` | Unlabeled transformers for final prediction | 34 |

The sensor and load measurements were aggregated at transformer level before being merged with the transformer registry information.

The labeled dataset contains the target variable used for supervised learning:

- `Fault`
- `Normal`

The 34 unlabeled transformer records were kept separate from model training and were used only during the final prediction stage.

**Dataset source:** Project-provided classification dataset.

## Technologies and Libraries

- Python
- Jupyter Notebook
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

The complete dependency list is available in [`requirements.txt`](requirements.txt).

## Methodology

The project follows the following workflow:

1. Inspect the raw datasets
2. Check data types, missing values, duplicates, and distributions
3. Clean and prepare the data
4. Aggregate historical sensor and load readings at transformer level
5. Merge the datasets using `transformer_id`
6. Perform exploratory data analysis
7. Engineer useful features
8. Encode categorical variables
9. Scale numerical features where required
10. Train and compare classification models
11. Evaluate models using stratified cross-validation
12. Tune the selected model
13. Retrain the final model using the labeled dataset
14. Predict the condition of the previously unlabeled transformers

## Models Evaluated

The following classification algorithms were evaluated:

- Logistic Regression
- Random Forest
- Support Vector Machine (SVM)

Because the dataset contains fewer Fault cases than Normal cases, class imbalance was addressed using balanced class weights.

## Evaluation Metrics

Overall accuracy alone can be misleading when the classes are imbalanced.

Therefore, the project focuses particularly on:

- **Fault Precision** — proportion of transformers predicted as Fault that were actually Fault.
- **Fault Recall** — proportion of actual Fault transformers correctly identified.
- **Fault F1-score** — balance between Fault precision and Fault recall.

Fault Recall is particularly important in this problem because failing to identify an actual faulty transformer can result in a missed maintenance opportunity.

## Final Model

The selected model is a tuned **Logistic Regression** model with:

- `class_weight="balanced"`
- `C=10`

The model was selected based primarily on its Fault detection performance, with particular attention to Fault Recall and Fault F1-score.

## Results

The final tuned Logistic Regression achieved the following cross-validation results:

| Metric | Score |
|---|---:|
| Accuracy | 0.676 |
| Fault Precision | 0.409 |
| Fault Recall | 0.596 |
| Fault F1-score | 0.479 |

### Interpretation

The model correctly identifies approximately **59.6% of the Fault cases** in the cross-validation evaluation.

The Fault Precision of **40.9%** indicates that a considerable number of transformers flagged as Fault were actually Normal.

This means the model should be treated as an **early-warning and prioritization tool**, rather than a definitive fault diagnosis system.

Fault Recall was prioritized because missing an actual fault can be more costly than performing an additional inspection on a transformer that turns out to be normal.

After model selection, the final Logistic Regression model was retrained using all **198 labeled transformers** and used to generate predictions for the **34 previously unlabeled transformers**.

## Project Structure

```text
transformer-fault-detection/
│
├── data/
│   ├── processed_data/
│   ├── fault_status_labeled.csv
│   ├── fault_status_to_predict.csv
│   ├── load_logs.csv
│   ├── oil_thermal_readings.csv
│   └── transformer_registry.csv
│
├── notebooks/
│   ├── 01_dataset_inspection.ipynb
│   ├── 02_data_cleaning_&_aggregation.ipynb
│   ├── 03_EDA.ipynb
│   ├── 04_feature_engineering_modeling_predictions.ipynb
│   └── 05_conclusion.ipynb
│
├── src/
│
├── .gitignore
├── problem_statement.md
├── README.md
└── requirements.txt
```
## How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/nithin3114/transformer-fault-detection.git
cd transformer-fault-detection
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

### 3. Activate the Environment

For Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the Project Pipeline

The `src/` directory contains the Python pipeline developed to run the main project workflow automatically.

The pipeline handles the required data preparation, feature processing, model workflow, and prediction steps without requiring each notebook to be executed manually.

Run the pipeline using:

```bash
python src/transformer_fault_prediction_pipeline.py
```
### 6. Explore the Project Using Jupyter Notebook

To inspect the analysis and workflow step by step:

```bash
jupyter notebook
```

The notebooks are organized in numerical order:

```text
01 → 02 → 03 → 04 → 05
```

They contain the detailed dataset inspection, data cleaning and aggregation, exploratory data analysis, feature engineering, modeling, prediction, and conclusion workflow.

## Challenges

Some of the main challenges encountered during the project included:

- The raw datasets were stored at different levels of detail, requiring historical oil/thermal and load measurements to be aggregated before merging with transformer-level information.
- The target classes were imbalanced, making accuracy alone insufficient for evaluating the model.
- Initial model approaches did not provide the desired Fault detection performance, which led to comparison of multiple algorithms and the use of balanced class weights.
- The limited number of labeled transformers restricted the amount of training data available for the classification model.

## Limitations

- The labeled dataset contains only 198 transformers, which limits the amount of data available for model training and evaluation.
- Model performance is dependent on the quality and representativeness of the provided transformer, thermal/oil, and load data.
- The Fault Recall of 0.596 means that the model does not identify every actual Fault case.
- The model should not be treated as a standalone diagnostic system. Predictions should be followed by appropriate engineering inspection and verification.
- The dataset is specific to the provided project scenario, so model performance may not generalize directly to other transformer populations or operating environments.

## Future Work

Possible improvements include:

- Collecting a larger and more diverse set of labeled transformer records.
- Adding additional electrical, environmental, and maintenance-history features.
- Exploring additional models and hyperparameter optimization techniques.
- Evaluating probability thresholds to prioritize higher Fault Recall when appropriate.
- Testing the model on new real-world transformer data.
- Developing a monitoring interface that can help maintenance teams review model predictions and prioritize inspections.

## Conclusion

This project demonstrates an end-to-end machine learning workflow for transformer fault classification, including data inspection, cleaning, aggregation, exploratory analysis, feature engineering, model comparison, evaluation, tuning, and final prediction.

The final Logistic Regression model provides a **selected early-warning approach** for identifying potentially faulty transformers. Its performance also highlights the limitations of working with a small and imbalanced labeled dataset, making further data collection and validation important for any real-world deployment.
