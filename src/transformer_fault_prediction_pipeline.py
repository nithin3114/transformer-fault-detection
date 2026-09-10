# imports 
import pandas as pd
import numpy as np

from IPython.display import display

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.model_selection import StratifiedKFold, cross_validate

from sklearn.metrics import (
    make_scorer,
    precision_score,
    recall_score,
    f1_score
)

# ============================================================
# 1. LOAD DATA
# ============================================================

def load_data(data_path):
    """
    Load the five raw transformer datasets.
    """

    registry = pd.read_csv(
        data_path + r"\transformer_registry.csv"
    )

    oil = pd.read_csv(
        data_path + r"\oil_thermal_readings.csv"
    )

    load = pd.read_csv(
        data_path + r"\load_logs.csv"
    )

    labels = pd.read_csv(
        data_path + r"\fault_status_labeled.csv"
    )

    prediction = pd.read_csv(
        data_path + r"\fault_status_to_predict.csv"
    )

    return registry, oil, load, labels, prediction

# ============================================================
# 2. CHECK INPUT FILES
# ============================================================

def check_input_files(data_path):
    """
    Check whether all required CSV files are present
    in the given data folder.
    """

    required_files = [
        "transformer_registry.csv",
        "oil_thermal_readings.csv",
        "load_logs.csv",
        "fault_status_labeled.csv",
        "fault_status_to_predict.csv"
    ]

    print("=" * 70)
    print("CHECKING INPUT FILES")
    print("=" * 70)

    for file_name in required_files:

        file_path = data_path + "\\" + file_name

        try:
            pd.read_csv(file_path, nrows=1)
            print(f"✓ {file_name}")

        except FileNotFoundError:
            raise FileNotFoundError(
                f"Required file not found: {file_name}"
            )

    print("\n✓ All required files found.") 


# ============================================================
# 3. VALIDATE INPUT DATA
# ============================================================

def validate_data(registry, oil, load, labels, prediction):
    """
    Validate the structure and basic integrity of the
    five input datasets.
    """

    print("\n" + "=" * 70)
    print("VALIDATING INPUT DATA")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. REQUIRED COLUMNS
    # --------------------------------------------------------

    required_columns = {

        "registry": [
            "transformer_id",
            "install_year",
            "rated_kva",
            "cooling_type",
            "zone"
        ],

        "oil": [
            "record_id",
            "transformer_id",
            "reading_date",
            "dissolved_gas_ppm",
            "oil_temp_c"
        ],

        "load": [
            "record_id",
            "transformer_id",
            "log_date",
            "load_pct_of_rated"
        ],

        "labels": [
            "transformer_id",
            "fault_status"
        ],

        "prediction": [
            "transformer_id"
        ]
    }

    datasets = {
        "registry": registry,
        "oil": oil,
        "load": load,
        "labels": labels,
        "prediction": prediction
    }

    for name, df in datasets.items():

        missing_columns = set(required_columns[name]) - set(df.columns)

        if missing_columns:
            raise ValueError(
                f"{name} is missing columns: {missing_columns}"
            )

        print(f"✓ {name} columns validated")

    # --------------------------------------------------------
    # 2. TRANSFORMER ID CHECK
    # --------------------------------------------------------

    registry_ids = set(registry["transformer_id"])
    oil_ids = set(oil["transformer_id"])
    load_ids = set(load["transformer_id"])
    label_ids = set(labels["transformer_id"])
    prediction_ids = set(prediction["transformer_id"])

    if not oil_ids.issubset(registry_ids):
        raise ValueError(
            "Oil data contains transformer IDs not present in registry."
        )

    if not load_ids.issubset(registry_ids):
        raise ValueError(
            "Load data contains transformer IDs not present in registry."
        )

    if not label_ids.issubset(registry_ids):
        raise ValueError(
            "Label data contains transformer IDs not present in registry."
        )

    if not prediction_ids.issubset(registry_ids):
        raise ValueError(
            "Prediction data contains transformer IDs not present in registry."
        )

    print("✓ Transformer ID relationships validated")

    # --------------------------------------------------------
    # 3. DUPLICATE ID CHECK
    # --------------------------------------------------------

    if registry["transformer_id"].duplicated().any():
        raise ValueError(
            "Duplicate transformer IDs found in registry."
        )

    if labels["transformer_id"].duplicated().any():
        raise ValueError(
            "Duplicate transformer IDs found in labels."
        )

    if prediction["transformer_id"].duplicated().any():
        raise ValueError(
            "Duplicate transformer IDs found in prediction data."
        )

    print("✓ Duplicate ID checks passed")

    # --------------------------------------------------------
    # 4. TARGET VALUE CHECK
    # --------------------------------------------------------

    valid_statuses = {"Normal", "Fault"}

    actual_statuses = set(labels["fault_status"].dropna().unique())

    if not actual_statuses.issubset(valid_statuses):
        raise ValueError(
            f"Unexpected fault_status values found: "
            f"{actual_statuses - valid_statuses}"
        )

    print("✓ Fault status values validated")

    # --------------------------------------------------------
    # 5. LABEL / PREDICTION OVERLAP CHECK
    # --------------------------------------------------------

    overlap = label_ids.intersection(prediction_ids)

    if overlap:
        raise ValueError(
            f"Transformers appear in both labeled and prediction data: "
            f"{overlap}"
        )

    print("✓ Label/prediction separation validated")

    print("\n✓ INPUT DATA VALIDATION PASSED")

# ============================================================
# 4. CLEAN DATA
# ============================================================

def clean_data(registry, oil, load, labels, prediction):
    """
    Clean the five input datasets based on the data-quality
    decisions identified during exploratory analysis.
    """

    print("\n" + "=" * 70)
    print("CLEANING DATA")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. CREATE COPIES
    # --------------------------------------------------------

    registry = registry.copy()
    oil = oil.copy()
    load = load.copy()
    labels = labels.copy()
    prediction = prediction.copy()

    # --------------------------------------------------------
    # 2. CONVERT DATE COLUMNS
    # --------------------------------------------------------

    oil["reading_date"] = pd.to_datetime(
        oil["reading_date"],
        errors="coerce"
    )

    load["log_date"] = pd.to_datetime(
        load["log_date"],
        errors="coerce"
    )

    print("✓ Date columns converted")

    # --------------------------------------------------------
    # 3. HANDLE INVALID NEGATIVE GAS VALUES
    # --------------------------------------------------------

    negative_gas = (
        oil["dissolved_gas_ppm"] < 0
    ).sum()

    oil.loc[
        oil["dissolved_gas_ppm"] < 0,
        "dissolved_gas_ppm"
    ] = np.nan

    print(
        f"✓ Negative gas values converted to NaN: "
        f"{negative_gas}"
    )

    # --------------------------------------------------------
    # 4. HANDLE MISSING COOLING TYPE
    # --------------------------------------------------------

    missing_cooling = registry["cooling_type"].isna().sum()

    registry["cooling_type"] = registry[
        "cooling_type"
    ].fillna("Unknown")

    print(
        f"✓ Missing cooling_type values replaced: "
        f"{missing_cooling}"
    )

    # --------------------------------------------------------
    # 5. REMOVE DUPLICATE ROWS
    # --------------------------------------------------------

    duplicate_counts = {
        "registry": registry.duplicated().sum(),
        "oil": oil.duplicated().sum(),
        "load": load.duplicated().sum(),
        "labels": labels.duplicated().sum(),
        "prediction": prediction.duplicated().sum()
    }

    for name, count in duplicate_counts.items():

        if count > 0:
            print(
                f"✓ Removing {count} duplicate rows from {name}"
            )

    registry = registry.drop_duplicates()
    oil = oil.drop_duplicates()
    load = load.drop_duplicates()
    labels = labels.drop_duplicates()
    prediction = prediction.drop_duplicates()

    # --------------------------------------------------------
    # 6. KEEP MISSING LOAD VALUES
    # --------------------------------------------------------

    missing_load = load["load_pct_of_rated"].isna().sum()

    print(
        f"✓ Missing load values retained as NaN: "
        f"{missing_load}"
    )

    # --------------------------------------------------------
    # 7. CLEANING SUMMARY
    # --------------------------------------------------------

    print("\n✓ DATA CLEANING COMPLETE")

    return registry, oil, load, labels, prediction

# ============================================================
# 5. AGGREGATE SENSOR DATA
# ============================================================

def aggregate_data(oil, load):
    """
    Aggregate multiple oil/thermal readings and load logs
    into transformer-level features.
    """

    print("\n" + "=" * 70)
    print("AGGREGATING SENSOR DATA")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. AGGREGATE OIL / THERMAL DATA
    # --------------------------------------------------------

    oil_agg = oil.groupby("transformer_id").agg(

        gas_mean=("dissolved_gas_ppm", "mean"),
        gas_max=("dissolved_gas_ppm", "max"),
        gas_std=("dissolved_gas_ppm", "std"),

        oil_temp_mean=("oil_temp_c", "mean"),
        oil_temp_max=("oil_temp_c", "max"),
        oil_temp_std=("oil_temp_c", "std")

    ).reset_index()

    print(
        "✓ Oil/thermal data aggregated:",
        oil_agg.shape
    )

    # --------------------------------------------------------
    # 2. AGGREGATE LOAD DATA
    # --------------------------------------------------------

    load_agg = load.groupby("transformer_id").agg(

        load_mean=("load_pct_of_rated", "mean"),
        load_max=("load_pct_of_rated", "max"),
        load_std=("load_pct_of_rated", "std")

    ).reset_index()

    print(
        "✓ Load data aggregated:",
        load_agg.shape
    )

    # --------------------------------------------------------
    # 3. RETURN AGGREGATED DATA
    # --------------------------------------------------------

    return oil_agg, load_agg

# ============================================================
# 6. MERGE DATA
# ============================================================

def merge_data(registry, oil_agg, load_agg, labels, prediction):
    """
    Merge registry, aggregated sensor data, and target IDs
    to create labeled and prediction datasets.
    """

    print("\n" + "=" * 70)
    print("MERGING DATA")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. MERGE REGISTRY + OIL DATA
    # --------------------------------------------------------

    features = registry.merge(
        oil_agg,
        on="transformer_id",
        how="left"
    )

    print(
        "✓ Registry + oil data merged:",
        features.shape
    )

    # --------------------------------------------------------
    # 2. MERGE LOAD DATA
    # --------------------------------------------------------

    features = features.merge(
        load_agg,
        on="transformer_id",
        how="left"
    )

    print(
        "✓ Load data merged:",
        features.shape
    )

    # --------------------------------------------------------
    # 3. CREATE LABELED DATASET
    # --------------------------------------------------------

    labeled_data = features.merge(
        labels,
        on="transformer_id",
        how="inner"
    )

    print(
        "✓ Labeled dataset created:",
        labeled_data.shape
    )

    # --------------------------------------------------------
    # 4. CREATE PREDICTION DATASET
    # --------------------------------------------------------

    prediction_data = features.merge(
        prediction,
        on="transformer_id",
        how="inner"
    )

    print(
        "✓ Prediction dataset created:",
        prediction_data.shape
    )

    print("\n✓ DATA MERGING COMPLETE")

    return labeled_data, prediction_data

# ============================================================
# 7. PREPARE FEATURES
# ============================================================

def prepare_features(labeled_data):
    """
    Separate features and target and create the preprocessing
    configuration for machine learning models.
    """

    print("\n" + "=" * 70)
    print("PREPARING FEATURES")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. SEPARATE FEATURES AND TARGET
    # --------------------------------------------------------

    X = labeled_data.drop(
        columns=["fault_status", "transformer_id"]
    )

    y = labeled_data["fault_status"].map({
        "Normal": 0,
        "Fault": 1
    })

    print("✓ Features and target separated")

    # --------------------------------------------------------
    # 2. DEFINE FEATURE TYPES
    # --------------------------------------------------------

    categorical_features = [
        "cooling_type",
        "zone"
    ]

    numerical_features = [
        "install_year",
        "rated_kva",
        "gas_mean",
        "gas_max",
        "gas_std",
        "oil_temp_mean",
        "oil_temp_max",
        "oil_temp_std",
        "load_mean",
        "load_max",
        "load_std"
    ]

    print(
        f"✓ Numerical features: {len(numerical_features)}"
    )

    print(
        f"✓ Categorical features: {len(categorical_features)}"
    )

    # --------------------------------------------------------
    # 3. CREATE PREPROCESSOR
    # --------------------------------------------------------

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                MinMaxScaler(),
                numerical_features
            ),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features
            )
        ]
    )

    print("✓ Preprocessor created")

    # --------------------------------------------------------
    # 4. TRAIN / TEST SPLIT
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("\n✓ Train/test split completed")

    print(
        "Training samples:",
        X_train.shape[0]
    )

    print(
        "Testing samples :",
        X_test.shape[0]
    )

    print(
        "Training faults :",
        y_train.sum()
    )

    print(
        "Testing faults  :",
        y_test.sum()
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessor,
        X,
        y
    )

# ============================================================
# 8. EVALUATE CANDIDATE MODELS
# ============================================================

def evaluate_models(
    X_train,
    y_train,
    preprocessor
):
    """
    Evaluate candidate classification models using
    stratified cross-validation on the training data.
    """

    print("\n" + "=" * 70)
    print("EVALUATING CANDIDATE MODELS")
    print("=" * 70)

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    scoring = {
        "accuracy": "accuracy",
        "precision": make_scorer(
            precision_score,
            zero_division=0
        ),
        "recall": make_scorer(
            recall_score,
            zero_division=0
        ),
        "f1": make_scorer(
            f1_score,
            zero_division=0
        ),
        "roc_auc": "roc_auc"
    }

    models = {
        "Logistic Regression": LogisticRegression(
            C=10,
            class_weight="balanced",
            random_state=42
        ),

        "Random Forest": RandomForestClassifier(
            class_weight="balanced",
            random_state=42
        ),

        "SVM": SVC(
            class_weight="balanced",
            random_state=42
        )
    }

    results = []

    for model_name, model in models.items():

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        cv_result = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring
        )

        results.append({
            "Model": model_name,
            "Accuracy": cv_result["test_accuracy"].mean(),
            "Precision": cv_result["test_precision"].mean(),
            "Recall": cv_result["test_recall"].mean(),
            "F1": cv_result["test_f1"].mean(),
            "ROC-AUC": cv_result["test_roc_auc"].mean()
        })

        print(f"✓ {model_name} evaluated")

    results_df = pd.DataFrame(results)

    print("\nMODEL COMPARISON")
    print("-" * 70)

    display(
        results_df.round(3)
    )

    return results_df

# ============================================================
# 9. SELECT BEST MODEL
# ============================================================

def select_best_model(results_df):
    """
    Select the best model based on the project's
    fault-detection priorities.
    """

    print("\n" + "=" * 70)
    print("SELECTING BEST MODEL")
    print("=" * 70)

    ranked_results = results_df.sort_values(
        by=[
            "Recall",
            "F1",
            "ROC-AUC",
            "Accuracy",
            "Precision"
        ],
        ascending=False
    ).reset_index(drop=True)

    best_model = ranked_results.iloc[0]["Model"]

    print("\nModel ranking:")
    display(ranked_results.round(3))

    print(
        f"\n✓ Selected model: {best_model}"
    )

    return best_model, ranked_results

# ============================================================
# 10. EVALUATE SELECTED MODEL
# ============================================================

def evaluate_selected_model(
    best_model,
    X_train,
    X_test,
    y_train,
    y_test,
    preprocessor
):
    """
    Train the automatically selected model on the training data
    and evaluate it on the untouched test data.
    """

    print("\n" + "=" * 70)
    print("EVALUATING SELECTED MODEL")
    print("=" * 70)

    models = {
        "Logistic Regression": LogisticRegression(
            C=10,
            class_weight="balanced",
            random_state=42
        ),

        "Random Forest": RandomForestClassifier(
            class_weight="balanced",
            random_state=42
        ),

        "SVM": SVC(
            class_weight="balanced",
            random_state=42
        )
    }

    selected_model = models[best_model]

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", selected_model)
    ])

    pipeline.fit(
        X_train,
        y_train
    )

    y_pred = pipeline.predict(
        X_test
    )

    y_score = pipeline.predict_proba(
        X_test
    )[:, 1]

    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        roc_auc_score,
        confusion_matrix
    )

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        y_score
    )

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    print(
        f"\nSelected model: {best_model}"
    )

    print("\nTEST SET PERFORMANCE")
    print("-" * 40)

    print(f"Accuracy  : {accuracy:.3f}")
    print(f"Precision : {precision:.3f}")
    print(f"Recall    : {recall:.3f}")
    print(f"F1 Score  : {f1:.3f}")
    print(f"ROC-AUC   : {roc_auc:.3f}")

    print("\nConfusion Matrix")
    print("-" * 40)
    print(cm)

    return pipeline, {
        "Model": best_model,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "ROC-AUC": roc_auc
    }

# ============================================================
# 11. TRAIN FINAL MODEL
# ============================================================

def train_final_model(
    best_model,
    X,
    y,
    preprocessor
):
    """
    Train the selected model on the complete labeled dataset.
    """

    print("\n" + "=" * 70)
    print("TRAINING FINAL MODEL")
    print("=" * 70)

    models = {
        "Logistic Regression": LogisticRegression(
            C=10,
            class_weight="balanced",
            random_state=42
        ),

        "Random Forest": RandomForestClassifier(
            class_weight="balanced",
            random_state=42
        ),

        "SVM": SVC(
            class_weight="balanced",
            random_state=42
        )
    }

    selected_model = models[best_model]

    final_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", selected_model)
    ])

    final_pipeline.fit(
        X,
        y
    )

    print(
        f"✓ Final model trained: {best_model}"
    )

    print(
        f"✓ Training samples used: {X.shape[0]}"
    )

    print("\n✓ FINAL MODEL TRAINING COMPLETE")

    return final_pipeline

# ============================================================
# 12. GENERATE PREDICTIONS
# ============================================================

def generate_predictions(
    final_pipeline,
    prediction_data
):
    """
    Generate fault predictions and fault scores
    for the unlabeled transformers.
    """

    print("\n" + "=" * 70)
    print("GENERATING PREDICTIONS")
    print("=" * 70)

    prediction_ids = prediction_data[
        "transformer_id"
    ]

    X_prediction = prediction_data.drop(
        columns=["transformer_id"]
    )

    predictions = final_pipeline.predict(
        X_prediction
    )

    fault_scores = final_pipeline.predict_proba(
        X_prediction
    )[:, 1]

    results = pd.DataFrame({
        "transformer_id": prediction_ids,
        "predicted_status": np.where(
            predictions == 1,
            "Fault",
            "Normal"
        ),
        "fault_score": fault_scores
    })

    print(
        f"✓ Predictions generated for "
        f"{len(results)} transformers"
    )

    print(
        f"✓ Fault predictions: "
        f"{(results['predicted_status'] == 'Fault').sum()}"
    )

    print(
        f"✓ Normal predictions: "
        f"{(results['predicted_status'] == 'Normal').sum()}"
    )

    print("\nPREDICTIONS")
    print("-" * 70)

    display(
        results.round({
            "fault_score": 3
        })
    )

    return results

# ============================================================
# 13. ASSIGN INSPECTION PRIORITY
# ============================================================

def assign_priority(prediction_results):
    """
    Assign inspection priority based on the predicted
    fault score.
    """

    print("\n" + "=" * 70)
    print("ASSIGNING INSPECTION PRIORITY")
    print("=" * 70)

    prediction_results = prediction_results.copy()

    prediction_results["inspection_priority"] = pd.cut(
        prediction_results["fault_score"],
        bins=[-np.inf, 0.40, 0.70, np.inf],
        labels=["Low", "Medium", "High"]
    )

    print("✓ Inspection priorities assigned")

    print("\nPRIORITY DISTRIBUTION")
    print("-" * 40)

    print(
        prediction_results[
            "inspection_priority"
        ].value_counts()
    )

    print("\nFINAL PREDICTION RESULTS")
    print("-" * 70)

    display(
        prediction_results.round({
            "fault_score": 3
        })
    )

    return prediction_results

# ============================================================
# 14. SAVE OUTPUTS
# ============================================================

def save_outputs(
    data_path,
    output_path,
    labeled_data,
    prediction_data,
    model_results,
    ranked_results,
    test_results,
    prediction_results
):
    """
    Save processed datasets and final pipeline outputs.
    """

    print("\n" + "=" * 70)
    print("SAVING OUTPUTS")
    print("=" * 70)

    import os

    # Create output folder if it does not exist

    output_path_final = output_path + r"\output"

    os.makedirs(output_path_final, exist_ok=True)

    # Save processed datasets back into data folder
    processed_path = data_path + r"\processed_data"

    os.makedirs(processed_path, exist_ok=True)

    labeled_data.to_csv(
        processed_path + r"\labeled_data.csv",
        index=False
    )

    prediction_data.to_csv(
        processed_path + r"\prediction_data.csv",
        index=False
    )

    # Save model evaluation results
    model_results.to_csv(
        output_path_final + r"\model_comparison.csv",
        index=False
    )

    ranked_results.to_csv(
        output_path_final + r"\model_ranking.csv",
        index=False
    )

    pd.DataFrame([test_results]).to_csv(
        output_path_final + r"\final_test_evaluation.csv",
        index=False
    )

    # Save final predictions
    prediction_results.to_csv(
        output_path_final + r"\final_predictions.csv",
        index=False
    )

    print("\n✓ Processed datasets updated in data folder")
    print("✓ Model comparison saved")
    print("✓ Model ranking saved")
    print("✓ Test evaluation saved")
    print("✓ Final predictions saved")

    print("\nOUTPUT LOCATION")
    print("-" * 40)
    print(output_path_final)

# ============================================================
# RUN COMPLETE PIPELINE
# ============================================================

def run_pipeline():

    data_path = input(
        "Enter the path to the folder containing the 5 CSV files: "
    ).strip()

    check_input_files(data_path)

    registry, oil, load, labels, prediction = load_data(data_path)

    print("\nDATA LOADED")
    print("-" * 40)
    print("Registry shape   :", registry.shape)
    print("Oil readings     :", oil.shape)
    print("Load logs        :", load.shape)
    print("Labels           :", labels.shape)
    print("Prediction data  :", prediction.shape)

    validate_data(
        registry,
        oil,
        load,
        labels,
        prediction
    )

    registry, oil, load, labels, prediction = clean_data(
        registry,
        oil,
        load,
        labels,
        prediction
    )

    oil_agg, load_agg = aggregate_data(
        oil,
        load
    )

    labeled_data, prediction_data = merge_data(
        registry,
        oil_agg,
        load_agg,
        labels,
        prediction
    )

    (
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessor,
        X,
        y
    ) = prepare_features(labeled_data)

    model_results = evaluate_models(
        X_train,
        y_train,
        preprocessor
    )

    best_model, ranked_results = select_best_model(
        model_results
    )

    selected_pipeline, test_results = evaluate_selected_model(
        best_model,
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessor
    )

    final_pipeline = train_final_model(
        best_model,
        X,
        y,
        preprocessor
    )

    prediction_results = generate_predictions(
        final_pipeline,
        prediction_data
    )

    prediction_results = assign_priority(
        prediction_results
    )

    output_path = input(
        "Enter the path where the output folder should be created: "
    ).strip()

    save_outputs(
        data_path,
        output_path,
        labeled_data,
        prediction_data,
        model_results,
        ranked_results,
        test_results,
        prediction_results
    )


if __name__ == "__main__":
    run_pipeline()