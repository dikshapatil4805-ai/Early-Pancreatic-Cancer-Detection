"""
Clinical data loader and preprocessing for multimodal pancreatic cancer detection.

This module:
- Loads train/validation/test clinical CSV files.
- Uses only the approved clinical model features.
- Fits preprocessing on the training set only.
- One-hot encodes categorical variables.
- Preserves missingness indicators.
- Applies the same fitted preprocessing to validation and test sets.
"""

from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CLINICAL_DATA_DIR = (
    PROJECT_ROOT / "data" / "processed" / "clinical"
)


# ---------------------------------------------------------------------------
# Feature definitions
# ---------------------------------------------------------------------------

TARGET_COLUMN = "target_pdac"

NUMERIC_FEATURES = [
    "patient_age_years",
]

CATEGORICAL_FEATURES = [
    "patient_sex",
    "scanner",
]

MISSINGNESS_FEATURES = [
    "age_missing",
    "sex_missing",
    "scanner_missing",
]

MODEL_FEATURES = (
    NUMERIC_FEATURES
    + CATEGORICAL_FEATURES
    + MISSINGNESS_FEATURES
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_clinical_csvs(
    data_dir: Path = CLINICAL_DATA_DIR,
) -> Dict[str, pd.DataFrame]:
    """
    Load the prepared clinical train/validation/test datasets.

    Parameters
    ----------
    data_dir : Path
        Directory containing train.csv, validation.csv, and test.csv.

    Returns
    -------
    dict
        Dictionary containing:
        - train
        - validation
        - test
    """

    data_dir = Path(data_dir)

    files = {
        "train": data_dir / "train.csv",
        "validation": data_dir / "validation.csv",
        "test": data_dir / "test.csv",
    }

    for split_name, file_path in files.items():
        if not file_path.exists():
            raise FileNotFoundError(
                f"Clinical {split_name} file not found: {file_path}"
            )

    datasets = {
        split_name: pd.read_csv(file_path)
        for split_name, file_path in files.items()
    }

    return datasets


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_clinical_columns(
    datasets: Dict[str, pd.DataFrame],
) -> None:
    """
    Verify that all required model columns are present.

    Raises
    ------
    ValueError
        If required columns are missing.
    """

    required_columns = set(MODEL_FEATURES + [TARGET_COLUMN])

    for split_name, df in datasets.items():
        missing_columns = required_columns - set(df.columns)

        if missing_columns:
            raise ValueError(
                f"{split_name} dataset is missing required columns: "
                f"{sorted(missing_columns)}"
            )


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def build_preprocessor() -> ColumnTransformer:
    """
    Create the clinical preprocessing pipeline.

    Numeric:
        - StandardScaler for patient age.

    Categorical:
        - OneHotEncoder for sex and scanner.
        - Unknown categories are ignored.

    Missingness indicators:
        - Passed through unchanged.

    Returns
    -------
    ColumnTransformer
        Unfitted preprocessing transformer.
    """

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                StandardScaler(),
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                CATEGORICAL_FEATURES,
            ),
            (
                "missingness",
                "passthrough",
                MISSINGNESS_FEATURES,
            ),
        ],
        remainder="drop",
    )

    return preprocessor


# ---------------------------------------------------------------------------
# Prepare datasets
# ---------------------------------------------------------------------------

def prepare_clinical_data(
    data_dir: Path = CLINICAL_DATA_DIR,
) -> Tuple[
    object,
    pd.DataFrame,
    pd.Series,
    pd.DataFrame,
    pd.Series,
    pd.DataFrame,
    pd.Series,
]:
    """
    Load and preprocess the clinical datasets.

    The preprocessing transformer is fitted ONLY on the training set.

    Returns
    -------
    preprocessor
        Fitted ColumnTransformer.

    X_train
        Preprocessed training features.

    y_train
        Training labels.

    X_validation
        Preprocessed validation features.

    y_validation
        Validation labels.

    X_test
        Preprocessed test features.

    y_test
        Test labels.
    """

    datasets = load_clinical_csvs(data_dir)

    validate_clinical_columns(datasets)

    train_df = datasets["train"]
    validation_df = datasets["validation"]
    test_df = datasets["test"]

    # Select only approved model features.
    X_train_raw = train_df[MODEL_FEATURES].copy()
    X_validation_raw = validation_df[MODEL_FEATURES].copy()
    X_test_raw = test_df[MODEL_FEATURES].copy()

    # Extract target.
    y_train = train_df[TARGET_COLUMN].astype(int)
    y_validation = validation_df[TARGET_COLUMN].astype(int)
    y_test = test_df[TARGET_COLUMN].astype(int)

    # Build and fit preprocessing ONLY on training data.
    preprocessor = build_preprocessor()

    X_train_array = preprocessor.fit_transform(X_train_raw)

    # Apply the fitted training transformation to validation/test.
    X_validation_array = preprocessor.transform(X_validation_raw)
    X_test_array = preprocessor.transform(X_test_raw)

    # Convert transformed arrays into DataFrames with feature names.
    feature_names = preprocessor.get_feature_names_out()

    X_train = pd.DataFrame(
        X_train_array,
        columns=feature_names,
        index=train_df.index,
    )

    X_validation = pd.DataFrame(
        X_validation_array,
        columns=feature_names,
        index=validation_df.index,
    )

    X_test = pd.DataFrame(
        X_test_array,
        columns=feature_names,
        index=test_df.index,
    )

    return (
        preprocessor,
        X_train,
        y_train,
        X_validation,
        y_validation,
        X_test,
        y_test,
    )


# ---------------------------------------------------------------------------
# Simple inspection helper
# ---------------------------------------------------------------------------

def print_dataset_summary(
    data_dir: Path = CLINICAL_DATA_DIR,
) -> None:
    """
    Print a summary of the prepared clinical datasets.
    """

    datasets = load_clinical_csvs(data_dir)

    validate_clinical_columns(datasets)

    print("\nClinical Dataset Summary")
    print("=" * 50)

    for split_name, df in datasets.items():
        print(f"\n{split_name.upper()}")
        print(f"Rows: {len(df)}")
        print(f"Columns: {len(df.columns)}")
        print(
            "PDAC:",
            int((df[TARGET_COLUMN] == 1).sum()),
        )
        print(
            "Non-PDAC:",
            int((df[TARGET_COLUMN] == 0).sum()),
        )

    print("\nModel features:")
    for feature in MODEL_FEATURES:
        print(f"  - {feature}")


# ---------------------------------------------------------------------------
# Main test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print_dataset_summary()

    (
        preprocessor,
        X_train,
        y_train,
        X_validation,
        y_validation,
        X_test,
        y_test,
    ) = prepare_clinical_data()

    print("\nPreprocessing successful.")
    print("=" * 50)

    print("Training feature shape:", X_train.shape)
    print("Validation feature shape:", X_validation.shape)
    print("Test feature shape:", X_test.shape)

    print("\nTraining labels:", y_train.shape)
    print("Validation labels:", y_validation.shape)
    print("Test labels:", y_test.shape)

    print("\nTransformed feature names:")
    for name in X_train.columns:
        print(f"  - {name}")