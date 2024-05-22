import argparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from data_preparation.Dataset import Dataset


def preprocess_data(
    filename,
    cols_to_remove=[
        "GameID",
        "Round",
        "Site",
        "White",
        "Black",
        "White_tosViolation",
        "Black_tosViolation",
    ],
    cols_to_fill_numbers={
        "WhiteElo": "int",
        "WhiteRatingDiff": "int",
        "White_playTime_total": "float",
        "White_count_all": "float",
        "BlackElo": "int",
        "BlackRatingDiff": "int",
        "Black_playTime_total": "float",
        "Black_count_all": "float",
    },
    fill_string_values={
        "White_profile_flag": "Unknown",
        "White_title": "None",
        "Black_profile_flag": "Unknown",
        "Black_title": "None",
        "Opening": "Unknown",
    },
    clean_outliers=False,
    cols_to_rename=None,
    cols_to_transform={
        "Result": {"1-0": "White", "0-1": "Black", "1/2-1/2": "Draw"},
    },
    clean_missing_vals=True,
    cols_to_normalize=None,
    categorical_features=[
        "Event",
        "White_is_deleted",
        "White_profile_flag",
        "White_title",
        "Black_is_deleted",
        "Black_profile_flag",
        "Black_title",
        "ECO",
        "Opening",
        "TimeControl",
        "Termination",
    ],
    train=0.8,
    test=0.10,
    validation=0.10,
    seed=50,
    n_estimators_pipeline=100,
    random_state_pipeline=42,
):
    """Preprocesses a dataset for machine learning.

    Loads a dataset from a CSV file, performs various preprocessing steps, and
    splits it into training, testing, and validation sets.

    Args:
        filename (str): Path to the CSV file.
        cols_to_remove (list): List of column names to remove.
        cols_to_fill_numbers (dict): Dictionary mapping column names to data types for numeric missing value imputation.
        fill_string_values (dict): Dictionary mapping column names to strings for missing string value replacement.
        clean_outliers (bool): Whether to clean outliers (using IQR method).
        cols_to_rename (dict): Dictionary mapping old column names to new column names for renaming.
        cols_to_transform (dict): Dictionary mapping column names to dictionaries specifying value replacements.
        clean_missing_vals (bool): Whether to remove rows with missing values.
        cols_to_normalize (list): List of column names to normalize.
        categorical_features (list): List of column names for categorical features (for one-hot encoding).
        train (float): Proportion of data to use for training (default=0.8).
        test (float): Proportion of data to use for testing (default=0.1).
        validation (float): Proportion of data to use for validation (default=0.1).
        seed (int): Random seed for reproducibility (default=50).
        n_estimators_pipeline (int): Number of trees in the Random Forest classifier (default=100).
        random_state_pipeline (int): Random state for the Random Forest classifier (default=42).

    Returns:
        tuple: A tuple containing the following preprocessed data:

            * x_train (pd.DataFrame): Features for the training set.
            * y_train (pd.Series): Target variable for the training set.
            * x_test (pd.DataFrame): Features for the testing set.
            * y_test (pd.Series): Target variable for the testing set.
            * validate_set (pd.DataFrame): Features for the validation set.
            * pipeline (Pipeline): The fitted preprocessing pipeline (including StandardScaler and OneHotEncoder).
    """
    dataset = Dataset(
        filename, train=train, test=test, validation=validation, seed=seed
    )
    if cols_to_remove is not None:
        dataset.remove_columns(cols_to_remove)
    if cols_to_fill_numbers is not None:
        dataset.fill_missing_number_vals(cols_to_fill_numbers)
    if fill_string_values is not None:
        dataset.fill_missing_string_vals(fill_string_values)
    if cols_to_transform is not None:
        dataset.transform_text_values(cols_to_transform)

    # Calculate custom features
    dataset["Average_White_Play_Time"] = (
        dataset["White_playTime_total"] / dataset["White_count_all"]
    )
    dataset["Average_Black_Play_Time"] = (
        dataset["Black_playTime_total"] / dataset["Black_count_all"]
    )

    # Debug: Print columns after transformation
    print("Columns after transformation:", dataset.full_dataset.columns)

    if clean_outliers is True:
        dataset.clean_outliers()
    if cols_to_rename is not None:
        dataset.rename_columns(cols_to_rename)
    if clean_missing_vals is True:
        dataset.clean_missing_vals()
    if cols_to_normalize is not None:
        dataset.normalize(cols_to_normalize)

    # Debug: Check if 'Result' column is present
    if "Result" not in dataset.full_dataset.columns:
        print("Error: 'Result' column is missing in the dataset after preprocessing.")
        print("Columns in dataset:", dataset.full_dataset.columns)

    # Remove 'Result' from features for preprocessing
    features = dataset.full_dataset.drop(columns=["Result"])
    numeric_features = features.select_dtypes(include=["int64", "float64"]).columns
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=n_estimators_pipeline,
                    random_state=random_state_pipeline,
                ),
            ),
        ]
    )
    dataset.full_dataset.to_csv(
        "data\\02_processed_data\\processed_data.csv", index=False
    )
    dataset.split_data("Result")

    return (
        dataset.x_train,
        dataset.y_train,
        dataset.x_test,
        dataset.y_test,
        dataset.validate_set,
        pipeline,
    )
