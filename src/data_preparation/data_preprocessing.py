import logging
import os
from typing import Dict, List, Tuple
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pandas as pd
from data_preparation.Dataset import Dataset

current_path = os.path.realpath(__file__)
parent_dir = os.path.dirname(os.path.dirname(current_path))


def create_error_logger() -> logging.Logger:
    """
    Creates a logger to record errors during pipeline execution.

    Returns:
        logging.Logger: The configured logger object.
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.ERROR)
    return logger


logger = create_error_logger()


def calculate_average_time(
    dataset: Dataset,
    time_columns: List[str] = ["White_playTime_total", "Black_playTime_total"],
    count_columns: List[str] = ["White_count_all", "Black_count_all"],
    output_column_names: List[str] = [
        "Average_White_Play_Time",
        "Average_Black_Play_Time",
    ],
):
    """Calculates average play time for specified player groups.

    Args:
        dataset (Dataset): The dataset object containing the relevant columns.
        time_columns (List[str]): List of columns with total play time for each player group.
        count_columns (List[str]): List of columns with the number of games played for each group.
        output_column_names (List[str]): List of names for the new columns containing average play times.
    """
    try:
        for time_col, count_col, output_col in zip(
            time_columns, count_columns, output_column_names
        ):
            non_zero_rows = dataset.full_dataset[count_col] != 0
            dataset.full_dataset.loc[non_zero_rows, output_col] = (
                dataset.full_dataset.loc[non_zero_rows, time_col]
                / dataset.full_dataset.loc[non_zero_rows, count_col]
            ).fillna(0)
    except KeyError as e:
        logger.error("Missing column(s): %s", e)
    except ZeroDivisionError as e:
        logger.error(f"Division by zero error in column '{count_col}': %s", e)
    except TypeError as e:
        logger.error("Type error in calculation (check data types): %s", e)
    except Exception as e:
        logger.error("Unexpected error: %s", e)


def round_columns_to_int(dataset: Dataset, columns: List[str]):
    """Rounds values in the specified columns to the nearest integer.

    Args:
        columns (List[str]): List of column names to round.
    """

    try:
        for column in columns:
            dataset.full_dataset[column] = (
                dataset.full_dataset[column].round().astype(int)
            )
    except (TypeError, ValueError) as e:
        logger.error("Error rounding values in column '%s': %s", column, e)
        raise


def transform_data(
    filename: str,
    cols_to_remove: List[str] = [
        "WhiteRatingDiff",
        "BlackRatingDiff",
        "Black_count_all",
        "Black_createdAt",
        "Black_is_deleted",
        "Black_playTime_total",
        "Black_profile_flag",
        "Black_title",
        "Black_tosViolation",
        "Date",
        "ECO",
        "GameID",
        "Moves",
        "Opening",
        "Round",
        "Site",
        "Termination",
        "Time",
        "TimeControl",
        "TotalMoves",
        "White_count_all",
        "White_createdAt",
        "White_is_deleted",
        "White_playTime_total",
        "White_profile_flag",
        "White_title",
        "White_tosViolation",
    ],
    cols_to_fill_numbers: Dict[str, str] = {
        "WhiteElo": "int",
        "WhiteRatingDiff": "int",
        "White_playTime_total": "float",
        "White_count_all": "float",
        "BlackElo": "int",
        "BlackRatingDiff": "int",
        "Black_playTime_total": "float",
        "Black_count_all": "float",
    },
    fill_string_values: Dict[str, str] = {
        "White_profile_flag": "Unknown",
        "White_title": "None",
        "Black_profile_flag": "Unknown",
        "Black_title": "None",
        "Opening": "Unknown",
    },
    clean_outliers: bool = False,
    cols_to_transform: Dict[str, Dict[str, str]] = {
        "Result": {"1-0": "White", "0-1": "Black", "1/2-1/2": "Draw"},
        "Event": {"tournament*": "tournament", "swiss*": "swiss"},
    },
    clean_missing_vals: bool = True,
    cols_to_normalize: List[str] = None,
    categorical_features: List[str] = [
        "Event",
        "Day",  # Remove "Day" if it doesn't exist
        "Time_TimeOfDay",
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
    train: float = 0.8,
    test: float = 0.10,
    validation: float = 0.10,
    seed: int = 50,
    n_estimators_pipeline: int = 100,
    random_state_pipeline: int = 42,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, Pipeline]:
    """Preprocesses a dataset for machine learning.

    Loads, cleans, transforms, and splits a dataset from a CSV file into training, testing, and validation sets.

    Args:
        filename (str): Path to the CSV file.
        cols_to_remove (List[str], optional): Columns to remove. Defaults to ["GameID", "Round", "Site", "White", "Black", "White_tosViolation", "Black_tosViolation"].
        cols_to_fill_numbers (Dict[str, str], optional): Numeric columns to fill with their mean, and their data types. Defaults to a predefined dictionary.
        fill_string_values (Dict[str, str], optional): String columns and values to fill missing values with. Defaults to a predefined dictionary.
        clean_outliers (bool, optional): Whether to remove outliers. Defaults to False.
        cols_to_transform (Dict[str, Dict[str, str]], optional): Dictionary mapping columns and value replacements. Defaults to {"Result": {"1-0": "White", "0-1": "Black", "1/2-1/2": "Draw"}}.
        clean_missing_vals (bool, optional): Whether to drop rows with missing values. Defaults to True.
        cols_to_normalize (List[str], optional): List of columns to normalize to [0, 1]. Defaults to None.
        categorical_features (List[str], optional): Categorical columns for one-hot encoding. Defaults to a predefined list.
        train (float, optional): Proportion of data for training (0-1). Defaults to 0.8.
        test (float, optional): Proportion of data for testing (0-1). Defaults to 0.1.
        validation (float, optional): Proportion of data for validation (0-1). Defaults to 0.1.
        seed (int, optional): Random seed for reproducibility. Defaults to 50.
        n_estimators_pipeline (int, optional): Number of estimators for Random Forest. Defaults to 100.
        random_state_pipeline (int, optional): Random state for Random Forest. Defaults to 42.

    Returns:
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, Pipeline]:
            - x_train (pd.DataFrame): Features for the training set.
            - y_train (pd.Series): Target variable for the training set.
            - x_test (pd.DataFrame): Features for the testing set.
            - y_test (pd.Series): Target variable for the testing set.
            - validate_set (pd.DataFrame): Features for the validation set.
            - pipeline (Pipeline): The fitted preprocessing and modeling pipeline.

    Raises:
        KeyError: If the "Result" column is not found after preprocessing.
        Exception: If there is an unexpected error during preprocessing.
    """
    dataset = Dataset(
        filename, train=train, test=test, validation=validation, seed=seed
    )

    pipeline = None

    try:
        if cols_to_fill_numbers is not None:
            dataset.fill_missing_number_vals(cols_to_fill_numbers)
        if fill_string_values is not None:
            dataset.fill_missing_string_vals(fill_string_values)
        if cols_to_transform is not None:
            dataset.transform_text_values(cols_to_transform)

        # Calculate custom features
        # calculate_average_time(dataset)

        # Round numeric values
        # if cols_to_round is not None:
        #    round_columns_to_int(dataset, cols_to_round)

        # Debug: Print columns after transformation
        print("Columns after transformation:", dataset.full_dataset.columns)

        if clean_outliers is True:
            dataset.clean_outliers()
        if clean_missing_vals is True:
            dataset.clean_missing_vals()
        if cols_to_normalize is not None:
            dataset.normalize(cols_to_normalize)

        # Debug: Check if 'Result' column is present
        if "Result" not in dataset.full_dataset.columns:
            print(
                "Error: 'Result' column is missing in the dataset after preprocessing."
            )
            print("Columns in dataset:", dataset.full_dataset.columns)
            logger.error(
                "Error: 'Result' column is missing in the dataset after preprocessing."
            )
            logger.error("Columns in dataset: %s", dataset.full_dataset.columns)
            raise KeyError("'Result' column not found.")

        if cols_to_remove is not None:
            dataset.remove_columns(cols_to_remove)

        # Remove 'Day' if it doesn't exist in the final dataframe
        available_categorical_features = [col for col in categorical_features if col in dataset.full_dataset.columns]

        # Remove 'Result' from features for preprocessing
        features = dataset.full_dataset.drop(columns=["Result"])
        print(features)
        numeric_features = features.select_dtypes(include=["int64", "float64"]).columns
        print(numeric_features)
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), numeric_features),
                ("cat", OneHotEncoder(handle_unknown="ignore"), available_categorical_features),
            ]
        )
        print(pipeline)
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

        project_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        )
        output_path = os.path.join(project_root, "data", "02_transformed_data")
        os.makedirs(output_path, exist_ok=True)
        dataset.full_dataset.to_csv(
            os.path.join(output_path, "transformed_data.csv"), index=False
        )
        dataset.split_data("Result")
    except Exception as e:
        logger.error("Unexpected error in transformed_data function: %s", e)

    return (
        dataset.x_train,
        dataset.y_train,
        dataset.x_test,
        dataset.y_test,
        dataset.validate_set,
        pipeline,
    )

