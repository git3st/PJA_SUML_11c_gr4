from typing import Dict, List
import logging
import re
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


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


class Dataset:
    """A class representing a dataset for machine learning tasks.

    This class provides methods for reading, preprocessing, and splitting data.
    It handles errors during these operations, logging them for troubleshooting.

    Attributes:
        full_dataset (pd.DataFrame): The entire dataset.
        train_set (pd.DataFrame): The training set (features).
        validate_set (pd.DataFrame): The validation set (features).
        test_set (pd.DataFrame): The testing set (features).
        x_train (pd.DataFrame): The training set without the target variable.
        x_test (pd.DataFrame): The testing set without the target variable.
        y_train (pd.Series): The target variable for the training set.
        y_test (pd.Series): The target variable for the testing set.
        train_proportion (float): The proportion of data to use for training.
        test_proportion (float): The proportion of data to use for testing.
        validate_proportion (float): The proportion of data to use for validation.
        seed (int): The random seed for reproducibility.
    """

    def __init__(
        self,
        filename: str,
        train: float,
        test: float,
        validation: float,
        seed: int,
    ):
        """Initializes the Dataset object.

        Args:
            filename (str): Path to the CSV file containing the dataset.
            train (float): Proportion of data to use for training.
            test (float): Proportion of data to use for testing.
            validation (float): Proportion of data to use for validation.
            seed (int): Random seed for reproducibility.
        """
        self.full_dataset: pd.DataFrame = pd.read_csv(filename)
        self.validate_set: pd.DataFrame = pd.DataFrame
        self.x_train: pd.DataFrame = pd.DataFrame
        self.x_test: pd.DataFrame = pd.DataFrame
        self.y_train: pd.Series = pd.Series(dtype="object")
        self.y_test: pd.Series = pd.Series(dtype="object")
        self.train_proportion: float = train
        self.test_proportion: float = test
        self.validate_proportion: float = validation
        self.seed: int = seed

    def remove_columns(self, cols_to_remove: List[str]):
        """Removes specified columns from the full dataset.

        Args:
            cols_to_remove: A list of column names to remove.

        Raises:
            KeyError: If one or more of the specified columns do not exist.
        """
        try:
            self.full_dataset.drop(cols_to_remove, axis=1, inplace=True)
        except KeyError as ke:
            logger.error("KeyError while removing columns: %s", ke)
            raise

    def clean_missing_vals(self):
        """Removes rows with missing values from the full dataset."""
        try:
            self.full_dataset.dropna(axis=0, inplace=True)
        except Exception as e:
            logger.error("Error while cleaning missing values: %s", e)
            raise

    def fill_missing_number_vals(self, columns_and_conversion: Dict[str, str]):
        """Fills missing values in specified numerical columns with rounded means.

        Args:
            columns_and_conversion: A dictionary mapping column names (keys) to the
                desired data type for filling (values), e.g., {"column_name": "int"}.

        Raises:
            TypeError, ValueError: If there's a type mismatch during the conversion or
                the column does not contain numeric values.
        """
        for column, conversion_type in columns_and_conversion.items():
            try:
                col_mean = self.full_dataset[column].mean()
                is_missing = self.full_dataset[column].isnull()
                self.full_dataset.loc[is_missing, column] = col_mean.round().astype(
                    conversion_type
                )
            except (
                TypeError,
                ValueError,
            ) as e:
                logger.error(
                    "Error filling missing values in column '%s': %s", column, e
                )
                raise

    def fill_missing_string_vals(self, columns_and_values: Dict[str, str]):
        """Fills missing values (NaN) in specified string columns with the given replacement values.

        Args:
            columns_and_values: A dictionary mapping column names (keys) to the replacement string values (values).

        Raises:
            KeyError: If a specified column name is not found in the dataset.
        """
        try:
            for column, fill_string_value in columns_and_values.items():
                self.full_dataset[column] = self.full_dataset[column].fillna(
                    fill_string_value
                )
        except KeyError as ke:
            logger.error("KeyError while filling string values: %s", ke)
            raise

    def clean_outliers(self):
        """Removes outliers from numerical columns in the dataset using the Interquartile Range (IQR) method.

        Outliers are identified as values that fall below Q1 - 1.5 * IQR or above Q3 + 1.5 * IQR, where Q1 and Q3 are the 25th and 75th percentiles respectively.
        """
        try:
            q1 = self.full_dataset.quantile(0.25)
            q3 = self.full_dataset.quantile(0.75)
            iqr = q3 - q1
            outliers = (self.full_dataset < (q1 - 1.5 * iqr)) | (
                self.full_dataset > (q3 + 1.5 * iqr)
            )
            self.full_dataset = self.full_dataset[~outliers.any(axis=1)]
        except Exception as e:
            logger.error("Error while cleaning outliers: %s", e)
            raise

    def rename_columns(self, cols_to_rename: Dict[str, str]):
        """Renames specified columns in the dataset.

        Args:
            cols_to_rename: A dictionary mapping old column names (keys) to new column names (values).

        Raises:
            KeyError: If an old column name specified for renaming is not found.
        """
        try:
            for column_name, new_column_name in cols_to_rename.items():
                self.full_dataset.rename(
                    columns={column_name: new_column_name}, inplace=True
                )
        except Exception as e:
            logger.error("Error while renaming columns: %s", e)
            raise

    def transform_text_values(self, trans_dict: Dict[str, Dict[str, str]]):
        """Replaces specific values in text columns of the dataset according to a given mapping.

        Args:
            trans_dict: A dictionary where keys are column names and values are dictionaries specifying the replacements (e.g., {"column1": {"old_value": "new_value"}}).
        """
        try:
            for column, mapping in trans_dict.items():
                if column in self.full_dataset.columns:
                    for key in mapping.keys():
                        if "*" in key:  # Identify patterns with "*"
                            pattern = re.compile(
                                key.replace("*", r".*")
                            )  # Create regex
                            self.full_dataset[column] = (
                                self.full_dataset[column]
                                .astype(str)
                                .str.replace(pattern, mapping[key], regex=True)
                            )
                        else:
                            self.full_dataset.loc[
                                self.full_dataset[column] == key, column
                            ] = mapping[key]
        except Exception as e:
            logger.error("Error transforming text values: %s", e)
            raise

    def transform_date_to_day(self, cols_to_transform: List[str]) -> None:
        """Converts date columns to the day of the week.

        Args:
            cols_to_transform (list): List of column names to convert to day of week.

        Raises:
            ValueError: If a date in the specified format cannot be parsed.
        """
        for column in cols_to_transform:
            try:
                self.full_dataset[column] = pd.to_datetime(
                    self.full_dataset[column], format="%Y.%m.%d"
                ).dt.day_name()
            except ValueError:  # Handle invalid date formats
                logger.error(
                    "ValueError: Unable to convert column '%s' to datetime using format '%%Y.%%m.%%d'.",
                    column,
                )
                raise

    def transform_time_to_category(self, cols_to_transform: List[str]) -> None:
        """Converts time columns to categorical time-of-day features.

        Args:
            cols_to_transform (list): List of column names to transform.

        Raises:
            ValueError: If a time in the specified format cannot be parsed.
        """
        for column in cols_to_transform:
            try:
                self.full_dataset[column] = pd.to_datetime(
                    self.full_dataset[column], format="%H:%M:%S"
                )
                hour = self.full_dataset[column].dt.hour
                self.full_dataset[column + "_TimeOfDay"] = pd.cut(
                    hour,
                    bins=[-1, 6, 12, 18, 24],
                    labels=["Night", "Morning", "Afternoon", "Evening"],
                    right=False,
                )
                self.full_dataset.drop(column, axis=1, inplace=True)
            except ValueError:
                logger.error(
                    "ValueError: Unable to convert column '%s' to datetime using format '%%H:%%M:%%S'.",
                    column,
                )
                raise

    def normalize(self, cols_to_normalize: List[str]):
        """Normalizes specified numeric columns to a range between 0 and 1.

        This function iterates over the specified columns in the dataset,
        and if they are numeric, it scales each column's values to the range [0, 1]
        using min-max scaling.

        Args:
            cols_to_normalize: A list of column names to normalize.

        Raises:
            TypeError: If the columns to normalize include non-numeric data types.
            ValueError: If there are issues with values in the numeric columns.
            ZeroDivisionError: If a column contains only zeros, preventing normalization.
        """
        try:
            for column in self.full_dataset.columns:
                if column in cols_to_normalize:
                    self.full_dataset[column] = (
                        self.full_dataset[column]
                        / self.full_dataset[column].abs().max()
                    )
        except (
            TypeError,
            ValueError,
            ZeroDivisionError,
        ) as e:
            logger.error("Error normalizing columns: %s", e)
            raise

    def get_numeric_features(self) -> pd.Index:
        """
        Identifies and returns the names of numeric columns in the dataset.

        Returns:
            pd.Index: An index object containing the names of numeric columns.
        """
        numeric_features = self.full_dataset.select_dtypes(
            include=["int64", "float64"]
        ).columns
        return numeric_features

    def transform_columns(self, categorical_features: List[str]) -> ColumnTransformer:
        """
        Creates a ColumnTransformer for preprocessing numeric and categorical features.

        Args:
            categorical_features (list): List of column names representing categorical features.

        Returns:
            ColumnTransformer: A configured ColumnTransformer object for preprocessing data.
        """
        try:
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", StandardScaler(), self.get_numeric_features()),
                    (
                        "cat",
                        OneHotEncoder(handle_unknown="ignore"),
                        categorical_features,
                    ),
                ]
            )
            return preprocessor
        except Exception as e:
            logger.error("Error during column transformation: %s", e)
            raise

    def split_data(self, value: str):
        """Splits the dataset into training, validation, and testing sets.

        This function splits the dataset into three sets: training, validation,
        and testing, ensuring the specified target variable ('value') is present
        in the dataset. The split is done according to the proportions defined
        in the object's attributes.

        Args:
            value (str): The name of the target variable column to use for splitting.

        Returns:
            Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
                A tuple containing:
                    - x_train (pd.DataFrame): Features for the training set.
                    - y_train (pd.Series): Target variable for the training set.
                    - x_test (pd.DataFrame): Features for the testing set.
                    - y_test (pd.Series): Target variable for the testing set.
                    - validate_set (pd.DataFrame): Features for the validation set.
                    - validate_y (pd.Series): Target variable for the validation set.
        Raises:
            KeyError: If the specified target variable ('value') is not found in the dataset.
        """
        # Debug: Check if predicted column is present before splitting
        if value not in self.full_dataset.columns:
            print(
                f"Error: '{value}' column is missing in the dataset before splitting."
            )
            print("Columns in dataset:", self.full_dataset.columns)
            logger.error(
                "Error: %s column is missing in the dataset before splitting.", value
            )
            logger.error("Columns in dataset: %s", self.full_dataset.columns)
            raise KeyError(f"'{value}' column not found.")

        x = self.full_dataset.drop(columns=[value])
        y = self.full_dataset[value]

        # Split into train and test sets
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            x,
            y,
            test_size=1 - self.train_proportion,
            random_state=self.seed,
        )

        # Split test set into test and validation sets
        validate_size = self.validate_proportion / (
            self.validate_proportion + self.test_proportion
        )
        self.x_test, self.validate_set, self.y_test, self.validate_y = train_test_split(
            self.x_test, self.y_test, test_size=validate_size, random_state=self.seed
        )

        # Debug: Check shapes of the split datasets
        print(
            f"x_train shape: {self.x_train.shape}, y_train shape: {self.y_train.shape}"
        )
        print(f"x_test shape: {self.x_test.shape}, y_test shape: {self.y_test.shape}")
        print(
            f"validate_set shape: {self.validate_set.shape}, validate_y shape: {self.validate_y.shape}"
        )
