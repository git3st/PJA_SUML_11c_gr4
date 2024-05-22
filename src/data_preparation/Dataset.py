import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


class Dataset:
    """A class representing a dataset for machine learning tasks.

    This class provides methods for reading, preprocessing, and splitting data into
    training, testing, and validation sets.

    Attributes:
        full_dataset (pd.DataFrame): The entire dataset.
        train_set (pd.DataFrame): The training set.
        validate_set (pd.DataFrame): The validation set.
        test_set (pd.DataFrame): The testing set.
        x_train (pd.DataFrame): The training set without the target variable.
        x_test (pd.DataFrame): The testing set without the target variable.
        y_train (pd.Series): The target variable for the training set.
        y_test (pd.Series): The target variable for the testing set.
        train_proportion (float): The proportion of data to use for training.
        test_proportion (float): The proportion of data to use for testing.
        validate_proportion (float): The proportion of data to use for validation.
        seed (int): The random seed for reproducibility.
    """

    def __init__(self, filename, train, test, validation, seed):
        """Initializes the Dataset object.

        Args:
            filename (str): Path to the CSV file containing the dataset.
            train (float): Proportion of data to use for training.
            test (float): Proportion of data to use for testing.
            validation (float): Proportion of data to use for validation.
            seed (int): Random seed for reproducibility.
        """
        self.full_dataset = pd.read_csv(filename)
        self.validate_set = pd.DataFrame
        self.x_train = pd.DataFrame
        self.x_test = pd.DataFrame
        self.y_train = pd.Series(dtype="object")
        self.y_test = pd.Series(dtype="object")
        self.train_proportion = train
        self.test_proportion = test
        self.validate_proportion = validation
        self.seed = seed

    def remove_columns(self, cols_to_remove):
        """Removes specified columns from the full dataset.

        Args:
            cols_to_remove (list): List of column names to remove.
        """
        self.full_dataset.drop(cols_to_remove, axis=1, inplace=True)

    def clean_missing_vals(self):
        """Removes rows with missing values from the full dataset."""
        self.full_dataset.dropna(axis=0, inplace=True)

    def fill_missing_number_vals(self, columns_and_conversion):
        """Fills missing values in numerical columns with the mean or median, and rounds/converts them to the specified type.

        Args:
            columns_and_conversion (dict): Dictionary of column names (keys) and the desired data type for filling (values), e.g., {"column_name": "int"}.
        """
        for column, conversion_type in columns_and_conversion.items():
            col_mean = self.full_dataset[column].mean()
            is_missing = self.full_dataset[column].isnull()
            self.full_dataset.loc[is_missing, column] = col_mean.round().astype(
                conversion_type
            )

    def fill_missing_string_vals(self, columns_and_values):
        """Fills missing string values with provided values per column.

        Args:
            columns_and_values (dict): A dictionary where keys are column names
                and values are the replacement strings.
        """
        for column, fill_string_value in columns_and_values.items():
            self.full_dataset[column] = self.full_dataset[column].fillna(
                fill_string_value
            )

    def clean_outliers(self):
        """Removes outliers from numerical columns in the full dataset using the interquartile range (IQR) method."""
        q1 = self.full_dataset.quantile(0.25)
        q3 = self.full_dataset.quantile(0.75)
        iqr = q3 - q1
        outliers = (self.full_dataset < (q1 - 1.5 * iqr)) | (
            self.full_dataset > (q3 + 1.5 * iqr)
        )
        self.full_dataset = self.full_dataset[~outliers.any(axis=1)]

    def rename_columns(self, cols_to_rename):
        """Renames specified columns in the full dataset.

        Args:
            cols_to_rename (dict): A dictionary mapping old column names (keys) to new column names (values).
        """
        for column_name, new_column_name in cols_to_rename.items():
            self.full_dataset.rename(
                columns={column_name: new_column_name}, inplace=True
            )

    def transform_text_values(self, trans_dict):
        """Replaces specific values in text columns according to a given mapping.

        Args:
            trans_dict (dict): A dictionary mapping column names (keys) to dictionaries (values) that specify the replacements for those columns (e.g., {"column1": {"old_value": "new_value"}}).
        """
        for column, mapping in trans_dict.items():
            if column in self.full_dataset.columns:
                self.full_dataset.loc[
                    self.full_dataset[column].isin(mapping.keys()), column
                ] = self.full_dataset[column].map(mapping)

    def normalize(self, cols_to_normalize):
        """Normalizes specified numerical columns to a range between 0 and 1.

        Args:
            cols_to_normalize (list): List of column names to normalize.
        """
        for column in self.full_dataset.columns:
            if column in cols_to_normalize:
                self.full_dataset[column] = (
                    self.full_dataset[column] / self.full_dataset[column].abs().max()
                )

    def get_numeric_features(self):
        """
        Identifies and returns the names of numeric columns in the dataset.

        Returns:
            list: A list of column names that are of numeric datatypes (int64 or float64).
        """
        numeric_features = self.full_dataset.select_dtypes(
            include=["int64", "float64"]
        ).columns
        return numeric_features

    def transform_columns(self, categorical_features):
        """
        Creates a ColumnTransformer for preprocessing numeric and categorical features.

        Args:
            categorical_features (list): List of column names representing categorical features.

        Returns:
            ColumnTransformer: A configured ColumnTransformer object for preprocessing data.
        """
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), self.get_numeric_features()),
                ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ]
        )
        return preprocessor

    def split_data(self, value):
        """
        Splits the dataset into training, validation, and testing sets.

        Args:
            value (str): The name of the target variable column to use for splitting.
        """
        # Debug: Check if 'Result' column is present before splitting
        if value not in self.full_dataset.columns:
            print(
                f"Error: '{value}' column is missing in the dataset before splitting."
            )
            print("Columns in dataset:", self.full_dataset.columns)

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
