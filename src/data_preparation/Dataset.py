from typing import Any
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


class Dataset:
    def __init__(self, filename, train, test, validation, seed):
        self.full_dataset = pd.read_csv(filename)
        self.train_set = pd.DataFrame
        self.validate_set = pd.DataFrame
        self.test_set = pd.DataFrame
        self.train_proportion = train
        self.test_proportion = test
        self.validate_proportion = validation
        self.seed = seed

    def remove_columns(self, cols_to_remove):
        self.full_dataset.drop(cols_to_remove, axis=1, inplace=True)

    def clean_missing_vals(self):
        self.full_dataset.dropna(axis=0, inplace=True)

    def fill_missing_number_vals(self, columns_and_conversion):
        for column, conversion_type in columns_and_conversion.items():
            col_mean = self.full_dataset[column].mean()
            is_missing = self.full_dataset[column].isnull()
            self.full_dataset.loc[is_missing, column] = col_mean.round().astype(
                conversion_type
            )
            # self.full_dataset[column] = (
            #     self.full_dataset[column].fillna(self.full_dataset[column]).mean()
            # )

    def fill_missing_string_vals(self, columns_and_values):
        """Fills missing string values with provided values per column.

        Args:
            columns_and_values: A dictionary where keys are column names
                            and values are the replacement strings.
        """
        for column, fill_string_value in columns_and_values.items():
            self.full_dataset[column] = self.full_dataset[column].fillna(
                fill_string_value
            )

    def clean_outliers(self):
        q1 = self.full_dataset.quantile(0.25)
        q3 = self.full_dataset.quantile(0.75)
        iqr = q3 - q1
        outliers = (self.full_dataset < (q1 - 1.5 * iqr)) | (
            self.full_dataset > (q3 + 1.5 * iqr)
        )
        self.full_dataset = self.full_dataset[~outliers.any(axis=1)]

    def transform_text_values(self, trans_dict):
        for column, mapping in trans_dict.items():
            if column in self.full_dataset.columns:
                self.full_dataset[column].replace(mapping, inplace=True)

    def normalize(self, cols_to_normalize):
        for column in self.full_dataset.columns:
            if column in cols_to_normalize:
                self.full_dataset[column] = (
                    self.full_dataset[column] / self.full_dataset[column].abs().max()
                )

    def split_data(self):
        self.train_set, self.test_set = train_test_split(
            self.full_dataset,
            test_size=1 - self.train_proportion,
            random_state=self.seed,
        )
        validate_size = self.validate_proportion / (
            self.validate_proportion + self.test_proportion
        )
        self.test_set, self.validate_set = train_test_split(
            self.test_set, test_size=validate_size, random_state=self.seed
        )
