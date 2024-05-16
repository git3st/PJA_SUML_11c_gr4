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
    random_state_pipeline=42
):
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
    if 'Result' not in dataset.full_dataset.columns:
        print("Error: 'Result' column is missing in the dataset after preprocessing.")
        print("Columns in dataset:", dataset.full_dataset.columns)

    # Remove 'Result' from features for preprocessing
    features = dataset.full_dataset.drop(columns=['Result'])
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
            ("classifier", RandomForestClassifier(n_estimators=n_estimators_pipeline, random_state=random_state_pipeline)),
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
