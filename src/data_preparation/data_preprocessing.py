from data_preparation.Dataset import Dataset

"""
if you need to transform something from text to numbers
format:
cols_to_transform = {
    'column1': {'oldvalue1': 'newvalue1', 'oldvalue2': 'newvalue2'},
    'column2': {'oldvalue3': 'newvalue3'}
}
"""


def preprocess_data(
    filename,
    cols_to_remove=["GameID", "Round", "Site", "White", "Black"],
    clean_missing_vals=False,
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
    cols_to_transform=None,
    cols_to_normalize=None,
    train=0.8,
    test=0.10,
    validation=0.10,
    seed=50,
):
    dataset = Dataset(
        filename, train=train, test=test, validation=validation, seed=seed
    )
    if cols_to_remove is not None:
        dataset.remove_columns(cols_to_remove)
    if clean_missing_vals is True:
        dataset.clean_missing_vals()
    if cols_to_fill_numbers is not None:
        dataset.fill_missing_number_vals(cols_to_fill_numbers)
    if fill_string_values is not None:
        dataset.fill_missing_string_vals(fill_string_values)
    if clean_outliers is True:
        dataset.clean_outliers()
    if cols_to_transform is not None:
        dataset.transform_text_values(cols_to_transform)
    if cols_to_normalize is not None:
        dataset.normalize(cols_to_normalize)

    # Split Data
    dataset.split_data()
    return dataset.train_set, dataset.test_set, dataset.validate_set
