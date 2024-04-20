from src.data_preparation.Dataset import Dataset

"""
if you need to transform something from text to numbers
format:
cols_to_transform = {
    'column1': {'oldvalue1': 'newvalue1', 'oldvalue2': 'newvalue2'},
    'column2': {'oldvalue3': 'newvalue3'}
}
"""


def preprocess_data(
    filename, cols_to_normalize=None, cols_to_transform=None, cols_to_remove=None, train=0.8, test=0.10,
        validation=0.10, seed=50
):
    dataset = Dataset(filename, train=train, test=test, validation=validation, seed=seed)
    if cols_to_remove is not None:
        dataset.remove_columns(cols_to_remove)
    if cols_to_transform is not None:
        dataset.transform_text_values(cols_to_transform)
    if cols_to_normalize is not None:
        dataset.normalize(cols_to_normalize)
    dataset.split_data()

    return dataset.train_set, dataset.test_set, dataset.validate_set
