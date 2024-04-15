from Dataset import Dataset

'''
if you need to transform something from text to numbers
format:
cols_to_transform = {
    'column1': {'oldvalue1': 'newvalue1', 'oldvalue2': 'newvalue2'},
    'column2': {'oldvalue3': 'newvalue3'}
}
'''


def preprocess_data(filename, cols_to_normalize=None, cols_to_transform=None, cols_to_remove=None):
    dataset = Dataset(filename)
    dataset.remove_columns(cols_to_remove)
    dataset.transform_text_values(cols_to_transform)
    dataset.normalize(cols_to_normalize)
    dataset.split_data()

    return dataset.train_set, dataset.test_set, dataset.validate_set

