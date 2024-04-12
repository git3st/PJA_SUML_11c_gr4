from data_prep import Dataset


# prepare dataset
dataset = Dataset('PLACEHOLDER')
cols_to_normalize = []
# if you need to transform something from text to numbers
# format:
# cols_to_transform = {
#     'column1': {'oldvalue1': 'newvalue1', 'oldvalue2': 'newvalue2'},
#     'column2': {'oldvalue3': 'newvalue3'}
# }
cols_to_transform = {}
cols_to_remove = []
dataset.remove_columns(cols_to_remove)
dataset.transform_text_values(cols_to_transform)
dataset.normalize(cols_to_normalize)
dataset.split_data()

# Use following for train/test/validate
# dataset.full_dataset
# dataset.train_set
# dataset.test_set
# dataset.validate_set

