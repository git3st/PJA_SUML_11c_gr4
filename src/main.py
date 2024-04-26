from data_preparation.merge_files import merge_files
from data_preparation.data_preprocessing import preprocess_data

merge_files("data\\games_metadata_profile_2024_01", 16, "data\\full_dataset.csv")
train, test, validate = preprocess_data(filename="data\\full_dataset.csv")
train.to_csv("data\\data_train.csv")
# model = machine_learning(train, validate)
