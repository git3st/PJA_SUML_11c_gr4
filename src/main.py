from src.data_preparation.merge_files import merge_files
from src.data_preparation.data_preprocessing import preprocess_data

merge_files("data\\games_metadata_profile_2024_01", 16, "data\\full_dataset.csv")
train, test, validate = preprocess_data("data\\full_dataset.csv")
# model, x_test, y_test = machine_learning(train, test, "Result")
