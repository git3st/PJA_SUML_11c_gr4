from data_preprocessing import preprocess_data
from merge_files import merge_files

merge_files("data\\games_metadata_profile_2024_01", 16, "data\\full_dataset.csv")
train, test, validate = preprocess_data("data\\full_dataset.csv")
