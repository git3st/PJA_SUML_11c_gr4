from merge_files import merge_files
from data_preprocessing import preprocess_data
from machine_learning import machine_learning
from sklearn.ensemble import RandomForestClassifier

merge_files("data\\games_metadata_profile_2024_01", 16, "data\\full_dataset.csv")
train, test, validate = preprocess_data("data\\full_dataset.csv")
# model, x_test, y_test = machine_learning(train, test, "Result")
