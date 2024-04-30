from data_preparation.merge_files import merge_files
from data_preparation.data_preprocessing import preprocess_data
from data_science.machine_learning import machine_learning
from data_science.evaluate_model import evaluate_model
from data_science.release_model import release_model

merge_files(
    "data\\01_raw_data\\games_metadata_profile_2024_01",
    16,
    "data\\01_raw_data\\full_dataset.csv",
)
x_train, y_train, x_test, y_test, validate_set, pipeline = preprocess_data(
    filename="data\\01_raw_data\\full_dataset.csv"
)
model = machine_learning(x_train, y_train, validate_set, pipeline)
# evaluate_model(x_test, y_test, model)
# release_model()
