import pickle
from sklearn.base import is_classifier
from data_preparation.data_preprocessing import preprocess_data


# Release model
def release_model():
    with open("data\\04_models\\chess_game_result_classifier.pkl", "wb") as f:
        pickle.dump((preprocess_data, is_classifier), f)
