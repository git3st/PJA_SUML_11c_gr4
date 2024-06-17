import os
import requests
import streamlit as st
import pickle
import pandas as pd
import argparse
import logging

# TO DO
# Test api call
# Test output
# Beautify


def create_error_logger() -> logging.Logger:
    """
    Creates a logger to record errors during pipeline execution.

    Returns:
        logging.Logger: The configured logger object.
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.ERROR)
    return logger


logger = create_error_logger()


def get_elo(user, api_key):
    """
    Retrieves the Lichess Elo rating for the given user.

    Args:
        user (str): The Lichess username.
        api_key (str): Your Lichess API key.

    Returns:
        int: The user's Elo rating.
        None: If there was an error fetching the Elo.

    Raises:
        requests.exceptions.RequestException: For any HTTP request error.
        KeyError: If the Elo rating data is not found in the response.
        ValueError: If the Elo rating cannot be converted to an integer.
        Exception: For any other unexpected errors.
    """

    rating = 1500
    try:
        url = f"https://lichess.org/api/user/{user}"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        response = requests.get(url=url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            rating = int(data["profile"]["fideRating"])
        else:
            print("Error: {response.status_code}")
    except requests.exceptions.RequestException as e:
        logger.error("Error fetching Lichess data for user '{user}': %s", e)
    except KeyError as e:
        logger.error("Elo rating data not found for user '{user}': %s", e)
    except ValueError as e:
        logger.error("Invalid Elo rating format for user '{user}': %s", e)
    except Exception as e:  # Catch-all for unexpected errors
        logger.error("Unexpected error fetching Elo for '{user}': %s", e)
    return rating


parser = argparse.ArgumentParser()
parser.add_argument(
    "--api",
    help="Lichess API key",
)

args = parser.parse_args()
api = args.api


dirname = os.path.dirname(os.path.dirname(__file__))
model_path = os.path.join(dirname, "models", "models", "KNeighborsDist", "model.pkl")

with open(model_path, "rb") as model_file:
    model = pickle.load(model_file)
    metadata = model.feature_metadata
    print(metadata)

st.title("Chess Match Prediction")

st.write(
    """
## Provide the player names to see which one would win!
"""
)


white = st.text_input("White:")
black = st.text_input("Black:")


if st.button("Predict"):
    white_elo = get_elo(white, api)
    black_elo = get_elo(black, api)
    white_diff = white_elo - black_elo
    black_diff = black_elo - white_elo
    input_data = pd.DataFrame(
        {
            "White": [white],
            "Black": [black],
            "WhiteElo": [white_elo],
            "BlackElo": [black_elo],
        }
    )

    prediction = model.predict(input_data)

    st.write(f"Prediction: {prediction[0]} would win!")
    st.balloons()
