import json
import os
import time

import requests
import streamlit as st
import pickle
import pandas as pd
import argparse
import logging


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


def get_elo(user, game_mode, api_key):
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

    try:
        url = f"https://lichess.org/api/user/{user}"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        response = requests.get(url=url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            rating = int(data["perfs"][game_mode]["rating"])
        else:
            st.error(
                "Error - we could not communicate with the Lichess backend server (not our fault!)"
            )

    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching Lichess data for user '{user}': {e}")
        st.error(
            "Error - we could not communicate with the Lichess backend server (not our fault!)"
        )
    except KeyError as e:
        logger.error(f"Elo rating data not found for user '{user}': {e}")
        st.error(
            "Error - we could not communicate with the Lichess backend server (not our fault!)"
        )
    except ValueError as e:
        logger.error(f"Invalid Elo rating format for user '{user}': {e}")
        st.error(
            "Error - we could not communicate with the Lichess backend server (not our fault!)"
        )
    except Exception as e:  # Catch-all for unexpected errors
        logger.error(f"Unexpected error fetching Elo for '{user}': {e}")
        st.error(
            "Error - we could not communicate with the Lichess backend server (not our fault!)"
        )
    return rating


def fake_loader(final_result):
    msg = st.toast("Fetching data for the players...")
    time.sleep(2)
    msg.toast("Evaluating game performance...")
    time.sleep(2)
    msg.toast("Comparing players...")
    time.sleep(2)
    msg.toast("Ready!")
    st.success(final_result)


parser = argparse.ArgumentParser()
parser.add_argument(
    "--api",
    help="Lichess API key",
)

args = parser.parse_args()
api = args.api

project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
)
model_path = os.path.join(
    project_root, "models", "models", "KNeighborsDist", "model.pkl"
)

with open(model_path, "rb") as model_file:
    model = pickle.load(model_file)
    metadata = model.feature_metadata
    print(metadata)


st.title("Prophet - AI Chess Prediction")

st.write(
    """
## Provide the player names to see which one would win!
"""
)
st.logo(os.path.join(project_root, "data", "ichess.png"))
st.sidebar.image(os.path.join(project_root, "data", "Checkmate-Prophet-03b.jpeg"))
st.sidebar.markdown("POWERED BY PJAIT STUDENTS - PLEASE LET US PASS")

white = st.text_input("White:")
black = st.text_input("Black:")
modes = ["bullet", "blitz", "rapid", "classical"]
game_mode = st.selectbox("Game mode:", modes)

if st.button("Predict"):
    white_elo = get_elo(white, game_mode, api)
    black_elo = get_elo(black, game_mode, api)
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
    logger.error(input_data)

    prediction = model.predict(input_data)
    if prediction[0] == 0:
        final_result = f"{black} would win with {white} in a {game_mode} game!"
    elif prediction[0] == 2:
        final_result = f"{white} would win with {black} in a {game_mode} game!"
    fake_loader(final_result=final_result)
    st.balloons()
