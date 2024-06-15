import os
import requests
import streamlit as st
import pickle
import pandas as pd
import argparse

# TO DO
# Test api call
# Test output
# Beautify


def get_elo(user, api_key):
    rating = 1500
    url = f'https://lichess.org/api/user/{user}'
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }
    response = requests.get(url=url, headers=headers)
    if response.status_code == 200:
        data = response.json()
        rating = int(data['profile']['fideRating'])
    else:
        print(f'Error: {response.status_code}')
    return rating


parser = argparse.ArgumentParser()
parser.add_argument(
    "--api",
    help="Lichess API key",
)

args = parser.parse_args()
api = args.api


dirname = os.path.dirname(os.path.dirname(__file__))
model_path = os.path.join(dirname, 'models', 'models', 'KNeighborsDist', 'model.pkl')

with open(model_path, 'rb') as model_file:
    model = pickle.load(model_file)
    metadata = model.feature_metadata
    print(metadata)

st.title("Chess Match Prediction")

st.write("""
## Provide the player names to see which one would win!
""")


white = st.text_input("White:")
black = st.text_input("Black:")


if st.button('Predict'):
    white_elo = get_elo(white, api)
    black_elo = get_elo(black, api)
    white_diff = white_elo - black_elo
    black_diff = black_elo - white_elo
    input_data = pd.DataFrame({
        'White': [white],
        'Black': [black],
        'WhiteElo': [white_elo],
        'BlackElo': [black_elo],
    })

    prediction = model.predict(input_data)

    st.write(f"Prediction: {prediction[0]} would win!")
    st.baloons()
