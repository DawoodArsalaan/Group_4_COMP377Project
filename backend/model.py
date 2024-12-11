import requests
import numpy as np
import pandas as pd
from tensorflow import keras
from keras.api.models import Sequential
from keras.api.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

def fetch_data(coin, days):
    url = f"https://api.coingecko.com/api/v3/coins/{coin}/market_chart"
    params = {"vs_currency": "usd", "days": days}
    response = requests.get(url, params=params)
    if response.status_code != 200:
        raise Exception("Failed to fetch data")
    data = response.json()
    prices = [item[1] for item in data['prices']]  # Extract prices
    return np.array(prices)

def fetch_and_train_model(coin, days):
    prices = fetch_data(coin, days)

    # Prepare data
    scaler = MinMaxScaler()
    prices_scaled = scaler.fit_transform(prices.reshape(-1, 1))

    X, y = [], []
    for i in range(60, len(prices_scaled)):
        X.append(prices_scaled[i-60:i, 0])
        y.append(prices_scaled[i, 0])
    X, y = np.array(X), np.array(y)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    # Build LSTM model
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
        LSTM(50),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=10, batch_size=32, verbose=0)

    # Predict
    predicted_prices = model.predict(X)
    predicted_prices = scaler.inverse_transform(predicted_prices)
    return predicted_prices.flatten().tolist()

def predict_prices(coin, days):
    return fetch_and_train_model(coin, days)
