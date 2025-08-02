
import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import datetime

st.set_page_config(layout="centered")
st.title("📈 Stock Price Forecasting using LSTM")
st.markdown("This app uses an LSTM model to predict stock prices for the next 7 days.")

model = load_model("lstm_stock_model.keras")

ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, RELIANCE.NS)", "AAPL")

if st.button("Predict"):
    end = datetime.datetime.today()
    start = end - datetime.timedelta(days=100)
    df = yf.download(ticker, start=start, end=end)

    if df.empty:
        st.error("❌ Unable to fetch data. Please check ticker symbol.")
    else:
        st.success("✅ Data successfully loaded.")
        data = df[['Close']]
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data)
        x_input = scaled_data[-60:]
        x_input = np.reshape(x_input, (1, 60, 1))

        temp_input = list(x_input[0])
        predictions = []

        for _ in range(7):
            x_pred = np.array(temp_input[-60:]).reshape(1, 60, 1)
            yhat = model.predict(x_pred, verbose=0)
            predictions.append(yhat[0][0])
            temp_input.append(yhat[0])

        forecasted_prices = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
        future_days = list(range(1, 8))
        st.line_chart(pd.DataFrame({
            'Day': future_days,
            'Forecasted Price': forecasted_prices.flatten()
        }).set_index('Day'))

        st.write("### Forecasted Prices:")
        st.table(pd.DataFrame({
            "Day": future_days,
            "Forecasted Price": forecasted_prices.flatten().round(2)
        }))
