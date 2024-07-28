import streamlit as st
from keras import models
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

st.title("Stock Price Predictor App")

stock = st.text_input("Enter the Stock ID", "NVDA")

from datetime import datetime
end = datetime.now()
start = datetime(end.year-5, end.month, end.day)

nvidia_data = yf.download(stock, start, end)

model = models.load_model("stock_predictor.h5")
st.subheader("Stock Data")
st.write(nvidia_data)

splitting_len = int(len(nvidia_data)*0.7)
x_test = pd.DataFrame(nvidia_data.Close[splitting_len:])

def plot_graph(fig_size, values, full_data):
    fig = plt.figure(figsize=fig_size)
    plt.plot(values, 'Orange')
    plt.plot(full_data.Close, 'b')
    return fig

st.subheader('Original Close Price and MA for 250 days')
nvidia_data['MA_for_250_days'] = nvidia_data.Close.rolling(250).mean()
st.pyplot(plot_graph(15,6), nvidia_data['MA_for_250_days', nvidia_data])

st.subheader('Original Close Price and MA for 200 days')
nvidia_data['MA_for_200_days'] = nvidia_data.Close.rolling(200).mean()
st.pyplot(plot_graph(15,6), nvidia_data['MA_for_200_days', nvidia_data])

st.subheader('Original Close Price and MA for 100 days')
nvidia_data['MA_for_100_days'] = nvidia_data.Close.rolling(100).mean()
st.pyplot(plot_graph(15,6), nvidia_data['MA_for_100_days', nvidia_data])

st.subheader('Original Close Price and MA for 100 days')
nvidia_data['MA_for_100_days'] = nvidia_data.Close.rolling(100).mean()
st.pyplot(plot_graph(15,6), nvidia_data['MA_for_100_days', nvidia_data])
