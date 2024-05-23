import numpy as np
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from keras.models import load_model
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR

# Function to plot predictions vs original
def plot_predictions(y_test, y_prediction, label, color):
    plt.plot(y_test, label='Original Price', color='b')
    plt.plot(y_prediction, label=label, color=color)
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()

# Load data
start = '2013-01-01'
end = '2023-12-31'
st.title('Stock Trend Prediction')
user_input = st.text_input('Enter Stock ticker', 'AAPL')
df = yf.download(user_input, start=start, end=end)
st.subheader('Data from 2013 - 2023')
st.write(df.describe())

# Visualization
st.subheader('Closing Price vs Time Chart')
fig = plt.figure(figsize=(12, 6))
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with 100MA')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize=(12, 6))
plt.plot(ma100)
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with 100MA & 200MA')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize=(12, 6))
plt.plot(ma100, 'r')
plt.plot(ma200, 'g')
plt.plot(df.Close, 'b')
st.pyplot(fig)

# Prepare training and testing data
data_training = pd.DataFrame(df['Close'][0:int(len(df) * 0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df) * 0.70): int(len(df))])

# Load the pre-trained LSTM model
model = load_model('keras_model.h5')

from sklearn.preprocessing import MinMaxScaler


# Prepare the testing data for LSTM
scaler = MinMaxScaler(feature_range=(0, 1))
data_training_array = scaler.fit_transform(data_training)

past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i - 100: i])
    y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)

# Make LSTM predictions
y_prediction_lstm = model.predict(x_test)

# Plot LSTM predictions vs original
st.subheader('LSTM Prediction vs Original')
fig2 = plt.figure(figsize=(12, 6))
plot_predictions(y_test, y_prediction_lstm, 'LSTM Predicted Price', 'r')
st.pyplot(fig2)

# Prepare data for other algorithms
X_train = np.arange(len(data_training)).reshape(-1, 1)
y_train = data_training.values.ravel()
X_test = np.arange(len(data_training), len(df)).reshape(-1, 1)

# Train and predict using Linear Regression
model_lr = LinearRegression()
model_lr.fit(X_train, y_train)
y_prediction_lr = model_lr.predict(X_test)

# Plot Linear Regression predictions vs original
st.subheader('Linear Regression Prediction vs Original')
fig3 = plt.figure(figsize=(12, 6))
plot_predictions(data_testing.values, y_prediction_lr, 'Linear Regression Predicted Price', 'g')
st.pyplot(fig3)

# Train and predict using Random Forest
model_rf = RandomForestRegressor()
model_rf.fit(X_train, y_train)
y_prediction_rf = model_rf.predict(X_test)

# Plot Random Forest predictions vs original
st.subheader('Random Forest Prediction vs Original')
fig4 = plt.figure(figsize=(12, 6))
plot_predictions(data_testing.values, y_prediction_rf, 'Random Forest Predicted Price', 'c')
st.pyplot(fig4)

# Train and predict using SVM
model_svm = SVR()
model_svm.fit(X_train, y_train)
y_prediction_svm = model_svm.predict(X_test)

# Plot SVM predictions vs original
st.subheader('SVM Prediction vs Original')
fig5 = plt.figure(figsize=(12, 6))
plot_predictions(data_testing.values, y_prediction_svm, 'SVM Predicted Price', 'm')
st.pyplot(fig5)

# Train and predict using Gradient Boosting
model_gb = GradientBoostingRegressor()
model_gb.fit(X_train, y_train)
y_prediction_gb = model_gb.predict(X_test)

# Plot Gradient Boosting predictions vs original
st.subheader('Gradient Boosting Prediction vs Original')
fig6 = plt.figure(figsize=(12, 6))
plot_predictions(data_testing.values, y_prediction_gb, 'Gradient Boosting Predicted Price', 'y')
st.pyplot(fig6)
