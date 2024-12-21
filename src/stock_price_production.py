# stock_price_prediction.py

# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error

# Load your data
data = pd.read_csv('data/stock_prices.csv')

# Ensure the 'Date' column is in datetime format and set as the index
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Plot the stock data to visualize
plt.figure(figsize=(10, 6))
plt.plot(data['Close'], label='Stock Price')
plt.title('Stock Price History')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.show()

# Example preprocessing steps
# Assuming 'Close' is the column for stock closing prices
stock_data = data[['Close']].values

# Scale the data to a range between 0 and 1
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(stock_data)

# Prepare the data for training
# We'll use 60 previous days to predict the next day's stock price
X_train, y_train = [], []
for i in range(60, len(scaled_data)):
    X_train.append(scaled_data[i-60:i, 0])  # Taking 60 previous days
    y_train.append(scaled_data[i, 0])      # Next day's price

# Convert to numpy arrays
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshape X_train to be in 3D shape [samples, time steps, features] for LSTM input
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Build the LSTM model
model = Sequential()

# First LSTM layer with dropout to prevent overfitting
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))

# Second LSTM layer with dropout
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))

# Fully connected layer to predict the stock price
model.add(Dense(units=1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32)

# To make predictions, you need to prepare the test data.
# Assuming you have a separate test dataset, here is an example of using the model to predict.
# Here, I will demonstrate how to prepare the data based on the training data.

# Create a test set by using the last 60 days of data for prediction
test_data = data[-60:]  # Last 60 days of data
test_scaled = scaler.transform(test_data[['Close']].values)

X_test = []
y_test = test_data[['Close']].values

# Prepare the test set (similar to the training data)
for i in range(60, len(test_scaled)):
    X_test.append(test_scaled[i-60:i, 0])  # Taking 60 previous days
    y_test.append(test_scaled[i, 0])      # Actual next day's price

# Convert to numpy arrays
X_test, y_test = np.array(X_test), np.array(y_test)

# Reshape X_test to match the LSTM input
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Make predictions on the test set
predictions = model.predict(X_test)

# Inverse transform the predictions and y_test to get the actual values
predictions = scaler.inverse_transform(predictions)
y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

# Calculate Mean Squared Error (MSE) to evaluate the model
mse = mean_squared_error(y_test, predictions)
print(f"Mean Squared Error: {mse}")

# Plot the actual vs predicted stock prices
plt.figure(figsize=(10, 6))
plt.plot(y_test, color='blue', label='Actual Stock Price')
plt.plot(predictions, color='red', label='Predicted Stock Price')
plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()
