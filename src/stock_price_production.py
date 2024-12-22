# stock_price_prediction_long_term.py

# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error

# Load the stock data from the local CSV
file_path = r"C:\Users\augus\OneDrive\Desktop\Tesla Stock Prices (2010-2023).csv"
  # Update the path if needed
data = pd.read_csv(file_path)

# Ensure the 'Date' column is in datetime format and set as the index
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Visualize the stock data
plt.figure(figsize=(10, 6))
plt.plot(data['Close'], label='Tesla Stock Price')
plt.title('Stock Price History')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.show()

# Preprocess the data
# Assuming 'Close' is the column for stock closing prices
stock_data = data[['Close']].values

# Scale the data to a range between 0 and 1
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(stock_data)

# Prepare the data for training
lookback = 60  # Number of past days to use
horizon = 20   # Number of future days to predict

def create_sequences(data, lookback, horizon):
    X, y = [], []
    for i in range(len(data) - lookback - horizon + 1):
        X.append(data[i:i + lookback])
        y.append(data[i + lookback:i + lookback + horizon])
    return np.array(X), np.array(y)

X, y = create_sequences(scaled_data, lookback, horizon)

# Split data into training and testing sets
split_ratio = 0.8
train_size = int(len(X) * split_ratio)

X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Reshape X_train and X_test for LSTM input
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Build the LSTM model
model = Sequential()

# First LSTM layer with dropout
model.add(LSTM(units=100, return_sequences=True, input_shape=(lookback, 1)))
model.add(Dropout(0.3))

# Second LSTM layer with dropout
model.add(LSTM(units=100, return_sequences=False))
model.add(Dropout(0.3))

# Fully connected layer to predict multiple days
model.add(Dense(units=horizon))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
history = model.fit(
    X_train, y_train, 
    epochs=100, 
    batch_size=64, 
    validation_data=(X_test, y_test), 
    verbose=1
)

# Plot training and validation loss
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Evaluate the model on test data
predictions = model.predict(X_test)

# Inverse transform the predictions and y_test
predictions = scaler.inverse_transform(predictions)
y_test = scaler.inverse_transform(y_test.reshape(-1, horizon))

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(y_test.flatten(), predictions.flatten())
print(f"Mean Squared Error: {mse}")

# Plot actual vs predicted stock prices for a sample
plt.figure(figsize=(12, 6))
plt.plot(range(horizon), y_test[0], label='Actual Prices', marker='o')
plt.plot(range(horizon), predictions[0], label='Predicted Prices', linestyle='--', marker='x')
plt.title('Sample Long-Term Stock Price Prediction')
plt.xlabel('Days')
plt.ylabel('Stock Price')
plt.legend()
plt.show()
