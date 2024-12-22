import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Load your data (replace with your dataset)
data = pd.read_csv(r"C:\Users\augus\OneDrive\Desktop\Tesla Stock Prices (2010-2023).csv")

# Preprocess the data: Assuming the 'Close' column is the target for stock prices
prices = data['Close'].values.reshape(-1, 1)

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
prices_scaled = scaler.fit_transform(prices)

# Define lookback and horizon
lookback = 60  # Number of previous days to use for prediction
horizon = 5    # Number of days to predict

# Create datasets for training and testing
def create_dataset(data, lookback, horizon):
    X, y = [], []
    for i in range(lookback, len(data) - horizon + 1):
        X.append(data[i - lookback:i, 0])  # Take lookback days
        y.append(data[i:i + horizon, 0])  # Take next 'horizon' days as target
    return np.array(X), np.array(y)

# Split the data into training and testing sets
train_size = int(len(prices_scaled) * 0.8)
train_data, test_data = prices_scaled[:train_size], prices_scaled[train_size:]

X_train, y_train = create_dataset(train_data, lookback, horizon)
X_test, y_test = create_dataset(test_data, lookback, horizon)

# Reshape X to be 3D for LSTM [samples, time steps, features]
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Build the LSTM model
model = Sequential()

# First LSTM layer with dropout and L2 regularization
model.add(LSTM(units=30, return_sequences=True, input_shape=(lookback, 1), kernel_regularizer='l2'))
model.add(Dropout(0.3))  # Increased dropout

# Second LSTM layer with dropout and L2 regularization
model.add(LSTM(units=30, return_sequences=False, kernel_regularizer='l2'))
model.add(Dropout(0.3))  # Increased dropout

# Fully connected layer to predict multiple days
model.add(Dense(units=horizon))

# Compile the model with a reduced learning rate and use a scheduler
optimizer = Adam(learning_rate=0.0001)  # Reduced learning rate
model.compile(optimizer=optimizer, loss='mean_squared_error')

# Implement early stopping with a patience of 10
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
history = model.fit(
    X_train, y_train, 
    epochs=50,  # Reduced number of epochs
    batch_size=64, 
    validation_data=(X_test, y_test),
    callbacks=[early_stopping],  # Use early stopping
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
