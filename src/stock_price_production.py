import pandas as pd  #for data manipulation operations
import numpy as np   #for linear algebra

#Libraries for visualisation

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


import datetime as dt

from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import LSTM

from itertools import cycle

#Loading the required data
df = pd.read_csv('C:\\Users\\augus\\OneDrive\\Desktop\\Tesla Stock Prices (2010-2023).csv')
df.set_index('Date',inplace=True)
df.head()

print('Number of days present in the dataset: ',df.shape[0])
print('Number of fields present in the dataset: ',df.shape[1])

df.info()
df.describe()

#EDA & Feature Engineering

from plotly.offline import init_notebook_mode
init_notebook_mode(connected=True)

#Check for null values
df.isnull() .sum()

#Plots

data = df.iloc[2300:].copy()

plt.figure(figsize=(30, 15))
ax = sns.lineplot(x=data.index, y=data['Close'])
plt.xticks(['19/08/2019', '16/03/2020', '26/02/2021', '15/03/2022', '05/01/2023'])
plt.title('Tesla Stock Close Price Over Time', fontsize=20)  # Title added here
plt.show()

# Plotly Plot for Stock Price Analysis (Open, High, Low, Close)
data = df.iloc[2300:].copy()

names = cycle(['Stock Open Price', 'Stock High Price', 'Stock Low Price', 'Stock Close Price'])

fig = px.line(data, x=data.index, y=[data['Open'], data['High'], data['Low'], data['Close']],
              labels={'date': 'Date', 'value': 'Stock value'})
fig.update_layout(title_text='Tesla Stock Price Analysis (2019 - 2023)',  # Title added here
                  font_size=15, font_color='black', legend_title_text='Stock Parameters')
fig.update_xaxes(showgrid=False)
fig.update_yaxes(showgrid=False)
fig.for_each_trace(lambda t: t.update(name=next(names)))
fig.show()

#Moving Averages(MA)

ma_day = [30, 60, 120,150]

for ma in ma_day:
        column_name = f"MA for {ma} days"
        data[column_name] = data['Close'].rolling(ma).mean()

plt.figure(figsize=(30,15))
plt.plot(data['Close'],label='Close Price')
plt.plot(data['MA for 30 days'],label='30 days MA')
plt.plot(data['MA for 60 days'],label='60 days MA')
plt.plot(data['MA for 120 days'],label='120 days MA')
plt.plot(data['MA for 150 days'],label='150 days MA')
plt.xticks(['19/08/2019','16/03/2020','26/02/2021','15/03/2022','05/01/2023'])
plt.legend()
plt.show()

names = cycle(['Close Price','MA 30 days','MA 60 days','MA 120 days','MA 150 days'])

fig = px.line(data, x=data.index ,y=[data['Close'],data['MA for 30 days'],data['MA for 60 days'],data['MA for 120 days'], data['MA for 150 days']],labels={'date': 'Date','value':'Stock value'})
fig.update_layout(title_text='Moving Average Analysis', font_size=15, font_color='black',legend_title_text='Stock Parameters')
fig.update_xaxes(showgrid=False)
fig.update_yaxes(showgrid=False)
fig.for_each_trace(lambda t:  t.update(name = next(names)))
fig.show()

#Splitting the Time-Series Data
# Creating a new dataframe with only 'Close'
new_df = data['Close']
new_df.index = data.index

final_df=new_df.values

train_data=final_df[0:646,]
test_data=final_df[646:,]

train_df = pd.DataFrame()
test_df = pd.DataFrame()

train_df['Close'] = train_data
train_df.index = new_df[0:646].index
test_df['Close'] = test_data
test_df.index = new_df[646:].index

print("train_data: ", train_df.shape)
print("test_data: ", test_df.shape)

#Scaling Data Using MIN-MAX Scaler
# Using Min-Max scaler to scale data
scaler=MinMaxScaler(feature_range=(0,1))
scaled_data=scaler.fit_transform(final_df.reshape(-1,1))

X_train_data,y_train_data=[],[]

for i in range(60,len(train_df)):
    X_train_data.append(scaled_data[i-60:i,0])
    y_train_data.append(scaled_data[i,0])
    
X_train_data,y_train_data=np.array(X_train_data),np.array(y_train_data)

X_train_data=np.reshape(X_train_data,(X_train_data.shape[0],X_train_data.shape[1],1))

#Model Building
# Initializing the LSTM model
model = Sequential()
model.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train_data.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.2))
model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.2))
model.add(LSTM(units = 50))
model.add(Dropout(0.2))
model.add(Dense(units = 1))

model.summary()
model.compile(optimizer = 'adam', loss = 'mean_squared_error')
model.fit(X_train_data, y_train_data, epochs = 150, batch_size = 32);

#Predictions

# Preparing input data and generating predictions

# Ensure the input data is properly reshaped and scaled
input_data = new_df[len(new_df) - len(test_df) - 60:].values
input_data = input_data.reshape(-1, 1)
input_data = scaler.transform(input_data)

# Prepare the X_test dataset for prediction
X_test = []
for i in range(60, input_data.shape[0]):
    X_test.append(input_data[i - 60:i, 0])

X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Generate predictions using the trained model
predicted = model.predict(X_test)
predicted = scaler.inverse_transform(predicted)

# Check if predictions are being created correctly
print(f"Predicted values:\n{predicted[:5]}")  # Print first few predicted values
print(f"Length of Predictions: {len(predicted)}")
print(f"Length of Test DataFrame: {len(test_df)}")

# Add predictions to the test_df DataFrame
test_df['Predictions'] = np.nan  # Initialize the 'Predictions' column
test_df.iloc[60:, test_df.columns.get_loc('Predictions')] = predicted.flatten()  # Add predictions from index 60 onwards

# Check if the Predictions column exists and is populated
print(f"Test DataFrame columns: {test_df.columns}")
print(f"Test DataFrame with Predictions:\n{test_df[['Close', 'Predictions']].head()}")

# Plotting using Matplotlib
plt.figure(figsize=(50, 10))
plt.plot(train_df['Close'], label='Training Data')
plt.plot(test_df['Close'], label='Test Data')

# Plot predictions only if available
if 'Predictions' in test_df and not test_df['Predictions'].isnull().all():
    plt.plot(test_df['Predictions'], label='Prediction')
else:
    print("Predictions are either missing or empty.")

plt.xticks(['19/08/2019', '16/03/2020', '26/02/2021', '15/03/2022', '05/01/2023'])
plt.legend()
plt.title('Stock Price Prediction vs Actual Data')
plt.show()

# Plotting using Plotly for better visualization
fig = go.Figure()

# Adding Training Data
fig.add_trace(go.Scatter(x=train_df.index, y=train_df['Close'],
                         mode='lines', name='Training Data'))

# Adding Test Data
fig.add_trace(go.Scatter(x=test_df.index, y=test_df['Close'],
                         mode='lines', name='Test Data'))

# Adding Predictions if available
if 'Predictions' in test_df and not test_df['Predictions'].isnull().all():
    fig.add_trace(go.Scatter(x=test_df.index, y=test_df['Predictions'],
                             mode='lines', name='Prediction'))
else:
    print("Predictions are either missing or empty.")

# Customize the layout
fig.update_layout(title_text='Stock Price Prediction Using LSTM Model',
                  xaxis_title='Date',
                  yaxis_title='Stock Price',
                  font_size=15, font_color='black')

# Show the plot
fig.show()



#MSE and MAE
print('The Mean Squared Error is',mean_squared_error(test_df['Close'].values,test_df['Predictions'].values))
print('The Mean Absolute Error is',mean_absolute_error(test_df['Close'].values,test_df['Predictions'].values))
print('The Root Mean Squared Error is',np.sqrt(mean_squared_error(test_df['Close'].values,test_df['Predictions'].values)))
