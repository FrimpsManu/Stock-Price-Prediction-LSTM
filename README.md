# Stock Price Prediction using LSTM 
 
This project uses a Long Short-Term Memory (LSTM) neural network to predict daily stock prices based on historical data. The model is trained on Tesla stock price data (from 2010 to 2023), sourced from Kaggle. By leveraging the power of LSTM, the project aims to predict future stock price movements based on historical patterns.


## Project Overview
The goal of this project is to use deep learning techniques to build a model that predicts stock prices, focusing on Tesla Inc. The model utilizes an LSTM network, which is particularly effective for time series forecasting. Stock prices exhibit temporal dependencies, making LSTM a good fit for this task.

The project involves:
- Collecting and preparing historical stock price data for Tesla (2010–2023) from Kaggle.
- Preprocessing the data (including normalization and transformation into a format suitable for LSTM training).
- Building and training an LSTM model to predict future stock prices.
- Visualizing the predictions and comparing them with actual values.

 
## Project Structure 
 
-Stock-Price-Prediction-LSTM
1. Data
    -Folder to store datasets stock_prices.csv  -Tesla stock prices dataset (2010-2023) from    Kaggle

2. Src
    -Source code for the project stock_price_prediction.py
    -Python file for the core functionality

3. Requirements.txt
    -List of project dependencies README.md
    -Project description and instructions .gitignore
    -Gitignore to exclude unnecessary files
 

## Requirements 
 
To install the required dependencies, run the following command: 
 
pip install the libraries in the requirements.txt 

 
## Usage 
 
To run the stock price prediction model, use the following command: 
 
-python src/stock_price_prediction.py 

The script will load the Tesla stock price data (from 2010 to 2023), preprocess it, build the LSTM model, train it, and generate predictions for future stock prices.

 
## License 
 
This project is licensed under the MIT License. 
