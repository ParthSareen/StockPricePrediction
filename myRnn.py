#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 22:43:30 2019

@author: parthsareen
LSTM to capture upward and downward google stock trends
many layers- dropout reg. 
5 years worth of google data

"""
# libs
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Data preprocessing
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[: ,1:2].values
#using normalization for rnn with sigmoid

#feature scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1))
training_set_scaled = sc.fit_transform(training_set)

#data structure with 6- timesteps and 1 output
"""
60 timesteps of past information to understand trends from
basically 3 months of data
"""
X_train = []
y_train = []

for i in range(60, 1258):
    X_train.append(training_set_scaled[i - 60 : i, 0])
    y_train.append(training_set_scaled[i , 0])
 
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping the data, can change num of indicators at the end
# X_train.shape[0] = lines of x_train, X_train.shape[1] = columns of x_train
# last is number of indicators

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
# Building the RNN
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Initializing the RNN
# regressor since comntinuous output
regressor = Sequential()

# First lstm with drop reg. to prevent overfitting
# units, return sequences, shape of x_train (3 dimension,)
regressor.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
regressor.add(Dropout(0.2)) # ignore 20% of neurons, 10 neurons ignored

# second layer, input shape is not needed anymore, recognizes shape this time
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2)) # ignore 20% of neurons, 10 neurons ignored

# third layer
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2)) # ignore 20% of neurons, 10 neurons ignored

# fourth layer, no more sequences returned
regressor.add(LSTM(units=50, return_sequences=False))
regressor.add(Dropout(0.2)) # ignore 20% of neurons, 10 neurons ignored

# Output layer, one neural, stock price at time t+1
regressor.add(Dense(units=1))

# Compiling the RNN
#regressor.compile(optimizer='RMSprop')
# loss this time is the mean squared error since continuous problem
regressor.compile(optimizer='adam', loss='mean_squared_error')

# Fitting the RNN to the Training set
# x -> input, y_train -> ground truth?, epochs forward and back prop
regressor.fit(X_train, y_train, epochs=100, batch_size=32)

# Prediction
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_train.iloc[: ,1:2].values
# Getting the real google stock price (from test set)

# Getting the predicted stock price of 2017, it is based on last 2016
# needs prev 60 days to predict any day, will need t-60 days worth of data, will need training and test date
# what we wanna concatenate, 0 is vertical 1 is horizontal
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis=0)
# lower bound is first financial day - 60 days
# len total is final index and len test is 20, which gets to jan 3
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
#need to reshape for numpy, might get warning for 3d shape of inputs
inputs = inputs.reshape(-1,1)
#scaling inputs, not test vals
inputs = sc.transform(inputs)

x_test = []
# upper bound is for 20 days
for i in range(60, 80):
    x_test.append(training_set_scaled[i - 60 : i, 0])
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

predicted_stock_price = regressor.predict(x_test)
#inverse scaling of regressor, inv transform method
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# Visualizing the results
plt.plot(real_stock_price, color='red', label='Real Google Stock Price')
plt.plot(predicted_stock_price, color='blue', label='Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()