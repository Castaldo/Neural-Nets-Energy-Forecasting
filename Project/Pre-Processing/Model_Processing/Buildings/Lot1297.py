import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from tensorflow.keras.layers import Dense, TimeDistributed, Flatten, Dropout
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

# Model pre-proecessing class 
# Creates a variable with only the relevent model features  

Lot_1297 = pd.read_csv('Project/Data/Processed/Lot_1297.csv')

# Drop null values, drop non-numerical columns, convert all vlaues to the same data typer
# Select the relevent model features  

Lot_1297 = Lot_1297.dropna()
Lot_1297 = Lot_1297.astype('float64')

Features = ['Temperature', 'Relative Humidity', 'Short-wave irradiation', 'Hour', 'Month',
            'Day', 'Holiday', 'Season', 'Weekend', 'Wind speed']

# Function to create the training, validation, and test datasets

def Nueral_Network (dataset, target, start, end, timesteps):
    
    data = []
    labels = []

    start = start + timesteps
    
    if end is None:
        end = len(dataset)
        
    for i in range(start, end):
        
        indices = range(i - timesteps, i)
        data.append(dataset[indices])
        labels.append(target[i])

    return np.array(data), np.array(labels)

# Sets the dataset to only inlcude the selected model features and sets the number of time lags

MLP_Dataset = Lot_1297[Features]
timesteps = 1

# Splits the data 70/15/15

train_set = 6148
validation_set = 7466
test_set = 8783

# Scales all the values between 0 and 1 

scaler_all = MinMaxScaler(feature_range=(0, 1))
scaler_all.fit(MLP_Dataset[:train_set])
scaled_dataset = scaler_all.transform(MLP_Dataset)

# Defines the metric the network is trying to predict
# Scales and combines the datasets

Lot_Energy = Lot_1297['Lot Hourly Usage'].values
scaler = MinMaxScaler(feature_range=(0, 1))

Energy_reshaped = Lot_Energy.reshape(-1, 1)
scaler.fit(Energy_reshaped[:train_set])

scaled_demand = scaler.transform(Energy_reshaped)
scaled_dataset = np.concatenate((scaled_dataset, scaled_demand), axis=1)

# Passes the proper parameters that are passed to the neural network funtion 
# This creates the training, validation, and test sets

X_test, y_test = Split_Data(scaled_dataset, scaled_dataset[:, -1], validation_set, test_set, timesteps)




    
