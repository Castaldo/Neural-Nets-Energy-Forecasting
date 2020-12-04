import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler

# Model pre-proecessing class 
# Creates a variable with only the relevent model features   

SF = pd.read_csv("Project/Data/Processed/SF.csv")
SF = SF.fillna(0)

Features = ['Temperature', 'Short-wave irradiation', 'Hour', 'Season', 'Wind speed', 'Day', 'Weekend', 'Relative Humidity']

# Function to create the training, validation, and test datasets

def Nueral_Network (dataset, target, start, end, timesteps):
    
    data = []
    labels = []

    if end is None:
        end = len(dataset)
        
    for i in range(start, end):
        
        indices = range(i - timesteps, i)
        data.append(dataset[indices])
        labels.append(target[i])

    return np.array(data), np.array(labels)

# Sets the dataset to only inlcude the selected model features and sets the number of time lags

Dataset = SF[Features]
timesteps = 3

# Splits the data 70/15/15

train_set = 30678
validation_set = 37251
test_set = 43180

# Scales all the values between 0 and 1 

scaler_all = MinMaxScaler(feature_range=(0, 1))
scaler_all.fit(Dataset[:train_set])
scaled_dataset = scaler_all.transform(Dataset)

# Defines the metric the network is trying to predict
# Scales and combines the datasets

Energy = SF['Energy'].values
scaler = MinMaxScaler(feature_range=(0, 1))

Energy_reshaped = Energy.reshape(-1, 1)
scaler.fit(Energy_reshaped[:train_set])

scaled_variable = scaler.transform(Energy_reshaped)
scaled_dataset = np.concatenate((scaled_dataset, scaled_variable), axis=1)

# Passes the proper parameters that are passed to the neural network funtion 
# This creates the training, validation, and test sets

X_test, y_test = Split_Data(scaled_dataset, scaled_dataset[:, -1], validation_set, test_set, timesteps)
