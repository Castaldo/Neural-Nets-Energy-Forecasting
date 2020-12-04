import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler

# Model pre-proecessing class 
# Creates a variable with only the relevent model features  

France = pd.read_csv("Project/Data/Processed/France.csv")

Features = ['Temperature_y', 'Month', 'Hour', 'Month', 'Season', 'Wind speed', 'Wind Velocity (10m)']

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

Dataset = France[Features]
timesteps = 5

# Splits the data 70/15/15

train_set = 6143
validation_set = 7459
test_set = 8776

# Scales all the values between 0 and 1 

scaler_all = MinMaxScaler(feature_range=(0, 1))
scaler_all.fit(Dataset[:train_set])
scaled_dataset = scaler_all.transform(Dataset)

# Defines the metric the network is trying to predict
# Scales and combines the datasets

Wind = France['Wind'].values
scaler = MinMaxScaler(feature_range=(0, 1))

Wind_reshaped = Wind.reshape(-1, 1)
scaler.fit(Wind_reshaped[:train_set])

scaled_variable = scaler.transform(Wind_reshaped)
scaled_dataset = np.concatenate((scaled_dataset, scaled_variable), axis=1)

# Passes the proper parameters that are passed to the neural network funtion 
# This creates the training, validation, and test sets

X_test, y_test = Split_Data(scaled_dataset, scaled_dataset[:, -1], validation_set, test_set, timesteps)
