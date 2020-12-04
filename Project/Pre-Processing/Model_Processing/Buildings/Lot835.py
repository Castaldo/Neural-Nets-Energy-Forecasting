import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler

# Model pre-proecessing class 

Lot_835 = pd.read_csv('Project/Data/Processed/Lot_835.csv')

# Drop null values, drop non-numerical columns, convert all vlaues to the same data typer
# Select the relevent model features  

Lot_835 = Lot_835.dropna()
Lot_835 = Lot_835.astype('float64')

Features = ['Temperature', 'Relative Humidity', 'Short-wave irradiation', 'Hour', 'Month', 'Day', 'Holiday', 'Season', 'Weekend', 'Wind speed']

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

# Slims the dataset down to only the relevent features and defines the time lag

MLP_Dataset = Lot_835[Features]
timesteps = 1

# Splits the data 70/15/15

train_set = 6148
validation_set = 7466
test_set = 8783

# Scales all the values between 0 and 1 

scaler_all = MinMaxScaler(feature_range=(0, 1))
scaler_all.fit(MLP_Dataset[:train_set])
scaled_dataset = scaler_all.transform(MLP_Dataset)

# Defines the metric the network is trying to predict and sets the datastes accordingly 

Lot_Energy = Lot_835['Lot Hourly Usage'].values
scaler = MinMaxScaler(feature_range=(0, 1))

Energy_reshaped = Lot_Energy.reshape(-1, 1)
scaler.fit(Energy_reshaped[:train_set])

scaled_variable = scaler.transform(Energy_reshaped)
scaled_dataset = np.concatenate((scaled_dataset, scaled_variable), axis=1)

# Passes the parameters to the neural network funtion to create the various sets

X_test, y_test = Split_Data (scaled_dataset, scaled_dataset[:, -1], validation_set, test_set, timesteps)
