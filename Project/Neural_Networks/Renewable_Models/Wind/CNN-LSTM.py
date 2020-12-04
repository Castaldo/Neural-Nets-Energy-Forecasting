import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import warnings

from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout, LSTM
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import TimeSeriesSplit
from array import array

from math import sqrt
from sklearn.metrics import mean_squared_error

import plotly.express as px

# Runtime path to the Processing directory in order to import the pre-procesisng moudles for each city datatset

import sys
sys.path.insert(1, 'Project/Pre-Processing/Model_Processing/Renewable')
import AU_Wind
import DK_Wind
import FR_Wind

# Plots the MAPE and Loss charts. These charts show the MAPE after each epoch and the loss after each epoch
# It then saves the plots in a seperate folder. It uses the 'history' script which comes pre-build with Keras  

def plot_model_mape(history, string):
    
    train_mape = history.history['root_mean_squared_error']
    val_mape = history.history['val_root_mean_squared_error']

    plt.figure(figsize=(5, 5))
    plt.plot(train_mape, label='Training MAPE')
    plt.plot(val_mape, label='Validation MAPE')
    plt.legend()
    plt.title('Epochs vs. Training and Validation MAPE')

    plt.savefig('Project/Dashboards/assets/images/Wind/' + string + '_MAPE_CNN_LSTM.png', bbox_inches='tight')

def plot_model_loss(history, string):
    
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(5, 5))
    plt.plot(train_loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend()
    plt.title('Epochs vs. Training and Validation Loss')
    
    plt.savefig('Project/Dashboards/assets/images/Wind/' + string + '_Loss_CNN_LSTM.png', bbox_inches='tight')

# Convolutional neural network moudle.

def CNN_LSTM(data, string):

    tscv = TimeSeriesSplit()
    TimeSeriesSplit(max_train_size=None, n_splits=5)
    a = []

    for train_index, test_index in tscv.split(data.scaled_dataset):
    
        print("TRAIN:", train_index, "TEST:", test_index)
        X_train, y_train = data.Nueral_Network(data.scaled_dataset, data.scaled_dataset[:, -1], 0, data.train_set, data.timesteps)
        X_val, y_val = data.Nueral_Network(data.scaled_dataset, data.scaled_dataset[:, -1], data.train_set, data.validation_set, data.timesteps)
        X_test, y_test = data.Nueral_Network(data.scaled_dataset, data.scaled_dataset[:, -1], data.validation_set, data.test_set, data.timesteps)

        # Defines the models input shape, the loss fucntion, and the metric used for the error function.
        # The 'data' passed into the moudle as an argument calls on the each lots pre-preocessing moudle to
        # obtain the training, testing, and validation data sets

        input_shape = X_train.shape[-2:]
        loss = tf.keras.losses.MeanSquaredError()
        metric = tf.keras.metrics.RootMeanSquaredError()

        # Reshapes the y_test numpy array so it cna be passes into the mean_absolute_percentage_error function
        # Reverses the scaler to re-obtain the atcual values of the data

        y_test_reshaped = y_test.reshape(-1, 1)
        y_test_inv = data.scaler.inverse_transform(y_test_reshaped)

        # Sets the amount of test sample to use in each iteration and shuffles the data to prevent over-fitting

        batch_size = 64
        shuffle_size = 64

        val = tf.data.Dataset.from_tensor_slices((X_val, y_val))
        val = val.cache().shuffle(shuffle_size).batch(shuffle_size).prefetch(1)

        train = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        train = train.cache().shuffle(shuffle_size).batch(shuffle_size).prefetch(1)

        # Builds the model. Filters defines the amopunt of sliding widnow that will move of the time series data
        # Kernal defines the size of the window
        # Strides defines how many inputs the window will move after each convultional 
        # Padding handles null vlaues that may result from the other parameters 
        # After the convultional layer, the dats's dimensions are reduced by the flatten() method and passed to a 
        # traditonal MLP network with 50 layers and 1 output layer

        CNN_LSTM = tf.keras.models.Sequential([Conv1D(filters=100, kernel_size=2,strides=1, padding='causal',
        activation='relu', input_shape = input_shape),

            LSTM(80, return_sequences=True),
            Flatten(),
            Dense(50, activation='relu'),
            Dense(1)
        ])

        optimizer = tf.keras.optimizers.Adam(lr = .0001, amsgrad=True)
        CNN_LSTM.compile(loss = loss, optimizer = optimizer, metrics = metric)
        tf.keras.backend.set_epsilon(1)

        Model = CNN_LSTM.fit(train, epochs = 100, validation_data = val) 

        # predict is a build in keras model that appleis the trianed network to new data
        # The forecats sclaer values are then transformed back to real vlaues and passed to the MAPE fucntion

        forecast = CNN_LSTM.predict(data.X_test)
        CNN_LSTM_forecast = data.scaler.inverse_transform(forecast)
        rms = sqrt(mean_squared_error(y_test_inv, CNN_LSTM_forecast)) 

        a.append(np.array(rms))    

        # MAPE and Loss are plotted

        plot_model_mape(Model, string)
        plot_model_loss(Model, string)

        # The modle and the wights are saved as JSON ands h5 files 

    CNN_LSTM_JSON = CNN_LSTM.to_json()
    with open("Project/Saved_Models/Wind/" + string + "/CNN_LSTM/" + string + "_CNN-LSTM.json", "w") as json_file:
        json_file.write(CNN_LSTM_JSON)
        
    CNN_LSTM.save_weights("Project/Saved_Models/Wind/" + string + "/CNN_LSTM/" + string + "_CNN-LSTM.h5")

    print('MLP forecast MAPE of hour-ahead electricity demand: {}'.format(a))
    return CNN_LSTM

# Run each lot through the model and clear the network afterward

AU_Wind_CNN_LSTM = CNN_LSTM(AU_Wind, 'AU_Wind')
tf.keras.backend.clear_session()

FR_Wind_CNN_LSTM = CNN_LSTM(FR_Wind, 'FR_Wind')
tf.keras.backend.clear_session()

DK_Wind_CNN_LSTM = CNN_LSTM(DK_Wind, 'DK_Wind')
tf.keras.backend.clear_session()