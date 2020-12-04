import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import warnings

from tensorflow.keras.layers import Dense, LSTM, Flatten, Dropout
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import TimeSeriesSplit
from array import array

from math import sqrt
from sklearn.metrics import mean_squared_error

import plotly.express as px

# Runtime path to the Processing directory in order to import the pre-procesisng moudles for each building lot

import sys
sys.path.insert(1, 'Project/Pre-Processing/Model_Processing/Renewable')
import AU_Solar
import DK_Solar
import FR_Solar

# Plots the MAPE and Loss charts. These charts show the MAPE after each epoch and the loss after each epoch
# It then  saves the plots in a seperate folder. It uses the 'history' script which comes pre-build with Keras

def plot_model_mape(history, string):
    
    train_mape = history.history['root_mean_squared_error']
    val_mape = history.history['val_root_mean_squared_error']

    plt.figure(figsize=(5, 5))
    plt.plot(train_mape, label='Training MAPE')
    plt.plot(val_mape, label='Validation MAPE')
    plt.legend()
    plt.title('Epochs vs. Training and Validation MAPE')

    plt.savefig('Project/Dashboards/assets/images/Solar/' + string + '_MAPE_LSTM.png', bbox_inches='tight')

def plot_model_loss(history, string):
    
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(5, 5))
    plt.plot(train_loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend()
    plt.title('Epochs vs. Training and Validation Loss')
    
    plt.savefig('Project/Dashboards/assets/images/Solar/' + string + '_Loss_LSTM.png', bbox_inches='tight')

# LSTM neural network moudle.

def LSTM_Netwrok(data, string):

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

        # Builds the model. LSTM defines the amount of LSTM cells in the network
        # Return sequence defiens wetaher the modle return the final h valeu of the full array of results form each cell
        # flatten() reduces the datas dimensions and it gets passed to a Dense layer of 160 nuerons
        # Dropout radnomly drops 10% of nuerons to prevent over-fitting

        LSTM_Model = tf.keras.models.Sequential([
            LSTM(80, input_shape = input_shape, return_sequences = True),
            Flatten(),
            Dense(160, activation='tanh'),
            Dropout(0.1),
            Dense(1)
        ])

        optimizer = tf.keras.optimizers.Adam(lr = .0009)
        LSTM_Model.compile(loss = loss, optimizer = optimizer, metrics = metric)
        tf.keras.backend.set_epsilon(1)

        Model = LSTM_Model.fit(train, epochs = 100, validation_data = val)  

        # Forecats is a build in keras model that appleis the trianed network to new data
        # The forecats sclaer values are then transformed back to real vlaues and passed to the MAPE fucntion

        forecast = LSTM_Model.predict(data.X_test)
        LSTM_forecast = data.scaler.inverse_transform(forecast)
        rms = sqrt(mean_squared_error(y_test_inv, LSTM_forecast))

        a.append(np.array(rms)) 

        # MAPE and Loss are plotted

        plot_model_mape(Model, string)
        plot_model_loss(Model, string)

    # The modle and the wights are saved as JSON ands h5 files 

    LSTM_JSON = LSTM_Model.to_json()
    with open("Project/Saved_Models/Solar/" + string + "/LSTM/" + string + "_LSTM.json", "w") as json_file:
        json_file.write(LSTM_JSON)
        
    LSTM_Model.save_weights("Project/Saved_Models/Solar/" + string + "/LSTM/" + string + "_LSTM.h5")

    print('CNN forecast MAPE of hour-ahead electricity demand: {}'.format(a))
    return LSTM

# Run each lot through the model and clear the network afterward


AU_Solar_LSTM = LSTM_Netwrok(AU_Solar, 'AU_Solar')
tf.keras.backend.clear_session()

DK_Solar_LSTM = LSTM_Netwrok(DK_Solar, 'DK_Solar')
tf.keras.backend.clear_session()

FR_Solar_LSTM = LSTM_Netwrok(FR_Solar, 'FR_Solar')
tf.keras.backend.clear_session()
