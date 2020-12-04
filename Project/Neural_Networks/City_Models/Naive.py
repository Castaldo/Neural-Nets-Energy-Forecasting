import pandas as pd 
import numpy as np

# Naive class that simple meaures the average error rate beteween each index in the test data. This is the metric
# we want our models to surpass

NYC = pd.read_csv('Project/Data/Processed/NY.csv')
Pittsburgh = pd.read_csv('Project/Data/Processed/PA.csv')
SanDeigo = pd.read_csv('Project/Data/Processed/SD.csv')
SanFran = pd.read_csv('Project/Data/Processed/SF.csv')
Vancouver = pd.read_csv('Project/Data/Processed/VA.csv')

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def NY(Data):
    
    y = Data['Energy'].values

    y_train = y[:30678]
    y_cv = y[30678 : 37251]
    y_test = y[37251:]

    naive_day_ahead = y[37251 - 1 : 43456 - 1]
    rmse_naive_day = mean_absolute_percentage_error(y_test, naive_day_ahead)

    print('RMSE of day-ahead electricity price naive forecast: {}'.format(round(rmse_naive_day, 3)))

NY(NYC)

def Pitt(Data):
    
    y = Data['Energy'].values

    y_train = y[:30678]
    y_cv = y[30678 : 37251]
    y_test = y[37251:]

    naive_day_ahead = y[37251 - 1 : 43342 - 1]
    rmse_naive_day = mean_absolute_percentage_error(y_test, naive_day_ahead)

    print('RMSE of day-ahead electricity price naive forecast: {}'.format(round(rmse_naive_day, 3)))

Pitt(Pittsburgh)

def SF(Data):
    
    y = Data['Energy'].values

    y_train = y[:30678]
    y_cv = y[30678 : 37251]
    y_test = y[37251:]

    naive_day_ahead = y[37251 - 1 : 43180- 1]      
    rmse_naive_day = mean_absolute_percentage_error(y_test, naive_day_ahead)

    print('RMSE of day-ahead electricity price naive forecast: {}'.format(round(rmse_naive_day, 3)))

SF(SanFran)

def SD(Data):
    
    Data = Data.fillna(0)

    y = Data['Energy'].values

    y_train = y[:30678]
    y_cv = y[30678 : 37251]
    y_test = y[37251:]

    naive_day_ahead = y[37251 - 1 : 43231- 1]      
    rmse_naive_day = mean_absolute_percentage_error(y_test, naive_day_ahead)

    print('RMSE of day-ahead electricity price naive forecast: {}'.format(round(rmse_naive_day, 3)))

SD(SanDeigo)

def VA(Data):
    
    y = Data['Energy'].values

    y_train = y[:30678]
    y_cv = y[30678 : 37251]
    y_test = y[37251:]

    naive_day_ahead = y[37251 - 1 : 43775 - 1]
    rmse_naive_day = mean_absolute_percentage_error(y_test, naive_day_ahead)

    print('RMSE of day-ahead electricity price naive forecast: {}'.format(round(rmse_naive_day, 3)))

VA(Vancouver)


