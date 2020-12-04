import pandas as pd 
import numpy as np

# Naive class that simple meaures the average error rate beteween each index in the test data. This is the metric
# we want our models to surpass

Lot_835 = pd.read_csv('Project/Data/Processed/Lot_835.csv')
Lot_1265 = pd.read_csv('Project/Data/Processed/Lot_1265.csv')
Lot_1297 = pd.read_csv('Project/Data/Processed/Lot_1297.csv')

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def naive(Data):
    y = Data['Lot Hourly Usage'].values

    y_train = y[:6148]
    y_cv = y[6148 : 7466]
    y_test = y[7466:]

    naive_day_ahead = y[7466 - 1 : 8783 - 1]
    rmse_naive_day = mean_absolute_percentage_error(y_test, naive_day_ahead)

    print('RMSE of day-ahead electricity price naive forecast: {}'.format(round(rmse_naive_day, 3)))

naive(Lot_835)
naive(Lot_1265)
naive(Lot_1297)