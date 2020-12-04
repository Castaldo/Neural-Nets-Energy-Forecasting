import pandas as pd 
import numpy as np
from math import sqrt
from sklearn.metrics import mean_squared_error, r2_score

# Naive class that simple meaures the average error rate beteween each index in the test data. This is the metric
# we want our models to surpass

Germany = pd.read_csv("Project/Data/Processed/Germany.csv")
Austria = pd.read_csv("Project/Data/Processed/Austria.csv")
France = pd.read_csv("Project/Data/Processed/France.csv")

def naive_2(Data):
    
    y  = Data['Solar'].values

    _train = y[:6143]
    y_cv = y[6143 : 7459]
    y_test = y[7459:]


    naive_day_ahead = y[7459 - 1 : 8784 - 1]
    rmse_naive = sqrt(mean_squared_error(y_test, naive_day_ahead))

    print('RMSE of day-ahead electricity price naive forecast: {}'.format(round(rmse_naive, 3)))

naive_2(Austria)
naive_2(France)

def naive(Data):
    
    y  = Data['Solar'].values

    _train = y[:6143]
    y_cv = y[6143 : 7459]
    y_test = y[7459:]


    naive_day_ahead = y[7459 - 1 : 8784 - 1]
    rmse_naive = sqrt(mean_squared_error(y_test, naive_day_ahead))

    print('RMSE of day-ahead electricity price naive forecast: {}'.format(round(rmse_naive, 3)))

naive(Germany)
