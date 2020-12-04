# Import all dependencies 

import plotly.express as px
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

import dash as dash
import  dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.io as pio
pio.templates
import plotly.figure_factory as ff
import dash_table

from keras.models import model_from_json
from keras.models import load_model
import requests
import json

from math import sqrt
from sklearn.metrics import mean_squared_error

# Runtime path to the Processing and Database directories to import the pre-procesisng and database modules

import sys
sys.path.insert(1, 'Project/Pre-Processing/Model_Processing/Renewable')
import AU_Wind
import DK_Wind
import FR_Wind

sys.path.append('Project/Database')
import DB_Write

# Module to calculate the mean absolute percentage error between two passed arrays

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# Transformer to encode target values

def y_test_inv (data):

    y_test_reshaped = data.y_test.reshape(-1, 1)
    y_test_inv = data.scaler.inverse_transform(y_test_reshaped)

    return y_test_inv

# Loads the weights and JSON files for the saved models from the Saved_Models folder

def select_model(country, passed_model):
    
    model_json_file = open ('Project/Saved_Models/Wind/' + country + '/' + passed_model + '/' + country + '_' + passed_model + '.json', 'r')
    model = model_json_file.read()
    model_json_file.close()
    model = model_from_json(model)
    model.load_weights('Project/Saved_Models/Wind/' + country + '/' + passed_model + '/' + country + '_' + passed_model + '.h5')
    model.save('Project/Saved_Models/Wind/' + country + '/' + passed_model + '/' + country + '_' + passed_model + '.hdf5')
    model = load_model('Project/Saved_Models/Wind/' + country + '/' + passed_model + '/' + country + '_' + passed_model + '.hdf5', compile = False)   

    return model

# Passes the chosen model and building daat to the model processing script and runs the test data for the chosen building through the chosen model 

def test(data, Lot): 

    test = data.predict(Lot.X_test)
    forecast = Lot.scaler.inverse_transform(test)
    return forecast

app = dash.Dash(__name__) 
server = app.server

# Design parameters for the Dash frontend 

# Dash HTML - this format was cloned from a sample Dash applictaion maintained by Plotly. 

app.layout = html.Div(
    
    [
        dcc.Store(id="aggregate_data"),
        html.Div(id="output-clientside"),
        html.Div(
            [
                html.Div(
                    [
                        html.Img(
                            src = "assets/images/Dash_Logo.png",
                            id="plotly-image",
                            style={
                                "height": "80px",
                                "width": "auto",
                                "left": "300px;",

                            },
                        )
                    ],
                    className="one-third column",
                ),

                html.Div(
                    [
                        html.Div(
                            [
                                html.H3(
                                    " Next Hour Energy Usage",
                                    style={"margin-bottom": "0px"},
                                ),
                                html.H5(
                                    "Wind Generation", style={"margin-top": "0px"}
                                ),
                            ]
                        )
                    ],
                    className="one-half column",
                    id="title",
                ),
                    html.Div([
                        dcc.Dropdown(id='select_model',
                            options=[
                            {'label': 'MLP', 'value': 4},
                            {'label': 'LSTM', 'value': 5}, 
                            {'label': 'CNN', 'value': 6},
                            {'label': 'CNN-LSTM', 'value': 7}],

                        multi = False,
                        value = 4,
                    ), 
                ], className = 'dropDown'), 

            ],
            id="header",
            className="row flex-display",
            style={"margin-bottom": "25px"},
        ),

        html.Div(
            [
                html.Div(
                    html.Img(id = "image", src='', style={ "margin-top": "20px", "height": "520px", "width": "370px"}),
                className="pretty_container four columns",

                ),

                html.Div(
                    [
                        html.Div(
                            [
                                html.Div(
                                    dcc.Dropdown(id='select_building',
                                        options=[
                                        {'label': 'Austria', 'value': 1},
                                        {'label': 'France', 'value': 2}, 
                                        {'label': 'Germany', 'value': 3}],

                                    multi = False,
                                    value = 1,
                                ),),

                            ],
                            id="info-container",
                            className="LotdropDown",

                        ),
                        html.Div(
                            [dcc.Graph(id="graph", figure={})],
                            id="countGraphContainer",
                            className="pretty_container",
                        ),
                    ],
                    id="right-column",
                    className="eight columns",
                ),
            ],
            className="row flex-display",
        ),
        html.Div(
            [

                html.Div(
                    html.Img(id = "MAPE", src=''),
                    className="pretty_container seven columns",
                ),

                html.Div(
                    html.Img(id = "Loss", src=''),
                    className="pretty_container five columns",
                ),
            ],
            className="row flex-display",
        ),
    ],

    id="mainContainer",
    style={"display": "flex", "flex-direction": "column"},

)

#---------------------------------------------------------------------
    
# Dash callback that loads the main image of each building lot and displays it on the application 
    
@app.callback(
    Output('image','src'),
    [Input('select_building','value'),])

@server.route('/Wind')
def choose_city(data):
    
    country_code = {1:"Austria", 2:"France", 3:"German"}

    image = ('assets/Renewable/' + country_code[data] + '.png')

    return (image)

#---------------------------------------------------------------------

# Dash callback that loads the prediction graph displayed on the dash application 

@server.route('/Wind')
@app.callback(
    Output('graph','figure'),
    [Input('select_building','value'),
    Input('select_model','value')],)

def inital_graph(country, model_value):

    country_str = {1: 'AU_Wind', 2: 'FR_Wind', 3: 'DK_Wind'}
    country_code = {1: AU_Wind, 2: FR_Wind, 3: DK_Wind}
    model_code = {4:'MLP', 5:'LSTM', 6:'CNN', 7:'CNN_LSTM'}

    model = select_model(country_str[country], model_code[model_value])

    predict = test(model, country_code[country])
    y = y_test_inv(country_code[country])
    string = country_str[country] + '_' + model_code[model_value]

    dataset = pd.DataFrame({'Model': string, 'predicted': predict[0 : 370:, 0].round(2), 'actual': y[0 : 370:, 0].round(2)})
    dataset = dataset.reset_index()
    rms = sqrt(mean_squared_error(predict, y))
    dataset.to_csv(r'Project/Data/Predicted/' + string + '.csv', index=False) 

    fig = px.scatter(title = 'December 2018 - Avereage Root Mean Sqaured Error = {}'.format(round(rms, 3)))
    fig.add_trace(go.Scatter(x = dataset.index, y = dataset['predicted'], name = 'predicetd', line = dict(width=3, dash='dot')))
    fig.add_trace(go.Scatter(x = dataset.index, y = dataset['actual'], mode = 'lines', name = 'real'))

    fig.update_layout(
        font_family="Courier New",
        title_font_family="Times New Roman",
        title_font_color="blue",
        xaxis_title="Time Steps (Hourly)",
        yaxis_title="Energy Usage (MwH)",
    )

    fig.update_xaxes(title_font_family="Arial")
    fig.update_yaxes(title_font_family="Arial")

    # Writes the predicetd values to the Database

    try: 
        DB_Write.write_city(string)
    except Exception as e:
        print('Error:', e)

    return(fig)

#---------------------------------------------------------------------

# Dash callback that loads the MAPE plot displayed on the application 

@app.callback(
    Output('MAPE','src'),
    [Input('select_building','value'),
    Input('select_model','value'),])

def graph_mape (city, model):

    city_code = {1:"AU", 2:"FR", 3:"DK"}
    model_code = {4:"MLP", 5:"LSTM", 6:"CNN", 7:'CNN_LSTM'}

    MAPE = "assets/images/Wind/" + city_code[city] + "_wind_MAPE_" + model_code[model] + ".png"

    return (MAPE)

#---------------------------------------------------------------------

# Dash callback that loads the Loss plot displayed on the  application 

@app.callback(
    Output('Loss','src'),
    [Input('select_building','value'),
    Input('select_model','value')],)

def graph_loss (city, model):

    city_code = {1:"AU", 2:"FR", 3:"DK"}
    model_code = {4:"MLP", 5:"LSTM", 6:"CNN", 7:'CNN_LSTM'}

    Loss = "assets/images/Wind/" + city_code[city] + "_wind_Loss_" + model_code[model] + ".png"

    return (Loss)

#---------------------------------------------------------------------

if __name__ == '__main__':
    app.run_server(debug=True)