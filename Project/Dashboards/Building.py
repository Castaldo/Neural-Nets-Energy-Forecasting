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

# Runtime path to the Processing and Database directories to import the pre-procesisng and database modules 

import sys
sys.path.insert(1, 'Project/Pre-Processing/Model_Processing/Buildings')
import Lot835
import Lot1265
import Lot1297

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

def select_model(buidling, passed_model):
    
    model_json_file = open ('Project/Saved_Models/Buildings/' + buidling + '/' + passed_model + '/' + buidling + '_' + passed_model + '.json', 'r')
    model = model_json_file.read()
    model_json_file.close()
    model = model_from_json(model)
    model.load_weights('Project/Saved_Models/Buildings/' + buidling + '/' + passed_model + '/' + buidling + '_' + passed_model + '.h5')
    model.save('Project/Saved_Models/Buildings/' + buidling + '/' + passed_model + '/' + buidling + '_' + passed_model + '.hdf5')
    model = load_model('Project/Saved_Models/Buildings/' + buidling + '/' + passed_model + '/' + buidling + '_' + passed_model + '.hdf5', compile = False)    

    return model

# Passes the chosen model and building daat to the model processing script and runs the test data for the chosen building through the chosen model 

def test(model, Lot): 

    test = model.predict(Lot.X_test)
    forecast = Lot.scaler.inverse_transform(test)
    return forecast

app = dash.Dash(__name__) 

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
                            id = "plotly-image",
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
                                    "Next Hour Energy Usage",
                                    style={"margin-bottom": "0px"},
                                ),
                                html.H5(
                                    "Building Usage", style={"margin-top": "0px"}
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
                                        {'label': 'Lot_835', 'value': 1},
                                        {'label': 'Lot_1265', 'value': 2}, 
                                        {'label': 'Lot_1297', 'value': 3}],

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

def choose_building(data):
    
    city_code = {1:"ESB", 2:"Rock", 3:"Chrysler"}

    image = ('assets/Buildings/' + city_code[data] + '.png')

    return (image)

#---------------------------------------------------------------------

# Dash callback that loads the prediction graph displayed on the dash application 

@app.callback(
    Output('graph','figure'),
    [Input('select_building','value'),
    Input('select_model','value')],)

def inital_graph(building, model_value):

    building_str = {1: 'Lot_835', 2: 'Lot_1265', 3: 'Lot_1297'}
    building_code = {1: Lot835, 2: Lot1265, 3: Lot1297}
    model_code = {4:'MLP', 5:'LSTM', 6:'CNN', 7:'CNN_LSTM'}

    # Sets the choosen model

    model = select_model(building_str[building], model_code[model_value])

    predict = test(model, building_code[building])
    y = y_test_inv(building_code[building])
    string = building_str[building] + '_' + model_code[model_value]

    # Plots the first 168 hours of the test set

    dataset = pd.DataFrame({'Model': string, 'predicted': predict[0 : 168:, 0], 'actual': y[0 : 168:, 0]})
    dataset = dataset.reset_index()
    MAPE = mean_absolute_percentage_error(predict, y)
    dataset.to_csv(r'Project/Data/Predicted/' + string + '.csv', index=False) 

    fig = px.scatter(title = '7 Day Prediction: Avereage Mean Percent Error = {}'.format(round(MAPE, 3)))
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
    Input('select_model','value')],)

def graph_MAPE (building, model):

    building_code = {1:"Lot_835", 2:"Lot_1265", 3:"Lot_1297"}
    model_code = {4:"MLP", 5:"LSTM", 6:"CNN", 7:'CNN-LSTM'}

    MAPE = 'assets/images/Buildings/' + building_code[building] + '_MAPE_' + model_code[model] + '.png'

    return (MAPE)


#---------------------------------------------------------------------

# Dash callback that loads the Loss plot displayed on the  application 

@app.callback(
    Output('Loss','src'),
    [Input('select_building','value'),
    Input('select_model','value')],)

def graph_loss (building, model):

    building_code = {1:"Lot_835", 2:"Lot_1265", 3:"Lot_1297"}
    model_code = {4:"MLP", 5:"LSTM", 6:"CNN", 7:'CNN-LSTM'}

    Loss = 'assets/images/Buildings/' + building_code[building] + '_Loss_' + model_code[model] + '.png'

    return (Loss)

#---------------------------------------------------------------------

if __name__ == '__main__':
    app.run_server(debug=True)

