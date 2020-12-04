## Introduction 

This an end-to-end data scienece project using Nueral Netwroks to make short-term energy consumption and generation forecasts. A more detailed report on thsi project can be found in the **Applications of Neural Networks for Time Series Energy Forecasting.pdf**

## Notebooks 

The **Notebooks** folder contains .ipynb files with some of the data used in this project. The notebooks include pre-processing, EDA, and neural net implementations. They can be viewed in the links below 
  
**- France Renewable Genertaion:** https://nbviewer.jupyter.org/github/Castaldo/Neural-Nets-Energy-                     Forecasting/blob/master/Notebooks/Renewable%20Generation/French_Generation.ipynb

**- Residnetial Household Consumption:** https://nbviewer.jupyter.org/github/Castaldo/Neural-Nets-Energy-Forecasting/blob/master/Notebooks/Granular%20Consumption/House.ipynb

**- Large Hotel Consumption:** https://nbviewer.jupyter.org/github/Castaldo/Neural-Nets-Energy-Forecasting/blob/master/Notebooks/Granular%20Consumption/Hotel.ipynb

## Code 

The **Project** contains all the Python implementation of the project. 

**Dashboards** - This folder contains the four GUI's for the applictaion. When you run either of the scripts, the GUI will appear on http://127.0.0.1:8050/

**Data** - This folder contains all the CSV filed used in this porject. The 'Predicetd' folder contains the predicted values that each network made on the training data.

**Database** - This folder conatins the files that creates the Postgres DB, connect the application, and write the data.

**Neural_Networks** - This folder contains all of the Neural Networks used in this project.

**Pre-Processing** - This folder contains all the files that transformed the data. There are two seperate scripts. One contains the transformations on the unprocessed data and the other contains the transfomrtaion necessary for the model.

**Saved_Models** - This foder contains each datasets saved model
