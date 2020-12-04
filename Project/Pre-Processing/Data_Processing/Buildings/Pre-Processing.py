import pandas as pd
from scipy.constants import convert_temperature
import numpy as np

Buildings = pd.read_csv('Project/Data/Unprocessed/Energy/BuildingsDF.csv')
Buildings = Buildings.dropna()
New_York_Load_Demand = pd.read_csv('Project/Data/Unprocessed/Energy/NYC_Demand.csv')
Weather_2016 = pd.read_csv('Project/Data/Unprocessed/Weather/NycWeather.csv')

# Inserts space or zero into certain column values. This was necessary to transform some of the values into the proper datatypes 
 
def insert_space(string, integer):
    return string[0:integer] + ' ' + string[integer:]

def insert_zero(string, integer):
    return string[0:integer] + ':00:00' + string[integer:]

# Maps the weekend to the weekdays and seasons to the months to apply over the database

seasons = [1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 1]
weekend = [0, 0, 0, 0, 0, 1, 1]
day_to_weekend = dict(zip(range(0,7), weekend))
month_to_season = dict(zip(range(1,13), seasons))

# Standardizes the naming conventions of all the datastes 
# Deletes null values in certain metrics
# Eliminates all buildings that are not in Manhattan and drops nulls 

def Rename(Data):
    
    Data = Data.rename(columns={"Building Usage": "Usage"})
    Data = Data.rename(columns={"Total GHG Emissions (Metric Tons CO2e)": "Pollution"})
    Data = Data.rename(columns={"Electricity Usage (kWh)": "Electricity"})
    Data = Data[Data.Electricity != 'Not Available']
    Data = Data[Data.Pollution != 'Not Available']
    Data = Data[Data.Borough == 'Manhattan']
    
    return Data

Buildings = Rename(Buildings)   

# Selects the building lots from the database that are being tested

Lot_835 = Buildings[Buildings['BBL - 10 digits'].str.contains('100835')] 
Lot_1265 = Buildings[Buildings['BBL - 10 digits'].str.contains('101265')] 
Lot_1297 = Buildings[Buildings['BBL - 10 digits'].str.contains('101297')]  

# Gets rid of buildings that are accidentally in the lot 
# Converts electricty usage form KWH to MWH 
# Creates a column that sums the electricity usage of all the buildings in the lots
# Creates index and sums the buildings in the lot

def Normalzie(Building):

    Building = Building[Building.Address != '445 East 69th ST']
    Building = Building[Building.Address != '420 East 70th St']
    Building = Building[Building.Address != '299 Riverside Drive']
    
    Building['Electricity'] = Building['Electricity'].astype('float64')
    Building['Electricity'] = Building['Electricity'] / 1000
    Building['Index'] = range(1, len(Building) + 1)
    Building.reset_index
    Building['Lot_Electricty'] = Building['Electricity'].sum() 
    
    return Building

# Selects NYC region from New York's hourly energy usage dataset
# Standardizes naming conventions

def Clean_Load(Data):
    
    Data = Data[Data['Zone Name'].str.contains("N.Y.C")]
    Data = Data.rename(columns={"Eastern Date Hour": "Date"})
    Data = Data.rename(columns={"DAM Forecast Load": "Energy_Total"})
    Data = Data.rename(columns={"GMT Start Hour": "Hour"})
    
    return Data

New_York_Load_Demand = Clean_Load(New_York_Load_Demand)

# Cleans the weather data to make it mergable and standarized with the other datasets
# Creates hour column

def Clean_Weather(Data):
    
    Data = Data.rename(columns={"# Date": "Date"})
    Data['Date'] = Data['Date']  + ' ' + Data['UT time']
    Data['Date'] = Data['Date'].str.replace(r'\b24:00\b', '00:00')
    Data['Date'] = Data['Date'].astype('datetime64[ns]')
    Data['Hour'] = Data['Date'].dt.hour
    del Data['UT time']
    
    return Data

Weather_2016 = Clean_Weather(Weather_2016)

# Creates a weighted sum of the buildings energy usage to be mapped over NYC's hourly usage
# This is used to estimate how much energy each building was consuming at a given hour

Load_Total = New_York_Load_Demand['Energy_Total'].sum()

def Weight(Building, Load_Total, NY):

    Building['Lot_Weight'] = Building['Lot_Electricty'] / Load_Total
    Building = pd.concat([Building]*8784, ignore_index=True)
    
    return Building

# Concates the wetaher and energy data so that they are able to be merged with lot 835 dataset
# Creates an index column in which the datasets can be merged on
# Drops nulls

def Lot_835_Fix(Building, NY, Weather):
    
    Weather = pd.concat([Weather]*3, ignore_index=True)
    NY = pd.concat([NY]*3, ignore_index=True)

    NY = NY.sort_values(['Date'], ascending=[True])
    NY['Index'] = range(1, len(NY) + 1)
    
    Building['Index'] = range(1, len(Building) + 1)
    Building = Building.sort_values(['Index'], ascending=[True])

    Weather = Weather.sort_values(['Date'], ascending=[True])
    Weather['Index'] = range(1, len(Weather) + 1)

    Merge = pd.merge(NY, Building, on= 'Index', how='left')
    Building = pd.merge(Merge, Weather, on= 'Index', how='left')

    Building.dropna()
    
    return Building

# Concates the wetaher and energy data so that they are able to be merged with lot 1265 dataset
# Creates an index column in which the datasets can be merged on
# Drops nulls

def Lot_1265_Fix(Building, NY, Weather):
    
    Weather = pd.concat([Weather]*2, ignore_index=True)
    NY = pd.concat([NY]*2, ignore_index=True)

    NY = NY.sort_values(['Date'], ascending=[True])
    NY['Index'] = range(1, len(NY) + 1)
    
    Building['Index'] = range(1, len(Building) + 1)
    Building = Building.sort_values(['Index'], ascending=[True])

    Weather = Weather.sort_values(['Date'], ascending=[True])
    Weather['Index'] = range(1, len(Weather) + 1)

    Merge = pd.merge(NY, Building, on= 'Index', how='left')
    Building = pd.merge(Merge, Weather, on= 'Index', how='left')
    
    Building.dropna()

    return Building

# Concates the wetaher and energy data so that they are able to be merged with lot 1297 dataset
# Creates an index column in which the datasets can be merged on
# Drops nulls

def Lot_1297_Fix(Building, NY, Weather):
    
    Weather = pd.concat([Weather]*2, ignore_index=True)
    NY = pd.concat([NY]*2, ignore_index=True)

    NY = NY.sort_values(['Date'], ascending=[True])
    NY['Index'] = range(1, len(NY) + 1)
    
    Building['Index'] = range(1, len(Building) + 1)
    Building = Building.sort_values(['Index'], ascending=[True])

    Weather = Weather.sort_values(['Date'], ascending=[True])
    Weather['Index'] = range(1, len(Weather) + 1)

    Merge = pd.merge(NY, Building, on= 'Index', how='left')
    Building = pd.merge(Merge, Weather, on= 'Index', how='left')

    Building.dropna()
    
    return Building

# Creates a 'lot hourly usage' column by summing the enrgy usage of all the buildings in the lot
# This is the metric that the model is testing
# Fixes naming conventions
# Sets the varaibles to be included in the final dataset

def Clean(Building, NY, Weather):
    
    Building['Lot Hourly Usage'] = Building['Energy_Total'] * Building['Lot_Weight']
    Building['Lot % Change'] = Building['Lot Hourly Usage'] / Building['Lot_Electricty']
    
    Building = Building.rename(columns={"Energy_Total": "NYC_Hourly"})
    Building = Building.rename(columns={"Date_x": "Date"})
    Building = Building.rename(columns={"Hour_y": "Hour"})
    
    Varaibles = ['Date', 'NYC_Hourly', 'Temperature', 'Relative Humidity', 'Pressure', 'Wind speed', 'Wind direction', 'Rainfall',
                'Snowfall', 'Snow depth', 'Short-wave irradiation', 'Hour', 'Lot Hourly Usage', 'Lot % Change']
    
    Building = Building[Varaibles]
    Building['Date'] = Building['Date'].astype('datetime64[ns]')

    return Building

# Adds temporal data to help map energy consumption patterns

def add_values(data):
    
    Holiday_List = ['25/12/15', '25/12/16', '25/12/17', '25/12/18', '25/12/19', '31/12/15', '31/12/16', '31/12/17', '31/12/18', '31/12/19',
    '04/07/15', '04/07/16', '04/07/17', '04/07/18', '04/07/19', '26/11/15', '24/11/16', '23/11/17', '22/11/18', '28/11/19', '17/03/15', '17/03/16', 
    '17/03/17', '17/03/18', '17/03/19']

    data['Check_Holiday'] = pd.to_datetime(data["Date"]).dt.strftime('%d/%m/%y')
    data['Check_Holiday'] = data['Check_Holiday'].isin(Holiday_List)
    data['Holiday'] = data['Check_Holiday']
    data['Holiday'] = data['Holiday'].astype('float64')
    del data['Check_Holiday']
    
    data['Hour'] = data['Hour'].astype('float64')

    data['Day'] = data['Date'].dt.dayofweek
    data['Day'] = data['Day'].astype('float64')

    data['Month'] = data['Date'].dt.month
    data['Month'] = data['Month'].astype('float64')
    
    data['Season'] = data.Month.map(month_to_season)
    data['Weekend'] = data.Day.map(day_to_weekend)

    data = data.round(2)
    
    return data

# Re-orders the columns and delets duplicate rows

def Re_order(data):

    data = data.drop_duplicates(subset=['Date'], keep='first')
    
    Order_list = ['NYC_Hourly', 'Temperature', 'Relative Humidity', 'Pressure' ,
    'Wind speed', 'Wind direction', 'Rainfall', 'Snowfall', 'Snow depth', 'Short-wave irradiation',
    'Hour', 'Lot % Change', 'Holiday', 'Day', 'Month', 'Season', 'Weekend', 'Lot Hourly Usage']

    data = data [(Order_list)]

    return data

# Remove any vlaues that are within four standard deviations form the mean

def remove_outliers(data):

    for value in data:
        mean_feat = data[value].mean()
        dev = data[value].std()
        upper = mean_feat + 3 * dev
        lower = mean_feat - 3 * dev
        data.loc[data[value] > upper, value] = np.nan
        data.loc[data[value] < lower, value] = np.nan

        return data

# Runs each building through the fucntions sequentially 

Lot_835 = Normalzie(Lot_835)
Lot_835 = Weight(Lot_835, Load_Total, New_York_Load_Demand)
Lot_835 = Lot_835_Fix(Lot_835, New_York_Load_Demand, Weather_2016)
Lot_835 = Clean(Lot_835, New_York_Load_Demand, Weather_2016)
Lot_835 = add_values(Lot_835)
Lot_835 = Re_order(Lot_835)
Lot_835 = remove_outliers(Lot_835)

Lot_1265 = Normalzie(Lot_1265)
Lot_1265 = Weight(Lot_1265, Load_Total, New_York_Load_Demand)
Lot_1265 = Lot_1265_Fix(Lot_1265, New_York_Load_Demand, Weather_2016)
Lot_1265 = Clean(Lot_1265, New_York_Load_Demand, Weather_2016)
Lot_1265 = add_values(Lot_1265)
Lot_1265 = Re_order(Lot_1265)
Lot_1265 = remove_outliers(Lot_1265)

Lot_1297 = Normalzie(Lot_1297)
Lot_1297 = Weight(Lot_1297, Load_Total, New_York_Load_Demand)    
Lot_1297 = Lot_1297_Fix(Lot_1297, New_York_Load_Demand, Weather_2016)
Lot_1297 = Clean(Lot_1297, New_York_Load_Demand, Weather_2016)
Lot_1297 = add_values(Lot_1297)
Lot_1297 = Re_order(Lot_1297)
Lot_1297 = remove_outliers(Lot_1297)

########################################################################

# Write the final dataframe to a csv and save 

Lot_835.to_csv(r'Project/Data/Processed/Lot_835.csv', index=False) 
print('Lot 835 Data Processed')

Lot_1265.to_csv(r'Project/Data/Processed/Lot_1265.csv', index=False) 
print('Lot 1265 Data Processed')

Lot_1297.to_csv(r'Project/Data/Processed/Lot_1297.csv', index=False) 
print('Lot 1297 Data Processed')