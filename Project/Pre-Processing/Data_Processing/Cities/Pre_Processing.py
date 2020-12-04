import pandas as pd
from scipy.constants import convert_temperature
import numpy as np
import datetime

# Inserts space or zero into certain column vlaues. This was necessary to transform some of the values into the proper datatypes 

def insert_space(string, integer):
    return string[0:integer] + ' ' + string[integer:]

def insert_zero(string, integer):
    return string[0:integer] + ':00:00' + string[integer:]

# Maps the weekend to the weekdays and seasons to the months to apply over the database

seasons = [1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 1]
weekend = [0, 0, 0, 0, 0, 1, 1]
day_to_weekend = dict(zip(range(0,7), weekend))
month_to_season = dict(zip(range(1,13), seasons))

# Import data

NYC_Weather = pd.read_csv('Project/Data/Unprocessed/Weather/NYC_Weather.csv')
Pittsburgh_Weather = pd.read_csv('Project/Data/Unprocessed//Weather/Pittsburgh_Weather.csv')
SanDiego_Weather = pd.read_csv('Project/Data/Unprocessed/Weather/SanDiego_Weather.csv')
SanFran_Weather = pd.read_csv('Project/Data/Unprocessed/Weather/SanFran_Weather.csv')
Vancouver_Weather = pd.read_csv('Project/Data/Unprocessed/Weather/Vancouver_Weather.csv')

# Cleans the weather data and fixes the date format to make the datasets able to merge

def clean_weather(data):

    data['Temperature'] = convert_temperature(data['Temperature'], 'Kelvin', 'Fahrenheit').round(0)
    data = data.rename(columns={"# Date": "Date"})
    data['Date'] = data['Date']  + ' ' + data['UT time']
    data['Date'] = data['Date'].str.replace(r'\b24:00\b', '00:00')
    data['Date'] = data['Date'].astype('datetime64[ns]')
    
    del data['UT time']

    return data

NYC_Weather = clean_weather(NYC_Weather)
Pittsburgh_Weather = clean_weather(Pittsburgh_Weather)
SanDiego_Weather = clean_weather(SanDiego_Weather)
SanFran_Weather = clean_weather(SanFran_Weather)
Vancouver_Weather = clean_weather(Vancouver_Weather)

# Formats the date column for the New York's dataset
# Standardizes naming conventions
# Cleans the date column for the NYC's dataset
# Adds temporal value and deletes unused columns

New_York_Load_Demand = pd.read_csv('Project/Data/Unprocessed/Energy/NYC_Load.csv')

def clean_NYC(data):
    
    data = data[data['Zone Name'].str.contains("N.Y.C")]
    data = data.rename(columns={"Eastern Date Hour": "Date"})
    data = data.rename(columns={"DAM Forecast Load": "New York"})
    data['New York'] = data['New York'].astype('float64')
    
    data['Date'] = data['Date'].astype('datetime64[ns]')
    data['Hour'] = data['Date'].dt.hour
    
    del data['Zone Name']
    del data['GMT Start Hour']
    
    return data
    
New_York_Load_Demand = clean_NYC(New_York_Load_Demand)

# Formats the date column for the Vancouver's dataset
# Standardizes naming conventions
# Eliminates anomolies
# Multiplies the energy usage vlaue to properly capture Vancouver consumption

Vancouver_Load_Demand = pd.read_csv('Project/Data/Unprocessed/Energy/Vancouver_Load.csv')

def clean_Vancouver(data):
        
    data = data[['Hourly Balancing Authority Load Report', 'Unnamed: 1', 'Unnamed: 2']]
    data = data.reset_index()
    del data['index']
    data["id"] = data.index
    data = data[data.id != 0]
    
    data = data.rename(columns={"Hourly Balancing Authority Load Report": "Date"})
    data = data.rename(columns={"Unnamed: 1": "Hour"})
    data = data.rename(columns={"Unnamed: 2": "Vancouver"})
    data = data.drop(data["Date"].loc[data["Date"]=='Date'].index)
    
    data['Hour_Con'] = data['Hour']
    data['Hour_Con'] = data['Hour_Con'].apply(lambda x: '{0:0>2}'.format(x))
    data['Hour_Con'] = data['Hour_Con'].apply(lambda x: insert_zero(x, 2))
    data['Hour_Con'] = data['Hour_Con'].str.replace(r'\b24:00:00\b', '00:00:00')
    data["Hour_Con"] = data['Hour_Con'].apply(lambda x: insert_space(x, 0))
    data["Date"] = data["Date"] + data["Hour_Con"].astype(str)
    del data['Hour_Con']
        
    data = data.drop(data["Date"].loc[data["Date"]=='11/3/19 2*:00:00'].index)
    data = data.drop(data["Date"].loc[data["Date"]=='11/1/15 2*:00:00'].index)
    data = data.drop(data["Date"].loc[data["Date"]=='11/6/16 2*:00:00'].index)
    data = data.drop(data["Date"].loc[data["Date"]=='11/5/17 2*:00:00'].index)
    data = data.drop(data["Date"].loc[data["Date"]=='11/4/18 2*:00:00'].index)
    data['Date'] = data['Date'].astype('datetime64[ns]')

    data['Vancouver'] = data['Vancouver'].str.replace(r',', '')
    data['Vancouver'] = data['Vancouver'].astype('float64')
    data = data.fillna(0)
    data['Vancouver'] = pd.to_numeric(data['Vancouver']).astype(int)
    data['Vancouver'] = (data[['Vancouver']] * .23).round(0)
    
    return data
    
Vancouver_Load_Demand = clean_Vancouver(Vancouver_Load_Demand)

# Formats the date column for the Califronia's dataset
# Standardizes naming conventions
# Multiplies the energy usage vlaue to properly capture San Francisco consumption

California_Load_Demand = pd.read_csv('Project/Data/Unprocessed/Energy/California_Load.csv')

def clean_Cali(data):

    data = data.rename(columns={"HE": "Hour"})
    data = data.rename(columns={"Dates": "Date"})
    data['Date'] = data['Date'].astype('datetime64[ns]')

    data = data.rename(columns={"PGE": "San Francisco"})
    data = data.rename(columns={"SDGE": "San Diego"})
    data['San Francisco'] = data['San Francisco'].astype('float64')
    data['San Diego'] = data['San Diego'].astype('float64')


    data["San Francisco"] = data["San Francisco"].astype('float64')
    data["San Diego"] = data["San Diego"].astype('float64')
    data['San Francisco'] = (data[['San Francisco']] * .055).round(0)
    
    return data

California_Load_Demand = clean_Cali(California_Load_Demand)

# Formats the date column for the Pittsburgh's datasest
# Standardizes naming conventions
# Adds temporal values

Pennsylvania_Load_Demand = pd.read_csv('Project/Data/Unprocessed/Energy/Pittsburgh_Load.csv')

def clean_PA(data):
    
    data = data[['datetime_beginning_ept', 'zone', 'mw']]
    data = data.rename(columns={"datetime_beginning_ept": "Date"})
    data = data.rename(columns={"mw": "Pittsburgh"})
    data = data[data['Date'] != 'datetime_beginning_ept']
    data['Date'] = data['Date'].astype('datetime64[ns]')
    data['Hour'] = data['Date'].dt.hour
    data['Pittsburgh'] = data['Pittsburgh'].astype('float64')
    
    return data
    
Pennsylvania_Load_Demand = clean_PA(Pennsylvania_Load_Demand)

# Adds additonal temporal data to help map energy consumption patterns

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

    data = data.round(2)
    
    return data

New_York_Load_Demand = add_values(New_York_Load_Demand)
Vancouver_Load_Demand = add_values(Vancouver_Load_Demand)
California_Load_Demand = add_values(California_Load_Demand)
Pennsylvania_Load_Demand = add_values(Pennsylvania_Load_Demand)

# Choose NYC columns and merge datasets
# Add season and month variables
# Create previous hour column 

def NYC_final(data):
    
    data = data[['Date', 'New York', 'Hour', 'Day', 'Month', 'Holiday']]
    data = pd.merge(data, NYC_Weather, on= "Date", how="left")
    data = data.rename(columns={"New York": "Energy"})
    data['Previous Hour'] = data['Energy'].shift(+1)
    data['Date'] = pd.to_datetime(data['Date']).dt.strftime('%m/%d/%Y') 
    data['Season'] = data.Month.map(month_to_season)
    data['Weekend'] = data.Day.map(day_to_weekend)
    
    return data
    
NY = NYC_final(New_York_Load_Demand)

# Choose Vancover columns and merge datasets
# Add season and month variables
# Create previous hour column 

def VA_final(data):
    
    data = data[['Date', 'Vancouver', 'Hour', 'Day', 'Month', 'Holiday']]
    data = pd.merge(data, Vancouver_Weather, on= "Date", how="left")
    data = data.rename(columns={"Vancouver": "Energy"})
    data['Previous Hour'] = data['Energy'].shift(+1)
    data['Date'] = pd.to_datetime(data['Date']).dt.strftime('%m/%d/%Y') 
    data['Season'] = data.Month.map(month_to_season)
    data['Weekend'] = data.Day.map(day_to_weekend)
    
    
    return data
    
VA = VA_final(Vancouver_Load_Demand)

# Choose Pittsburgh columns and merge datasets
# Add season and month variables
# Create previous hour column 

def Pitt_final(data):
    
    data = data[['Date', 'Pittsburgh', 'Hour', 'Day', 'Month', 'Holiday']]
    data = pd.merge(data, Pittsburgh_Weather, on= "Date", how="left")
    data = data.rename(columns={"Pittsburgh": "Energy"})
    data['Previous Hour'] = data['Energy'].shift(+1)
    data['Date'] = pd.to_datetime(data['Date']).dt.strftime('%m/%d/%Y') 
    data['Season'] = data.Month.map(month_to_season)
    data['Weekend'] = data.Day.map(day_to_weekend)
    
    return data
    
PA = Pitt_final(Pennsylvania_Load_Demand)

# Choose San Francisco columns and merge datasets
# Add season and month variables
# Create previous hour column 

def SF_final(data):
    
    data = data[['Date', 'San Francisco', 'Hour', 'Day', 'Month', 'Holiday']]
    data = pd.merge(data, SanFran_Weather, on= "Date", how="left")
    data = data.rename(columns={"San Francisco": "Energy"})
    data['Previous Hour'] = data['Energy'].shift(+1)
    data['Date'] = pd.to_datetime(data['Date']).dt.strftime('%m/%d/%Y') 
    data['Season'] = data.Month.map(month_to_season)
    data['Weekend'] = data.Day.map(day_to_weekend)
    
    return data
    
SF = SF_final(California_Load_Demand)

# Choose San Deigo columns and merge datasets
# Add season and month variables
# Create previous hour column 

def SD_final(data):
    
    data = data[['Date', 'San Diego', 'Hour', 'Day', 'Month', 'Holiday']]
    data = pd.merge(data, SanDiego_Weather, on= "Date", how="left")
    data = data.rename(columns={"San Diego": "Energy"})
    data['Previous Hour'] = data['Energy'].shift(+1)
    data['Date'] = pd.to_datetime(data['Date']).dt.strftime('%m/%d/%Y') 
    data['Season'] = data.Month.map(month_to_season)
    data['Weekend'] = data.Day.map(day_to_weekend)
    
    return data
    
SD = SD_final(California_Load_Demand)

# Remove any vlaues that are within four standard deviations form the mean

def remove_outliers(data):

    del data['Date']
    
    for value in data:
        mean_feat = data[value].mean()
        dev = data[value].std()
        upper = mean_feat + 4 * dev
        lower = mean_feat - 4 * dev
        data.loc[data[value] > upper, value] = np.nan
        data.loc[data[value] < lower, value] = np.nan

        return data

NY = remove_outliers(NY)
PA = remove_outliers(PA)
SD = remove_outliers(SD)
SF = remove_outliers(SF)
VA = remove_outliers(VA)

# Drop null values

def drop(data):
    

    data = data.dropna()
    data = data.reset_index()  

    return data

NY = drop(NY)
PA = drop(PA)
SD = drop(SD)
SF = drop(SF)
VA = drop(VA)


# Write the final dataframe to a csv and save 

NY.to_csv(r'Project/Data/Processed/NY.csv', index=False) 
print('New York Data Processed')

VA.to_csv(r'Project/Data/Processed/VA.csv', index=False) 
print('Vancouver Data Processed')

PA.to_csv(r'Project/Data/Processed/PA.csv', index=False) 
print('Pittsburgh Data Processed')

SD.to_csv(r'Project/Data/Processed/SD.csv', index=False) 
print('San Deigo Data Processed')

SF.to_csv(r'Project/Data/Processed/SF.csv', index=False)
print('San Francisco Data Processed')