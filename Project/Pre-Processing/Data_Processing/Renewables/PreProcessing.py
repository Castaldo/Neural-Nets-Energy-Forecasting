import pandas as pd
import numpy as np

Austria_Weather = pd.read_csv("Project/Data/Unprocessed/Weather/AustriaWeather.csv")
Germany_Weather = pd.read_csv("Project/Data/Unprocessed/Weather/GermanWeather.csv")
France_Weather = pd.read_csv("Project/Data/Unprocessed/Weather/FranceWeather.csv")

# Maps the weekend to the weekdays and seasons to the months to apply over the database

seasons = [1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 1]
month_to_season = dict(zip(range(1,13), seasons))

# Selects Austria, Germany, and France from the European renewables datatset

Austria_Renewable = pd.read_csv("Project/Data/Unprocessed/Energy/Europe_Renewables.csv",
                    usecols=(lambda s: s.startswith('utc') | s.startswith('AT')),
                    parse_dates=[0], index_col=0)

Austria_Renewable = Austria_Renewable[['AT_solar_generation_actual', 'AT_wind_onshore_generation_actual']]

Germany_Renewable = pd.read_csv("Project/Data/Unprocessed/Energy/Europe_Renewables.csv",
                    usecols=(lambda s: s.startswith('utc') | s.startswith('DE')),
                    parse_dates=[0], index_col=0)      

Germany_Renewable = Germany_Renewable[['DE_solar_generation_actual', 'DE_wind_generation_actual']]                  


France_Renewable = pd.read_csv("Project/Data/Unprocessed/Energy/Europe_Renewables.csv",
                   usecols=(lambda s: s.startswith('utc') | s.startswith('FR')),
                    parse_dates=[0], index_col=0)    

France_Renewable = France_Renewable[['FR_solar_generation_actual', 'FR_wind_onshore_generation_actual']]

# Selects Austria, Germany, and France from the European weather datatset

Austria_Wind_Solar = pd.read_csv("Project/Data/Unprocessed/Renewable/Austria.csv",
                                parse_dates=[0], index_col=0)

Germany_Wind_Solar = pd.read_csv("Project/Data/Unprocessed/Renewable/Germany.csv",
                                parse_dates=[0], index_col=0)

France_Wind_Solar = pd.read_csv("Project/Data/Unprocessed/Renewable/France.csv",
                               parse_dates=[0], index_col=0)

Austria_Wind_Solar = Austria_Wind_Solar[['AT_windspeed_10m', 'AT_temperature', 'AT_radiation_direct_horizontal', 'AT_radiation_diffuse_horizontal']]

# Selects data only from the year 2016
# Creates index and standardizes naming conventions

def clean_renewable_data(data):
    
    data = data.loc[data.index.year == 2016, :]
    data.reset_index(inplace=True)
    data = data.rename(columns={"utc_timestamp": "Date"})
    data['Date'] = data['Date'].astype('datetime64[ns]')
    data.columns = data.columns.str.replace(r'(^.*solar.*$)', 'Solar')
    data.columns = data.columns.str.replace(r'(^.*wind.*$)', 'Wind')
    data['Index'] = range(1, len(data) + 1)
    data = data.round(2)

    return data 

Austria_Renewable = clean_renewable_data(Austria_Renewable)
Germany_Renewable = clean_renewable_data(Germany_Renewable)
France_Renewable = clean_renewable_data(France_Renewable)

# Cleans weather data
# Creates index and standardizes naming conventions
# Genereates the mean value for all the data collected from each region

def clean_weather_data(weather_data):
    
    weather_data = weather_data.rename(columns={"# Date": "Date"})
    weather_data['Date'] = weather_data['Date']  + ' ' + weather_data['UT time']
    weather_data['Date'] = weather_data['Date'].str.replace(r'\b24:00\b', '00:00')
    weather_data['Date'] = weather_data['Date'].astype('datetime64[ns]')
    
    weather_data = weather_data.groupby(weather_data.Date).mean()
    weather_data['Index'] = weather_data.reset_index(inplace=True)
    weather_data['Index'] = range(1, len(weather_data) + 1)

    return weather_data 

Austria_Weather = clean_weather_data(Austria_Weather)
Germany_Weather = clean_weather_data(Germany_Weather)
France_Weather = clean_weather_data(France_Weather)

# Creates index and standardizes naming conventions
# Adds temporal columns

def clean_wind_solar(wind_solar_data): 

    wind_solar_data.reset_index(inplace=True)
    wind_solar_data.columns = wind_solar_data.columns.str.replace(r'(^.*windspeed_.*$)', 'Wind Velocity (10m)')
    wind_solar_data.columns = wind_solar_data.columns.str.replace(r'(^.*temperature.*$)', 'Temperature')
    wind_solar_data.columns = wind_solar_data.columns.str.replace(r'(^.*direct.*$)', 'Atmospheric Horizontal Radiation')
    wind_solar_data.columns = wind_solar_data.columns.str.replace(r'(^.*diffuse.*$)', 'Atmospheric Ground Radiation')

    wind_solar_data = wind_solar_data.rename(columns={"utc_timestamp": "Date"})
    wind_solar_data['Date'] = wind_solar_data['Date'].astype('datetime64[ns]')
    wind_solar_data['Index'] = range(1, len(wind_solar_data) + 1)

    wind_solar_data ['Date'] = wind_solar_data['Date'].astype('datetime64')
    wind_solar_data ['Month'] = wind_solar_data['Date'].dt.month
    wind_solar_data ['Day'] = wind_solar_data['Date'].dt.dayofweek
    wind_solar_data ['Hour'] = wind_solar_data['Date'].dt.hour
    
    return wind_solar_data

Austria_Wind_Solar = clean_wind_solar(Austria_Wind_Solar)
France_Wind_Solar = clean_wind_solar(France_Wind_Solar)

# Standardizes German naming conventions
# Renames German columns to standardize with French and Austrian dataset
# Uses French dataset to add temporal values

def clean_german(data, France):
    
    data.reset_index(inplace=True)
    data = data[['lat', 'lon', 'v2', 'SWTDN', 'SWGDN', 'T']]
    data = data.rename(columns={"v2": "Wind Velocity (10m)"})
    data = data.rename(columns={"SWTDN": "Atmospheric Horizontal Radiation"})
    data = data.rename(columns={"SWGDN": "Atmospheric Ground Radiation"})
    data = data.rename(columns={"T": "Temperature"})
    data = data.rename(columns={"timestamp": "Date"})

    data['Index'] = range(1, len(data) + 1)
    del data['lat']
    del data['lon']
    
    France['Date'] = France['Date'].astype('datetime64')
    y = France['Date'].dt.month
    z = France['Date'].dt.dayofweek
    a = France['Date'].dt.hour

    data['Month'] = y
    data['Day'] = z
    data['Hour'] = a
    
    return data
    
Germany_Wind_Solar = clean_german(Germany_Wind_Solar, France_Wind_Solar)

# Merge the three datastes on the index column

def merge(Weather, WS, Country):
    
    Merge = pd.merge(Weather, WS, on= "Index", how="left")
    Country = pd.merge(Country, Merge , on = "Index", how = "left")
    
    return Country

Austria = merge(Austria_Weather, Austria_Wind_Solar, Austria_Renewable)
Germany = merge(Germany_Weather, Germany_Wind_Solar, Germany_Renewable)
France = merge(France_Weather, France_Wind_Solar, France_Renewable)

# Adds Seasonal parameter 

def add_season(data):

        data['Season'] = data.Month.map(month_to_season)
        return data

Austria = add_season(Austria)
France = add_season(France)
Germany = add_season(Germany)

# Delete repeat columns and drop all nulls

def clean_finalframe(final_data):
    
    del final_data["Temperature_x"]
    del final_data["Index"]
    del final_data["Date_y"]
    final_data.columns = final_data.columns.str.replace(r'(^.*Date.*$)', 'Date')
    final_data = final_data.dropna()
    
    return final_data

Austria = Austria.set_index('Date')
Germany = Germany.set_index('Date_x')
France = France.set_index('Date')

Austria = clean_finalframe(Austria)
France = clean_finalframe(France)

# Write to CSV

Austria.to_csv(r'Project/Data/Processed/Autria.csv', index=False) 
print('Austria Data Clean')

Germany.to_csv(r'Project/Data/Processed/Germany.csv', index=False) 
print('Germany Data Clean')

France.to_csv(r'Project/Data/Processed/France.csv', index=False)
print('France Data Clean')
