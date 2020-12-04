import psycopg2

# Set Database parameters

t_host = "localhost"
t_port = "5432"
t_dbname = "Neural_Network_Predictions"
t_user = "postgres"
t_pw = "Strawberry327"
db_conn = psycopg2.connect(host=t_host, port=t_port, dbname=t_dbname, user=t_user, password=t_pw)
db_cursor = db_conn.cursor()

# Writes the CSV's of predicted to the Postgres DB

def write_building(string):
    try:
        with open('Project/Data/Predicted/' + string + '.csv', 'r') as f:
            next(f)
            db_cursor.copy_from(f, 'building_predictions', sep=',')
        db_conn.commit()
        
    except Exception as e:
        print(e)
        db_conn.close

# Writes the predicted values of the Cities CSV's to the Postgres DB

def write_city(string):
    try:
        with open('Project/Data/Predicted/' + string + '.csv', 'r') as f:
            next(f)
            db_cursor.copy_from(f, 'city_predictions', sep=',')
        db_conn.commit()
        
    except Exception as e:
        print(e)
        db_conn.close

# Writes the predicted values of the Solar CSV's to the Postgres DB

def write_solar(string):
    try:
        with open('Project/Data/Predicted/' + string + '.csv', 'r') as f:
            next(f)
            db_cursor.copy_from(f, 'solar_predictions', sep=',')
        db_conn.commit()
        
    except Exception as e:
        print(e)
        db_conn.close

# Writes the predicted values of the Wind CSV's to the Postgres DB

def write_wind(string):
    try:
        with open('Project/Data/Predicted/' + string + '.csv', 'r') as f:
            next(f)
            db_cursor.copy_from(f, 'wind_predictions', sep=',')
        db_conn.commit()
        
    except Exception as e:
        print(e)
        db_conn.close