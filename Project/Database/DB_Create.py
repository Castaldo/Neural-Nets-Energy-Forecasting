import psycopg2

# Set Database parameters and connect

t_host = "localhost"
t_port = "5432"
t_dbname = "Neural_Network_Predictions"
t_user = "postgres"
t_pw = "Strawberry327"
db_conn = psycopg2.connect(host=t_host, port=t_port, dbname=t_dbname, user=t_user, password=t_pw)
db_cursor = db_conn.cursor()

# Functions to Postgres Database Tables

def building_tables():
    
    db_cursor.execute("""CREATE TABLE building_predictions(
        index integer,
        model char(20),
        predicted float,
        actual float
        )
        """)
        
    db_conn.commit()

    print('Building Tables Created')

def city_tables():
    
    db_cursor.execute("""CREATE TABLE  city_predictions(
        index integer,
        model char(20),
        predicted float,
        actual float
        )
        """)
        
    db_conn.commit()

    print('City Table Created')

def wind_tables():
    
    db_cursor.execute("""CREATE TABLE wind_predictions(
        index integer,
        model char(20),
        predicted float,
        actual float
        )
        """)

    db_conn.commit()

    print('Wind Table Created')

def solar_tables():
    
    db_cursor.execute("""CREATE TABLE solar_predictions(
        index integer,
        model char(20),
        predicted float,
        actual float
        )
        """)
        
    db_conn.commit()

    print('Solar Table Created')

building_tables()
city_tables()
wind_tables()
solar_tables()