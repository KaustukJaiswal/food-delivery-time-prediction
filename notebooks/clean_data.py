import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno

def change_column_names(data:pd.DataFrame)-> pd.DataFrame:
    return (data.rename({
 'Delivery_person_ID': 'rider_id',
 'Delivery_person_Age': 'age',
 'Delivery_person_Ratings': 'ratings',
 'Delivery_location_latitude': 'delivery_latitude',
 'Delivery_location_longitude': 'delivery_longitude',
 'Time_Orderd': 'order_time',
 'Time_Order_picked': 'order_picked_time',
 'Weatherconditions':'weather',
 'Road_traffic_density':'traffic',
 'City':'city_type',
 'Time_taken(min)': 'time_taken'},axis=1
).rename(str.lower,axis=1)
)


def clean_lat_long(data:pd.DataFrame, threshold=1):
    location_columns=['restaurant_latitude',
 'restaurant_longitude',
 'delivery_latitude',
 'delivery_longitude']
    return(
        data.assign(**{col: (np.where(data[col]<threshold,np.nan,data[col].values)) for col in location_columns})
    )



def extract_datetime_features(ser):
    date_column=pd.to_datetime(ser,dayfirst=True)
    return (pd.DataFrame(
        {
            "day": date_column.dt.day,
            "month":date_column.dt.month ,
            "year":date_column.dt.year,
            "day_of_week": date_column.dt.day_name(),
            "is_weekend":date_column.dt.day_name().isin(['Saturday','Sunday']).astype(int)
        }
    )
    )
    
    
def time_of_day(time_col):
    return(
        pd.cut(time_col,bins=[0,6,12,17,20,24],labels=['after_midnight','morning','afternoon','evening','night'])
    )

    


def calculate_haversine_distance(df: pd.DataFrame) -> pd.DataFrame:
    location_columns = ['restaurant_latitude',
                        'restaurant_longitude',
                        'delivery_latitude',
                        'delivery_longitude']
    lat1 = df[location_columns[0]]
    lon1 = df[location_columns[1]]
    lat2 = df[location_columns[2]]
    lon2 = df[location_columns[3]]

    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2-lon1
    dlat = lat2-lat1
    a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2
    c = 2*np.arcsin(np.sqrt(a))
    distance = 6371 * c

    return (
        df.assign(distance = distance)
    )

def create_distance_type(data: pd.DataFrame) -> pd.DataFrame:
    return (
        data.assign(
            distance_type = pd.cut(data['distance'], bins = [0, 5, 10, 15, 25], right = False,
                                   labels = ['short', 'medium', 'long', 'very_long'])
        ))


def data_cleaning(data:pd.DataFrame)->pd.DataFrame:
    minors_data=data.loc[data['age'].astype('float')<18]
    six_star_data=data.loc[data['ratings']=='6']
    minors_index=minors_data.index.tolist()
    six_star_index=six_star_data.index.tolist()

    return(
        data
        .drop(columns='id')
        .drop(index=minors_index)
        .drop(index=six_star_index)
        .replace("NaN ",np.nan)
        .assign(
            city_name=lambda x:x['rider_id'].str.split('RES').str[0],
            age= lambda x:x['age'].astype(float),
            ratings=lambda x: x['ratings'].astype(float),
            #locations
            restaurant_latitude = lambda x: x['restaurant_latitude'].abs(),
            restaurant_longitude = lambda x: x['restaurant_longitude'].abs(),
            delivery_latitude = lambda x: x['delivery_latitude'].abs(),
            delivery_longitude = lambda x: x['delivery_longitude'].abs(),
            #order date
            order_date = lambda x: pd.to_datetime(x['order_date'],
                                                  dayfirst = True),
            order_day = lambda x: x['order_date'].dt.day,
            order_month = lambda x: x['order_date'].dt.month,
            order_day_of_week = lambda x: x['order_date'].dt.day_name().str.lower(),
            is_weekend = lambda x: (x['order_date']
                                    .dt.day_name()
                                    .isin(["Saturday", "Sunday"])
                                    .astype(int)),
            # time based columns
            order_time = lambda x: pd.to_datetime(x['order_time'],
                                                  format = "mixed"),
            order_picked_time = lambda x: pd.to_datetime(x['order_picked_time'],
                                                         format= "mixed"),
            # time taken to pickup order
            pickup_time_minutes = lambda x: (
                (x['order_picked_time'] - x['order_time'])
                .dt.seconds / 60
            ),
            # hour in which order was placed
            order_time_hour = lambda x: x['order_time'].dt.hour,
            # time of the when order was placed
            order_time_of_day = lambda x: (x['order_time_hour'].pipe(time_of_day)),
            # categorical columns
            weather = lambda x: (
                x['weather']
                .str.replace("conditions ", "")
                .str.rstrip()
                .replace("NaN", np.nan)
            ),
            traffic = lambda x: x['traffic'].str.rstrip().str.lower(),
            type_of_order = lambda x: x['type_of_order'].str.rstrip().str.lower(),
            type_of_vehicle = lambda x: x['type_of_vehicle'].str.rstrip().str.lower(),
            festival = lambda x: x['festival'].str.rstrip().str.lower(),
            city_type = lambda x: x['city_type'].str.rstrip().str.lower(),
            multiple_deliveries = lambda x: x['multiple_deliveries'].astype(float),
            time_taken = lambda x: (x['time_taken']
                                    .str.replace("(min) ", "")
                                    .astype(int)))
        .drop(columns = ['order_time', 'order_picked_time'])
        )

def perform_data_cleaning(data:pd.DataFrame):
        d=(data
        .pipe(change_column_names)
        .pipe(data_cleaning)
        .pipe(clean_lat_long)
        .pipe(calculate_haversine_distance)
        .pipe(create_distance_type))
        d.to_csv("../data/raw/cleaned_data.csv",index=False)
        