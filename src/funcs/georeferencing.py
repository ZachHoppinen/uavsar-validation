from geopy.distance import geodesic
def resolution_to_meters(resolution_deg, ul_lat):
    """
    Calculation resolution in meters from a resolution in degrees/pixel
    """

    from geopy.distance import geodesic

    resolution_deg = float(resolution_deg)
    ul_lat = float(ul_lat)

    origin = (ul_lat, 0)  # (latitude, longitude) don't confuse
    dist = (ul_lat + 1, 0)

    ms_per_deg = geodesic(origin, dist).meters

    resolution_m = resolution_deg * ms_per_deg

    return resolution_m

import requests
import urllib
import pandas as pd
url = r'https://epqs.nationalmap.gov/v1/json?'
def pd_add_elevation(df, lat_column: str, lon_column: str):
    """
    Query service using lat, lon. add the elevation values as a new column.
    https://gis.stackexchange.com/questions/338392/getting-elevation-for-multiple-lat-long-coordinates-in-python

    Args:
    df: pd.dataframe
    lat_column: name of column
    lon_column: name of column
    """
    elevations = []
    for lat, lon in zip(df[lat_column], df[lon_column]):
                
        # define rest query params
        params = {
            'output': 'json',
            'x': lon,
            'y': lat,
            'units': 'Meters'
        }
        
        # format query string and return query value
        result = requests.get((url + urllib.parse.urlencode(params)))
        #elevations.append(result.json()['USGS_Elevation_Point_Query_Service']['Elevation_Query']['Elevation'])
        #new 2023:
        elevations.append(result.json()['value'])

    df['elev_meters'] = elevations

def triangle(x, y):
    return np.sqrt(x*x + y*y)