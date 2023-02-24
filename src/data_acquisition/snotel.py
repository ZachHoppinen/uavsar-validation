import metloom
import pandas as pd
from snotel_extract import get_snotel_data
from datetime import datetime

snotels = {'bogus':'978:ID:SNTL', 'mores':'637:ID:SNTL' , 'jackson': '550:ID:SNTL',\
            'banner': '312:ID:SNTL'}

dates = (datetime(2019, 12, 1), datetime(2021, 4, 1))

dfs = []

for name, site_id in snotels.items():
    df = get_snotel_data(name, site_id, dates)
    df['lon'] = df['geometry'].x
    df['lat'] = df['geometry'].y
    df['elev'] = df['geometry'].z
    df = df.drop(['geometry'], axis = 1)
    dfs.append(df)

df = pd.concat(dfs, axis = 0)

df.to_parquet('/bsuhome/zacharykeskinen/uavsar-validation/data/insitu/snotel.parq')