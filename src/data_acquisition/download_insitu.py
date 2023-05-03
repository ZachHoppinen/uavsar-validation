## First get snotel data and save

import metloom
import pandas as pd
from datetime import datetime
from snotel_extract import get_snotel_data

snotels = {'bogus':'978:ID:SNTL', 'mores':'637:ID:SNTL', 'banner': '312:ID:SNTL'}

dates = (datetime(2019, 12, 1), datetime(2021, 4, 1))

dfs = []

for name, site_id in snotels.items():
    print(name)
    df = get_snotel_data(name, site_id, dates)
    df['lon'] = df['geometry'].x
    df['lat'] = df['geometry'].y
    df['elev'] = df['geometry'].z
    df = df.drop(['geometry'], axis = 1)
    dfs.append(df)

df_snotel = pd.concat(dfs, axis = 0)

import warnings; warnings.filterwarnings('ignore', message='.*initial implementation of Parquet.*')
df.to_parquet('/bsuhome/zacharykeskinen/uavsar-validation/data/insitu/snotel.parq')

# next add in snowpits

# snowpit first
import pandas as pd
from pathlib import Path
from pandas.errors import EmptyDataError

full_df = pd.read_parquet('/bsuhome/zacharykeskinen/uavsar-validation/data/insitu/snotel.parq')
full_df = full_df.tz_localize(None, level = 0)
# add 2020, 2021 pits
dir = '/bsuhome/zacharykeskinen/scratch/data/uavsar/snowpits'
fps = {f'{dir}/2020/SNEX20_TS_SP_Summary_SWE_v01.csv': f'{dir}/2020/SNEX20_TS_SP_Summary_Environment_v01.csv',\
        f'{dir}/2021/SNEX21_TS_SP_Summary_SWE_v01.csv': f'{dir}/2021/SNEX21_TS_SP_Summary_Environment_v01.csv'}
pit_path = Path(dir)

for fp, env_fp in fps.items():
    df = pd.read_csv(fp)

    env = pd.read_csv(env_fp)

    df = df.loc[df.PitID.str.contains('ID'), :]
    env = env.loc[env.PitID.str.contains('ID'), :]

    pitIds = df.PitID
    df = df.set_index('PitID')
    env = env.set_index('PitID')

    for id, r in df.iterrows():
        if '/2020/' in fp:
            pit_fp = next(Path(next(pit_path.joinpath('2020/pits/').glob(f'{id}*'))).glob('*strat*'))
        elif '/2021/' in fp:
            pit_fp = next(pit_path.joinpath('2021/pits/IDAHO-BoiseRiver', id + '_SNOW_PIT').glob('*strat*'))
        try:
            pit = pd.read_csv(pit_fp, comment = '#', names = ['Top', 'bottom', 'grain-size', 'grain-type','hand-hardness', 'wetness', 'comments'])
            
            df.loc[id, 'pit_fp'] = str(pit_fp)
            if (pit.wetness.dropna() != 'D').any():
                df.loc[id, 'all_dry'] = 0
            else:
                df.loc[id, 'all_dry'] = 1
            
            if pit.wetness[(pit.wetness == 'W')].size == 0:
                df.loc[id, 'any_wet'] = 0
            else:
                df.loc[id, 'any_wet'] = 1
            

        except EmptyDataError:
            pass


    df['lat'] = df.loc[:, 'Latitude (deg)']
    df['lon'] = df.loc[:, 'Longitude (deg)']
    df['SD'] = df.loc[:, 'Snow Depth (cm)'] / 100
    df['SWE'] = df.loc[:, 'SWE (mm)'] / 1000
    df['SWE-A'] = df.loc[:, 'SWE A (mm)'] / 1000
    df['SWE-B'] = df.loc[:, 'SWE B (mm)'] / 1000
    df['density'] = df.loc[:, 'Density Mean (kg/m^3)']
    df['datetime'] = pd.to_datetime(df.loc[:, 'Date/Local Standard Time'].str.split('T').str[0])
    df['site'] = df['Site']
    df['site_name'] = df['Site']
    df.loc[:, 'canopy'] = env['Canopy']
    df.loc[:, 'ground_wet'] = env['Ground Condition']
    df.loc[:, 'ground_rough'] = env['Ground Roughness']
    df.loc[:, 'ground_veg'] = env['Ground Vegetation']
    df.loc[:, 'datasource'] = 'SnowEx'
    
    df = df.set_index(pd.MultiIndex.from_frame(df[['datetime', 'site']]))
    
    df = df[['SWE','SD', 'datasource', 'site_name', 'lon', 'lat' ,'SWE-A', 'SWE-B', 'density', 'canopy', 'ground_wet', 'ground_rough', 'ground_veg',
             'pit_fp', 'all_dry', 'any_wet']]

    full_df = pd.concat([full_df, df], axis = 0)

full_df = full_df.set_index(pd.to_datetime(full_df.index.get_level_values('datetime'), utc = True).date)

full_df = full_df.sort_index()

import warnings; warnings.filterwarnings('ignore', message='.*initial implementation of Parquet.*')
full_df.to_parquet('/bsuhome/zacharykeskinen/uavsar-validation/data/insitu/all_insitu.parq')

## add storm board data

import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

from stats import get_stats

interval_dir = Path('/bsuhome/zacharykeskinen/scratch/data/uavsar/interval')
outdir = Path('/bsuhome/zacharykeskinen/uavsar-validation/data/interval/')

# clean up and combo original data frames

combo = []
for fp in interval_dir.glob('*.csv'):
    print(f'Starting year: {fp.stem}')
    if '20' in fp.name:
        df = pd.read_csv(fp, parse_dates=['Date/Local Standard Time'])
        df['date_t2'] = pd.to_datetime(df['Date/Local Standard Time'].dt.strftime('%Y-%m-%d'))
        df['swe'] = df['AVG SWE (mm)']
        df['hn'] = df['AVG HN (cm)']
        df['time_t2'] = df['Date/Local Standard Time'].dt.strftime('%T')
        df['obs_year'] = '2019-2020'
    elif '21' in fp.name:
        df = pd.read_csv(fp, parse_dates=['Date'])
        df['date_t2'] = df['Date']
        df['swe'] = df['avg SWE (mm)']
        df['hn'] = df['avg HN (cm)']
        df['time_t2'] = df['Local \nTime']
        df['obs_year'] = '2020-2021'
    
    for l in ['A', 'B', 'C']:
        df[f'swe_{l.lower()}'] = df[f'SWE (mm) {l}']
        df[f'hn_{l.lower()}'] = df[f'HN (cm) {l}']
    
    df = df[df.swe != ' ']
    df = df[df.hn != ' ']

    df['swe'] = df.swe.astype(float) / 1000
    df['hn'] = df.hn.astype(float) / 100
    
    df['den'] = df['swe'] / df['hn'] * 997

    df['week'] = df['Week No.'].astype(int)

    gdf = gpd.GeoDataFrame(df, geometry = gpd.points_from_xy(df.Longitude, df.Latitude))
    gdf = gdf.loc[gdf.State == 'ID']
    combo.append(gdf)
combo = pd.concat(combo)
combo = combo.drop(['Date/Local Standard Time', 'Date', 'avg SWE (mm)', 'AVG SWE (mm)', 'Local \nTime', 'AVG HN (cm)','avg HN (cm)', 'avg DEN (kg/m^3)', 'HN (cm) A', 'HN (cm) B',
       'HN (cm) C', 'SWE (mm) A', 'SWE (mm) B', 'SWE (mm) C', 'DEN (kg/m^3) A', 'DEN (kg/m^3) B','DEN (kg/m^3) C', 'Week No.'], axis = 1)

for col in combo.columns:
    if col.lower() == col:
        continue
    combo[col.lower()] = combo[col]
    combo = combo.drop(col, axis = 1)

add_t1_dfs = []
for (year, site), df in combo.groupby(['obs_year', 'site']):
    df['date_t1'] = df['date_t2'].shift(1)
    if site == 'Banner Open':
        df.loc[df.comments == 'placed interval board on 2019-12-18. ','date_t1'] = pd.to_datetime('2019-12-18')
    df['days_between'] = df['date_t2'] - df['date_t1']
    add_t1_dfs.append(df)

combo = pd.concat(add_t1_dfs)
combo = combo.dropna(subset = ['date_t1'], axis = 0)
combo['hn_a'] = combo['hn_a'].astype(float)
combo.parquet('/bsuhome/zacharykeskinen/uavsar-validation/data/insitu/storm_boards.parq')