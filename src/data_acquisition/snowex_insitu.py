# snowpit first
import pandas as pd

full_df = pd.read_parquet('/bsuhome/zacharykeskinen/uavsar-validation/data/insitu/snotel.parq')

# add 2020 pits

fps = ['/bsuhome/zacharykeskinen/scratch/data/uavsar/snowpits/2020/SNEX20_TS_SP_Summary_SWE_v01.csv',\
        '/bsuhome/zacharykeskinen/scratch/data/uavsar/snowpits/2021/SNEX21_TS_SP_Summary_SWE_v01.csv']

for fp in fps:
    df = pd.read_csv(fp)

    df = df.loc[df.PitID.str.contains('ID'), :]

    df['lat'] = df.loc[:, 'Latitude (deg)']
    df['lon'] = df.loc[:, 'Longitude (deg)']
    df['SD'] = df.loc[:, 'Snow Depth (cm)'] / 100
    df['SWE'] = df.loc[:, 'SWE (mm)'] / 1000
    df['SWE-A'] = df.loc[:, 'SWE A (mm)'] / 1000
    df['SWE-B'] = df.loc[:, 'SWE B (mm)'] / 1000
    df['density'] = df.loc[:, 'Density Mean (kg/m^3)']
    df['datetime'] = pd.to_datetime(df.loc[:, 'Date/Local Standard Time'])
    df['site'] = df['Site']
    df['site_name'] = df['Site']
    df['datasource'] = 'SnowEx'

    df = df.set_index(pd.MultiIndex.from_frame(df[['datetime', 'site']]))

    df = df[['SWE','SD', 'datasource', 'site_name', 'lon', 'lat' ,'SWE-A', 'SWE-B', 'density']]

    full_df = pd.concat([full_df, df], axis = 0)

full_df = full_df.set_index(pd.to_datetime(full_df.index.get_level_values('datetime'), utc = True).date)

full_df = full_df.sort_index()

import warnings; warnings.filterwarnings('ignore', message='.*initial implementation of Parquet.*')
full_df.to_parquet('/bsuhome/zacharykeskinen/uavsar-validation/data/insitu/all_insitu.parq')