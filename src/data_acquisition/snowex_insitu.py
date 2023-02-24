# snowpit first
import pandas as pd

full_df = pd.read_parquet('/bsuhome/zacharykeskinen/uavsar-validation/data/insitu/snotel.parq')

if 'SNOWDEPTH' in full_df.columns:
    full_df = full_df.drop(['SNOWDEPTH'], axis = 1)

if len(full_df[full_df['datasource'] == 'SnowEx']) < 47:
    fp = '/bsuhome/zacharykeskinen/scratch/data/uavsar/snowpits/SNEX20_TS_SP_Summary_SWE_v01.csv'
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

full_df.to_parquet('/bsuhome/zacharykeskinen/uavsar-validation/data/insitu/snotel.parq')