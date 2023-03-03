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