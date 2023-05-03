import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm

from stats import get_stats

interval_dir = Path('/bsuhome/zacharykeskinen/scratch/data/uavsar/interval')
ncs_dir = Path('/bsuscratch/zacharykeskinen/data/uavsar/ncs/')
outdir = Path('/bsuhome/zacharykeskinen/uavsar-validation/data/interval/')

class uavsar_fp:

    def __init__(self, fp, f1, f2):
        self.fp = fp
        self.f1 = f1
        self.f2 = f2


    def check_dates(self, t1, t2):
        if (self.f1 <= t1 + pd.Timedelta('2 days')) & (self.f1 >= t1 - pd.Timedelta('2 days')):
            if (self.f2 <= t2 + pd.Timedelta('2 days')) & (self.f2 >= t2 - pd.Timedelta('2 days')):
                return self

sd_fps = list(ncs_dir.glob('232*.sd.nc'))
sd_fps = [uavsar_fp(fp, pd.to_datetime(fp.name.split('_')[1]), pd.to_datetime(fp.stem.split('_')[2].strip('.sd'))) for fp in sd_fps]

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

# add snotel diffs and elevation

# select only interval board observations with +- 2 days uavsar flights

combo['uv_fp'] = np.nan
for i, r in combo.iterrows():
    sub_set = [fp.check_dates(r.date_t1, r.date_t2) for fp in sd_fps]
    sub_set = list(filter(lambda item: item is not None, sub_set))
    if len(sub_set) == 0:
        continue
    assert len(sub_set) == 1
    combo.loc[i, 'uv_fp'] = sub_set[0].fp

combo = combo.dropna(subset = 'uv_fp')

# loop through and add uavsar data

for i, r in tqdm(combo.iterrows(), total = len(combo)):
    ds = xr.load_dataset(r.uv_fp)

    combo.loc[i, 'inc'] = ds['inc'].sel(x = r.geometry.x, y = r.geometry.y, method = 'nearest')
    for band in ['VV','VH', 'HV','HH']:
        combo.loc[i, 'cor'] = ds['cor'].sel(band = band).sel(x = r.geometry.x, y = r.geometry.y, method = 'nearest')

        combo.loc[i, f'int_hn_{band}'] = ds['sd_delta_int'].sel(band = band).sel(x = r.geometry.x, y = r.geometry.y, method = 'nearest')
        if 'sd_delta_unw' in ds.data_vars:
            combo.loc[i, f'unw_hn_{band}'] = ds['sd_delta_unw'].sel(band = band).sel(x = r.geometry.x, y = r.geometry.y, method = 'nearest')
        
        # tolerance of ~ 100 m in arc degrees
        tol = 0.00090009
        combo.loc[i, f'100_cor_{band}'] = ds['cor'].sel(band = band).sel(x = slice(r.geometry.x - tol, r.geometry.x + tol), y = slice(r.geometry.y + tol, r.geometry.y - tol)).mean()
        combo.loc[i, f'100_int_hn'] = ds['sd_delta_int'].sel(band = band).sel(x = slice(r.geometry.x - tol, r.geometry.x + tol), y = slice(r.geometry.y + tol, r.geometry.y - tol)).mean()
        if 'sd_delta_unw' in ds.data_vars:
            combo.loc[i, f'100_unw_hn_{band}'] = ds['sd_delta_unw'].sel(band = band).sel(x = slice(r.geometry.x - tol, r.geometry.x + tol), y = slice(r.geometry.y + tol, r.geometry.y - tol)).mean()

# save output

combo.to_csv('/bsuhome/zacharykeskinen/uavsar-validation/data/interval/interval_sd_v5.csv')