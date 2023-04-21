import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import matplotlib.pyplot as plt
from stats import get_stats
from pathlib import Path
from tqdm import tqdm 

interval_dir = Path('/bsuhome/zacharykeskinen/scratch/data/uavsar/interval')
ncs_dir = Path('/bsuscratch/zacharykeskinen/data/uavsar/ncs/')

class uavsar_fp:

    def __init__(self, fp, f1, f2):
        self.fp = fp
        self.f1 = f1
        self.f2 = f2

    def check_dates(self, t1, t2):
        if (self.f1 < t1 + pd.Timedelta('2 days')) & (self.f1 > t1 - pd.Timedelta('2 days')):
            if (self.f2 < t2 + pd.Timedelta('2 days')) & (self.f2 > t2 - pd.Timedelta('2 days')):
                return self

sd_fps = list(ncs_dir.glob('232*.sd.nc'))
sd_fps = [uavsar_fp(fp, pd.to_datetime(fp.name.split('_')[1]), pd.to_datetime(fp.stem.split('_')[2].strip('.sd'))) for fp in sd_fps]

res = []
for fp in interval_dir.glob('*.csv'):
    print(f'Starting year: {fp.stem}')
    if '20' in fp.name:
        df = pd.read_csv(fp, parse_dates=['Date/Local Standard Time'])
        df['date_t2'] = df['Date/Local Standard Time']
    elif '21' in fp.name:
        df = pd.read_csv(fp, parse_dates=['Date'])
        df['date_t2'] = df['Date']
        
    gdf = gpd.GeoDataFrame(df, geometry = gpd.points_from_xy(df.Longitude, df.Latitude))
    gdf = gdf.loc[gdf.Location == 'Boise River Basin']

    for id, df in tqdm(gdf.groupby('PitID')):
        df['date_t1'] = df.shift(1)['date_t2']
        for i, r in df.iterrows():
            sub_set = [fp.check_dates(r.date_t1, r.date_t2) for fp in sd_fps]
            sub_set = list(filter(lambda item: item is not None, sub_set))
            if len(sub_set) == 0:
                continue
            assert len(sub_set) == 1

            gdf.loc[i, 'int_sd_delta'] = np.random.random()
            gdf.loc[i, 'unw_sd_delta'] = np.random.random()
            ds = xr.load_dataset(sub_set[0].fp)

            tol = 0.00090009
            for band in ['VV','VH','HV','HH']:
                gdf.loc[i, f'100_int_sd_delta_{band}'] = ds['sd_delta_int'].sel(band = band).sel(x = slice(r.geometry.x - tol, r.geometry.x + tol), y = slice(r.geometry.y + tol, r.geometry.y - tol)).mean()
                if 'sd_delta_unw' in ds.data_vars:
                    gdf.loc[i, f'100_unw_sd_delta_{band}'] = ds['sd_delta_unw'].sel(band = band).sel(x = slice(r.geometry.x - tol, r.geometry.x + tol), y = slice(r.geometry.y + tol, r.geometry.y - tol)).mean()
                
                tol = 0.00090009* 3
                gdf.loc[i, f'300_int_sd_delta_{band}'] = ds['sd_delta_int'].sel(band = band).sel(x = slice(r.geometry.x - tol, r.geometry.x + tol), y = slice(r.geometry.y + tol, r.geometry.y - tol)).mean()
                if 'sd_delta_unw' in ds.data_vars:
                    gdf.loc[i, f'300_unw_sd_delta_{band}'] = ds['sd_delta_unw'].sel(band = band).sel(x = slice(r.geometry.x - tol, r.geometry.x + tol), y = slice(r.geometry.y + tol, r.geometry.y - tol)).mean()
                
                tol = 0.00090009* 5
                gdf.loc[i, f'500_int_sd_delta_{band}'] = ds['sd_delta_int'].sel(band = band).sel(x = slice(r.geometry.x - tol, r.geometry.x + tol), y = slice(r.geometry.y + tol, r.geometry.y - tol)).mean()
                if 'sd_delta_unw' in ds.data_vars:
                    gdf.loc[i, f'500_unw_sd_delta_{band}'] = ds['sd_delta_unw'].sel(band = band).sel(x = slice(r.geometry.x - tol, r.geometry.x + tol), y = slice(r.geometry.y + tol, r.geometry.y - tol)).mean()
                
                gdf.loc[i, f'nearest_int_sd_delta_{band}'] = ds['sd_delta_int'].sel(band = band).sel(x = r.geometry.x, y = r.geometry.y, method = 'nearest')
                if 'sd_delta_unw' in ds.data_vars:
                    gdf.loc[i, f'nearest_unw_sd_delta_{band}'] = ds['sd_delta_unw'].sel(band = band).sel(x = r.geometry.x, y = r.geometry.y, method = 'nearest')
            
            tol = 0.00090009* 3
            band = 'VV'
            sub = ds.sel(band = band).sel(x = slice(r.geometry.x - tol, r.geometry.x + tol), y = slice(r.geometry.y + tol, r.geometry.y - tol))
            gdf.loc[i, '300_coh'] = sub['cor'].mean()
            gdf.loc[i, f'cor_0.5_300_int_sd_delta_{band}'] = sub.where(sub['cor'] > 0.5)['sd_delta_int'].mean()
            if 'sd_delta_unw' in ds.data_vars:
                gdf.loc[i, f'cor_0.5_300_unw_sd_delta_{band}'] = sub.where(sub['cor'] > 0.5)['sd_delta_unw'].mean()

            gdf.loc[i, f'cor_0.6_300_int_sd_delta_{band}'] = sub.where(sub['cor'] > 0.6)['sd_delta_int'].mean()
            if 'sd_delta_unw' in ds.data_vars:
                gdf.loc[i, f'cor_0.6_300_unw_sd_delta_{band}'] = sub.where(sub['cor'] > 0.6)['sd_delta_unw'].mean()
    
    res.append(gdf)
res = pd.concat(res, ignore_index= True)
res.to_csv('/bsuhome/zacharykeskinen/uavsar-validation/data/interval/interval_sd_v3.csv')