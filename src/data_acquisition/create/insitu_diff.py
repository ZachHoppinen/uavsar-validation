from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import rioxarray as rxa
import matplotlib.pyplot as plt
import shapely
from itertools import combinations

from uavsar_pytools.snow_depth_inversion import phase_from_depth, depth_from_phase

ncs_dir = Path('/bsuhome/zacharykeskinen/scratch/data/uavsar/ncs')
ds = xr.open_dataset(ncs_dir.joinpath('final_uv_geom_atm_veg_model_v1.nc')).isel(x = slice(0, -1, 100), y = slice(0, -1, 100))
ds = ds.sortby('time1')

data_dir = Path('/bsuhome/zacharykeskinen/uavsar-validation/data')
insitu_dir = data_dir.joinpath('insitu')
insitu = pd.read_parquet(insitu_dir.joinpath('all_insitu.parq'))
insitu = gpd.GeoDataFrame(insitu, geometry=gpd.points_from_xy(insitu.lon, insitu.lat), crs="EPSG:4326")
ds_dates = pd.to_datetime(ds.time.data).normalize()
ds_dates = [[d + pd.Timedelta(f'{i} days') for i in range (-3, 3)] for d in ds_dates]
ds_dates = [item for sublist in ds_dates for item in sublist]
snowex = insitu[insitu.datasource == 'SnowEx'].loc[pd.to_datetime(insitu[insitu.datasource == 'SnowEx'].index).normalize().isin(ds_dates), :]
snotels = insitu[insitu.datasource == 'NRCS'].loc[pd.to_datetime(insitu[insitu.datasource == 'NRCS'].index).normalize().isin(pd.to_datetime(ds.time.data).normalize()), :]
insitu = pd.concat([snowex, snotels])

res = pd.DataFrame()
i = 0
# tolerance around each site 5 * 100 m\n",
tol = 5* 0.00090009
for name, df in insitu.groupby('site_name'):
    if len(df) < 2:
        continue

    print(name)
    for r1, r2 in combinations(df.index, 2):

        r1, r2 = sorted([r1, r2])
        try:
            t2 = pd.to_datetime(ds.time2.sel(time1 = r1, method = 'nearest', tolerance = '2 days').data.ravel()[0]).date()
        except KeyError:
            continue
        if not r2 + pd.Timedelta('3 days') > t2 or not r2 - pd.Timedelta('3 days') < t2:
            continue
        res.loc[i, 'time1'] = pd.to_datetime(ds.time1.sel(time1 = r1, method = 'nearest', tolerance = '2 days').data.ravel()[0])
        res.loc[i, 'time2'] = pd.to_datetime(ds.time2.sel(time1 = r1, method = 'nearest', tolerance = '2 days').data.ravel()[0])
        res.loc[i, 'field_time1'] = r1
        res.loc[i, 'field_time2'] = r2

        row1 = df.loc[r1]
        row2 = df.loc[r2]

        res.loc[i, 'dSWE'] = row2['SWE'] - row1['SWE']
        res.loc[i, 'dSD'] = row2['SD'] - row1['SD']
        for k in ['datasource', 'site_name', 'lon','lat','elev','canopy',\
            'pit_fp', 'ground_rough','ground_veg', 'canopy']:
            res.loc[i, k] = row1[k]
        
        for k in ['SWE', 'SD', 'temp', 'SWE-A','SWE-B','density','ground_wet',\
            'all_dry','any_wet']:
            res.loc[i, f't1_{k}'] = row1[k]
            res.loc[i, f't2_{k}'] = row2[k]
        if row1['datasource'] == 'NRCS':
            res.loc[i, f't1_density'] = row1['SWE']/row1['SD']*997
            res.loc[i, f't2_density'] = row2['SWE']/row2['SD']*997

        day_ds = ds.sel(time1 = r1, method = 'nearest', tolerance = '2 days').sel(x = row1.geometry.x, y = row1.geometry.y, method = 'nearest', tolerance = tol)
        # res.loc[i, 'int_phase'] = day_ds['int_phase']
        # res.loc[i, 'unw_phase'] = day_ds['unw']
        res.loc[i, 'inc'] = day_ds['inc']
        res.loc[i, 'cor'] = day_ds['cor']
        res.loc[i, 'tree_perc'] = day_ds['tree_perc']

        i += 1
# res['int_sd'] = depth_from_phase(res['int_phase'].values, res['inc'].values, density = res['t1_density'].values)
# res['unw_sd'] = depth_from_phase(res['unw_phase'].values, res['inc'].values, density = res['t1_density'].values)

res.to_parquet('/bsuhome/zacharykeskinen/uavsar-validation/data/insitu/all_difference.parq')
