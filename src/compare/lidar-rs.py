import numpy as np
import pandas as pd
import xarray as xr
import rioxarray as rxa
from tqdm import tqdm

from pathlib import Path
from stats import get_stats, clean_xs_ys

from scipy.ndimage import gaussian_filter

from xrspatial import slope, aspect, curvature

snotel_locs = {'Dry_Creek.sd': [-116.09685, 43.76377], 'Banner.sd': [-115.23447, 44.30342], 'Mores.sd': [-115.66588, 43.932]}

import socket
hn = socket.gethostname()
lidar_dir = Path('/bsuhome/zacharykeskinen/scratch/data/uavsar/ncs/lidar')
if hn == 'Zachs-MacBook-Pro.local':
    lidar_dir = Path('/Users/zachkeskinen/Desktop/')

out_dir = Path('/bsuhome/zacharykeskinen/uavsar-validation/results/lidar')

## Slope

res = pd.DataFrame(index = pd.MultiIndex.from_arrays([[],[],[]], names=('stat', 'loc', 'date')))

for lidar_fp in lidar_dir.glob('*.sd.nc'):
    print(lidar_fp)

    lidar = xr.open_dataset(lidar_fp)
    
    if isinstance(lidar.attrs['lidar_times'] , str):
        lidar.attrs['lidar_times'] = [lidar.attrs['lidar_times']]
    sntl_x, sntl_y = snotel_locs[lidar_fp.stem]
    
    for t in lidar.attrs['lidar_times']:
        t = pd.to_datetime(t)

        ds_orig = lidar.sel(time = slice(t - pd.Timedelta('180 days'), t))
        ds_orig = ds_orig.sel(band = 'VV')

        lidar_slope = slope(lidar['lidar-dem'].rio.reproject('EPSG:32611')).rio.reproject_match(lidar['lidar-sd'].sel(time = t))

        delta = 1 # degrees
        for high in tqdm(np.arange(delta, lidar_slope.max(), delta)):
            res = res.sort_index()
            low = high - delta
            
            ds = ds_orig.copy(deep = True)

            lidar_sd = ds['lidar-sd'].sel(time = t).where((lidar_slope > low) & (lidar_slope < high))
            lidar_sd.data = gaussian_filter(lidar_sd, 3)

            ds = ds.sel(time = ds.snotel_dSWE > 0)
            
            sum = ds['232-sd_delta_int'].sum(dim = 'time')
            sum.data = gaussian_filter(sum, 3)

            xs, ys = clean_xs_ys(lidar_sd.values.ravel(), sum.values.ravel(), clean_zeros = True)

            if len(xs) < 2 or len(ys) < 2:
                print(f"No values for {low}-{high}.")
                rmse, r, n =  np.nan, np.nan, np.nan
            else:
                rmse, r, n = get_stats(xs, ys)
            
            loc = lidar_fp.name.replace('.sd.nc', '')
            date = t.strftime('%Y-%m-%d')

            for stat_name, stat in zip(['n', 'r', 'rmse'], [n, r, rmse]):
                res.loc[(stat_name, loc, date), f'{low}-{high}'] = stat

res.to_csv(out_dir.joinpath('slope.csv'))

# ## Aspect

res = pd.DataFrame(index = pd.MultiIndex.from_arrays([[],[],[]], names=('stat', 'loc', 'date')))

for lidar_fp in lidar_dir.glob('*.sd.nc'):
    print(lidar_fp)

    lidar = xr.open_dataset(lidar_fp)
    
    if isinstance(lidar.attrs['lidar_times'] , str):
        lidar.attrs['lidar_times'] = [lidar.attrs['lidar_times']]
    sntl_x, sntl_y = snotel_locs[lidar_fp.stem]
    
    for t in lidar.attrs['lidar_times']:
        t = pd.to_datetime(t)

        ds_orig = lidar.sel(time = slice(t - pd.Timedelta('180 days'), t))
        ds_orig = ds_orig.sel(band = 'VV')

        cond = aspect(lidar['lidar-dem'].rio.reproject('EPSG:32611')).rio.reproject_match(lidar['lidar-sd'].sel(time = t))

        delta = 1 # degrees
        for high in tqdm(np.arange(delta, cond.max(), delta)):
            res = res.sort_index()
            low = high - delta
            
            ds = ds_orig.copy(deep = True)

            lidar_sd = ds['lidar-sd'].sel(time = t).where((cond > low) & (cond < high))
            lidar_sd.data = gaussian_filter(lidar_sd, 3)

            ds = ds.sel(time = ds.snotel_dSWE > 0)
            
            sum = ds['232-sd_delta_int'].sum(dim = 'time')
            sum.data = gaussian_filter(sum, 3)

            xs, ys = clean_xs_ys(lidar_sd.values.ravel(), sum.values.ravel(), clean_zeros = True)

            if len(xs) < 2 or len(ys) < 2:
                print(f"No values for {low}-{high}.")
                rmse, r, n =  np.nan, np.nan, np.nan
            else:
                rmse, r, n = get_stats(xs, ys)
            
            loc = lidar_fp.name.replace('.sd.nc', '')
            date = t.strftime('%Y-%m-%d')

            for stat_name, stat in zip(['n', 'r', 'rmse'], [n, r, rmse]):
                res.loc[(stat_name, loc, date), f'{low}-{high}'] = stat

res.to_csv(out_dir.joinpath('aspect.csv'))

# ## Incidence Angle

res = pd.DataFrame(index = pd.MultiIndex.from_arrays([[],[],[]], names=('stat', 'loc', 'date')))

for lidar_fp in lidar_dir.glob('*.sd.nc'):
    print(lidar_fp)

    lidar = xr.open_dataset(lidar_fp)
    
    if isinstance(lidar.attrs['lidar_times'] , str):
        lidar.attrs['lidar_times'] = [lidar.attrs['lidar_times']]
    sntl_x, sntl_y = snotel_locs[lidar_fp.stem]
    
    for t in lidar.attrs['lidar_times']:
        t = pd.to_datetime(t)

        ds_orig = lidar.sel(time = slice(t - pd.Timedelta('180 days'), t))
        ds_orig = ds_orig.sel(band = 'VV')

        cond = np.rad2deg(lidar['232-inc'].isel(time = 0))

        delta = 1 # degrees
        for high in tqdm(np.arange(delta, cond.max(), delta)):
            res = res.sort_index()
            low = high - delta
            
            ds = ds_orig.copy(deep = True)

            lidar_sd = ds['lidar-sd'].sel(time = t).where((cond > low) & (cond < high))
            lidar_sd.data = gaussian_filter(lidar_sd, 3)

            ds = ds.sel(time = ds.snotel_dSWE > 0)
            
            sum = ds['232-sd_delta_int'].sum(dim = 'time')
            sum.data = gaussian_filter(sum, 3)

            xs, ys = clean_xs_ys(lidar_sd.values.ravel(), sum.values.ravel(), clean_zeros = True)

            if len(xs) < 2 or len(ys) < 2:
                print(f"No values for {low}-{high}.")
                rmse, r, n =  np.nan, np.nan, np.nan
            else:
                rmse, r, n = get_stats(xs, ys)
            
            loc = lidar_fp.name.replace('.sd.nc', '')
            date = t.strftime('%Y-%m-%d')

            for stat_name, stat in zip(['n', 'r', 'rmse'], [n, r, rmse]):
                res.loc[(stat_name, loc, date), f'{low}-{high}'] = stat

res.to_csv(out_dir.joinpath('inc.csv'))

# tree height

res = pd.DataFrame(index = pd.MultiIndex.from_arrays([[],[],[]], names=('stat', 'loc', 'date')))

for lidar_fp in lidar_dir.glob('*.sd.nc'):
    print(lidar_fp)

    lidar = xr.open_dataset(lidar_fp)
    
    if isinstance(lidar.attrs['lidar_times'] , str):
        lidar.attrs['lidar_times'] = [lidar.attrs['lidar_times']]
    sntl_x, sntl_y = snotel_locs[lidar_fp.stem]
    
    for t in lidar.attrs['lidar_times']:
        t = pd.to_datetime(t)

        ds_orig = lidar.sel(time = slice(t - pd.Timedelta('180 days'), t))
        ds_orig = ds_orig.sel(band = 'VV')

        cond = lidar['lidar-vh'].sel(time = t)

        delta = 1 # meters
        for high in tqdm(np.arange(delta, cond.max(), delta)):
            res = res.sort_index()
            low = high - delta
            
            ds = ds_orig.copy(deep = True)

            lidar_sd = ds['lidar-sd'].sel(time = t).where((cond > low) & (cond < high))
            lidar_sd.data = gaussian_filter(lidar_sd, 3)

            ds = ds.sel(time = ds.snotel_dSWE > 0)
            
            sum = ds['232-sd_delta_int'].sum(dim = 'time')
            sum.data = gaussian_filter(sum, 3)

            xs, ys = clean_xs_ys(lidar_sd.values.ravel(), sum.values.ravel(), clean_zeros = True)

            if len(xs) < 2 or len(ys) < 2:
                pass
            
            loc = lidar_fp.name.replace('.sd.nc', '')
            date = t.strftime('%Y-%m-%d')

            for stat_name, stat in zip(['n', 'r', 'rmse'], [n, r, rmse]):
                res.loc[(stat_name, loc, date), f'{low}-{high}'] = stat

res.to_csv(out_dir.joinpath('tree_height.csv'))

# tree density

res = pd.DataFrame(index = pd.MultiIndex.from_arrays([[],[],[]], names=('stat', 'loc', 'date')))

for lidar_fp in lidar_dir.glob('*.sd.nc'):
    print(lidar_fp)

    lidar = xr.open_dataset(lidar_fp)
    
    if isinstance(lidar.attrs['lidar_times'] , str):
        lidar.attrs['lidar_times'] = [lidar.attrs['lidar_times']]
    sntl_x, sntl_y = snotel_locs[lidar_fp.stem]
    
    for t in lidar.attrs['lidar_times']:
        t = pd.to_datetime(t)

        ds_orig = lidar.sel(time = slice(t - pd.Timedelta('180 days'), t))
        ds_orig = ds_orig.sel(band = 'VV')

        tree_height = lidar['lidar-vh'].sel(time = t)
        trees = xr.where(tree_height > 5, 0, 1).where(~tree_height.isnull())
        tree_density = trees.rolling(x = 5, y = 5).sum()/25

        delta = 0.01 # percentage (10 % )
        for high in tqdm(np.arange(delta, tree_density.max(), delta)):
            res = res.sort_index()
            low = high - delta
            
            ds = ds_orig.copy(deep = True)

            lidar_sd = ds['lidar-sd'].sel(time = t).where((tree_density > low) & (tree_density < high))
            lidar_sd.data = gaussian_filter(lidar_sd, 3)

            ds = ds.sel(time = ds.snotel_dSWE > 0)
            
            sum = ds['232-sd_delta_int'].sum(dim = 'time')
            sum.data = gaussian_filter(sum, 3)

            xs, ys = clean_xs_ys(lidar_sd.values.ravel(), sum.values.ravel(), clean_zeros = True)

            if len(xs) < 2 or len(ys) < 2:
                print(f"No values for {low}-{high}.")
                rmse, r, n =  np.nan, np.nan, np.nan
            else:
                rmse, r, n = get_stats(xs, ys)
            
            loc = lidar_fp.name.replace('.sd.nc', '')
            date = t.strftime('%Y-%m-%d')

            for stat_name, stat in zip(['n', 'r', 'rmse'], [n, r, rmse]):
                res.loc[(stat_name, loc, date), f'{low}-{high}'] = stat

res.to_csv(out_dir.joinpath('tree_density.csv'))
