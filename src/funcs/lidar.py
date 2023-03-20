import xarray as xr
import numpy as np
import pandas as pd
import rioxarray as rxa
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib as mpl
import pandas as pd
import os
from os.path import join, exists, basename
from glob import glob

def make_site_ds(site: str, lidar_dir: str) -> xr.Dataset:
    """
    Makes a dataset of snow depth, veg height, and DEM for a specific site abbreviation
    in the lidar directory. Returns it reprojected to EPSG4326.
    Args:
    site: Site abbreviation to search for
    lidar_dir: Direction of lidar tiffs
    Returns:
    dataset: xarray dataset of dem, sd, vh for site
    """

    dataset = xr.Dataset()

    for img_type in ['SD', 'VH', 'DEM']:
        files = glob(join(lidar_dir, f'*_{img_type}_*_{site}_*.tif'))
        assert files, f"No files found for {img_type} at {site}"
        
        imgs = []
        for f in files:

            date = pd.to_datetime(basename(f).split('_')[-2])

            img = rxa.open_rasterio(f)

            img = img.squeeze(dim = 'band')
            
            img = img.expand_dims(time = [date])

            imgs.append(img)
    
        dataset['lidar-' + img_type.lower()] = xr.concat(imgs, dim = 'time')

    dataset = dataset.rio.reproject('EPSG:4326')
    
    return dataset

def get_site_dSWE(site_name, d1, d2, df):
    site = df[df.site_name == site_name]
    if not d1.tz:
        d1 = d1.tz_localize('UTC')
    if not d2.tz:
        d2 = d2.tz_localize('UTC')
    site_pair = site.loc[d1:d2, 'SWE']
    if len(site_pair) < 2:
        return np.nan
    dSWE = site_pair.iloc[-1] - site_pair.iloc[0]
    return dSWE

def add_delta_t(ds):
    t_deltas = [pd.Timedelta(t.values) for t in ds.time.diff(dim = 'time')]
    t_deltas.append(np.nan)
    ds = ds.assign_coords({'t_delta': ('time', t_deltas)})

    return ds

def add_dSWEs_dTime(ds, df):

    ds = add_delta_t(ds)

    dSWEs = []
    for sub_t in ds.time:
        start = pd.to_datetime(ds['time'].sel(time = sub_t).values)
        end = pd.to_datetime(start + ds['t_delta'].sel(time = sub_t).values)
        dSWE = get_site_dSWE(ds.attrs['site'].lower(), start, end, df)
        dSWEs.append(dSWE)
    
    ds = ds.assign_coords({'dSWE': ('time', dSWEs)})

    return ds

import numpy as np
import pandas as pd
import xarray as xr
import rioxarray as rxa

from pathlib import Path
from stats import get_stats, clean_xs_ys

from scipy.ndimage import gaussian_filter

from uavsar_pytools.snow_depth_inversion import depth_from_phase, phase_from_depth

def get_r(ds, low, high, condition):
    """
    calculates pearson r from ds uavsar-sd and lidar-sd filted between low and high
    on DA condition.

    Args:
    ds - xarray Dataset with 'lidar-sd', '232-sd_delta_int
    low - float of low end
    high - float of high end
    condition - xarray dataset of same shape as ds (x, y)

    Returns:
    r - pearson r correlation
    n - number of valid observations
    """
    # tolerance around snotel ~100 m
    tol = 0.00090009

    # loop through dataset times
    for ds_t in ds.time:

        # select out our time
        sub = ds.sel(time = ds_t)

        # if all nans continue
        if sub['232-int'].sum() == 0:
            ds['232-sd_delta_int'].loc[dict(time = ds_t)] = np.nan
            continue
        
        # if SWE increased in this time period
        if sub['snotel_dSWE'] > 0:
            # get current phase
            cur_phase = sub['232-int'].sel(x = slice(sntl_x - tol, sntl_x + tol), y = slice(sntl_y + tol, sntl_y - tol)).mean()
            # get snotel incidence angle
            sntl_inc = sub['232-inc'].sel(x = slice(sntl_x - tol, sntl_x + tol), y = slice(sntl_y + tol, sntl_y - tol)).mean()
            # calculate sd change using 250 density
            snotel_sd_change = sub['snotel_dSWE'] * (997 / 250)
            # calculate phase we think snotel should be at
            sd_phase = phase_from_depth(snotel_sd_change, sntl_inc, density = 250)
            # calculate inversion after changing mean value from current to expect snotel phase
            data = depth_from_phase(sub['232-int'] + (sd_phase - cur_phase), sub['232-inc'], density = 250)
            # smooth
            data.data = gaussian_filter(data, 3)
            # remove values that don't meet condition
            data = data.where((condition > low) & (condition < high))
            # set data
            ds['232-sd_delta_int'].loc[dict(time = ds_t)] = data

        else:
            # if swe didn't increase
            ds['232-sd_delta_int'].loc[dict(time = ds_t)] = np.nan
    
    # sum changes in SD across time
    sum = ds['232-sd_delta_int'].sum(dim = 'time')

    # blur lidar and check condition
    ds['lidar-sd'].loc[dict(time = t)] = gaussian_filter(ds['lidar-sd'].sel(time = t), sigma = 3)
    ds['lidar-sd'].loc[dict(time = t)] = ds['lidar-sd'].sel(time = t).where((condition > low) & (condition < high))

    # remove nans, infinites, and 0s
    xs, ys = clean_xs_ys(ds['lidar-sd'].sel(time = t).values.ravel(), sum.values.ravel())
    xs_tmp = xs[(xs != 0) & (ys != 0)]
    ys = ys[(xs != 0) & (ys != 0)]
    xs = xs_tmp

    # calculate stats on lidar and uavsar
    if len(xs) == 0 or len(ys) == 0:
        return np.nan, np.nan
    rmse, r, n = get_stats(xs, ys)

    return r, n
    