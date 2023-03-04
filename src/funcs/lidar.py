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