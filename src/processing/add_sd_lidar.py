import xarray as xr
import numpy as np
import pandas as pd
import rioxarray as rxa
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

# loop through each flight pair and calculate rmse, r2, plot

re_ds = xr.open_dataset('/bsuscratch/zacharykeskinen/data/uavsar/ncs/052_2021-02-03_2021-02-10.sd.nc')

ncs_dir = Path('/bsuscratch/zacharykeskinen/data/uavsar/ncs')

lidar_dir = Path('/bsuhome/zacharykeskinen/scratch/data/uavsar/ncs/lidar')

for lidar_fp in lidar_dir.glob('*.nc'):

    if len(lidar_fp.name.split('.')) > 2:
        continue

    lidar = xr.open_dataset(lidar_fp)
    re_ds_lidar = re_ds.rio.clip_box(*lidar.rio.bounds())
    lidar = lidar.rio.reproject_match(re_ds_lidar)

    ints = {'052': [], '232':[]}
    sd_unws = {'052': [], '232':[]}

    cors = {'052': [], '232':[]}
    incs = {'052': [], '232':[]}

    unws = {'052': [], '232':[]}
    ints = {'052': [], '232':[]}

    lidar_times = lidar.time.data

    for fp in tqdm(ncs_dir.glob('*.sd.nc')):

        if '2020' in fp.name or '-03-' in fp.name:
            continue

        ds = xr.open_dataset(fp)

        t = pd.to_datetime(ds.attrs['time1'])

        int = ds['sd_delta_int'].rio.reproject_match(lidar['lidar-sd'])
        int = int.expand_dims(time = [t])
        int.attrs = ds.attrs

        direction = fp.stem.split('_')[0]
        ints[direction].append(int)

        cor = ds['cor'].rio.reproject_match(lidar['lidar-sd'])
        cor = cor.expand_dims(time = [t])
        cors[direction].append(cor)

        inc = ds['inc'].rio.reproject_match(lidar['lidar-sd'])
        inc = inc.expand_dims(time = [t])
        incs[direction].append(inc)

        if 'unw' in ds.data_vars:
            unw = ds['unw'].rio.reproject_match(lidar['lidar-sd'])
            unw = unw.expand_dims(time = [t])
            unws[direction].append(unw)

        int = ds['int'].rio.reproject_match(lidar['lidar-sd'])
        int = int.expand_dims(time = [t])
        ints[direction].append(int)

        if 'sd_delta_unw' in ds.data_vars:
            unw = ds['sd_delta_unw'].rio.reproject_match(lidar['lidar-sd'])
            unw = unw.expand_dims(time = [t])
            unw.attrs = ds.attrs
            sd_unws[direction].append(unw)
    
    for direction, xlist in ints.items():
        lidar = xr.merge([lidar, xr.concat(xlist, dim = 'time', combine_attrs = 'drop_conflicts').rename(f'dSD_{direction}_int')], combine_attrs = 'drop_conflicts')
    for direction, xlist in sd_unws.items():
        lidar = xr.merge([lidar, xr.concat(xlist, dim = 'time', combine_attrs = 'drop_conflicts').rename(f'dSD_{direction}_unw')], combine_attrs = 'drop_conflicts')
    for direction, xlist in cors.items():
        lidar = xr.merge([lidar, xr.concat(xlist, dim = 'time', combine_attrs = 'drop_conflicts').rename(f'{direction}_cor')], combine_attrs = 'drop_conflicts')
    for direction, xlist in incs.items():
        lidar = xr.merge([lidar, xr.concat(xlist, dim = 'time', combine_attrs = 'drop_conflicts').rename(f'{direction}_inc')], combine_attrs = 'drop_conflicts')
    for direction, xlist in unws.items():
        lidar = xr.merge([lidar, xr.concat(xlist, dim = 'time', combine_attrs = 'drop_conflicts').rename(f'{direction}_unw')], combine_attrs = 'drop_conflicts')
    for direction, xlist in ints.items():
        lidar = xr.merge([lidar, xr.concat(xlist, dim = 'time', combine_attrs = 'drop_conflicts').rename(f'{direction}_int')], combine_attrs = 'drop_conflicts')
    
    lidar.attrs['lidar_times'] = [str(t) for t in lidar_times]
                                  
    lidar = lidar.sortby('time')

    print(lidar.attrs)

    lidar.to_netcdf(lidar_fp.with_suffix('.phase.nc'))
