import xarray as xr
import numpy as np
import pandas as pd
import rioxarray as rxa
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

def add_data_vars(fp, lidar):
    ds = xr.open_dataset(fp)

    # get directions and times
    direction = fp.stem.split('_')[0]
    t = pd.to_datetime(ds.attrs['time1'])
    t2 = pd.to_datetime(ds.attrs['time2'])
    snotel_id = snotels[lidar_fp.stem]
    dSWE, dSD = get_snotel_change(df, ds, snotel_id)

    # loop through data vars and add them as time dimension

    DAs = []
    for data_var in ds.data_vars:
        print(data_var)
        data = ds[data_var].rio.reproject_match(lidar['lidar-sd'])
        data = data.expand_dims(time = [t])
        data = data.assign_coords(time2 = ("time", [t2]))
        data = data.assign_coords(snotel_dSWE = ("time", [dSWE]))
        data = data.assign_coords(snotel_dSD = ("time", [dSD]))
        data = data.rename(f"{direction}-{data_var}")
        DAs.append(data)
    
    lidar = xr.merge([lidar, xr.merge(DAs)])

    return lidar

def get_snotel_change(df, ds, site_id):

    t1 = pd.to_datetime(ds.attrs['time1'])
    t2 = pd.to_datetime(ds.attrs['time2'])

    if t1.tz == None:
        t1 = t1.tz_localize('UTC')
    if t2.tz == None:
        t2 = t2.tz_localize('UTC')

    sub = df.loc[t1:t2]

    sub = sub.loc[sub.site == site_id]
    diff = sub[['SWE', 'SD']].iloc[-1] -sub[['SWE', 'SD']].iloc[0]

    return diff

df = pd.read_parquet('/bsuhome/zacharykeskinen/uavsar-validation/data/insitu/snotel.parq')
df = df.reset_index(level= 1)
df = df.sort_index()

re_ds = xr.open_dataset('/bsuscratch/zacharykeskinen/data/uavsar/ncs/052_2021-02-03_2021-02-10.sd.nc')

lidar_dir = Path('/bsuhome/zacharykeskinen/scratch/data/uavsar/ncs/lidar')
ncs_dir = Path('/bsuhome/zacharykeskinen/scratch/data/uavsar/ncs')
dem_dir = Path('/bsuhome/zacharykeskinen/scratch/data/uavsar/lidar/idaho')

prefixes = {'Banner': 'USIDBS', 'Dry_Creek':'USIDDC', 'Mores':'USIDMC'}
snotels = {'Banner': '312:ID:SNTL', 'Dry_Creek':'978:ID:SNTL', 'Mores':'637:ID:SNTL'}

for lidar_fp in lidar_dir.glob('*.nc'):

    if '.sd.' in lidar_fp.name:
        continue

    if Path(lidar_fp.with_suffix('.sd.nc')).exists():
        print(lidar_fp.with_suffix('.sd.nc').name + ' exists.')
        continue

    print(lidar_fp)

    site_id = prefixes[lidar_fp.stem]
    lidar = xr.open_dataset(lidar_fp)

    # add in DEM that didn't work last time
    dem = xr.open_dataarray(next(dem_dir.glob(f'*DEM*{site_id}*.tif')))
    dem = dem.rio.reproject_match(lidar['lidar-sd'])
    dem = dem.squeeze('band')
    lidar['lidar-dem'] = dem

    # clip and reproject lidar
    re_ds_lidar = re_ds.rio.clip_box(*lidar.rio.bounds())
    lidar = lidar.rio.reproject_match(re_ds_lidar)

    lidar_times = lidar.time.data

    # loop through UAVSAR
    for fp in ncs_dir.glob('*sd.nc'):
        print(fp)
        lidar = add_data_vars(fp, lidar)
        print(lidar)

    lidar.attrs['lidar_times'] = [str(t) for t in lidar_times]
                                  
    lidar = lidar.sortby('time')

    lidar.to_netcdf(lidar_fp.with_suffix('.sd.nc'))
