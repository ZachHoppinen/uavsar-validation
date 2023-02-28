import xarray as xr
import numpy as np
import rioxarray as rxa
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path

# loop through each flight pair and calculate rmse, r2, plot

re_ds = xr.open_dataset('/bsuscratch/zacharykeskinen/data/uavsar/ncs/052_2021-02-03_2021-02-10.sd.nc')

ncs_dir = Path('/bsuscratch/zacharykeskinen/data/uavsar/ncs')

lidar_dir = Path('/bsuhome/zacharykeskinen/scratch/data/uavsar/ncs/lidar')

for lidar_fp in lidar_dir.glob('*.nc'):
    lidar = xr.open_dataset(lidar_fp)
    re_ds_lidar = re_ds.rio.clip_box(*lidar.rio.bounds())
    lidar = lidar.rio.reproject_match(re_ds_lidar)

    for fp in ncs_dir.glob('*.sd.nc'):
        print(fp)
        ds = xr.open_dataset(fp)

        lidar = lidar.assign({f"{fp.stem.strip('.sd')}_dSD_int": ds['sd_delta_int'].rio.reproject_match(lidar['lidar-sd'])})

        if 'sd_delta_unw' in ds.data_vars:
            lidar = lidar.assign({f"{fp.stem.strip('.sd')}_dSD_unw": ds['sd_delta_unw'].rio.reproject_match(lidar['lidar-sd'])})
            
    lidar.to_netcdf(lidar_fp.with_suffix('.sd.nc'))
