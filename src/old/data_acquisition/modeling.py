import xarray as xr
from glob import glob
from os.path import join

model_dir = '/bsuhome/zacharykeskinen/scratch/data/uavsar/modeling/lowman'

full_ds = []

for t in ['swed', 'smlt']:
    fs = glob(join(model_dir, f'*{t}*.nc'))
    
    type_dss = []
    for f in fs:
        ds = xr.open_dataset(f)
        type_dss.append(ds)
    
    full_ds.append(xr.merge(type_dss))

ds_swe, ds_mel = full_ds

ds = xr.Dataset()

ds['swe'] = ds_swe['SWED']
ds['melt'] = ds_mel['SMLT']

ds.to_netcdf('/bsuhome/zacharykeskinen/scratch/data/uavsar/ncs/model.nc')