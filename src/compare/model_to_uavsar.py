import xarray as xr
import rioxarray as rxa
from pathlib import Path

# first make datasets of modeled swe difference for each flight pair
model = xr.open_dataset('/bsuhome/zacharykeskinen/scratch/data/uavsar/ncs/model/model.re.nc')
model = model.rio.write_crs('EPSG:4326')

ncs_dir = Path('/bsuscratch/zacharykeskinen/data/uavsar/ncs/')

for fp in ncs_dir.glob('*.sd.nc'):

    if Path(f'/bsuhome/zacharykeskinen/scratch/data/uavsar/ncs/model/{fp.stem}.model.nc').exists():
        continue

    print(fp)

    ds = xr.open_dataset(fp)

    diff_model = model.sel(time = ds.attrs['time2']) - model.sel(time = ds.attrs['time1'])
    diff_model['uavsar_dSD_int'] = ds['sd_delta_int'].rio.reproject_match(diff_model['swe'])
    if 'unw' in ds.data_vars:
        diff_model['uavsar_dSD_unw'] = ds['sd_delta_unw'].rio.reproject_match(diff_model['swe'])
    diff_model.attrs = ds.attrs
    diff_model = diff_model.rio.clip_box(*ds.rio.bounds())

    diff_model.to_netcdf(f'/bsuhome/zacharykeskinen/scratch/data/uavsar/ncs/model/{fp.stem}.model.nc')

# loop through each flight pair and calculate rmse, r2, plot