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
    lc = xr.open_dataset('/bsuscratch/zacharykeskinen/data/uavsar/ncs/landcover/lc.nc')
    lc = lc.rio.write_crs('EPSG:4326')

    diff_model = model.sel(time = ds.attrs['time2']) - model.sel(time = ds.attrs['time1'])
    diff_model['cor'] = ds['cor'].rio.reproject_match(diff_model['swe'])
    diff_model['melt_t1'] = model.sel(time = ds.attrs['time1'])['melt']
    diff_model['melt_t2'] = model.sel(time = ds.attrs['time2'])['melt']
    diff_model['swe_t1'] = model.sel(time = ds.attrs['time1'])['swe']
    diff_model['swe_t2'] = model.sel(time = ds.attrs['time2'])['swe']
    diff_model['uavsar_dSD_int'] = ds['sd_delta_int'].rio.reproject_match(diff_model['swe'])
    lc = lc.squeeze('band')
    lc = lc.rio.reproject_match(diff_model['swe'])
    diff_model['lc'] = lc['band_data']
    if 'unw' in ds.data_vars:
        diff_model['uavsar_dSD_unw'] = ds['sd_delta_unw'].rio.reproject_match(diff_model['swe'])
    diff_model.attrs = ds.attrs
    diff_model = diff_model.rio.clip_box(*ds.rio.bounds())

    vars_list = list(diff_model.data_vars)  
    for var in vars_list:
        try:
            del diff_model[var].attrs['grid_mapping']
        except KeyError:
            pass

    diff_model.to_netcdf(f'/bsuhome/zacharykeskinen/scratch/data/uavsar/ncs/model/{fp.stem}.model.nc')