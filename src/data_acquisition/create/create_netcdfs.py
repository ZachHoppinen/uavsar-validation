from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import rioxarray as rxa
import matplotlib.pyplot as plt

data_dir = Path('/bsuhome/zacharykeskinen/scratch/data/uavsar')
ncs_dir = data_dir.joinpath('ncs')
tif_dir = data_dir.joinpath('tifs', 'Lowman, CO')

dems = []
ints = []
cohs = []
unws = []
for i, pair_dir in enumerate(tif_dir.glob('lowman_232*_grd')):

    print(pair_dir.stem)

    # get files
    ann = pd.read_csv(next(pair_dir.glob('*.csv')), index_col=0)
    int = rxa.open_rasterio(next(pair_dir.glob('*VV_*int.grd.tiff'))).squeeze('band', drop = True)
    coh = rxa.open_rasterio(next(pair_dir.glob('*VV_*cor.grd.tiff'))).squeeze('band', drop = True)
    
    # snag dem and use it as reference image
    if i == 0:
        dem = rxa.open_rasterio(next(pair_dir.glob('*VV_*hgt.grd.tiff'))).squeeze('band', drop = True)
        dem.attrs['units'] = 'meters'
        dem.attrs['long_name'] = 'elevation'
        ref = dem
    
    # add attributes
    int.attrs = ann.to_dict()
    int.attrs['units'] = 'radians'
    int.attrs['long_name'] = 'wrapped phase'
    coh.attrs = ann.to_dict()
    coh.attrs['units'] = ''
    coh.attrs['long_name'] = 'coherence'

    # add time and flight2 dimensions
    int = int.expand_dims(time = [pd.to_datetime(ann.loc['value', 'start time of acquisition for pass 1'])])
    int = int.assign_coords(time2 = ('time', [pd.to_datetime(ann.loc['value', 'start time of acquisition for pass 2'])]))

    coh = coh.expand_dims(time = [pd.to_datetime(ann.loc['value', 'start time of acquisition for pass 1'])])
    coh = coh.assign_coords(time2 = ('time', [pd.to_datetime(ann.loc['value', 'start time of acquisition for pass 2'])]))
    # dems.append(dem.interp_like(ref).expand_dims(time = [pd.to_datetime(ann.loc['value', 'start time of acquisition for pass 1'])]))
    
    # add wrapped and coherence to lists
    ints.append(int.interp_like(ref))
    cohs.append(coh.interp_like(ref))

    if len(list(pair_dir.glob('*VV_*unw.grd.tiff'))) == 1:
        # if we have unwrapped phase
        unw = rxa.open_rasterio(next(pair_dir.glob('*VV_*unw.grd.tiff'))).squeeze('band', drop = True)
        # add attributes
        unw.attrs = ann.to_dict()
        unw.attrs['units'] = 'radians'
        unw.attrs['long_name'] = 'unwrapped phase'

        # add time and flight 2 dim and coords
        unw = unw.expand_dims(time = [pd.to_datetime(ann.loc['value', 'start time of acquisition for pass 1'])])
        unw = unw.assign_coords(time2 = ('time', [pd.to_datetime(ann.loc['value', 'start time of acquisition for pass 2'])]))

        # add to list
        unws.append(unw.interp_like(ref))

# make dataset and add variables
ds = xr.Dataset()
ds['unw'] = xr.concat(unws, dim = 'time')
ds['int'] = xr.concat(ints, dim = 'time')
ds['cor'] = xr.concat(cohs, dim = 'time')
ds['dem'] = ref

# fix time from nanoseconds to pd datetime
ds['time'] = pd.to_datetime(ds['time'].data)

# save as intermediate file
ds.to_netcdf(ncs_dir.joinpath('tmp_uv.nc'))