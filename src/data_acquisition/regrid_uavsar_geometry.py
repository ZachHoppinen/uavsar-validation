from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import rioxarray as rxa
import matplotlib.pyplot as plt

from scipy.interpolate import griddata

geometry_dir = Path('/bsuhome/zacharykeskinen/scratch/data/uavsar/uavsar_geometry/')

DAs = []
for heading in ['05208_01_BU', '23205_01_BC']:
    print(heading)
    lats = []
    longs = []
    full_x = []
    full_y = []
    full_z = []
    for swath in [1,2,3]:
        lkv = np.fromfile(geometry_dir.joinpath(f'lowman_{heading}_s{swath}_2x8.lkv'), dtype = np.dtype('<f'))
        x, y, z = lkv[::3], lkv[1::3], lkv[2::3]
        mag = (x**2 + y**2 + z**2)**0.5

        llh = np.fromfile(geometry_dir.joinpath(f'lowman_{heading}_s{swath}_2x8.llh'), dtype = np.dtype('<f'))
        lat, lon, height = llh[::3], llh[1::3], llh[2::3]

        n = 1
        lat, lon, height = lat[::n], lon[::n], height[::n]
        x, y, z = x[::n], y[::n], z[::n]
    
        lats.extend(lat)
        longs.extend(lon)
        full_x.extend(x)
        full_y.extend(y)
        full_z.extend(z)


    lon = np.array(longs)
    lat = np.array(lats)
    # regrid to regular grid
    xs =  np.linspace(lon.min(), lon.max(), int(lon.shape[0]**0.5))
    ys = np.linspace(lat.min(), lat.max(), int(lat.shape[0]**0.5))
    xg, yg = np.meshgrid(xs, ys)

    points = (lon, lat)

    for var_name, arr in zip(['lkv_x','lkv_y','lkv_z'], [full_x, full_y, full_z]):
        data = griddata(points, arr, (xg, yg), method = 'cubic', fill_value = np.nan)

        # add to list of dataarrays
        da = xr.DataArray(data = np.expand_dims(data, axis = 0), coords = [[heading.split('_')[0][:3]], ys, xs], dims = ['heading', 'y', 'x']).rename(var_name)
        da = da.rio.write_crs('epsg:4326')
        da = da.rio.set_nodata(np.nan)
        DAs.append(da)

# reproject all to the same grid
for i, da in enumerate(DAs):
    if i == 0:
        continue
    da = da.rio.reproject_match(DAs[0])
    DAs[i] = da
ds = xr.merge(DAs)

# save output
ds.to_netcdf('/bsuhome/zacharykeskinen/scratch/data/uavsar/ncs/geometry/uavsar_geom.nc')