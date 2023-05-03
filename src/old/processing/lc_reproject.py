import xarray as xr
import rioxarray

fp = '/bsuhome/zacharykeskinen/scratch/data/uavsar/ncs/232_2020-02-21_2020-03-11.nc'
ds = xr.open_dataset(fp)

lc = xr.open_dataset('/bsuhome/zacharykeskinen/scratch/data/uavsar/land-cover/nlcd_2019_land_cover_l48_20210604.img', )
lc = lc.rio.reproject('EPSG:4326')
lc = lc.rio.clip_box(*ds.rio.bounds())

lc.to_netcdf('/bsuhome/zacharykeskinen/scratch/data/uavsar/ncs/landcover/lc.nc')