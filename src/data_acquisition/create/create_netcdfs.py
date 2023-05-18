from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import rioxarray as rxa
import matplotlib.pyplot as plt
from pyproj import Transformer

xr.set_options(keep_attrs = True)

data_dir = Path('/bsuhome/zacharykeskinen/scratch/data/uavsar')
ncs_dir = data_dir.joinpath('ncs')
# tif_dir = data_dir.joinpath('tifs', 'Lowman, CO')

# ints = [None] * len(list(tif_dir.glob('lowman_232*_grd')))
# cohs = [None] * len(list(tif_dir.glob('lowman_232*_grd')))
# unws = [None] * len(list(tif_dir.glob('lowman_232*_grd')))
# for i, pair_dir in enumerate(tif_dir.glob('lowman_232*_grd')):

#     if i > 1:
#         break

#     print(pair_dir.stem)

#     # get files
#     ann = pd.read_csv(next(pair_dir.glob('*.csv')), index_col=0)
#     int = rxa.open_rasterio(next(pair_dir.glob('*VV_*int.grd.tiff'))).squeeze('band', drop = True)
#     coh = rxa.open_rasterio(next(pair_dir.glob('*VV_*cor.grd.tiff'))).squeeze('band', drop = True)
    
#     # snag dem and use it as reference image
#     if i == 0:
#         dem = rxa.open_rasterio(next(pair_dir.glob('*VV_*hgt.grd.tiff'))).squeeze('band', drop = True)
#         dem.attrs['units'] = 'meters'
#         dem.attrs['long_name'] = 'elevation'
#         ref = dem
    
#     # add attributes
#     int.attrs = ann.loc['value',:].to_dict()
#     int.attrs['units'] = 'radians'
#     int.attrs['long_name'] = 'wrapped phase'
#     coh.attrs = ann.loc['value',:].to_dict()
#     coh.attrs['units'] = ''
#     coh.attrs['long_name'] = 'coherence'

#     # add time and flight2 dimensions
#     int = int.expand_dims(time = [pd.to_datetime(ann.loc['value', 'start time of acquisition for pass 1'])])
#     int = int.assign_coords(time2 = ('time', [pd.to_datetime(ann.loc['value', 'start time of acquisition for pass 2'])]))

#     coh = coh.expand_dims(time = [pd.to_datetime(ann.loc['value', 'start time of acquisition for pass 1'])])
#     coh = coh.assign_coords(time2 = ('time', [pd.to_datetime(ann.loc['value', 'start time of acquisition for pass 2'])]))
#     # dems.append(dem.interp_like(ref).expand_dims(time = [pd.to_datetime(ann.loc['value', 'start time of acquisition for pass 1'])]))
    
#     # add wrapped and coherence to lists
#     ints[i] = int.interp_like(ref)
#     cohs[i] = coh.interp_like(ref)

#     if len(list(pair_dir.glob('*VV_*unw.grd.tiff'))) == 1:
#         # if we have unwrapped phase
#         unw = rxa.open_rasterio(next(pair_dir.glob('*VV_*unw.grd.tiff'))).squeeze('band', drop = True)
#         # add attributes
#         unw.attrs = ann.loc['value',:].to_dict()
#         unw.attrs['units'] = 'radians'
#         unw.attrs['long_name'] = 'unwrapped phase'

#         # add time and flight 2 dim and coords
#         unw = unw.expand_dims(time = [pd.to_datetime(ann.loc['value', 'start time of acquisition for pass 1'])])
#         unw = unw.assign_coords(time2 = ('time', [pd.to_datetime(ann.loc['value', 'start time of acquisition for pass 2'])]))

#         # add to list
#         unws[i] = unw.interp_like(ref)

# # make dataset and add variables
# ds = xr.Dataset()
# # concat each list of DataArrays to one dataset variable by combining on time
# ds['unw'] = xr.concat([i for i in unws if i is not None], dim = 'time', combine_attrs = 'drop_conflicts')
# ds['int'] = xr.concat([i for i in ints if i is not None], dim = 'time', combine_attrs = 'drop_conflicts')
# ds['cor'] = xr.concat([i for i in cohs if i is not None], dim = 'time', combine_attrs = 'drop_conflicts')
# ds['dem'] = ref

# # remove complex dtype can't save to netcdf
# ds['int_re'] = ds['int'].real
# ds['int_im'] = ds['int'].imag
# ds['int_phase'] = xr.apply_ufunc(np.angle, ds['int'])
# ds['int_mag'] = xr.apply_ufunc(np.abs, ds['int'])
# ds = ds.drop_vars('int')

# # fix time from nanoseconds to pd datetime
# ds['time'] = pd.to_datetime(ds['time'].data)
# ds['time2'] = pd.to_datetime(ds['time2'].data)

# ds['time2'] = pd.to_datetime(ds['time2'].data)

# # save as intermediate file
# ds.to_netcdf(ncs_dir.joinpath('tes_tmp_uv.nc'))

## get geometry (incidence angle and path vectors)

# ds = xr.open_dataset(ncs_dir.joinpath('tmp_uv.nc'))

geom_dir = ncs_dir.joinpath('geometry')

# inc = rxa.open_rasterio(next(geom_dir.glob('*.inc.tiff'))).squeeze('band', drop = True).interp_like(ds['dem'])
# ds['inc'] = inc.drop('spatial_ref')

# geom = xr.open_dataset(next(geom_dir.glob('*geom.nc'))).sel(heading = '232').interp_like(ds['dem'])
# for var in geom.data_vars:
#     ds[var] = geom[var].drop('spatial_ref')

# # save as intermediate file
# ds.to_netcdf(ncs_dir.joinpath('tmp_uv_geom.nc'))

## get atmosphere

# ds = xr.open_dataset(ncs_dir.joinpath('tmp_uv_geom.nc'))

# geom_atm = xr.open_dataset(next(geom_dir.glob('*v2.nc'))).sel(heading = '232')
# geom_atm['inc'] = ds['inc'].interp_like(geom_atm['dem'])

# geom_atm = geom_atm.rename({'y':'latitude', 'x':'longitude'})

# atm_dir = ncs_dir.joinpath('atmosphere')

# ds['delay'] = xr.zeros_like(ds['int_phase'])

# from phase_o_matic import presto_phase_delay
# for t1, t2 in zip(ds.time.data, ds.time2.data):
#     print(t1)
#     # get phase delay for each specific day interpolated across the dem and divided by cos(inc)
#     d1 = presto_phase_delay(pd.to_datetime(t1), geom_atm['dem'], geom_atm['inc'], work_dir = atm_dir)
#     d2 = presto_phase_delay(pd.to_datetime(t2), geom_atm['dem'], geom_atm['inc'], work_dir = atm_dir)
#     # difference bteween two days delay
#     delay = d2.isel(time = 0) - d1.isel(time = 0)
#     # rename back to dimensions of our dataset
#     delay_interp = delay['delay'].rename({'latitude':'y', 'longitude':'x'}).interp_like(ds['dem'])

#     ds['delay'].loc[{'time':t1}] = delay_interp

# ds.to_netcdf(ncs_dir.joinpath('tmp_uv_geom_atm.nc'))

## get tree data

tree_dir = data_dir.joinpath('trees')
nlcd = tree_dir.joinpath('nlcd')
biomass = tree_dir.joinpath('biomass')

ds = xr.open_dataset(ncs_dir.joinpath('tmp_uv_geom_atm.nc'))

# get tree height
print('tree height')
tree_height = xr.open_dataarray(tree_dir.joinpath('Forest_height_2019_NAM.tif')).rio.clip_box(*ds.rio.bounds()).squeeze('band', drop = True)
# get old max for masking out interpolated edges
height_max = tree_height.max() + 100
# interpolate to match our dataset
tree_height = tree_height.interp_like(ds['dem'])
# mask out outliers from interpolation
tree_height = tree_height.where((tree_height >= 0) & (tree_height <= height_max))
# add to dataset
ds['tree_height'] =  tree_height.drop('spatial_ref')

# get tree percentage 
print('tree percentage')

tree_perc = xr.open_dataarray(nlcd.joinpath('nlcd_2016_treecanopy_2019_08_31.img')).squeeze('band', drop =  True)

# find bounds in this datasets crs to clip it before reprojecting
# https://pyproj4.github.io/pyproj/stable/api/transformer.html
transformer = Transformer.from_crs("epsg:4326","epsg:5070", always_xy = True)

bds = list(transformer.transform(*ds.rio.bounds()[:2]))
bds.extend(list(transformer.transform(*ds.rio.bounds()[2:])))
# clip big raster to our area
tree_perc = tree_perc.rio.clip_box(*bds)
# add our crs to reproject
ds = ds.rio.write_crs('epsg:4326')
# reproject and mask out areas where the interpolation led to artifacts
tree_perc = tree_perc.rio.reproject_match(ds['dem'])
tree_perc = tree_perc.where((tree_perc >= 0) & (tree_perc < 100)) # percentage bounds 0-100
# add to dataset
ds['tree_perc'] = tree_perc

# get vegetation biomass
print('tree biomass')

bm = xr.open_dataarray(biomass.joinpath('conus_forest_biomass_mg_per_ha.img')).squeeze('band', drop = True)
# get old max for masking
bm_max = bm.max() + 1000
# clip big raster to our study area
transformer = Transformer.from_crs("epsg:4326","epsg:5069", always_xy = True)
bds = list(transformer.transform(*ds.rio.bounds()[:2]))
bds.extend(list(transformer.transform(*ds.rio.bounds()[2:])))
bm = bm.rio.clip_box(*bds)
# reproject to our crs and mask out artificats
bm = bm.rio.reproject_match(ds['dem'])
bm = bm.where((bm < bm_max) & (bm >= 0))
# add to dataset
ds['tree_biomass'] = bm

## add attributes
ds['tree_biomass'].attrs['long_name'] = '2008 USDS Biomass'
ds['tree_biomass'].attrs['units'] = 'megagrams / hectare'
ds['tree_biomass'].attrs['data_url'] = 'https://data.fs.usda.gov/geodata/rastergateway/biomass/conus_forest_biomass.php'

ds['tree_height'].attrs['long_name'] = '2019 GLAD tree height'
ds['tree_height'].attrs['units'] = 'meters'
ds['tree_height'].attrs['data_url'] = 'https://glad.umd.edu/dataset/gedi'

ds['tree_perc'].attrs['long_name'] = '2016 NLCD Tree Percentage'
ds['tree_perc'].attrs['units'] = 'Percent'
ds['tree_perc'].attrs['data_url'] = 'https://www.mrlc.gov/data/nlcd-2016-usfs-tree-canopy-cover-conus'

# save temporary file

ds.to_netcdf(ncs_dir.joinpath('tmp_uv_geom_atm_veg.nc'))