from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import rioxarray as rxa
import matplotlib.pyplot as plt
from pyproj import Transformer

from uavsar_pytools.snow_depth_inversion import phase_from_depth, depth_from_phase

xr.set_options(keep_attrs = True)

def create_netcdf(tif_dir, ncs_dir):
    ints = [None] * len(list(tif_dir.glob('lowman_232*_grd')))
    cohs = [None] * len(list(tif_dir.glob('lowman_232*_grd')))
    unws = [None] * len(list(tif_dir.glob('lowman_232*_grd')))
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
        int.attrs = ann.loc['value',:].to_dict()
        int.attrs['units'] = 'radians'
        int.attrs['long_name'] = 'wrapped phase'
        coh.attrs = ann.loc['value',:].to_dict()
        coh.attrs['units'] = ''
        coh.attrs['long_name'] = 'coherence'

        # add time and flight2 dimensions
        int = int.expand_dims(time = [pd.to_datetime(ann.loc['value', 'start time of acquisition for pass 1'])])
        int = int.assign_coords(time2 = ('time', [pd.to_datetime(ann.loc['value', 'start time of acquisition for pass 2'])]))

        coh = coh.expand_dims(time = [pd.to_datetime(ann.loc['value', 'start time of acquisition for pass 1'])])
        coh = coh.assign_coords(time2 = ('time', [pd.to_datetime(ann.loc['value', 'start time of acquisition for pass 2'])]))
        # dems.append(dem.interp_like(ref).expand_dims(time = [pd.to_datetime(ann.loc['value', 'start time of acquisition for pass 1'])]))
        
        # add wrapped and coherence to lists
        ints[i] = int.interp_like(ref)
        cohs[i] = coh.interp_like(ref)

        if len(list(pair_dir.glob('*VV_*unw.grd.tiff'))) == 1:
            # if we have unwrapped phase
            unw = rxa.open_rasterio(next(pair_dir.glob('*VV_*unw.grd.tiff'))).squeeze('band', drop = True)
            # add attributes
            unw.attrs = ann.loc['value',:].to_dict()
            unw.attrs['units'] = 'radians'
            unw.attrs['long_name'] = 'unwrapped phase'

            # add time and flight 2 dim and coords
            unw = unw.expand_dims(time = [pd.to_datetime(ann.loc['value', 'start time of acquisition for pass 1'])])
            unw = unw.assign_coords(time2 = ('time', [pd.to_datetime(ann.loc['value', 'start time of acquisition for pass 2'])]))

            # add to list
            unws[i] = unw.interp_like(ref)

    # make dataset and add variables
    ds = xr.Dataset()
    # concat each list of DataArrays to one dataset variable by combining on time
    ds['unw'] = xr.concat([i for i in unws if i is not None], dim = 'time', combine_attrs = 'drop_conflicts')
    ds['int'] = xr.concat([i for i in ints if i is not None], dim = 'time', combine_attrs = 'drop_conflicts')
    ds['cor'] = xr.concat([i for i in cohs if i is not None], dim = 'time', combine_attrs = 'drop_conflicts')
    ds['dem'] = ref

    # remove complex dtype can't save to netcdf
    ds['int_re'] = ds['int'].real
    ds['int_im'] = ds['int'].imag
    ds['int_phase'] = xr.apply_ufunc(np.angle, ds['int'])
    ds['int_mag'] = xr.apply_ufunc(np.abs, ds['int'])
    ds = ds.drop_vars('int')

    # fix time from nanoseconds to pd datetime
    ds['time'] = pd.to_datetime(ds['time'].data)
    ds['time2'] = pd.to_datetime(ds['time2'].data)

    ds['time2'] = pd.to_datetime(ds['time2'].data)

    # save as intermediate file
    ds.to_netcdf(ncs_dir.joinpath('tmp_uv.nc'))

## get geometry (incidence angle and path vectors)

def add_inc_geom(ncs_dir):
    ds = xr.open_dataset(ncs_dir.joinpath('tmp_uv.nc'))

    geom_dir = ncs_dir.joinpath('geometry')

    inc = rxa.open_rasterio(next(geom_dir.glob('*.inc.tiff'))).squeeze('band', drop = True).interp_like(ds['dem'])
    ds['inc'] = inc.drop('spatial_ref')

    geom = xr.open_dataset(next(geom_dir.glob('*geom.nc'))).sel(heading = '232').interp_like(ds['dem'])
    for var in geom.data_vars:
        ds[var] = geom[var].drop('spatial_ref')

    # save as intermediate file
    ds.to_netcdf(ncs_dir.joinpath('tmp_uv_geom.nc'))

## get atmosphere

def add_atmosphere(ncs_dir):
    ds = xr.open_dataset(ncs_dir.joinpath('tmp_uv_geom.nc'))

    geom_atm = xr.open_dataset(next(geom_dir.glob('*v2.nc'))).sel(heading = '232')
    geom_atm['inc'] = ds['inc'].interp_like(geom_atm['dem'])

    geom_atm = geom_atm.rename({'y':'latitude', 'x':'longitude'})

    atm_dir = ncs_dir.joinpath('atmosphere')

    ds['delay'] = xr.zeros_like(ds['int_phase'])

    from phase_o_matic import presto_phase_delay
    for t1, t2 in zip(ds.time.data, ds.time2.data):
        print(t1)
        # get phase delay for each specific day interpolated across the dem and divided by cos(inc)
        d1 = presto_phase_delay(pd.to_datetime(t1), geom_atm['dem'], geom_atm['inc'], work_dir = atm_dir)
        d2 = presto_phase_delay(pd.to_datetime(t2), geom_atm['dem'], geom_atm['inc'], work_dir = atm_dir)
        # difference bteween two days delay
        delay = d2.isel(time = 0) - d1.isel(time = 0)
        # rename back to dimensions of our dataset
        delay_interp = delay['delay'].rename({'latitude':'y', 'longitude':'x'}).interp_like(ds['dem'])

        ds['delay'].loc[{'time':t1}] = delay_interp

    ds.to_netcdf(ncs_dir.joinpath('tmp_uv_geom_atm.nc'))

## get tree data

def add_vegetation(ncs_dir):
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
    # remove attribute grid_mapping that appears somehow
    del ds['dem'].attrs['grid_mapping']

    ds.to_netcdf(ncs_dir.joinpath('tmp_uv_geom_atm_veg.nc'))

# ## get model data

def add_model(ncs_dir):
    model_dir = ncs_dir.joinpath('model')

    # first we need to fix up our time data
    # we need a time for model time slices, time1 for uavsar flight 1s and a coordinate not dim time2 of the second flight
    ds = xr.open_dataset(ncs_dir.joinpath('tmp_uv_geom_atm_veg.nc'))
    # round first to make sure we don't have any second mismatches
    t2 = ds.time2.dt.round('min').data
    # drop time2 as an index should just be a coordinate and reliant on time1
    ds = ds.drop_indexes('time2')
    # swap time name to time1
    ds = ds.swap_dims({'time':'time1'})
    # round data
    ds['time1'] = ds.time.dt.round('min').data
    # drop the time dimension now it is names time1
    ds = ds.drop('time')
    # drop index on time2 that reappears when we drop time
    ds = ds.drop_indexes('time2')
    # open model dataset
    model = xr.open_dataset(model_dir.joinpath('model.re.nc'))

    # now we calculate time dimension which should be all time steps in time1 and time2 with no duplicates
    times = np.unique(np.concatenate([ds.time1, ds.time2]))
    # add this new time dimension in
    ds['time'] = times

    # calculate the new shape of our data
    old_shape = list(ds['int_phase'].shape)
    old_shape[0] = len(ds['time'])

    # make datarrays with time slices for each time1 and time2 iwth no duplicates
    model_swe = xr.DataArray(np.zeros(old_shape),
                coords = [times, ds.y.data, ds.x.data],
                dims = ['time','y','x']
    )
    model_melt = model_swe.copy()

    # these are a function of time1 so we can just copy our existing data variable
    model_d_swe = xr.zeros_like(ds['int_phase'])
    cum_melt = xr.zeros_like(ds['int_phase'])

    # now cycle through each uavsar flight pair
    for t1, t2 in zip(ds.time1.data, ds.time2.data):
        print([t1, t2])
        t1, t2 = pd.to_datetime(t1), pd.to_datetime(t2)
        # get relevant time slice of the model
        m1 = model.sel(time = t1, method = 'nearest').interp_like(ds['dem'])
        m2 = model.sel(time = t2, method = 'nearest').interp_like(ds['dem'])

        # this is the difference in swe and melt
        diff = m2 - m1
        # get SWE and Melt at t1 and t2 and save to "time" dimension
        model_swe.loc[dict(time = t1)] = m1['swe']
        model_swe.loc[dict(time = t2)] = m2['swe']

        model_melt.loc[dict(time = t1)] = m1['melt']
        model_melt.loc[dict(time = t2)] = m2['melt']

        # get change in SWE and save to "time1" dimension
        model_d_swe.loc[dict(time1 = t1)] = m2['swe'] - m1['swe']

        # get cumulative melt and save to "time1" dimension
        cum_melt.loc[dict(time1 = t1)] = model.sel(time = slice(t1, t2))['melt'].sum(dim = ['time']).interp_like(ds['dem'])

    # now when we add these as data variables they will know to set either time1 or time
    # based on what dimension are in the dataarray
    ds['model_swe'] = model_swe
    ds['model_melt'] = model_melt
    ds['cum_melt'] = cum_melt
    ds['model_d_swe'] = model_d_swe

    # add attrs
    ds['model_swe'].attrs['long_name'] = 'SNOWMODEL SWE'
    ds['model_melt'].attrs['long_name'] = 'SNOWMODEL Melt'
    ds['cum_melt'].attrs['long_name'] = 'SNOWMODEL Cumulative Melt'
    ds['model_d_swe'].attrs['long_name'] = 'SNOWMODEL Change in SWE'

    for var in ['model_d_swe', 'cum_melt', 'model_swe', 'model_melt']:
        ds[var].attrs['units'] = 'meters'

    # fix time2 to be a coordinate not an index one more time
    t2 = ds['time2']
    ds = ds.drop('time2')
    ds = ds.assign_coords(time2 = ('time1', t2.data))

    # mask out some extreme outliers in delay
    ds['delay'] = ds.delay.where((ds.delay > ds.delay.quantile(0.01)) & (ds.delay < ds.delay.quantile(0.99)))

    # finally sort by time
    ds = ds.sortby('time1')
    ds = ds.sortby('time')

    # save dataset
    ds.to_netcdf(ncs_dir.joinpath('tmp_uv_geom_atm_veg_model_v2.nc'))

# # now correct by insitu data

def insitu_correct(ncs_dir):
    data_dir = Path('/bsuhome/zacharykeskinen/uavsar-validation/data')
    insitu_dir = data_dir.joinpath('insitu')
    insitu = pd.read_parquet(insitu_dir.joinpath('all_difference.parq'))
    insitu = insitu[insitu.site_name != 'jackson']
    insitu = gpd.GeoDataFrame(insitu, geometry=gpd.points_from_xy(insitu.lon, insitu.lat), crs="EPSG:4326")

    ds = xr.open_dataset(ncs_dir.joinpath('tmp_uv_geom_atm_veg_model_v2.nc'))
    ds['delay'] = ds.delay.where((ds.delay > ds.delay.quantile(0.01)) & (ds.delay < ds.delay.quantile(0.99)))
    ds = ds.sortby('time1')
    ds = ds.sortby('time')

    ds['unw_atm'] = ds['unw'] - ds['delay']
    ds['int_atm'] = ds['int_phase'] - ds['delay']

    inc_mean = ds['inc'].mean()

    snotels = insitu#[insitu.datasource == 'NRCS']

    for t1 in insitu.time1.unique():
        print(t1)
        insitu_time = snotels[snotels.time1 == t1]
        # correct by SWE converted to SD
        insitu_density = pd.concat([insitu_time['t1_density'], insitu_time['t2_density']]).mean()

        insitu_dSD = (insitu_time['dSWE'] * insitu_time['t1_density'] / 997).mean()
        expected_phase = phase_from_depth(insitu_dSD, inc_angle = inc_mean, density = insitu_density)

        t2 = insitu_time.iloc[0]['time2']
        snow_on = (ds['model_swe'].sel(time = t1) > 0.1) & (ds['model_swe'].sel(time = t2) > 0.1)
        # snow_cor = (snow_on & (ds['cor'].sel(time1 = t1) > 0.35))

        for k in ['unw', 'int_phase','unw_atm','int_atm']:
            cur_phase = ds[k].sel(time1 = t1).where(snow_on).mean()

            ds[k].loc[{'time1':t1}] = ds[k].sel(time1 = t1) - (cur_phase - expected_phase)

        # sd corrected
        # ds.to_netcdf(ncs_dir.joinpath('final_insitu.nc'))
        # swe corrected
        ds.to_netcdf(ncs_dir.joinpath('final_insitu_swe_all_no_cor.nc'))

def coarsen(ncs_dir):
    ds = xr.open_dataset(ncs_dir.joinpath('final_insitu_swe_all_no_cor.nc'))# .isel(x = slice(0, -1, 100), y = slice(0, -1, 100))

    ds = ds.chunk()

    ds_10 = ds.coarsen(x = 10, y = 10, boundary = 'trim').mean()
    ds_10.to_netcdf(ncs_dir.joinpath('final_swe_all_no_cor_10_10.nc'))

    ds_50 = ds.coarsen(x = 50, y = 50, boundary = 'trim').mean()
    ds_50.to_netcdf(ncs_dir.joinpath('final_swe_all_no_cor_50_50.nc'))

if __name__ == '__main__':
    data_dir = Path('/bsuhome/zacharykeskinen/scratch/data/uavsar')
    ncs_dir = data_dir.joinpath('ncs')
    tif_dir = data_dir.joinpath('tifs', 'Lowman, CO')
    # print('Starting')
    # create_netcdf(tif_dir, ncs_dir)
    # print('Add Incidence Angle')
    # add_inc_geom(ncs_dir)
    # print('Add atmosphere')
    # add_atmosphere(ncs_dir)
    # print('Add vegetation')
    # add_vegetation(ncs_dir)
    # print('Add model')
    # add_model(ncs_dir)
    print('Insitu Correct')
    insitu_correct(ncs_dir)
    print('Coarsen')
    coarsen(ncs_dir)