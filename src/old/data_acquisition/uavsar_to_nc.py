from os.path import join, exists, basename
from glob import glob
import numpy as np
import pandas as pd
import xarray as xr
import rioxarray as rxa
from tqdm import tqdm

import rasterio as rio

from uavsar_pytools import UavsarCollection

from georeferencing import resolution_to_meters

import os

import sys

import certifi

os.environ['REQUESTS_CA_BUNDLE'] = os.path.join(os.path.dirname(sys.argv[0]), certifi.where())

#directory of images
img_dir = '/bsuhome/zacharykeskinen/scratch/data/uavsar/ncs/images'

if len(glob(join(img_dir, '*_grd'))) < 17:
    lowman_col = UavsarCollection(collection = 'Lowman, CO', work_dir = img_dir, clean = False, inc = True)
    lowman_col.collection_to_tiffs()

# holds all dataset one for each image pair
datasets = []

# amount to coarsen each image
coarsen_factor = 1

for img_pair in tqdm(glob(join(img_dir, '*')), desc = 'Creating Lowman netcdfs.'):

    # get all the coherence, unwrapped, wrapped, incidence angle tiffs
    tiffs = [i for i in glob(join(img_pair, '*.tiff')) if any(ss in i for ss in ['cor', 'int', 'unw', 'inc'])]

    if len(tiffs) == 0:
        print(f'No images for {img_pair}')
        continue

    # get metadata
    ann = pd.read_csv(glob(join(img_pair, '*.csv'))[0])
    time1 = pd.to_datetime(ann['start time of acquisition for pass 1'].iloc[0])
    time2 = pd.to_datetime(ann['start time of acquisition for pass 2'].iloc[0])

    flight_dir = basename(img_pair).split('_')[1][:3]

    nc_dir = '/bsuhome/zacharykeskinen/scratch/data/uavsar/ncs'
    out_fp = join(nc_dir , f"{flight_dir}_{time1.strftime('%Y-%m-%d')}_{time2.strftime('%Y-%m-%d')}.nc")

    if exists(out_fp):
        print(f'{out_fp} exists. Skipping.')
        continue

    resolution_deg = float(ann.loc[0, 'grd_mag.col_mult']) # deg/pixel
    ul_lat = ann.loc[0, 'grd_mag.row_addr']
    ul_lon = ann.loc[0, 'grd_mag.col_addr']

    # calculate approximate resolution in meters
    res_meters = resolution_to_meters(resolution_deg, ul_lat)

    # make a dataset to hold all this pairs images
    img_ds = xr.Dataset()

    # loop through each type of image and make a DataArray of all bands
    for img_type in ['cor', 'int', 'unw', 'inc']:

        type_imgs = [i for i in glob(join(img_pair, '*.tiff')) if f'.{img_type}.' in i]

        if len(type_imgs) == 0:
            print(f'No images for {img_type} in {img_pair}')
            continue

        band_imgs = []

        # VV, VH, HV, HH
        for tiff in type_imgs:

            # open image
            img = rxa.open_rasterio(tiff)

            # no band information for incidence angle
            if img_type != 'inc':
                img['band'] = [basename(tiff).split('_')[-2][4:]]

            # otherwise rename appropriately
            else:
                img = img.squeeze(dim = 'band')
                img = img.drop_vars('band')

            if img_type == 'int':
                img = xr.ufuncs.angle(img)
            
            # coarsen to approximately 30 meters
            img = img.coarsen(x = coarsen_factor, boundary = 'trim').mean().coarsen(y = coarsen_factor, boundary = 'trim').mean()

            if band_imgs:
               img = img.rio.reproject_match(band_imgs[0])

            band_imgs.append(img)
        
        # concat all bands into a data var of img dataset

        if img_type != 'inc': 
            img = xr.concat(band_imgs, dim = 'band')
        
        else:
            img = img.rio.reproject_match(img_ds['cor'])
        
        img_ds[img_type] = img

    img_ds.attrs['time1'] = time1.strftime("%Y-%m-%d")
    img_ds.attrs['time2'] = time2.strftime("%Y-%m-%d")

    img_ds.attrs['range_looks'] = int(ann.loc[0, 'number of looks in range']) * coarsen_factor
    img_ds.attrs['azimuth_looks'] = int(ann.loc[0, 'number of looks in azimuth']) * coarsen_factor
    img_ds.attrs['resolution_m'] = res_meters * coarsen_factor
    img_ds.attrs['resolution_deg'] = resolution_deg * coarsen_factor

    img_ds.to_netcdf(out_fp)