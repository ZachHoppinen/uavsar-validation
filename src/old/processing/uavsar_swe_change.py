from pathlib import Path
from glob import glob
import numpy as np
import xarray as xr
import pandas as pd
import rioxarray as rxa
import matplotlib.pyplot as plt

from insitu import get_uavsar_insitu

from uavsar_pytools.snow_depth_inversion import phase_from_depth, depth_from_phase

print('Starting python processing script.')

inc_232 = rxa.open_rasterio('/bsuhome/zacharykeskinen/scratch/data/uavsar/ncs/images/lowman_23205_20002-007_20007-003_0013d_s01_L090_01_int_grd/lowman_23205_20007_003_200213_L090_CX_01.inc.tiff', mask_and_scale = True)
inc_052 = rxa.open_rasterio('/bsuhome/zacharykeskinen/scratch/data/uavsar/ncs/images/lowman_05208_21009-005_21012-004_0007d_s01_L090_01_int_grd/lowman_05208_21012_004_210210_L090_CX_01.inc.tiff', mask_and_scale = True)

df = pd.read_parquet('/bsuhome/zacharykeskinen/uavsar-validation/data/insitu/all_insitu.parq')

print('Opened incs and df')

ncs_fp = Path('/bsuhome/zacharykeskinen/scratch/data/uavsar/ncs/')
a = []
for fp in ncs_fp.glob('*-*.nc'):
    if fp.with_suffix('.sd.nc').exists():
        continue

    if '.sd.nc' in str(fp):
        continue

    print(fp)

    img = xr.open_dataset(fp)
    df1, df2 = get_uavsar_insitu(df, img)

    diff = df2.set_index('site_name')[['SWE','SD']] - (df1.set_index('site_name')[['SWE','SD']])
    delta = diff.mean()
    den = pd.concat([df1, df2])['density'].mean()
    print(f"{img.attrs['time1']} to {img.attrs['time2']} SWE change is {delta['SWE']}. SD change is {delta['SD']}. Density is {den}")
    img.attrs['dSWE'] = delta['SWE']
    img.attrs['dSD'] = delta['SD']
    img.attrs['mean_density'] = den
    
    if fp.stem[:3] == '232':
        inc = inc_232.rio.reproject_match(img['cor'])
    
    elif fp.stem[:3] == '052':
        inc = inc_052.rio.reproject_match(img['cor'])
    
    inc = inc.squeeze(dim = 'band')
    inc = inc.drop_vars('band')

    inc = inc.where(inc < 10000).where(inc > -10000)

    inc = inc.rio.reproject_match(img['cor'])

    img['inc'] = inc

    mean_inc = img['inc'].mean()

    phase_theor = phase_from_depth(img.attrs['dSD'], mean_inc, density = den)

    for img_type in ['unw', 'int']:

        if img_type not in img.data_vars:

            continue
    
        img[f'sd_delta_{img_type}'] = xr.zeros_like(img[img_type])

        for band in ['VV', 'VH', 'HV', 'HH']:

            mean_phase = img[img_type].sel(band = band).where((img[img_type].sel(band = band) < 1e6) & (img[img_type].sel(band = band) > -1e6)).mean()

            img[img_type].loc[dict(band = band)] = img[img_type].sel(band = band) + (phase_theor - mean_phase)

            img[f'sd_delta_{img_type}'].loc[dict(band = band)] = depth_from_phase(img[img_type].sel(band = band), img['inc'], density = img.attrs['mean_density'])

    img.to_netcdf(fp.with_suffix('.sd.nc'))