# snowpit first
import pandas as pd

df = pd.read_parquet('/bsuhome/zacharykeskinen/uavsar-validation/data/insitu/all_insitu.parq')
df = df.loc[df.site_name != 'jackson']

import rioxarray as rxa
import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

ncs_dir = Path('/bsuscratch/zacharykeskinen/data/uavsar/ncs/')

from insitu import get_uavsar_insitu

from itertools import product

insitu_UV = pd.DataFrame()

for f in ncs_dir.glob('*.sd.nc'):
    print(f)
    ds = xr.open_dataset(f)
    df1, df2 = get_uavsar_insitu(df, ds)
    df1, df2 = df1.reset_index(names = 'datetime'), df2.reset_index(names = 'end_datetime')
    df1, df2 = df1.set_index(['site_name']), df2.set_index(['site_name'])
    df2['dSWE'] = df2['SWE'] - df1['SWE']
    df2['dSWE-A'] = df2['SWE-A'] - df1['SWE-A']
    df2['dSWE-B'] = df2['SWE-B'] - df1['SWE-B']
    df2['dSD'] = df2['SD'] - df1['SD']
    df2['dTime'] = df2['end_datetime'] - df1['datetime']
    for i, r in df2.iterrows():
        tol = 0.00090009 # ~0.001 degrees or ~100m
        df2.loc[i, 'VV_cor'] = ds['cor'].sel(band = 'VV', y = slice(r.lat + tol, r.lat - tol), x = slice(r.lon - tol, r.lon + tol)).mean(skipna = True)

        for img_type, band in product(['int', 'unw'], ['VV','VH','HV','HH']):
            if img_type not in ds.data_vars or band not in ds.band:
                continue

            tol = 0.00090009 # ~0.001 degrees or ~100m
            data = ds[f'sd_delta_{img_type}'].sel(band = band, y = slice(r.lat + tol, r.lat - tol), x = slice(r.lon - tol, r.lon + tol))
            data = data.where((data < 10000) & (data > -10000))
            if ~np.isfinite(data.mean(skipna = True)) and img_type =='int':
                print()
            df2.loc[i, f'UV_{img_type}_{band}_sd'] = data.mean(skipna = True)
    
            
    
    insitu_UV = pd.concat([insitu_UV, df2.reset_index(names = 'site_name')], axis = 0)

import warnings; warnings.filterwarnings('ignore', message='.*initial implementation of Parquet.*')
insitu_UV.to_parquet('/bsuhome/zacharykeskinen/uavsar-validation/data/insitu/insitu_uavsar_v2.parq')