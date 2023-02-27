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

import pandas as pd
df = pd.read_parquet('/bsuhome/zacharykeskinen/uavsar-validation/data/insitu/insitu_uavsar.parq')
df = df.loc[df.site_name != 'jackson']
df = df.loc[df.dSD > 0.8]
df[['site_name', 'dSD', 'end_datetime', 'dTime']]

import contextily as ctx
import matplotlib.pyplot as plt
import xarray as xr
df = pd.read_parquet('/bsuhome/zacharykeskinen/uavsar-validation/data/insitu/insitu_uavsar.parq')

f, ax = plt.subplots(figsize = (12, 8))

ds = xr.open_dataset('/bsuhome/zacharykeskinen/scratch/data/uavsar/ncs/232_2021-02-03_2021-02-10.sd.nc')

ds['sd_delta_int'].sel(band = 'VV').plot(ax = ax)

df.plot.scatter(x = 'lon', y = 'lat', ax= ax)
ctx.add_basemap(ax = ax, crs= 'EPSG:4326')

plt.savefig('/bsuhome/zacharykeskinen/uavsar-validation/figures/insitu/map_insitu.png')
plt.show()

fig, axes = plt.subplots(1, 2, figsize = (12,8))

df.plot.scatter(x = 'dSD', y = 'UV_int_VV_sd', ax= axes[0], label = 'VV')
df.plot.scatter(x = 'dSD', y = 'UV_int_VH_sd', ax= axes[0], color = 'C1', label = 'VH')
df.plot.scatter(x = 'dSD', y = 'UV_int_HV_sd', ax= axes[0], color = 'C2', label = 'HV')
df.plot.scatter(x = 'dSD', y = 'UV_int_HH_sd', ax= axes[0], color = 'C1', label = 'HH')
axes[0].set_title('Lowman Insitu vs Wrapped UAVSAR Snow Depth Changes')
axes[0].set_xlabel('Insitu')
axes[0].set_ylabel('Wrapped UAVSAR SD')

df.plot.scatter(x = 'dSD', y = 'UV_unw_VV_sd', ax= axes[1], label = 'VV')
df.plot.scatter(x = 'dSD', y = 'UV_unw_VH_sd', ax= axes[1], color = 'C1', label = 'VH')
df.plot.scatter(x = 'dSD', y = 'UV_unw_HV_sd', ax= axes[1], color = 'C2', label = 'HV')
df.plot.scatter(x = 'dSD', y = 'UV_unw_HH_sd', ax= axes[1], color = 'C1', label = 'HH')
axes[1].set_title('Lowman Insitu vs Unwrapped UAVSAR Snow Depth Changes')
axes[1].set_xlabel('Insitu')
axes[1].set_ylabel('Unwrapped UAVSAR SD')
plt.savefig('/bsuhome/zacharykeskinen/uavsar-validation/figures/insitu/insitu_uavsar_scatter.png')