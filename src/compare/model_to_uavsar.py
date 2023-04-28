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

    diff_model = model.sel(time = ds.attrs['time2']) - model.sel(time = ds.attrs['time1'])
    diff_model['cor'] = ds['cor'].rio.reproject_match(diff_model['swe'])
    diff_model['uavsar_dSD_int'] = ds['sd_delta_int'].rio.reproject_match(diff_model['swe'])
    if 'unw' in ds.data_vars:
        diff_model['uavsar_dSD_unw'] = ds['sd_delta_unw'].rio.reproject_match(diff_model['swe'])
    diff_model.attrs = ds.attrs
    diff_model = diff_model.rio.clip_box(*ds.rio.bounds())

    diff_model.to_netcdf(f'/bsuhome/zacharykeskinen/scratch/data/uavsar/ncs/model/{fp.stem}.model.nc')

# loop through each flight pair and calculate rmse, r2, plot

import xarray as xr
import numpy as np
import rioxarray as rxa
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path

# loop through each flight pair and calculate rmse, r2, plot

ncs_dir = Path('/bsuscratch/zacharykeskinen/data/uavsar/ncs/model')

fig, axes = plt.subplots(1, 2, figsize = (12, 8))


for i, var in enumerate(['int','unw']):
    xs = []
    ys = []
    var = f'uavsar_dSD_{var}'
    for fp in ncs_dir.glob('*.sd.model.nc'):
        ds = xr.open_dataset(fp)
        if var in ds.data_vars:
            ds = ds#.coarsen(x = 3, boundary = 'pad').mean().coarsen(y = 3, boundary = 'pad').mean()
            xs.append(ds['swe'].values.ravel())
            ys.append(ds[var].sel(band = 'VV').values.ravel() * (ds.attrs['mean_density']/ 997))

    # stack arrays
    xs = np.hstack(xs)
    ys = np.hstack(ys)

    xs_tmp = xs[(~np.isnan(xs)) & (~np.isnan(ys))]
    ys = ys[(~np.isnan(xs)) & (~np.isnan(ys)) ]
    xs = xs_tmp

    # range = [[-0.001, 0.002], [-0.1, 0.1]]
    axes[i].hist2d(xs, ys, bins = 100, norm=mpl.colors.LogNorm(), cmap=mpl.cm.inferno) #, range = range)
    axes[i].set_xlabel('Modeled SWE Change')
    if i == 0:
        axes[i].set_title(f'Wrapped SWE Change (n = {len(ys):.2e})')
    else:
        axes[i].set_title(f'Unwrapped SWE Change (n = {len(ys):.2e})')
    #axes[i].set_xlim(range[0])
    #axes[i].set_ylim(range[1])

    from sklearn.metrics import mean_squared_error
    rms = mean_squared_error(xs, ys, squared=False)

    from scipy.stats import pearsonr
    r, p = pearsonr(xs, ys)
    axes[i].text(.01, .99, f'RMSE: {rms:.2}, r: {r:.2}', ha='left', va='top', transform=axes[i].transAxes)
plt.suptitle('UAVSAR vs SNOWMODEL SWE Change @ 100m Resolution')

axes[0].set_ylabel('UAVSAR SWE Change')
# identity line
for ax in axes:
    ax.plot([-1,1], [-1,1], label = '1-to-1')
axes[1].legend(loc = 'lower right')

plt.savefig('/bsuhome/zacharykeskinen/uavsar-validation/figures/model/hist2d_all_100.png')

import xarray as xr
import numpy as np
import rioxarray as rxa
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path

# loop through each flight pair and calculate rmse, r2, plot

ncs_dir = Path('/bsuscratch/zacharykeskinen/data/uavsar/ncs/model')

fig, axes = plt.subplots(1, 2, figsize = (12, 8))


for i, var in enumerate(['int','unw']):
    xs = []
    ys = []
    var = f'uavsar_dSD_{var}'
    for fp in ncs_dir.glob('*.sd.model.nc'):
        ds = xr.open_dataset(fp)
        if ds.attrs['dSWE'] > 0 and 'uavsar_dSD_unw' in ds.data_vars:
            if var in ds.data_vars:
                ds = ds#.coarsen(x = 3, boundary = 'pad').mean().coarsen(y = 3, boundary = 'pad').mean()
                xs.append(ds['swe'].values.ravel())
                ys.append(ds[var].sel(band = 'VV').values.ravel() * (ds.attrs['mean_density']/ 997))

    # stack arrays
    xs = np.hstack(xs)
    ys = np.hstack(ys)

    xs_tmp = xs[(~np.isnan(xs)) & (~np.isnan(ys)) & (xs > 0)]
    ys = ys[(~np.isnan(xs)) & (~np.isnan(ys)) & (xs > 0)]
    xs = xs_tmp

    range = [[-0.001, 0.2], [-0.12, 0.3]]
    axes[i].hist2d(xs, ys, bins = 100, norm=mpl.colors.LogNorm(), cmap=mpl.cm.inferno, range = range)
    axes[i].set_xlabel('Modeled SWE Change')
    if i == 0:
        axes[i].set_title(f'Wrapped SWE Change (n = {len(ys):.2e})')
    else:
        axes[i].set_title(f'Unwrapped SWE Change (n = {len(ys):.2e})')

    from sklearn.metrics import mean_squared_error
    rms = mean_squared_error(xs, ys, squared=False)

    from scipy.stats import pearsonr
    r, p = pearsonr(xs, ys)
    axes[i].text(.99, .99, f'RMSE: {rms:.2}, r: {r:.2}', ha='right', va='top', transform=axes[i].transAxes)
plt.suptitle('UAVSAR vs SNOWMODEL SWE Change for increasing SWE periods and pixels')

axes[0].set_ylabel('UAVSAR SWE Change')
# identity line
for ax in axes:
    ax.plot([0,1], [0,1], label = '1-to-1')
axes[1].legend(loc = 'lower right')
plt.savefig('/bsuhome/zacharykeskinen/uavsar-validation/figures/model/hist2d_increasing.png')

# mean through time!!!
n = len(list(ncs_dir.glob('*.sd.model.nc')))

for var in ['swe', 'uavsar_dSD_int']:
    for i, fp in enumerate(ncs_dir.glob('*.sd.model.nc')):
        ds = xr.open_dataset(fp)
        
        if i == 0:
            shape = np.array([(*ds.swe.shape, n)], dtype = object).ravel()
            full_array = np.zeros(shape)
            main_ds = ds
        else:
            ds = ds.rio.reproject_match(main_ds)

        if 'uavsar' not in var:   
            full_array[:, :, i-1] = ds[var].values
        else:
            full_array[:, :, i-1] = ds[var].sel(band = 'VV').values

    if 'uavsar' not in var:
        main_ds[var].data = np.mean(full_array, axis = 2)
    else:
        main_ds[var].loc[dict(band = 'VV')].data = np.mean(full_array, axis = 2)

import contextily as ctx
ds = main_ds
ds = ds.sel(y = slice(43.96, 44.35), x = slice(-115.66, -115.2))
fig, axes = plt.subplots(1, 2, figsize = (24,16))
# ds['swe'].data = gaussian_filter(ds['swe'], 0.1)
ds['swe'].where(~ds['uavsar_dSD_int'].sel(band = 'VV').isnull()).plot(ax = axes[0], vmin = 0, vmax= 0.11, alpha = 0.6, cbar_kwargs = {'label': '', 'shrink': 0.55})
ds['uavsar_dSD_int'].loc[dict(band = 'VV')] = gaussian_filter(ds['uavsar_dSD_int'].sel(band = 'VV'), 0.7)
(ds['uavsar_dSD_int'].sel(band = 'VV') * (ds.attrs['mean_density']/ 997)).plot(ax = axes[1], vmin = 0.0, vmax = 0.155, alpha = 0.6, cbar_kwargs = {'label': '', 'shrink': 0.55})

axes[0].set_title('Mean Modeled SWE Change')
axes[1].set_title('Mean UAVSAR SWE Change')

for ax in axes:
    ax.set_xlabel('')
    ax.set_ylabel('')
    ctx.add_basemap(ax = ax, crs = 'EPSG:4326', source = ctx.providers.Stamen.Terrain)

plt.savefig('/bsuhome/zacharykeskinen/uavsar-validation/figures/model/mean_maps.png')

# binned by conditions

from stats import get_stats
import xarray as xr
import numpy as np
import rioxarray as rxa
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path

def clean_xs_ys(xs, ys):
        # stack arrays
    xs = np.hstack(xs)
    ys = np.hstack(ys)

    xs_tmp = xs[(~np.isnan(xs)) & (~np.isnan(ys)) & (np.isfinite(xs)) & (np.isfinite(ys))]
    ys = ys[(~np.isnan(xs)) & (~np.isnan(ys))  & (np.isfinite(xs)) & (np.isfinite(ys))]
    xs = xs_tmp

    return xs, ys

#https://www.mrlc.gov/downloads/sciweb1/shared/mrlc/metadata/nlcd_2019_land_cover_l48_20210604.xml
evergreen = [42]
decidous = [41, 43] # and mixed
open = [12, 31, 52, 71]
water = [11, 90, 95]
developed = [21, 22, 23, 24, 81, 82]

fig = plt.figure(constrained_layout=True, figsize = (12, 12))
fig.suptitle('UAVSAR SWE Retrievals vs SNOWMODEL')
ncs_dir = Path('/bsuscratch/zacharykeskinen/data/uavsar/ncs/model')

# create 3x1 subfigs
subfigs = fig.subfigures(nrows=3, ncols=1)
for row, subfig in enumerate(subfigs):


    # create 1x3 subplots per subfig
    axes = subfig.subplots(nrows=1, ncols=3)

        # identity line
    for ax in axes.ravel():
        ax.plot([-1,1], [-1,1], label = '1-to-1')

    if row == 0:
        subfig.suptitle(f'Coherence', fontweight = 'bold')
        for i, (coh_low, coh_high) in enumerate([[0, 0.3], [0.3, 0.6],[0.6, 1]]):
            xs = []
            ys = []
            var = f'uavsar_dSD_unw'
            for fp in ncs_dir.glob('*.sd.model.nc'):
                ds = xr.open_dataset(fp)
                # ds = ds.where(~ds['lc'].isin(water.extend(developed)))
                ds = ds.where((ds['cor'].sel(band = 'VV') < coh_high) & (ds['cor'].sel(band = 'VV') > coh_low))

                if var in ds.data_vars:
                    xs.append(ds['swe'].values.ravel())
                    ys.append(ds[var].sel(band = 'VV').values.ravel() * (ds.attrs['mean_density']/ 997))

            xs, ys = clean_xs_ys(xs, ys)

            rmse, r, n = get_stats(xs, ys)

            axes[i].hist2d(xs, ys, bins = 100, norm=mpl.colors.LogNorm(), cmap=mpl.cm.inferno)

            axes[i].text(.01, .99, f'RMSE: {rmse:.2}, r: {r:.2}\nn = {len(xs):.2e}', ha='left', va='top', transform=axes[i].transAxes)
            axes[i].set_title(f'Coherence {coh_low} - {coh_high}')

            if i == 0:
                axes[i].set_ylabel('UAVSAR SWE Change')
            else:
                axes[i].set_yticks([])

            # axes[i].set_xticks([])
        
    if row == 1:
        subfig.suptitle(f'Snow Wetness', fontweight = 'bold')
        for i, (low, high) in enumerate([[0, 0.001], [0.001, 0.006],[0.006, 0.013]]):
            xs = []
            ys = []
            var = f'uavsar_dSD_unw'

            for fp in ncs_dir.glob('*.sd.model.nc'):
                ds = xr.open_dataset(fp)
                # ds = ds.where(~ds['lc'].isin(water.extend(developed)))
                ds['melt_sum'] = ds['melt_t1'] + ds['melt_t2']
                ds = ds.where((ds['melt_sum'] < high) & (ds['melt_sum'] > low))

                if var in ds.data_vars:
                    xs.append(ds['swe'].values.ravel())
                    ys.append(ds[var].sel(band = 'VV').values.ravel() * (ds.attrs['mean_density']/ 997))
            xs, ys = clean_xs_ys(xs, ys)

            rmse, r, n = get_stats(xs, ys)

            axes[i].hist2d(xs, ys, bins = 100, norm=mpl.colors.LogNorm(), cmap=mpl.cm.inferno)

            axes[i].text(.01, .99, f'RMSE: {rmse:.2}, r: {r:.2}\nn = {len(xs):.2e}', ha='left', va='top', transform=axes[i].transAxes)
            axes[i].set_title(f'Melt {low} - {high}')

            if i == 0:
                axes[i].set_ylabel('UAVSAR SWE Change')
            else:
                axes[i].set_yticks([])
            # axes[i].set_xticks([])
    
    if row == 2:
        subfig.suptitle(f'Land Cover', fontweight = 'bold')
        for i, (lc_list) in enumerate([open, decidous, evergreen]):
            xs = []
            ys = []
            var = f'uavsar_dSD_unw'

            for fp in ncs_dir.glob('*.sd.model.nc'):
                ds = xr.open_dataset(fp)
                # ds = ds.where(~ds['lc'].isin(water.extend(developed)))
                ds = ds.where(ds['lc'].isin(lc_list))

                if var in ds.data_vars:
                    xs.append(ds['swe'].values.ravel())
                    ys.append(ds[var].sel(band = 'VV').values.ravel() * (ds.attrs['mean_density']/ 997))
            xs, ys = clean_xs_ys(xs, ys)

            rmse, r, n = get_stats(xs, ys)

            axes[i].hist2d(xs, ys, bins = 100, norm = mpl.colors.LogNorm(), cmap=mpl.cm.inferno)

            axes[i].text(.01, .99, f'RMSE: {rmse:.2}, r: {r:.2}\nn = {len(xs):.2e}', ha='left', va='top', transform=axes[i].transAxes)
            if i == 0:
                axes[i].set_title(f'Open')
                axes[i].set_ylabel('UAVSAR SWE Change')
            elif i == 1:
                axes[i].set_title('Deciduous')
                axes[i].set_yticks([])

            elif i == 2:
                axes[i].set_title('Evergreen')
                axes[i].set_yticks([])
            axes[i].set_xlabel('Modeled SWE Change')



for ax in fig.get_axes():
    ax.set_xlim(-0.21, 0.2)
    ax.set_ylim(-0.3, 0.3)

plt.savefig('/bsuhome/zacharykeskinen/uavsar-validation/figures/model/modeling_binned_v2.png')