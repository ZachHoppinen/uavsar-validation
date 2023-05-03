import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import rioxarray as rxa
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from uavsar_pytools.snow_depth_inversion import depth_from_phase
from stats import get_stats, clean_xs_ys
from pathlib import Path
from scipy.ndimage import gaussian_filter

subsets = gpd.read_file(Path('~/scratch/data/uavsar/lidar/subset/date.shp').expanduser())
lidar_dir = Path('/bsuhome/zacharykeskinen/scratch/data/uavsar/ncs/lidar/')
figdir = Path('/bsuhome/zacharykeskinen/uavsar-validation/figures')

fig, axes = plt.subplots(4, 3, figsize = (12, 8))

for i, (_, sub) in enumerate(subsets.iterrows()):
    loc, year = sub.date.split('-')
    lidar = xr.open_dataset(lidar_dir.joinpath(loc).with_suffix('.sd.nc'))

    if sub.id == 3 or sub.id == 5:
        xmin, ymin, xmax, ymax = sub.geometry.bounds
        from shapely.geometry import box
        subset = lidar.rio.clip_box(*box(xmin + 0.02, ymin, xmax - 0.02, ymax).bounds)
    else:
        subset = lidar.rio.clip_box(*sub.geometry.bounds)
    subset = subset.sortby('y')
    if year == '2020':
        if loc != 'Dry_Creek':
            lidar_sub = subset['lidar-sd'].sel(time = subset.lidar_times[0])
        else:
            lidar_sub = subset['lidar-sd'].sel(time = subset.lidar_times)
        
        if loc == 'Mores':
            uv_sub = subset['232-sd_delta_unw'].sel(band = 'VV').isel(time = 3)
            t1, t2 = subset['232-sd_delta_unw'].sel(band = 'VV').isel(time = 3)['time'], subset['232-sd_delta_int'].sel(band = 'VV').isel(time = 3)['time2']
        else:
            uv_sub = subset['232-sd_delta_unw'].sel(band = 'VV').isel(time = 2)
            t1, t2 = subset['232-sd_delta_unw'].sel(band = 'VV').isel(time = 2)['time'], subset['232-sd_delta_int'].sel(band = 'VV').isel(time = 2)['time2']
            
    if year == '2021':
        lidar_sub = subset['lidar-sd'].sel(time = subset.lidar_times[1])
        uv_sub = subset['232-sd_delta_unw'].sel(band = 'VV').isel(time = 7)
        t1, t2 = subset['232-sd_delta_unw'].sel(band = 'VV').isel(time = 7)['time'], subset['232-sd_delta_int'].sel(band = 'VV').isel(time = 7)['time2']
    
    t1, t2 = t1.dt.strftime('%Y-%m-%d').values, t2.dt.strftime('%m-%d').values

    # filter
    lidar_sub.data = gaussian_filter(lidar_sub.data, 3)

    uv_sub = uv_sub.where(~lidar_sub.isnull())
    uv_sub = uv_sub.interpolate_na(method = 'quadratic', limit = 10, dim = 'x')
    uv_sub = uv_sub.interpolate_na(method = 'quadratic', limit = 10, dim = 'y')
    uv_sub.data = gaussian_filter(uv_sub.data, 1)

    lidar_sub.plot(ax = axes[i, 0], vmin = lidar_sub.quantile(0.01), vmax = lidar_sub.quantile(0.999), cmap = 'Blues', cbar_kwargs={'label': None})
    uv_sub.plot(ax = axes[i, 1], vmin = uv_sub.quantile(0.01), vmax = uv_sub.quantile(0.99), cmap = 'Blues', cbar_kwargs={'label': None})

    # get r and n
    rmse, r, n, bias = get_stats(lidar_sub, uv_sub, bias = True)
    axes[i, 2].text(.99, .01, f'r: {r:.2}\nn = {n:.1e}', ha = 'right', va = 'bottom', transform = axes[i, 2].transAxes, color = 'white', fontsize = 12)

    xs, ys = clean_xs_ys(lidar_sub, uv_sub)
    axes[i, 2].hist2d(xs, ys, bins = 80, range = [np.quantile(xs, [0.01, 0.99]), np.quantile(ys, [0.01, 0.99])])

    # if i == 0:
    #     axes[0, 0].set_title(f"Lidar", weight = 'bold')
    #     # axes[0, 2].set_title('')
    # else:
    #     axes[i, 0].set_title('')
    axes[i, 0].set_title(f"{year} Lidar", weight = 'bold')
    axes[i, 1].set_title(f"{t1} to {t2} UAVSAR", weight = 'bold')

    axes[i, 0].set_ylabel(f"{loc.replace('_', ' ')}", weight="bold")
    axes[i, 1].set_ylabel('')

    import matplotlib.font_manager as fm
    fontprops = fm.FontProperties(size=12)

    scalebar = AnchoredSizeBar(axes[i, 0].transData,
                           0.009009009009009009, '1 km', 'upper left', 
                           pad=0.1,
                           color='black',
                           frameon=False,
                           size_vertical=0.00003,
                           fontproperties=fontprops)

    axes[i, 0].add_artist(scalebar)

    

for ax in axes[:, :2].ravel():
    ax.set_xlabel('')
    # ax.set_ylabel('')
    ax.set_xticks([])
    ax.set_yticks([])

for ax in axes[:, 2].ravel():
    ax.xaxis.set_major_locator(plt.MaxNLocator(3))
    ax.yaxis.set_major_locator(plt.MaxNLocator(3))

plt.tight_layout()
plt.savefig(figdir.joinpath('lidar', 'subset', 'quad-lidar-img.png'))

# failed ifgures

fig, axes = plt.subplots(2, 3, figsize = (12, 8))

for i, fp in enumerate(Path('/bsuhome/zacharykeskinen/scratch/data/uavsar/ncs/lidar/').glob('*.sd.nc')):

    # if i != 0:
    #     continue

    print(fp)
    ds = xr.open_dataset(fp).load()
    if isinstance(ds.lidar_times, str):
        ds['lidar_times'] = [ds.lidar_times]

    for j, time in enumerate(ds.lidar_times):
        ax = axes[j, i]
        lidar = ds['lidar-sd'].sel(time = time)
        # veg = ds['lidar-vh'].sel(time = time)
        if isinstance(time, str):
            year = pd.to_datetime(time).year
        else:
            year = pd.to_datetime(time.data).year


        uavsar = ds.sel(time = ds.time.dt.year == year)
        uavsar = uavsar.isel(time = uavsar['snotel_dSWE'] > 0.01 )
        short_baseline = pd.to_timedelta(uavsar['time2'] - uavsar['time']) < '15 days'
        uavsar = uavsar.isel(time = short_baseline)
        imgs = uavsar.time.size
        uavsar = uavsar['232-sd_delta_int'].sel(band = 'VV').sum(dim = 'time')
        uavsar.data = gaussian_filter(uavsar.data, 3)

        lidar, uavsar = clean_xs_ys(lidar, uavsar, clean_zeros = True)

        uavsar = uavsar[lidar > 0]
        lidar = lidar[lidar > 0]

        rmse, r, n = get_stats(lidar, uavsar)
        x05, x99 = np.quantile(lidar, [0.001, 0.9999])
        y05, y95 = np.quantile(uavsar, [0.001, 0.9999])

        range = [[0, x99], [y05, y95]]
        range = [[0, 5.5], [-0.1, 0.45]]

        ax.hist2d(x = lidar, y = uavsar, bins = 100, norm = mpl.colors.LogNorm(), range = range)
        ax.text(.99, .99, f'r: {r:.2}\nn = {n:.1e}', ha = 'right', va = 'top', transform = ax.transAxes)
        ax.set_xlim(0, x99)
        ax.set_title(f"{fp.stem.split('.')[0].replace('_',' ')} {year} - {imgs} images")

        x05, x95 = np.quantile(lidar, [0.05, 0.95])
        y05, y95 = np.quantile(uavsar, [0.05, 0.95])

        axes[1, 2].plot([x05, x95], [np.median(uavsar), np.median(uavsar)], label = f"{fp.stem.split('.')[0].replace('_',' ')} {year}")
        axes[1, 2].plot([np.median(lidar), np.median(lidar)], [y05, y95])

for ax in axes.ravel():
    ax.xaxis.set_major_locator(plt.MaxNLocator(2))
    ax.yaxis.set_major_locator(plt.MaxNLocator(2))
    ax.set_xlim(0, 5.5)
    ax.set_ylim(-0.1, 0.45)
axes[1,2].legend()
axes[1,2].set_title('All Site 95% Confidence Intervals')
axes[1,2].set_xlim(0, 3)

fig, axes = plt.subplots(2, 3, figsize = (12, 8))

for i, fp in enumerate(Path('/bsuhome/zacharykeskinen/scratch/data/uavsar/ncs/lidar/').glob('*.sd.nc')):

    # if i != 0:
    #     continue

    print(fp)
    ds = xr.open_dataset(fp).load()
    if isinstance(ds.lidar_times, str):
        ds['lidar_times'] = [ds.lidar_times]

    for j, time in enumerate(ds.lidar_times):
        ax = axes[j, i]
        lidar = ds['lidar-sd'].sel(time = time)
        # veg = ds['lidar-vh'].sel(time = time)
        if isinstance(time, str):
            year = pd.to_datetime(time).year
        else:
            year = pd.to_datetime(time.data).year


        uavsar = ds.sel(time = ds.time.dt.year == year)
        uavsar = uavsar.isel(time = uavsar['snotel_dSWE'] > 0.01 )
        short_baseline = pd.to_timedelta(uavsar['time2'] - uavsar['time']) < '15 days'
        uavsar = uavsar.isel(time = short_baseline)
        imgs = uavsar.time.size
        uavsar = uavsar['232-sd_delta_int'].sel(band = 'VV').sum(dim = 'time')
        uavsar.data = gaussian_filter(uavsar.data, 3)

        lidar, uavsar = clean_xs_ys(lidar, uavsar, clean_zeros = True)

        uavsar = uavsar[lidar > 0]
        lidar = lidar[lidar > 0]

        uavsar = (np.median(uavsar) - uavsar) / np.median(uavsar)
        lidar = (np.median(lidar) - lidar) / np.median(lidar)

        rmse, r, n = get_stats(lidar, uavsar)
        x05, x99 = np.quantile(lidar, [0.001, 0.9999])
        y05, y95 = np.quantile(uavsar, [0.001, 0.9999])

        range = [[x05, x99], [y05, y95]]
        # range = [[0, 5.5], [-0.1, 0.45]]

        ax.hist2d(x = lidar, y = uavsar, bins = 100, norm = mpl.colors.LogNorm()) # range = range
        ax.text(.99, .99, f'r: {r:.2}\nn = {n:.1e}', ha = 'right', va = 'top', transform = ax.transAxes)
        # ax.set_xlim(0, x99)
        ax.set_title(f"{fp.stem.split('.')[0].replace('_',' ')} {year} - {imgs} images")

        x05, x95 = np.quantile(lidar, [0.05, 0.95])
        y05, y95 = np.quantile(uavsar, [0.05, 0.95])
        # axes[1, 2].errorbar(np.median(lidar), np.median(uavsar), xerr = np.array([x05, x99]).reshape(2, 1), yerr = np.array([y05, y95]).reshape(2, 1), label = f"{fp.stem.split('.')[0].replace('_',' ')} {year}")
        axes[1, 2].plot([x05, x95], [np.median(uavsar), np.median(uavsar)], label = f"{fp.stem.split('.')[0].replace('_',' ')} {year}")
        axes[1, 2].plot([np.median(lidar), np.median(lidar)], [y05, y95])

# for ax in axes.ravel():
    # ax.xaxis.set_major_locator(plt.MaxNLocator(2))
    # ax.yaxis.set_major_locator(plt.MaxNLocator(2))
    # ax.set_xlim(0, 5.5)
    # ax.set_ylim(-0.1, 0.45)
axes[1,2].legend()
axes[1,2].set_title('All Site 95% Confidence Intervals')
# axes[1,2].set_xlim(0, 3)