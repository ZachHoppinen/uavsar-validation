import xarray as xr
import numpy as np
import pandas as pd
import rioxarray as rxa
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib as mpl
import pandas as pd

# loop through each flight pair and calculate rmse, r2, plot
lidar = None

lidar_dir = Path('/bsuhome/zacharykeskinen/scratch/data/uavsar/ncs/lidar')

for lidar_fp in lidar_dir.glob('*.sd.nc'):
    print(lidar_fp)
    lidar = xr.open_dataset(lidar_fp)

    if isinstance(lidar.attrs['lidar_times'] , str):
        lidar.attrs['lidar_times'] = [lidar.attrs['lidar_times']]

    for t in lidar.attrs['lidar_times']:

        try:
            print(t)

            t = pd.to_datetime(t)
            ds = lidar.sel(time = slice(t - pd.Timedelta('180 days'), t))

            cum_sd232 = ds['dSD_232_unw'].sum(dim = 'time').where(~ds['lidar-sd'].sel(time = t).isnull())
            cum_sd052 = ds['dSD_052_unw'].sum(dim = 'time').where(~ds['lidar-sd'].sel(time = t).isnull())

            vmax = cum_sd232.sel(band = 'VV').quantile(0.9)

            if cum_sd052.sum() > 0:
                fig, axes = plt.subplots(1, 4, figsize = (20,10))
                cum_sd232.sel(band = 'VV').plot(vmin = 0, vmax = vmax, ax = axes[1], cmap = 'cividis')
                cum_sd052.sel(band = 'VV').plot(vmin = 0, vmax = vmax, ax = axes[2], cmap = 'cividis')
                ax = axes[3]
            else:
                fig, axes = plt.subplots(1, 3, figsize = (20,10))
                cum_sd232.sel(band = 'VV').plot(vmin = 0, vmax = vmax, ax = axes[1], cmap = 'cividis')
                ax = axes[2]

            ds['lidar-sd'].sel(time = t).plot(vmin = 0, vmax = 3, ax = axes[0])

            xs = lidar['lidar-sd'].sel(time = t).values.ravel()
            ys = cum_sd232.sel(band = 'VV').values.ravel()

            xs_tmp = xs[(~np.isnan(xs)) & (~np.isnan(ys)) & (ys != 0)]
            ys = ys[(~np.isnan(xs)) & (~np.isnan(ys)) & (ys != 0)]
            xs = xs_tmp

            range = [[-.2, 5.5], [-.2, np.max(ys) + 0.3]]
            ax.hist2d(xs_tmp, ys, bins = 100, norm=mpl.colors.LogNorm(), cmap=mpl.cm.inferno, range = range)

            from scipy.stats import pearsonr
            r, p = pearsonr(xs, ys)
            ax.text(.01, .99, f'r: {r:.2}\nn = {len(ys):.2e}', ha = 'left', va = 'top', transform = ax.transAxes)

            plt.savefig(f"/bsuhome/zacharykeskinen/uavsar-validation/figures/lidar/all/unw_{lidar.attrs['site']}_{t.strftime('%Y-%m-%d')}.png")
        except ValueError:
            pass

# INT NOW

import xarray as xr
import numpy as np
import pandas as pd
import rioxarray as rxa
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib as mpl
import pandas as pd

# loop through each flight pair and calculate rmse, r2, plot
lidar = None

lidar_dir = Path('/bsuhome/zacharykeskinen/scratch/data/uavsar/ncs/lidar')

for lidar_fp in lidar_dir.glob('*.sd.nc'):
    print(lidar_fp)
    lidar = xr.open_dataset(lidar_fp)

    if isinstance(lidar.attrs['lidar_times'] , str):
        lidar.attrs['lidar_times'] = [lidar.attrs['lidar_times']]

    for t in lidar.attrs['lidar_times']:

        try:
            print(t)

            t = pd.to_datetime(t)
            ds = lidar.sel(time = slice(t - pd.Timedelta('180 days'), t))

            cum_sd232 = ds['dSD_232_int'].sum(dim = 'time').where(~ds['lidar-sd'].sel(time = t).isnull())
            cum_sd052 = ds['dSD_052_int'].sum(dim = 'time').where(~ds['lidar-sd'].sel(time = t).isnull())

            vmax = cum_sd232.sel(band = 'VV').quantile(0.9)

            if cum_sd052.sum() > 0:
                fig, axes = plt.subplots(1, 4, figsize = (20,10))
                cum_sd232.sel(band = 'VV').plot(vmin = 0, vmax = vmax, ax = axes[1], cmap = 'cividis')
                cum_sd052.sel(band = 'VV').plot(vmin = 0, vmax = vmax, ax = axes[2], cmap = 'cividis')
                ax = axes[3]
            else:
                fig, axes = plt.subplots(1, 3, figsize = (20,10))
                cum_sd232.sel(band = 'VV').plot(vmin = 0, vmax = vmax, ax = axes[1], cmap = 'cividis')
                ax = axes[2]

            ds['lidar-sd'].sel(time = t).plot(vmin = 0, vmax = 3, ax = axes[0])

            xs = lidar['lidar-sd'].sel(time = t).values.ravel()
            ys = cum_sd232.sel(band = 'VV').values.ravel()

            xs_tmp = xs[(~np.isnan(xs)) & (~np.isnan(ys)) & (ys != 0)]
            ys = ys[(~np.isnan(xs)) & (~np.isnan(ys)) & (ys != 0)]
            xs = xs_tmp

            range = [[-.2, 5.5], [-.2, np.max(ys) + 0.3]]
            ax.hist2d(xs_tmp, ys, bins = 100, norm=mpl.colors.LogNorm(), cmap=mpl.cm.inferno, range = range)

            from scipy.stats import pearsonr
            r, p = pearsonr(xs, ys)
            ax.text(.01, .99, f'r: {r:.2}\nn = {len(ys):.2e}', ha = 'left', va = 'top', transform = ax.transAxes)

            plt.savefig(f"/bsuhome/zacharykeskinen/uavsar-validation/figures/lidar/all/int_{lidar.attrs['site']}_{t.strftime('%Y-%m-%d')}.png")
        except ValueError:
            pass

# now plot all individual images pairs

import xarray as xr
import numpy as np
import pandas as pd
import rioxarray as rxa
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib as mpl
import pandas as pd
import os

# loop through each flight pair and calculate rmse, r2, plot
lidar = None

lidar_dir = Path('/bsuhome/zacharykeskinen/scratch/data/uavsar/ncs/lidar')

for lidar_fp in lidar_dir.glob('*.all.nc'):

    print(lidar_fp)
    lidar = xr.open_dataset(lidar_fp)

    if isinstance(lidar.attrs['lidar_times'] , str):
        lidar.attrs['lidar_times'] = [lidar.attrs['lidar_times']]

    for img_type in ['int', 'unw']:

        for t in lidar.attrs['lidar_times']:
            print(t)

            t = pd.to_datetime(t)
            ds = lidar.sel(time = slice(t - pd.Timedelta('180 days'), t))

            os.makedirs(f"/bsuhome/zacharykeskinen/uavsar-validation/figures/lidar/sub-images/{img_type}_{ds.attrs['site']}", exist_ok = True)

            print(''.center(50, '-'))
            for sub_t in ds.time:

                out_fp = f"/bsuhome/zacharykeskinen/uavsar-validation/figures/lidar/sub-images/{img_type}_{ds.attrs['site']}/{img_type}-{ds.attrs['site']}-{pd.to_datetime(sub_t.values).strftime('%Y-%m-%d')}.png"
                
                if Path(out_fp).exists():
                    continue

                print(sub_t.data)
                fig, axes = plt.subplots(1, 4, figsize = (20, 10))

                sub = ds.sel(time = sub_t)[f'dSD_232_{img_type}'].sel(band = 'VV')
                cor = ds.sel(time = sub_t)['232_cor'].sel(band = 'VV')

                vmax = sub.quantile(0.9)

                ds['lidar-sd'].sel(time = t).plot(vmin = 0, vmax = 3, ax = axes[0])
                sub.where(~ds['lidar-sd'].sel(time = t).isnull()).plot(vmin = 0, vmax = vmax, ax = axes[1])
                cor.where(~ds['lidar-sd'].sel(time = t).isnull()).plot(vmin = 0, vmax = 1, ax = axes[2])

                ax = axes[3]

                xs = lidar['lidar-sd'].sel(time = t).values.ravel()
                ys = sub.values.ravel()
                cors = cor.values.ravel()

                xs_tmp = xs[(~np.isnan(xs)) & (~np.isnan(ys)) & (ys != 0)]
                ys = ys[(~np.isnan(xs)) & (~np.isnan(ys)) & (ys != 0)]
                xs = xs_tmp

                if len(ys) == 0:
                    plt.clf()
                    continue

                range = [[-.2, 5.5], [-.2, np.max(ys) + 0.3]]
                ax.hist2d(xs_tmp, ys, bins = 100, norm=mpl.colors.LogNorm(), cmap=mpl.cm.inferno, range = range)

                from scipy.stats import pearsonr
                r, p = pearsonr(xs, ys)
                ax.text(.01, .99, f'r: {r:.2}\nn = {len(ys):.2e}', ha = 'left', va = 'top', transform = ax.transAxes)

                plt.savefig(out_fp)
                plt.clf()

# incidence angle removel

df = pd.read_parquet('/bsuhome/zacharykeskinen/uavsar-validation/data/insitu/snotel.parq')
df = df.reset_index(level= 1)
df = df.sort_index()
# loop through each flight pair and calculate rmse, r2, plot
lidar = None

lidar_dir = Path('/bsuhome/zacharykeskinen/scratch/data/uavsar/ncs/lidar')

for lidar_fp in lidar_dir.glob('*.all.nc'):

    print(lidar_fp)
    lidar = xr.open_dataset(lidar_fp)

    if isinstance(lidar.attrs['lidar_times'] , str):
        lidar.attrs['lidar_times'] = [lidar.attrs['lidar_times']]

    for t in lidar.attrs['lidar_times']:
        print(t)

        t = pd.to_datetime(t)
        ds = lidar.sel(time = slice(t - pd.Timedelta('180 days'), t))
        
        lidar_img = ds['lidar-sd'].sel(time = t)

        if len(ds.time) < 2:
            continue

        ds = ds.isel(time = slice(0, len(ds.time)-1))
        ds = add_dSWEs_dTime(ds, df)
        ds = ds.sel(time = ds.dSWE > 0.00254*2) # more than 0.2 inches of swe
        # ds = ds.sel(time = ds.t_delta < pd.Timedelta('10 days'))

        if len(ds.time) < 2:
            continue

        os.makedirs(f"/bsuhome/zacharykeskinen/uavsar-validation/figures/lidar/masked/{ds.attrs['site']}", exist_ok = True)

        for img_type in ['int']:

            out_fp = f"/bsuhome/zacharykeskinen/uavsar-validation/figures/lidar/masked/{ds.attrs['site']}/inc_less_{img_type}-{ds.attrs['site']}-{(t).strftime('%Y-%m-%d')}.png"
            
            # if Path(out_fp).exists():
            #     continue

            fig, axes = plt.subplots(1, 3, figsize = (20, 10))

            sub = ds[f'dSD_232_{img_type}'].sel(band = 'VV')
            cor = ds['232_cor'].sel(band = 'VV')

            sub = sub.sum(dim = 'time')
            cor = cor.mean(dim = 'time')
            inc = ds['232_inc'].isel(time = 0)


            lidar_img.plot(vmin = 0, vmax = 3, ax = axes[0])

            vmin, vmax = sub.quantile([0.2, 0.85]).values
            sub = sub.rolling(x = 5).mean().rolling(y = 5).mean().where(inc < np.deg2rad(90))
            sub.where(~lidar_img.isnull()).plot(vmin = vmin, vmax = vmax, ax = axes[1], cmap = 'viridis')
            # cor.where(~lidar_img.isnull()).plot(vmin = 0, vmax = 1, ax = axes[2])

            ax = axes[2]

            xs = lidar_img.values.ravel()
            ys = sub.values.ravel()
            cors = cor.values.ravel()
            incs = inc.values.ravel()

            xs_tmp = xs[(~np.isnan(xs)) & (~np.isnan(ys)) & (ys != 0) & (incs < np.deg2rad(90))]
            ys = ys[(~np.isnan(xs)) & (~np.isnan(ys)) & (ys != 0) & (incs < np.deg2rad(90))]
            xs = xs_tmp

            if len(ys) == 0:
                plt.clf()
                continue

            range = [[-.2, 5.5], [-.2, np.max(ys) + 0.3]]
            ax.hist2d(xs_tmp, ys, bins = 100, norm=mpl.colors.LogNorm(), cmap=mpl.cm.inferno, range = range)

            from scipy.stats import pearsonr
            r, p = pearsonr(xs, ys)
            ax.text(.01, .99, f'r: {r:.2}\nn = {len(ys):.2e}', ha = 'left', va = 'top', transform = ax.transAxes)

            plt.savefig(out_fp)
            plt.clf()

# combo - use 052 or 232 depending on which has better coherence

df = pd.read_parquet('/bsuhome/zacharykeskinen/uavsar-validation/data/insitu/snotel.parq')
df = df.reset_index(level= 1)
df = df.sort_index()
# loop through each flight pair and calculate rmse, r2, plot
lidar = None

lidar_dir = Path('/bsuhome/zacharykeskinen/scratch/data/uavsar/ncs/lidar')

for lidar_fp in lidar_dir.glob('*.all.nc'):

    print(lidar_fp)
    lidar = xr.open_dataset(lidar_fp)

    if isinstance(lidar.attrs['lidar_times'] , str):
        lidar.attrs['lidar_times'] = [lidar.attrs['lidar_times']]

    for t in lidar.attrs['lidar_times']:
        print(t)

        t = pd.to_datetime(t)
        ds = lidar.sel(time = slice(t - pd.Timedelta('180 days'), t))
        
        lidar_img = ds['lidar-sd'].sel(time = t)

        if len(ds.time) < 2:
            continue

        ds = ds.isel(time = slice(0, len(ds.time)-1))
        ds = add_dSWEs_dTime(ds, df)
        ds = ds.sel(time = ds.dSWE > 0.00254*2) # more than 0.2 inches of swe
        # ds = ds.sel(time = ds.t_delta < pd.Timedelta('10 days'))

        if len(ds.time) < 2:
            continue

        os.makedirs(f"/bsuhome/zacharykeskinen/uavsar-validation/figures/lidar/combo/{ds.attrs['site']}", exist_ok = True)

        for img_type in ['int']:

            out_fp = f"/bsuhome/zacharykeskinen/uavsar-validation/figures/lidar/combo/{ds.attrs['site']}/coh_{img_type}-{ds.attrs['site']}-{(t).strftime('%Y-%m-%d')}.png"
            
            # if Path(out_fp).exists():
            #     continue

            fig, axes = plt.subplots(1, 3, figsize = (20, 10))

            cor232 = ds['232_cor'].sel(band = 'VV')
            cor052 = ds['052_cor'].sel(band = 'VV')
            sub232 = ds[f'dSD_232_{img_type}'].sel(band = 'VV')
            sub052 = ds[f'dSD_052_{img_type}'].sel(band = 'VV')
            
            sub = sub052.where(cor052 > cor232).fillna(sub232.where(cor232 > cor052))

            sub = sub.sum(dim = 'time')

            inc = ds['232_inc'].isel(time = 0)

            lidar_img.plot(vmin = 0, vmax = 3, ax = axes[0])

            vmin, vmax = sub.quantile([0.2, 0.85]).values
            sub = sub.rolling(x = 5).mean().rolling(y = 5).mean()
            sub.where(~lidar_img.isnull()).plot(vmin = vmin, vmax = vmax, ax = axes[1], cmap = 'viridis')
            # cor.where(~lidar_img.isnull()).plot(vmin = 0, vmax = 1, ax = axes[2])

            ax = axes[2]

            xs = lidar_img.values.ravel()
            ys = sub.values.ravel()

            xs_tmp = xs[(~np.isnan(xs)) & (~np.isnan(ys)) & (ys != 0)]
            ys = ys[(~np.isnan(xs)) & (~np.isnan(ys)) & (ys != 0)]
            xs = xs_tmp

            if len(ys) == 0:
                plt.clf()
                continue

            range = [[-.2, 5.5], [-.2, np.max(ys) + 0.3]]
            ax.hist2d(xs_tmp, ys, bins = 100, norm=mpl.colors.LogNorm(), cmap=mpl.cm.inferno, range = range)

            from scipy.stats import pearsonr
            r, p = pearsonr(xs, ys)
            ax.text(.01, .99, f'r: {r:.2}\nn = {len(ys):.2e}', ha = 'left', va = 'top', transform = ax.transAxes)

            plt.savefig(out_fp)
            plt.clf()

# subset out for variables

import numpy as np
import pandas as pd
import xarray as xr
import rioxarray as rxa

from pathlib import Path
from stats import get_stats, clean_xs_ys

from scipy.ndimage import gaussian_filter

from xrspatial import slope, aspect, curvature

import matplotlib as mpl
import matplotlib.pyplot as plt

from uavsar_pytools.snow_depth_inversion import depth_from_phase, phase_from_depth

snotel_locs = {'Dry_Creek.sd': [-116.09685, 43.76377], 'Banner.sd': [-115.23447, 44.30342], 'Mores.sd': [-115.66588, 43.932]}

lidar_dir = Path('/bsuhome/zacharykeskinen/scratch/data/uavsar/ncs/lidar')

for lidar_fp in lidar_dir.glob('*.sd.nc'):
    print(lidar_fp)
    lidar = xr.open_dataset(lidar_fp)

    if isinstance(lidar.attrs['lidar_times'] , str):
        lidar.attrs['lidar_times'] = [lidar.attrs['lidar_times']]
    
    sntl_x, sntl_y = snotel_locs[lidar_fp.stem]
    
    for t in lidar.attrs['lidar_times']:
        fig, axes = plt.subplots(1, 3, figsize = (12, 6))
        print(t)

        t = pd.to_datetime(t)
        ds = lidar.sel(time = slice(t - pd.Timedelta('180 days'), t))
        ds = ds.sel(band = 'VV')

        # trees = ds['lidar-vh'].sel(time = t)
        # elev = ds['lidar-dem']
        # lidar_slope = slope(lidar['lidar-dem'].rio.reproject('EPSG:32611')).rio.reproject_match(lidar['lidar-sd'].sel(time = t))
        # lidar_aspect = aspect(ds['lidar-dem'])
        # lidar_curvature = curvature(ds['lidar-dem'])
        
        tol = 0.00090009 # ~0.001 degrees or ~100m

        for ds_t in ds.time:

            sub = ds.sel(time = ds_t)
            if sub['232-int'].sum() == 0:
                ds['232-sd_delta_int'].loc[dict(time = ds_t)] = np.nan
                continue
            
            if sub['snotel_dSWE'] > 0:
                cur_phase = sub['232-int'].sel(x = slice(sntl_x - tol, sntl_x + tol), y = slice(sntl_y + tol, sntl_y - tol)).mean()
                sntl_inc = sub['232-inc'].sel(x = slice(sntl_x - tol, sntl_x + tol), y = slice(sntl_y + tol, sntl_y - tol)).mean()
                snotel_sd_change = sub['snotel_dSWE'] * (997 / 250)
                sd_phase = phase_from_depth(snotel_sd_change, sntl_inc, density = 250)

                data = depth_from_phase(sub['232-int'] + (sd_phase - cur_phase), sub['232-inc'], density = 250)
                # data = gaussian_filter(data, 3)
                # trees below snow surface?
                # data = data.where(trees > 5)
                # low angle
                data = data.where(lidar_slope < 10)
                # south faces
                # data = data.where((lidar_aspect) > 145 & (lidar_aspect < 220))

                ds['232-sd_delta_int'].loc[dict(time = ds_t)] = data

            else:
                ds['232-sd_delta_int'].loc[dict(time = ds_t)] = np.nan
        
        sum = ds['232-sd_delta_int'].sum(dim = 'time')

        ds['lidar-sd'].loc[dict(time = t)] = gaussian_filter(ds['lidar-sd'].sel(time = t), sigma = 3)
        ds['lidar-sd'].sel(time = t).where(lidar_slope < 10).plot(ax = axes[0], vmin = 0, vmax = 3)

        sum.where(~ds['lidar-sd'].sel(time = t).isnull()).plot(ax = axes[1], vmin = 0, vmax = 3)

        xs, ys = clean_xs_ys(ds['lidar-sd'].sel(time = t).values.ravel(), sum.values.ravel())
        xs_tmp = xs[(xs != 0) & (ys != 0)]
        ys = ys[(xs != 0) & (ys != 0)]
        xs = xs_tmp

        axes[2].hist2d(xs, ys, bins = 100, cmap = mpl.cm.inferno, range = [[0,3.5],[0,3.5]]) # , norm = mpl.colors.LogNorm()

        rmse, r, n = get_stats(xs, ys)
        axes[2].text(.01, .99, f'RMSE: {rmse:.2}, r: {r:.2}\nn = {len(xs):.2e}', ha='left', va='top', transform=axes[2].transAxes, color = 'white')

        plt.show()
        