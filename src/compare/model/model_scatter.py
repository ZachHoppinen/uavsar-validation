from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import rioxarray as rxa
import matplotlib as mpl
import matplotlib.pyplot as plt
from pyproj import Transformer
import xskillscore as xskill

from uavsar_pytools.snow_depth_inversion import phase_from_depth, depth_from_phase

from stats import clean_xs_ys, get_stats

xr.set_options(keep_attrs = True)

data_dir = Path('/bsuhome/zacharykeskinen/scratch/data/uavsar')
ncs_dir = data_dir.joinpath('ncs')
fig_dir = Path('/bsuhome/zacharykeskinen/uavsar-validation/figures/model')

ds = xr.open_dataset(ncs_dir.joinpath('final_10_10.nc'))

insitu = pd.read_parquet('/bsuhome/zacharykeskinen/uavsar-validation/data/insitu/all_difference.parq')
insitu = insitu[insitu.site_name != 'jackson']
insitu = gpd.GeoDataFrame(insitu, geometry = gpd.points_from_xy(insitu.lon, insitu.lat), crs="EPSG:4326")
# insitu = insitu.set_index('time1')

interval = pd.read_parquet('/bsuhome/zacharykeskinen/uavsar-validation/data/insitu/storm_boards.parq')
interval = gpd.GeoDataFrame(interval, geometry = gpd.points_from_xy(interval.longitude, interval.latitude), crs="EPSG:4326")

sub = ds[['dem', 'int_phase', 'unw', 'model_d_swe','cum_melt','cor', 'tree_perc','tree_height','model_swe', 'inc', 'delay']]
sub = sub.where((ds['model_swe'].min(dim = 'time') > 0) & (~sub['cor'].mean(dim = 'time1').isnull()))

# get insitu info
densities = np.zeros(len(ds.time1.data))
phases = np.zeros(len(ds.time1.data))
for i, time in enumerate(ds.time1.data):
    insitu_t = insitu[insitu.time1 == time]
    density = np.mean(insitu_t[['t1_density', 't2_density']].mean(axis = 1))
    densities[i] = density
    phases[i] = np.mean(phase_from_depth(insitu_t['dSWE']*997/density, insitu_t['inc'], density = insitu_t[['t1_density', 't2_density']].mean(axis = 1)))
densities = xr.DataArray(densities,
    coords = {'time1':ds.time1.data})
phases = xr.DataArray(phases,
    coords = {'time1':ds.time1.data})

# sub = sub.where(sub['cor']>0.45)
# sub = sub.drop_sel(time1 = '2021-03-16T17:50:00.000000000')
sub['unw_atm'] = sub['unw'] - sub['delay']

# sub['unw_atm'] = sub['unw_atm'] - (sub['unw_atm'].mean(dim = ['x','y']) - phases)

model_phases = phase_from_depth(sub['model_d_swe']*997/densities, sub['inc'], density = densities)
sub['unw_atm'] = sub['unw_atm'] - (sub['unw_atm'].mean(dim = ['x','y']) - model_phases.mean(dim = ['x','y']))

unw_d_swe = depth_from_phase(sub['unw_atm'], sub['inc'], density = densities) * densities / 997
unw_d_swe = unw_d_swe.rolling(x = 10, y = 10).mean()
fig, axes = plt.subplots(3,1, figsize = (8, 12))

xs, ys = clean_xs_ys(unw_d_swe.data.ravel(), sub['model_d_swe'].data.ravel())
xs_dry, ys_dry = clean_xs_ys(unw_d_swe.data.ravel(), sub['model_d_swe'].where((sub['cum_melt'] < 0.01)).data.ravel())

best = sub.where((sub['cum_melt'] < 0.01) & (sub['tree_perc'] < 20))

model_phases = phase_from_depth(best['model_d_swe']*997/densities, best['inc'], density = densities)
best['unw_atm'] = best['unw_atm'] - (best['unw_atm'].mean(dim = ['x','y']) - model_phases.mean(dim = ['x','y']))
# best['unw_atm'] = best['unw_atm'] - (best['unw_atm'].mean(dim = ['x','y']) - phases)

unw_d_swe = depth_from_phase(best['unw_atm'], best['inc'], density = densities) * densities / 997
unw_d_swe = unw_d_swe.rolling(x = 10, y = 10).mean()

xs_best, ys_best = clean_xs_ys(unw_d_swe.data.ravel(), best['model_d_swe'].data.ravel())
for i, (ax, [xs, ys]) in enumerate(zip(axes, [[xs, ys], [xs_dry, ys_dry], [xs_best, ys_best]])):

    # ax.hist2d(xs, ys, bins = 100, norm=mpl.colors.LogNorm(), cmap=mpl.cm.inferno, range = [[-0.1, 0.075],[-0.1, 0.075]]) # , norm=mpl.colors.LogNorm(), cmap=mpl.cm.inferno
    # ax.scatter(xs, ys, alpha = 0.1, s = 1)
    ax.set_xlabel('Snow Model SWE Change')
    ax.set_ylabel('UAVSAR SWE Change')
    if i == 0:

        ax.hist2d(xs, ys, bins = 150, norm=mpl.colors.LogNorm(), cmap=mpl.cm.inferno, range = [[-0.1, 0.075],[-0.1, 0.075]])
        ax.set_xlim(-0.1, 0.075)
        ax.set_ylim(-0.1, 0.075)
    elif i == 1:
        ax.hist2d(xs, ys, bins = 100, norm=mpl.colors.LogNorm(), cmap=mpl.cm.inferno, range = [[-0.03, 0.075],[-0.03, 0.075]])
        ax.set_xlim(-0.1, 0.075)
        ax.set_ylim(-0.1, 0.075)
    else:
        ax.hist2d(xs, ys, bins = 100, norm=mpl.colors.LogNorm(), cmap=mpl.cm.inferno, range = [[-0.03, 0.075],[-0.03, 0.075]])
        ax.set_xlim(-0.1, 0.075)
        ax.set_ylim(-0.1, 0.075)
    ax.plot([-0.1,0.2], [-0.1,0.2], color = 'blue', linestyle = 'dashed')
    rmse, r, n, bias = get_stats(xs, ys, bias = True)
    ax.text(s = f'rmsd: {rmse:.3f}, r: {r:.2f}\nn: {n:.2e}', x = 0.01, y= 0.99 , ha = 'left', va = 'top', transform = ax.transAxes)

# axes[0].text(s = 'A', x = 0.99,y= 0.94 , ha = 'right', va = 'top', weight = 'bold', transform = axes[0].transAxes)
# axes[1].text(s = 'B', x = 0.99,y= 0.94 , ha = 'right', va = 'top',weight = 'bold',transform = axes[1].transAxes)


axes[0].set_title('SNOWMODEL vs UAVSAR Retrieved SWE')
axes[1].set_title('SNOWMODEL vs UAVSAR Retrieved SWE for Modeled Dry Snow')
axes[2].set_title('SNOWMODEL vs UAVSAR Retrieved SWE for Dry Snow in Treeless Regions')

plt.tight_layout()

plt.savefig(fig_dir.joinpath('uavsar_snowmodel_scatter_unw_v2.png'))