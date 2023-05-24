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

from stats import clean_xs_ys

xr.set_options(keep_attrs = True)

data_dir = Path('/bsuhome/zacharykeskinen/scratch/data/uavsar')
ncs_dir = data_dir.joinpath('ncs')
fig_dir = Path('/bsuhome/zacharykeskinen/uavsar-validation/figures/model')

ds = xr.open_dataset(ncs_dir.joinpath('final_insitu.nc'))# .isel(x = slice(0, -1, 100), y = slice(0, -1, 100))

ds_10 = ds.rolling(x = 10, y = 10).mean()
ds_10.to_netcdf(ncs_dir.joinpath('final_10_10.nc'))

ds_50 = ds.rolling(x = 50, y = 50).mean()
ds_50.to_netcdf(ncs_dir.joinpath('final_50_50.nc'))

sub = ds[['dem', 'int_phase','int_atm','unw_atm', 'model_d_swe','model_swe','cor', 'tree_perc', 'inc']]
sub = sub.where((sub['model_swe'].min(dim = 'time') > 0) & (~sub['cor'].mean(dim = 'time1').isnull()))

model_phase = phase_from_depth(sub['model_d_swe']*997/250, sub['inc'], density = 250)

sub['int_atm'] = sub['int_atm'] - (sub['int_atm'].mean(dim = ['x', 'y']) - model_phase.mean(dim = ['x', 'y']))
sub['unw_atm'] = sub['unw_atm'] - (sub['unw_atm'].mean(dim = ['x', 'y']) - model_phase.mean(dim = ['x', 'y']))

model_phase = phase_from_depth(sub['model_d_swe']*997/250, sub['inc'], density = 250)
fig, axes = plt.subplots(2, 2, figsize = (12, 8))
xskill.pearson_r(sub['int_atm'], model_phase, dim = 'time1', skipna = True).plot(vmax = 1, vmin = 0, ax= axes[0,0])
axes[0,0].set_title('Pearson $r^{2}$')
sub['cor'].mean(dim = 'time1').plot(ax = axes[0,1])
axes[0,1].set_title('Temporal Coherence')
sub['tree_perc'].plot(ax = axes[1, 0])
axes[1, 0].set_title('Tree Percentage')
sub['dem'].plot(ax = axes[1,1])
axes[1,1].set_title('Elevation')
plt.tight_layout()
plt.savefig(fig_dir.joinpath('r2_maps.png'))

int_sd = depth_from_phase(sub['unw_atm'], sub['inc'], density = 250)*250/997
fig, axes = plt.subplots(2, 2, figsize = (12, 8))
sub['rmse'] = xskill.rmse(sub['model_d_swe'], int_sd, dim = 'time1', skipna = True)
sub['rmse'].plot(ax= axes[0,0], vmax = 0.08)
axes[0,0].set_title('RMSE SWE (m)')
sub['cor'].mean(dim = 'time1').plot(ax = axes[0,1])
axes[0,1].set_title('Temporal Coherence')
sub['tree_perc'].plot(ax = axes[1, 0])
axes[1, 0].set_title('Tree Percentage')
sub['dem'].plot(ax = axes[1,1])
axes[1,1].set_title('Elevation')
plt.tight_layout()
plt.savefig(fig_dir.joinpath('rmse_maps.png'))

fig, axes = plt.subplots(2,2, figsize = (12, 8))
dem_group = sub.groupby_bins('dem', np.arange(1600, 2700, 200)).mean()
dem_group['rmse'].plot(ax = axes[0,0])
axes[0,0].set_title('Elevation (m)')

tree_group = sub.groupby_bins('tree_perc', np.arange(0, 100, 10)).mean()
tree_group['rmse'].plot(ax = axes[0, 1])
axes[0, 1].set_title('Tree Percentage')

cor_group = sub.groupby_bins('cor', np.arange(0, 1, 0.1)).mean()
cor_group['rmse'].plot(ax = axes[1,0 ])
axes[1, 0].set_title('Coherence')

inc_group = sub.groupby_bins('inc', np.arange(0, np.deg2rad(90), np.deg2rad(5))).mean()
inc_group['rmse'].plot(ax = axes[1,1])
axes[1,1].set_title('Incidence Angle (deg)')
# convert to degrees
tick_locs = np.deg2rad([0, 30, 60, 90])
tick_lbls = [0, 30, 60, 90]
plt.xticks(tick_locs, tick_lbls)

for ax in axes[:,0]:
    ax.set_ylabel('RMSE')

for ax in axes[:,1]:
    ax.set_ylabel('')

for ax in axes.ravel():
    ax.set_xlabel('')

plt.tight_layout()

plt.savefig(fig_dir.joinpath('rmse_binned.png'))