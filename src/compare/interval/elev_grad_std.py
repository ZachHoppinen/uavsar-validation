from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import rioxarray as rxa
import matplotlib.pyplot as plt
import shapely

from uavsar_pytools.snow_depth_inversion import phase_from_depth, depth_from_phase

ncs_dir = Path('/bsuhome/zacharykeskinen/scratch/data/uavsar/ncs')
ds = xr.open_dataset(ncs_dir.joinpath('final_uv_geom_atm_veg_model_v1.nc')) # .isel(x = slice(0, -1, 50), y = slice(0, -1, 50))
ds = ds.sortby('time1')

data_dir = Path('/bsuhome/zacharykeskinen/uavsar-validation/data')
insitu_dir = data_dir.joinpath('insitu')
insitu = pd.read_parquet(insitu_dir.joinpath('all_insitu.parq'))
boards = pd.read_parquet(insitu_dir.joinpath('storm_boards.parq'))
boards = gpd.GeoDataFrame(boards, geometry=gpd.points_from_xy(boards.longitude, boards.latitude), crs="EPSG:4326")
# add inc
for i, r in boards.iterrows():
    boards.loc[i, 'inc'] = ds.sel(x = r.geometry.x, y = r.geometry.y, method = 'nearest')['inc'].data.ravel()[0]
    boards.loc[i, 'elev'] = ds.sel(x = r.geometry.x, y = r.geometry.y, method = 'nearest')['dem'].data.ravel()[0]

interval_dir = Path('/bsuhome/zacharykeskinen/uavsar-validation/figures/interval')

fig, axes = plt.subplots(4, 2, figsize = (12, 8))
i = 0
for t1, t2 in zip(ds.time1.data, ds.time2.data):
    flight_boards = boards.loc[(boards.date_t2 <= t2 + pd.Timedelta('2 days')) & (boards.date_t1 >= t1 - pd.Timedelta('2 days')), :].copy()
    for site_id in ['BB','LR', 'MC']:
        flight_boards_site = flight_boards.loc[flight_boards.site.str.contains(site_id), :].copy()
        flight_boards_site = flight_boards_site.dropna(subset=['hn','den','inc'])
        xmin, ymin, xmax, ymax = flight_boards_site.total_bounds
        ds_site = ds.sel(x = slice(xmin, xmax), y = slice(ymax, ymin), time1 = t1)
        ds_site = ds_site.where((ds_site['cor'] > 0.5))
        if ds_site.x.size > 2 and ds_site.y.size > 2 and len(flight_boards_site) > 4 and ds_site['int_phase'].isnull().sum() != ds_site['int_phase'].size:
            flight_boards_site = flight_boards_site.sort_values('elev')

            board_phase = phase_from_depth(flight_boards_site.hn.values, inc_angle=flight_boards_site.inc.values, density = flight_boards_site.den.values)
            board_phase = board_phase - board_phase[0]

            flight_boards_site.loc[:, 'phase'] = board_phase

            elev_phase = ds_site.groupby_bins('dem', np.arange(flight_boards_site.elev.min()-50, flight_boards_site.elev.max() + 250, 100))
            
            elev_phase_var = elev_phase.std(skipna = True)['int_phase']
            elev_phase_mean = elev_phase.mean(skipna = True)
            
            elev_phase_mean = elev_phase_mean - elev_phase_mean.isel(dem_bins = 0)
            # elev_phase = elev_phase.rolling({'dem_bins': 3}, min_periods=1).mean()

            ax = axes.ravel()[i]
            # elev_phase['int_phase'].plot(ax = ax, label = 'Uncorrected Wrapped Phase')
            atmospheric = (elev_phase_mean['int_phase'] - elev_phase_mean['delay'])

            atmospheric = atmospheric.dropna(dim = 'dem_bins')
            elev_phase_var = elev_phase_var.dropna(dim = 'dem_bins')
            ax.fill_between([v.mid for v in atmospheric.dem_bins.values], (atmospheric - elev_phase_var).data, (atmospheric + elev_phase_var).data, color = 'red', alpha = 0.5, label = 'Wrapped Phase')

            if i == 0:
                flight_boards_site.plot.scatter(x = 'elev', y = 'phase', ax = ax, label = 'Interval Board Expected Phase')
                ax.legend()
            else:
                flight_boards_site.plot.scatter(x = 'elev', y = 'phase', ax = ax)
            ax.set_title(site_id)
            ax.set_xlabel('')
            i += 1
plt.tight_layout()
# plt.legend(loc = 'upper left')
plt.savefig(interval_dir.joinpath('std_elev_grad.png'))