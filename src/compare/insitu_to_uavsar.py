import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from stats import get_stats
df_full = pd.read_parquet('/bsuhome/zacharykeskinen/uavsar-validation/data/insitu/insitu_uavsar.parq')
df_full = df_full.loc[df_full.site_name != 'jackson']

fig, axes = plt.subplots(1, 2, figsize = (12,8))

symbol_dic = {'NRCS':'o', 'SnowEx':'x'}

for source in ['NRCS', 'SnowEx']:
    df = df_full.loc[df_full.datasource == source]
    symbol = symbol_dic[source]

    for j, img_type in enumerate(['int', 'unw']):
        for i, band in enumerate(['VV']):
            if source == 'NRCS':
                df.plot.scatter(x = 'dSD', y = f'UV_{img_type}_{band}_sd', ax = axes[j], label = f'Snotel', marker = symbol, color = f'C{i}')
            else:
                df.plot.scatter(x = 'dSD', y = f'UV_{img_type}_{band}_sd', ax = axes[j], label = f'Snowpit', marker = symbol, color = f'C{i}')

axes[0].set_title('Insitu vs Wrapped UAVSAR Snow Depth Changes')
axes[0].set_xlabel('Insitu')
axes[0].set_ylabel('Wrapped UAVSAR SD')
axes[0].legend(loc = 'lower right')


axes[1].set_title('Insitu vs Unwrapped UAVSAR Snow Depth Changes')
axes[1].set_xlabel('Insitu')
axes[1].set_ylabel('Unwrapped UAVSAR SD')
axes[1].legend(loc = 'lower right')

# add RMSE, R, N to fig
rmse, r, n = get_stats(df.dSD.values.ravel(), df.UV_int_VV_sd.values.ravel())
axes[0].text(.01, .99, f'RMSE: {rmse:.3}, r: {r:.2}\nn = {n}', ha='left', va='top', transform=axes[0].transAxes)

rmse, r, n = get_stats(df.dSD.values.ravel(), df.UV_unw_VV_sd.values.ravel())
axes[1].text(.01, .99, f'RMSE: {rmse:.3}, r: {r:.2}\nn = {n}', ha='left', va='top', transform=axes[1].transAxes)


plt.savefig('/bsuhome/zacharykeskinen/uavsar-validation/figures/insitu/insitu_uavsar_scatterv2.png')