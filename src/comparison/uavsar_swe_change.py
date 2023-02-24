from pathlib import Path
from glob import glob
import xarray as xr
import pandas as pd

from insitu import get_uavsar_insitu

df = pd.read_parquet('/bsuhome/zacharykeskinen/uavsar-validation/data/insitu/all_insitu.parq')

ncs_fp = Path('/bsuhome/zacharykeskinen/scratch/data/uavsar/ncs/')
a = []
for fp in ncs_fp.glob('*-*.nc'):
    if '.insitu.' not in str(fp):
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
        img.to_netcdf(fp.with_suffix('.insitu.nc'))