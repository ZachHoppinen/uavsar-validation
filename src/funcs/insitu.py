import pandas as pd
import xarray as xr

def get_uavsar_insitu(df, img):
    """
    get insitu values for flight1 and flight2

    Args:
    df : dataframe with times in index
    img: xarray dataset with time1 and time2 as attributes.

    Returns:
    flight1_df: Dataframe for snotels on flight 1 and insitu +- 2 days
    flight2_df: Dataframe for snotels on flight 2 and insitu +- 2 days
    """

    assert type(df) == pd.DataFrame
    assert type(img) == xr.Dataset

    t1 = pd.to_datetime(img.attrs['time1']).date()
    t2 = pd.to_datetime(img.attrs['time2']).date()

    f1_df = df.loc[(df.index == t1) & (df.datasource == 'NRCS')]

    time1_start = t1 - pd.Timedelta('2 days')
    time1_end = t1 + pd.Timedelta('2 days')

    sub = df.loc[time1_start:time1_end]
    sub = sub.loc[sub.datasource != 'NRCS']

    f1_df = pd.concat([f1_df, sub])

    f2_df = df.loc[(df.index == t2) & (df.datasource == 'NRCS')]

    time2_start = t2 - pd.Timedelta('2 days')
    time2_end = t2 + pd.Timedelta('2 days')

    sub = df.loc[time2_start:time2_end]
    sub = sub.loc[sub.datasource != 'NRCS']

    f2_df = pd.concat([f2_df, sub])

    return f1_df, f2_df

