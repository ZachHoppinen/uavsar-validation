import numpy as np
import pandas as pd

def get_stats(xs, ys, clean = True, bias = False):

    if clean:
        xs, ys = clean_xs_ys(xs, ys)

    from sklearn.metrics import mean_squared_error
    rmse = mean_squared_error(xs, ys, squared=False)

    from scipy.stats import pearsonr
    r, p = pearsonr(xs, ys)

    if bias:
        MBE = np.mean(ys - xs)
        return rmse, r, len(xs), MBE

    return rmse, r, len(xs)

def clean_xs_ys(xs, ys, clean_zeros = False):
        # stack arrays
    xs = np.hstack(xs)
    ys = np.hstack(ys)

    xs_tmp = xs[(~np.isnan(xs)) & (~np.isnan(ys)) & (np.isfinite(xs)) & (np.isfinite(ys))]
    ys = ys[(~np.isnan(xs)) & (~np.isnan(ys))  & (np.isfinite(xs)) & (np.isfinite(ys))]
    xs = xs_tmp

    if clean_zeros:
        xs_tmp = xs[(xs != 0) & (ys != 0)]
        ys = ys[(xs != 0) & (ys != 0)]
        xs = xs_tmp

    return xs, ys

def two_arr_clean(arr1, arr2):
    idx = np.logical_not(np.logical_or(np.isnan(arr1), np.isnan(arr2)))     
    arr1, arr2 = arr1[idx], arr2[idx]
    return arr1, arr2

def get_r(ds, lidar):
    xs = lidar.values.ravel()
    ys = ds.values.ravel()

    xs, ys = clean_xs_ys(xs, ys, clean_zeros = True)
    r, p = pearsonr(xs, ys)
    return r

def bootstrap(df):
    randlist = pd.DataFrame(index=np.random.randint(len(df), size=len(df)))
    df.merge(randlist, left_index=True, right_index=True, how='right')
