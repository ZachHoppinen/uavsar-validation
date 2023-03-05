import numpy as np

def get_stats(xs, ys):
    xs_tmp = xs[(~np.isnan(xs)) & (~np.isnan(ys)) & (np.isfinite(xs)) & (np.isfinite(ys))]
    ys = ys[(~np.isnan(xs)) & (~np.isnan(ys))  & (np.isfinite(xs)) & (np.isfinite(ys))]
    xs = xs_tmp

    from sklearn.metrics import mean_squared_error
    rmse = mean_squared_error(xs, ys, squared=False)

    from scipy.stats import pearsonr
    r, p = pearsonr(xs, ys)

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