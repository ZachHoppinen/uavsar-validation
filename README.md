# uavsar-validation

This is a research repository comparing UAVSAR L-band InSAR images against snow depth observations over the Sawtooth Mountains of Idaho.

See: https://tc.copernicus.org/articles/18/575/2024/ for full article.

## Workplan

- [x] Download data
    - [x] Download UAVSAR unwrapped phase, wrapped phase, coherence, and incidence angles to netcdfs.
    - [x] Download Lidar datasets for Banner Summit (QSI), Dry Creek (QSI and maybe helicopter), Mores Creek (QSI).
    - [x] Download SnowEx pits, probing, other in-situ observations to pandas dataframe
    - [x] Download snotel observations of flight dates for all snotel sites
    - [x] Download SNOWMODEL data

- [x] Prepare UAVSAR
    - [x] Run phase, inc -> SWE change using in-situ observations
    - [x] Sum seasonal SWE for time series - failure (missing large precip increase due to UAVSAR not flying)
    - [x] Add UAVSAR snow deltas to in-situ dataset

- [x] In-situ compare
    - [x] Compare r2, rmse of all in-situ change vs UAVSAR SD delta
    - [x] Compare r2, rmse for individual types of observations
    - [x] Create time series vs snotel sites of UAVSAR changes vs snotel changes
    - [x] Compare r2, rmse for in-situ changes binned by coherence, snow depth, trees
    - [x] r2 and rmse for other insitu probed depths independent of setting phase - failed interval boards were biased by under board settlement, probed depths were too few

- [x] Model compare
    - [x] Compare r2, rmse of all model observations (100 m) vs UAVSAR SD delta
    - [x] Compare r2, rmse for binned SD classes
    - [x] Compare r2, rmse for binned vegetation classes
    - [x] Compare r2, rmse for binned wetness classes
    - [x] Compare r2, rmse binned by coherence

- [x] Lidar compare
    - [x] Compare r2, rmse of summed SWE vs lidar for each location - failed due to lack of summed SWEs (see prepare UAVSAR section pt 2.) 
    - [x] Compare r2 of UAVSAR sd delta vs snotel SD
    - [x] plot of r2 vs binned slope, aspect, tree %
    - [x] check covariance of independent datasets

- [x] Ancillary figures
    - [x] Figure showing UAVSAR flight box, insitu sites, lidar flight boxes
    - [x] Met summary figure for each study/lidar site's met data

- [x] Writing - rough draft
    - [x] introduction
    - [x] methods
    - [x] results
    - [x] discussion
    - [x] conclusion

- [x] Writing - polished draft
    - [x] introduction
    - [x] methods
    - [x] results
    - [x] discussion
    - [x] conclusion
