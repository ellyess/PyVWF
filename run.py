import vwf.VWF_Model as model
import time
import xarray as xr
import numpy as np
import pandas as pd

mode = 'era5' # 'era5' or 'merra2'
method = 'method_2' # 'method_1' or 'method_2'
time_res = 'two_month' #  the options are: 'year', 'season', 'two_month', 'month'
num_clu = 10 # the higher the number of clusters the longer training will take

year_star = 2016 # start year of training period
year_end = 2019 # end year of training period
test_year = 2020 # year you wish to receive a time series for

vwf_model = model.VWF(mode, method)
vwf_model.load_files(year_star, year_end, test_year)

vwf_model.train(num_clu)

vwf_model.test(time_res)