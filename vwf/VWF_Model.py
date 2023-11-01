import pandas as pd
import xarray as xr
from pathlib import Path
import time

from vwf.preprocessing import (
    prep_era5,
    prep_obs,
    prep_obs_test,
)
# make a way for the country we want to work on to be read

from vwf.bias_correction import (
    generate_training_data,
    training_bias,
    format_bcfactors,
    closest_cluster
)

from vwf.simulation import simulate_wind

pd.options.mode.chained_assignment = None  # default='warn'

class VWF():
    """
    This class allows both the training and testing of the VWF model.
    """
    def __init__(self, country, year_star, year_end, year_test):
        self.country = country
        self.year_star = year_star
        self.year_end = year_end
        self.year_test = year_test
        
        # fixed inputs files for now
    
    def prep(self):
        
        powerCurveFileLoc = 'data/turbine_info/Wind Turbine Power Curves.csv'
        self.powerCurveFile = pd.read_csv(powerCurveFileLoc)
        
        # files for training
        # self.era5_train = prep_era5(self.year_star, self.year_end, True)
        self.era5_train = prep_era5(True)
        self.obs_cf, self.turb_info_train = prep_obs(self.country, self.year_star, self.year_end)
        # self.obs_cf = pd.read_csv('data/wind_data/'+self.country+'/obs_cf_train.csv')
        # self.turb_info_train = pd.read_csv('data/wind_data/'+self.country+'/turb_info_train.csv')
        self.unc_ws_train, self.unc_cf_train = simulate_wind(self.era5_train, self.turb_info_train, self.powerCurveFile)     
           
        # files for testing
        # this file is currently saved from atlite's cutout
        # ncFile = 'data/reanalysis/test/'+str(self.year_test)+'-'+str(self.year_test)+'_clean.nc'
        # era5_test = xr.open_dataset(ncFile)
        # self.era5_test = era5_test.resample(time='1D').mean() # hourly is too slow
        self.era5_test = prep_era5()
        self.user_input = prep_obs_test(self.country, self.year_test) # probably should be in test but for research its here
        
        return self
        
        
    def train(self, num_clu):
        print(len(self.turb_info_train), " turbines in training.")
        
        my_file = Path('data/bias_correction/bc_factors_'+str(self.year_star)+'-'+str(self.year_end)+'_'+str(num_clu)+'.csv')
        
        if my_file.is_file():
            print(num_clu,  "clusters is trained already.")
            print(" ")
            
            self.clus_info = pd.read_csv(
                'data/bias_correction/clus_labels_'+str(self.year_star)+'-'+str(self.year_end)+'_'+str(num_clu)+'.csv',
                index_col=None
                )
            self.bc_factors = pd.read_csv(
                'data/bias_correction/bc_factors_'+str(self.year_star)+'-'+str(self.year_end)+'_'+str(num_clu)+'.csv',
                index_col=None
                )
        
        else:  
            print("Training on ", num_clu,  " clusters.")
            start_time = time.time()
            
            train_data, self.clus_info = generate_training_data(self.unc_cf_train, self.obs_cf, self.turb_info_train, num_clu)
            bcfactors = training_bias(train_data, self.era5_train, self.clus_info, self.year_star, self.year_end, self.powerCurveFile)
            self.bc_factors = format_bcfactors(bcfactors)
            
            # SAVING FILES FOR FUTURE RE RUNS
            self.clus_info.to_csv(
                'data/bias_correction/clus_labels_'+str(self.year_star)+'-'+str(self.year_end)+'_'+str(num_clu)+'.csv',
                index = None
                )
             
            self.bc_factors.to_csv(
                'data/bias_correction/bc_factors_'+str(self.year_star)+'-'+str(self.year_end)+'_'+str(num_clu)+'.csv',
                index = None
                )
            
            # bcfactors.to_csv(
            #     'data/bias_correction/bc_fact_'+str(self.year_star)+'-'+str(self.year_end)+'_'+str(num_clu)+'_raw.csv',
            #     index = None
            #     )
                
            end_time = time.time()
            elapsed_time = end_time - start_time
            print("Training parameters completed and saved. Elapsed time: {:.2f} seconds".format(elapsed_time))
            print(" ")
            
        self.num_clu = num_clu

        return self

    
    def test(self, time_res):
        
        # user_input = pd.read_csv('data/wind_data/'+self.country+'/turb_info.csv')
        # obs_cf_test, user_input = prep_obs(self.country, self.year_test, self.year_test)
        # user_input = pd.read_csv('data/input.csv') # for single input or to change for future
                                 
        turb_info = closest_cluster(self.clus_info, self.user_input)
        
        # producing uncorrected results
        unc_ws, unc_cf = simulate_wind(self.era5_test, turb_info, self.powerCurveFile)
        
        unc_ws.to_csv('data/results/raw/'+str(self.year_test)+'_unc_ws.csv', index = None)
        unc_cf.to_csv('data/results/raw/'+str(self.year_test)+'_unc_cf.csv', index = None)
            
        # producing corrected results
        print("Test for ", self.year_test, " using ", self.num_clu, " clusters with time resolution: ", time_res,  " is taking place.")
        start_time = time.time()
        
        cor_ws, cor_cf = simulate_wind(self.era5_test, turb_info, self.powerCurveFile, time_res, False, True, self.bc_factors)
        cor_ws.to_csv('data/results/raw/'+str(self.year_test)+'_'+time_res+'_'+str(self.num_clu)+'_cor_ws.csv', index = None)
        cor_cf.to_csv('data/results/raw/'+str(self.year_test)+'_'+time_res+'_'+str(self.num_clu)+'_cor_cf.csv', index = None)
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        print("Results completed and saved. Elapsed time: {:.2f} seconds".format(elapsed_time))
        print(" ")
        
        self.time_res = time_res
        
        return self

        

