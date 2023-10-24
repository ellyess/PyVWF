import pandas as pd
import xarray as xr
from pathlib import Path
import time

from vwf.preprocessing import (
    prep_era5,
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
    def __init__(self, country):

        # variables for training and testing
        self.mode = 'era5'
        self.method = 'method_2'
        self.country = country
        
        # fixed inputs files for now
    
    def load_files(self, year_star, year_end, year_test):
        
        powerCurveFileLoc = 'data/turbine_info/Wind Turbine Power Curves.csv'
        self.powerCurveFile = pd.read_csv(powerCurveFileLoc)
        
        # files for training
        self.era5_train = prep_era5(year_star, year_end)
        self.obs_cf = pd.read_csv('data/wind_data/'+self.country+'/obs_cf.csv')
        self.turb_info_train = pd.read_csv('data/wind_data/'+self.country+'/turb_info.csv')
        self.unc_ws_train, self.unc_cf_train = simulate_wind(self.era5_train, self.turb_info_train, self.powerCurveFile)     
           
        # files for testing
        # this file is currently saved from atlite's cutout
        ncFile = 'data/reanalysis/era5/'+str(year_test)+'-'+str(year_test)+'_clean.nc'
        era5_test = xr.open_dataset(ncFile)
        self.era5_test = era5_test.resample(time='1D').mean() # hourly is too slow
           
        self.year_star = year_star
        self.year_end = year_end
        self.year_test = year_test
        
        return self
        
        
    def train(self, num_clu):

        my_file = Path('data/bias_correction/'+self.mode+'_'+self.method+'_bcfactors_'+str(self.year_star)+'-'+str(self.year_end)+'_'+str(num_clu)+'_clusters.csv')
        
        if my_file.is_file():
            print(self.method, " using ", num_clu,  "clusters is trained already.")
            print(" ")
            
            self.clus_info = pd.read_csv(
                'data/bias_correction/'+self.mode+'_'+self.method+'_clus_labels_'+str(self.year_star)+'-'+str(self.year_end)+'_'+str(num_clu)+'_clusters.csv',
                index_col=None
                )
            self.bc_factors = pd.read_csv(
                'data/bias_correction/'+self.mode+'_'+self.method+'_bcfactors_'+str(self.year_star)+'-'+str(self.year_end)+'_'+str(num_clu)+'_clusters.csv',
                index_col=None
                )
        
        else:  
            print("Training for ", self.method, " using ", num_clu,  " is taking place.")
            start_time = time.time()
            
            train_data, self.clus_info = generate_training_data(self.unc_cf_train, self.obs_cf, self.turb_info_train, num_clu)
            bcfactors = training_bias(train_data, self.era5_train, self.clus_info, self.year_star, self.year_end, self.powerCurveFile)
            self.bc_factors = format_bcfactors(bcfactors)
            
            self.clus_info.to_csv(
                'data/bias_correction/'+self.mode+'_'+self.method+'_clus_labels_'+str(self.year_star)+'-'+str(self.year_end)+'_'+str(num_clu)+'_clusters.csv',
                index = None
                )
             
            self.bc_factors.to_csv(
                'data/bias_correction/'+self.mode+'_'+self.method+'_bcfactors_'+str(self.year_star)+'-'+str(self.year_end)+'_'+str(num_clu)+'_clusters.csv',
                index = None
                )
            
            bcfactors.to_csv(
                'data/bias_correction/'+self.mode+'_'+self.method+'_bcfactors_'+str(self.year_star)+'-'+str(self.year_end)+'_'+str(num_clu)+'_clusters_raw.csv',
                index = None
                )
                
            end_time = time.time()
            elapsed_time = end_time - start_time
            print("Training parameters completed and saved. Elapsed time: {:.2f} seconds".format(elapsed_time))
            print(" ")
            
        self.num_clu = num_clu

        return self

    
    def test(self, time_res):
        
        turb_info_test = pd.read_csv('data/wind_data/'+self.country+'/turb_info_test.csv')                            
        turb_info_test = closest_cluster(self.clus_info, turb_info_test)
        
        # producing uncorrected results
        uncorr_file = Path('data/results/raw/'+self.mode+'_'+self.method+'_'+str(self.year_test)+'_speed_uncorr.csv')
        if uncorr_file.is_file():
            unc_ws_test = pd.read_csv('data/results/raw/'+self.mode+'_'+self.method+'_'+str(self.year_test)+'_speed_uncorr.csv', parse_dates=['time'])
            unc_cf_test = pd.read_csv('data/results/raw/'+self.mode+'_'+self.method+'_'+str(self.year_test)+'_cf_uncorr.csv', parse_dates=['time'])
        
        else:
            unc_ws_test, unc_cf_test = simulate_wind(self.era5_test, turb_info_test, self.powerCurveFile)
            
            unc_ws_test.to_csv('data/results/raw/'+self.mode+'_'+self.method+'_'+str(self.year_test)+'_speed_uncorr.csv', index = None)
            unc_cf_test.to_csv('data/results/raw/'+self.mode+'_'+self.method+'_'+str(self.year_test)+'_cf_uncorr.csv', index = None)
            
        # producing corrected results
        corr_file = Path('data/results/raw/'+self.mode+'_'+self.method+'_'+str(self.year_test)+'_'+time_res+'_'+str(self.num_clu)+'_clusters_speed_corr.csv')
        if corr_file.is_file():
            print("Results for ", self.year_test, " using ", self.num_clu, " clusters with time resolution: ", time_res, " exist already.")
            print(" ")
                       
        else:
            print("Test for ", self.year_test, " using ", self.num_clu, " clusters with time resolution: ", time_res,  " is taking place.")
            start_time = time.time()
            
            corr_speed, corr_cf = simulate_wind(self.era5_test, turb_info_test, self.powerCurveFile, time_res, False, True, self.bc_factors)
            corr_speed.to_csv('data/results/raw/'+self.mode+'_'+self.method+'_'+str(self.year_test)+'_'+time_res+'_'+str(self.num_clu)+'_clusters_speed_corr.csv', index = None)
            corr_cf.to_csv('data/results/raw/'+self.mode+'_'+self.method+'_'+str(self.year_test)+'_'+time_res+'_'+str(self.num_clu)+'_clusters_cf_corr.csv', index = None)
            
            end_time = time.time()
            elapsed_time = end_time - start_time
            print("Results completed and saved. Elapsed time: {:.2f} seconds".format(elapsed_time))
            print(" ")
        

        self.time_res = time_res
        
        return self

        

