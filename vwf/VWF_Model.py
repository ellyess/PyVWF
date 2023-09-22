import pandas as pd
import xarray as xr
from pathlib import Path
import time

from vwf.preprocessing import (
    prep_era5_method_1,
    prep_era5_method_2,
    prep_merra2_method_1,
    prep_metadata_2020,
    prep_obs_and_turb_info
)

from vwf.bias_correction import (
    generate_training_data,
    training_bias,
    format_bcfactors,
    closest_cluster
)

from vwf.simulation import simulate_wind
from vwf.extras import hours_to_days


pd.options.mode.chained_assignment = None  # default='warn'

class VWF():
    """
    This class allows both the training and testing of the VWF model.
    """
    def __init__(self, mode, method):

        # variables for training and testing
        self.mode = mode
        self.method = method
        
        # fixed inputs files for now
    
    def load_files(self, year_star, year_end, year_):
        
        powerCurveFileLoc = 'data/turbine_info/Wind Turbine Power Curves.csv'
        self.powerCurveFile = pd.read_csv(powerCurveFileLoc)
        self.meta_2020 = prep_metadata_2020()
        
        if self.mode == 'merra2':
            self.reanal_data_test = prep_merra2_method_1(year_, year_)  
            
        else:
            # for train and test
            if self.method == 'method_1':
                self.reanal_data = prep_era5_method_1(year_star, year_end)
                self.reanal_data_test = prep_era5_method_1(year_, year_)  
            else:
                self.reanal_data = prep_era5_method_2(year_star, year_end)
                # this file is currently saved from atlite's cutout
                ncFile = 'data/reanalysis/era5/'+str(year_)+'-'+str(year_)+'_clean.nc'
                reanal_data_test = xr.open_dataset(ncFile)
                self.reanal_data_test = reanal_data_test.resample(time='1D').mean() # hourly is too slow
        
            self.obs_gen, self.turb_info_train = prep_obs_and_turb_info(self.meta_2020, year_star, year_end)
            self.uncorr_speed_train, self.uncorr_cf_train = simulate_wind(self.reanal_data, self.turb_info_train, self.powerCurveFile)     
           
        self.year_star = year_star
        self.year_end = year_end
        self.year_ = year_
        
        return self
        
        
    def train(self, num_clu):

        my_file = Path('data/bias_correction/'+self.mode+'_'+self.method+'_bcfactors_'+str(self.year_star)+'-'+str(self.year_end)+'_'+str(num_clu)+'_clusters.csv')
        if my_file.is_file():
            print(self.method, " using ", num_clu,  "clusters is trained already.")
            print(" ")
            
            self.clus_data = pd.read_csv(
                'data/bias_correction/'+self.mode+'_'+self.method+'_clus_labels_'+str(self.year_star)+'-'+str(self.year_end)+'_'+str(num_clu)+'_clusters.csv',
                index_col=None
                )
            self.bias_factors = pd.read_csv(
                'data/bias_correction/'+self.mode+'_'+self.method+'_bcfactors_'+str(self.year_star)+'-'+str(self.year_end)+'_'+str(num_clu)+'_clusters.csv',
                index_col=None
                )
        
        else:
            
            print("Training for ", self.method, " using ", num_clu,  " is taking place.")
            start_time = time.time()
            
            train_gen, self.clus_data = generate_training_data(self.uncorr_cf_train, self.obs_gen, self.turb_info_train, num_clu)
            train_factors = training_bias(train_gen, self.reanal_data, self.clus_data, self.year_star, self.year_end, self.powerCurveFile)
            self.bias_factors = format_bcfactors(train_factors)
            
            self.clus_data.to_csv(
                'data/bias_correction/'+self.mode+'_'+self.method+'_clus_labels_'+str(self.year_star)+'-'+str(self.year_end)+'_'+str(num_clu)+'_clusters.csv',
                index = None
                )
             
            self.bias_factors.to_csv(
                'data/bias_correction/'+self.mode+'_'+self.method+'_bcfactors_'+str(self.year_star)+'-'+str(self.year_end)+'_'+str(num_clu)+'_clusters.csv',
                index = None
                )
            
            train_factors.to_csv(
                'data/bias_correction/'+self.mode+'_'+self.method+'_bcfactors_'+str(self.year_star)+'-'+str(self.year_end)+'_'+str(num_clu)+'_clusters_raw.csv',
                index = None
                )
                
            end_time = time.time()
            elapsed_time = end_time - start_time
            print("Training parameters completed and saved. Elapsed time: {:.2f} seconds".format(elapsed_time))
            print(" ")
            
            return self
            
        self.num_clu = num_clu

        return self

    
    def test(self, time_res):
        
        if self.mode == 'merra2':
            self.num_clu = 1
            turb_info = self.meta_2020.copy()
            turb_info['cluster'] = 0

            bias_fact_dict = {'0': [0 , 0.597, 2.836]}
            self.bias_factors = pd.DataFrame.from_dict(bias_fact_dict, orient='index',
                                        columns=['cluster', 'scalar', 'offset'])
                                        
        else:                            
            turb_info = closest_cluster(self.clus_data, self.meta_2020)
        
        # producing uncorrected results
        uncorr_file = Path('data/results/raw/'+self.mode+'_'+self.method+'_'+str(self.year_)+'_speed_uncorr.csv')
        if uncorr_file.is_file():
            uncorr_speed = pd.read_csv('data/results/raw/'+self.mode+'_'+self.method+'_'+str(self.year_)+'_speed_uncorr.csv', parse_dates=['time'])
            uncorr_cf = pd.read_csv('data/results/raw/'+self.mode+'_'+self.method+'_'+str(self.year_)+'_cf_uncorr.csv', parse_dates=['time'])
        
        else:
            uncorr_speed, uncorr_cf = simulate_wind(self.reanal_data_test, turb_info, self.powerCurveFile)
            
            if self.mode == 'merra2':
                uncorr_speed = hours_to_days(uncorr_speed)
                uncorr_cf = hours_to_days(uncorr_cf)
            
            uncorr_speed.to_csv('data/results/raw/'+self.mode+'_'+self.method+'_'+str(self.year_)+'_speed_uncorr.csv', index = None)
            uncorr_cf.to_csv('data/results/raw/'+self.mode+'_'+self.method+'_'+str(self.year_)+'_cf_uncorr.csv', index = None)
            
        # producing corrected results
        corr_file = Path('data/results/raw/'+self.mode+'_'+self.method+'_'+str(self.year_)+'_'+time_res+'_'+str(self.num_clu)+'_clusters_speed_corr.csv')
        if corr_file.is_file():
            print("Results for ", self.year_, " using ", self.num_clu, " clusters with time resolution: ", time_res, " exist already.")
            print(" ")
                       
        else:
            print("Test for ", self.year_, " using ", self.num_clu, " clusters with time resolution: ", time_res,  " is taking place.")
            start_time = time.time()
            
            corr_speed, corr_cf = simulate_wind(self.reanal_data_test, turb_info, self.powerCurveFile, time_res, False, True, self.bias_factors)
            
            if self.mode == 'merra2':
                corr_speed = hours_to_days(corr_speed)
                corr_cf = hours_to_days(corr_cf)
                
            corr_speed.to_csv('data/results/raw/'+self.mode+'_'+self.method+'_'+str(self.year_)+'_'+time_res+'_'+str(self.num_clu)+'_clusters_speed_corr.csv', index = None)
            corr_cf.to_csv('data/results/raw/'+self.mode+'_'+self.method+'_'+str(self.year_)+'_'+time_res+'_'+str(self.num_clu)+'_clusters_cf_corr.csv', index = None)
            
            end_time = time.time()
            elapsed_time = end_time - start_time
            print("Results completed and saved. Elapsed time: {:.2f} seconds".format(elapsed_time))
            print(" ")
        

        self.time_res = time_res
        
        return self

        

