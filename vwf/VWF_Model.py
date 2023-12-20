import pandas as pd
import xarray as xr
import dask.dataframe as dd

from pathlib import Path
import time

from vwf.preprocessing import (
    prep_era5,
    prep_obs,
    merge_gen_cf,
)
from vwf.bias_correction import (
    cluster_turbines,
    train_data,
    find_offset,
    closest_cluster
)

from vwf.simulation import simulate_wind

pd.options.mode.chained_assignment = None  # default='warn'

class VWF():
    """
    This class allows both the training and testing of the VWF model.
    """
    def __init__(self, country):
        self.country = country
        
        # fixed inputs files for now
    
    def prep(self):
        
        powerCurveFileLoc = 'data/turbine_info/Wind Turbine Power Curves.csv'
        self.powerCurveFile = pd.read_csv(powerCurveFileLoc)
            
        return self
        
        
    def train(self, cluster_list, time_res_list):
        
        era5 = prep_era5(True)
        obs_cf, turb_info = prep_obs(self.country, True)
        turb_info.to_csv('data/correction_factors/simulated_turbines/'+self.country+'_train_turb_info.csv', index = None)
        gen_cf = merge_gen_cf(era5, obs_cf, turb_info, self.powerCurveFile)
        
        for num_clu in cluster_list:
            clus_info = cluster_turbines(num_clu, turb_info)
            clus_info.to_csv('data/correction_factors/simulated_turbines/'+self.country+'_clus_info_'+str(num_clu)+'.csv')
        
            for time_res in time_res_list:
                
                my_file = Path('data/correction_factors/'+self.country+'_factors_'+time_res+'_'+str(num_clu)+'.csv')
                if my_file.is_file():
                    print(time_res, " on ", num_clu,  " clusters is trained already.")
                    print(" ")
                
                else:
            
                    print("Training for ", num_clu, " clusters with time resolution: ", time_res, " is taking place.")
                    start_time = time.time()
                    
                    bias_data = train_data(time_res, gen_cf, clus_info)
                    ddf = dd.from_pandas(bias_data, npartitions=40)
                    
                    def find_offset_parallel(df):
                        return df.apply(find_offset, args=(clus_info, era5, self.powerCurveFile), axis=1)
                        
                    ddf["offset"] = ddf.map_partitions(find_offset_parallel, meta=('offset', 'float'))
                    ddf.to_csv(
                        'data/correction_factors/'+self.country+'_factors_'+time_res+'_'+str(num_clu)+'.csv', 
                        single_file=True, 
                        compute_kwargs={'scheduler':'processes'}
                    )
                    
                    end_time = time.time()
                    elapsed_time = end_time - start_time
                    print("Trained correction factors have been saved. Elapsed time: {:.2f} seconds".format(elapsed_time))
                    print(" ")
        
        return self

    
    def test(self, year_test, cluster_list, time_res_list):
        era5 = prep_era5()
        obs_cf, turb_info = prep_obs(self.country, False, year_test)
        obs_cf.to_csv('data/results/raw/'+self.country+"_"+str(year_test)+'_obs_ws.csv', index = None)
        turb_info.to_csv('data/correction_factors/simulated_turbines/'+self.country+'_'+str(year_test)+'_turb_info.csv', index = None)
        
        unc_ws, unc_cf = simulate_wind(era5, turb_info, self.powerCurveFile)
        unc_ws.to_csv('data/results/raw/'+self.country+"_"+str(year_test)+'_unc_ws.csv', index = None)
        unc_cf.to_csv('data/results/raw/'+self.country+"_"+str(year_test)+'_unc_cf.csv', index = None)
            
        for num_clu in cluster_list:
            clus_info = pd.read_csv('data/correction_factors/simulated_turbines/'+self.country+'_clus_info_'+str(num_clu)+'.csv')
            clus_info = closest_cluster(clus_info, turb_info)
        
            for time_res in time_res_list:
                print("Test for ", num_clu, " clusters with time resolution: ", time_res, " is taking place.")
                start_time = time.time()
                
                bias_data = pd.read_csv('data/correction_factors/'+self.country+'_factors_'+time_res+'_'+str(num_clu)+'.csv')
                bc_factors = bias_data.groupby(['cluster', 'time_slice'], as_index=False).agg({'scalar': 'mean', 'offset': 'mean'})
                bc_factors.columns = ['cluster',time_res,'scalar','offset']
        
                cor_ws, cor_cf = simulate_wind(era5, clus_info, self.powerCurveFile, bc_factors, time_res)
                cor_ws.to_csv('data/results/raw/'+self.country+"_"+str(year_test)+'_'+time_res+'_'+str(num_clu)+'_cor_ws.csv', index = None)
                cor_cf.to_csv('data/results/raw/'+self.country+"_"+str(year_test)+'_'+time_res+'_'+str(num_clu)+'_cor_cf.csv', index = None)
        
                end_time = time.time()
                elapsed_time = end_time - start_time
                print("Results completed and saved. Elapsed time: {:.2f} seconds".format(elapsed_time))
                print(" ")
        
        return self
    