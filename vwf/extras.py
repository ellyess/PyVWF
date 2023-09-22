import numpy as np
import pandas as pd


def hours_to_days(ds):
    return ds.groupby(pd.Grouper(key='time',freq='D')).mean().reset_index()
    
    
def add_times(data):
    data['year'] = pd.DatetimeIndex(data['time']).year
    data['month'] = pd.DatetimeIndex(data['time']).month
    data.insert(1, 'year', data.pop('year'))
    data.insert(2, 'month', data.pop('month'))
    return data
    

def run_all_metrics(cluster_list = [1,2,3,5,10,25,50,100,200]):
    method_list = ['method_1', 'method_2']
    test_year = 2020
    time_res_list = ['year', 'season', 'two_month', 'month'] 
    cluster_list = [1,2,3,5,10,25,50,100,200]
    equation_combo = ['_1_1', '_1_2', '_2_1', '_2_2']
    
    # importing observation for denmark 2020.
    cf_obs = pd.read_csv('data/results/denmark_obs_cf.csv', parse_dates=['time'])
    cf_obs_month = cf_obs.groupby(pd.Grouper(key='time',freq='M')).mean().transpose().mean().values
                 
    for equation in equation_combo:             
        era5 = calc_metrics_era5(cf_obs_month, test_year, cluster_list, time_res_list, equation)
        atlite = calc_metrics_atlite(cf_obs_month, test_year, cluster_list, time_res_list, equation)
                      
        all_metrics = pd.concat([era5,atlite]).reset_index(drop=True)
        all_metrics.to_csv('data/results/metrics/metrics'+equation+'.csv', index = None)
                      

    merra2 = calc_metrics_merra2(cf_obs_month)
    merra2['equation'] = '_1_1'
    merra2.to_csv('data/results/metrics/metrics_merra2.csv', index = None)
                      
        
def calc_metrics_era5(cf_obs_month, test_year, cluster_list, time_res_list, equation):
    
    method_list = ['method_1', 'method_2']
    abs_err_unc = []
    abs_diff_calc = []
    method_all = []
    cluster_all = []
    time_all = []
    month_all = []
    for method in method_list:
    
        cf_uncorr = pd.read_csv('../../../../results/raw'+equation+'/era5_'+method+'_'+str(test_year)+'_cf_uncorr.csv', parse_dates=['time'])
        cf_month_uncorr = cf_uncorr.groupby(pd.Grouper(key='time',freq='M')).mean().transpose().mean().values
        abs_err = abs(cf_obs_month - cf_month_uncorr)
        abs_err_unc.append(abs_err)
        
        for num_clu in cluster_list:
            
            for time_res in time_res_list:
                
                cf_corr = pd.read_csv('../../../../results/raw'+equation+'/era5_'+method+'_'+str(test_year)+'_'+time_res+'_'+str(num_clu)+'_clusters_cf_corr.csv', parse_dates=['time'])
                cf_month = cf_corr.groupby(pd.Grouper(key='time',freq='M')).mean().transpose().mean().values
                abs_diff = abs(cf_obs_month - cf_month)
                abs_diff_calc.append(abs_diff)
                
                method_all.append([method]*12)
                cluster_all.append([num_clu]*12)
                time_all.append([time_res]*12)
                month_all.append(np.array(range(1,13)))
                

    df_unc = pd.DataFrame(list(zip(np.ravel(method_all[::(len(cluster_list)*len(time_res))]), np.ravel(month_all), np.ravel(abs_err_unc))), 
             columns =['method', 'month', 'abs_err'])

    df_unc['mode'] = 'era5'
    df_unc['num_clu'] = 1
    df_unc['time_res'] = 'uncorr'
    df_unc['equation'] = 'uncorr'
    
    df_corr = pd.DataFrame(list(zip(np.ravel(method_all), np.ravel(cluster_all), np.ravel(time_all), np.ravel(month_all), np.ravel(abs_diff_calc))), 
                 columns =['method','num_clu', 'time_res', 'month', 'abs_err'])
    df_corr['mode'] = 'era5'
    columns = ['mode','method','num_clu', 'time_res', 'month', 'abs_err']
    df_corr = df_corr[columns]
    df_corr['equation'] = equation
    
    df_metrics = pd.concat([df_corr,df_unc]).reset_index(drop=True)
    
    return df_metrics

def calc_metrics_atlite(cf_obs_month, test_year, cluster_list, time_res_list, equation):

    abs_err_unc = []
    cf_uncorr = pd.read_csv('../../../../results/atlite'+equation+'/atlite_'+str(test_year)+'_cf_uncorr.csv', parse_dates=['time'])
    cf_month_uncorr = cf_uncorr.groupby(pd.Grouper(key='time',freq='M'))['cf'].mean().values
    abs_err = abs(cf_obs_month - cf_month_uncorr)
    abs_err_unc.append(abs_err)
        
    abs_err_corr = []
    cluster_all = []
    time_all = []
    month_all = []
    for num_clu in cluster_list:

        for time_res in time_res_list:

            cf_corr = pd.read_csv('../../../../results/atlite'+equation+'/atlite_'+str(test_year)+'_'+time_res+'_'+str(num_clu)+'_clusters_cf_corr.csv', parse_dates=['time'])
            cf_month = cf_corr.groupby(pd.Grouper(key='time',freq='M'))['cf'].mean().values
            abs_err = abs(cf_obs_month - cf_month)
            abs_err_corr.append(abs_err)

            cluster_all.append([num_clu]*12)
            time_all.append([time_res]*12)
            month_all.append(np.array(range(1,13)))
                

    df_unc = pd.DataFrame(list(zip(np.ravel(month_all), np.ravel(abs_err_unc))), 
             columns =['month', 'abs_err'])

    df_unc['mode'] = 'era5'
    df_unc['method'] = 'atlite'
    df_unc['num_clu'] = 1
    df_unc['time_res'] = 'uncorr'
    df_unc['equation'] = 'uncorr'
    
    df_corr = pd.DataFrame(list(zip(np.ravel(cluster_all), np.ravel(time_all), np.ravel(month_all), np.ravel(abs_err_corr))), 
                 columns =['num_clu', 'time_res', 'month', 'abs_err'])
    df_corr['mode'] = 'era5'
    df_corr['method'] = 'atlite'
    
    columns = ['mode','method','num_clu', 'time_res', 'month', 'abs_err']
    df_corr = df_corr[columns]
    df_corr['equation'] = equation
    
    df_metrics = pd.concat([df_corr,df_unc]).reset_index(drop=True)
    # df_metrics.to_csv('data/results/metrics/atlite_'+str(test_year)+'_all_metrics.csv', index = None)
    return df_metrics
    
def calc_metrics_merra2(cf_obs_month):
    
    cf_merra2_unc = pd.read_csv('data/results/merra2_method_1_2020_cf_uncorr.csv', parse_dates=['time']) # daily cf
    cf_month_unc = cf_merra2_unc.groupby(pd.Grouper(key='time',freq='M')).mean().transpose().mean().values
    abs_err_unc = abs(cf_obs_month - cf_month_unc)
    
    df_unc = pd.DataFrame()
    df_unc['month'] = np.array(range(1,13))
    df_unc['abs_err'] = abs_err_unc
    df_unc['mode'] = 'merra2'
    df_unc['method'] = 'method_1'
    df_unc['num_clu'] = 1
    df_unc['time_res'] = 'uncorr'
    df_unc['equation'] = 'uncorr'
    
    cf_merra2_corr = pd.read_csv('data/results/merra2_method_1_2020_year_1_clusters_cf_corr.csv', parse_dates=['time']) # daily cf
    cf_month_corr = cf_merra2_corr.groupby(pd.Grouper(key='time',freq='M')).mean().transpose().mean().values
    abs_err_corr = abs(cf_obs_month - cf_month_corr)
                
    df_corr = pd.DataFrame()
    df_corr['month'] = np.array(range(1,13))
    df_corr['abs_err'] = abs_err_corr
    df_corr['mode'] = 'merra2'
    df_corr['method'] = 'method_1'
    df_corr['num_clu'] = 1
    df_corr['time_res'] = 'year'
    columns = ['mode','method','num_clu', 'time_res', 'month', 'abs_err']
    df_corr = df_corr[columns]
    df_unc['equation'] = '_1_1'
    
    df_metrics = pd.concat([df_corr,df_unc]).reset_index(drop=True)     
    return df_metrics

    

