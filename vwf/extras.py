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