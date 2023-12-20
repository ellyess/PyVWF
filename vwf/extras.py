import numpy as np
import pandas as pd

def add_times(data):
    data['year'] = pd.DatetimeIndex(data['time']).year
    data['month'] = pd.DatetimeIndex(data['time']).month
    data.insert(1, 'year', data.pop('year'))
    data.insert(2, 'month', data.pop('month'))
    return data


def calc_metrics_era5(cluster_list):
    time_res_list = ['yearly', 'season', 'bimonth', 'month'] 
    year_test = 2020
    
    # importing observation for denmark 2020.
    cf_obs = obs = pd.read_csv('data/wind_data/DK/obs_cf_test.csv', parse_dates=['time'])
    cf_obs_month = cf_obs.groupby(pd.Grouper(key='time',freq='M')).mean().transpose().mean()
    
    
    months = np.array(range(1,13))
    
    abs_diff_calc = []
    cluster_all = []
    time_all = []
    month_all = []
    diff_calc = []

    
    cf_uncorr = pd.read_csv('data/results/raw/'+str(year_test)+'_unc_cf.csv', parse_dates=['time'])
    cf_month_uncorr = cf_uncorr.groupby(pd.Grouper(key='time',freq='M')).mean().transpose().mean().values
    diff = cf_month_uncorr - cf_obs_month 
    abs_err = abs(diff)

    df_unc = pd.DataFrame({'month':months, 'abs_err':abs_err, 'diff': diff})
    df_unc['num_clu'] = 1
    df_unc['time_res'] = 'uncorrected'
    
    for num_clu in cluster_list:
        for time_res in time_res_list:
            
            cf_corr = pd.read_csv('data/results/raw/'+str(year_test)+'_'+time_res+'_'+str(num_clu)+'_cor_cf.csv', parse_dates=['time'])
            cf_month = cf_corr.groupby(pd.Grouper(key='time',freq='M')).mean().transpose().mean().values
            diff = cf_month - cf_obs_month 
            abs_diff = abs(diff)
            abs_diff_calc.append(abs_diff)
            diff_calc.append(diff)
            
            cluster_all.append([num_clu]*12)
            time_all.append([time_res]*12)
            month_all.append(months)
                


    df_corr = pd.DataFrame(list(zip(np.ravel(cluster_all), np.ravel(time_all), np.ravel(month_all), np.ravel(abs_diff_calc), np.ravel(diff_calc))), 
                 columns =['num_clu', 'time_res', 'month', 'abs_err', 'diff'])

    columns = ['num_clu', 'time_res', 'month', 'abs_err', 'diff']
    df_corr = df_corr[columns]
    
    df_metrics = pd.concat([df_corr,df_unc]).reset_index(drop=True)
    
    
    # converting it to rmse and mae per trained model
    df_metrics['se'] = df_metrics['abs_err']**2
    all_metrics = df_metrics
    rmse = np.sqrt(df_metrics.groupby(['num_clu', 'time_res'])['se'].mean()).reset_index()
    clus_metrics = df_metrics.groupby(['num_clu', 'time_res'])['abs_err'].mean().reset_index()
    clus_metrics['RMSE'] = rmse['se']
    clus_metrics.columns = ['num_clu', 'time_res', 'MAE', 'RMSE']
    
    clus_metrics.to_csv('data/results/results_metrics.csv', index = None)
    
    return clus_metrics, all_metrics
    
def calc_metrics_merra2():
    # importing observation for denmark 2020.
    cf_obs = obs = pd.read_csv('data/wind_data/DK/obs_cf_test.csv', parse_dates=['time'])
    cf_obs_month = cf_obs.groupby(pd.Grouper(key='time',freq='M')).mean().transpose().mean()
    
    cf_merra2_unc = pd.read_csv('data/results/merra_2020_unc_cf.csv', parse_dates=['time']) # daily cf
    cf_month_unc = cf_merra2_unc.groupby(pd.Grouper(key='time',freq='M')).mean().transpose().mean().values
    abs_err_unc = abs(cf_obs_month - cf_month_unc)
    
    months = np.array(range(1,13))
    
    df_unc = pd.DataFrame({'month':months, 'abs_err':abs_err_unc})
    df_unc['num_clu'] = 1
    df_unc['time_res'] = 'uncorrected'
    
    cf_merra2_corr = pd.read_csv('data/results/merra_2020_year_1_cor_cf.csv', parse_dates=['time']) # daily cf
    cf_month_corr = cf_merra2_corr.groupby(pd.Grouper(key='time',freq='M')).mean().transpose().mean().values
    abs_err_corr = abs(cf_obs_month - cf_month_corr)
                
    df_corr = pd.DataFrame({'month':months, 'abs_err':abs_err_corr})
    df_corr['num_clu'] = 1
    df_corr['time_res'] = 'year'
    
    columns = ['num_clu', 'time_res', 'month', 'abs_err']
    df_corr = df_corr[columns]
    
    df_metrics = pd.concat([df_corr,df_unc]).reset_index(drop=True)  
    # converting it to rmse and mae per trained model
    df_metrics['se'] = df_metrics['abs_err']**2
    all_metrics = df_metrics
    rmse = np.sqrt(df_metrics.groupby(['num_clu', 'time_res'])['se'].mean()).reset_index()
    clus_metrics = df_metrics.groupby(['num_clu', 'time_res'])['abs_err'].mean().reset_index()
    clus_metrics['RMSE'] = rmse['se']
    clus_metrics.columns = ['num_clu', 'time_res', 'MAE', 'RMSE']
    
    return clus_metrics, all_metrics