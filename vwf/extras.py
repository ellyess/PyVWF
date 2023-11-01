import numpy as np
import pandas as pd

def add_times(data):
    data['year'] = pd.DatetimeIndex(data['time']).year
    data['month'] = pd.DatetimeIndex(data['time']).month
    data.insert(1, 'year', data.pop('year'))
    data.insert(2, 'month', data.pop('month'))
    return data


def calc_metrics_era5(cluster_list):
    time_res_list = ['year', 'season', 'two_month', 'month'] 
    year_test = 2020
    
    # importing observation for denmark 2020.
    cf_obs = pd.read_csv('data/results/denmark_obs_cf.csv', parse_dates=['time'])
    cf_obs_month = cf_obs.groupby(pd.Grouper(key='time',freq='M')).mean().transpose().mean()
    
    
    months = np.array(range(1,13))
    
    abs_diff_calc = []
    cluster_all = []
    time_all = []
    month_all = []

    
    cf_uncorr = pd.read_csv('data/results/raw/'+str(year_test)+'_unc_cf.csv', parse_dates=['time'])
    cf_month_uncorr = cf_uncorr.groupby(pd.Grouper(key='time',freq='M')).mean().transpose().mean().values
    abs_err = abs(cf_obs_month - cf_month_uncorr)

    df_unc = pd.DataFrame({'month':months, 'abs_err':abs_err})
    df_unc['num_clu'] = 1
    df_unc['time_res'] = 'uncorrected'
    
    for num_clu in cluster_list:
        for time_res in time_res_list:
            
            cf_corr = pd.read_csv('data/results/raw/'+str(year_test)+'_'+time_res+'_'+str(num_clu)+'_cor_cf.csv', parse_dates=['time'])
            cf_month = cf_corr.groupby(pd.Grouper(key='time',freq='M')).mean().transpose().mean().values
            abs_diff = abs(cf_obs_month - cf_month)
            abs_diff_calc.append(abs_diff)
            
            cluster_all.append([num_clu]*12)
            time_all.append([time_res]*12)
            month_all.append(months)
                


    df_corr = pd.DataFrame(list(zip(np.ravel(cluster_all), np.ravel(time_all), np.ravel(month_all), np.ravel(abs_diff_calc))), 
                 columns =['num_clu', 'time_res', 'month', 'abs_err'])

    columns = ['num_clu', 'time_res', 'month', 'abs_err']
    df_corr = df_corr[columns]
    
    df_metrics = pd.concat([df_corr,df_unc]).reset_index(drop=True)
    
    # converting it to rmse and mae per trained model
    df_metrics['se'] = df_metrics['abs_err']**2
    rmse = np.sqrt(df_metrics.groupby(['num_clu', 'time_res'])['se'].mean()).reset_index()
    df_metrics = df_metrics.groupby(['num_clu', 'time_res'])['abs_err'].mean().reset_index()
    df_metrics['RMSE'] = rmse['se']
    df_metrics.columns = ['num_clu', 'time_res', 'MAE', 'RMSE']
    df_metrics
    
    df_metrics.to_csv('data/results/results_metrics.csv', index = None)
    
    return df_metrics