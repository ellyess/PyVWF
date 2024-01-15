import numpy as np
import pandas as pd

def add_times(data):
    data['year'] = pd.DatetimeIndex(data['time']).year
    data['month'] = pd.DatetimeIndex(data['time']).month
    data.insert(1, 'year', data.pop('year'))
    data.insert(2, 'month', data.pop('month'))
    return data
    
def add_time_res(df):
    df.loc[df['month'] == 1, ['bimonth','season']] = ['1/6', 'winter']
    df.loc[df['month'] == 2, ['bimonth','season']] = ['1/6', 'winter']
    df.loc[df['month'] == 3, ['bimonth','season']] = ['2/6', 'spring']
    df.loc[df['month'] == 4, ['bimonth','season']] = ['2/6', 'spring']
    df.loc[df['month'] == 5, ['bimonth','season']] = ['3/6', 'spring']
    df.loc[df['month'] == 6, ['bimonth','season']] = ['3/6', 'summer']
    df.loc[df['month'] == 7, ['bimonth','season']] = ['4/6', 'summer']
    df.loc[df['month'] == 8, ['bimonth','season']] = ['4/6', 'summer']
    df.loc[df['month'] == 9, ['bimonth','season']] = ['5/6', 'autumn']
    df.loc[df['month'] == 10, ['bimonth','season']] = ['5/6', 'autumn']
    df.loc[df['month'] == 11, ['bimonth','season']] = ['6/6', 'autumn']
    df.loc[df['month'] == 12, ['bimonth','season']] = ['6/6', 'winter']
    df['yearly'] = 'year'
    return df

def weighted_monthly_cf(df, turb_info):
    df = df.groupby(pd.Grouper(key='time',freq='M')).mean()
    df = df.reset_index()
    df = df.melt(
            id_vars=["time"], # adding in turbine ID for merging
            var_name="ID", 
            value_name="cf"
        )
    
    df["ID"] = df["ID"].astype(str)
    turb_info["ID"] = turb_info["ID"].astype(str)
    df = pd.merge(df, turb_info[['ID','capacity']], on=['ID'], how='left')

    def weighted_avg(df, values, weights):
            return (df[values] * df[weights]).sum() / df[weights].sum()

    wavg = lambda x: weighted_avg(df.loc[x.index], 'cf', 'capacity')
    
    df = df.groupby(pd.Grouper(key='time',freq='M')).agg({"cf": wavg})
    return df.cf
    
def calc_metrics_era5(year_test, country, cluster_list, time_res_list=['yearly', 'season', 'bimonth', 'month']):
    turb_info = pd.read_csv('data/training/simulated-turbines/'+country+"_"+str(year_test)+'_turb_info.csv')
    cf_obs = pd.read_csv('data/results/capacity-factor/'+country+"_"+str(year_test)+'_obs_cf.csv', parse_dates=['time'])
    cf_unc = pd.read_csv('data/results/capacity-factor/'+country+"_"+str(year_test)+'_unc_cf.csv', parse_dates=['time'])

    cf_obs_month = weighted_monthly_cf(cf_obs, turb_info)
    cf_unc_month = weighted_monthly_cf(cf_unc, turb_info)

    months = np.array(range(1,13))
    diff = cf_unc_month - cf_obs_month
    abs_err = abs(diff)
    
    df_unc = pd.DataFrame({'month':months, 'abs_err':abs_err, 'diff': diff})
    df_unc['num_clu'] = 1
    df_unc['time_res'] = 'uncorrected'

    # loading and calculating every scenario we simulated for comparison
    abs_diff_calc = []
    cluster_all = []
    time_all = []
    month_all = []
    diff_calc = []
    for num_clu in cluster_list:
        for time_res in time_res_list:
            cf_cor = pd.read_csv('data/results/capacity-factor/'+country+"_"+str(year_test)+'_'+time_res+'_'+str(num_clu)+'_cor_cf.csv', parse_dates=['time'])
            cf_cor_month = weighted_monthly_cf(cf_cor, turb_info)
            
            diff = cf_cor_month - cf_obs_month 
            abs_diff = abs(diff)
            
            abs_diff_calc.append(abs_diff)
            diff_calc.append(diff)
            cluster_all.append([num_clu]*12)
            time_all.append([time_res]*12)
            month_all.append(months)
                

    df_cor = pd.DataFrame(list(zip(np.ravel(cluster_all), np.ravel(time_all), np.ravel(month_all), np.ravel(abs_diff_calc), np.ravel(diff_calc))), 
                 columns =['num_clu', 'time_res', 'month', 'abs_err', 'diff'])
    
    df_metrics = pd.concat([df_cor,df_unc]).reset_index(drop=True)
    
    # converting it to rmse and mae
    df_metrics['se'] = df_metrics['abs_err']**2
    rmse = np.sqrt(df_metrics.groupby(['num_clu', 'time_res'])['se'].mean()).reset_index()
    clus_metrics = df_metrics.groupby(['num_clu', 'time_res'])['abs_err'].mean().reset_index()
    clus_metrics['RMSE'] = rmse['se']
    clus_metrics.columns = ['num_clu', 'time_res', 'MAE', 'RMSE']

    df_metrics['se'] = np.sqrt(df_metrics['se'])
    df_metrics.columns = ['num_clu','time_res','month', 'AE', 'diff', 'RMSE']
    
    return clus_metrics, df_metrics
    
# def calc_metrics_era5(cluster_list, year_test, country):
#     time_res_list = ['yearly', 'season', 'bimonth', 'month'] 
    
#     # importing observation for denmark 2020.
#     cf_obs = obs = pd.read_csv('data/results/capacity-factor/'+country+"_"+str(year_test)+'_obs_cf.csv', parse_dates=['time'])
#     cf_obs_month = cf_obs.groupby(pd.Grouper(key='time',freq='M')).mean().transpose().mean()
    
#     months = np.array(range(1,13))
    
#     abs_diff_calc = []
#     cluster_all = []
#     time_all = []
#     month_all = []
#     diff_calc = []

    
#     cf_uncorr = pd.read_csv('data/results/capacity-factor/'+country+"_"+str(year_test)+'_unc_cf.csv', parse_dates=['time'])
#     cf_month_uncorr = cf_uncorr.groupby(pd.Grouper(key='time',freq='M')).mean().transpose().mean().values
#     diff = cf_month_uncorr - cf_obs_month 
#     abs_err = abs(diff)

#     df_unc = pd.DataFrame({'month':months, 'abs_err':abs_err, 'diff': diff})
#     df_unc['num_clu'] = 1
#     df_unc['time_res'] = 'uncorrected'
    
#     for num_clu in cluster_list:
#         for time_res in time_res_list:
            
#             cf_corr = pd.read_csv('data/results/capacity-factor/'+country+"_"+str(year_test)+'_'+time_res+'_'+str(num_clu)+'_cor_cf.csv', parse_dates=['time'])
#             cf_month = cf_corr.groupby(pd.Grouper(key='time',freq='M')).mean().transpose().mean().values
#             diff = cf_month - cf_obs_month 
#             abs_diff = abs(diff)
#             abs_diff_calc.append(abs_diff)
#             diff_calc.append(diff)
            
#             cluster_all.append([num_clu]*12)
#             time_all.append([time_res]*12)
#             month_all.append(months)
                


#     df_corr = pd.DataFrame(list(zip(np.ravel(cluster_all), np.ravel(time_all), np.ravel(month_all), np.ravel(abs_diff_calc), np.ravel(diff_calc))), 
#                  columns =['num_clu', 'time_res', 'month', 'abs_err', 'diff'])

#     columns = ['num_clu', 'time_res', 'month', 'abs_err', 'diff']
#     df_corr = df_corr[columns]
    
#     df_metrics = pd.concat([df_corr,df_unc]).reset_index(drop=True)
    
    
#     # converting it to rmse and mae per trained model
#     df_metrics['se'] = df_metrics['abs_err']**2
#     all_metrics = df_metrics
#     rmse = np.sqrt(df_metrics.groupby(['num_clu', 'time_res'])['se'].mean()).reset_index()
#     clus_metrics = df_metrics.groupby(['num_clu', 'time_res'])['abs_err'].mean().reset_index()
#     clus_metrics['RMSE'] = rmse['se']
#     clus_metrics.columns = ['num_clu', 'time_res', 'MAE', 'RMSE']
    
#     clus_metrics.to_csv('data/results/metrics/'+country+"_"+str(year_test)+'_metrics.csv', index = None)
    
#     return clus_metrics, all_metrics
    