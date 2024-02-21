import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

bg_colour = '#f0f0f0'
custom_params = {'xtick.bottom': True, 'axes.edgecolor': 'black', 'axes.spines.right': False, 'axes.spines.top': False, 'axes.facecolor': bg_colour, 'mathtext.default': 'regular'}
sns.set_theme(style='darkgrid', rc=custom_params)

def calc_error(type, df_sim, df_obs, turb_info, train=False):
    if train == True:
        df_obs = df_obs.pivot(
                index=['year','month'], 
                columns='ID', 
                values='obs').reset_index(drop=True)
        df_obs['time'] = np.arange(str(2015)+'-01', str(2019+1)+'-01', dtype='datetime64[M]')
        df_obs['month'] = df_obs.time.dt.month
        df_obs_monthly = df_obs.drop(columns=['time']).groupby('month').mean().reset_index()
        df_obs_monthly = df_obs_monthly.melt(id_vars=['month'], var_name='ID', value_name='cf')
    else:
        df_obs['time'] = pd.to_datetime(df_obs['time'])
        df_obs['month'] = df_obs.time.dt.month
        df_obs_monthly = df_obs.drop(columns=['time']).set_index('month').reset_index()
        df_obs_monthly = df_obs_monthly.melt(id_vars=['month'], var_name='ID', value_name='cf')
    
    df_sim['time'] = pd.to_datetime(df_sim['time'])
    df_sim['month'] = df_sim.time.dt.month
    df_sim_monthly = df_sim.drop(columns=['time']).groupby('month').mean().reset_index()
    df_sim_monthly = df_sim_monthly.melt(id_vars=['month'], var_name='ID', value_name='cf')

    turb_info['ID'] = turb_info['ID'].apply(str)
    df_obs_monthly['ID'] = df_obs_monthly['ID'].apply(str)
    df_sim_monthly['ID'] = df_sim_monthly['ID'].apply(str)
    
    merged = pd.merge(df_sim_monthly, df_obs_monthly, on=['ID', 'month'], suffixes=('_sim', '_obs'))
    if type == 'region':
        merged = pd.merge(merged, turb_info[['ID', 'capacity','region']], on='ID')
    else:
        merged = pd.merge(merged, turb_info[['ID', 'capacity']], on='ID')
    
    merged['diff'] = merged['cf_sim'] - merged['cf_obs']
    merged['abdiff'] = np.abs(merged['diff'])
    merged['sqdiff'] = merged['diff']**2
    
    merged = merged.dropna(subset=['cf_sim', 'cf_obs', 'capacity'])

    def weighted_avg(df, values, weights):
        return (df[values] * df[weights]).sum() / df[weights].sum()

    wavg = lambda x: weighted_avg(merged.loc[x.index], 'sqdiff', 'capacity')
    wavg2 = lambda x: weighted_avg(merged.loc[x.index], 'abdiff', 'capacity')
    wavg3 = lambda x: weighted_avg(merged.loc[x.index], 'cf_obs', 'capacity')

    # normalises the error i think???
    if type == 'total':
        error = merged.agg({"sqdiff": wavg, 'abdiff': wavg2, 'cf_obs': wavg3})
        rmse = (np.sqrt(error['sqdiff'])/error['cf_obs'])
        mae = (error['abdiff']/error['cf_obs'])
    else:
        error = merged.groupby(type).agg({"sqdiff": wavg, 'abdiff': wavg2, 'cf_obs': wavg3})
        rmse = (np.sqrt(error['sqdiff'])/error['cf_obs']).rename('rmse')
        mae = (error['abdiff']/error['cf_obs']).rename('mae')

    # no normalisation
    # if type == 'total':
    #     error = merged.agg({"sqdiff": wavg, 'abdiff': wavg2, 'cf_obs': wavg3})
    #     rmse = np.sqrt(error['sqdiff'])
    #     mae = error['abdiff']
    # else:
    #     error = merged.groupby(type).agg({"sqdiff": wavg, 'abdiff': wavg2, 'cf_obs': wavg3})
    #     rmse = np.sqrt(error['sqdiff']).rename('rmse')
    #     mae = error['abdiff'].rename('mae')

    return rmse, mae

def overall_error(run, country, turb_info, cluster_list, time_res_list, train, *args):
    rmse_all = []
    mae_all = []
    cluster_all = []
    time_all = []
    
    if train == True:
        obs_cf = pd.read_csv(run+'/results/capacity-factor/'+country+'_train_obs_cf.csv')
        unc_cf = pd.read_csv(run+'/results/capacity-factor/'+country+'_train_unc_cf.csv', parse_dates=['time'])
    else:
        year_test = args[0]
        obs_cf = pd.read_csv(run+'/results/capacity-factor/'+country+"_"+str(year_test)+'_obs_cf.csv', parse_dates=['time'])
        unc_cf = pd.read_csv(run+'/results/capacity-factor/'+country+"_"+str(year_test)+'_unc_cf.csv', parse_dates=['time'])
    
    rmse, mae = calc_error('total', unc_cf, obs_cf, turb_info, train)
    
    rmse_all.append(rmse)
    mae_all.append(mae)
    cluster_all.append(1)
    time_all.append('uncorrected')
    
    for num_clu in cluster_list:
        for time_res in time_res_list:
            if train == True:
                cor_cf = pd.read_csv(run+'/results/capacity-factor/'+country+'_train_'+time_res+'_'+str(num_clu)+'_cor_cf.csv', parse_dates=['time'])
            else:
                cor_cf = pd.read_csv(run+'/results/capacity-factor/'+country+"_"+str(year_test)+'_'+time_res+'_'+str(num_clu)+'_cor_cf.csv', parse_dates=['time'])
                
            rmse, mae = calc_error('total', cor_cf, obs_cf, turb_info, train)
            
            rmse_all.append(rmse)
            mae_all.append(mae)
            cluster_all.append(num_clu)
            time_all.append(time_res)
    
    
    df_metrics = pd.DataFrame(list(zip(np.ravel(cluster_all), np.ravel(time_all), np.ravel(rmse_all), np.ravel(mae_all))), 
                 columns =['num_clu', 'time_res', 'rmse', 'mae'])
    return df_metrics

def explore_error(metric_type, run, year_test, country, turb_info, cluster_list, time_res_list):
    rmse_all = []
    mae_all = []
    cluster_all = []
    time_all = []
    type_all = []
    
    obs_cf = pd.read_csv(run+'/results/capacity-factor/'+country+'_'+str(year_test)+'_obs_cf.csv', parse_dates=['time'])
    unc_cf = pd.read_csv(run+'/results/capacity-factor/'+country+'_'+str(year_test)+'_unc_cf.csv', parse_dates=['time'])

    rmse, mae = calc_error(metric_type, unc_cf, obs_cf, turb_info)
    
    rmse_all.append(rmse)
    mae_all.append(mae)
    cluster_all.append([1]*len(rmse))
    time_all.append(['uncorrected']*len(rmse))
    type_all.append(rmse.index)
    
    for num_clu in cluster_list:
        for time_res in time_res_list:
            cor_cf = pd.read_csv(run+'/results/capacity-factor/'+country+"_"+str(year_test)+'_'+time_res+'_'+str(num_clu)+'_cor_cf.csv', parse_dates=['time'])
            rmse, mae = calc_error(metric_type, cor_cf, obs_cf, turb_info)
            
            rmse_all.append(rmse)
            mae_all.append(mae)
            cluster_all.append([num_clu]*len(rmse))
            time_all.append([time_res]*len(rmse))
            type_all.append(rmse.index)
    
    
    df_metrics = pd.DataFrame(list(zip(np.ravel(type_all), np.ravel(cluster_all), np.ravel(time_all), np.ravel(rmse_all), np.ravel(mae_all))), 
                 columns =[metric_type,'num_clu', 'time_res', 'rmse', 'mae'])

    return df_metrics
    
    
def plot_overall_error(run, country, df_metrics, name):
    fig, axes = plt.subplots(1, 2, figsize=(12, 3))
    
    sns.lineplot(
        x='num_clu',
        y='rmse',
        hue ="time_res",
        style="time_res",
        hue_order = ['month', 'bimonth', 'season', 'yearly'],
        style_order= ['month', 'bimonth', 'season', 'yearly'],
        data = df_metrics[(df_metrics['time_res'] != 'uncorrected')],
        ax = axes[0],
        legend = False
    )
    axes[0].set_xscale('log')
    axes[0].set_ylabel('NRMSE')
    axes[0].set_xlabel('$n_{clu}$')
    axes[0].set_xticks([1, 10, 100,1000])
    axes[0].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    
    sns.lineplot(
        x='num_clu',
        y='mae',
        hue ="time_res",
        style= "time_res",
        hue_order = ['month', 'bimonth', 'season', 'yearly'],
        style_order= ['month', 'bimonth', 'season', 'yearly'],
        data = df_metrics[(df_metrics['time_res'] != 'uncorrected')],
        ax = axes[1],
        legend = True
    )
    
    axes[1].set_xscale('log')
    axes[1].set_ylabel('NMAE')
    axes[1].set_xlabel('$n_{clu}$')
    axes[1].set_xticks([1, 10, 100,1000])
    axes[1].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    
    # fixing the legend labels and sharing it over whole plot.
    axes[1].get_legend().remove()
    handles, labels = plt.gca().get_legend_handles_labels()
    labels = ['Monthly', 'Bimonthly',  'Seasonal', 'yearly'] # renaming labels
    plt.legend(handles, labels, ncol=1, loc='right', bbox_to_anchor=(1.3, 0.5), title='$t_{freq}$')
    plt.tight_layout()
    plt.savefig(run+'/plots/'+country+'_'+name+'_error.png', bbox_inches='tight')
    return plt.tight_layout()