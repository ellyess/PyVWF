"""
metrics module.

Summary
-------
Calculate the metrics for model evaluation.

Data conventions
----------------
Tabular inputs are assumed to be tidy (one observation per row) unless stated otherwise.
Datetime columns are assumed to be timezone-naive UTC unless specified.

Units
-----
Wind speed: [m s^-1]; Hub height: [m]; Power: [MW]; Energy: [MWh]; Capacity factor: [-] (unless stated otherwise).

Assumptions
-----------
- ERA5/reanalysis fields are treated as representative at the chosen spatial/temporal resolution.
- Wake effects, curtailment, availability losses are not modelled unless explicitly implemented in this module.

References
----------
Add dataset and methodological references relevant to this module.
"""
import numpy as np
import pandas as pd
# def cluster_metrics(num_clu, turb_train, train=False, *args):
#     # fitting clusters to training data
#     kmeans = KMeans(
#             init="random",
#             n_clusters = num_clu,
#             n_init = 10,
#             max_iter = 300,
#             random_state = 42
#         )
#     kmeans.fit(turb_info_train[['lat','lon']])
        
#     if train == True:
#         turb_info_train['cluster'] = kmeans.predict(turb_info_train[['lat','lon']])
#         silhouette = silhouette_score(turb_info[['lat','lon']], turb_info['cluster'])
#         sse = kmeans.inertia_
#         return turb_info_train, silhouette, sse
#     else:
#         turb_info = args[0]
#         turb_info['cluster'] = kmeans.predict(turb_info[['lat','lon']])
#         turb_info['distance'] = kmeans.transform(turb_info[['lat','lon']]).min(axis=1)
#         turb_info['silhouette_vals'] = silhouette_samples(turb_info[['lat','lon']], turb_info['cluster'])
#         return turb_info

def calculate_error(type, df_sim, df_obs, turb_info, train=False):
    """
    Calculate error.

        Args:
            type (Any): TODO.
            df_sim (Any): TODO.
            df_obs (Any): TODO.
            turb_info (Any): TODO.
            train (Any): TODO.
            *args (tuple): Additional positional arguments.

        Returns:
            None: TODO.

        Assumptions:
            - Datetime handling is assumed to be UTC unless stated otherwise.
            - Units are assumed to be consistent with SI conventions unless stated otherwise.
    """
    if train == True:
        df_obs = df_obs.pivot(
                index=['year','month'], 
                columns='ID', 
                values='obs').reset_index(drop=True)
        df_obs['time'] = np.arange(str(2015)+'-01', str(2019+1)+'-01', dtype='datetime64[M]')
        df_obs['month'] = df_obs.time.dt.month
        df_obs['year'] = df_obs.time.dt.year
        df_obs_monthly = df_obs.drop(columns=['time']).groupby(['year','month']).mean().reset_index()
        df_obs_monthly = df_obs_monthly.melt(id_vars=['year','month'], var_name='ID', value_name='cf')
    else:
        df_obs['time'] = pd.to_datetime(df_obs['time'])
        df_obs['month'] = df_obs.time.dt.month
        df_obs['year'] = df_obs.time.dt.year
        df_obs_monthly = df_obs.drop(columns=['time']).set_index('month').reset_index()
        df_obs_monthly = df_obs_monthly.melt(id_vars=['year','month'], var_name='ID', value_name='cf')
    
    df_sim['time'] = pd.to_datetime(df_sim['time'])
    df_sim['month'] = df_sim.time.dt.month
    df_sim['year'] = df_sim.time.dt.year
    df_sim_monthly = df_sim.drop(columns=['time']).groupby(['year','month']).mean().reset_index()
    df_sim_monthly = df_sim_monthly.melt(id_vars=['year','month'], var_name='ID', value_name='cf')

    turb_info['ID'] = turb_info['ID'].astype(str)
    df_obs_monthly['ID'] = df_obs_monthly['ID'].astype(str)
    df_sim_monthly['ID'] = df_sim_monthly['ID'].astype(str)
    
    merged = pd.merge(df_sim_monthly, df_obs_monthly, on=['ID', 'month', 'year'], suffixes=('_sim', '_obs'))
    if type == 'regional-error':
        merged = pd.merge(merged, turb_info[['ID', 'capacity','region', 'In training?']], on='ID')
    elif (type == 'cluster-error'):
        merged = pd.merge(merged, turb_info[['ID', 'capacity','cluster']], on='ID')
    elif (type == 'turbine-error'):
        merged = pd.merge(merged, turb_info[['ID', 'capacity','distance']], on='ID')
    elif type == 'monthly-error':
        merged = pd.merge(merged, turb_info[['ID', 'capacity','In training?']], on='ID')
    else:
        merged = pd.merge(merged, turb_info[['ID', 'capacity']], on='ID')
    
    
    merged = merged.dropna(subset=['cf_sim', 'cf_obs', 'capacity']).reset_index(drop=True)

    def weighted_avg(df, values, weights):
        """
        Weighted avg.

            Args:
                df (Any): TODO.
                values (Any): TODO.
                weights (Any): TODO.
                *args (tuple): Additional positional arguments.

            Returns:
                None: TODO.

            Assumptions:
                - Datetime handling is assumed to be UTC unless stated otherwise.
                - Units are assumed to be consistent with SI conventions unless stated otherwise.
        """
        return (df[values] * df[weights]).sum() / df[weights].sum()


    if type == 'monthly-error': # country-monthly
        wavg_obs = lambda x: weighted_avg(merged.loc[x.index], 'cf_obs', 'capacity')
        wavg_sim = lambda x: weighted_avg(merged.loc[x.index], 'cf_sim', 'capacity')
        count = lambda x: merged.loc[x.index]['ID'].count()
        averaged = merged.groupby(['month', 'In training?']).agg({'cf_obs': wavg_obs, 'cf_sim': wavg_sim, 'ID': count}).reset_index().set_index('month')
        # averaged['diff'] = (np.abs(averaged['cf_sim'] - averaged['cf_obs'])/averaged['cf_obs']) * 100
        averaged['diff'] = (averaged['cf_sim'] - averaged['cf_obs'])

        averaged_total = merged.groupby(['month']).agg({'cf_obs': wavg_obs, 'cf_sim': wavg_sim, 'ID': count})
        # averaged_total['diff'] = (np.abs(averaged_total['cf_sim'] - averaged_total['cf_obs'])/averaged_total['cf_obs'])*100
        averaged_total['diff'] = (averaged_total['cf_sim'] - averaged_total['cf_obs'])
        averaged_total['In training?'] = 'Both'

        averaged = pd.concat([averaged, averaged_total])
        
        be = averaged[['diff','In training?','ID']]
        return be

    
    elif type == 'regional-error': # region-yearly
        averaged = merged.groupby('ID').agg({'cf_obs': np.mean, 'cf_sim': np.mean, 'region': 'first', 'In training?': 'first', 'capacity':'first'}).reset_index()
        wavg_obs = lambda x: weighted_avg(averaged.loc[x.index], 'cf_obs', 'capacity')
        wavg_sim = lambda x: weighted_avg(averaged.loc[x.index], 'cf_sim', 'capacity')
        count = lambda x: averaged.loc[x.index]['ID'].count()
        averaged_type = averaged.groupby(['region','In training?']).agg({'cf_obs': wavg_obs, 'cf_sim': wavg_sim, 'ID': count})
        averaged_type['diff'] = (np.abs(averaged_type['cf_sim'] - averaged_type['cf_obs'])/averaged_type['cf_obs'])*100
        # averaged_type['diff'] = np.abs(averaged_type['cf_sim'] - averaged_type['cf_obs'])
        averaged_type = averaged_type.reset_index().set_index('region')

        averaged_total = averaged.groupby('region').agg({'cf_obs': wavg_obs, 'cf_sim': wavg_sim, 'ID': count})
        # averaged_total['diff'] = (np.abs(averaged_total['cf_sim'] - averaged_total['cf_obs'])/averaged_total['cf_obs'])*100
        averaged_total['diff'] = averaged_total['cf_sim'] - averaged_total['cf_obs']
        
        averaged_total['In training?'] = 'Both'
        
        averaged = pd.concat([averaged_type, averaged_total])
        be = averaged[['diff','In training?','ID','cf_obs','cf_sim']]
        return be
        
    elif type == 'turbine-error': # region-yearly
        averaged = merged.groupby('ID').mean()
        averaged['diff'] = averaged['cf_sim'] - averaged['cf_obs']
        # averaged['diff'] = (np.abs(averaged['cf_sim'] - averaged['cf_obs'])/averaged['cf_obs'])*100
        return averaged

    elif type == 'cluster-error': # cluster-yearly
        averaged = merged.groupby('ID').mean().reset_index()
        wavg_obs = lambda x: weighted_avg(averaged.loc[x.index], 'cf_obs', 'capacity')
        wavg_sim = lambda x: weighted_avg(averaged.loc[x.index], 'cf_sim', 'capacity')
        count = lambda x: averaged.loc[x.index]['ID'].count()
        averaged = averaged.groupby('cluster').agg({'cf_obs': wavg_obs, 'cf_sim': wavg_sim, 'ID': count})
        averaged['diff'] = averaged['cf_sim'] - averaged['cf_obs']
        
        be = averaged['diff']
        return be, averaged['ID']
        
    elif type == 'temporal-focus': # country-monthly
        wavg_obs = lambda x: weighted_avg(merged.loc[x.index], 'cf_obs', 'capacity')
        wavg_sim = lambda x: weighted_avg(merged.loc[x.index], 'cf_sim', 'capacity')
        averaged = merged.groupby('month').agg({'cf_obs': wavg_obs, 'cf_sim': wavg_sim})
        averaged['diff'] = averaged['cf_sim'] - averaged['cf_obs']
        
        rmse = np.sqrt((averaged['diff']**2).mean())
        mae = np.abs(averaged['diff']).mean()
        mbe = averaged['diff'].mean()
        
    elif type == 'spatial-focus': # turbine-yearly
        averaged = merged.groupby('ID').mean()
        averaged['diff'] = averaged['cf_sim'] - averaged['cf_obs']
        averaged['sqdiff'] = averaged['diff']**2
        averaged['abdiff'] = np.abs(averaged['diff'])
        wavg_sq = lambda x: weighted_avg(averaged.loc[x.index], 'sqdiff', 'capacity')
        wavg_ab = lambda x: weighted_avg(averaged.loc[x.index], 'abdiff', 'capacity')
        wavg_diff = lambda x: weighted_avg(averaged.loc[x.index], 'diff', 'capacity')
        
        rmse = np.sqrt(averaged.agg({'sqdiff': wavg_sq}))
        mae = averaged.agg({'abdiff': wavg_ab})
        mbe = averaged.agg({'diff': wavg_diff})
        
    elif type == 'total':
        merged['diff'] = merged['cf_sim'] - merged['cf_obs']
        merged['abdiff'] = np.abs(merged['diff'])
        merged['sqdiff'] = merged['diff']**2
        merged = merged.groupby('ID').mean()
        wavg_sq = lambda x: weighted_avg(merged.loc[x.index], 'sqdiff', 'capacity')
        wavg_ab = lambda x: weighted_avg(merged.loc[x.index], 'abdiff', 'capacity')
        wavg_diff = lambda x: weighted_avg(merged.loc[x.index], 'diff', 'capacity')
        averaged = merged.agg({"sqdiff": wavg_sq, 'abdiff': wavg_ab, 'diff': wavg_diff})
        
        rmse = np.sqrt(averaged['sqdiff'])
        mae = averaged['abdiff']
        mbe = averaged['diff']

    return rmse, mae, mbe
    
    
def overall_error(type, run, country, turb_info, cluster_list, time_res_list, train, *args):
    """
Overall error.

    Args:
        type (Any): TODO.
        run (Any): TODO.
        country (Any): TODO.
        turb_info (Any): TODO.
        cluster_list (Any): TODO.
        time_res_list (Any): TODO.
        train (Any): TODO.
        *args (tuple): Additional positional arguments.

    Returns:
        None: TODO.

    Assumptions:
        - Datetime handling is assumed to be UTC unless stated otherwise.
        - Units are assumed to be consistent with SI conventions unless stated otherwise.
"""
    rmse_all = []
    mae_all = []
    mbe_all = []
    cluster_all = []
    time_all = []
    
    if train == True:
        obs_cf = pd.read_csv(run+'/results/capacity-factor/'+country+'_train_obs_cf.csv')
        unc_cf = pd.read_csv(run+'/results/capacity-factor/'+country+'_train_unc_cf.csv', parse_dates=['time'])
    else:
        year_test = args[0]
        obs_cf = pd.read_csv(run+'/results/capacity-factor/'+country+"_"+str(year_test)+'_obs_cf.csv', parse_dates=['time'])
        unc_cf = pd.read_csv(run+'/results/capacity-factor/'+country+"_"+str(year_test)+'_unc_cf.csv', parse_dates=['time'])
    
    rmse, mae, mbe = calculate_error(type, unc_cf, obs_cf, turb_info, train)
    
    rmse_all.append(rmse)
    mae_all.append(mae)
    mbe_all.append(mbe)
    cluster_all.append(1)
    time_all.append('uncorrected')
    
    for num_clu in cluster_list:
        for time_res in time_res_list:
            if train == True:
                cor_cf = pd.read_csv(run+'/results/capacity-factor/'+country+'_train_'+time_res+'_'+str(num_clu)+'_cor_cf.csv', parse_dates=['time'])
            else:
                cor_cf = pd.read_csv(run+'/results/capacity-factor/'+country+"_"+str(year_test)+'_'+time_res+'_'+str(num_clu)+'_cor_cf.csv', parse_dates=['time'])
                
            rmse, mae, mbe = calculate_error(type, cor_cf, obs_cf, turb_info, train)
            
            rmse_all.append(rmse)
            mae_all.append(mae)
            mbe_all.append(mbe)
            cluster_all.append(num_clu)
            time_all.append(time_res)
    
    
    df_metrics = pd.DataFrame(list(zip(np.ravel(cluster_all), np.ravel(time_all), np.ravel(rmse_all), np.ravel(mae_all), np.ravel(mbe_all))), 
                 columns =['num_clu', 'time_res', 'rmse', 'mae', 'mbe'])
    
    # if train == True:
    #     df_metrics.to_csv(run+'/results/'+country+'_train_metrics.csv', index = None)
    # else:
    #     df_metrics.to_csv(run+'/results/'+country+'_test_metrics.csv', index = None)
    return df_metrics