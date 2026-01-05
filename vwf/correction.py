"""
correction module.

Summary
-------
Bias correction modules.

Data conventions
----------------
Expected dimensions follow xarray conventions (e.g., time × lat × lon) unless stated otherwise.
Time coordinates are assumed to be UTC unless explicitly converted by the caller.

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
import xarray as xr
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from calendar import monthrange

from vwf.wind import (
    train_simulate_wind
)


def calculate_scalar(gen_cf, time_res):
    """
    Calculate the scalar, multiplicative correction factor.

        Args:
            time_res (str): time resolution we want to simulate
            gen_data (pandas.DataFrame): dataframe with the observed and simulated cf for training period

        Returns:
            turb_info (pandas.DataFrame): turbine metadata with assigned cluster column
    """
    def weighted_avg(group_df, whole_df, values, weights):
        """
        Weighted avg.

            Args:
                group_df (Any): TODO.
                whole_df (Any): TODO.
                values (Any): TODO.
                weights (Any): TODO.
                *args (tuple): Additional positional arguments.

            Returns:
                None: TODO.

            Assumptions:
                - Datetime handling is assumed to be UTC unless stated otherwise.
                - Units are assumed to be consistent with SI conventions unless stated otherwise.
        """
        v = whole_df.loc[group_df.index, values]
        w = whole_df.loc[group_df.index, weights]
        return (v * w).sum() / w.sum()
        
    df = gen_cf.groupby([time_res, 'cluster', 'year']).agg({
                            "obs": lambda x: weighted_avg(x, gen_cf, 'obs', 'capacity'),
                            "sim": lambda x: weighted_avg(x, gen_cf, 'sim', 'capacity'),
                            })
        
    # bias_data['scalar'] = (0.6 * (bias_data['obs'] / bias_data['sim'])) + 0.2
    df['scalar'] = df['obs'] / df['sim']
    df = df.reset_index()
    df.columns = ['time_slice', 'cluster', 'year', 'obs', 'sim', 'scalar']
        
    return df[['year', 'time_slice', 'cluster', 'obs', 'sim', 'scalar']]
    
def find_offset(row, turb_info, reanalysis, powerCurveFile):
    """
    Optimize the offset, additive correction factor.
    
    This function applies the correction factors to the simulated wind speed
    which is converted into capacity factor, this is done repeatedly till an
    offset is calculated that makes the simulated capacity factor match the
    observed capacity factor.

        Args:
            row (pandas.Series): row has the year, cluster and time slice
            turb_info (pandas.DataFrame): turbine metadata including height and coordinates
            reanalysis (xarray.Dataset): wind parameters on a grid
            powerCurveFile (pandas.DataFrame): capacity factor at increasing wind speeds for different models
            
        Returns:
            offset (float): best offset value
    """
    if row['time_slice'] == 'spring':
        time_slice = [3,4,5]
    elif row['time_slice'] == 'summer':
        time_slice = [6,7,8]
    elif row['time_slice'] == 'autumn':
        time_slice = [9,10,11]
    elif row['time_slice'] == 'winter':
        time_slice = [1,2,12]
    elif row['time_slice'] == '1/6':
        time_slice = [1,2]
    elif row['time_slice'] == '2/6':
        time_slice = [3,4]
    elif row['time_slice'] == '3/6':
        time_slice = [5,6]
    elif row['time_slice'] == '4/6':
        time_slice = [7,8]
    elif row['time_slice'] == '5/6':
        time_slice = [9,10]
    elif row['time_slice'] == '6/6':
        time_slice = [11,12]
    elif row['time_slice'] == '1/1':
        time_slice = [1,2,3,4,5,6,7,8,9,10,11,12]
    else:
        time_slice = int(row['time_slice'])
    
    #initialize step size
    step = np.sign(row.obs - row.sim) * 10
    step_prev = step
    offset = 0
    n = 30 # max 30 iterations
    
    # loop will run until the step size is smaller than our power curve's resolution
    while np.abs(step) > 0.002: 
        if n == 0:
            offset = np.nan
            break
        n -= 1
        
        # calculate the mean simulated CF using the new offset
        mean_sim_cf = train_simulate_wind(
            reanalysis.sel(
                    time=np.logical_and(
                    reanalysis.time.dt.year == row.year, 
                    reanalysis.time.dt.month.isin(time_slice)
                    )
            ),
            turb_info.loc[turb_info['cluster'] == row.cluster],
            powerCurveFile, 
            row.scalar, 
            offset
        )
        
        # wind power is roughly cube of wind speed, this provides a good step size
        step = np.cbrt(row.obs - mean_sim_cf)
        # prevent the step size oscilating between + and -
        if np.sign(step) != np.sign(step_prev) and np.abs(step) > np.abs(step_prev):
            step = -step_prev/2

        # ensure the step size reduces through iterations
        elif np.sign(step) == np.sign(step_prev) and np.abs(step) > np.abs(step_prev):
            step = step_prev/2
            
        step_prev = step
        offset += step
    return offset

