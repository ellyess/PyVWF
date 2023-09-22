import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interpolate, optimize
import statsmodels.api as sm
import statsmodels.formula.api as smf
import itertools
from sklearn.metrics import mean_squared_error
from math import sqrt
from calendar import monthrange
pd.options.mode.chained_assignment = None
import os 
os.environ['HDF5_USE_FILE_LOCKING']='FALSE'

# Taking what we need from reanalysis data for everything else.
def prep_reanalysis(dataset, mode, min_lat, max_lat, min_lon, max_lon):
    # First take average of the hourly wind speed over each day (otherwise it takes 36 hours for one-year conversion)
    ds = dataset.groupby('time.date').mean()

    def wmeter(u, v):
        """ 
        Calculate wind speed magnitude from u- and v-components
        Inputs:
            u: Eastward component of wind at height h
            v: Northward component of Wind at height h
        Returns:
            Wind magnitude at height h
        """
        return np.sqrt(u**2 + v**2)
        
    if mode == 'ERA5':
        # Get 10m and 100m wind speed magnitude
        # Note that the longitude for original download contained more values than we need so is sliced here
        cropped_ds = ds.sel(latitude=slice(max_lat,min_lat), longitude=slice(min_lon,max_lon))
        u10m = cropped_ds.u10
        v10m = cropped_ds.v10
        u100m = cropped_ds.u100
        v100m = cropped_ds.v100
        
        w10m = wmeter(u10m, v10m)
        w100m = wmeter(u100m, v100m)
        
        return w10m, w100m, np.asarray(cropped_ds.latitude), np.asarray(cropped_ds.longitude)
        
        
    else:
        return "Please select a valid dataset. Either MERRA2 or ERA5"

# CREATING PROFILES (A,z) FOR HEIGHT EXTRAPOLATION
def extrapolate_log_law_ERA5(w10m, w100m, ratio=1):
    """ 
    Derive A and z for wind speed height interpolation formula w(h) = A * np.log(h / z)
    Inputs:
        w10m: Wind magnitude at 10m
        w100m: Wind magnitude at 100m
        ratio: If w100m < w10m, the ratio to scale w100m
    Returns:
        A and z values
    """
    # Check if w100m > w10m, if not scale it by ratio = w100m.mean()/w10m.mean()
    if w100m <= w10m:
        w100m = w10m*ratio

    # assemble our two heights and wind speeds
    h = [10, 100]
    v = [w10m, w100m]
    logh = np.log(h)
    df = pd.DataFrame(np.column_stack((v,logh)))

    # linearise and perform a ls fit, weight the data at 100m more strongly
    reg = smf.wls(formula='v ~ logh', data=df, weights=(1,2)).fit()
    
    # extract our coefficients
    # v = A log(h) - A log(z) therefore slope = A, exp(-intercept / A) = z
    A = reg.params[1]
    z = np.exp(-reg.params[0] / A)

    return A, z

    
def simulate_Az_ERA5(w10m, w100m):    
#     ratios = np.mean(w100m, axis=(1,2))/np.mean(w10m, axis=(1,2))
    ratio = (w100m.mean()/w10m.mean()).values
    log_law_MERRA2 = np.vectorize(extrapolate_log_law_ERA5)
    subA, subz = log_law_MERRA2(w10m, w100m, ratio)
    A = np.transpose(subA, (0, 1, 2))
    z = np.transpose(subz, (0, 1, 2))
    
    return A, z



    
def determine_farm_scalar(PR, match_method=2, iso = None):
    """ 
    Calculate scalar based on the error factor PR = CF_obs/CF_sim
    
    Inputs:
        PR: A ratio derived from PR = CF_obs/CF_sim
        match_method: 1 or 2, to signify which method to use (see below)

    Returns:
        Scalar alpha used in bias correction: w_corr = alpha*w + beta 
    """
    # This was used as an alternative method in RN
    if match_method == 1:
        scalar = 0.85
    
    # This is the method used in my study
    if match_method == 2:
        scalar_alpha = 0.6
        scalar_beta = 0.2
        scalar = (scalar_alpha * PR) + scalar_beta
    return scalar
    

def find_farm_offset(A, z, gendata, azfile, powerCurveFile, farm_ind, myScalar, energyInitial, energyTarget):
    """ 
    Iterative process to find the fixed offset beta in 
    bias correction formula w_corr = alpha*w + beta such that 
    the resulting load factor from simulation model equals energyTarget
    
    Inputs:
        A: Derived A value for height interpolation function: w(h) = A * np.log(h / z)
        z: Derived z value for height interpolation function: w(h) = A * np.log(h / z)
        gendata: DataFrame that contains the meta data for wind turbines
        farm_ind: Turbine row number in gendata
        myScalar: Scalar alpha in formula w_corr = alpha*w + beta 
        energyInitial: Uncorrected CF for this turbine
        energyTarget: Observed CF for this turbine

    Returns:
        Offset beta used in bias correction: w_corr = alpha*w + beta
    """
    myOffset = 0
    
    # decide our initial search step size
    stepSize = -0.64
    if (energyTarget > energyInitial):
        stepSize = 0.64
        
    # Stop when step-size is smaller than our power curve's resolution
    while np.abs(stepSize) > 0.002:
        # If we are still far from energytarget, increase stepsize
        myOffset += stepSize
        
        # Calculate the simulated CF using the new offset
        mylf = wind_speed_to_power_output(A, z, gendata, azfile, powerCurveFile, farm_ind, myScalar, myOffset, True)
        # print(mylf)
        
        
        # If we have overshot our target, then repeat, searching the other direction
        # ((guess < target & sign(step) < 0) | (guess > target & sign(step) > 0))
        if mylf != 0:
            energyGuess = np.mean(mylf)
            if np.sign(energyGuess - energyTarget) == np.sign(stepSize):
                stepSize = -stepSize / 2
            # If we have reached unreasonable places, stop
            if myOffset < -20 or myOffset > 20:
                break
        elif mylf == 0:
            myOffset = 0
            break
    
    return myOffset
    
    
def simulate_wind(azfile, gen_obs, powerCurveFile, scalar=1, offset=0):
    all_heights = np.sort(gen_obs['height'].unique())
    new_ds = azfile.assign_coords(
        height=('height', all_heights))



    w = new_ds.A * np.log(new_ds.height / new_ds.z)

    lat =  xr.DataArray(gen_obs['latitude'], dims='turbine', coords={'turbine':gen_obs['turb_match']})
    lon =  xr.DataArray(gen_obs['longitude'], dims='turbine', coords={'turbine':gen_obs['turb_match']})
    height =  xr.DataArray(gen_obs['height'], dims='turbine', coords={'turbine':gen_obs['turb_match']})

    f = w.interp(
            lat=lat, lon=lon, height=height,
            kwargs={"fill_value": None})
    
    def speed_to_power(speed_frame):
        power_frame = speed_frame.copy()

        if speed_frame.iloc[:,1].name == 'time_level_1':
            
            for i in range(3, len(power_frame.columns)+1):
                speed_single = power_frame.iloc[:,i-1]
                x = powerCurveFile['data$speed']
                y = powerCurveFile[speed_single.name]
                f2 = interpolate.interp1d(x, y, kind='cubic')
                power_frame.iloc[:,i-1] = f2(speed_single)
        else:            
            for i in range(2, len(power_frame.columns)+1):
                speed_single = power_frame.iloc[:,i-1]
                x = powerCurveFile['data$speed']
                y = powerCurveFile[speed_single.name]
                f2 = interpolate.interp1d(x, y, kind='cubic')   
                power_frame.iloc[:,i-1] = f2(speed_single)

        return power_frame
    

    f = (f+offset) * scalar
    f = f.where(f > 0 , 0)
    f = f.where(f < 40 , 40)
    speed_series = f.to_pandas()
    speed_series = speed_series.reset_index()
    power_series = speed_to_power(speed_series)
    
    return speed_series, power_series
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
def wind_speed_to_power_output(A, z, gendata, azfile, powerCurveFile, farm_ind, myScalar=0, myOffset=0, biascorrect=False):
    """ 
    Simulate wind CF from wind speed
    
    Inputs:
        A: Derived A value for height interpolation function: w(h) = A * np.log(h / z)
        z: Derived z value for height interpolation function: w(h) = A * np.log(h / z)
        gendata: DataFrame that contains the meta data for wind turbines
        farm_ind: Turbine row number in gendata
        myScalar: If biascorrect = True, the scalar used for bias correction
        myOffset: If biascorrect = True, the offset used for bias correction
        biascorrect: A boolean value indicating whether to do bias correction
        
    Returns:
        Wind capacity factor for the given turbine.
    """
    
    # for the selected turbine, extract meta information: turbine location, height, and model
    height = gendata['height'][farm_ind]
    lat_farm = gendata['latitude'][farm_ind]
    lon_farm = gendata['longitude'][farm_ind]
    key = gendata['turb_match'][farm_ind]

    if key != 0:
        # Height interpolation
        w = A * np.log(height / z)
        if biascorrect == True:
              w2 = myScalar*w.transpose() + myOffset # the bias correction step
        if biascorrect == False:
              w2 = w.transpose() # Transpose is needed for the interpolate.interp2d function

        # Location interpolation
        f = interpolate.interp2d(azfile.lat, azfile.lon, w2, kind='cubic')
        speed = f(lat_farm,lon_farm)
        
        # Power curve conversion
        x = powerCurveFile['data$speed']
        y = powerCurveFile[key]

        # calculate power from curve
        f2 = interpolate.interp1d(x, y, kind='cubic')
        if (speed > 0) & (speed < 40):
            power = f2(speed)
        else:
            power = 0

    return float(power)
    
    # def simulate_Az_ERA5(w10m, w100m):
#     # NOTE THAT THIS TAKES 20 MINUTES TO RUN
#     # Simulate A and z for every day
#     emptyA = []
#     emptyz = []
#     for k in range(np.shape(w10m)[0]):
#         A = []
#         z = []
#         for i in range(np.shape(w10m)[1]):
#             subA = []
#             subz = []
#             for j in range(np.shape(w10m)[2]):
#                 print('at: ',k,i,j)
#                 ratio_ = (w100m[k].mean()/w10m[k].mean()).values
#                 subA.append(extrapolate_log_law_ERA5(w10m[k,i,j], w100m[k,i,j], ratio_)[0])
#                 subz.append(extrapolate_log_law_ERA5(w10m[k,i,j], w100m[k,i,j], ratio_)[1])
#             A.append(subA)
#             z.append(subz)
            
#         emptyA.append(A)
#         emptyz.append(z)
    
#     fullA = np.asarray(emptyA)
#     fullz = np.asarray(emptyz)
    
#     return fullA, fullz

# def simulate_Az_MERRA2(w2m, w10m, w50m, disph):
#     emptyA = []
#     emptyz = []
#     for k in range(np.shape(w2m)[0]):
#         A = []
#         z = []
#         for i in range(np.shape(w2m)[1]):
#             subA = []
#             subz = []
#             for j in range(np.shape(w2m)[2]):
#                 print('at: ',k,i,j)
#                 subA.append(extrapolate_log_law_MERRA2(w2m[k,i,j], w10m[k,i,j], w50m[k,i,j], disph[k,i,j])[0])
#                 subz.append(extrapolate_log_law_MERRA2(w2m[k,i,j], w10m[k,i,j], w50m[k,i,j], disph[k,i,j])[1])
#             A.append(subA)
#             z.append(subz)
            
#         emptyA.append(A)
#         emptyz.append(z)

#     fullA = np.asarray(emptyA)
#     fullz = np.asarray(emptyz)
    
#     return fullA, fullz