import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

# calculate the wind extrapolation parameters for a given data point (one location and height)
# takes four parameters: wind speed at 2, 10, 50 metres, and the displacement height
# returns two parameters: the scale factor (A) and reference height (z)
# wind speed at any height can be calculated from: v = A * log(h / z)
def extrapolate_log_law(w2m, w10m, w50m, dh):
    # assemble our three heights and wind speeds
    h = np.array([2 + dh, 10 + dh, 50]).reshape(-1,1)
    v = np.array([w2m, w10m, w50m]).reshape(-1,1)
    df = pd.DataFrame(np.hstack((v,h)))

    # linearise and perform a ls fit
    # weight the data at 50m more strongly
    logh = np.log(h)
    reg = smf.wls(formula='v ~ logh', data=df, weights=(1,1,2)).fit()

    # extract our coefficients
    # v = A log(h) - A log(z) therefore slope = A, exp(-intercept / A) = z
    A = res.params[1]
    z = np.exp(-res.params[0] / A)

    return A, z

# calculate the wind extrapolation parameters for a given data point (one location and height)
# takes four parameters: wind speed at 2, 10, 50 metres, and the displacement height
# returns two parameters: the scale factor (epsilon) and shear coefficient (alpha)
# wind speed at any height can be calculated from: v = epsilon * h ^ alpha
def extrapolate_power_law(w2m, w10m, w50m, dh):
    # assemble our three heights and wind speeds
    h = np.array([2 + dh, 10 + dh, 50]).reshape(-1,1)
    v = np.array([w2m, w10m, w50m]).reshape(-1,1)
    df = pd.DataFrame(np.hstack((v,h)))

    # linearise and perform a ls fit
    # weight the data at 50m more strongly
    logh = np.log(h)
    logv = np.log(v)
    reg = smf.wls(formula='logv ~ logh', data=df, weights=(1,1,2)).fit()


    # extract our coefficients
    # v2 / v1 =  (h2 / h1) ^ alpha, therefore v2 = epsilon * h ^ alpha
    epsilon = np.exp(reg.params[0])
    alpha = reg.params[1]

    return epsilon, alpha



# wrapper function to calculate the extrapolation parameters for all data points
# takes the filename to be processed
# returns a list of arrays, one containing each parameter returned by the chosen extrapolation function
def extrapolate_ncdf(ncdf_file):

    # ds = xr.open_dataset(ncdf_file)
    # df = ds.to_dataframe()
