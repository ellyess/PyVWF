import numpy as np
import pandas as pd
import seaborn as sns
from scipy.interpolate import CubicSpline

def air_density_correction(w, rho_0, z):
    """
    Correct windspeeds based on air density
    Parameters:
    w : float
        wind speed at hub height (m/s)
    rho_0 : float
        air density at ground level (kg/m^3)
    z : float
        hub height (m)
    Returns:
    w : float
        corrected windspeed (m/s)

    """
    # estimate air density at hub height using the international standard atmosphere
    rho = rho_0 - 0.00012*z

    # decide the exponent based on the wind speed
    # 1/3 up to 8 m/s, then a smooth-step function up to 2/3 above 13 m/s
    # this was fitted in matlab from the form x = [0, 1]; y = -2x^3 + 3x^2;
    def m_func(w):
        if w < 8:
            m = 1/3
        elif w > 13:
            m = 2/3
        else:
            m = -2/375 * w**3  +  21/125 * w**2  -  208/125 * w  +  703/125

        return m

    # modify the wind speed
    return w * (rho / 1.225)^m


# I'M VERY CONFUSED ABOUT THIS FUNCTION I'M NOT SURE ABOUT THE CONVERSION INQUIRE
def wind_speed_to_power_output(windSpeed, powerCurve, myScalar=1.00, myOffset=0.00):
    """
    Take a time series of wind speeds and convert to load factors
    Parameters:
    windSpeed : array
        a vector of wind speeds
    powerCurve : array
        the power curve of the wind turbine / farm
    z : float
        the scale factor and offset for adjusting wind speeds
    Returns:
    w : float
        corrected windspeed (m/s)

    """
    # transform the wind speeds
    windSpeed = (windSpeed + myOffset) * myScalar

	# convert to a power curve index
	# (i.e. which element of powerCurve to access)
    # CHECK THIS AGAIN AS INDEXING MIGHT BE DIFF
    index = 100 * windSpeed # + 1
    index = np.where(index < 0, 0, index)
    index = np.where(index > 3999, 3999, index)

    return np.take(powerCurve, index)


# find the fixed offset to add to wind speeds such that the resulting load factor equals energyTarget
#
# pass:
#   speed = the wind speed time series
#   farmCurve = the power curve you want to use for this farm
#   myScalar = the scalar you want to use for this farm
#   energyInitial = the initial load factor coming from unmodified MERRA
#   energyTarget = the desired load factor that we are fitting to
def find_farm_offset(speed, farmCurve, myScalar, energyInitial, energyTarget, verbose=False):
    myOffset = 0

	# decide our initial search step size
	stepSize = -0.64
	if (energyTarget > energyInitial):
		stepSize = 0.64

    while np.abs(stepSize) > 0.002:
        # change our offset
		myOffset += stepSize

		# calculate the yield with our farm curve
		mylf = wind_speed_to_power_output(speed, farmCurve, myScalar, myOffset)
		energyGuess = np.mean(mylf)

        if np.sign(energyGuess - energyTarget) == np.sign(stepSize):
            stepSize = -stepSize / 2

        if myOffset < -20 or myOffset > 20:
            break

    return myOffset

# find the scalar to multiply wind speeds by such that the resulting load factor equals energyTarget
#
# pass:
#   speed = the wind speed time series
#   farmCurve = the power curve you want to use for this farm
#   myOffset = the offset you want to use for this farm
#   energyInitial = the initial load factor coming from unmodified MERRA
#   energyTarget = the desired load factor that we are fitting to
def find_farm_scalar(speed, farmCurve, myOffset, energyInitial, energyTarget, verbose=False):
    myScalar = 1.00

	# decide our initial search step size
	stepSize = -0.128
	if (energyTarget > energyInitial):
		stepSize = 0.128

    while np.abs(stepSize) > 0.0002:
        # change our offset
		myScalar += stepSize

		# calculate the yield with our farm curve
		mylf = wind_speed_to_power_output(speed, farmCurve, myScalar, myOffset)
		energyGuess = np.mean(mylf)

        if np.sign(energyGuess - energyTarget) == np.sign(stepSize):
            stepSize = -stepSize / 2

        if myScalar <= 0 or myScalar > 5:
            break

    return myScalar


# PLOT
def plot_power_curve_and_wind_spee(speed, myCurve, myOffset, mySpread):
    x = np.arange(0, 40, 0.01)
    y = myCurve
    y = y.astype('float')
    y[y==0] = np.nan

    sns.kdeplot(np.array(speed))
    sns.lineplot(x,y)

    return None



########################
# Reshape data 	#  function to reshape a wide format matrix (with only row and column numbers as descriptors)
	#  to long format, optionally specifying the labels to assign to each row and column number.

# could potentially just use pd.melt for this function???
#################


def interpolateTable(curve):
    return smoothCurve = CubicSpline(curve)

# 2 more functions I need to work on.
