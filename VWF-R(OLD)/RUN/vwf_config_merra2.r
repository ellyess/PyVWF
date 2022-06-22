
# some tips..
# save to a different physical drive to the one you read merra data from
# i'm not sure if you get much performance gain from more cores... 20 is ~1/3 utilisation of my CPUs



	## MOST COMMON OPTIONS

	# where to save everything - filenames are automatically generated
	baseSaveFolder = 'M:/WORK/Wind Modelling/2020.09 - Malte Offshore/Simulations/'
	baseSaveFile = get_text_before(basename(inputFile), '.', last=TRUE) %&% '.'

	# what format to save as: csv rds or rdata
	xtn = 'rds'

	# which years to calculate
	yearRange = 1980:2019

	# set how many cores you wish to use 
	n.cores = 8





	##  MERRA EXTRAPOLATED WIND DATA

	# do you want to use custom pre-compiled windspeeds?
	# set this to NULL to use the default speeds from a previous model run, or to calculate new ones 
	windSpeedFile = NULL

	# the files which contain our data
	# this relies on files being named YYYY.nc or YYYY-MM.nc
	merraFiles = 'K:/NINJA/MERRA2-WIND/' %&% yearRange %&% '.nc'

	# process most recent data first?
	processNewestFirst = FALSE

	# which model are we using 'MERRA1' or 'MERRA2' or 'ERA5' or 'CMIP5'
	reanalysis = 'MERRA2'





	##  POWER CONVERSION PARAMETERS

	# the parameters we use to determine the scalar and offset for each farm
	# scalar = (scalar_alpha * PR) + scalar_beta
	scalar_alpha = 1/2
	scalar_beta = 1/3

	# specify the parameters for multi-turbine farm curve
	convolverStdDev = 0.20			# must be > 0
	convolverSpeedSmooth = 0.20		# must be 0.00, 0.02, 0.04, ..., 0.30

	# the files our power curves live in (raw turbine curves and smoothed farm curves)
	turbCurveFile = 'M:/WORK/Wind Modelling/~ Wind Turbine Power Curves/R/Wind Turbine Power Curves ~ 8 (0.01ms with 0.00 w smoother).csv'
	farmCurveFile = 'M:/WORK/Wind Modelling/~ Wind Turbine Power Curves/R/Wind Turbine Power Curves ~ 8 (0.01ms with ' %&% sprintf("%.2f", convolverSpeedSmooth) %&% ' w smoother).csv'

	# should we inflate offshore farm PR by 1.16 (i.e. you haven't done it in your farms file, or you have separately calibrated onshore/offshore)
	inflate_offshore_pr = TRUE





	##  HOW TO TRANSFORM WIND SPEEDS

	# do we require speeds interpolated to half-hourly?
	halfHourly = FALSE

	# if so, inject noise into the half-hourly wind speeds to represent the distribution of power swings more closely
	# stdev of normal distribution that things are multiplied by
	noise_stdev = 0.04

	# do you want to factor air density into the calculations?
	# if so airDensityFile must be specified
	doAirDensity = FALSE
	airDensityFile = NULL

	# should we transform unfathomably low wind speeds?
	# i.e. sites with a long-run average < 4 m/s ... just wouldn't have a wind farm!
	doTransformLowestSpeeds = TRUE




	##  RESULTS FILES

	# should the original wind speeds for each farm be saved?
	# this will let you skip the first (slow) part of processing if you want to rerun
	# i'm not entirely sure the code would even work with this FALSE
	save_original_wind_speeds = TRUE

	# should you calculate the annual average wind speed for every year and location?
	# use this for bias correction :)
	save_annual_wind_speeds = FALSE

	# the VWF uses several CPU cores to calculate individual years of data
	# and then joins these together to a single multi-year file
	# avoiding that stage speeds things up, but beware the rest of the model 
	# won't work unless you go and do that seperately
	join_wind_speed_files = TRUE

	# should the modified wind speeds for each farm be saved?
	# these are the 'bias corrected' speeds which give the desired energy output
	save_modified_wind_speeds = FALSE

	# should the hourly power output / capacity factor for each farm be saved?
	# note, this file will be similar in size to the wind speeds (i.e. big)
	save_hourly_farm_mw = FALSE
	save_hourly_farm_cf = TRUE

	# should time aggregates of individual farms also be saved?
	save_farms_monthly = FALSE
	save_farms_annualy = FALSE



	# should MW/CF be aggregated across farms?
	# use '*' to aggregate all farms together (MW is the sum of all, CF is the capacity-weighted average)
	# use other columns names from windFarms to split into groups
	# use '' to do no aggregation at all
	#
	# e.g. c('iso', 'offshore') will give one file with columns for each country, one file with all onshore and offshore seperated
	#      if you'd like onshore/offshore seperated *in* each country, create a new column in windFarms (windFarms$iso_off = windFarms$iso %&% ifelse(tolower(windfarms$offshore) == 'no', 'ON', 'OFF'))
	#
	# '*' is quick... others are slow as they also create the 'evolving' total (where farms don't produce before they were born) for validation purposes
	#
	save_files_split_by = c('*')
	save_split_snapshot = FALSE
	save_split_evolving = FALSE
	save_split_averages = FALSE


	# ...
	save_model_parameters = FALSE



	##  GLORIOUS GUI TYPE STUFF

	# do you want a graphical / interactive experience (at the expense of speed)
	lots_of_plots = FALSE

	# do you any plots at all (at the expense of R claiming the foreground when redrawing)
	a_few_plots = FALSE
