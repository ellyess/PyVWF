############################################################################
#  
#  Copyright (C) 2012-2017  Iain Staffell  <staffell@gmail.com>
#  
#  Unauthorised copying of this file via any medium is strictly prohibited
#  Proprietary and confidential
#  
############################################################################


	# establish two plots
	if (a_few_plots)
	{
		dev.new(height=5.4, width=12, xpos=0, ypos=450)
		good_graphics()
		layout(t(1:3))

		plot_farms(windFarms, pch='+', cap.weight=1)
		dev.set()
	}





####################################

	##
	##  plot the mean wind speeds at each site
	##

	if (ncol(windSpeed) > 2) {
		site_means = colMeans(windSpeed[ , -1])
	} else {
		site_means = windSpeed[ , 2]
	}


	if (a_few_plots)
	{
		ylim = range(site_means/1.1, site_means*1.1)
		plot(sort(site_means), type='l', ylim=ylim, main='Average Speeds by Site')
	}



	##
	##  transform wind speeds based on air density
	##

	if (doAirDensity)
	{
		airDensityFile = dirname(windSpeedFile) %&% '/' %&% gsub('windspeed', 'density', basename(windSpeedFile))
		airDensity = read_data(airDensityFile)

		# safety check
		ok = all(colnames(windSpeed) == colnames(airDensity)) & (nrow(windSpeed) == nrow(airDensity))
		if (!ok) stop("VWF.MAIN.WINDPOWER.R: your windSpeed and airDensity datasets do not line up...\n")

		# calculate
		avgSpeedBefore = colMeans(windSpeed[ , -1])
		for (i in 2:ncol(windSpeed))
		{
			windSpeed[ , i] = air_density_correction(windSpeed[ , i], airDensity[ , i], windFarms$height[i-1])
		}
		avgSpeedAfter = colMeans(windSpeed[ , -1])

		# summary stats
		airLo = quantile(as.matrix(airDensity[ , -1]), 0.1)
		airHi = quantile(as.matrix(airDensity[ , -1]), 0.9)

		adjLo = quantile(avgSpeedAfter / avgSpeedBefore, 0.1) 
		adjHi = quantile(avgSpeedAfter / avgSpeedBefore, 0.9)

		flush("    Air density is", sigfig(airLo, 3), "-", sigfig(airHi, 3), "\b, adjusting wind speeds by", percent(adjLo, 1), "\b\b -", percent(adjHi, 1), "\n")

		# floppy
		if (ncol(windSpeed) > 2) {
			site_means = colMeans(windSpeed[ , -1])
		} else {
			site_means = windSpeed[ , 2]
		}

		if (a_few_plots)
			lines(sort(site_means), col='red')
	}



	##
	##  transform unfathomably low speeds
	##  use a cubic hermite to enforce a lower-bound floor
	##

	if (doTransformLowestSpeeds)
	{
		# our transformation parameters
		# floor is the lowest we can go (m/s)
		# width is how smooth the transition is..
		# e.g. 4 and 1.5 means speeds of 2.5 go up to 4, speeds of 5.5 are unchanged  
		trnsfrm_floor = 3.5
		trnsfrm_width = 1.5

		trnsfrm_min = trnsfrm_floor - trnsfrm_width
		trnsfrm_max = trnsfrm_floor + trnsfrm_width

		w = which(site_means < trnsfrm_min)

		if (length(w) > 0)
		{
			scalars = trnsfrm_floor / site_means[w]

			speed = windSpeed[ , w+1]
			speed = mult_cols_by_vector(speed, scalars)
			windSpeed[ , w+1] = speed
		}

		w = which(site_means >= trnsfrm_min & site_means < trnsfrm_max)

		if (length(w) > 0)
		{
			scalars = trnsfrm_max - site_means[w]
			scalars = scalars ^ 2 / (4 * trnsfrm_width)
			scalars = (site_means[w] + scalars) / site_means[w]

			speed = windSpeed[ , w+1]
			speed = mult_cols_by_vector(speed, scalars)
			windSpeed[ , w+1] = speed
		}

		if (ncol(windSpeed) > 2) {
			site_means_x = colMeans(windSpeed[ , -1])
		} else {
			site_means_x = windSpeed[ , 2]
		}

		if (a_few_plots)
		{
			lines(sort(site_means_x), col='red3')
			lines(sort(site_means), col='red')
		}
	}

#################################





	##
	##  process the wind speed data
	##

	# interpolate to half hourly
	if (halfHourly)
	{
		cat("    Interpolating to half hourly...\n")
		windSpeed = interpolate_wind_speed(windSpeed, resolution=30)
	}

	# separate the date column from the numerics
	#if (!exists('datecol'))
	if (colnames(windSpeed)[1] == 'Timestamp')
	{
		datecol = windSpeed[ , 1]
		windSpeed = windSpeed[ , -1]
	}

	# inject noise into the new profile
	if (halfHourly)
	{
		cat("    Injecting gaussian noise into profile...\n")
		windSpeed = inject_noise_into_wind_speeds(windSpeed, noise_stdev)
	}

	# make sure wind farm names line up in the two files
	windFarms$name = colnames(windSpeed)






#####
## ##  ESTABLISH THE PERFORMANCE RATIO OF ALL FARMS
#####

	##
	## transform wind speeds from the nasa ideal to the on-the-ground reality

	# set the base performance for all farms
	performance = rep(1.000, ncol(windSpeed))

	# modify these with the input file values if desired
	if (!is.null(PR_column))
	{
		performance = performance * windFarms[ , PR_column]
	}

	# show a histogram of performance ratios...
	if (a_few_plots)
		plot(performance)












#####
## ##  CONVERT FROM SPEED TO POWER -- GENERATE LOAD FACTORS
#####

	flush("> Generating load factors with ")

	if (pr_method == 1) flush("multiplicative PR values ")
	if (pr_method == 2) flush("additive PR values ")

	if (match_method == 1) flush("(fixed scalar same for every farm, finding offset)\n")
	if (match_method == 2) flush("(fixed scalar from each farm's PR, finding offset)\n")
	if (match_method == 3) flush("(fixed offset same for every farm, finding scalar)\n")
	if (match_method == 4) flush("(fixed offset from each farm's PR, finding scalar)\n")



	# to avoid passing the whole data.frame...
	colnames_windSpeed = colnames(windSpeed)


	# #WEIBULL_HAXX
	# wxx = seq(0, 1, length.out=1e6)
	# w20 = qweibull(wxx, 2.0, 10)
	# w17 = qweibull(wxx, 1.7, 10)
	# w17 = head(tail(w17, -1), -1)
	# w20 = head(tail(w20, -1), -1)
	# hist(w17, xlim=c(0,40), breaks=100)
	# hist(w20, xlim=c(0,40), breaks=100)
	# plot_l(w20, w20/w17)
	# weibull_spline = spline(w20, w20/w17, xout=seq(0,40,0.001))
	# #WEIBULL_HAXX


	# run through each farm in parallel building a huge mega-list of results
	results = foreach (speed=iter(windSpeed, by='col'), i=icount(), .combine='rbind') %dopar%
	{
		# farm name and turbine model
		myName = windFarms[i, 'name']
		myModel = windFarms[i, 'power_curve']

		# valid variable version of farm name (column name in csv)
		# numeric farm names don't need an X on the front
		#myCol = make.names(myName)
		myCol = myName
		if (is.numeric(myCol))
				myCol = as.character(myCol)

		myModel = make.names(myModel)

		# my individual power curve - unmodified
		myCurve = turbCurve[ , myModel]

		# check our name is ok
		if (myCol %notin% colnames_windSpeed)
			stop("Cannot find the farm:", myName, "\n\n")




		# # interpolate any NAs (which i believe are all zeroish)
		# missing = which(is.na(speed))
		# if (length(missing) > 0)
		# 	speed[missing] = spline(speed, xout=missing)$y


		# calculate load factors using the unmodified turbine power curve
		mylf = wind_speed_to_power_output(speed, myCurve)
		energyInitial = mean(mylf)

		# calculate our target energy yield
		if (pr_method == 2) {
			energyTarget = energyInitial + performance[i]
			performance[i] = energyTarget / energyInitial
		} else {
			energyTarget = energyInitial * performance[i]
		}
		energyTarget = max(0.00, min(energyTarget, 1.00))



		##
		##  calculate modified yield using convoluted farm curve

		# determine how to convolute this to a curve for this whole farm
		mySpread = get_power_curve_convolution_spread(windFarms[i, ])

		# convolute to produce the power curve for this farm
		myCurve = farmCurve[ , myModel]
		myCurve = convoluteFarmCurve(myCurve, 0, mySpread)


		##WEIBULL_HAXX
		#weibull_adjuster = spline(w20, w20/w17, xout=speed)
		#speed = speed * weibull_adjuster$y
		##WEIBULL_HAXX


		# we choose to match using offsets, with fixed scalars
		if (match_method == 1 | match_method == 2)
		{
			# get the scalar for this farm
			myScalar = determine_farm_scalar(performance[i], windFarms$iso[i])
		
			# find the additive convolution offset that gives us the correct yield
			myOffset = find_farm_offset(speed, myCurve, myScalar, energyInitial, energyTarget)
		}

		# we choose to match using scalars, with fixed offsets
		if (match_method == 3 | match_method == 4)
		{
			# determine the offset we should use for this farm
			myOffset = determine_farm_offset(performance[i])

			# find the multiplicative scalar for wind speeds that gives us the correct yield
			myScalar = find_farm_scalar(speed, myCurve, myOffset, energyInitial, energyTarget)
		}

		if (energyTarget == 0 & energyInitial == 0)
		{
			myOffset = 0
			myScalar = 1
		}

		# calculate the final load factors of this farm
		mylf = wind_speed_to_power_output(speed, myCurve, myScalar, myOffset)

		# make it clear if we got stuck
		if (myOffset < -20 | myOffset > 20 | myScalar < 0 | myScalar > 10)
		{
			# mylf = mylf * NA
			flush('\n ~!!~ Not happy calculating', basename(inputFile), 'with a PR of', percent(mean(windFarms$PR)))
			flush('\n ~!!~ I ended up with myOffset =', round(myOffset, 3), 'and myScalar =', round(myScalar, 3), '\n')
		}

		# calculate the power output in MW
		mymw = mylf * windFarms[i, 'capacity']



		# save the the corrected wind speeds if wanted
		if (save_modified_wind_speeds)
		{
			speed = (speed + myOffset) * myScalar

			# very low PR can lead to negative wind speeds with some match methods
			if (any(w < 0))
			{
				# zero breaks this calculation
				speed[speed==0] = 1e-6

				scalars = rep(1, length(speed))

				# our transformation parameters
				# floor is the lowest we can go (m/s)
				# width is how smooth the transition is..
				# e.g. 0 and 0.5 means speeds of -0.5 go up to 0, speeds over 0.5 are unchanged  
				trnsfrm_floor = 0
				trnsfrm_width = 0.5

				trnsfrm_min = trnsfrm_floor - trnsfrm_width
				trnsfrm_max = trnsfrm_floor + trnsfrm_width

				w = which(speed < trnsfrm_min)

				if (length(w) > 0)
				{
					scalars[w] = trnsfrm_floor / speed[w]
				}

				w = which(speed >= trnsfrm_min & speed < trnsfrm_max)

				if (length(w) > 0)
				{
					scalars[w] = trnsfrm_max - speed[w]
					scalars[w] = scalars[w] ^ 2 / (4 * trnsfrm_width)
					scalars[w] = (speed[w] + scalars[w]) / speed[w]
				}

				speed = speed * scalars
			}
		}



		# # print our VWF convolution parameters
		# mo = sprintf('%1.2f', myOffset)
		# cs = sprintf('%1.2f', convolverStdDev)
		# ms = sprintf('%3.1f%%', myScalar * 100)
		# flush('->  [ws + N(', mo, ', ', cs, ')] x ', ms, '\n', sep='')
	

		# save the VWF parameters for this farm
		parms = c(myName, energyInitial, performance[i], energyTarget, myOffset, myScalar, mySpread)

		results = list(
			speed=speed,
			mylf=mylf,
			mymw=mymw,
			parms=parms
		)

		return(results)
	}




#####
## ##  EXTRACT RESULTS
#####

	# now process the mega-results table

	# modified wind speeds
	windSpeed = as.data.frame(results[ , 1])
	colnames(windSpeed) = colnames_windSpeed

	# load factors
	loadFactor = as.data.frame(results[ , 2])
	colnames(loadFactor) = colnames_windSpeed

	# power outputs
	powerMW = as.data.frame(results[ , 3])
	colnames(powerMW) = colnames_windSpeed

	# VWF model parameters
	parms = as.data.frame(results[ , 4], stringsAsFactors=FALSE)
	parms = as.data.frame(t(parms), stringsAsFactors=FALSE)
	for (i in colnames(parms)[-1])
		parms[ , i] = as.numeric(parms[ , i])
	colnames(parms) = c('name', 'original_LF', 'PR', 'desired_LF', 'offset', 'scalar', 'stdev')


	# conserve memory
	rm(results)
	gc()







#####
## ##  POST PROCESSING
#####

	##
	##  transform wind speeds based on wind farm icing
	##

	if (!exists('doAirTemperature'))
		doAirTemperature = FALSE

	if (doAirTemperature)
	{
		airTemperatureFile = dirname(windSpeedFile) %&% '/' %&% gsub('windspeed', 'temperature', basename(windSpeedFile))
		airTemperature = read_data(airTemperatureFile)

		# safety check
		ok = all(c('Timestamp', colnames(windSpeed)) == colnames(airTemperature)) & (nrow(windSpeed) == nrow(airTemperature))
		if (!ok) stop("VWF.MAIN.WINDPOWER.R: your windSpeed and airDensity datasets do not line up...\n")

		# calculate
		avgSpeedBefore = colMeans(powerMW[ , -1])
		allIcing = NULL
		for (i in 2:ncol(powerMW))
		{
			icing = pmax(0, 0 - airTemperature[ , i])
			if (sum(icing) > 0)
			{
				powerMW[ , i] = powerMW[ , i] * (1 - (icing * 0.0089))
				loadFactor[ , i] = loadFactor[ , i] * (1 - (icing * 0.0089))
			}
			push(allIcing, sum(icing))
		}
		avgSpeedAfter = colMeans(powerMW[ , -1])

		# summary stats
		iceLo = quantile(allIcing, 0.1) * 365 / nrow(powerMW)
		iceHi = quantile(allIcing, 0.9) * 365 / nrow(powerMW)

		adjLo = quantile(avgSpeedAfter / avgSpeedBefore, 0.1)
		adjHi = quantile(avgSpeedAfter / avgSpeedBefore, 0.9)

		flush("    Icing degree days are", sigfig(iceLo, 2), "to", sigfig(iceHi, 2), "\b, adjusting power output by", percent(adjLo, 1), "to", percent(adjHi, 1), "\n")

	}



	if (a_few_plots)
	{
		# histogram of site load factors
		cf = parms$desired_LF * 100
		hist(cf, breaks=0:ceiling(max(cf)), col='grey90')
		abline(v=mean(cf), lwd=2, col='red3')

		# map of capacity factors
		dev.set()
		cfc = colour_ramp_data(cf)
		plot_farms(windFarms, cap.weight=1, add.points=TRUE, pch='+', col=cfc)
		dev.set()
	}


	# run garbage collection to keep memory in check
	junk = capture.output(gc())
	