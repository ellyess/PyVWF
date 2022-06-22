
	source('M:/WORK/Wind Modelling/VALIDATION/Hourly vs TNOs/national_wind_validate (version 2016).r')


	aggregate_output_by_farm_type = function(split_cat)
	{
		# establish results storage
		n = length(datecol)
		m = nrow(windFarms)
		snapshot_capacity = snapshot_output = snapshot_cf = data.frame(GMT=ymd_wtf(datecol))
		evolving_capacity = evolving_output = evolving_cf = data.frame(GMT=ymd_wtf(datecol))

		# understand what time period this is
		min_date = min(evolving_output$GMT)
		max_date = max(evolving_output$GMT)
		total_duration = (max_date - min_date) / dyears(1)

		iii = 0

		# go through each value in that column
		for (split_val in sort(unique(windFarms[ , split_cat])))
		{
			s_c = s_o = rep(0, n)
			e_c = e_o = rep(0, n)

			# filter our farms - filter gives the farm row / speed column
			filter = which(windFarms[ , split_cat] == split_val)
			if (length(filter) == 0)
				next

			# calculate the total output and capacity
			for (f in filter)
			{
				my_output = powerMW[ , f]
				my_capacity = rep(windFarms$capacity[f], n)
				my_start = dmy(windFarms$date[f])

				# simply add these on to create the snapshot
				s_o = s_o + my_output
				s_c = s_c + my_capacity

				# if we have a start date, then wipe out the period before birth
				if (!is.na(my_start))
				{
					unborn = (evolving_capacity$GMT < my_start)

					# what percentage of the whole period have you lived for?
					# take that to be your capacity - so that the oldest farms have
					# the greatest weight when we go so far back that we know nothing
					my_duration = (max_date - my_start) / dyears(1)
					my_duration = my_duration / total_duration
					my_scalar = (0.1 * my_duration) / windFarms$capacity[f]

					# now scale output and capacity by this
					my_output[unborn] = my_output[unborn] * my_scalar
					my_capacity[unborn] = my_capacity[unborn] * my_scalar
				}

				# now add these to create the evolving fleet
				e_o = e_o + my_output
				e_c = e_c + my_capacity


				# gui
				iii = iii + 1
				if (iii %% 5 == 0)
					clear('Processing', iii, '/', m, '-', split_val)
			}

			# push these into our results storage
			snapshot_output[ , split_val] = s_o
			snapshot_capacity[ , split_val] = s_c

			# push these into our results storage
			evolving_output[ , split_val] = e_o
			evolving_capacity[ , split_val] = e_c
		}

		clear()


		# now normalise to CF
		for (i in colnames(evolving_output)[-1])
		{
			snapshot_cf[ , i] = snapshot_output[ , i] / snapshot_capacity[ , i]
			evolving_cf[ , i] = evolving_output[ , i] / evolving_capacity[ , i]
		}


		# assemble a list to return results
		results = list()
		results$snapshot_capacity = snapshot_capacity
		results$snapshot_output = snapshot_output
		results$snapshot_cf = snapshot_cf
		results$evolving_capacity = evolving_capacity
		results$evolving_output = evolving_output
		results$evolving_cf = evolving_cf
		return(results)
	}

	actuals = read_csv('M:/WORK/Wind Modelling/VALIDATION/Hourly vs TNOs/national_hourly_wind_actuals.csv')
	colnames(actuals)[1] = 'GMT'
	actuals$GMT = ymd_hms(actuals$GMT)

	# timezones
	timeZone = list()
	timeZone$GB = 'Europe/London'
	timeZone$FR = 'Europe/Paris'
	timeZone$IE = 'Europe/Dublin'
	timeZone$ES = 'Europe/Madrid'
	timeZone$FR = 'UTC'
	timeZone$ES = 'UTC'
	timeZone$IE = 'Europe/Dublin'
	timeZone$DE = 'Europe/Berlin'
	timeZone$DK = 'Europe/Copenhagen'


	# what do you want to do today?
	test.timing = TRUE
	test.capacity = FALSE
	quick.validate = FALSE
	full.validate = !quick.validate


	assemble_cfdf = function(actuals, simulation, iso)
	{
		# assemble the actuals and simulation
		act = data.frame(gmt = actuals$GMT,    cf = actuals[ , iso])
		sim = data.frame(gmt = simulation$GMT, cf = simulation[ , iso])

		# push the simulation into the local timezone
		if (timeZone[[iso]] != 'UTC')
		{
			tz = tz_shift(sim$gmt, timeZone[[iso]])
			sim$gmt = sim$gmt + dhours(tz)
		}

		# remove blank actuals
		act = act[ !is.na(act$cf), ]

		# align data
		results = align_data(act, 'gmt', sim, 'gmt')
		cfdf = data.frame(time=results[[1]]$gmt, act=results[[1]]$cf*100, sim=results[[2]]$cf*100, cap=1)

		# make sure simulation and actuals hold the same amount of data
		cfdf = na.everyone(cfdf, c('sim', 'act', 'cap'))


		return(cfdf)
	}


	validation_merit = function(cfdf, iso)
	{
		# RMS OBVIOUSLY
		cf_rms = rms(cfdf$sim - cfdf$act)

		# R2 ALSO
		fit1 = lm(cfdf$sim ~ cfdf$act)
		cf_r2 = summary(fit1)$r.squared

		# RMS OF THE QUANTILES
		qq_act = quantile(cfdf$act, probs=(0:100)/100, na.rm=TRUE)
		qq_sim = quantile(cfdf$sim, probs=(0:100)/100, na.rm=TRUE)
		qq_rms = rms(qq_sim - qq_act)

		# RMS OF THE HISTOGRAM
		hst_act = hist(cfdf$act, breaks=0:100, plot=FALSE)$counts / length(cfdf$act) * 8760
		hst_sim = hist(cfdf$sim, breaks=0:100, plot=FALSE)$counts / length(cfdf$act) * 8760
		hst_rms = rms(hst_sim - hst_act)

		# RMS OF THE TWO POWER SWINGS
		res = power_swings(cfdf, max_swing=80, break_swing=0.5, swing_dur_1=1*2, swing_dur_2=4*2)
		swing_sd = res[[1]]
		ps_rms = rms(swing_sd[c(1,3)] - swing_sd[c(2,4)])

		# SEASONAL SWINGS
		monthly = aggregate_monthly('time', cfdf)
		monthly$month = month(monthly$date)
		monthly$err = monthly$sim - monthly$act
		seasonal = aggregate(monthly$err, by=list(monthly$month), median)

		ssn_summer = mean(seasonal$x[6:8] - seasonal$x[c(1,2,12)])
		ssn_spring = mean(seasonal$x[3:5] - seasonal$x[9:11])

		try(dev.off(), silent=TRUE)
		try(dev.off(), silent=TRUE)

		# WEIGHTING
		merit = c(cf_rms, 50*(1-cf_r2), hst_rms/4, qq_rms, ps_rms, ssn_summer, ssn_spring)
		weights = c(   5,           1,          4,      3,      2,          4,          1)
		merit = weighted.mean(merit, weights)

		stats = list(
			   'iso' = iso,
			   'rms' = cf_rms,
			    'r2' = cf_r2,
			  'hist' = hst_rms,
			 'quant' = qq_rms,
			 'swing_rms' = ps_rms,
			 'swing_1h' = swing_sd[2] - swing_sd[1],
			 'swing_4h' = swing_sd[4] - swing_sd[3],
			'summer' = ssn_summer,
			'spring' = ssn_spring,
			 'total' = merit
		)

		# PRETTY PICTURES
		fn_base = paste(iso, input$alpha, input$beta, input$smooth_base, input$smooth_speed, input$noise, sep='_') %&% '.png'
		fn_path = baseSaveFolder

		bxp = box_plot()
		good_png(fn_path %&% '/box_plot_' %&% fn_base)
		dev.off()

		hist_plot()
		good_png(fn_path %&% '/histogram_' %&% fn_base)
		dev.off()

		return(stats)
	}


