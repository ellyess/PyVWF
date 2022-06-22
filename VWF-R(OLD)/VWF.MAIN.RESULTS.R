############################################################################
#  
#  Copyright (C) 2012-2017  Iain Staffell  <staffell@gmail.com>
#  
#  Unauthorised copying of this file via any medium is strictly prohibited
#  Proprietary and confidential
#  
############################################################################


	# r is filthy and shitty and slow with data.frames
	# and we want to do lots of rounding on them to save space
	# so, break each frame into chunks and work on those...
	# this is quicker than apply() which is a crappy for-loop wrapper
	quick_sig_figs = function(data, sigfigs)
	{
		# don't bother
		if (ncol(data) < 10)
		{
			return( signif(data, sigfigs) )
		}

		# identify equal-sized chunks of data
		chunks = chunk(1:ncol(data), sqrt(ncol(data)))

		for (cols in chunks)
		{
			# break out this chunk
			extract = data[ , cols]

			# apply the rounding
			for (col in 1:ncol(extract))
				if (is.numeric(extract[ , col]))
					extract[ , col] = signif(extract[ , col], sigfigs)
			
			# re-assemble this chunk
			data[ , cols] = extract
		}

		data
	}


#####
## ##  SAVE HOURLY FILES
#####

	flush('> Saving results...\n')



	# save the bias corrected wind speeds for each farm
	if (save_modified_wind_speeds)
	{
		fn = baseSaveFolder %&% baseSaveFile %&% 'windspeed.corrected.' %&% xtn
		windSpeed = quick_sig_figs(windSpeed, 5)
		write_data(windSpeed, fn, xtn)
		flush('  Written', fn, '\n')
	}



	# save the time-averaged outputs for every farm
	if (save_farms_monthly)
	{
		farms_cf = aggregate_ts_faster(datecol, loadFactor, by='monthly', mean)
		fn = baseSaveFolder %&% baseSaveFile %&% '(a).farm.CF.monthly.csv'
		write_csv(farms_cf, fn, row.names=FALSE)
		clear('  Written', fn)
	}

	if (save_farms_annualy)
	{
		farms_cf = aggregate_ts_faster(datecol, loadFactor, by='yearly', mean)
		fn = baseSaveFolder %&% baseSaveFile %&% '(a).farm.CF.yearly.csv'
		write_csv(farms_cf, fn, row.names=FALSE)
		clear('  Written', fn)
	}




	# save the total aggregate across all farms
	if ('*' %in% save_files_split_by)
	{
		# hourly
		if (ncol(powerMW > 1)) 
		{
			total = rowSums(powerMW, na.rm=TRUE)

		} else {

			total = powerMW
		}

		total = data.frame(GMT=ymd_wtf(datecol), MW=total, CF=total/sum(windFarms$capacity))

		if (save_split_snapshot)
		{
			# save the hourly results
			fn = baseSaveFolder %&% baseSaveFile %&% '(b).hourly.MW.CF.csv'
			write_csv(total, fn, row.names=FALSE)
			clear('  Written', fn)

			# save the monthly results
			if (save_split_monthly)
			{
				total_m = aggregate_ts_faster('GMT', total, by='monthly', mean)
				fn = baseSaveFolder %&% baseSaveFile %&% '(b).monthly.MW.CF.csv'
				write_csv(total_m, fn, row.names=FALSE)
				clear('  Written', fn)
			}

			# save the annual results
			if (save_split_annualy)
			{
				total_y = aggregate_ts_faster('GMT', total, by='yearly', mean)
				fn = baseSaveFolder %&% baseSaveFile %&% '(b).yearly.MW.CF.csv'
				write_csv(total_y, fn, row.names=FALSE)
				clear('  Written', fn)
			}
		}



		if (save_split_evolving)
		{
###########################################################################################################
			# establish results storage
			n = length(datecol)
			m = nrow(windFarms)

			# understand what time period this is
			min_date = min(total$GMT)
			max_date = max(total$GMT)
			total_duration = (max_date - min_date) / dyears(1)

			e_c = e_o = rep(0, n)

			# calculate the total output and capacity
			for (f in 1:nrow(windFarms))
			{
				my_output = powerMW[ , f]
				my_capacity = rep(windFarms$capacity[f], n)
				my_start = dmy(windFarms$date[f])

				# if we have a start date, then wipe out the period before birth
				if (!is.na(my_start))
				{
					unborn = (total$GMT < my_start)

					# what percentage of the whole period have you lived for?
					# take that to be your capacity - so that the oldest farms have
					# the greatest weight when we go so far back that we know nothing
					my_duration = (as.Date(max_date) - as.Date(my_start)) / dyears(1)
					my_duration = pmax(0.01, my_duration / total_duration)
					my_scalar = (0.1 * my_duration) / windFarms$capacity[f]

					# now scale output and capacity by this
					my_output[unborn] = my_output[unborn] * my_scalar
					my_capacity[unborn] = my_capacity[unborn] * my_scalar
				}

				# now add these to create the evolving fleet
				e_o = e_o + my_output
				e_c = e_c + my_capacity
			}


			# assemble the overall results
			total = data.frame(GMT=ymd_wtf(datecol), MW=e_o, CF=e_o/e_c, CAP=e_c)

			# save the hourly results
			fn = baseSaveFolder %&% baseSaveFile %&% '(b).hourly.MW.CF.evolving.csv'
			write_csv(total, fn, row.names=FALSE)
			clear('  Written', fn)

			# save the monthly results
			if (save_split_monthly)
			{
				total_m = aggregate_ts_faster('GMT', total, by='monthly', mean)
				fn = baseSaveFolder %&% baseSaveFile %&% '(b).monthly.MW.CF.evolving.csv'
				write_csv(total_m, fn, row.names=FALSE)
				clear('  Written', fn)
			}

			# save the annual results
			if (save_split_annualy)
			{
				total_y = aggregate_ts_faster('GMT', total, by='yearly', mean)
				fn = baseSaveFolder %&% baseSaveFile %&% '(b).yearly.MW.CF.evolving.csv'
				write_csv(total_y, fn, row.names=FALSE)
				clear('  Written', fn)
			}


###########################################################################################################
		}
	}




	# save the total MW and average CF grouped by various things (e.g. by country)
	for (split_col in save_files_split_by)
	{
		# skip doing nothing or everything or something silly
		if (split_col == '') next
		if (split_col == '*') next
		if (split_col %notin% colnames(windFarms))
		{
			flush('  !! you want to split results by ' %&% split_col %&% ' which isnt a column in your wind farms file...\n')
			flush('  !! you need to look at `save_files_split_by` and `windFarmCols`...\n')
			next
		}


		# establish results storage
		n = length(datecol)
		m = nrow(windFarms)
		snapshot_capacity = snapshot_output = data.frame(GMT=ymd_wtf(datecol))
		evolving_capacity = evolving_output = data.frame(GMT=ymd_wtf(datecol))

		# understand what time period this is
		min_date = min(evolving_output$GMT)
		max_date = max(evolving_output$GMT)
		total_duration = (max_date - min_date) / dyears(1)

		# go through each value in that column
		for (split_val in sort(unique(windFarms[ , split_col])))
		{
			s_c = s_o = rep(0, n)
			e_c = e_o = rep(0, n)

			# filter our farms - filter gives the farm row / speed column
			filter = which(windFarms[ , split_col] == split_val)
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
					my_duration = (as.Date(max_date) - as.Date(my_start)) / dyears(1)
					my_duration = pmax(0.01, my_duration / total_duration)
					my_scalar = (0.1 * my_duration) / windFarms$capacity[f]

					# now scale output and capacity by this
					my_output[unborn] = my_output[unborn] * my_scalar
					my_capacity[unborn] = my_capacity[unborn] * my_scalar
				}

				# now add these to create the evolving fleet
				e_o = e_o + my_output
				e_c = e_c + my_capacity
			}

			# push these into our results storage
			snapshot_output[ , split_val] = s_o
			snapshot_capacity[ , split_val] = s_c

			# push these into our results storage
			evolving_output[ , split_val] = e_o
			evolving_capacity[ , split_val] = e_c
		}


		if (save_split_snapshot)
		{
			# save power output to file
			fn = baseSaveFolder %&% baseSaveFile %&% '(c).snapshot.MW.' %&% split_col %&% '.csv'
			write_csv(snapshot_output, fn, row.names=FALSE)
			clear('  Written', fn)

			# now normalise to CF
			for (i in 2:ncol(snapshot_output))
			{
				snapshot_output[ , i] = snapshot_output[ , i] / snapshot_capacity[ , i]
			}

			# save the hourly results
			fn = baseSaveFolder %&% baseSaveFile %&% '(c).snapshot.CF.' %&% split_col %&% '.csv'
			write_csv(snapshot_output, fn, row.names=FALSE)
			clear('  Written', fn)

			# save the monthly results
			if (save_split_monthly)
			{
				snapshot_cf = aggregate_ts_faster('GMT', snapshot_output, by='monthly', mean)
				fn = baseSaveFolder %&% baseSaveFile %&% '(c).snapshot.CF.monthly.' %&% split_col %&% '.csv'
				write_csv(snapshot_cf, fn, row.names=FALSE)
				clear('  Written', fn)
			}

			# save the annual results
			if (save_split_annualy)
			{
				snapshot_cf = aggregate_ts_faster('GMT', snapshot_output, by='yearly', mean)
				fn = baseSaveFolder %&% baseSaveFile %&% '(c).snapshot.CF.yearly.' %&% split_col %&% '.csv'
				write_csv(snapshot_cf, fn, row.names=FALSE)
				clear('  Written', fn)
			}
		}



		if (save_split_evolving)
		{
			# save power output to file
			fn = baseSaveFolder %&% baseSaveFile %&% '(d).evolving.MW.' %&% split_col %&% '.csv'
			write_csv(evolving_output, fn, row.names=FALSE)
			clear('  Written', fn)

			# now normalise to CF
			for (i in 2:ncol(evolving_output))
			{
				evolving_output[ , i] = evolving_output[ , i] / evolving_capacity[ , i]
			}

			# save the hourly capacity
			fn = baseSaveFolder %&% baseSaveFile %&% '(d).evolving.capacity.' %&% split_col %&% '.csv'
			write_csv(evolving_capacity, fn, row.names=FALSE)
			clear('  Written', fn)

			# save the hourly results
			fn = baseSaveFolder %&% baseSaveFile %&% '(d).evolving.CF.' %&% split_col %&% '.csv'
			write_csv(evolving_output, fn, row.names=FALSE)
			clear('  Written', fn)

			# save the monthly results
			if (save_split_monthly)
			{
				evolving_cf = aggregate_ts_faster('GMT', evolving_output, by='monthly', mean)
				fn = baseSaveFolder %&% baseSaveFile %&% '(d).evolving.CF.monthly.' %&% split_col %&% '.csv'
				write_csv(evolving_cf, fn, row.names=FALSE)
				clear('  Written', fn)
			}

			# save the annual results
			if (save_split_annualy)
			{
				evolving_cf = aggregate_ts_faster('GMT', evolving_output, by='yearly', mean)
				fn = baseSaveFolder %&% baseSaveFile %&% '(d).evolving.CF.yearly.' %&% split_col %&% '.csv'
				write_csv(evolving_cf, fn, row.names=FALSE)
				clear('  Written', fn)
			}

		}
	}

	flush('\n')


	# save the model parameters
	if (save_model_parameters)
	{
		fn = baseSaveFolder %&% baseSaveFile %&% '(e).parameters.csv'
		write_csv(parms, fn)
		flush('  Written', fn)
	}




	# save the hourly power outputs for every farm
	if (save_hourly_farm_mw)
	{
		fn = baseSaveFolder %&% baseSaveFile %&% '(a).farm.MW.' %&% xtn
		powerMW = quick_sig_figs(powerMW, 6)
		write_data(powerMW, fn, xtn)
		flush('  Written', fn, '\n')
	}

	# save the hourly capacity factors for every farm
	if (save_hourly_farm_cf)
	{
		fn = baseSaveFolder %&% baseSaveFile %&% '(a).farm.CF.' %&% xtn
		loadFactor = quick_sig_figs(loadFactor, 6)
		write_data(loadFactor, fn, xtn)
		flush('  Written', fn, '\n')
	}
