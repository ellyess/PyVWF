############################################################################
#  
#  Copyright (C) 2012-2017  Iain Staffell  <staffell@gmail.com>
#  
#  Unauthorised copying of this file via any medium is strictly prohibited
#  Proprietary and confidential
#  
############################################################################



#####
## ##  READ IN EXISTING SPEEDS IF WE HAVE THEM
#####

# VWF.PREP already tested whether our wind speed file exists...
use_cached_speeds = !is.null(windSpeedFile)

if (use_cached_speeds)
{
	flush('> Reading pre-compiled wind speeds...\n')
	windSpeed = read_data(windSpeedFile)
}





#####
## ##  RUN THROUGH EACH GROUP (MONTH / QUARTER) OF MERRA FILES, READ IN THE WIND SPEED FOR EACH WINDFARM & SAVE TO A TEMP FILE
#####

if (!use_cached_speeds)
{
	flush('> Generating wind speeds... temp files being written to', baseSaveFolder, '\n')

	windSpeed = foreach(g=seq_along(merra_wind$files), .combine=rbind, .packages=c('ncdf4', 'lubridate')) %dopar%
	{

		##
		##  READ IN THE NETCDF DATA
		##

		# prepare the filename and date
		fn = merra_wind$files[g]
		nc_wind$open_file(fn)

		# make a vector of our timestamps
		dt = merra_wind$date[g]
		timeVector = dt + hours(nc_wind$hour)

		# that seems not to work with our ERA5 files... 
		# so may the grotesque dogshite hacking commence...
		if (reanalysis %in% c('ERA5', 'ERA5NEW'))
		{
			# every element of this line of code speaks volumes about me
			# the mixed use of camel case and underscores
			# the use of a function sloppily pasted over from VWF.NETCDF.R
			# on a class from VWF.NCDF.R that ought to have been deprecated
			# back in 2015 before it was bloody well born...
			timeVector = nc_time(nc_wind$ncdf)
		}
	
		# extract our extrapolation parameters - format is A[lon, lat, time]
		A = nc_wind$get_var('A')
		z = nc_wind$get_var('z')
	
		# close this input file
		nc_wind$close_file()


		# the ERA5 format isn't perfected just yet :-S
		if (nc_wind$model %in% c('ERA5', 'ERA5NEW'))
		{
			# convert from log(z) to z
			z = exp(z)
		}



		# if we are crossing the date line, reshape our data to contain what we need
		if (dateLineMadness)
		{
			A = A[ c(dateLineWest, dateLineEast), , ]
			z = z[ c(dateLineWest, dateLineEast), , ]
			dimnames(A)[[1]] = dateLineDims
			dimnames(z)[[1]] = dateLineDims
			nc_wind$lon = dateLineDims
		}




		##
		##  SPATIAL INTERPOLATION AND HEIGHT EXTRAPOLATION
		##

		# prepare our results storage
		# this holds the estimated wind speeds, rows=hours, columns=farms
		speed = matrix(NA, nrow=length(timeVector), ncol=length(windFarms$name))


		# interpolate each hour of the day
		for (h in 1:nrow(speed))
		{
			# initialise the wind speeds for all farms
			s = speed[h, ]

			# run through each height in turn
			all_heights = sort(unique(windFarms$height))
			for (height in all_heights)
			{				
				# get wind speeds at this height
				w = A[ , , h] * log(height / z[ , , h])

				# locate the farms we care about
				my_farms = (windFarms$height == height)

				# loess interpolation
				if (spatial.method == 'loess')
				{
					# create a loess fit
					loess_w$value = as.vector(w)
					loess_fit = loess(value ~ lon*lat, data=loess_w, degree=2, span=loessSpan)

					# interpolate speeds to these farms
					s[my_farms] = predict(loess_fit, newdata=windFarms[my_farms, ])
				}

				# spline interpolation
				if (spatial.method == 'akima')
				{
					# cubic spline our speeds
					s[my_farms] = bicubic(nc_wind$lon, nc_wind$lat, w, windFarms$lon[my_farms], windFarms$lat[my_farms])$z
				}

			}

			# round to 5 d.p. to save time & space
			s = round(s, 5)

			# slot these speeds back in - forbidding silliness like negative speeds
			speed[h, ] = pmax(0, s)
		}

		clear()



		##
		##  SAVE THE WIND SPEEDS
		##

		# turn that raw matrix into a nice dataframe
		speed = data.frame(speed)
		speed = cbind(timeVector, speed)
		colnames(speed) = c("Timestamp", windFarms$name)


		# save the wind speeds if requested
		if (save_original_wind_speeds)
		{
			fn = baseSaveFolder %&% baseSaveFile %&% 'windspeed.' %&% merra_wind$names[g] %&% '.' %&% xtn
			write_data(speed, fn, xtn)
			flush('\n ~ Written', fn)
		}


		# save the annual average speeds if requested
		if (save_annual_wind_speeds)
		{
			fn = baseSaveFolder %&% baseSaveFile %&% 'annualspeed.' %&% merra_wind$names[g] %&% '.csv'

			ann_y = 'Y' %&% merra_wind$names[g]
			ann_f = colnames(speed[ , -1])
			ann_s = colMeans(speed[ , -1])
			z = data.frame(farm=ann_f, speed=ann_s)
			colnames(z)[2] = ann_y 
			
			write_data(z, fn, 'csv')
			flush('\n ~ Written', fn)
		}


		# if we are crossing the date line, repair the damage we did
		if (dateLineMadness)
		{
			nc_wind$lon = originalDims
		}

	} # finished processing all files - end of parallel loop







	###
	###  CLEAR OUT TEMPORARY FILES
	###  AND BIND ALL WIND SPEEDS TOGETHER
	###

	if (save_original_wind_speeds & join_wind_speed_files)
	{
		
		# concatenate all the results files
		fn = baseSaveFolder %&% baseSaveFile %&% 'windspeed.' %&% merra_wind$names %&% '.' %&% xtn
		windSpeed = read_data_bind_rows(fn, verbose=FALSE)

		# and save
		big_fn = baseSaveFolder %&% baseSaveFile %&% 'windspeed.' %&% xtn
		flush('~ Writing', big_fn, '\n')
		write_data(windSpeed, big_fn, xtn)

		# and remove the intermediates
		file.remove(fn)

	}

	if (save_annual_wind_speeds & join_wind_speed_files)
	{
		
		# concatenate all the results files
		fn = baseSaveFolder %&% baseSaveFile %&% 'annualspeed.' %&% merra_wind$names %&% '.csv'
		windSpeed = read_data_bind_cols(fn, verbose=FALSE)

		# get rid of junk date columns
		kill_cols = which(colnames(windSpeed) == 'farm')[-1]
		windSpeed = windSpeed[ , -kill_cols]

		# and save
		big_fn = baseSaveFolder %&% baseSaveFile %&% 'annualspeed.csv'
		flush('~ Writing', big_fn, '\n')
		write_data(windSpeed, big_fn, 'csv')

		# and remove the intermediates
		file.remove(fn)

	}


}




#####
## ##  FINISH OFF
#####

	# kill the plot of wind farm locations
	if (a_few_plots) try( dev.off(), silent=TRUE )
	flush(' ', nrow(windSpeed), 'hours of wind speeds calculated for', ncol(windSpeed)-1, 'farms\n')
