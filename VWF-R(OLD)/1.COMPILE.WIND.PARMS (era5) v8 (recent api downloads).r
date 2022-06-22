
	source('M:/WORK/Code/R/__inc.r')
	source_progress()
	source_netcdf()
	library(ncdf4.helpers)
	library(doSNOW)



	#####
	## ##  DEFINE YOUR INPUTS
	#####

	# folders to process
	folder_u10 = 'M:\\ERA5ANALYSIS\\u10m\\'
	folder_v10 = 'M:\\ERA5ANALYSIS\\v10m\\'
	folder_u100 = 'M:\\ERA5ANALYSIS\\u100m\\'
	folder_v100 = 'M:\\ERA5ANALYSIS\\v100m\\'

	# filenames to write to
	final_path = function(my_date) paste0('M:\\ERA5ANALYSIS\\ninja_Az\\era5-ninja-wind-Az-', my_date, '.nc')

	# other places we might have written files to (so you don't duplicate ones that were built elsewhere)
	potential_paths = final_path

	# how many cores to process A/z values - really shouldn't be many as the single-thread read and write is the slowest bit
	n_cores = 3

	# so also run this many versions in parallel - also not too many as the disk I/O destroys life
	n_instances = 1
	instance = 1




	#####
	## ##  SETUP
	#####

	# locate all our files
	files_u10 = list.files(folder_u10, full.names=TRUE, pattern='.nc$')
	files_v10 = gsub('u10', 'v10', files_u10, fixed=TRUE)
	files_u100 = gsub('u10', 'u100', files_u10, fixed=TRUE)
	files_v100 = gsub('u10', 'v100', files_u10, fixed=TRUE)

	if (any(!file.exists(files_v10))) stop('Some files are missing from folder_v10')
	if (any(!file.exists(files_u100))) stop('Some files are missing from folder_u100')
	if (any(!file.exists(files_v100))) stop('Some files are missing from folder_v100')

	# split into parallel instances
	if (n_instances > 1)
	{
		files_u10 = chunk(files_u10, groups=n_instances)[[instance]]
		files_v10 = chunk(files_v10, groups=n_instances)[[instance]]
		files_u100 = chunk(files_u100, groups=n_instances)[[instance]]
		files_v100 = chunk(files_v100, groups=n_instances)[[instance]]
	}


	F = length(files_u10)


	flush('\n\nBeginning to process', F, 'files of ERA-5 data at', as.character(Sys.time()), '\n')



	# build a snow cluster
	clear('Building cluster')
	cl = makeCluster(n_cores)
	registerDoSNOW(cl)

	# establish our progress bar
	PB = blam.progress(F)
	PB$pct_dp = 3
	PB$i_dp = 3
	# we assume the parallel stuff takes 50% of the time, the NetCDF writing takes the other 50%
	PB$parallel_tick = function(f, F, n, N) { PB$total( f - 1 + n/N/2, 'Calculating A/z: ' %&% n %&% ' / ' %&% N ) }


	# function to calculate A/z
	extrapolate_era5 = function(w10m, w100m)
	{
		# slope = (w100m - w10m) / (log(100) - log(10))
		A = (w100m - w10m) / 2.30258509299405

		# A = w10m - (log(10) * slope)
		intercept = w10m - (2.30258509299405 * A)

		z = exp(-intercept / A)

		return( c(A, z) )
	}





	#####
	## ##  PROCESS ALL FILES
	#####

	for (f in 1:F)
	{
		# sort out filenames
		my_date = get_text_before(basename(files_u10[f]), '.')
		output_file = final_path(my_date)

		if (any(file.exists(potential_paths(my_date))))
		{
			PB$skip(1)
			clear('Skipping', basename(output_file), 'as it already exists...\n')
			next
		}


		# get our dimensions
		nc = nc_open(files_u10[f])
		lon = nc_lon(nc)
		lat = nc_lat(nc)
		tim = nc_time(nc)
		nc_close(nc)


		# register our progress and combining functions
		N = length(tim)
		snow.prog = list(progress = function(n) PB$parallel_tick(f, F, n, N) )
		


		#####
		## ##  CALCULATE
		#####

		# run through each timestep in parallel
		profile = foreach(t=1:length(tim), .options.snow=snow.prog, .packages='ncdf4') %dopar%
		{
			# get the wind at 10 metres
			nc = nc_open(files_u10[f])
			U10M = ncvar_get(nc, 'u10', start=c(1,1,t), count=c(-1,-1,1))
			nc_close(nc)

			nc = nc_open(files_v10[f])
			V10M = ncvar_get(nc, 'v10', start=c(1,1,t), count=c(-1,-1,1))
			nc_close(nc)

			W10M = sqrt(U10M^2 + V10M^2)
			rm(U10M, V10M); quiet=gc()


			# get the wind at 100 metres
			nc = nc_open(files_u100[f])
			U100M = ncvar_get(nc, 'u100', start=c(1,1,t), count=c(-1,-1,1))
			nc_close(nc)

			nc = nc_open(files_v100[f])
			V100M = ncvar_get(nc, 'v100', start=c(1,1,t), count=c(-1,-1,1))
			nc_close(nc)

			W100M = sqrt(U100M^2 + V100M^2)
			rm(U100M, V100M); quiet=gc()


			# pre-allocate space for A and z
			A = z = W10M * NA

			# calculate A and log(z) for all locations
			for (i in 1:length(lon))
			{
				Az = mapply(FUN=extrapolate_era5, W10M[i, ], W100M[i, ])
				A[i, ] = Az[1, ]
				z[i, ] = log(Az[2, ])
			}

			# filter z to sensible values
			z[ z < log(1e-10) ] = log(1e-10)
			z[ z > log(1e+10) ] = log(1e+10)

			# truncate and bind into a list
			A = signif(A, 5)
			z = signif(z, 5)
			
			list(A=A, z=z)
		}


		# strip out A & z
		dims = c(dim(profile[[1]]$A), length(profile))
		A = z = array(NA, dims)

		for (t in 1:length(profile))
		{
			A[ , , t] = profile[[t]]$A
			z[ , , t] = profile[[t]]$z
		}

		rm(profile); gc()




		#####
		## ##  WRITE TO NETCDF
		#####

#		# rearrange this stupid data so that lat is ascending
#		lat_order = rev(seq_along(lat))

#		# rearrange this stupid data to be london centred
#		east = (lon > 180)
#		lon[east] = lon[east] - 360
#		lon_order = order(lon)

#		A = A[lon_order, lat_order, ]
#		z = z[lon_order, lat_order, ]

		# copy the dimensions from our input file
		x_dim = nc$dim$lon
		y_dim = nc$dim$lat
		t_tim = nc$dim$time

#		# rearrange these stupid dimensions
#		y_dim$vals = y_dim$vals[lat_order]

#		east = (x_dim$vals > 180)
#		x_dim$vals[east] = x_dim$vals[east] - 360
#		x_dim$vals = sort(x_dim$vals)

		# build netcdf containers for our data
		nc_A = ncvar_def('A', units='', longname='Scale factor for log-law extrapolation: W = A log(h / exp(z))',
			dim=list(x_dim,y_dim,t_tim), missval=1.e30, compression=4, chunksizes=c(1,1,length(tim)))

		nc_z = ncvar_def('z', units='', longname='Height for log-law extrapolation: W = A log(h / exp(z))',
			dim=list(x_dim,y_dim,t_tim), missval=1.e30, compression=4, chunksizes=c(1,1,length(tim)))

		# build a new netcdf file to populate
		#nc_close(nc)

		PB$tick(0, paste('Writing', output_file))

		nc = nc_create(output_file, list(nc_A, nc_z), force_v4=TRUE)

		PB$tick(0.5/3, paste('Writing', output_file))

		ncvar_put(nc, 'A', A)
		rm(A); quiet=gc()

		PB$tick(0.5/3, paste('Writing', output_file))

		ncvar_put(nc, 'z', z)
		rm(z); quiet=gc()

		PB$tick(0.5/3, paste('Writing', output_file, '\n'))

		nc_close(nc)

	}


	# tidy up everything
	flush('\nFinished at', as.character(Sys.time()), '\n\n')
	stopCluster(cl)






	# test what we made
	if (0)
	{

		files = c(
			'I:/ninja_wind_Az_2015-01.nc'
		)

		for (f in files)
		{
			nc = nc_open(f)
			lon = nc_lon(nc)
			lat = nc_lat(nc)
			time = nc.get.time.series(nc)

			A = ncvar_get(nc, 'A', c(1,1,1), c(-1,-1,1))
			z = ncvar_get(nc, 'z', c(1,1,1), c(-1,-1,1))

			W = A * log(123 / exp(z))

			nc_plot_gridded(lon, lat, W, col=blues9, zRange=c(0,25), nLevels=25)
		}

		W = NULL

		for (f in files)
		{
			clear(f)

			nc = nc_open(f)
			lon = nc_lon(nc)
			lat = nc_lat(nc)
			time = nc.get.time.series(nc)

			A = ncvar_get(nc, 'A')
			z = ncvar_get(nc, 'z')

			w = A * log(123 / exp(z))

			if (is.null(W)) {

				W = w

			} else {

				W = abind(W, w, along=3)
			}

		}

		clear()

	}
