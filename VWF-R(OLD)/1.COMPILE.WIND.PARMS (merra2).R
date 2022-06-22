 ##
#####
## ##################################################################################################
#####
 ##
 ##  this requires us to have a template NetCDF file with the right variables available
 ##  can't do that within R easily, so need to download NCO from http://nco.sourceforge.net/
 ##
 #### MERRA 1
  # ncks -x -v disph,t10m,t2m,ts,u10m,u50m,v10m,v50m MERRA_IN.nc MERRA_OUT.nc
  # ncrename -h -O -v u2m,A MERRA_OUT.nc
  # ncrename -h -O -v v2m,z MERRA_OUT.nc
  # ncatted -a long_name,A,o,c,"Scale factor for log-law extrapolation: W = A log(h / z)" MERRA_OUT.nc
  # ncatted -a long_name,z,o,c,"Ref height for log-law extrapolation: W = A log(h / z)" MERRA_OUT.nc
  #
  #
 #### MERRA 2
  # ncks -x -v CLDPRS,CLDTMP,DISPH,H1000,H250,H500,H850,OMEGA500,PBLTOP,PS,Q250,Q500,Q850,QV10M,QV2M,SLP,T10M,T250,T2M,T2MDEW,T2MWET,T500,T850,TO3,TOX,TQI,TQL,TQV,TROPPB,TROPPT,TROPPV,TROPQ,TROPT,TS,U10M,U250,U500,U50M,U850,V10M,V250,V500,V50M,V850,ZLCL MERRA_IN.nc4 MERRA_OUT.nc4
  # ncrename -h -O -v U2M,A MERRA_OUT.nc4
  # ncrename -h -O -v V2M,z MERRA_OUT.nc4
  # ncatted -a long_name,A,o,c,"Scale factor for log-law extrapolation: W = A log(h / z)" MERRA_OUT.nc4
  # ncatted -a long_name,z,o,c,"Ref height for log-law extrapolation: W = A log(h / z)" MERRA_OUT.nc4
  #
 ##
 ##
#####
## ##################################################################################################
#####
 ##
 ##  this runs through all merra files in the specified folders  
 ##  reads the wind speed data and calculates the extrapolation parameters
 ##  then saves the A and z values to NetCDF files
 ##
 ##  after running, you should move the resulting files into their own folder
 ##
 ##


#####
## ##  SETUP
#####




#### there are some advances in the merra1 version to copy over here once i know they work


	###  latest MERRA 2

	# the folders we wish to read data from
	merraFolder = 'C:/Users/istaffel/Downloads/wind_to_process/'

	# the folders we wish to write to
	outputFolder = 'C:/Users/istaffel/Downloads/wind_az/'

	# our blank NetCDF template file and the output heights it is to be filled with
	templateFile = 'M:/WORK/Wind Modelling/VWF CODE/TOOLS/MERRA2.log.law.template.A.z.nc4'





#####
## ##  READ IN DATA
#####

	# load the VWF model
	source('M:/WORK/Wind Modelling/VWF CODE/VWF.R')

	# find and prepare all our merra files
	merra = prepare_merra_files(merraFolder)

	# prepare a NetCDF file handler so our format is known
	f = merra$files[1]
	nc = NetCdfClass(f, 'MERRA2', TRUE)

	# subset if necessary
	if (exists('region'))
		nc$subset_coords(region)

	# close this input file
	nc$close_file()







#####
## ##  PREPARE OUR CLUSTER
#####

	# build a parallel cluster
	# note that it doesn't make sense going much beyond 8 cores
	library(doParallel)
	cl = makeCluster(8)
	registerDoParallel(cl)

	# provide the extrapoalte function to each core
	clusterExport(cl, varlist=c('extrapolate_log_law'))







#####
## ##  RUN
#####

	# run through each input merra file
	for (f in merra$files)
	{

		# get this file's time attributes
		nc$open_file(f)
		myTime = ncatt_get(nc$ncdf, "time", "units")$value
		nc$close_file()



		# do the extrapolation, getting the A,z parameters
		profile = extrapolate_ncdf(f)

		# safety checks
		err = which(profile$z > 100)
		profile$z[err] = 100

		err = which(profile$z < 10^-10)
 		profile$z[err] = 10^-10

		err = which(profile$A < 0)
		profile$A[err] = 0


	

		# build a filename for this output data
		o = gsub('prod.assim.tavg1_2d_slv', 'wind_profile', f)
		o = gsub('tavg1_2d_slv_Nx', 'wind_profile', f)
		o = gsub(merraFolder, outputFolder, o)

		# create the NetCDF file for this month
		file.copy(templateFile, o)

		# open this file for writing
		ncout = nc_open(o, write=TRUE)

		# set its time attributes
		ncatt_put(ncout, "time", "units", myTime)

		# put our data in
		for (var in names(profile))
		{
			ncvar_put(ncout, var, profile[[var]])
		}

		# save and close
		nc_close(ncout)
		cat("Written", o, "\n")
	}



	cat("\n\n\nDUN DA DUN DUN DADDA!\n\n")
	