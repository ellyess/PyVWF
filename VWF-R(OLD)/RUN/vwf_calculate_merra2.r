############################################################################
##  
##  Copyright (C) 2012-2017  Iain Staffell  <staffell@gmail.com>
##  
##  Unauthorised copying of this file via any medium is strictly prohibited
##  Proprietary and confidential
##  
############################################################################

	# where does our model config live
	configFile = 'M:/WORK/Wind Modelling/2020.09 - Malte Offshore/vwf_config_merra2.r'

	# get the list of input files we wish to process
	allInputFiles = c(
		'M:/WORK/Wind Modelling/2020.09 - Malte Offshore/Malte_Offshore_Farms_Gamesa14.r',
		'M:/WORK/Wind Modelling/2020.09 - Malte Offshore/Malte_Offshore_Farms_Haliade12.r'
	)


	# cycle through each file in turn
	for (inputFile in allInputFiles)
	{
		cat('\n\n\n', basename(inputFile), '\n\n')

		source('M:/WORK/Wind Modelling/VWF CODE/VWF.R')
		source(configFile)
		
		source('M:/WORK/Wind Modelling/VWF CODE/VWF.MAIN.PREP.R')
		source('M:/WORK/Wind Modelling/VWF CODE/VWF.MAIN.WINDSPEED.R')

		source('M:/WORK/Wind Modelling/VWF CODE/VWF.MAIN.WINDPOWER.R')
		source('M:/WORK/Wind Modelling/VWF CODE/VWF.MAIN.RESULTS.R')

	}

	cat('\n\nFLAWLESS\n\n')
