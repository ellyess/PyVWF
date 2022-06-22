############################################################################
#  
#  Copyright (C) 2012-2017  Iain Staffell  <staffell@gmail.com>
#  
#  Unauthorised copying of this file via any medium is strictly prohibited
#  Proprietary and confidential
#  
############################################################################


	# external dependencies (that are always needed)
	suppressPackageStartupMessages(library(ncdf4))
	suppressPackageStartupMessages(library(fields))
	suppressPackageStartupMessages(library(lubridate))
	#suppressPackageStartupMessages(library(rworldmap))
	suppressPackageStartupMessages(library(doParallel))
	suppressPackageStartupMessages(library(data.table))
	suppressPackageStartupMessages(library(akima))
	suppressPackageStartupMessages(library(plyr))

	# external dependencies (not needed by all things, so not loaded by default)
	#library(MASS)
	#library(fields)




	# figure out the working directory for this VWF code
	if (!is.null(parent.frame(2)$ofile))
	{
		vwf.dir = dirname(parent.frame(2)$ofile) %&% '/'

	# fallback default
	} else {
		vwf.dir = 'M:/WORK/Wind Modelling/VWF CODE/'
	}
	


	# load iain's standard library
	# source(vwf.dir %&% 'VWF.STDLIB.R')

	# load the background code
	source(vwf.dir %&% 'VWF.NCDF.R')
	source(vwf.dir %&% 'VWF.EXTRAPOLATE.R')
	source(vwf.dir %&% 'VWF.FARMS.R')
	source(vwf.dir %&% 'VWF.PLOTS.R')
	source(vwf.dir %&% 'VWF.EXTRAS.R')

	flush('Welcome to the VWF model v19.03.03\n')
