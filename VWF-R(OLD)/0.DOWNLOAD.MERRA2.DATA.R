
# to browse the datasets:
# https://disc.sci.gsfc.nasa.gov/datasets?keywords=%22MERRA-2%22&page=1&source=Models%2FAnalyses%20MERRA-2

# for http download:
# https://goldsmr4.gesdisc.eosdis.nasa.gov/data/MERRA2/M2T1NXSLV.5.12.4/

# this will generate a list of URLs to download a set of variables from a single table for a set of dates
# you can then put these into an olden firefox to download using downthemall... 
# 
# just paste the list of URLS into sublime, wrap around with <a href="url">filename</a><br> 
# and grab using downthemall.  
#
# VARIABLES FOR VWF: DISPH, U2M, V2M, U10M, V10M, U50M, V50M >>> A z
# VARIABLES FOR GSEE: T2M, SWGDN, SWTDN
# VARIABLES FOR WEATHER: CLDTOT, PRECSNOLAND, PRECTOTLAND, QV2M, RHOA, SNOMAS, T2MDEW, T2MWET
#
# these come from:
#
# M2T1NXRAD CLDTOT, SWGDN, SWTDN
# M2T1NXLND PRECTOTLAND, PRECSNOLAND, SNOMAS, TSOIL2, TSOIL5, TSOIL6
# M2T1NXFLX RHOA
# M2T1NXSLV DISPH, U2M, V2M, U10M, V10M, U50M, V50M, QV2M, T2M, T2MDEW, T2MWET 

	library(lubridate)

	MAKE_MERRA2_URL = function(table, date_from, date_to, vars)
	{
		dates = as.character(seq(dmy(date_from), dmy(date_to), by='day'))
		dates0 = gsub('-', '', dates)
		dates2F = gsub('-', '%2F', dates)
		dates2F = substr(dates2F, 1, 9)

		years = substr(dates, 1, 4)
		version = rep('400', length(years))
		version[ years %in% 1979:1991 ] = '100'
		version[ years %in% 1992:2000 ] = '200'
		version[ years %in% 2001:2010 ] = '300'
		version[ dates2F %in% '2020%2F09' ] = '401'

		vars2C = paste(vars, collapse='%2C')

		if (table == 'M2T1NXSLV')
		{
			url1 = 'http://goldsmr4.gesdisc.eosdis.nasa.gov/daac-bin/OTF/HTTP_services.cgi?FILENAME=%2Fdata%2FMERRA2%2FM2T1NXSLV.5.12.4%2F'
			url2 = dates2F
			url3 = '%2FMERRA2_' %&% version %&% '.tavg1_2d_slv_Nx.'
			url4 = dates0
			url5 = '.nc4&FORMAT=bmM0Lw&BBOX=-90%2C-180%2C90%2C180&LABEL=MERRA2_' %&% version %&% '.tavg1_2d_slv_Nx.'
			url6 = dates0
			url7 = '.SUB.nc4&SHORTNAME=M2T1NXSLV&SERVICE=SUBSET_MERRA2&VERSION=1.02&LAYERS=&VARIABLES='
			url8 = vars2C
		}

		if (table == 'M2T1NXRAD')
		{
			url1 = 'http://goldsmr4.gesdisc.eosdis.nasa.gov/daac-bin/OTF/HTTP_services.cgi?FILENAME=%2Fdata%2FMERRA2%2FM2T1NXRAD.5.12.4%2F'
			url2 = dates2F
			url3 = '%2FMERRA2_' %&% version %&% '.tavg1_2d_rad_Nx.'
			url4 = dates0
			url5 = '.nc4&FORMAT=bmM0Lw&BBOX=-90%2C-180%2C90%2C180&LABEL=MERRA2_' %&% version %&% '.tavg1_2d_rad_Nx.'
			url6 = dates0
			url7 = '.SUB.nc4&SHORTNAME=M2T1NXRAD&SERVICE=SUBSET_MERRA2&VERSION=1.02&LAYERS=&VARIABLES='
			url8 = vars2C
		}

		if (table == 'M2T1NXFLX')
		{
			url1 = 'http://goldsmr4.gesdisc.eosdis.nasa.gov/daac-bin/OTF/HTTP_services.cgi?FILENAME=%2Fdata%2FMERRA2%2FM2T1NXFLX.5.12.4%2F'
			url2 = dates2F
			url3 = '%2FMERRA2_' %&% version %&% '.tavg1_2d_flx_Nx.'
			url4 = dates0
			url5 = '.nc4&FORMAT=bmM0Lw&BBOX=-90%2C-180%2C90%2C180&LABEL=MERRA2_' %&% version %&% '.tavg1_2d_flx_Nx.'
			url6 = dates0
			url7 = '.SUB.nc4&SHORTNAME=M2T1NXFLX&SERVICE=SUBSET_MERRA2&VERSION=1.02&LAYERS=&VARIABLES='
			url8 = vars2C
		}

		if (table == 'M2T1NXLND')
		{
			url1 = 'http://goldsmr4.gesdisc.eosdis.nasa.gov/daac-bin/OTF/HTTP_services.cgi?FILENAME=%2Fdata%2FMERRA2%2FM2T1NXLND.5.12.4%2F'
			url2 = dates2F
			url3 = '%2FMERRA2_' %&% version %&% '.tavg1_2d_lnd_Nx.'
			url4 = dates0
			url5 = '.nc4&FORMAT=bmM0Lw&BBOX=-90%2C-180%2C90%2C180&LABEL=MERRA2_' %&% version %&% '.tavg1_2d_lnd_Nx.'
			url6 = dates0
			url7 = '.SUB.nc4&SHORTNAME=M2T1NXLND&SERVICE=SUBSET_MERRA2&VERSION=1.02&LAYERS=&VARIABLES='
			url8 = vars2C
		}

		paste0(url1, url2, url3, url4, url5, url6, url7, url8)
	}


	# recent weather grabs
	date_from = '01-01-2020'
	date_to = '31-12-2020'

	urls = MAKE_MERRA2_URL('M2T1NXSLV', date_from, date_to, 'T2M')
	wc(urls)

	urls = MAKE_MERRA2_URL('M2T1NXRAD', date_from, date_to, 'SWGDN')
	wc(urls)

	urls = MAKE_MERRA2_URL('M2T1NXSLV', date_from, date_to, 'QV2M')
	wc(urls)

	urls = MAKE_MERRA2_URL('M2T1NXSLV', date_from, date_to, c('U10M','V10M'))
	wc(urls)





	# full ninja grab
	date_from = '01-01-2020'
	date_to = '31-12-2020'


	# wind speeds (all vars together for converting to a/z)
	table = 'M2T1NXSLV'
	vars = c('DISPH', 'U2M', 'V2M', 'U10M', 'V10M', 'U50M', 'V50M') 
	urls = MAKE_MERRA2_URL(table, date_from, date_to, vars)
	wc(urls)


	# weather data (each var in a single file)
	urls = list()

	urls[[1]] = MAKE_MERRA2_URL('M2T1NXRAD', date_from, date_to, 'CLDTOT')
	urls[[2]] = MAKE_MERRA2_URL('M2T1NXRAD', date_from, date_to, 'SWGDN')
	urls[[3]] = MAKE_MERRA2_URL('M2T1NXRAD', date_from, date_to, 'SWTDN')

	urls[[4]] = MAKE_MERRA2_URL('M2T1NXLND', date_from, date_to, 'PRECTOTLAND')
	urls[[5]] = MAKE_MERRA2_URL('M2T1NXLND', date_from, date_to, 'PRECSNOLAND')
	urls[[6]] = MAKE_MERRA2_URL('M2T1NXLND', date_from, date_to, 'SNOMAS')

	urls[[7]] = MAKE_MERRA2_URL('M2T1NXFLX', date_from, date_to, 'RHOA')

	urls[[8]] = MAKE_MERRA2_URL('M2T1NXSLV', date_from, date_to, 'QV2M')
	urls[[9]] = MAKE_MERRA2_URL('M2T1NXSLV', date_from, date_to, 'T2M')
	urls[[10]] = MAKE_MERRA2_URL('M2T1NXSLV', date_from, date_to, 'T2MDEW')

	wc(unlist(urls))




	# extra stuff for heat pumps
	table = 'M2T1NXLND'
	vars = c('TSOIL2', 'TSOIL5', 'TSOIL6')
	urls[[11]] = MAKE_MERRA2_URL(table, date_from, date_to, vars)




	# how to mash these into mega annual files!
	#
	# fucking give up with these autists and install WSL (step 1 and 6 of https://docs.microsoft.com/en-us/windows/wsl/install-win10)
	#         then install nco (sudo apt-get update // sudo apt-get install nco // sudo apt-get install netcdf-bin)
	 
	# 1) rename your downloads to a simple numerical series: e.g. T2M<#001> (note, extension must be .nc)
	# 2) navigate to your downloads cd /mnt/c/nc/
	# 2) ncrcat -4 -n 365,3,1 T2M001.nc GRAVY.nc4
	# 3) nccopy -u -m 6500M -c time/880,latitude/32,longitude/32 GRAVY.nc4 GRAVY_CHUNKS.nc4