############################################################################
#  
#  Copyright (C) 2012-2016  Iain Staffell  <staffell@gmail.com>
#  
#  Unauthorised copying of this file via any medium is strictly prohibited
#  Proprietary and confidential
#  
############################################################################



#####
## ##  AN EXTENDED NETCDF4 LIBRARY
#####
#
#	# open
#	nc = nc_open(filename)
#   x = nc_describe(nc)
#
#	# describe
#	nc_open(filename)
#
#	# extract dimensions
#	lat = nc_lat(nc)
#	lon = nc_lon(nc)
#	time = nc_time(nc)
#
#   # subset data - region is list(lon=c(min,max), lat=c(min,max))
#   s = nc_subset(nc, region)
#
#	# extract data
#	d = ncvar_get(nc, varname)
#   d = ncvar_get(nc, varname, s$start2d, s$count2d)
#   d = ncvar_get(nc, varname, s$start3d, s$count3d)
#
#	# plot
#	nc_plot(lon, lat, d)
#   nc_plot(s$lon, s$lat, d)
#
#   nc_plot_gridded(lon, lat, d)
#   nc_plot_gridded(s$lon, s$lat, d)


#### TODO UPGRADE MANS PLOTTING TO USE IDEAL_ASPECT_RATIO
#### TODO FUCK ALL MY TIMESTAMP BOLLOCKS, USE nc.get.time.series

	library(ncdf4)
	library(fields)
	library(rworldmap)
	library(lubridate)
	library(ncdf4.helpers)




#####
## ######  NETCDF FUNCTIONS  #########################################################################
#####

	# a function to understand the variables and dimensions inside a NetCDF object
	# you can do print(nc) but it is ridiculously detailed
	# you can do nc_open(filename) but it doesn't give you back any objects
	# this instead returns a short list of the variables and dimensions
	nc_describe = function(nc, quiet=FALSE)
	{
		nc_vars = nc_desc = nc_dims = NULL

		# get the dimensions
		for (i in 1:nc$ndim)
		{
			dim = nc$dim[[i]]
			push(nc_dims, dim$name)
		}

		# get the variables and long descriptions
		for (i in 1:nc$nvar)
		{
			var = nc$var[[i]]
			push(nc_vars, var$name)

			desc = var$longname %&% ' [' %&% paste(var$varsize, collapse=', ') %&% ']'
			push(nc_desc, desc)
		}

		# print these to console
		if (!quiet)
		{
			cat('\nFound', length(nc_dims), 'Dimensions:\n')
			for (i in 1:length(nc_dims))
				cat(sprintf('%16s [%d]\n', nc_dims[[i]], nc$dim[[i]]$len))

			cat('\nFound', length(nc_vars), 'Variables:\n')
			for (i in 1:length(nc_vars))
				cat(sprintf('%16s', nc_vars[[i]]), '\t', nc_desc[[i]], '\n')
		}

		# and return them
		return( list(vars=nc_vars, desc=nc_desc, dims=nc_dims) )
	}



	# extract the longitude and latitude
	# this will try to find typical places where they are stored (e.g. in MERRA, CMIP5)
	# or you can specify either the name of the dim or the val to read
	nc_lon = function(nc, dim=NULL, var=NULL)
	{
		nc_lon_get = function(nc, dim, var)
		{
			# access the specific dim or var
			if (!is.null(dim)) return(nc$dim[dim]$vals)
			if (!is.null(var)) return(nc$var[var]$vals)

			# search the dims for typical names
			if (!is.null(nc$dim$longitude)) return(nc$dim$longitude$vals)
			if (!is.null(nc$dim$lon)) return(nc$dim$lon$vals)
			if (!is.null(nc$dim$XDim)) return(nc$dim$XDim$vals)

			# search the vars for typical names
			if (!is.null(nc$var$longitude)) return(nc$var$longitude$vals)
			if (!is.null(nc$var$lon)) return(nc$var$lon$vals)
		}

		lon = nc_lon_get(nc, dim, var)

		# do not tolerate 0-360 kind of people...
		if (min(lon)>=0) lon = lon - 180
		lon
	}

	nc_lat = function(nc, dim=NULL, var=NULL)
	{
		# access the specific dim or var
		if (!is.null(dim)) return(nc$dim[dim]$vals)
		if (!is.null(var)) return(nc$var[var]$vals)

		# search the dims for typical names
		if (!is.null(nc$dim$latitude)) return(nc$dim$latitude$vals)
		if (!is.null(nc$dim$lat)) return(nc$dim$lat$vals)
		if (!is.null(nc$dim$YDim)) return(nc$dim$YDim$vals)

		# search the vals for typical names
		if (!is.null(nc$var$latitude)) return(nc$var$latitude$vals)
		if (!is.null(nc$var$lat)) return(nc$var$lat$vals)
	}

	# extract the time dimension and convert into lubridate format
	# this will try to find typical places where they are stored (e.g. in MERRA, CMIP5)
	# or you can specify either the name of the dim or the val to read
	nc_time = function(nc, dim=NULL, var=NULL)
	{
		if (is.null(dim) & is.null(var))
		{
			if (!is.null(nc$var$time)) var = 'time'
			if (!is.null(nc$var$TIME)) var = 'TIME'
			if (!is.null(nc$dim$time)) dim = 'time'
			if (!is.null(nc$dim$TIME)) dim = 'TIME'
		}

		if (!is.null(var))
		{
			# get the series of steps
			steps = ncvar_get(nc, var)

			# get the starting point
			time = ncatt_get(nc, var)
		}

		if (!is.null(dim))
		{
			# get the series of steps
			steps = nc$dim[[dim]]$vals

			# get the starting point
			time = nc$dim[[dim]]
		}

		# we are expecting 'hours since XXXXXXX'
		date_string = strsplit(time$units, ' ')[[1]]
		date_start = ymd(date_string[3])


		# build the series
		if (date_string[1] == 'days')    date_series = date_start + ddays(steps)
		if (date_string[1] == 'hours')   date_series = date_start + dhours(steps)
		if (date_string[1] == 'minutes') date_series = date_start + dminutes(steps)


		# woah - fix stupid climate data!
		if (!is.null(time$calendar))
		{
			if (time$calendar == '360_day')
			{
				cat("nc_time: this as a 360 day calendar - assuming 30 day months...\n")

				num_months = floor(steps / 90) * 3
				date_series = date_start + months(num_months)

				day_fraction = steps %% 90
				date_series = date_series + ddays(day_fraction)
			}
		}


		# boom
		return(date_series)
	}


	# generate a list of parameters that can be used to 
	# subset your netcdf file down to a defined region
	# region needs to be a list containing lon[min, max] and lat[min, max]
	# optionally state where time comes in your list of dimensions ('first' or 'last')
	#
	nc_subset = function(nc, region, time_position=NULL)
	{
		# deal with people who just want a default subset (i.e. no subsetting)
		if (is.null(region))
		{
			start = c(1, 1)
			count = c(-1, -1)
			start2d = c(start, 1)
			count2d = c(count, -1)
			start3d = c(start2d, 1)
			count3d = c(count2d, -1)
			return(list(start=start, count=count, start2d=start2d, count2d=count2d, start3d=start3d, count3d=count3d, lon=nc_lon(nc), lat=nc_lat(nc)))
		}		

		# remove fools
		if (length(region$lon) != 2 | length(region$lat) != 2)
			stop("nc_subset() -- region must be a list of $lon[min,max] $lat[min,max]\n")



		# get the coordinates for our file
		lon = nc_lon(nc)
		lat = nc_lat(nc)

		# find the time dimension
		# can only cope if it's 1st or 3rd
		if (is.null(time_position))
		{
			time_position = 'last'
			if (nc$dim[[1]]$name == 'time')
				time_position = 'first'
		}

		if (time_position %notin% c('first', 'last'))
			stop("nc_subset: error - time_position must either be 'first' or 'last'...\n")

	
		# get our spatial resolution
		dx = abs(diff(lon)[1])
		dy = abs(diff(lat)[1])

		# expand the region to make sure we include the points adjacent to our range
		# but only if our region limits are not exactly on grid points
		region$lon[1] = lon[ which.closest(lon, region$lon[1], round='down') ]
		region$lon[2] = lon[ which.closest(lon, region$lon[2], round='up') ]
		region$lat[1] = lat[ which.closest(lat, region$lat[1], round='down') ]
		region$lat[2] = lat[ which.closest(lat, region$lat[2], round='up') ]


		# deal with proper London centred maps
		if (max(lon) <= 180)
		{
			# define which parts of the file live within our region
			myLon = which(lon >= region$lon[1] & lon <= region$lon[2])
			myLat = which(lat >= region$lat[1] & lat <= region$lat[2])

			# store the start and count parameters to use when reading variables
			start = c(min(myLon), min(myLat))
			count = c(length(myLon), length(myLat))

			# make life simpler for yourself
			if (time_position == 'last')
			{
				start2d = c(start, 1)
				count2d = c(count, -1)
			}

			if (time_position == 'first')
			{
				start2d = c(1, start)
				count2d = c(-1, count)
			}

			start3d = c(start2d, 1)
			count3d = c(count2d, -1)

			# chop down our lon and lat vectors to this region
			lon = lon[myLon]
			lat = lat[myLat]

			return(list(start=start, count=count, start2d=start2d, count2d=count2d, start3d=start3d, count3d=count3d, lon=lon, lat=lat))

			###
			###  THEN ADD A VARIANT TO DEAL WITH CROSSING THE DATE LINE
			###  AS IN MY NZ SIMULATIONS FOR STEVE
			###
		}


		# deal with stupid pacific centred maps
		if (max(lon) > 180)
		{
			# life is easy if we don't cross the meridian
			if (sign(region$lon[1]) == sign(region$lon[2]))
			{
				# move the western hemisphere
				if (region$lon[1] < 0)
					region$lon = region$lon + 360

				# define which parts of the file live within our region
				myLon = which(lon >= region$lon[1] & lon <= region$lon[2])
				myLat = which(lat >= region$lat[1] & lat <= region$lat[2])

				# store the start and count parameters to use when reading variables
				start = c(min(myLon), min(myLat))
				count = c(length(myLon), length(myLat))

				# make life simpler for yourself
				if (time_position == 'last')
				{
					start2d = c(start, 1)
					count2d = c(count, -1)
				}

				if (time_position == 'first')
				{
					start2d = c(1, start)
					count2d = c(-1, count)
				}
				
				start3d = c(start2d, 1)
				count3d = c(count2d, -1)

				# chop down our lon and lat vectors to this region
				lon = lon[myLon]
				lat = lat[myLat]

				return(list(start=start, count=count, start2d=start2d, count2d=count2d, start3d=start3d, count3d=count3d, lon=lon, lat=lat))
			}



			# we have to do two subsets if we do
			if (sign(region$lon[1]) != sign(region$lon[2]))
			{
				 cat("nc_subset: you have a pacific centred map and you want stuff that crosses the meridian...\n")
				stop("           try just getting the eastern or western hemisphere in one go...\n")
			}
		}


	}

	ncvar_get = function(nc, varid=NA, ...)
	{
		# get the variable as you would normally
		val = ncdf4::ncvar_get(nc, varid, ...)

		# then re-order the output from time,lon,lat to lon,lat,time if needed
		# e.g. 1,2,3 will become 2,3,1 or 1,2,3,4 will become 2,3,4,1
		if (nc$var[[varid]]$dim[[1]]$name == 'time')
		{
			dims = seq_along(dim(val))
			dims = c(tail(dims, -1), head(dims, 1))
			val = aperm(val, dims)
		}

		return(val)
	}




#####
#####  MAYBE I NEED MY OWN NC_GETVAR FUNCTION...
#####  WHICH DETECTS AND SHIFTS PACIFIC CENTRED MAPS
#####  DOES THE SUBSETTED READING FOR YOU
#####  AND CAN DO MULTI SUBSETS IF NEEDED
#####
#####  BUT... HOW TO TELL IT OTHER PARTS OF SUBSETTING I WANT TO DO :( :( :(
##### 
#####  ALSO - I CAN'T EVEN FIGURE OUT HOW TO STORE THE MULTIPLE START/COUNT VARIABLES...
#####





#####
## ######  PLOTTING FUNCTIONS  #########################################################################
#####

	##
	##  simplest plot using lattice
	##  for learning purposes......
	##
	nc_plot1 = function(lon, lat, data, ...)
	{
		# make sure our lon and lat are ordered
		if (!is.ordered(lon) | !is.ordered(lat))
		{
			o1 = order(lon)
			o2 = order(lat)

			lon = lon[o1]
			lat = lat[o2]

			data = data[o1, o2]
		}

		# plot
		filled.contour(lon, lat, data, ...)
	}




	##
	##  slightly more complex, with london centering and map overlay
	##  for learning purposes......
	##
	nc_plot2 = function(lon, lat, data, ...)
	{
		# shift pacific-centred maps back to london...
		if (max(lon) > 180)
		{
			east = (lon > 180)
			lon[east] = lon[east] - 360
		}

		# make sure our lon and lat are ordered
		if (!is.ordered(lon) | !is.ordered(lat))
		{
			o1 = order(lon)
			o2 = order(lat)

			lon = lon[o1]
			lat = lat[o2]

			data = data[o1, o2]
		}

		# get our world map shape (high res for better visuals)
		if (!exists('world.map'))
			world.map <<- get_world_map(res='high')

		# plot
		filled.contour(
			lon, lat, data, 
			xlab="Longitude", ylab="Latitude",
			plot.axes = { plot(world.map, add=TRUE); axis(1); axis(2) },
			...
		)
	}





	##
	##  more complex still, with colours, better legend, etc.
	##  for learning purposes......
	##
	nc_plot3 = function(lon, lat, data, colours=NULL, nLevels=64, aspect=NULL, ...)
	{
		# shift pacific-centred maps back to london...
		if (max(lon) > 180)
		{
			east = (lon > 180)
			lon[east] = lon[east] - 360
		}

		# make sure our lon and lat are ordered
		if (!is.ordered(lon) | !is.ordered(lat))
		{
			o1 = order(lon)
			o2 = order(lat)

			lon = lon[o1]
			lat = lat[o2]

			data = data[o1, o2]
		}

		# get our world map shape (high res for better visuals)
		if (!exists('world.map'))
			world.map <<- get_world_map(res='high')


		# specify a default colour scheme for the plot
		if (is.null(colours))
			colours = c('#3F168A', '#2049D0', '#3288BD', '#66C2A5', '#ABDDA4', '#E6F598', '#FFFFBF', '#FEE08B', '#FDAE61', '#F46D43', '#D53E4F', '#9E0142')

		# convert into a colour ramp function
		colours = colorRampPalette(colours)


		# aspect ratio of the map
		if (is.null(aspect))
			aspect = 2.5

		aspect = aspect * (max(lat)-min(lat)) / (max(lon)-min(lon))


		# plot
		filled.contour(
			lon, lat, data, 
			asp=aspect, nlevels=nLevels, color.palette=colours,
			xlab="Longitude", ylab="Latitude",
			plot.axes = { plot(world.map, add=TRUE); axis(1); axis(2) },
			...
		)
	}




	##
	##  plot 2d netcdf data
	##     lon = vector of longitudes
	##     lat = vector of latitudes
	##     data = 2d array of data corresponding to those
	##
	##    (optional)
	##     xlim = the min/max longitude to cover (vector of 2)
	##     ylim = the min/max latitude to cover (vector of 2)
	##     colours = a colour ramp (vector of hex codes)
	##     border = a colour for the map borders
	##     nLevels = the number of colour levels to plot (more looks smoother, but is slower)
	##     zRange = the min/max data value to show (vector of 2)
	##     aspect = the aspect ratio to plot the map
	##     mainTitle = title text to go above main plot
	##     legendTitle = title text to go above the scale bar
	##     fullScreen = plot fullscreen for videos and other glamourous things (no axes, no legend, no nothing)
	##     myPoints = lattice makes it hard to plot points on a chart afterwards - so wrap your points(...) call in a function and pass it
	##                e.g. m = function() { points(x, y) }; nc_plot(..., myPoints=m)
	##
	nc_plot = function(lon, lat, data, xlim=range(lon), ylim=range(lat), colours=NULL, border='black', nLevels=64, zRange=range(data, na.rm=TRUE), aspect=1, mainTitle='', legendTitle='', fullScreen=FALSE, myPoints=NULL, lwd=1, ...)
	{
		# fix data
		data[ is.infinite(data) ] = NA

		# shift pacific-centred maps back to london...
		if (max(lon) > 180)
		{
			east = (lon > 180)
			lon[east] = lon[east] - 360
		}

		# make sure our lon and lat are ordered
		if (!is.ordered(lon) | !is.ordered(lat))
		{
			o1 = order(lon)
			o2 = order(lat)

			lon = lon[o1]
			lat = lat[o2]

			data = data[o1, o2]
		}

		# get our world map shape (high res for better visuals)
		if (!exists('world.map'))
			world.map <<- get_world_map(res='high')



		# specify a default colour scheme for the plot
		if (is.null(colours))
			colours = c('#3F168A', '#2049D0', '#3288BD', '#66C2A5', '#ABDDA4', '#E6F598', '#FFFFBF', '#FEE08B', '#FDAE61', '#F46D43', '#D53E4F', '#9E0142')

		# convert into a colour ramp function
		colours = colorRampPalette(colours)

		# aspect ratio of the map
		aspect = aspect / cos((min(lat)+max(lat)) * pi / 360)

		# make sure our data sits within the zRange (so it can be plotted)
		if (any(zRange != range(data, na.rm=TRUE)))
		{
			data[data < zRange[1]] = zRange[1]
			data[data > zRange[2]] = zRange[2]
		}


		if (!fullScreen)
		{
			# plot
			filled.contour.is(
				lon, lat, data, 
				asp=aspect, nlevels=nLevels, color.palette=colours,
				xlim=xlim, ylim=ylim, zlim=zRange,
				key.title = title(main=legendTitle), plot.title = title(main=mainTitle, xlab='Longitude', ylab='Latitude'),
				plot.axes = { plot(world.map, xlim=xlim, ylim=ylim, border=border, lwd=lwd, add=TRUE); if (!is.null(myPoints)) myPoints(); axis(1); axis(2) },
				...
			)
		}


		if (fullScreen)
		{
			# remove the borders for full-screen
			omar = par(mar = par('mar'))
			on.exit(par(omar))
			par(mar=rep(0,4))

			# plot
			filled.contour.is(
				lon, lat, data, 
				asp=aspect, nlevels=nLevels, color.palette=colours,
				xlim=xlim, ylim=ylim, zlim=zRange,
				plot.axes = { plot(world.map, xlim=xlim, ylim=ylim, add=TRUE) },
				axes=FALSE, add.legend=FALSE
			)
		}
	}

	# nc_plot(lon, lat, results, aspect=3, nLevels=128, mainTitle='Model Level 2', legendTitle='Height (m)')





	nc_plot_gridded = function(lon, lat, data, xlim=range(lon), ylim=range(lat), colours=NULL, border='black', nLevels=64, zRange=range(data, na.rm=TRUE), zAt=NULL, aspect=1, mainTitle='', legendTitle='', fullScreen=FALSE, lwd=1, ...)
	{
		# fix data
		data[ is.infinite(data) ] = NA

		# shift pacific-centred maps back to london...
		if (max(lon) > 180)
		{
			east = (lon > 180)
			lon[east] = lon[east] - 360
		}

		# make sure our lon and lat are ordered
		if (!is.ordered(lon) | !is.ordered(lat))
		{
			o1 = order(lon)
			o2 = order(lat)

			lon = lon[o1]
			lat = lat[o2]

			data = data[o1, o2]
		}

		# get our world map shape (high res for better visuals)
		if (!exists('world.map'))
			world.map <<- get_world_map(res='high')



		# specify a default colour scheme for the plot
		if (is.null(colours))
			colours = c('#3F168A', '#2049D0', '#3288BD', '#66C2A5', '#ABDDA4', '#E6F598', '#FFFFBF', '#FEE08B', '#FDAE61', '#F46D43', '#D53E4F', '#9E0142')


		colours = colorRampPalette(colours)(nLevels)

		# aspect ratio of the map
		aspect = aspect / cos((min(lat)+max(lat)) * pi / 360)


		# make sure our data sits within the zRange (so it can be plotted)
		if (any(zRange != range(data, na.rm=TRUE)))
		{
			data[data < zRange[1]] = zRange[1]
			data[data > zRange[2]] = zRange[2]
		}


		if (fullScreen)
		{
			# remove the borders for full-screen
			omar = par(mar = par('mar'))
			on.exit(par(omar))
			par(mar=rep(0,4))
		}

		{
			# plot
			if (is.null(zAt))
			{
				image.plot(
					lon, lat, data,
					asp=aspect, nlevel=nLevels, col=colours,
					xlim=xlim, ylim=ylim, zlim=zRange,
					main=mainTitle, xlab='Longitude', ylab='Latitude'
				)
			} else {
				image.plot(
					lon, lat, data,
					asp=aspect, nlevel=nLevels, col=colours,
					xlim=xlim, ylim=ylim, zlim=zRange,
					main=mainTitle, xlab='Longitude', ylab='Latitude',
					axis.args=list(at=zAt, labels=zAt)
				)
			}

			# we could use legend.lab=legendTitle
			# but that adds the title to the right of the legend
			# could also try legend.args=list(text=legendTitle, side=3)
			# but that gives an error!
			if (legendTitle != '')
			{
				lx = range(lon)
				if (!is.null(xlim)) lx = xlim

				lx = max(lx) + 0.14 * diff(lx)
				mtext(legendTitle, side=3, at=lx, font=2, cex=1.2)
			}

			plot(world.map, border=border, lwd=lwd, add=TRUE)
			box()
		}

	}



	#######################################################################################################################
	##
	##  a better filled contour function
	##  iain staffell ~ 2014
	##
	##  by default it no longer draws borders between levels in the legend
	##  this lets you draw smooth plots with lots of levels..
	##
	##  if you run with the new option add.legend=FALSE, you just end up with
	##  the level plot.  this means subsequent calls to points(), lines(), etc..
	##  will work as expected, as you remain in the coordinate system of the plot body
	##
	##  if you call par(mar=rep(0,4)), then call this with axes=FALSE, add.legend=FALSE
	##  you can obtain a full-screen plot with no borders, no nothing. 
	##
	filled.contour.is = function (x = seq(0, 1, length.out = nrow(z)), y = seq(0, 1, 
		length.out = ncol(z)), z, xlim = range(x, finite = TRUE), 
		ylim = range(y, finite = TRUE), zlim = range(z, finite = TRUE), 
		levels = pretty(zlim, nlevels), nlevels = 20, color.palette = cm.colors, 
		col = color.palette(length(levels) - 1), plot.title, plot.axes, 
		key.title, key.axes, asp = NA, xaxs = "i", yaxs = "i", las = 1, 
		axes = TRUE, frame.plot = axes, add.legend=TRUE, ...) 
	{
		# sort out data
		if (missing(z)) {
			if (!missing(x)) {
				if (is.list(x)) {
					z <- x$z
					y <- x$y
					x <- x$x
				}
				else {
					z <- x
					x <- seq.int(0, 1, length.out = nrow(z))
				}
			}
			else stop("no 'z' matrix specified")
		}
		else if (is.list(x)) {
			y <- x$y
			x <- x$x
		}
		if (any(diff(x) <= 0) || any(diff(y) <= 0)) 
			stop("increasing 'x' and 'y' values expected")

		mar.orig <- (par.orig <- par(c("mar", "las", "mfrow")))$mar
		on.exit(par(par.orig))

		# plot legend
		if (add.legend)
		{
			w <- (3 + mar.orig[2L]) * par("csi") * 2.54
			layout(matrix(c(2, 1), ncol = 2L), widths = c(1, lcm(w)))
			par(las = las)
			mar <- mar.orig
			mar[4L] <- mar[2L]
			mar[2L] <- 1
			par(mar = mar)

			plot.new()
			plot.window(xlim = c(0, 1), ylim = range(levels), xaxs = "i", yaxs = "i")
			rect(0, levels[-length(levels)], 1, levels[-1L], col = col, border = NA)  ## this removes legend borders
			if (missing(key.axes)) {
				if (axes) 
					axis(4)
			}
			else key.axes
			box()
			if (!missing(key.title)) 
				key.title
			mar <- mar.orig
			mar[4L] <- 1
			par(mar = mar)
		}

		# plot body
		plot.new()
		plot.window(xlim, ylim, "", xaxs = xaxs, yaxs = yaxs, asp = asp)
		.filled.contour(x, y, z, levels, col)
		if (missing(plot.axes)) {
			if (axes) {
				title(main = "", xlab = "", ylab = "")
				Axis(x, side = 1)
				Axis(y, side = 2)
			}
		}
		else plot.axes
		if (frame.plot) 
			box()
		if (missing(plot.title)) 
			title(...)
		else plot.title
		invisible()
	}

