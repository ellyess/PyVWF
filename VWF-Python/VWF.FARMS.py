# a function to prepare wind farms and check they are VALID

from os.path import exists

file_exists = exists(path_to_file)

if file_exists = True:


#we check if the wind farm info exits
# allow turbine to interchangibe with power curve if we have it in windfarmcols
# allow NaN to exist
# readin windfarm file
# check wind farm has following columns  ('name', 'lon', 'lat', 'height', 'power_curve', 'offshore')
# and that the columns have values >0

# assign defgult capacity if none assigned 1 MW
# assign 1 performance ratio if not specified.
# assign date to today if none existent -
# some form of search for if it says offshore
# check and fix date formats to one
# checking for power curves in files

#hack(?) function that be deleted by july 2020
# convert wind farm names into valid column names (to match those in the wind speed file)
#simplify farm heights to speed up extrapolation
# have these sort-of equally spaced in log terms (~0.015 resolution in log10)
