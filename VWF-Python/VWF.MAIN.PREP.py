import sys

# Bias correction FUNCTIONS

# decide where to get performance ratio data for each farm
# either set PR_column to be a column name in WindFarmFile to use the csv data
# or set to NULL to use hard-coded defaults...
PR_column = 'PR'

if 'match_method' not in locals():
    match_method = 2

if 'pr_method' not in locals():
    pr_method = 1

# determine the scalar for each farm (if we are using fixed scalars)
# this gives the number that we multiply all wind speeds by, before finding the offset which matches our desired capacity factor
def determine_farm_scalar(PR, iso = None):
    if match_method == 1:
        scalar = 0.85

    if match_method == 2:
        scalar = (scalar_alpha * PR) + scalar_beta

    return scalar


# determine the offset for each farm (if we are using fixed offsets)
# this gives the number we subtract from all wind speeds (in m/s), before finding the scalar which matches our desired capacity factor
def determine_farm_offset(PR):
    if match_method == 3:
        offset = 1.00

    if match_method == 4:
        offset = 3.00 * PR**0.333 / 0.85**0.333

    return offset


# this function makes no sense and says should be improved in r code
def get_power_curve_convolution_spread(farm):
    return convolverStdDev


# perform post-process bias correction to the files
# bias is a list where the names equal the values of windFarms$column
# and its values are what the PR of windFarms should be multiplied/added
def farms_bias_correct(farms, column, bias):
    if column not in df:
        sys.stdout.flush("farm_bias_correct: cannot find", column, "column..." )
        sys.stdout.flush("i am not altering the PR for your farms..")

    farms["iso"].replace('UK', 'GB', inplace=True)
    farms["iso"].replace('EL', 'GR', inplace=True)

    for n in bias:
        f = farms[column] == n
        if any(f) == True:
            if pr_method == 1:
                farms.loc[f, 'PR'] = farms.loc[f, 'PR'] * bias.at[0, n]
            if pr_method == 2:
                farms.loc[f, 'PR'] = farms.loc[f, 'PR'] + bias.at[0, n]

    # complex code to report what bias correction we adding look at later
    return farms

# NEEDS WORK
def farms_inflate_offshore_pr(farms, offshore_factor, column='offshore'):
    if column not in df:
        sys.stdout.flush("farm_bias_correct: cannot find", column, "column..." )
        sys.stdout.flush("i am not altering the PR for your farms..")

    farms["iso"].replace('UK', 'GB', inplace=True)
    farms["iso"].replace('EL', 'GR', inplace=True)

    for n in bias:
        f = farms[column] == n
        if any(f) == True
            if pr_method == 1:
                farms.loc[f, 'PR'] = farms.loc[f, 'PR'] * bias.at[0, n]
            if pr_method == 2:
                farms.loc[f, 'PR'] = farms.loc[f, 'PR'] + bias.at[0, n]

    # complex code to report what bias correction we adding look at later
    return farms

# code to let you know windSpeedFile doesn't exist

# ALL THE CODE AFTER THIS IS MORE LIKE UI TO LET YOU KNOW IF RIGHT FILES AND CALL ALL NEEDED FUNCTIONS
