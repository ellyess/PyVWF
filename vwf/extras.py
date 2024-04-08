import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

bg_colour = '#f0f0f0'
custom_params = {'xtick.bottom': True, 'axes.edgecolor': 'black', 'axes.spines.right': False, 'axes.spines.top': False, 'mathtext.default': 'regular'}
sns.set_theme(style='ticks', rc=custom_params)

def add_time_res(df):
    """
    Add columns to identify time resolutions.
    """
    df.loc[df['month'] == 1, ['bimonth','season']] = ['1/6', 'winter']
    df.loc[df['month'] == 2, ['bimonth','season']] = ['1/6', 'winter']
    df.loc[df['month'] == 3, ['bimonth','season']] = ['2/6', 'spring']
    df.loc[df['month'] == 4, ['bimonth','season']] = ['2/6', 'spring']
    df.loc[df['month'] == 5, ['bimonth','season']] = ['3/6', 'spring']
    df.loc[df['month'] == 6, ['bimonth','season']] = ['3/6', 'summer']
    df.loc[df['month'] == 7, ['bimonth','season']] = ['4/6', 'summer']
    df.loc[df['month'] == 8, ['bimonth','season']] = ['4/6', 'summer']
    df.loc[df['month'] == 9, ['bimonth','season']] = ['5/6', 'autumn']
    df.loc[df['month'] == 10, ['bimonth','season']] = ['5/6', 'autumn']
    df.loc[df['month'] == 11, ['bimonth','season']] = ['6/6', 'autumn']
    df.loc[df['month'] == 12, ['bimonth','season']] = ['6/6', 'winter']
    df['fixed'] = '1/1'
    return df