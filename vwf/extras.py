import numpy as np
import pandas as pd

def add_times(data):
    data['year'] = pd.DatetimeIndex(data['time']).year
    data['month'] = pd.DatetimeIndex(data['time']).month
    data.insert(1, 'year', data.pop('year'))
    data.insert(2, 'month', data.pop('month'))
    return data