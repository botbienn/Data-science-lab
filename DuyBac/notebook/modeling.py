import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib

'''------------------------------------------------------------------------'''

def get_file_list(): 
    # import required module
    import os
    # assign directory
    directory = '../data/'

    # iterate over files in that directory
    result = [f[:-4] for f in os.listdir(directory)]
    return result 

LOCATIONS_LIST = get_file_list()

def read_location_data(loc_list): 
    result = [pd.read_csv(f'../data/{loc}.csv') 
                             for loc in loc_list]
    return result

pd_list = read_location_data(LOCATIONS_LIST) 

print(pd_list[0].head())

'''------------------------------------------------------------------------'''

print('end')
