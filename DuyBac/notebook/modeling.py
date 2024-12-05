import pandas as pd
import numpy as np
from pandas.core.array_algos.replace import compare_or_regex_search
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


def read_location_data(loc_list): 
    result = [pd.read_csv(f'../data/{loc}.csv') for loc in loc_list]
    return result


'''------------------------------------------------------------------------'''
# Phân tích lựa chọn các thuộc tính cần để sử dụng cho model

def extract_corr_matrix(df_s: list):
    result = [df.select_dtypes(exclude='object').corr() 
        for df in df_s]    
    return result

'''------------------------------------------------------------------------'''
LOCATIONS_LIST = get_file_list()

df_list = read_location_data(LOCATIONS_LIST) 
corr_mats = extract_corr_matrix(df_list)

print(corr_mats[0])

print('end')
