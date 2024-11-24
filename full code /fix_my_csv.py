"""module for dataframes"""
import pandas as pd

def fix_csv(location):
    """fix son la lang son csv files """
    string = f'raw_data/{location}.csv'

    df = pd.read_csv(string, index_col = 0)

    # df.drop('Unnamed: 0')
    # print(df.columns)
    # print(df.head())

    df.to_csv(string, index = False)

# fix_csv('LangSon')
# fix_csv('SonLa')

def add_dt_epoch(location):
    """make sure HaNoi and NgheAn have dateTime epoch"""
    og_data_file = 'raw_data/HoChiMinh.csv'
    df_temp = pd.read_csv(og_data_file)
    date_time =  df_temp['DatetimeEpoch'].tolist()
    # print(date_time)

    string = f'raw_data/{location}.csv'
    df = pd.read_csv(string)
    df['DatetimeEpoch'] = date_time
    df.to_csv(string, index=False)

add_dt_epoch('HaNoi')
add_dt_epoch('NgheAn')
