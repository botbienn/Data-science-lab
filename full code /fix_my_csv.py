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

fix_csv('LangSon')
fix_csv('SonLa')
