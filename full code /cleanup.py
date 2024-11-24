import pandas as pd

RAW_DATA_PATH = '../raw_data/'
CLEANED_DATA_PATH = '../cleaned_data/'
LOCATION_FILE_NAME = ['SonLa','LangSon', 'HaNoi',
                    'NgheAn', 'DaNang', 'LamDong',
                    'HoChiMinh','BenTre']
dataframe_list = []

def read_raw_datas() -> list:
    """Takes a location file name and return 
    the pandas dataframe from that csv """

    result_list = []
    for file_name in LOCATION_FILE_NAME:
        result_list.append(pd.read_csv(RAW_DATA_PATH + f'{file_name}.csv'))

    return result_list

