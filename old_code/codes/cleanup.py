"""dataframe module"""
import pandas as pd

RAW_DATA_PATH = 'raw_data/'

CLEANED_DATA_PATH = 'cleaned_data/'

LOCATION_FILE_NAME = ['SonLa','LangSon', 'HaNoi',
                    'NgheAn', 'DaNang', 'LamDong',
                    'HoChiMinh','BenTre']

LOCATION_VALUES = ['Sơn La','Lạng Sơn', 'Hà Nội',
                    'Nghệ An', 'Đà Nẵng', 'Lâm Đồng',
                    'Hồ Chí Minh','Bến Tre']

# LOCATION_NAME_MAP = {LOCATION_FILE_NAME[i] : LOCATION_VALUES[i]
#                      for i in range(len(LOCATION_VALUES)) }

SAVE_COLS = [ 'Address', 'Datetime', 'DatetimeEpoch',
            'Tempmax', 'Tempmin', 'Temp', 'Dew', 
            'Humidity', 'Precip', 'Precipprob', 'Precipcover',
            'Preciptype', 'Windgust', 'Windspeed', 'Winddir', 
            'Pressure', 'Cloudcover', 'Visibility', 'Solarradiation', 
            'Solarenergy', 'Uvindex', 'Moonphase' ]

def read_raw_datas() -> list:
    """Takes a location file name and return 
    the pandas dataframe from that csv """
    result_list = []

    for file_name in LOCATION_FILE_NAME:
        df = pd.read_csv(RAW_DATA_PATH + f'{file_name}.csv')
        result_list.append(df)

    return result_list

def add_address_column(df, location_name) -> pd.DataFrame:
    """Drop old address columns and add standardized one"""
    if 'Address' in df.columns:
        df.drop('Address', axis=1)
    if 'address' in df.columns:
        df.drop('address', axis=1)

    address_list = [location_name for _ in range(len(df.index))]
    df = df.assign(Address = pd.Series(address_list))
    return df

def drop_unnecessary_columns(df) -> pd.DataFrame:
    """Drop unnecessary features to clean up datas,
    also rearrange columns with a fixed order"""
    df = df[SAVE_COLS]
    return df

def upper_case_name(df) -> pd.DataFrame:
    """Make all columns name uppercase"""
    col_names = df.columns
    upper_col_name = [col_name[0].upper() + col_name[1:]
                      for col_name in col_names]

    df.rename({col_names[i] : upper_col_name[i]
               for i in range(len(col_names))},
              axis=1, inplace=True)

    return df

def export_cleaned_df(df, location_name):
    """export cleaned up dataframe to csv files"""
    df.to_csv(CLEANED_DATA_PATH + f'{location_name}.csv', index=False)


def main():
    """Main function"""
    dataframe_list = read_raw_datas()
    for i, df in enumerate(dataframe_list):
        df = add_address_column(df, LOCATION_VALUES[i])
        df = upper_case_name(df)
        df = drop_unnecessary_columns(df)
        export_cleaned_df(df, LOCATION_FILE_NAME[i])

if __name__ == "__main__":
    main()