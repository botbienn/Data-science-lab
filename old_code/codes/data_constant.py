"""These are all the data constants"""
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

