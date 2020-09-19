import pandas as pd
result_file_list = ['barclays_lags_result.txt','barclays_lags_weather_result.txt','barclays_lags_weather_event_result.txt','barclays_lags_weather_event_text_result.txt']

for result_file in result_file_list:
    print('{result_file} result statistics'.format(result_file = result_file))
    data = pd.read_table(result_file, sep='\t')
    print(data.describe())


