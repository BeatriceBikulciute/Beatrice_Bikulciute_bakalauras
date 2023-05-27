import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.api import ExponentialSmoothing
import matplotlib.dates as mdates
from graph_generator import get_graph_url
from sklearn.metrics import mean_squared_error
import numpy as np
from flask import Flask, session
from statsmodels.tsa.api import ExponentialSmoothing

def readFile(fileData):
    csv_data = StringIO('\n'.join(fileData))

    df = pd.read_csv(csv_data, names=['date', 'value'])
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)

    return df

def dlt_data(data, date_diff_data, original_data, prognosis_num, checkbox, period, type_t):

    if checkbox == 1:
        data = readFile(date_diff_data)
    else:
        data = readFile(data)

    original_data = readFile(original_data)

    prognosis_num = int(prognosis_num)

    scaler = MinMaxScaler()
    data['value'] = scaler.fit_transform(data[['value']])

    model = ExponentialSmoothing(data, trend=type_t, damped_trend=True, seasonal=None)
    results = model.fit()

    forecast = results.forecast(steps=prognosis_num)

    forecast_unscaled = scaler.inverse_transform(
        forecast.values.reshape(-1, 1))

    forecast_df = pd.DataFrame(index=pd.date_range(start=data.index[-1], periods=prognosis_num+1, freq=period)[1:],
                               columns=['value'], data=forecast_unscaled.flatten())

    if checkbox == 1:
        anti_diff_data = forecast_df.cumsum()
        forecast_result = anti_diff_data
    else:
        forecast_result = forecast_df

    plt.figure()
    plt.plot(data.index, data['value'])
    plt.plot(forecast_result.index, forecast_result['value'], color='red', label='Forecast')
    plt.xticks([])
    prognosis_graph_url = get_graph_url(plt.gcf())

    if (prognosis_num) < 6:
        y_true = np.array(original_data['value'])[:prognosis_num]
        y_pred = np.array(forecast_result['value'])[:prognosis_num]
    else:
        y_true = np.array(original_data['value'])
        y_pred = np.array(forecast_result['value'])[:6]

    rmse = mean_squared_error(y_true, y_pred)

    result_dict = {
        'forecast_result': forecast_result,
        'prognosis_graph_url': prognosis_graph_url,
        'rmse': round(rmse,5)
    }

    return result_dict