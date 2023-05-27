import pandas as pd
import numpy as np
from io import StringIO
import matplotlib.pyplot as plt
import statsmodels.api as sm
from graph_generator import get_graph_url
from sklearn.metrics import mean_squared_error
import numpy as np
from flask import Flask, session

def readFile(fileData):
    csv_data = StringIO('\n'.join(fileData))

    df = pd.read_csv(csv_data, names=['date', 'value'])
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)

    return df

p = d = q = range(0, 3)

parametru_kombinacijos = [(i, j, k) for i in p for j in d for k in q]

geriausias_aic = float('inf')
geriausias_p = geriausias_d = geriausias_q = None

for parametrai in parametru_kombinacijos:
    try:
        model = ARIMA(data, order=parametrai)
        model_fit = model.fit()
        aic = model_fit.aic
        if aic < geriausias_aic:
            geriausias_aic = aic
            geriausias_p, geriausias_d, geriausias_q = parametrai
    except:
        continue

def arima_data(data, date_diff_data, original_data, prognosis_num, checkbox, period, ar, ma):

    if checkbox == 1:
        I = 1
        data = readFile(date_diff_data)
    else:
        I = 0
        data = readFile(data)

    original_data = readFile(original_data)

    prognosis_num = int(prognosis_num)

    if ar.isnumeric():
        AR = int(ar)
    elif geriausias_p is None:
        AR =1
    else: AR = geriausias_p

    if ma.isnumeric():
        MA = int(ma)
    elif geriausias_q is None:
        MA =1
    else: MA = geriausias_q

    max_value = data['value'].max()
    data['value_normalized'] = data['value'] / max_value

    model = sm.tsa.ARIMA(data['value_normalized'], order=(AR, I, MA))

    results = model.fit()

    forecast_normalized = results.forecast(steps=prognosis_num)

    forecast = forecast_normalized * max_value

    forecast_dates = pd.date_range(
        start=data.index[-1], periods=prognosis_num+1, freq=period)[1:]

    forecast_data = pd.DataFrame(
        {'date': forecast_dates, 'value': forecast})
    forecast_result = forecast_data.set_index('date')

    if checkbox == 1:
        forecast_result = forecast_result.cumsum()

    forecast_series = forecast_result['value']

    fig, ax = plt.subplots()
    plt.plot(data.index, data['value'], label='Data')
    plt.plot(forecast_series.index, forecast_series, label='Prognozė')
    plt.xticks([])
    prognosis_graph_url = get_graph_url(ax.figure)

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