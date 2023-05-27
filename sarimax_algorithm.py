import pandas as pd
import numpy as np
from io import StringIO
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import statsmodels.api as sm
from graph_generator import get_graph_url
from sklearn.metrics import mean_squared_error
import numpy as np
from flask import Flask, session
import itertools

def readFile(fileData):
    csv_data = StringIO('\n'.join(fileData))

    df = pd.read_csv(csv_data, names=['date', 'value'])
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)

    return df

geriausias_p = None
geriausias_q = None
geriausias_P = None
geriausias_Q = None
p = d = q = range(0, 3)
P = Q = range(0, 3)

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

parametru_kombinacijos = list(itertools.product(P, Q))

geriausias_aic = float('inf')
geriausieji_parametrai = None

for param in parametru_kombinacijos:
    try:
        model = SARIMAX(data, order=(0, 0, 0), seasonal_order=(param[0], 0, param[1], 12))
        model_fit = model.fit()
        aic = model_fit.aic
        if aic < geriausias_aic:
            geriausias_aic = aic
            geriausieji_parametrai = param
    except:
        continue

if geriausieji_parametrai is not None:
    # Išskleiskite geriausius parametrus
    geriausias_P, geriausias_Q = geriausieji_parametrai

def sarimax_data(data, date_diff_data, original_data, prognosis_num, checkbox, period, ar, ma, sar, sma, sarimax_num):
    if checkbox == 1:
        data = readFile(date_diff_data)
        SI = 1
        I= 1
    else:
        data = readFile(data)
        SI = 0
        I=0

    original_data = readFile(original_data)
    prognosis_num = int(prognosis_num)

    if sarimax_num.isnumeric():
        periods_per_season = int(sarimax_num)
    else:
        periods_per_season = 12

    if ar.isnumeric():
        AR = int(ar)
    elif geriausias_p is None:
        AR = 1
    else:
        AR = geriausias_p

    if ma.isnumeric():
        MA = int(ma)
    elif geriausias_q is None:
        MA = 1
    else:
        MA = geriausias_q

    if sar.isnumeric():
        SAR = int(sar)
    elif geriausias_P is None:
        SAR = 1
    else:
        SAR = geriausias_P

    if sma.isnumeric():
        SMA = int(sma)
    elif geriausias_Q is None:
        SMA = 1
    else:
        SMA = geriausias_Q

    scaler = MinMaxScaler()
    data['value'] = scaler.fit_transform(data[['value']])

    model = sm.tsa.statespace.SARIMAX(data, order=(AR, I, MA), seasonal_order=(SAR, SI, SMA, periods_per_season), enforce_stationarity=False, enforce_invertibility=False)

     # Prideriname modelį
    results = model.fit()

    forecast = results.forecast(steps=prognosis_num)

    forecast_original = scaler.inverse_transform(
        forecast.values.reshape(-1, 1))

    forecast_dates = pd.date_range(
        start=data.index[-1], periods=prognosis_num+1, freq=period)[1:]
    forecast_data = pd.DataFrame(
        {'date': forecast_dates, 'value': forecast_original.flatten()})
    forecast_result = forecast_data.set_index('date')

    if checkbox == 1:
        forecast_result = forecast_result.cumsum()

    fig, ax = plt.subplots()
    plt.plot(data.index[::5], scaler.inverse_transform(data[['value']])[::5], label='Data')
    plt.plot(forecast_dates, forecast_result['value'], label='Prognozė')
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
        'rmse': round(rmse, 5)
    }
    return result_dict