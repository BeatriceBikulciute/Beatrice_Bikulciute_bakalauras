import pandas as pd
import numpy as np
from io import StringIO
from numpy import array
import matplotlib.pyplot as plt
from graph_generator import get_graph_url
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error
import numpy as np
from flask import Flask, session

def readFile(fileData):
    csv_data = StringIO('\n'.join(fileData))

    df = pd.read_csv(csv_data, names=['date', 'value'])
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)

    return df

def lstm_data(data, date_diff_data, original_data, prognosis_num, checkbox, period, epoch, neur_sk):

    if checkbox == 1:
        data = readFile(date_diff_data)
    else:
        data = readFile(data)

    original_data = readFile(original_data)

    prognosis_num = int(prognosis_num)

    if epoch.isnumeric():
        EPOCH = int(epoch)
    else: EPOCH = 300

    if neur_sk.isnumeric():
        N_SK = int(neur_sk)
    else: N_SK=50

    sequence = data['value'].values

    X, y = list(), list()
    for i in range(len(sequence)):

        end_ix = i + prognosis_num

        if end_ix > len(sequence)-1:
            break

        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    X, y = array(X), array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    model = Sequential()
    model.add(LSTM(N_SK, activation='relu', input_shape=(prognosis_num, 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    neur_sk = 10
    neur_sk = int(N_SK)
    model.fit(X, y, epochs=N_SK, verbose=0)

    last_value = sequence[-1]
    forecast_values = []
    for i in range(prognosis_num):
        x_input = array(sequence[-prognosis_num:])
        x_input = x_input.reshape((1, prognosis_num, 1))
        yhat = model.predict(x_input, verbose=0)

        if checkbox == 1:
            yhat = yhat + last_value

        sequence = list(sequence) + [yhat[0, 0]]
        sequence = sequence[1:]

        if checkbox == 1:
            yhat = yhat + last_value
            forecast_values.append(yhat[0, 0])
        else:
            forecast_values.append(sequence[-1])

    forecast_dates = pd.date_range(
        start=data.index[-1], periods=prognosis_num+1, freq=period)[1:]
    forecast_result = pd.DataFrame(
        {'date': forecast_dates, 'value': forecast_values})
    forecast_result.set_index('date', inplace=True)

    fig, ax = plt.subplots()
    plt.plot(sequence, label='Seka')
    plt.plot(range(len(sequence), len(sequence) +
             len(yhat[0])), yhat[0], label='Prognozuotos reikšmės')
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