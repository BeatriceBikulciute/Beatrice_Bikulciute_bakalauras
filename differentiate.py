import pandas as pd
import numpy as np
from io import StringIO
from statsmodels.tsa.stattools import adfuller, acf, pacf
import matplotlib
import matplotlib.pyplot as plt
from graph_generator import get_graph_url
from flask import Flask, session
import statsmodels.api as sm

def differentiate_data(fileData):
    csv_data = StringIO('\n'.join(fileData))

    df = pd.read_csv(csv_data, names=['date', 'value'])
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    data = df['value'].values

    h = 1
    diff_data = []
    dates = []
    for i in range(len(data) - 1):
        diff = (data[i+1] - data[i]) / h
        diff_data.append(diff)
        dates.append(df.index[i+1])
    date_diff_data = pd.DataFrame({'date': dates, 'diff_data': diff_data})

    date_diff_data = date_diff_data.apply(lambda x: ','.join(
        [str(x['date'].date()), str(x['diff_data'])]), axis=1).tolist()

    session['date_diff_data'] = date_diff_data

    adf_result = adfuller(diff_data)
    adf_statistic = adf_result[0]
    adf_pvalue = adf_result[1]
    adf_lags = adf_result[2]
    adf_crit_values = adf_result[4]
    crit_value_10_percent = adf_result[4]['10%']

    lag_acf = acf(diff_data, nlags=5)
    lag_pacf = pacf(diff_data, nlags=5, method='ols')

    corr_rez = sm.tsa.acf(df['value'], nlags=12)
    corr_rezz = corr_rez[1:]
    max_corr = np.max(corr_rezz)  
    max_lag = np.argmax(corr_rezz)  
    min_corr = np.min(corr_rezz)  
    min_lag = np.argmin(corr_rezz)  
    if (abs(max_corr) >= abs(min_corr)) :
        corr_koef = max_corr
        lag_rez = max_lag+1
    else:
        corr_koef = min_corr
        lag_rez = min_lag+1    

    if adf_statistic < crit_value_10_percent and adf_pvalue < 0.05 and (np.all(corr_koef < -0.5) or np.all(corr_koef > 0.5)):
        recommendation = "SARIMAX, LSTM, EG, SARIMAX periodas: {lag_rez}"
    elif adf_statistic >= crit_value_10_percent or adf_pvalue >= 0.05:
        recommendation = "Duomenys nestacionarūs, EG, LSTM"
    elif adf_statistic < crit_value_10_percent and adf_pvalue < 0.05:
        recommendation = "ARIMA, LSTM, EG"

    fig, ax1 = plt.subplots()
    ax1.stem(lag_acf)
    ax1.set_title('Autokoreliacijos funkcija')
    acf_url = get_graph_url(ax1.figure)

    fig, ax2 = plt.subplots()
    ax2.stem(lag_pacf)
    ax2.set_title('Dalines autokoreliacijos funkcija')
    pacf_url = get_graph_url(ax2.figure)

    fig, ax3 = plt.subplots()
    ax3.plot(diff_data)
    ax3.set_title('Diferencijuotos reikšmės')
    diff_url = get_graph_url(ax3.figure)

    fig, ax4 = plt.subplots()
    ax4.boxplot(diff_data)
    ax4.set_title('Duomenų pasisikirstymo grafikas')
    box_plot_url = get_graph_url(ax4.figure)

    session['recom_diff'] = recommendation

    result_dict = {
        'adf_statistic': round(adf_statistic, 4),
        'adf_pvalue': round(adf_pvalue, 4),
        'adf_crit_values': adf_crit_values,
        'corr_coef_val': round(corr_koef, 2),
        'corr_period_val': lag_rez,
        'acf_url': acf_url,
        'pacf_url': pacf_url,
        'diff_url': diff_url,
        'box_plot_url': box_plot_url,
        'recommendation': recommendation,
    }

    return result_dict