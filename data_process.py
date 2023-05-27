import pandas as pd
import numpy as np
from io import StringIO
from statsmodels.tsa.stattools import adfuller
import matplotlib
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from graph_generator import get_graph_url
from flask import Flask, session
import statsmodels.api as sm

def process_data(fileData):
    matplotlib.use('Agg')
    csv_data = StringIO('\n'.join(fileData))

    df = pd.read_csv(csv_data, names=['date', 'value'])
    x = df['value'].to_numpy()
    y = df.index.to_numpy()
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)

    mean_val = df['value'].mean()
    median_val = df['value'].median()
    std_val = df['value'].std()
    rolling_mean = df['value'].rolling(window=10).mean().tolist()
    min_val = df['value'].min()
    max_val = df['value'].max()

    time_range = np.arange(len(df))
    trend_coef = np.polyfit(time_range, df['value'], 1)[0]

    adf_result = adfuller(df['value'])
    adf_statistic = adf_result[0]
    adf_pvalue = adf_result[1]
    adf_lags = adf_result[2]
    adf_crit_values = adf_result[4]
    crit_value_10_percent = adf_result[4]['10%']

    q25, q50, q75 = np.percentile(df['value'], [25, 50, 75])

    fig, ax1 = plt.subplots()
    plot_acf(df['value'], lags=6, ax=ax1)
    ax1.set_title('ACF grafikas')
    plt.tight_layout()
    acf_url = get_graph_url(ax1.figure)

    fig, ax2 = plt.subplots()
    plot_pacf(df['value'], lags=6, ax=ax2)
    ax2.set_title('PACF grafikas')
    plt.tight_layout()
    pacf_url = get_graph_url(ax2.figure)

    fig, ax3 = plt.subplots()
    ax3.boxplot(df['value'])
    ax3.set_xlabel('Reikšmės')
    ax3.set_ylabel('Vertės')
    ax3.set_title('Duomenų pasiskirstymo grafikas')
    box_plot_url = get_graph_url(ax3.figure)

    df['MA'] = df['value'].rolling(window=7).mean().tolist()

    fig, ax4 = plt.subplots()
    plt.plot(df['value'], label='Reikšmė')
    plt.plot(df['MA'], label='Slankusis vidurkis')
    plt.legend()
    moving_average_url = get_graph_url(ax4.figure)

    fig, ax5 = plt.subplots()
    pd.plotting.autocorrelation_plot(df['value'], ax=ax5)
    ax5.set_title('Autokoreliacijos grafikas')
    plt.tight_layout()
    autocorr_url = get_graph_url(ax5.figure)
    lags = ax5.lines[1].get_xdata()

    corr_rez = sm.tsa.acf(df['value'], nlags=12)
    corr_rezz = corr_rez[1:]
    max_corr = np.max(corr_rezz)
    max_lag = np.argmax(corr_rezz)
    min_corr = np.min(corr_rezz)
    min_lag = np.argmin(corr_rezz)
    if (abs(max_corr) >= abs(min_corr)) :
        corr_koef = max_corr
        lag_rez = max_lag+1
        corr_koef_index = np.where(corr_rezz == max_corr)[0][0]+1
    else:
        corr_koef = min_corr
        lag_rez = min_lag+1
        corr_koef_index = np.where(corr_rezz == min_corr)[0][0]+1

    fig, ax6 = plt.subplots()
    plt.plot(corr_rez)
    ax6.set_title('Koreliacijos funkcijos grafikas')
    plt.tight_layout()
    corr_url = get_graph_url(ax6.figure)

    if adf_statistic < crit_value_10_percent and adf_pvalue < 0.05 and (corr_koef < -0.5 or corr_koef > 0.5):
        recommendation = f"SARIMAX, LSTM, EG, SARIMAX periodas: {corr_koef_index}"
    elif adf_statistic >= crit_value_10_percent or adf_pvalue >= 0.05:
        recommendation = "Siūlomas diferencijavimas"
    elif adf_statistic < crit_value_10_percent and adf_pvalue< 0.05:
        recommendation = "ARIMA, LSTM, EG"

    session['recom'] = recommendation

    result_dict = {
        'adf_statistic': round(adf_statistic, 4),
        'adf_pvalue': round(adf_pvalue, 4),
        'adf_lags': round(adf_lags, 4),
        'adf_crit_values': adf_crit_values,
        'corr_val': round(corr_koef, 4),
        'mean_val': round(mean_val, 4),
        'median_val': round(median_val, 4),
        'std_val': round(std_val**2, 4),
        'min_val': round(min_val, 4),
        'max_val': round(max_val, 4),
        'trend_coef': round(trend_coef, 4),
        'q25': round(q25, 4),
        'q50': round(q50, 4),
        'q75': round(q75, 4),
        'corr_coef_val': corr_koef_index,
        'acf_url': acf_url,
        'pacf_url': pacf_url,
        'box_plot_url': box_plot_url,
        'autocorr_url': autocorr_url,
        'corr_url': corr_url,
        'moving_average_url': moving_average_url,
        'recommendation': recommendation
    }

    return result_dict