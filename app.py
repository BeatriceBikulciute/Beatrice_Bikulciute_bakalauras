from flask import Flask, render_template, request, flash, redirect, session, jsonify
from flask_caching import Cache
from data_process import process_data
from differentiate import differentiate_data
from dlt_algorithm import dlt_data
from arima_algorithm import arima_data
from sarimax_algorithm import sarimax_data
from lstm_algorithm import lstm_data
import matplotlib.pyplot as plt

app = Flask(__name__)
app.secret_key = '*2(mf8^$4)_)$)DS2#$F'

cache = Cache(app)

@app.route('/')
def index():
    return redirect('/upload')

@app.route('/upload', methods=['GET', 'POST'])
@cache.cached(timeout=3600)
def upload():
    session.clear()

    if request.method == 'POST':
        if 'selectedFile' not in request.files:
            return redirect(request.url)
        csv_file = request.files['selectedFile']
        if csv_file.filename == '':
           return redirect(request.url)

        data = csv_file.read().decode('utf-8').splitlines()
        first_data = data[:6]
        remaining_data = data[6:]

        session['original_data'] = first_data
        session['data'] = remaining_data

        return redirect('/data_process')

    return render_template('upload.html')

@app.route('/data_process', methods=['GET'])
@cache.cached(timeout=3600)
def data_process():
    if 'data' not in session:
        return redirect('/upload')

    data = session['data']
    result_dict1 = process_data(data)
    return render_template('data_process.html', result_dict=result_dict1)

    try:
        data = session['data']
        result_dict = process_data(data)
        return render_template('data_process.html', result_dict=result_dict)
    except Exception as e:
        flash(f'Error processing file: {str(e)}')
        return redirect('/upload')

@app.route('/differentiate', methods=['GET', 'POST'])
@cache.cached(timeout=3600)
def differentiate():
    if request.method == 'GET':

        data = session['data']
        result_dict2 = differentiate_data(data)

        return render_template('differentiate.html', result_dict=result_dict2)

@app.route('/prognosis', methods=['GET', 'POST'])
@cache.cached(timeout=3600)
def prognosis():
    if request.method == 'POST':
        prognosis_num = request.form.get('prognosis_num')
        sarimax_num = request.form.get('sarimax_num')
        ar = request.form.get('ar')
        ma = request.form.get('ma')
        sar = request.form.get('sar')
        sma = request.form.get('sma')
        epoch = request.form.get('epoch')
        neur_sk = request.form.get('neur_sk')
        trend_type = request.form.get('trend_type')

        if request.form.get('checkbox') is None:
            checkbox = 0
        else:
            checkbox = 1

        if request.form.get('period') == 'month':
            period = 'MS'
        elif request.form.get('period') == 'day':
            period = 'D'
        else:
            period = 'MS'

        if request.form.get('type') == 'add':
            type_t = 'add'
        elif request.form.get('type') == 'mul':
            type_t = 'add'
        else:
            type_t = 'add'  

        data, date_diff_data, original_data, = get_data()
        recom, recom_diff = get_recommendation()

        if request.form['dropdown'] == 'option_dlt':
            result_dict3 = dlt_data(
                data, date_diff_data, original_data, prognosis_num, checkbox, period, type_t)
            return render_template('prognosis.html', result_dict=result_dict3, recom=recom, recom_diff=recom_diff)

        elif request.form['dropdown'] == 'option_arima':
            result_dict3 = arima_data(
                data, date_diff_data, original_data, prognosis_num, checkbox, period, ar, ma)
            return render_template('prognosis.html', result_dict=result_dict3, recom=recom, recom_diff=recom_diff)

        elif request.form['dropdown'] == 'option_sarimax':
            result_dict3 = sarimax_data(
                data, date_diff_data, original_data, prognosis_num, checkbox, period, ar, ma, sar, sma, sarimax_num)
            return render_template('prognosis.html', result_dict=result_dict3, recom=recom, recom_diff=recom_diff)

        elif request.form['dropdown'] == 'option_lstm':
            result_dict3 = lstm_data(
                data, date_diff_data, original_data, prognosis_num, checkbox, period, epoch, neur_sk)
            return render_template('prognosis.html', result_dict=result_dict3, recom=recom, recom_diff=recom_diff)

    if request.method == 'GET':
        data, date_diff_data, original_data = get_data()
        recom, recom_diff = get_recommendation()

    return render_template('prognosis.html', recom=recom, recom_diff=recom_diff)

def get_data():
    data = session['data']
    if 'date_diff_data' not in session:
        differentiate_data(data)
    date_diff_data = session['date_diff_data']
    original_data = session['original_data']
    return data, date_diff_data, original_data

def get_recommendation():
    recom_diff = session['recom_diff']
    recom = session['recom']
    return recom, recom_diff

if __name__ == '__main__':
    app.run(debug=True)
