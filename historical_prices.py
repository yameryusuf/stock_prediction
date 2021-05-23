#!/usr/bin/env python 
"""
Retrieve intraday stock data from Google Finance.
"""
import numpy as np
import os
import pathlib
from dateutil.relativedelta import *
import time
import pandas as pd
import datetime
import json
from ta import add_all_ta_features
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from train_RNN import train_model
from test_RNN import test_model
from conversion_file import financial_columns

plt.close("all")
today_date = datetime.datetime.today().date()
historical = {'startmonth': (today_date.month % 12) + 1, 'startyear': today_date.year - 2,
              'endmonth': today_date.month - 1,
              'endyear': today_date.year}
current_month = {'startmonth': today_date.month, 'startyear': today_date.year, 'endmonth': today_date.month,
                 'endyear': today_date.year}
uri = 'https://www.alphavantage.co/query?'

counter = 0;
arg = 'MSFT';
symbol_add = False;
symbol_update = True;
function = 'Intraday_(Extended_History)';
interval = '60min';
time_period = '60';
series_type = 'close';
output_size = 'full';
skipfigure = True
timestep = 50
epochs = 10
test_case = 'extended'
feature_list = \
    ["volume_adi",
     "volume_obv",
     "volatility_kcw",
     "volatility_atr",
     ]


def process_times(startmonth, startyear, endmonth, endyear):
    now = datetime.date(datetime.datetime.now().year, datetime.datetime.now().month, 1)
    startdate = datetime.date(startyear, startmonth, 1)
    enddate = datetime.date(endyear, endmonth, 1)
    if now - startdate > datetime.timedelta(days=730) or enddate - now > datetime.timedelta(days=0):
        raise ValueError("Date range out of bounds")
    else:
        relative_months = range(
            relativedelta(enddate, startdate).months + 1 + relativedelta(enddate, startdate).years * 12)
        return [((relativedelta(now, dates).months + 1) % 13, (relativedelta(now, dates).years + 1)) for dates in
                (startdate + relativedelta(months=+n) for n in relative_months)]


def function_call(function, parameters):
    with open('functions.txt') as json_file:
        functions = json.load(json_file)
        cf = functions[function]
        for i in parameters.keys():
            cf[i] = parameters[i]
        url_call = uri
        for i in cf.keys():
            if cf[i] != i:
                url_call += (i + '=' + cf[i] + '&')

        url_call = url_call[0:len(url_call) - 1]
        df = pd.read_csv(url_call)
        return df


def get_indicator_metrics(functions, parameters, counter):
    function_dfs = {}
    for function in functions:
        files = pathlib.Path(function + '_' + parameters['symbol'] + '.csv')
        exists = False
        if files.exists():
            modified_date = datetime.datetime.fromtimestamp(files.stat().st_mtime).date()
            today_date = datetime.datetime.today().date()
            if modified_date == today_date:
                df = pd.read_csv(function + '_' + parameters['symbol'] + '.csv')
                exists = True
        if not exists:
            print("current function call:" + function + ",symbol:" + parameters['symbol'] + ",counter:" + str(counter))
            df = function_call(function, parameters)
            df.to_csv(function + '_' + parameters['symbol'] + '.csv')
            counter = counter + 1
        if counter % 5 == 0:
            time.sleep(65)
        function_dfs[function] = df
    return function_dfs


def get_historical_data(function, parameters, period, counter):
    if function == "Daily":
        date_range = [[1, 1]]
    else:
        date_range = process_times(period['startmonth'], period['startyear'], period['endmonth'], period['endyear'])
    # testing - date_range = [[1,1],[1,2]]
    net_data = []
    for period in date_range:
        counter = counter + 1
        parameters["slice"] = "year" + str(period[1]) + 'month' + str(period[0])
        print("current function call:" + function + ",symbol:" + parameters['symbol'] + ",slice:" + parameters[
            "slice"] + ",counter:" + str(counter))
        df = function_call(function, parameters)
        header = list(df.columns.values)
        readings = df.values.tolist()
        if counter % 5 == 0:
            time.sleep(65)
        net_data.extend(readings)
    final_data = pd.DataFrame(net_data, columns=header)
    final_data.rename(columns={final_data.columns[0]: "time"}, inplace=True)
    final_data = final_data[['time', 'close', 'open', 'high', 'low', 'volume']]
    final_data = final_data.set_index(final_data.columns[0])
    final_data.index = pd.to_datetime(final_data.index)
    return final_data.sort_index()


def save_data(generic_indicatior_parameters):
    historical_data_csv = generic_indicatior_parameters['symbol'] + '/data' + '/' + function + '_base_historical_data' \
                                                                                               '.csv '
    exists = False
    if pathlib.Path(historical_data_csv).exists():
        modified_date = datetime.datetime.fromtimestamp(pathlib.Path(historical_data_csv).stat().st_mtime).date()
        historical_data = pd.read_csv(historical_data_csv, index_col=0)
        historical_data.index = pd.to_datetime(historical_data.index)
        if modified_date == today_date:
            exists = True
        else:
            historical_data_current = get_historical_data(function=function,
                                                          parameters=generic_indicatior_parameters,
                                                          period=historical,
                                                          counter=counter).reset_index()
            data = historical_data.values.tolist()
            data.extend(historical_data_current.values.tolist())
            historical_data = pd.DataFrame.from_records(data, columns=historical_data_current.columns.values)
            historical_data.set_index('time')
            historical_data.drop_duplicates(subset=['time'], keep='first')
            historical_data.index = pd.to_datetime(historical_data.index)
            historical_data.to_csv(
                generic_indicatior_parameters['symbol'] + '/data' + '/' + function + '_base_historical_data.csv')
    if not exists:
        historical_data = get_historical_data(function=function,
                                              parameters=generic_indicatior_parameters,
                                              period=historical, counter=counter)
        historical_data.to_csv(
            generic_indicatior_parameters['symbol'] + '/data' + '/' + function + '_base_historical_data.csv')
    return historical_data


def plot_chart(extended_historical_data, symbol_folder, function):
    for i, col in enumerate(extended_historical_data.columns):
        print("loading figure: " + col)
        plt.figure(figsize=(10, 8), dpi=100)
        extended_historical_data[col].plot(fig=plt.figure(i), x_compat=True)
        plt.xlabel("Dates")
        plt.setp(plt.gca().xaxis.get_majorticklabels(), rotation=90)
        if function == 'Daily':
            plot_interval = 365
        else:
            plot_interval = 20
        plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=plot_interval))
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%d-%m-%Y"))
        plt.title(col)
        fig = plt.gcf()
        plt.tight_layout()
        print("saving figure: " + col)
        fig.savefig(symbol_folder + '/plots/' + str(i) + '_' + col + '.png', dpi=100)
        plt.close()


def extend_data_with_financials(historical_data, symbol_folder, feature_list):
    extended_historical_data_csv = symbol_folder + '/data' + '/' + function + '_extended_historical_data.csv'
    if pathlib.Path(extended_historical_data_csv).exists():
        modified_date = datetime.datetime.fromtimestamp(pathlib.Path(extended_historical_data_csv).stat().st_mtime).date()
        if modified_date != today_date:
            extended_historical_data = add_all_ta_features(historical_data, open='open', close='close', high='high',
                                                           low='low', volume='volume')
            extended_historical_data.index = pd.to_datetime(extended_historical_data.index).values.astype(
                np.int64) // 10 ** 6
            extended_historical_data.to_csv(
                symbol_folder + '/data' + '/' + function + '_extended_historical_data.csv')
        else:
            extended_historical_data = pd.read_csv(extended_historical_data_csv, index_col=0)
    else:
        extended_historical_data = add_all_ta_features(historical_data, open='open', close='close', high='high',
                                                       low='low', volume='volume')
        extended_historical_data.index = pd.to_datetime(extended_historical_data.index)
        extended_historical_data.to_csv(symbol_folder + '/data' + '/' + function + '_extended_historical_data.csv')
    extended_historical_data.fillna(0, inplace=True)
    extended_historical_data = extended_historical_data[['close', 'open', 'high', 'low', 'volume'] + feature_list]
    return extended_historical_data


def create_symbol_folders(generic_indicatior_parameters):
    pathlib.Path(generic_indicatior_parameters['symbol']).mkdir(
        parents=True, exist_ok=True)
    pathlib.Path(generic_indicatior_parameters['symbol'] + '/plots').mkdir(
        parents=True, exist_ok=True)
    pathlib.Path(generic_indicatior_parameters['symbol'] + '/data').mkdir(
        parents=True, exist_ok=True)
    pathlib.Path(generic_indicatior_parameters['symbol'] + '/models').mkdir(
        parents=True, exist_ok=True)


def main():
    directories = next(os.walk('.'))[1]
    symbols = [i for i in directories if i not in ['.git', '.idea', 'venv', '__pycache__']]
    if symbol_add:
        symbols.append(arg)
    elif symbol_update:
        symbols = [arg]
    for symbol in symbols:
        generic_indicatior_parameters = {'symbol': symbol,
                                         'interval': interval,
                                         'time_period': time_period,
                                         'datatype': 'csv',
                                         'series_type': series_type,
                                         'outputsize': output_size,
                                         'apikey': '1A673HM6LF1CZG2I'
                                         }
        create_symbol_folders(generic_indicatior_parameters)
        historical_data = save_data(generic_indicatior_parameters)
        extended_historical_data = extend_data_with_financials(historical_data, generic_indicatior_parameters['symbol'],
                                                               feature_list)
        if not skipfigure:
            plot_chart(extended_historical_data, function)
        if test_case == 'base':
            features = []
            feature_count = len(historical_data.columns) - 1
        elif test_case == 'extended':
            features = feature_list
            feature_count = len(extended_historical_data.columns) - 1
        train_model(generic_indicatior_parameters['symbol'], function + '_' + test_case + '_historical_data.csv',
                         timestep, feature_count, epochs, ['time','close', 'open', 'high', 'low', 'volume'] + features)
        test_model(generic_indicatior_parameters['symbol'], timestep, feature_count,function, interval)


if __name__ == "__main__":
    main()
