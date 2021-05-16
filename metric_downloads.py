#!/usr/bin/env python 
"""
Retrieve intraday stock data from Google Finance.
"""

import csv
import datetime
import re
import pathlib
from dateutil.relativedelta import *
import time
import pandas as pd
import requests
import datetime
import json

uri = 'https://www.alphavantage.co/query?'


def process_times(startmonth, startyear, endmonth, endyear):
    now = datetime.date(datetime.datetime.now().year, datetime.datetime.now().month, 1)
    startdate = datetime.date(startyear, startmonth, 1)
    enddate = datetime.date(endyear, endmonth, 1)
    if now - startdate > datetime.timedelta(days=730) or enddate - now > datetime.timedelta(days=0):
        raise ValueError("Date range out of bounds")
    else:
        return [((relativedelta(now, dates).months + 1) % 13, (relativedelta(now, dates).years + 1)) for dates in
                (startdate + relativedelta(months=+n) for n in range(relativedelta(enddate, startdate).months + 1))]


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


def get_indicator_metrics(functions, parameters,counter):
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


def get_historical_intraday_data(parameters, startmonth, startyear, endmonth, endyear, counter):
    date_range = process_times(startmonth, startyear, endmonth, endyear)
    net_data = []
    historical_data_csv = pathlib.Path('historical_data.csv')
    exists = False
    if historical_data_csv.exists():
        modified_date = datetime.datetime.fromtimestamp(historical_data_csv.stat().st_mtime).date()
        today_date = datetime.datetime.today().date()
        if modified_date == today_date:
            final_data = pd.read_csv('historical_data.csv')
            exists = True
    if not exists:
        for period in date_range:
            counter = counter + 1
            parameters["slice"] = "year" + str(period[1]) + 'month' + str(period[0])
            print("current function call:Intraday_(Extended_History),symbol:" + parameters['symbol'] + ",slice:" + parameters["slice"] + ",counter:" + str(counter))
            df = function_call("Intraday_(Extended_History)", parameters)
            header = list(df.columns.values)
            readings = df.values.tolist()
            if counter % 5 == 0:
                time.sleep(65)
            net_data.extend(readings)
        final_data = pd.DataFrame(net_data, columns=header)
        final_data.to_csv('historical_data.csv')
    return final_data.sort_values(by=['time'])


def main():
    counter = 0
    generic_indicatior_parameters = {'symbol': 'PLTR',
                                     'interval': '15min',
                                     'time_period': '60',
                                     'datatype': 'csv',
                                     'series_type': 'close',
                                     'apikey': '1A673HM6LF1CZG2I'
                                     }
    get_historical_intraday_data(
        parameters=generic_indicatior_parameters, startmonth=9,
        startyear=2020, endmonth=5, endyear=2021,counter=counter)
    time.sleep(65)
    metric_dfs = get_indicator_metrics(
        functions=["Earnings", "Income_Statement", "Balance_Sheet", "Cash_Flow", "SMA_High_Usage", "EMA_High_Usage",
                   "T3",
                   "STOCH_High_Usage", "RSI_High_Usage", "ADX_High_Usage", "CCI_High_Usage", "AROON_High_Usage",
                   "BBANDS_High_Usage", "ULTOSC", "HT_TRENDLINE"], parameters=generic_indicatior_parameters, counter=counter)


if __name__ == "__main__":
    main()
