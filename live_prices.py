import requests
import os
import pathlib
import csv
import datetime
import schedule
import sys
import time

directories = next(os.walk('.'))[1]
symbols = [i for i in directories if i not in ['.git', '.idea', 'venv']]
parameters = {'token': 'c2hc30qad3ifd59bm8m0', 'symbol': 'symbol'}
def call_scheduled_price_dump():
    for tick in symbols:
        parameters['symbol'] = tick
        query = 'https://finnhub.io/api/v1/quote?symbol={symbol}&token={token}'.format(symbol=parameters['symbol'],
                                                                                       token=parameters['token'])
        json_prices = requests.get(query).json()
        current_price = list(json_prices.values())
        current_price[5] = datetime.datetime.fromtimestamp(int(current_price[5])).strftime('%Y-%m-%d %H:%M:%S')
        symbol_file = tick + '/live_prices.csv'
        print(current_price)
        if not pathlib.Path(symbol_file).exists():
            file = open(symbol_file,'w+', newline='')
            header = ['close', 'high', 'low', 'open', 'previous close', 'time']
            write = csv.writer(file)
            write.writerows([header])
            write.writerows([current_price])
            file.close()
        else:
            file = open(symbol_file, "r")
            last_price = file.readlines()[-1]
            file.close()
            if last_price.split(',')[5][:-1] != str(current_price[5]):
                file = open(symbol_file,'a+', newline='')
                write = csv.writer(file)
                write.writerows([current_price])
                file.close()

def exit_script():
    for tick in symbols:
        symbol_file = tick + '/live_prices.csv'
        os.remove(symbol_file)
    sys.exit(0)

while True:
    schedule.every().day.at("20:30").do(exit_script)
    call_scheduled_price_dump()
    time.sleep(60)

