import pandas as pd
import numpy as np
from bs4 import BeautifulSoup as soup
from urllib.request import Request, urlopen

pd.set_option('display.max_colwidth', 25)

# Set up scraper
def get_html(symbol):
    url = ("http://finviz.com/quote.ashx?t=" + symbol.lower())
    req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    webpage = urlopen(req).read()
    html = soup(webpage, "html.parser")
    return html


def get_fundamentals(symbol):
    try:
        # Find fundamentals table
        html = get_html(symbol)
        fundamentals = pd.read_html(str(html), attrs={'class': 'snapshot-table2'})[0]

        # Clean up fundamentals dataframe
        fundamentals.columns = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11']
        colOne = []
        colLength = len(fundamentals)
        for k in np.arange(0, colLength, 2):
            colOne.append(fundamentals[f'{k}'])
        attrs = pd.concat(colOne, ignore_index=True)

        colTwo = []
        colLength = len(fundamentals)
        for k in np.arange(1, colLength, 2):
            colTwo.append(fundamentals[f'{k}'])
        vals = pd.concat(colTwo, ignore_index=True)

        fundamentals = pd.DataFrame()
        fundamentals['Attributes'] = attrs
        fundamentals['Values'] = vals
        fundamentals = fundamentals.set_index('Attributes')
        return fundamentals

    except Exception as e:
        return e


def get_news(symbol):
    try:
        # Find news table
        html = get_html(symbol)
        news = pd.read_html(str(html), attrs={'class': 'fullview-news-outer'})[0]
        links = []
        for a in html.find_all('a', class_="tab-link-news"):
            links.append(a['href'])

        # Clean up news dataframe
        news.columns = ['Date', 'News Headline']
        news['Article Link'] = links
        news = news.set_index('Date')
        return news

    except Exception as e:
        return e


def get_insider(symbol):
    try:
        # Find insider table
        html = get_html(symbol)
        insider = pd.read_html(str(html), attrs={'class': 'body-table'})[0]

        # Clean up insider dataframe
        insider = insider.iloc[1:]
        insider.columns = ['Trader', 'Relationship', 'Date', 'Transaction', 'Cost', '# Shares', 'Value ($)',
                           '# Shares Total', 'SEC Form 4']
        insider = insider[
            ['Date', 'Trader', 'Relationship', 'Transaction', 'Cost', '# Shares', 'Value ($)', '# Shares Total',
             'SEC Form 4']]
        insider = insider.set_index('Date')
        return insider

    except Exception as e:
        return e