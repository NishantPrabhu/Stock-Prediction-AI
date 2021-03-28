
"""
Data handlers
"""

import os
import json
import pandas as pd
import numpy as np


def load_data(data_root, ticker):
    price_data = pd.read_csv(os.path.join(data_root, "prices", ticker+".csv"))
    news_data = json.load(open(os.path.join(data_root, "news", ticker+".json"), "r"))
    price_dates = price_data['date'].values.tolist()
    prices = price_data['adj_close'].values.tolist()
    news_dates = [d['date'] for d in news_data]
    news = [d['text'] for d in news_data]
    return price_dates, prices, news_dates, news


def align_data(price_dates, prices, news_dates, news):
    dates, prices, news = [], [], []
    


class DataLoader:

    def __init__(self, root, batch_size, shuffle):
        if not "tickers.txt" in os.listdir(root):
            raise NotImplementedError(f"Could not find tickers.txt in {root}")
        else:
            with open(os.path.join(root, "tickers.txt"), "r") as f:
                self.tickers = f.read().split("\n")

        all_price_dates, all_prices = [], []
        all_news_dates, all_news = [], []
        for t in self.tickers:
            price_dates, prices, news_dates, news = load_data(root, t)
            all_price_dates.append(price_dates)
            all_prices.append(prices)
            all_news_dates.append(news_dates)
            all_news.append(news)
