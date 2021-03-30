
"""
Data handlers
"""

import os
import json
import ntpath
import random
import pandas as pd
import numpy as np
from tqdm import tqdm


def load_data(data_root, ticker):
    price_data = pd.read_csv(os.path.join(data_root, "prices", ticker+".csv"))
    news_data = json.load(open(os.path.join(data_root, "news", ticker+".json"), "r"))
    news_dates = [d['date'] for d in news_data]
    news = [d['text'] for d in news_data]
    return price_data, news_dates, news


class DataLoader:

    def __init__(self, root, batch_size, device):
        if not (os.path.exists(os.path.join(root, "prices")) & os.path.exists(os.path.join(root, "news"))):
            raise NotImplementedError(f"Could not find 'prices' and 'news' in root ({root})")

        files = os.listdir(os.path.join(root, "prices"))
        self.root = root
        self.device = device
        self.batch_size = batch_size
        self.tickers = [ntpath.basename(f).split('.')[0] for f in files]
        self.data = {ticker: {"prices": [], "news": []} for ticker in self.tickers}
        self.ptr = 0

        # Align dates for prices data
        tickers = self.tickers.copy()
        all_price_data = [pd.read_csv(os.path.join(root, "prices", f+".csv")) for f in self.tickers]
        self.dates = self._align_price_data(all_price_data, tickers)
        self._compile_news_data()

    def _align_price_data(self, all_price_data, tickers):
        keep_cols = ["Date", "Low", "High", "Adj Close"]
        price_cols = ["Low", "High", "Adj Close"]
        maxlen_df = np.argmax([len(df) for df in all_price_data])
        df = all_price_data.pop(maxlen_df)
        tick = tickers.pop(maxlen_df)

        df["Date"] = pd.to_datetime(df["Date"])
        df.drop([c for c in df.columns if c not in keep_cols], axis=1, inplace=True)
        self.data[tick]["prices"] = df[price_cols].values.tolist()
        df.drop(price_cols, axis=1, inplace=True)

        for i in range(len(tickers)):
            temp_df = all_price_data[i]
            temp_df["Date"] = pd.to_datetime(temp_df["Date"])
            temp_df.drop([c for c in temp_df.columns if c not in keep_cols], axis=1, inplace=True)
            df = pd.merge(left=df, right=temp_df, on="Date", how="left")
            self.data[tickers[i]]["prices"] = df[price_cols].values.tolist()
            df.drop(price_cols, axis=1, inplace=True)
        return df["Date"].values

    def _compile_news_data(self):
        for t in self.tickers:
            js = json.load(open(os.path.join(self.root, "news", t+".json"), "r"))
            dates = pd.to_datetime(pd.Series([d['time'][3:] for d in js])).values
            texts = []
            for i in range(len(js)):
                if "text" in js[i].keys():
                    texts.append(js[i]["text"])
                else:
                    texts.append(js[i]["title"])
            texts = np.asarray(texts)
            for d in self.dates:
                locs = np.where(dates == d)[0]
                dtexts = texts[locs].tolist()
                self.data[t]["news"].append(dtexts)

    def __len__(self):
        t = list(self.data.keys())[0]
        return len(self.data[t]["prices"]) // self.batch_size

    def flow(self):
        prices, news = [], []
        for _ in range(self.batch_size):
            price_temp, news_temp = [], []
            for t in self.tickers:
                price_temp.append(self.data[t]["prices"][self.ptr])
                news_temp.append(self.data[t]["news"][self.ptr])
            prices.append(price_temp)
            news.append(news_temp)
            self.ptr += 1

            if self.ptr >= len(self.data[t]["prices"]):
                self.ptr = 0

        return torch.from_numpy(prices).float().to(self.device), news


def get_dataloaders(root, batch_size):
    pass
