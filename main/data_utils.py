
"""
Data handlers
"""

import os
import json
import torch
import pickle
import ntpath
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import BertTokenizer


def load_data(root):
    with open(os.path.join(root, "prices_aligned.pkl"), "rb") as f:
        prices = pickle.load(f)
    with open(os.path.join(root, "news_aligned.pkl"), "rb") as f:
        news = pickle.load(f)
    return prices, news 


class DataLoader:

    def __init__(self, price_data, news_data, lookback_length):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.prices, self.news = price_data, news_data
        self.tickers = self.news.keys()
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.lookback = lookback_length
        self.ptr = 0

    def __len__(self):
        keys = list(self.prices.keys())
        return len(self.prices[keys[0]]) - self.lookback 

    def flow(self):
        prices, news, targets = [], [], []

        for i in range(self.lookback):
            prices_temp, news_temp = [], []
            for t in self.tickers:
                prices_temp.append(self.prices[t][self.ptr+i])
                news_data = self.news[t][self.ptr+i]
                input_tokens = self.tokenizer(news_data, padding=True, truncation=True, return_tensors='pt')['input_ids']
                news_temp.append(input_tokens)

            prices.append(prices_temp)
            news.append(news_temp)

        for t in self.tickers:
            targets.append(1 if self.prices[t][self.ptr+self.lookback][-1] > 1.0 else 0)                
        
        self.ptr += 1
        if self.ptr >= (len(self.prices) - self.lookback):
            self.ptr = 0

        prices, targets = np.array(prices), np.array(targets)
        prices = torch.from_numpy(prices).float().permute(1, 0, 2).to(self.device)
        targets = torch.from_numpy(targets).long().to(self.device)
        return prices, news, targets


def get_dataloaders(root, test_size, lookback_length):
    prices, news = load_data(root)
    keys = list(prices.keys())
    train_size = int((1 - test_size) * len(prices[keys[0]]))
    train_prices, train_news = {}, {}
    test_prices, test_news = {}, {}

    for t in prices.keys():
        train_prices[t], test_prices[t] = prices[t][:train_size], prices[t][train_size:]
        train_news[t], test_news[t] = news[t][:train_size], news[t][train_size:]

    train_loader = DataLoader(train_prices, train_news, lookback_length)
    test_loader = DataLoader(test_prices, test_news, lookback_length)
    return train_loader, test_loader


if __name__ == "__main__":

    loader = DataLoader(root="../data", lookback_length=7)
    prices, news, targets = loader.flow()

    print(f"Prices: {prices.size()}")
    print(f"Targets: {targets.size()}")
    print(targets)