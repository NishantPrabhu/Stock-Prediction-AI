
""" 
Data handlers.

OHLC series for several companies should be available in the root
directory as CSV files named after the company's ticker. Column 
names will be date, open, high, low, close, adj_close.
adj_close will be used for forecasting by default. 
"""

import os
import torch
import numpy as np 
import pandas as pd 


class ForecastingDataLoader:

    def __init__(self, data, config, device):
        self.data = data                                                        # numpy array of adj_close values for a certain company
        self.batch_size = config['batch_size']                                  # number of history/horizon samples to provide
        self.horizon = config['horizon']                                        # number of points to forecast
        self.lookback = config['lookback_horizon_ratio'] * self.horizon         # number of points (as K * horizon) to use as history
        self.ptr = self.lookback
        self.device = device

    def __len__(self):
        return len(self.data - self.horizon - self.lookback) // self.batch_size

    def flow(self):
        X, y = [], []
        for i in range(self.batch_size):
            X.append(self.data[self.ptr-self.lookback: self.ptr])
            y.append(self.data[self.ptr: self.ptr+self.horizon])
            self.ptr += 1

            if self.ptr > len(self.data)-self.horizon:
                self.ptr = self.lookback

        return torch.FloatTensor(X).squeeze(-1).to(self.device), torch.FloatTensor(y).squeeze(-1).to(self.device)


def get_dataloaders(root, loader_config, horizon, device):
    data = load_data(root, loader_config['ticker'])
    val_size = loader_config['forecast_horizon_ratio'] * horizon
    train_data, val_data = data[:len(data)-val_size], data[-val_size:]
    train_loader = ForecastingDataLoader(train_data, loader_config, device)
    train_data = torch.FloatTensor(train_data)
    val_data = torch.FloatTensor(val_data)
    return train_loader, train_data, val_data


def load_data(path, ticker):
    table = pd.read_csv(os.path.join(path, ticker+'.csv'))
    col = [c for c in table.columns if 'adj' in c.lower()]
    seq = table[col].diff().values.reshape(-1,)[1:]
    return seq


# Checking loader functioning
if __name__ == "__main__":

    data = np.arange(1, 40, 1)
    config = {
        'batch_size': 3,
        'horizon': 5,
        'lookback_horizon_ratio': 2 
    }

    ldr = ForecastingDataLoader(data, config)
    for _ in range(10):
        x, y = ldr.flow()
        print(x)
        print(y)
        print() 
