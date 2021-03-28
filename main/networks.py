
"""
Network definitions.
Individual components of the final model defined here.
"""

import bert
import torch
import torch.nn as nn
import torch.nn.functional as F


class PriceEncoder(nn.Module):
    """
    Encodes historical prices of multiple stocks into
    feature vectors using temporal attention.

    Input:
        Tensor of historical prices (num_stocks, lookback_length, feature_size)
    Output:
        Tensor of encoded features (num_stocks, hidden_size)
    """
    def __init__(self, config):
        super().__init__()
        self.encoder = bert.Encoder(config['encoder'])
        self.conv = nn.Conv1d(config['lookback'], 1, kernel_size=3, stride=1, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.encoder(x)                     # (num_stocks, lookback_length, hidden_size)
        x = self.conv(x)                        # (num_stocks, 1, hidden_size)
        x = self.relu(x.squeeze(-2))            # (num_stocks, hidden_size)
        return x


class NewsEncoder(nn.Module):
    """
    Encodes historical news data of multiple stocks into
    features vectors using self attention.

    Input:
        Tensor of news data embeddings (num_stocks, lookback_length, embedding_size)
    Output:
        Tensor of encoded features (num_stocks, hidden_size)
    """
    def __init__(self, config):
        super().__init__()
        self.encoder = bert.Encoder(config['encoder'])
        self.conv = nn.Conv1d(config['lookback'], 1, kernel_size=3, stride=1, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.encoder(x)                     # (num_stocks, lookback_length, hidden_size)
        x = self.conv(x)                        # (num_stocks, 1, hidden_size)
        x = self.relu(x.squeeze(-2))            # (num_stocks, hidden_size)
        return x


class GraphAttention(nn.Module):
    """
    Graph attention network to capture relations between encoded features
    of stocks related to each other.

    Input:
        Encoded price and news features of size (num_stocks, hidden_size) each
    Output:
        Feature tensor of size (num_stocks, feature_size)
    """
    def __init__(self, config):
        super().__init__()
        pass

    def forward(self, x_prices, x_news):
        pass
