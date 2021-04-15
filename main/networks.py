
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

    [NOTE]  This network is WIP. Need to add pretrained Pegasus and
            possibly pretrained BERT as well. Also going to work out the math to
            ensure the transformations for all stocks parallely has no issues.
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
        self.heads = config['heads']
        self.fc_transform = nn.Linear(config['hidden_size'], config['graph_size']//self.heads, bias=False)
        self.fc_attention = nn.Linear(2*config['graph_size']//self.heads, 1, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x, return_attn=False):
        x = self.fc_transform(x)
        concat_tensors = []
        for i in range(x.size(0)):
            for j in range(x.size(0)):
                concat_tensors.append(torch.cat([x[i], x[j]], dim=-1))
        concat_tensors = torch.cat(concat_tensors, dim=0).view(x.size(0), x.size(0), -1)
        attn_probs = self.softmax(self.relu(self.fc_attention(concat_tensors)))
        out = torch.einsum("nm,md->nd", attn_probs, x)

        if not return_attn:
            return out
        else:
            return out, attn_probs.detach().cpu().numpy()


class MultiHeadGraphAttention(nn.Module):
    """ Multihead extension of Graph Attention """

    def __init__(self, config):
        super().__init__()
        self.num_heads = config['heads']
        self.heads = nn.ModuleList([GraphAttention(config) for _ in range(self.num_heads)])
        self.bilinear = nn.Bilinear(config['hidden_size'], config['hidden_size'], config['hidden_size'])
        self.tanh = nn.Tanh()

    def forward(x_prices, x_news, return_attn=False):
        x = self.bilinear(x_prices, x_news)
        outputs, attn_probs = [], []
        for _ in range(self.num_heads):
            out, attn = self.heads[i](x)
            outputs.append(out)
            attn_probs.append(attn)

        outputs = self.tanh(torch.cat(outputs, dim=-1))
        if return_attn:
            return outputs
        else:
            return outputs, attn_probs


class Classifier(nn.Module):
    """ Linear classification network """

    def __init__(self, config):
        super().__init__()
        self.fc_out = nn.Linear(config['graph_size'], 2)
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        return self.log_softmax(self.fc_out(x))
