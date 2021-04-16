
"""
Network definitions.
Individual components of the final model defined here.
"""

import bert
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers


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
        self.upscale_fc = nn.Linear(3, config['model_dim'], bias=True)
        self.encoder = bert.Encoder(config)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.upscale_fc(x))                                   # (num_stocks, lookback, model_dim)
        x = self.encoder(x)                                                 # (num_stocks, lookback, model_dim)
        x = self.avgpool(x.permute(0, 2, 1).contiguous()).squeeze(-1)       # (num_stocks, model_dim)
        return x


class NewsEncoder(nn.Module):
    """
    Encodes historical news data of multiple stocks into
    features vectors using self attention.

    Input:
        List of tensors of news data tokens (from transformers.BertTokenizer)
    Output:
        Tensor of encoded features (num_stocks, hidden_size)
    """
    def __init__(self, device):
        super().__init__()
        self.encoder = transformers.BertModel.from_pretrained("bert-base-uncased")
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.relu = nn.ReLU()
        self.device = device

    def forward(self, x):
        """ 
        Input is a list of tensors with input vocab tokens from BertTokenizer. Sequence 
        lengths could be different for different elements of the list, so iterating over 
        them to generate individual outputs. 

        For instance if for past seven days, the 10 stocks have [N1, N2, ..., N10] news
        articles, then after extracting the embedding of [CLS] for each output we get 
        tensors of size (1, Nx, model_dim) where x = {1,2,...,10}. Each of these tensors 
        is average pooled along Nx dimension and then concatenated along batch dimension 
        to finally get (10, model_dim).
        """
        outputs = []
        for company_news in x:
            temp = []
            for tokens in company_news:
                tokens = tokens.to(self.device)
                out = self.encoder(tokens)
                temp.append(out[1])
            
            temp = torch.cat(temp, dim=0)
            out = self.avgpool(temp.unsqueeze(0).permute(0, 2, 1)).squeeze(-1)
            outputs.append(out)
        outputs = torch.cat(outputs, dim=0)
        return outputs


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
        self.heads = config['num_heads']
        self.fc_transform = nn.Linear(config['hidden_dim'], config['graph_dim']//self.heads, bias=False)
        self.fc_attention = nn.Linear(2*config['graph_dim']//self.heads, 1, bias=False)
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
        self.num_heads = config['num_heads']
        self.heads = nn.ModuleList([GraphAttention(config) for _ in range(self.num_heads)])
        self.bilinear = nn.Bilinear(config['prices_dim'], config['news_dim'], config['hidden_dim'])
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
        self.fc_out = nn.Linear(config['graph_dim'], 2)
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        return self.log_softmax(self.fc_out(x))