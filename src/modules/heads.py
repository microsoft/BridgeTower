import torch
import torch.nn as nn
import torch.nn.functional as F

from .bert_model import BertPredictionHeadTransform

class LinkTower(nn.Module):
    def __init__(self, config):
        super(LinkTower, self).__init__()
        self.LayerNorm = nn.LayerNorm(config['hidden_size'])

    def forward(self, hidden_states, cross_modal_hidden_states):
        return self.LayerNorm(hidden_states + cross_modal_hidden_states)


class Pooler(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class ITCHead(nn.Module):
    def __init__(self, hidden_size, embed_size):
        super().__init__()
        self.fc = nn.Linear(hidden_size, embed_size)

    def forward(self, x):
        return self.fc(x)


class ITMHead(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x):
        return self.fc(x)


class MLMHead(nn.Module):
    def __init__(self, config, weight=None):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        if weight is not None:
            self.decoder.weight = weight

    def forward(self, x):
        x = self.transform(x)
        x = self.decoder(x) + self.bias
        return x
