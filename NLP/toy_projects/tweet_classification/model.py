import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, hidden_dim, input_dim, n_layers, n_classes, vocab_size, dropout_p=0.5, , device="cuda"):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dropout = nn.Dropout(dropout_p)
        self.device = device

        self.embed = nn.Embedding(vocab_size, input_dim)
        self.gru = nn.GRU(input_dim, hidden_dim, n_layers, batch_first =True)
        self.out = nn.Linear(hidden_dim, n_classes)
        self.relu = nn.ReLU()

    def _init_hidden(self, batch_size):
        weight= next(self.parameters()).data
        hidden = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(self.device)
        return hidden

    def forward(self, x):
        x = self.embed(x)
        h = self._init_hidden(batch_size = x.size(0))
        out, _ = self.gru(x,h)
        out = self.relu(out[:,-1,:])
        self.dropout(out)
        out = self.out(out)
        return out
