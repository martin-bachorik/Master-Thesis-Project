import torch
import torch.nn as nn

__all__ = ['FFNN', 'LSTMModel', 'RNN']


class FFNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FFNN, self).__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

        # Non linear functions
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

    def forward(self, x):
        # HIDDEN LAYERS
        out1 = self.fc1(x)
        out1 = self.tanh(out1)

        out2 = self.fc2(out1)
        out2 = self.tanh(out2)

        # OUTPUT LAYER
        out3 = self.fc3(out2)
        return out3


class RNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(RNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.rnn = nn.RNN(input_dim, hidden_dim, layer_dim, batch_first=True, nonlinearity='relu')
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)
        out, hn = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        # batch_first: If True, then the input and output tensors are provided as (batch, seq, feature).
        # Default: False (seq_len, batch, feature)
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sig = nn.Sigmoid()
        self.fc = nn.Linear(hidden_dim, output_dim)  # Output layer

    def forward(self, x):
        #  x = (batch, seq_len, input_size); h = (num_layers * num_directions, batch, hidden_size)
        # out = (batch, seq_len, num_directions * hidden_size):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)
        out, (hn, cn) = self.lstm(x, (h0, c0))
        # OUTPUT layer
        out = self.fc(out[:, -1, :])
        return out
