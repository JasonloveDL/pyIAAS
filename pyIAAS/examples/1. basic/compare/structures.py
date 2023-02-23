import torch
from torch import nn


class lstm_model(nn.Module):
    def __init__(self):
        super().__init__()
        time_len = 168
        hidden_size = 48
        self.prelstm = nn.Linear(168, time_len)
        self.norm1 = torch.nn.BatchNorm1d(3)
        # self.prelstm2 = nn.Linear(3, hidden_size)
        # self.norm2 = torch.nn.BatchNorm1d(time_len)
        self.lstm = nn.LSTM(batch_first=True,
                            input_size=3,
                            hidden_size=hidden_size,
                            num_layers=3,
                            dropout=0.3)
        self.norm3 = torch.nn.BatchNorm1d(hidden_size)
        self.dense1 = nn.Linear(hidden_size, hidden_size)
        self.dense2 = nn.Linear(hidden_size, 1)
        self.active = nn.SELU()

    def forward(self, x):
        x = self.prelstm(x)
        x = self.norm1(x)
        x = self.active(x)
        x = x.transpose(1, 2)
        # x = self.prelstm2(x)
        # x = self.norm2(x)
        # x = self.active(x)
        # x = x.transpose(1, 2)
        _, (x, _) = self.lstm(x)
        x = x[-1]
        x = self.norm3(x)
        x = self.active(x)
        x = self.dense1(x)
        x = self.active(x)
        x = self.dense2(x)
        x = torch.flatten(x)
        return x


class cnn_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn1 = nn.Conv1d(3, 4, 3, padding=1)
        self.pool1 = nn.AvgPool1d(2)
        self.cnn2 = nn.Conv1d(4, 16, 3, padding=1)
        self.pool2 = nn.AvgPool1d(2)
        self.dense1 = nn.Linear(42, 64)
        self.dense2 = nn.Linear(64, 32)
        self.lstm = nn.LSTM(batch_first=True,
                            input_size=32,
                            hidden_size=64,
                            num_layers=3,
                            dropout=0.3)
        self.dense3 = nn.Linear(64, 8)
        self.dense4 = nn.Linear(8, 1)
        self.active = nn.ELU()

    def forward(self, x):
        # x = x.transpose(1, 2)
        x = self.cnn1(x)
        x = self.pool1(x)
        x = self.active(x)
        x = self.cnn2(x)
        x = self.pool2(x)
        x = self.active(x)
        x = self.dense1(x)
        x = self.active(x)
        x = self.dense2(x)
        _, (x, _) = self.lstm(x)
        x = x[0, :]
        x = self.dense3(x)
        x = self.active(x)
        x = self.dense4(x)
        x = torch.flatten(x)
        return x
