
import torch
import torch.nn as nn

class SimpleLSTM(nn.Module):
    def __init__(self):
        super(SimpleLSTM, self).__init__()
        self.lstm = nn.LSTM(
                input_size=257,
                hidden_size=256,
                num_layers=3,
                bidirectional=True,
                batch_first=True
        )
        self.fc = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Linear(512, 257),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out)
        return out

class RealImg_LSTM(nn.Module):
    def __init__(self):
        super(RealImg_LSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=514,
            hidden_size=512,
            num_layers=3,
            bidirectional=True,
            batch_first=True
        )
        self.fc = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Linear(1024, 514)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out)
        return out


class Wiener_LSTM(nn.Module):
    def __init__(self):
        super(Wiener_LSTM, self).__init__()
        self.input_size = 257
        self.hidden_size = 256

        self.lstm1 = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, bidirectional=True, batch_first=True)
        self.act1 = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.ReLU(inplace=False)
        )

        self.norm1 = nn.GroupNorm(self.hidden_size, self.hidden_size)

        self.lstm2 = nn.LSTM(input_size=self.hidden_size, hidden_size=self.hidden_size, bidirectional=True, batch_first=True)
        self.act2 = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.ReLU(inplace=False)
        )

        self.norm2 = nn.GroupNorm(self.hidden_size, self.hidden_size)

        self.lstm3 = nn.LSTM(input_size=self.hidden_size, hidden_size=self.hidden_size, bidirectional=True, batch_first=True)
        self.act3 = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.input_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.act1(out)
        out = torch.transpose(out, 1, 2)
        out = self.norm1(out)
        out = torch.transpose(out, 1, 2)
        
        out, _ = self.lstm2(out)
        out = self.act2(out)
        out = torch.transpose(out, 1, 2)
        out = self.norm2(out)
        out = torch.transpose(out, 1, 2)

        out, _ = self.lstm3(out)
        out = self.act3(out)
    
        return out
