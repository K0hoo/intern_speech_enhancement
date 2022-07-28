
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


class RealImg_Norm_LSTM(nn.Module):
    def __init__(self):
        super(RealImg_Norm_LSTM, self).__init__()
        
        self.lstm1 = nn.LSTM(input_size=514, hidden_size=512, bidirectional=True, batch_first=True)
        self.act1 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(inplace=False)
        )

        self.norm1 = nn.GroupNorm(512, 512)

        self.lstm2 = nn.LSTM(input_size=512, hidden_size=512, bidirectional=True, batch_first=True)
        self.act2 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(inplace=False)
        )

        self.norm2 = nn.GroupNorm(512, 512)

        self.lstm3 = nn.LSTM(input_size=512, hidden_size=512, bidirectional=True, batch_first=True)
        self.act3 = nn.Sequential(
            nn.Linear(1024, 514),
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
    
        return x * out