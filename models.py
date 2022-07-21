
import torch
import torch.nn as nn
import torch.functional as F

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
