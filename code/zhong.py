from torch import nn


class MiddleFlow(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, groups=128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1)
        )

    def forward(self, x):
        out = self.block(x)
        out += x
        return x


class Zhong(nn.Module):
    def __init__(self, frame, column, output_size, dropout=0.3):
        super().__init__()
        self.frame = frame
        self.column = column

        self.output = output_size
        self.start = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, groups=64),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
        )
        self.times = 3
        self.hidden_size = 32
        self.middle_flow = nn.ModuleList(
            MiddleFlow() for _ in range(self.times)
        )
        self.gru = nn.GRU(input_size=1280, hidden_size=self.hidden_size, bidirectional=True, batch_first=True)
        self.final = nn.Linear(5888, out_features=output_size)

    def forward(self, x):
        mfcc_data = x[:, :self.frame * self.column]
        x_image = mfcc_data.reshape(-1, 1, self.frame, self.column)
        out = self.start(x_image)  # [64,64,350,20]
        for mod in self.middle_flow:
            out = mod(out)
        batch, channel, h, w = out.shape
        out = out.reshape(-1, h, channel * w)  # (64,92,1280)
        gru_out, _ = self.gru(out)  # (64,92,1280)
        out = self.final(gru_out.reshape(-1, h * self.hidden_size * 2))
        return out
