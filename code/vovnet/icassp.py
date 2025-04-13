from torch import nn
import torch


class Part1(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.AvgPool2d(kernel_size=2, stride=1, ceil_mode=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(9, 1), padding=(4, 0)),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.AvgPool2d(kernel_size=2, stride=1, ceil_mode=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(1, 11), padding=(0, 5)),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.AvgPool2d(kernel_size=2, stride=1, ceil_mode=True),
        )

    def forward(self, inputs):
        x1 = self.conv1(inputs)
        x2 = self.conv2(inputs)
        x3 = self.conv3(inputs)
        x = torch.cat([x1, x2, x3], dim=1)
        return x


class Part2(nn.Module):
    def __init__(self, output, frame=187, column=40, dropout=0.3):
        super().__init__()
        self.part1 = Part1()
        self.frame = frame
        self.column = column
        self.seq1 = nn.Sequential(
            nn.Conv2d(in_channels=96, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, ceil_mode=True),
        )
        self.seq2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, ceil_mode=True),
        )
        self.seq3 = nn.Sequential(
            nn.Conv2d(in_channels=96, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=(2, 1), ceil_mode=True),
        )
        self.seq4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=160, kernel_size=3, padding=1),
            nn.BatchNorm2d(160),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=(2, 1), ceil_mode=True),
        )
        self.seq5 = nn.Sequential(
            nn.Conv2d(in_channels=160, out_channels=320, kernel_size=1),
            nn.BatchNorm2d(320),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.final = nn.Sequential(
            nn.Linear(in_features=320, out_features=output),
            nn.Dropout(p=dropout),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        x = x.reshape(-1, 1, self.frame, self.column)
        batch = x.shape[0]
        x = self.part1(x)
        x = self.seq1(x)
        x = self.seq2(x)
        x = self.seq3(x)
        x = self.seq4(x)
        x = self.seq5(x)
        x = x.view(batch, 320)
        x = self.final(x)
        return x
