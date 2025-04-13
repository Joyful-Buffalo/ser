import torch
from torch import nn


class InceptionBlock(nn.Module):
    def __init__(self, in_channels, c1, c5_in, c5_out, c3_in, c3_out, cb, dropout=0.5):
        super(InceptionBlock, self).__init__()
        self.branch1x1 = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=(1, 1)),
            nn.BatchNorm2d(c1)
        )
        self.branch_pool = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=(1, 1), padding=(1, 1)),
            nn.Conv2d(in_channels, cb, kernel_size=(1, 1)),
            nn.BatchNorm2d(cb)
        )
        self.branch3x3 = nn.Sequential(
            nn.Conv2d(in_channels, c3_in, kernel_size=(1, 1)),
            nn.BatchNorm2d(c3_in),
            nn.Conv2d(c3_in, c3_out, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(c3_out)
        )
        self.branch5x5 = nn.Sequential(
            nn.Conv2d(in_channels, c5_in, kernel_size=(1, 1)),
            nn.BatchNorm2d(c5_in),
            nn.Conv2d(c5_in, c5_out, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(c5_out),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Conv2d(c5_out, c5_out, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(c5_out),
        )

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch_pool = self.branch_pool(x)
        branch5x5 = self.branch5x5(x)
        branch3x3 = self.branch3x3(x)
        out_puts = [branch1x1, branch5x5, branch3x3, branch_pool]
        return torch.cat(out_puts, dim=1)


class InceptionNet1(nn.Module):
    def __init__(self, dropout=0.5):
        super(InceptionNet1, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(7, 7), padding=(3, 3)),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), ceil_mode=True),
            nn.Dropout(p=dropout),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(1, 1)),
            nn.BatchNorm2d(16),
            nn.Conv2d(in_channels=16, out_channels=64, kernel_size=(5, 5), padding=(2, 2)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), ceil_mode=True),
            nn.Dropout(p=dropout),
        )

        #        c1, c5_in, c5_out c3_in, c3_out, cb,
        channel2 = [8, 16, 64, 8, 32, 24]  # 128
        channel4 = [16, 32, 128, 16, 64, 48]  # 256
        # channel4 = [8, 16, 64, 8, 32, 24]
        self.inception3 = InceptionBlock(64, channel2[0], channel2[1], channel2[2], channel2[3], channel2[4],
                                         channel2[5])
        self.inception4 = InceptionBlock(128, channel4[0], channel4[1], channel4[2], channel4[3], channel4[4],
                                         channel4[5])
        channel5 = [32, 64, 256, 32, 128, 96]
        self.inceptionB1 = InceptionBlock(256, channel5[0], channel5[1], channel5[2], channel5[3], channel5[4],
                                          channel5[5])
        channel4 = [16, 32, 128, 16, 64, 48]
        self.inceptionB2 = InceptionBlock(512, channel4[0], channel4[1], channel4[2], channel4[3], channel4[4],
                                          channel4[5])
        self.mp2 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.sequential = nn.Sequential(
            # nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            # nn.Dropout(p=dropout),  [256, 128, 19, 1]
        )

    def forward(self, x):
        x = self.conv1(x)  # [256, 24, 150, 7]
        x = self.conv2(x)  # [256, 64, 75, 4]
        x = self.sequential(self.inception3(x))  # [256, 128, 38, 2]
        x = self.sequential(self.inception4(x))  # [256, 128, 19, 1]
        x = self.sequential(self.inceptionB1(x))
        x = self.sequential(self.inceptionB2(x))
        return x


class FullConvInception(nn.Module):
    def __init__(self, frame, column, output_size, dropout=0.5):
        super(FullConvInception, self).__init__()
        self.frame = frame
        self.column = column
        self.output = output_size
        self.inception = InceptionNet1(dropout=dropout)
        # channel4 = [16, 32, 128, 16, 64, 48]  # 256
        self.final = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=output_size, kernel_size=1),
            nn.AdaptiveAvgPool2d((1, 1))
        )

    def forward(self, x):
        mfcc_data = x[:, :self.frame * self.column]
        x_image = mfcc_data.reshape(-1, 1, self.frame, self.column)
        conv_out = self.inception(x_image)  # [256, 128, 19, 1]
        out = self.final(conv_out).squeeze(-1).squeeze(-1)
        return out
