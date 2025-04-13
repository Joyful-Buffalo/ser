from torch import nn
import torch


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


class InceptionNet(nn.Module):
    def __init__(self, dropout=0.5):
        super(InceptionNet, self).__init__()
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
            # nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(2, 2), stride=(2, 2)),
            # nn.BatchNorm2d(64),
            nn.Dropout(p=dropout),
        )

        #        c1, c5_in, c5_out c3_in, c3_out, cb,
        channel2 = [8, 16, 64, 8, 32, 24]  # 128
        channel4 = [16, 32, 128, 16, 64, 48]  # 256
        self.inception3 = InceptionBlock(64, channel2[0], channel2[1], channel2[2], channel2[3], channel2[4],
                                         channel2[5])
        self.inception4 = InceptionBlock(128, channel4[0], channel4[1], channel4[2], channel4[3], channel4[4],
                                         channel4[5])
        self.mp8 = nn.MaxPool2d(kernel_size=8, stride=8, ceil_mode=True)
        self.mp4 = nn.MaxPool2d(kernel_size=4, stride=4, ceil_mode=True)
        self.mp2 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.sequential = nn.Sequential(
            # nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            # nn.Dropout(p=dropout),
        )

    def forward(self, x):
        x = self.conv1(x)  # [256, 24, 150, 7]
        x = self.conv2(x)  # [256, 64, 75, 4]
        x = self.sequential(self.inception3(x))  # [256, 128, 38, 2]
        x = self.sequential(self.inception4(x))  # [256, 128, 19, 1]
        return x


class VovNet(nn.Module):
    def __init__(self, dropout=0.5):
        super(VovNet, self).__init__()
        self.inception = InceptionNet(dropout=dropout)
        self.vov1_1x1 = nn.Sequential(
            nn.Conv2d(kernel_size=(1, 1), in_channels=97, out_channels=64),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=dropout),
        )
        self.vov2_1x1 = nn.Sequential(
            nn.Conv2d(kernel_size=(1, 1), in_channels=449, out_channels=256),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=dropout),
        )

    def forward(self, inputs):  # [256, 1, 300, 13]
        x11 = self.inception.mp4(inputs)  # [256, 1, 75, 4]
        x12 = self.inception.conv1(inputs)  # [256, 24, 75, 4]
        x13 = self.inception.conv2(x12)  # [256, 64, 38, 2]
        x12 = self.inception.mp2(x12)  # [256, 24, 38, 2]
        x1 = self.vov1_1x1(torch.cat([x11, x12, x13], dim=1))

        # x21 = self.mp4(x11)  # [256, 1, 19, 1]
        x21 = self.inception.mp4(self.inception.mp4(inputs))
        x22 = self.inception.mp4(x1)  # [256, 64, 19, 1]
        x23 = self.inception.sequential(self.inception.inception3(x1))  # [256, 128, 75, 4]
        x24 = self.inception.mp2(x23)
        x25 = self.inception.sequential(self.inception.inception4(x23))
        out = self.vov2_1x1(torch.cat([x21, x22, x24, x25], dim=1))

        return out