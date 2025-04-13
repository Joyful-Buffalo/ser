import torch
from torch import nn


# 尾部1x1加dropout
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
            # nn.Dropout(p=dropout),
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
        self.mp2 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.sequential = nn.Sequential(
            # nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            nn.Dropout(p=dropout),
        )

    def forward(self, x):
        x = self.conv1(x)  # [256, 24, 150, 7]
        x = self.conv2(x)  # [256, 64, 75, 4]
        x = self.sequential(self.inception3(x))  # [256, 128, 38, 2]
        x = self.sequential(self.inception4(x))  # [256, 128, 19, 1]
        return x


# 3x3 dropout 去掉，seq加， 末尾1x1加
class FullInception7(nn.Module):
    def __init__(self, frame, column, output_size, dropout=0.5):
        super(FullInception7, self).__init__()
        self.frame = frame
        self.column = column
        self.output = output_size
        self.cnn = InceptionNet1(dropout=dropout)
        self.final = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            nn.Dropout(p=dropout),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            nn.Dropout(p=dropout),
            nn.Conv2d(in_channels=64, out_channels=output_size, kernel_size=1),
        )
        in_feature = frame
        in_final = column
        for i in range(6):
            if in_feature // 2 < in_feature / 2:
                in_feature = in_feature // 2 + 1
            else:
                in_feature /= 2
        for i in range(6):
            if in_final // 2 < in_final / 2:
                in_final = in_final // 2 + 1
            else:
                in_final /= 2
        self.view = int(in_feature * in_final)
        self.fc = nn.Linear(in_features=self.view, out_features=1)

    def forward(self, x):
        mfcc_data = x[:, :self.frame * self.column]
        x_image = mfcc_data.reshape(-1, 1, self.frame, self.column)
        conv_out = self.cnn(x_image)  # [256, 128, 19, 1]
        conv_out = self.final(conv_out)
        conv_out = conv_out.view(-1, self.output, self.view)
        out = self.fc(conv_out).squeeze(-1)
        return out
