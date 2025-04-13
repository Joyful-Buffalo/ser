from torch import nn


class LogHh(nn.Module):
    def __init__(self, dropout=0.5):
        super(LogHh, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(3, 32, kernel_size=(3, 5, 5), stride=(1, 1, 1), padding=(1, 2, 2)),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=(2, 2), padding=0, ceil_mode=True),
            nn.Dropout(p=dropout)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=(2, 2), padding=0, ceil_mode=True),
            nn.Dropout(p=dropout)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=(2, 2), padding=0, ceil_mode=True),
            nn.Dropout(p=dropout)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=(2, 2), padding=0, ceil_mode=True),
            nn.Dropout(p=dropout)
        )

    def forward(self, x_image):
        h_conv1 = self.conv1(x_image)  # [256, 32, 150, 7]# batch_size, channels, height, width
        h_conv2 = self.conv2(h_conv1)  # [256, 64, 75, 4]
        h_conv3 = self.conv3(h_conv2)  # [256, 128, 38, 2]
        h_conv4 = self.conv4(h_conv3)  # [256, 256, 19, 1]
        return h_conv4
