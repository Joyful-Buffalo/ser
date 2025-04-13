from torch import nn
import torch


class VovNet(nn.Module):
    def __init__(self, csc_out, in_channel, mid_channel, out_channel, in2, mid2, out2, out, dropout=0.5):
        super(VovNet, self).__init__()
        self.csc = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=csc_out, kernel_size=1),
            nn.BatchNorm2d(csc_out),
            nn.ReLU(inplace=True),
        )
        self.vov1_1x1 = nn.Sequential(
            nn.Conv2d(kernel_size=(1, 1), in_channels=in_channel + out_channel * 2, out_channels=in2),
            nn.BatchNorm2d(in2),
            nn.ReLU(inplace=True),
        )
        self.vov2_1x1 = nn.Sequential(
            nn.Conv2d(kernel_size=(1, 1), in_channels=in2 + out2 * 2 + csc_out, out_channels=out),
            nn.BatchNorm2d(out),
        )
        self.cnnBlock1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
        )
        self.cnnBlock2 = nn.Sequential(
            nn.Conv2d(in_channels=out_channel, out_channels=mid_channel, kernel_size=1),
            nn.BatchNorm2d(mid_channel),
            nn.Conv2d(in_channels=mid_channel, out_channels=out_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
        )
        self.cnnBlock3 = nn.Sequential(
            nn.Conv2d(in_channels=in2, out_channels=out2, kernel_size=3, padding=1),
            nn.BatchNorm2d(out2),
            nn.ReLU(inplace=True),
        )
        self.cnnBlock4 = nn.Sequential(
            nn.Conv2d(in_channels=out2, out_channels=mid2, kernel_size=1),
            nn.BatchNorm2d(mid2),
            nn.Conv2d(in_channels=mid2, out_channels=out2, kernel_size=3, padding=1),
            nn.BatchNorm2d(out2),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputs):  # [256, 1, 300, 13]
        x11 = inputs  # [256, 1, 75, 4]
        x12 = self.cnnBlock1(inputs)  # [256, 24, 75, 4]
        x13 = self.cnnBlock2(x12)  # [256, 64, 38, 2]
        x1 = self.vov1_1x1(torch.cat([x11, x12, x13], dim=1))
        x21 = self.csc(inputs)
        x22 = x1
        x23 = self.cnnBlock3(x1)
        x24 = self.cnnBlock4(x23)
        out = self.vov2_1x1(torch.cat([x21, x22, x23, x24], dim=1))
        return out


class BigNet(nn.Module):
    def __init__(self, dropout=0.5):
        super().__init__()
        self.head1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            nn.Dropout(p=dropout)
        )
        self.head2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            nn.Dropout(p=dropout),
        )
        self.down = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            nn.Dropout(p=dropout)
        )
        self.same = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout)
        )
        #         csc_out, in_channel, mid_channel, out_channel, in2, mid2, out2, out
        vov1_c = [8, 64, 32, 128, 192, 96, 64, 128, 128]
        self.vovsame = VovNet(vov1_c[0], vov1_c[1], vov1_c[2], vov1_c[3], vov1_c[4], vov1_c[5], vov1_c[6], vov1_c[7],
                              dropout)
        vov2_c = [12, 128, 48, 192, 288, 144, 96, 192, 192]
        self.vovdown1 = VovNet(vov2_c[0], vov2_c[1], vov2_c[2], vov2_c[3], vov2_c[4], vov2_c[5], vov2_c[6], vov2_c[7],
                               dropout)
        vov3_c = [16, 192, 64, 256, 384, 192, 128, 256, 256]
        self.vovsame2 = VovNet(vov3_c[0], vov3_c[1], vov3_c[2], vov3_c[3], vov3_c[4], vov3_c[5], vov3_c[6], vov3_c[7],
                               dropout)
        vov4_c = [24, 256, 96, 384, 576, 288, 192, 384, 384]
        self.vovdown2 = VovNet(vov4_c[0], vov4_c[1], vov4_c[2], vov4_c[3], vov4_c[4], vov4_c[5], vov4_c[6], vov4_c[7],
                               dropout)

    def forward(self, x):
        x = self.head1(x)
        x = self.head2(x)
        x = self.same(self.vovsame(x))
        x = self.down(self.vovdown1(x))
        x = self.same(self.vovsame2(x))
        x = self.down(self.vovdown2(x))

        return x
