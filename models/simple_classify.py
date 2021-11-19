import torch
import torch.nn as nn


class ResNext3dBlock(nn.Module):
    def __init__(self, in_channels, out_channels=-1, stride=1):
        super().__init__()

        if out_channels == -1:
            out_channels = in_channels
            self.downsample = lambda x: x
        else:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False),
                nn.BatchNorm3d(out_channels)
            )

        hidden_channels = out_channels
        assert hidden_channels % 32 == 0

        self.act = nn.ReLU(inplace=True)

        self.cv1 = nn.Conv3d(in_channels, hidden_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm3d(hidden_channels)

        self.cv2 = nn.Conv3d(hidden_channels, hidden_channels, kernel_size=3, stride=stride, padding=1, groups=32, bias=False)
        self.bn2 = nn.BatchNorm3d(hidden_channels)

        self.cv3 = nn.Conv3d(hidden_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        output = self.act(self.bn1(self.cv1(x)))
        output = self.act(self.bn2(self.cv2(output)))
        output = self.bn3(self.cv3(output))

        output += self.downsample(x)
        output = self.act(output)

        return output


class Classifier3d(nn.Module):
    """
    3d CNN for simple classification. 
    
    Adapted ResNeXt-101 net structure of classifier in
    "Real-time Hand Gesture Detection and Classification Using Convolutional Neural Networks": https://arxiv.org/abs/1901.10323

    Also refered to 
    "Can Spatiotemporal 3D CNNs Retrace the History of 2D CNNs and ImageNet?": https://arxiv.org/abs/1711.09577
    """

    def __init__(self, channels_list, block_nums, color=True):
        super().__init__()

        in_channels = 3 if color else 1
        strides = [1, 2, 2, 2]

        self.act = nn.ReLU(inplace=True)

        self.cv1 = nn.Conv3d(in_channels, channels_list[0], kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(channels_list[0])
        self.mp1 = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        self.cv_layers = nn.Sequential(*[
            self._make_layer(channels_list[i], channels_list[i+1], block_num=block_nums[i], stride=strides[i])
            for i in range(4)
        ])

        self.avp = nn.AdaptiveAvgPool3d(output_size=(1, 1, 1))
        self.flt = nn.Flatten()
        self.fc = nn.Linear(channels_list[4], 3)
    
    def _make_layer(self, in_channels, out_channels, block_num, stride=1):
        blocks = []

        blocks.append(ResNext3dBlock(in_channels, out_channels, stride=stride))
        for i in range(1, block_num):
            blocks.append(ResNext3dBlock(out_channels))

        return nn.Sequential(*blocks)
        
    def forward(self, x):
        """
        Return raw scores of 3 classes.
        """
        output = self.act(self.bn1(self.cv1(x)))
        output = self.mp1(output)

        output = self.cv_layers(output)

        output = self.fc(self.flt(self.avp(output)))

        return output
