import torch
import torch.nn as nn
import torchvision


class ResNet3dBlock(nn.Module):
    def __init__(self, in_channels, out_channels=-1, stride=1):
        super().__init__()

        if out_channels == -1 or in_channels == out_channels:
            out_channels = in_channels
            self.downsample = lambda x: x
        else:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False),
                nn.BatchNorm3d(out_channels)
            )

        self.act = nn.ReLU(inplace=True)

        self.cv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)

        self.cv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        output = self.act(self.bn1(self.cv1(x)))
        output = self.bn2(self.cv2(output))

        output += self.downsample(x)
        output = self.act(output)

        return output


class Classifier3d(nn.Module):
    """
    3d CNN for simple classification. 
    
    Adapted ResNet-10 net structure of detector in
    "Real-time Hand Gesture Detection and Classification Using Convolutional Neural Networks": https://arxiv.org/abs/1901.10323

    Also refered to 
    "Can Spatiotemporal 3D CNNs Retrace the History of 2D CNNs and ImageNet?": https://arxiv.org/abs/1711.09577
    """

    def __init__(self, channel_nums, block_nums, color=True):
        super().__init__()

        in_channels = 3 if color else 1
        strides = [1, 2, 2, 2]

        self.act = nn.ReLU(inplace=True)

        self.cv1 = nn.Conv3d(in_channels, channel_nums[0], kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(channel_nums[0])
        self.mp1 = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        self.cv_layers = nn.Sequential(*[
            self._make_layer(channel_nums[i], channel_nums[i+1], block_num=block_nums[i], stride=strides[i])
            for i in range(4)
        ])

        self.avp = nn.AdaptiveAvgPool3d(output_size=(1, 1, 1))
        self.flt = nn.Flatten()
        self.fc = nn.Linear(channel_nums[4], 3)
    
    def _make_layer(self, in_channels, out_channels, block_num, stride=1):
        blocks = []

        blocks.append(ResNet3dBlock(in_channels, out_channels, stride=stride))
        for i in range(1, block_num):
            blocks.append(ResNet3dBlock(out_channels))

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


class Classifier2dLSTM(nn.Module):
    def __init__(self):
        super().__init__()

        self.resnet = torchvision.models.resnet18()
        self.lstm = nn.LSTM(
            1000,
            32,
            num_layers=2,
            batch_first=True
        )
        self.fc = nn.Linear(32, 4)

    def forward(self, x):
        B, C, T, H, W = x.shape

        output = self.resnet(x.transpose(1, 2).reshape(B * T, C, H, W))
        output = output.reshape(B, T, 1000)
        output = self.lstm(output)[0]
        output = self.fc(output)

        return output  # shape: (B, T, 4)

    def get_features(self, frame, device):
        data = torch.from_numpy(frame).to(device=device, dtype=torch.float)
        data.div_(255 * 0.5).sub_(1.0)
        return self.resnet(data.unsqueeze(0))

    def next_output(self, features, state=None):
        state = [state] if state else []

        output, new_state = self.lstm(features.unsqueeze(0), *state) # output shape: (1, 1, 32)
        output = self.fc(output.squeeze(0)) # output shape: (1, 4)

        return output, new_state
