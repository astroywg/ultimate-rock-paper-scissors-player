import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
from torchvision import datasets, transforms
# from torchsummary.torchsummary import summary

from models.simple_classify import Classifier3d


def main():

    # Original ResNext101 size [64, 128, 256, 512, 1024], [3, 24, 36, 3]
    # Just can't fit that model into my GPU

    net = Classifier3d(
        channels_list=[64, 128, 128, 128, 128],
        block_nums=[3, 4, 6, 3]
    ).cuda()
    # summary(net, input_size=(3, 90, 112, 112))

    test_batch_size = 2
    data = torch.randn(test_batch_size, 3, 90, 112, 112, device='cuda')
    target = torch.randint(3, (test_batch_size,), dtype=torch.int64, device='cuda')
    output = net(data)
    loss = F.cross_entropy(output, target)
    print(loss.item())
    loss.backward()

if __name__ == '__main__':
    main()
