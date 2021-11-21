import time

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
from torchvision import datasets, transforms
# from torchsummary.torchsummary import summary

from models.simple_classify import Classifier3d


def train(net : nn.Module, device, train_loader, optimizer, epoch: int):

    print('Epoch {:#2d} ['.format(epoch), end='')
    net.train()

    criterion = nn.CrossEntropyLoss()

    loss_list = []
    acc_list = []
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device)
        target = target.to(device)

        output = net(data)
        loss = criterion(output, target)
        acc = (output.argmax(1) == target).float().mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_list.append(loss.item())
        acc_list.append(acc.item())
        if batch_idx % (len(train_loader) // 20) == 0:
            print('=', end='')

    train_loss = sum(loss_list) / len(loss_list)
    train_acc = 100. * sum(acc_list) / len(acc_list)
        
    print('] Train loss: {:6.3f}, Train acc: {:7.3f}%'.format(train_loss, train_acc), end='')


def test(net : nn.Module, device, test_loader):
    
    net.eval()

    criterion = nn.CrossEntropyLoss()

    loss_list = []
    acc_list = []

    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device)

            output = net(data)
            loss = criterion(output, target)
            acc = (output.argmax(1) == target).float().mean()
    
            loss_list.append(loss.item())
            acc_list.append(acc.item())

    test_loss = sum(loss_list) / len(loss_list)
    test_acc = 100. * sum(acc_list) / len(acc_list)
        
    print('] Test loss: {:6.3f}, Test acc: {:7.3f}%'.format(test_loss, test_acc), end='')


def main():

    # Original ResNext101 size [64, 128, 256, 512, 1024], [3, 24, 36, 3]
    # Just can't fit that model into my GPU

    net = Classifier3d(
        channels_list=[64, 128, 128, 128, 128],
        block_nums=[3, 4, 6, 3]
    )


if __name__ == '__main__':
    main()
