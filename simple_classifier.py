import time
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
import torchvision
from torchvision import datasets, transforms

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


def filename_2_video(name: str) -> torch.Tensor:
    video = torchvision.io.read_video(name)[0]
    return torch.permute(video, (3, 0, 1, 2))

def discriminate(mode: str) -> Callable[[str], bool]:

    assert mode in ['train', 'test']
    train = mode == 'train'

    def decide(name: str) -> bool:
        if 'RSP' in name:
            return False

        num_of_clip = int(name.split()[2].split('o')[0])
        return train ^ (num_of_clip % 10 == 0)

    return decide


def main():

    train_transform = transforms.Compose([
        transforms.Resize((224, 224))
    ])
    test_transform = transforms.Compose([
        transforms.Resize((224, 224))
    ])

    train_dataset = datasets.DatasetFolder(
        'dataset', 
        filename_2_video, 
        transform=train_transform, 
        is_valid_file=discriminate('train')
    )
    test_dataset = datasets.DatasetFolder(
        'dataset', 
        filename_2_video, 
        transform=test_transform, 
        is_valid_file=discriminate('test')
    )

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True)
    print(len(train_loader), len(test_loader))

    # Original ResNext101 size [64, 128, 256, 512, 1024], [3, 24, 36, 3]
    # Just can't fit that model into my GPU

    # net = Classifier3d(
    #     channels_list=[64, 128, 128, 128, 128],
    #     block_nums=[3, 4, 6, 3],
    # )

if __name__ == '__main__':
    main()
