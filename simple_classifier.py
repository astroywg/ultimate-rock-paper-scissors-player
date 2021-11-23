import time
from datetime import datetime
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
import torchvision
from torchvision import datasets, transforms

from tqdm import tqdm
from torchsummary import torchsummary

from models.simple_classify import Classifier3d


MIN_CLIP_FRAME_NUM = 87


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(net : nn.Module, device, train_loader, optimizer, epoch: int):

    print('Epoch {:#2d}'.format(epoch))
    net.train()

    criterion = nn.CrossEntropyLoss()

    loss_list = []
    acc_list = []
    
    with tqdm(total=len(train_loader), ncols=120, desc='Train') as t:
        for data, target in train_loader:
            data = data.to(device)
            target = target.to(device)

            optimizer.zero_grad()

            output = net(data)
            loss = criterion(output, target)
            acc = (output.argmax(1) == target).float().mean()

            loss.backward()
            optimizer.step()

            loss_list.append(loss.item())
            acc_list.append(acc.item())
            
            t.set_postfix(loss=loss.item(), acc=acc.item())
            t.update()

    train_loss = sum(loss_list) / len(loss_list)
    train_acc = 100. * sum(acc_list) / len(acc_list)
        
    print('Train loss: {:6.3f}, Train acc: {:7.3f}%'.format(train_loss, train_acc))
    return train_loss, train_acc


def test(net : nn.Module, device, test_loader):
    
    net.eval()

    criterion = nn.CrossEntropyLoss()

    loss_list = []
    acc_list = []

    with tqdm(total=len(test_loader), ncols=120, desc='Test') as t:
        with torch.no_grad():
            for data, target in test_loader:
                data = data.to(device)
                target = target.to(device)

                output = net(data)
                loss = criterion(output, target)
                acc = (output.argmax(1) == target).float().mean()
        
                loss_list.append(loss.item())
                acc_list.append(acc.item())

                t.set_postfix(loss=loss.item(), acc=acc.item())
                t.update()

    test_loss = sum(loss_list) / len(loss_list)
    test_acc = 100. * sum(acc_list) / len(acc_list)
        
    print('Test loss: {:6.3f}, Test acc: {:7.3f}%'.format(test_loss, test_acc), end='')
    return test_loss, test_acc


convert = transforms.ToTensor()

transform = transforms.Compose([
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


def load_from_filename(name: str) -> torch.Tensor:
    video = torchvision.io.read_video(name)[0]
    frames = video.shape[0]
    offset = (frames - MIN_CLIP_FRAME_NUM) // 2
    video = video[offset:(offset+MIN_CLIP_FRAME_NUM)] 

    images = list(video.numpy())
    return torch.stack([transform(convert(image).to(device)) for image in images], dim=1) # shape: (T, C, H, W)


def discriminate(train: bool) -> Callable[[str], bool]:

    def decide(name: str) -> bool:
        if 'RSP' in name:
            return False

        num_of_clip = int(name.split()[2].split('o')[0])
        return train ^ (num_of_clip % 10 == 0)

    return decide


def save_model(net, path=None):
    if path is None:
        path = 'ckpts/simple_classify/' + datetime.now().strftime('%Y%m%d_%H%M%S') + '.pt'
    torch.save(net.state_dict(), path)
    print(' -> Saved', end='')


def main():

    train_dataset = datasets.DatasetFolder(
        'dataset', 
        load_from_filename, 
        is_valid_file=discriminate(train=True)
    )
    test_dataset = datasets.DatasetFolder(
        'dataset', 
        load_from_filename, 
        is_valid_file=discriminate(train=False)
    )

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=6, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=6, shuffle=True)

    print('Loaded', len(train_loader.dataset), 'train data,', len(test_loader.dataset), 'test data')

    net = Classifier3d(
        channel_nums=[16, 16, 32, 64, 128],
        block_nums=[1, 1, 1, 1],
    ).to(device)
    torchsummary.summary(net, (3, 87, 240, 320))
    optimizer = optim.Adam(net.parameters(), lr=0.1)

    # train/test pipeline
    best_acc = 0

    for epoch in range(50):
        train(net, device, train_loader, optimizer, epoch)
        loss, acc = test(net, device, test_loader)

        if acc > best_acc: 
            best_acc = acc
            save_model(net)
        print()

if __name__ == '__main__':
    main()
