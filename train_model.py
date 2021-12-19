import glob, os
import torch, torchvision
import numpy as np
from pathlib import Path
from datetime import datetime
import time
from torchsummary import torchsummary

import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm

from models.simple_classify import Classifier2dLSTM, Classifier3d


torch.backends.cudnn.benchmark = True


CUT_SIZE = 200
load_directory = 'data'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def hand_to_num(filename):
    if 'rock' in filename:
        return 1
    elif 'scissors' in filename:
        return 2
    elif 'paper' in filename:
        return 3
    else:
        raise ValueError('filename does not contain hand shape')


def load_from_directory(video_directory):
    video_paths = glob.glob(video_directory + '*.mp4')
    load_dir = Path(load_directory)
    load_dir.mkdir()

    for video_path in video_paths:
        if not Path(video_path + '.npy').exists():
            continue

        video_name = Path(video_path).stem
        video = torchvision.io.read_video(video_path)[0]

        assert video.size(1) == 240
        assert video.size(2) == 320

        directory = load_dir / video_name
        directory.mkdir()

        for i in range(0, video.size(0), CUT_SIZE):
            torch.save(video[i:(i+CUT_SIZE)].clone(), str(directory / f'{i}.pt'))

        del video

        frames = np.load(video_path + '.npy')
        np.save(str(directory / 'frames.npy'), frames)

        print(
            str(directory), 
            '{:,} Bytes'.format(sum(
                f.stat().st_size 
                for f 
                in directory.glob('**/*') 
                if f.is_file()
            )),
            'Frames {}, {}, ...'.format(
                frames[0],
                frames[1]
            )
        )


def get_video_frames(name, start_frame, end_frame):
    directory = Path(load_directory) / name

    start = (start_frame // CUT_SIZE) * CUT_SIZE
    end = (end_frame // CUT_SIZE) * CUT_SIZE

    tensors = []
    for i in range(start, end+1, CUT_SIZE):
        loaded = torch.load(str(directory / f'{i}.pt'))
        
        left = max(0, start_frame - i)
        right = min(CUT_SIZE, end_frame - i)
        tensors.append(loaded[left:right])

    return torch.cat(tensors, dim=0)


class VideoDataset(torch.utils.data.Dataset):
    '''
    Dataset for video.
    '''

    def __init__(
        self, 
        net_type, 
        is_valid_video, 
        clip_tolerance, 
        frame_stride,
        offset,
        clip_len
    ) -> None:

        super().__init__()
        self.net_type = net_type
        self.clip_tolerance = clip_tolerance
        self.frame_stride = frame_stride
        self.clip_len = clip_len

        assert self.net_type in ['3d', '2d+lstm']

        self.videos = []
        video_lens = []

        for npy_path in Path(load_directory).glob('*/frames.npy'):
            name = npy_path.parent.stem
            if not is_valid_video(name):
                continue
            frames = np.load(str(npy_path)) - offset
            min_f, max_f = np.min(frames), np.max(frames)

            if net_type == '2d+lstm':
                min_f += clip_len + clip_tolerance
            
            self.videos.append({
                'name': name,
                'type': hand_to_num(str(npy_path)),
                'frames': frames,
                'start': min_f,
                'end': max_f
            })
            video_lens.append(
                max_f - min_f
            )

        video_lens = np.array(video_lens)
        self.total_clips = np.sum(video_lens)
        self.cumulative = np.cumsum(video_lens)

    def __len__(self):
        return self.total_clips // self.frame_stride

    def __getitem__(self, index):
        
        if index not in range(len(self)):
            raise ValueError('Wrong index!')
        frame_num = index * self.frame_stride

        video_num = np.searchsorted(self.cumulative, frame_num, side='right')
        if video_num == 0:
            idx = frame_num
        else:
            idx = frame_num - self.cumulative[video_num - 1]

        hand_frames = self.videos[video_num]['frames']
        idx += self.videos[video_num]['start']

        start_frame = idx - self.clip_len + 1
        end_frame = idx + 1

        frames = get_video_frames(
            self.videos[video_num]['name'], start_frame, end_frame
        ) # shape: (T, H, W, C)

        frames = frames.permute(3, 0, 1, 2).to(
            dtype=torch.get_default_dtype()
        ).div(255)
        # shape: (C, T, H, W)

        frames.sub_(0.5).div_(0.5)

        target_frame = np.array([idx]) if self.net_type == '3d' else np.arange(start_frame, end_frame)
        closest_hand_shape = np.searchsorted(hand_frames, target_frame, side='right') - 1
        hand_shape_distance = target_frame - hand_frames[closest_hand_shape]

        target = self.videos[video_num]['type'] * (hand_shape_distance < self.clip_tolerance).astype(int)
        target = torch.tensor(target)

        return frames, target


def collate_clips(batch):
    data = torch.stack([clip[0] for clip in batch])
    target = torch.squeeze(torch.stack([clip[1] for clip in batch])).to(dtype=torch.long)
    return data, target


def train(net : nn.Module, device, train_loader, optimizer, epoch: int, null_class_weight):

    print('Epoch {:#2d}'.format(epoch))
    net.train()

    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(
            [null_class_weight, 3, 3, 3], 
            dtype=torch.float, 
            device=device
        ), 
        reduction='mean'
    )

    total_loss = torch.tensor(0., device=device)
    total_right = torch.zeros(4, dtype=torch.float, device=device, requires_grad=False)
    total_num = torch.zeros(4, dtype=torch.float, device=device, requires_grad=False)
    
    with tqdm(total=len(train_loader), desc='Train') as t:
        for idx, (data, target) in enumerate(train_loader):

            start = time.time()
            data = data.to(device)
            target = target.to(device)

            optimizer.zero_grad(set_to_none=True)

            output = net(data)

            output = output.flatten(end_dim=-2)
            target = target.flatten()

            loss = criterion(output, target)

            target_nums = torch.bincount(
                target,
                minlength = 4
            ).detach()
            rights = torch.bincount(
                target,
                weights = (target == output.argmax(1)),
                minlength = 4
            ).detach()
            
            loss.backward()
            optimizer.step()

            total_loss += loss.detach()
            total_num += target_nums
            total_right += rights
            elapsed = time.time() - start

            if idx % 25 == 0:
                print(
                    'batch', idx, 
                    'loss', loss.detach().item(), 
                    'num', target_nums.cpu().numpy(), 
                    'right', rights.cpu().numpy()
                )

                train_loss = total_loss.item() / (idx + 1)
                train_acc = 100. * total_right / total_num
                train_acc = train_acc.cpu().numpy()
                print('Train loss: {:6.3f}, Train acc: {}%'.format(train_loss, train_acc))
            
            t.set_postfix(t=f'{(elapsed):.4f}')
            t.update()

    train_loss = total_loss.item() / len(train_loader)

    train_acc = 100. * total_right / total_num
    train_acc = train_acc.cpu().numpy()

    print('Train loss: {:6.3f}, Train acc: {}%'.format(train_loss, train_acc))
    return train_loss, train_acc


@torch.no_grad()
def test(net : nn.Module, device, test_loader, null_class_weight):

    net.eval()

    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(
            [null_class_weight, 3, 3, 3], 
            dtype=torch.float, 
            device=device
        ), 
        reduction='mean'
    )

    total_loss = torch.tensor(0., device=device)
    total_right = torch.zeros(4, dtype=torch.float, device=device, requires_grad=False)
    total_num = torch.zeros(4, dtype=torch.float, device=device, requires_grad=False)
    
    with tqdm(total=len(test_loader), desc='Test') as t:
        for idx, (data, target) in enumerate(test_loader):

            start = time.time()
            data = data.to(device)
            target = target.to(device)

            output = net(data)

            output = output.flatten(end_dim=-2)
            target = target.flatten()

            loss = criterion(output, target)

            target_nums = torch.bincount(
                target,
                minlength = 4
            ).detach()
            rights = torch.bincount(
                target,
                weights = (target == output.argmax(1)),
                minlength = 4
            ).detach()

            total_loss += loss.detach()
            total_num += target_nums
            total_right += rights
            elapsed = time.time() - start

            if idx % 25 == 0:
                print(
                    'batch', idx, 
                    'loss', loss.detach().item(), 
                    'num', target_nums.cpu().numpy(), 
                    'right', rights.cpu().numpy()
                )
            
            t.set_postfix(t=f'{(elapsed):.4f}')
            t.update()

    test_loss = total_loss.item() / len(test_loader)

    test_acc = 100. * total_right / total_num
    test_acc = test_acc.cpu().numpy()

    print('Test loss: {:6.3f}, Test acc: {}%'.format(test_loss, test_acc))
    return test_loss, test_acc


def save_model(net, path=None, epoch=0):
    if path is None:
        Path('ckpts').mkdir(exist_ok=True)
        path = 'ckpts/{}_{}.pt'.format(
            datetime.now().strftime('%Y%m%d_%H%M%S'),
            str(epoch)
        )
    torch.save(net.state_dict(), path)
    print(' -> Saved', end='')


def main_3d():

    clip_tolerance = 12
    null_class_weight = clip_tolerance / (90 - clip_tolerance)

    train_dataset = VideoDataset(
        net_type = '3d',
        is_valid_video = (lambda x: x[-1] in 'abc'),
        clip_tolerance = clip_tolerance,
        frame_stride = 4,
        clip_len = 16,
        offset = 6
    )

    test_dataset = VideoDataset(
        net_type = '3d',
        is_valid_video = (lambda x: x[-1] == 'e'),
        clip_tolerance = clip_tolerance,
        frame_stride = 4,
        clip_len = 16,
        offset = 6
    )

    print('Train', len(train_dataset), 'Test', len(test_dataset))

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        collate_fn=collate_clips
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        collate_fn=collate_clips
    )

    net = Classifier3d(
        channel_nums=[16, 16, 32, 64, 128],
        block_nums=[1, 1, 1, 1]
    ).to(device)

    optimizer = optim.Adam(net.parameters(), lr=1e-4)

    best_acc = 0

    for epoch in range(1,10):
        train(net, device, train_loader, optimizer, epoch, null_class_weight)
        test_loss, test_acc = test(net, device, test_loader, null_class_weight)

        if best_acc < test_acc.mean():
            save_model(net, epoch=epoch)
            best_acc = test_acc.mean()


def main_2dlstm():

    clip_tolerance = 12
    null_class_weight = clip_tolerance / (90 - clip_tolerance)

    train_dataset = VideoDataset(
        net_type = '2d+lstm',
        is_valid_video = (lambda x: x[-1] in 'abc'),
        clip_tolerance = clip_tolerance,
        frame_stride = 64,
        clip_len = 128,
        offset = 6
    )

    test_dataset = VideoDataset(
        net_type = '2d+lstm',
        is_valid_video = (lambda x: x[-1] == 'e'),
        clip_tolerance = clip_tolerance,
        frame_stride = 64,
        clip_len = 128,
        offset = 6
    )

    print('Train', len(train_dataset), 'Test', len(test_dataset))

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        collate_fn=collate_clips
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=2,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        collate_fn=collate_clips
    )

    net = Classifier2dLSTM().to(device)

    optimizer = optim.Adam(net.parameters(), lr=1e-4)

    best_acc = 0

    for epoch in range(1,10):
        train(net, device, train_loader, optimizer, epoch, null_class_weight)
        test_loss, test_acc = test(net, device, test_loader, null_class_weight)

        if best_acc < test_acc.mean():
            save_model(net, epoch=epoch)
            best_acc = test_acc.mean()


def main():
    load_from_directory('dataset/')
    main_3d()

if __name__ == '__main__':
    main()