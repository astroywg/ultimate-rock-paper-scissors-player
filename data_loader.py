import torch
import numpy as np
from pathlib import Path

CUT_SIZE = 200

def hand_to_num(filename):
    if 'rock' in filename:
        return 1
    elif 'scissors' in filename:
        return 2
    elif 'paper' in filename:
        return 3
    else:
        raise ValueError('filename does not contain hand shape')

def get_video_frames(name, start_frame, end_frame):
    directory = Path('data') / name

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
    """
    Dataset for video.
    """

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

        for npy_path in Path('data').glob('*/frames.npy'):
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
    target = torch.squeeze(torch.stack([clip[1] for clip in batch]))
    return data, target
