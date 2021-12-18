import numpy as np
from matplotlib import pyplot as plt

from data_loader import *
from models.simple_classify import Classifier3d, Classifier2dLSTM

import torch
import torch.nn as nn
import torch.optim as optim

torch.backends.cudnn.benchmark = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

categories = ['Misc.', 'Rock', 'Scissors', 'Paper']
abbr_categories = ['M', 'R', 'S', 'P']

def build_confusion_matrix(predict, truth):
    confusion_matrix = np.zeros((len(categories), len(categories)))
    for i in range(predict.size):
        row = categories.index(truth[i])
        column = categories.index(predict[i])
        confusion_matrix[row, column] += 1

    num_test_per_cat = truth.size / len(categories)
    confusion_matrix = confusion_matrix / num_test_per_cat
    accuracy = np.mean(np.diag(confusion_matrix))
    print('Accuracy (mean diagonal of confusion matrix) is %.3f' % accuracy)

    plt.figure()
    plt.imshow(confusion_matrix)
    plt.xticks(range(4), abbr_categories)
    plt.yticks(range(4), categories)

    plt.savefig('results/confusion_matrix.png', bbox_inches='tight')

clip_tolerance = 12
null_class_weight = clip_tolerance / (90 - clip_tolerance)

test_dataset = VideoDataset(
    net_type = '3d',
    is_valid_video = (lambda x: x[-1] == 'e'),
    clip_tolerance = clip_tolerance,
    frame_stride = 4,
    clip_len = 16,
    offset = 6
)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=2,
    pin_memory=True,
    collate_fn=collate_clips
)

net = Classifier3d(
        channel_nums=[16, 16, 32, 64, 128],
        block_nums=[1, 1, 1, 1],
    ).to(device)

optimizer = optim.Adam(net.parameters(), lr=1e-4)


@torch.no_grad()
def test(net: nn.Module):
    predict, truth = [], []

    net.load_state_dict(torch.load('Classifier3d.pt', map_location=device))
    net.eval()

    for idx, (data, target) in enumerate(test_loader):
        data = data.to(device)
        target = target.to(device)

        optimizer.zero_grad(set_to_none=True)

        output = net(data)
        output = output.flatten(end_dim=-2)
        target = target.flatten()

        predict.append(output.argmax(1))
        truth.append(target.argmax(1))

    return predict, truth


def main():
    build_confusion_matrix(test(net))


if __name__ == '__main__':
    main()
