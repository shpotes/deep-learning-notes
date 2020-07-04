from typing import Tuple

import math
import sys

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.nn import functional as F


def positions_to_sequences(seq_length: int,
                           tr = None,
                           bx = None,
                           noise_level: float = 0.3) -> torch.Tensor:
    
    st = torch.arange(seq_length).float()
    st = st[None, :, None]
    tr = tr[:, None, :, :]
    bx = bx[:, None, :, :]

    xtr = torch.relu(tr[..., 1] - torch.relu(torch.abs(st - tr[..., 0]) - 0.5) * 2 * tr[..., 1] / tr[..., 2])
    xbx = torch.sign(torch.relu(bx[..., 1] - torch.abs((st - bx[..., 0]) * 2 * bx[..., 1] / bx[..., 2]))) * bx[..., 1]

    x = torch.cat((xtr, xbx), 2)

    u = F.max_pool1d(x.sign().permute(0, 2, 1), kernel_size = 2, stride = 1).permute(0, 2, 1)

    collisions = (u.sum(2) > 1).max(1).values
    y = x.max(2).values

    return y + torch.rand_like(y) * noise_level - noise_level / 2, collisions


def generate_sequences(nb: int,
                       seq_length: int,
                       group_by_locations: bool = False,
                       seq_height_min: float = 1.0,
                       seq_height_max: float = 25.0,
                       seq_width_min: float = 5.0,
                       seq_width_max: float = 11.0) -> Tuple[torch.Tensor]:
    
    # Position / height / width
    tr = torch.empty(nb, 2, 3)
    tr[:, :, 0].uniform_(seq_width_max/2, seq_length - seq_width_max/2)
    tr[:, :, 1].uniform_(seq_height_min, seq_height_max)
    tr[:, :, 2].uniform_(seq_width_min, seq_width_max)

    bx = torch.empty(nb, 2, 3)
    bx[:, :, 0].uniform_(seq_width_max/2, seq_length - seq_width_max/2)
    bx[:, :, 1].uniform_(seq_height_min, seq_height_max)
    bx[:, :, 2].uniform_(seq_width_min, seq_width_max)

    if group_by_locations:
        a = torch.cat((tr, bx), 1)
        v = a[:, :, 0].sort(1).values[:, 2:3]
        mask_left = (a[:, :, 0] < v).float()
        h_left = (a[:, :, 1] * mask_left).sum(1) / 2
        h_right = (a[:, :, 1] * (1 - mask_left)).sum(1) / 2
        valid = (h_left - h_right).abs() > 4
    else:
        valid = (torch.abs(tr[:, 0, 1] - tr[:, 1, 1]) > 4) & (torch.abs(tr[:, 0, 1] - tr[:, 1, 1]) > 4)

    input, collisions = positions_to_sequences(seq_length, tr, bx)

    if group_by_locations:
        a = torch.cat((tr, bx), 1)
        v = a[:, :, 0].sort(1).values[:, 2:3]
        mask_left = (a[:, :, 0] < v).float()
        h_left = (a[:, :, 1] * mask_left).sum(1, keepdim = True) / 2
        h_right = (a[:, :, 1] * (1 - mask_left)).sum(1, keepdim = True) / 2
        a[:, :, 1] = mask_left * h_left + (1 - mask_left) * h_right
        tr, bx = a.split(2, 1)
    else:
        tr[:, :, 1:2] = tr[:, :, 1:2].mean(1, keepdim = True)
        bx[:, :, 1:2] = bx[:, :, 1:2].mean(1, keepdim = True)

    targets, _ = positions_to_sequences(seq_length, tr, bx)

    valid = valid & ~collisions
    tr = tr[valid]
    bx = bx[valid]
    input = input[valid][:, None, :]
    targets = targets[valid][:, None, :]

    if input.size(0) < nb:
        input2, targets2, tr2, bx2 = generate_sequences(nb - input.size(0), seq_length, group_by_locations)
        input = torch.cat((input, input2), 0)
        targets = torch.cat((targets, targets2), 0)
        tr = torch.cat((tr, tr2), 0)
        bx = torch.cat((bx, bx2), 0)

    return input, targets, tr, bx


def plot_sequence_images(sequences: torch.Tensor, tr = None, bx = None, seq_length=100, seq_height_max=25.0):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    ax.set_xlim(0, seq_length)
    ax.set_ylim(-1, seq_height_max + 4)

    for u in sequences:
        ax.plot(
            torch.arange(u[0].size(0)) + 0.5, u[0], color = u[1], label = u[2]
        )

    ax.legend(frameon = False, loc = 'upper left')

    delta = -1.
    if tr is not None:
        ax.scatter(test_tr[k, :, 0], torch.full((test_tr.size(1),), delta), color = 'black', marker = '^', clip_on=False)

    if bx is not None:
        ax.scatter(test_bx[k, :, 0], torch.full((test_bx.size(1),), delta), color = 'black', marker = 's', clip_on=False)

    plt.show()
