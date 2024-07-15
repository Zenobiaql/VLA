# Copyright (c) Meta Platforms, Inc. All Rights Reserved

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

def shift_dim(x, src_dim=-1, dest_dim=-1, make_contiguous=True):
    n_dims = len(x.shape)
    if src_dim < 0:
        src_dim = n_dims + src_dim
    if dest_dim < 0:
        dest_dim = n_dims + dest_dim

    assert 0 <= src_dim < n_dims and 0 <= dest_dim < n_dims

    dims = list(range(n_dims))
    del dims[src_dim]

    permutation = []
    ctr = 0
    for i in range(n_dims):
        if i == dest_dim:
            permutation.append(src_dim)
        else:
            permutation.append(dims[ctr])
            ctr += 1
    x = x.permute(permutation)
    if make_contiguous:
        x = x.contiguous()
    return x

class Codebook(nn.Module):
    def __init__(self, n_codes, embedding_dim):
        super().__init__()
        self.register_buffer('embeddings', torch.randn(n_codes, embedding_dim))
    #     self.register_buffer('N', torch.zeros(n_codes))
    #     self.register_buffer('z_avg', self.embeddings.data.clone())

        self.n_codes = n_codes
        self.embedding_dim = embedding_dim
    #     self._need_init = True
    #     self.no_random_restart = no_random_restart
    #     self.restart_thres = restart_thres

    # def _tile(self, x):
    #     d, ew = x.shape
    #     if d < self.n_codes:
    #         n_repeats = (self.n_codes + d - 1) // d
    #         std = 0.01 / np.sqrt(ew)
    #         x = x.repeat(n_repeats, 1)
    #         x = x + torch.randn_like(x) * std
    #     return x

    # def _init_embeddings(self, z):
    #     # z: [b, c, t, h, w]
    #     self._need_init = False
    #     flat_inputs = shift_dim(z, 1, -1).flatten(end_dim=-2)
    #     y = self._tile(flat_inputs)

    #     d = y.shape[0]
    #     _k_rand = y[torch.randperm(y.shape[0])][:self.n_codes]
    #     if dist.is_initialized():
    #         dist.broadcast(_k_rand, 0)
    #     self.embeddings.data.copy_(_k_rand)
    #     self.z_avg.data.copy_(_k_rand)
    #     self.N.data.copy_(torch.ones(self.n_codes))

    def forward(self, encodings):
        embeddings = F.embedding(encodings, self.embeddings)
        return embeddings
