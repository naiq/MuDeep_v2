import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def TripHard(feature, p, k, norm=True):
    if norm:
        fnorm = torch.norm(feature, p=2, dim=1, keepdim=True)
        feature = feature.div(fnorm.expand_as(feature))  # normalization
    feat1 = torch.unsqueeze(feature, dim=1).expand(feature.size(0), feature.size(0), feature.size(1)) # k*p x k*p x feature_dim
    feat2 = torch.unsqueeze(feature, dim=0).expand(feature.size(0), feature.size(0), feature.size(1)) # k*p x k*p x feature_dim
    delta = feat1 - feat2
    distmat = torch.sqrt(torch.sum(delta**2, dim=2) + 1e-8)    # k*p x k*p
                                                                # 1e-8: Avoid gradients becoming NAN

    positive = distmat[0:k, 0:k]  # torch.cat([(k x k), (k x k), ...], dim=0) = p*k x k
    negative = distmat[0:k, k:]   # torch.cat([(k x (p*k-k)), (k x (p*k-k)), ...], dim=0) = p*k x (p*k-k)
    for i in range(1, p):
        positive = torch.cat([positive, distmat[i*k:(i+1)*k, i*k:(i+1)*k]], dim=0)
        if i != p-1:
            negs = torch.cat([distmat[i*k:(i+1)*k, 0:i*k], distmat[i*k:(i+1)*k, (i+1)*k:]], dim=1)
        else:
            negs = distmat[i*k:(i+1)*k, 0:i*k]
        negative = torch.cat([negative, negs], dim=0)
    positive, _ = torch.max(positive, dim=1)    # k*p x 1
    negative, _ = torch.min(negative, dim=1)    # k*p x 1

    return positive, negative

