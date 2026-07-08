#!/usr/bin/env python3
""" 
physics-prior encoder
"""
from __future__ import annotations
import torch
import torch.nn as nn


class PhysicsPriorEncoder(nn.Module):

    def forward(self, S, I, SINR, building_mask, I_field, k2=None):
        feats = [S, I, SINR, building_mask, I_field]
        if k2 is not None:
            feats.append(k2)
        return torch.cat(feats, dim=1)
