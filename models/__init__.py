#!/usr/bin/env python3
"""
SFAFNet-DiT model (self-contained)
"""
from .blocks import (_safe_groups, ConvBNAct, ChannelAttention, InterferenceLocalizationHead, REMHead,
                     peak_nms, ResBlock, SFAFBlock, ResCompress, _TransformerBlock,
                     ConditionEncoder, ZeroInitProjection)
from .sfafnet import SharedEncoder, SFAFDecoder, RHead, DecompHeadsTF, DecompSFAFNetTF
from .physics_dit import TimeEmbed, DiTBlock, PhysicsDiT, DiffusionSchedule
from .prior_encoder import PhysicsPriorEncoder

__all__ = ["DecompSFAFNetTF", "PhysicsDiT", "DiffusionSchedule", "PhysicsPriorEncoder",
           "SharedEncoder", "SFAFDecoder", "RHead", "DecompHeadsTF", "TimeEmbed", "DiTBlock",
           "SFAFBlock", "ResBlock", "ConvBNAct", "ChannelAttention", "REMHead",
           "InterferenceLocalizationHead", "ConditionEncoder", "ZeroInitProjection",
           "ResCompress", "peak_nms", "_safe_groups"]
