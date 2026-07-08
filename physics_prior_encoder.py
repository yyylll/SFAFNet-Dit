#!/usr/bin/env python3
"""
render the interference apply the physics-prior encoder.
"""
from __future__ import annotations
import physics_prior as PP
from models.prior_encoder import PhysicsPriorEncoder

_DEFAULT_ENCODER = PhysicsPriorEncoder()         
def assemble_decomp_mphy(S, I, SINR, building_mask, q_IN_list, sigma=4.0, k2=None, _encoder=None):
    H, W = S.shape[-2], S.shape[-1]
    I_field = PP.interference_field_from_estimates(q_IN_list, H, W, sigma=sigma, device=S.device)
    enc = _encoder if _encoder is not None else _DEFAULT_ENCODER
    return enc(S, I, SINR, building_mask, I_field, k2=k2)
