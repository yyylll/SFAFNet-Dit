#!/usr/bin/env python3
"""TotalFirstLoss — the total-first / decompose-after objective, split into the TWO CAGrad task groups."""
import torch
import torch.nn as nn
from losses import DecompLoss


def _masked_mse(pred, tgt, building):
    valid = (building < 0.5).float()
    return ((pred - tgt) ** 2 * valid).sum() / valid.sum().clamp(min=1.0)


class TotalFirstLoss(nn.Module):
    def __init__(self, mu_total_light: float = 0.1, lambda_cons: float = 0.1, **decomp_kw):
        super().__init__()
        self.lambda_cons = lambda_cons
        decomp_kw.setdefault("mu_total", mu_total_light)
        self.decomp = DecompLoss(**decomp_kw)

    def consistency(self, combine, R_direct, building):
        """Hole A: λ_cons · masked_mse(combine(Ŝ,Î), R_direct.detach()) — grad flows to the decomposition only."""
        return self.lambda_cons * _masked_mse(combine, R_direct.detach(), building)

    def forward(self, out, batch, r_only=False):
        """Returns (L_R, L_decomp_group, parts)."""
        rem = batch["rem"]; bld = batch["building_mask"]
        L_R = _masked_mse(out["R_direct"], rem, bld)
        if r_only:
            return L_R, None, {"L_R": float(L_R), "L_cons": 0.0}
        L_dec, parts = self.decomp(
            S_pred=out["S"], I_pred=out["I"], R_pred=out["R_phys"], SINR_pred=out["SINR"],
            H_pred=out["H_IN"], gt_positions=batch["gt_interference_positions"],
            gt_signal=batch["gt_signal"], gt_interf=batch["gt_interf"],
            rem=rem, gt_sinr=batch["gt_sinr"], building_mask=bld, logvar_pred=out["logvar"])
        L_cons = self.consistency(out["R_phys"], out["R_direct"], bld)
        L_decomp_group = L_dec + L_cons
        parts = dict(parts); parts["L_R"] = float(L_R); parts["L_cons"] = float(L_cons)
        return L_R, L_decomp_group, parts
