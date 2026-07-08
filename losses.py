#!/usr/bin/env python3
"""
Multi-objective loss 
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

_SOBEL_X = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3)
_SOBEL_Y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).view(1, 1, 3, 3)


def gradient_magnitude(R):
    kx = _SOBEL_X.to(R.device, R.dtype); ky = _SOBEL_Y.to(R.device, R.dtype)
    gx = F.conv2d(R, kx, padding=1); gy = F.conv2d(R, ky, padding=1)
    return torch.sqrt(gx ** 2 + gy ** 2 + 1e-12)


def boundary_weighted_grad_loss(R_ref, R, building_mask, snr_weight=None, lambda_bnd=2.0):
    band = (F.max_pool2d(building_mask, 3, stride=1, padding=1) - building_mask).clamp(0, 1)
    w = 1.0 + lambda_bnd * band
    se = (gradient_magnitude(R_ref) - gradient_magnitude(R)) ** 2
    per_sample = (w * se).flatten(1).sum(1) / (w.flatten(1).sum(1) + 1e-8)
    if snr_weight is not None:
        per_sample = snr_weight.to(per_sample.device, per_sample.dtype).view(-1) * per_sample
    return per_sample.mean()

def shadow_band_amplitude_loss(R_ref, R, building_mask, band=4, snr_weight=None):
    k = 2 * band + 1
    dil = (F.max_pool2d(building_mask, k, stride=1, padding=band) > 0.5).float()
    shadow = dil * (1.0 - building_mask)
    se = (R_ref - R) ** 2
    per_sample = (se * shadow).flatten(1).sum(1) / (shadow.flatten(1).sum(1) + 1e-8)
    if snr_weight is not None:
        per_sample = snr_weight.to(per_sample.device, per_sample.dtype).view(-1) * per_sample
    return per_sample.mean()

def gradient_consistency_loss(R_ref, R, weight=None):
    if weight is None:
        return F.mse_loss(gradient_magnitude(R_ref), gradient_magnitude(R))
    se = (gradient_magnitude(R_ref) - gradient_magnitude(R)) ** 2
    per_sample = se.flatten(1).mean(1)
    w = weight.to(per_sample.device, per_sample.dtype).view(-1)
    return (w * per_sample).mean()


def gaussian_splat_heatmap(positions, H, W, sigma=2.0, device=None):
    B = len(positions)
    device = device or (positions[0].device if torch.is_tensor(positions[0]) and positions[0].numel() else "cpu")
    ys = torch.arange(H, device=device).view(1, H, 1).float()
    xs = torch.arange(W, device=device).view(1, 1, W).float()
    out = torch.zeros(B, 1, H, W, device=device); two = 2 * sigma * sigma
    for b in range(B):
        p = positions[b]
        if not torch.is_tensor(p):
            p = torch.as_tensor(p, device=device, dtype=torch.float32)
        p = p.view(-1, 2)
        p = p[(p[:, 0] >= 0) & (p[:, 1] >= 0)]
        if p.shape[0] == 0:
            continue
        g = torch.exp(-((ys - p[:, 0].view(-1, 1, 1)) ** 2 + (xs - p[:, 1].view(-1, 1, 1)) ** 2) / two)
        out[b, 0] = g.max(0).values
    return out


def localization_loss(H_pred, H_gt, kappa=4.0, building_mask=None):
    w = 1.0 + kappa * H_gt
    se = (H_pred - H_gt) ** 2 * w
    if building_mask is not None:
        v = (building_mask < 0.5).float()
        return (se * v).sum() / (v.sum() + 1e-6)
    return se.mean()


def interference_power_loss(P_dBm, gt_positions, gt_powers, lambda_bg=0.05, src_radius=2):
    B, _, H, W = P_dBm.shape
    device = P_dBm.device
    sup_terms = []
    src_mask = torch.zeros(B, 1, H, W, device=device)
    for b in range(B):
        pos = gt_positions[b]; pw = gt_powers[b]
        if not torch.is_tensor(pos):
            pos = torch.as_tensor(pos, device=device, dtype=torch.float32)
        if not torch.is_tensor(pw):
            pw = torch.as_tensor(pw, device=device, dtype=torch.float32)
        pos = pos.view(-1, 2)
        keep = (pos[:, 0] >= 0) & (pos[:, 1] >= 0)
        pos = pos[keep]; pw = pw.view(-1)[keep]
        for i in range(pos.shape[0]):
            y = int(pos[i, 0].item()); x = int(pos[i, 1].item())
            if 0 <= y < H and 0 <= x < W:
                sup_terms.append((P_dBm[b, 0, y, x] - pw[i]) ** 2)
                y0, y1 = max(0, y - src_radius), min(H, y + src_radius + 1)
                x0, x1 = max(0, x - src_radius), min(W, x + src_radius + 1)
                src_mask[b, 0, y0:y1, x0:x1] = 1.0
    sup = torch.stack(sup_terms).mean() if sup_terms else P_dBm.sum() * 0.0
    bg = (1.0 - src_mask)
    bg_pen = (torch.pow(10.0, P_dBm / 10.0) * bg).sum() / (bg.sum() + 1e-6)
    return sup + lambda_bg * bg_pen


def focal_localization_loss(H_pred, H_gt, alpha=2.0, beta=4.0, eps=1e-6):
    H_pred = H_pred.clamp(eps, 1 - eps)
    pos = (H_gt >= 1.0 - 1e-3).float()
    neg = 1.0 - pos
    pos_loss = -((1 - H_pred) ** alpha) * torch.log(H_pred) * pos
    neg_loss = -((1 - H_gt) ** beta) * (H_pred ** alpha) * torch.log(1 - H_pred) * neg
    n_pos = pos.sum().clamp(min=1.0)
    return (pos_loss.sum() + neg_loss.sum()) / n_pos

def focus_loss(R_ref, R, focus_mask, weight=None):
    if weight is None:
        n = focus_mask.sum()
        if n == 0:
            return torch.zeros((), device=R_ref.device)
        return (focus_mask * (R_ref - R).abs()).sum() / n
    num = (focus_mask * (R_ref - R).abs()).flatten(1).sum(1)
    den = focus_mask.flatten(1).sum(1) + 1e-8
    w = weight.to(num.device, num.dtype).view(-1)
    return (w * (num / den)).mean()


class SFAFNetDiTLoss(nn.Module):
    def __init__(self, l1=1.0, l2=0.5, l3=0.5, l4=0.5, kappa=4.0, sigma_loc=2.0):
        super().__init__()
        self.l1, self.l2, self.l3, self.l4, self.kappa, self.sigma = l1, l2, l3, l4, kappa, sigma_loc
    def forward(self, *, eps, eps_pred, R_ref, R, H_pred, gt_positions,
                base_mask=None, inter_mask=None, building_mask=None, aux_weight=None):
        H = R.shape[-2]; W = R.shape[-1]
        L_diff = F.mse_loss(eps_pred, eps)
        L_grad = gradient_consistency_loss(R_ref, R, weight=aux_weight)
        H_gt = gaussian_splat_heatmap(gt_positions, H, W, self.sigma, device=R.device)
        L_loc = localization_loss(H_pred, H_gt, self.kappa, building_mask)
        L_base = focus_loss(R_ref, R, base_mask, weight=aux_weight) if base_mask is not None else torch.zeros((), device=R.device)
        L_int = focus_loss(R_ref, R, inter_mask, weight=aux_weight) if inter_mask is not None else torch.zeros((), device=R.device)
        total = L_diff + self.l1 * L_grad + self.l2 * L_base + self.l3 * L_int + self.l4 * L_loc
        return total, {"diffusion": L_diff.item(), "grad": L_grad.item(), "loc": L_loc.item(),
                       "focus_base": float(L_base), "focus_inter": float(L_int), "total": total.item()}


def _masked_mse(a, b, valid):
    se = (a - b) ** 2
    if valid is None:
        return se.mean()
    return (se * valid).sum() / (valid.sum() + 1e-8)

def uncertainty_loss(R_pred, rem, logvar, building_mask=None, clamp=(-7.0, 7.0)):
    s = logvar.clamp(*clamp)
    nll = 0.5 * torch.exp(-s) * (R_pred - rem) ** 2 + 0.5 * s
    valid = (building_mask < 0.5).float() if building_mask is not None else None
    if valid is None:
        return nll.mean()
    return (nll * valid).sum() / (valid.sum() + 1e-8)


class DecompLoss(nn.Module):
    def __init__(self, mu_total=1.0, mu_sinr=0.5, lambda_loc=0.5, kappa=4.0, sigma_loc=2.0,
                 lambda_bnd=0.0, lambda_power=0.0, lambda_bg=0.05, use_focal_loc=False,
                 lambda_grad=0.0, lambda_grad_bnd=2.0, lambda_unc=0.0):
        super().__init__()
        self.mu_total, self.mu_sinr, self.lambda_loc = mu_total, mu_sinr, lambda_loc
        self.kappa, self.sigma, self.lambda_bnd = kappa, sigma_loc, lambda_bnd
        self.lambda_power, self.lambda_bg, self.use_focal_loc = lambda_power, lambda_bg, use_focal_loc
        self.lambda_grad, self.lambda_grad_bnd = lambda_grad, lambda_grad_bnd
        self.lambda_unc = lambda_unc

    def forward(self, *, S_pred, I_pred, R_pred, SINR_pred, H_pred, gt_positions,
                gt_signal, gt_interf, rem, gt_sinr, building_mask=None, P_pred=None, gt_powers=None,
                logvar_pred=None):
        H, W = rem.shape[-2], rem.shape[-1]
        valid = (building_mask < 0.5).float() if building_mask is not None else None
        w_s = valid
        if valid is not None and self.lambda_bnd > 0 and building_mask is not None:
            band = (F.max_pool2d(building_mask, 3, 1, 1) - building_mask).clamp(0, 1)
            w_s = valid * (1.0 + self.lambda_bnd * band)
        L_s = _masked_mse(S_pred, gt_signal, w_s)
        L_i = _masked_mse(I_pred, gt_interf, valid)
        L_t = _masked_mse(R_pred, rem, valid)
        L_sinr = _masked_mse(SINR_pred, gt_sinr, valid)
        H_gt = gaussian_splat_heatmap(gt_positions, H, W, self.sigma, device=rem.device)
        if self.use_focal_loc:
            L_loc = focal_localization_loss(H_pred, H_gt)
        else:
            L_loc = localization_loss(H_pred, H_gt, self.kappa, building_mask)
        L_power = (interference_power_loss(P_pred, gt_positions, gt_powers, self.lambda_bg)
                   if (self.lambda_power > 0 and P_pred is not None and gt_powers is not None)
                   else R_pred.sum() * 0.0)
        L_grad_s = (boundary_weighted_grad_loss(S_pred, gt_signal, building_mask, lambda_bnd=self.lambda_grad_bnd)
                    if building_mask is not None else S_pred.sum() * 0.0)
        L_unc = (uncertainty_loss(R_pred, rem, logvar_pred, building_mask)
                 if (self.lambda_unc > 0 and logvar_pred is not None) else R_pred.sum() * 0.0)
        total = (L_s + L_i + self.mu_total * L_t + self.mu_sinr * L_sinr
                 + self.lambda_loc * L_loc + self.lambda_power * L_power + self.lambda_grad * L_grad_s
                 + self.lambda_unc * L_unc)
        return total, {"sig": float(L_s), "interf": float(L_i), "total_rem": float(L_t),
                       "sinr": float(L_sinr), "loc": float(L_loc), "power": float(L_power),
                       "grad_s": float(L_grad_s), "unc": float(L_unc), "total": float(total)}


if __name__ == "__main__":
    torch.manual_seed(0)
    B, H, W = 2, 64, 64
    crit = SFAFNetDiTLoss()
    eps = torch.randn(B, 1, H, W); eps_pred = (eps + 0.1 * torch.randn_like(eps)).requires_grad_(True)
    R = torch.randn(B, 1, H, W); R_ref = (R + 0.1 * torch.randn_like(R)).requires_grad_(True)
    H_pred = (torch.rand(B, 1, H, W) * 0.1).requires_grad_(True)
    gt_pos = [torch.tensor([[20., 30.]]), torch.tensor([[40., 12.], [12., 50.]])]
    bld = (torch.rand(B, 1, H, W) > 0.9).float()
    base = (torch.rand(B, 1, H, W) > 0.8).float(); inter = (torch.rand(B, 1, H, W) > 0.9).float()
    total, log = crit(eps=eps, eps_pred=eps_pred, R_ref=R_ref, R=R, H_pred=H_pred,
                      gt_positions=gt_pos, base_mask=base, inter_mask=inter, building_mask=bld)
    print("loss components:", {k: round(v, 4) for k, v in log.items()})
    assert total.item() >= 0 and all(v >= 0 for v in log.values())
    total.backward()
