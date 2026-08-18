import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class DynamicOTThresh(nn.Module):
    """
    多策略阈值模块：
        mode = "topk"   →  Top-K 自适应
        mode = "global" →  全局分位数
        mode = "sigmoid"→  软判定 (sigmoid)
        mode = "learn"  →  可训练标量阈值
        mode = "otsu"   →  Otsu 自分割
    """
    def __init__(
        self,
        mode: str = "topk",
        k_ratio: float = 0.1,       # top-k 用
        q: float = 0.9,             # global quantile 用
        init_alpha: float = 10.0    # sigmoid 温度
    ):
        super().__init__()
        self.mode = mode.lower()
        self.k_ratio = k_ratio
        self.q = q

        if self.mode == "sigmoid":
            # α 可训练，让网络自己调陡峭度
            self.alpha = nn.Parameter(torch.tensor(init_alpha))
        elif self.mode == "learn":
            # γ, β 可分别学 now->past 与 past->now
            self.gamma_now = nn.Parameter(torch.tensor(0.5))
            self.gamma_past = nn.Parameter(torch.tensor(0.5))

    # ───────────────────────────────────────────────
    # 主接口：传入两个方向的匹配强度，输出 new_mask / disappear_mask
    # C_now2past, C_past2now shape = [B, N]
    # ───────────────────────────────────────────────
    def forward(self, C_now2past: torch.Tensor,
                      C_past2now: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        if self.mode == "topk":
            # 每张片单独取 Top-k% 作为显著
            τ_now  = torch.quantile(C_now2past, 1 - self.k_ratio, dim=1, keepdim=True)
            τ_past = torch.quantile(C_past2now, 1 - self.k_ratio, dim=1, keepdim=True)

            new_mask       = (C_now2past > τ_now)  & (C_past2now <= τ_past)
            disappear_mask = (C_past2now > τ_past) & (C_now2past <= τ_now)

        elif self.mode == "global":
            # 在初始化或每个 epoch 统计一次全局分位数
            τ_now  = torch.quantile(C_now2past.flatten(), self.q)
            τ_past = torch.quantile(C_past2now.flatten(), self.q)
            new_mask       = (C_now2past > τ_now)  & (C_past2now <= τ_past)
            disappear_mask = (C_past2now > τ_past) & (C_now2past <= τ_now)

        elif self.mode == "sigmoid":
            # 连续概率 → 0.5 判定，或直接输出 prob
            prob_new       = torch.sigmoid(self.alpha * (C_now2past - C_past2now))
            prob_disappear = 1 - prob_new
            new_mask       = prob_new       > 0.5
            disappear_mask = prob_disappear > 0.5
            # 你也可以返回 prob_new 作为 soft-mask

        elif self.mode == "learn":
            # learnable γ_now / γ_past，经 softplus 保正
            τ_now  = F.softplus(self.gamma_now)
            τ_past = F.softplus(self.gamma_past)
            new_mask       = (C_now2past > τ_now)  & (C_past2now <= τ_past)
            disappear_mask = (C_past2now > τ_past) & (C_now2past <= τ_now)

        elif self.mode == "otsu":
            # 逐样本 Otsu；用 numpy 版本即可
            new_mask, disappear_mask = [], []
            for b in range(C_now2past.size(0)):
                τ_now  = _otsu(C_now2past[b])
                τ_past = _otsu(C_past2now[b])
                new_mask.append((C_now2past[b] > τ_now)  & (C_past2now[b] <= τ_past))
                disappear_mask.append((C_past2now[b] > τ_past) & (C_now2past[b] <= τ_now))
            new_mask       = torch.stack(new_mask)
            disappear_mask = torch.stack(disappear_mask)

        else:
            raise ValueError(f"Unknown mode {self.mode}")

        return new_mask, disappear_mask


# ──── 辅助：简版 Otsu（单张 patch 向量）────
def _otsu(vec: torch.Tensor) -> float:
    """ vec: [N] """
    # 直方图 256 bins
    h = torch.histc(vec, bins=256, min=float(vec.min()), max=float(vec.max()))
    p = h / h.sum()
    omega = torch.cumsum(p, 0)
    mu = torch.cumsum(p * torch.arange(256, device=vec.device), 0)
    mu_t = mu[-1]
    sigma_b = (mu_t * omega - mu) ** 2 / (omega * (1 - omega) + 1e-8)
    k = torch.argmax(sigma_b)
    thr = vec.min() + (vec.max() - vec.min()) * (k.item() / 255.)
    return thr
