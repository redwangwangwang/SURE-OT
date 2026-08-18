from .dynamic_selection import DynamicOTThresh
import torch.nn.functional as F
import torch

class SinkhornOTChangeDetector:
    def __init__(self, eps=0.1, max_iter=50, thresh_mode=0.05, k_ratio=0.1, global_q=0.9, init_alpha=10.0):
        self.eps = eps
        self.max_iter = max_iter
        # self.thresh = DynamicOTThresh(
        #     mode=thresh_mode,
        #     k_ratio=k_ratio,
        #     q=global_q,
        #     init_alpha=init_alpha
        # )
        self.thresh = thresh_mode

    def compute_sinkhorn_transport(self, source_embeds, target_embeds, return_loss=False):
        """
        计算 OT transport plan 和 Wasserstein 距离
        source_embeds: [B, N, D]
        target_embeds: [B, N, D]
        """
        B, N, D = source_embeds.shape

        source = F.normalize(source_embeds, dim=-1)
        target = F.normalize(target_embeds, dim=-1)

        cost = torch.cdist(source, target, p=2).pow(2)  # [B, N, N]
        mu = torch.full((B, N), 1.0 / N, device=source.device)
        nu = torch.full((B, N), 1.0 / N, device=target.device)

        u = torch.ones_like(mu)
        v = torch.ones_like(nu)

        K = torch.exp(-cost / self.eps)

        for _ in range(self.max_iter):
            u = mu / (K @ v.unsqueeze(-1)).squeeze(-1)
            v = nu / (K.transpose(1, 2) @ u.unsqueeze(-1)).squeeze(-1)

        transport_plan = u.unsqueeze(-1) * K * v.unsqueeze(1)  # [B, N, N]
        ot_distance = torch.sum(transport_plan * cost, dim=[1, 2])

        if return_loss:
            # 计算熵  H(T*) = -Σ T log T
            entropy = -(transport_plan.clamp_min(1e-12) * transport_plan.log()).sum(dim=[1, 2])
            sinkhorn_loss = ot_distance - self.eps * entropy  # 论文里的 d_OT
            return transport_plan, sinkhorn_loss
        else:
            return transport_plan, ot_distance

    @staticmethod
    def _get_patch_mass(plan_p2n, plan_n2p):
        """
        根据双向 transport plan 计算两个方向的匹配强度向量
            plan_p2n: [B, N_past, N_now]   (历史→当前)
            plan_n2p: [B, N_now, N_past]   (当前→历史)
        返回：
            C_now2past: [B, N_now]   当前 patch → 历史 发送质量
            C_past2now: [B, N_now]   当前 patch ← 历史 接收质量
        """
        # 当前 patch 向过去发送质量（沿 past 维求和）
        C_now2past = plan_n2p.sum(dim=2)  # [B, N_now]
        # 历史 patch 向当前发送质量，再聚合到当前 patch（沿 past 维求和）
        C_past2now = plan_p2n.sum(dim=1)  # [B, N_now]
        return C_now2past, C_past2now

    def extract_change_masks(self, plan_p2n, plan_n2p):
        """
        plan_p2n: [B, N_past, N_now]   (历史 → 当前)
        plan_n2p: [B, N_now, N_past]   (当前 → 历史)
        返回 new_mask / disappear_mask 形状统一为 [B, N_now]
        """
        # ① 当前 patch 主动发送给历史（沿 past 维求和）
        mass_n2p_sent = plan_n2p.sum(dim=2)  # [B, N_now]

        # # ② 当前 patch 从历史接收（沿 past 维求和）
        mass_p2n_recv = plan_p2n.sum(dim=1)  # [B, N_now]

        # ③ 历史 patch 主动发送给当前
        mass_p2n_sent = plan_p2n.sum(dim=2)  # [B, N_past]

        # # ④ 历史 patch 从当前接收
        mass_n2p_recv = plan_n2p.sum(dim=1)  # [B, N_past]

        # # # ── 联合不对称判定 ─────────────────────────
        # # # 新增：当前主动发出多，历史回馈少
        new_mask = (mass_n2p_sent > self.thresh) & (mass_p2n_recv < self.thresh)  # [B, N_now]
        #
        # # 消失：历史主动发出多，当前接收少
        disappear_mask = (mass_p2n_sent > self.thresh) & (mass_n2p_recv < self.thresh)  # [B, N_past]

        return new_mask, disappear_mask

    def build_prompt(self, new_mask, disappear_mask, region_names=None):
        """
        构造每个样本的变化描述 prompt
        """
        B, N = new_mask.shape
        prompts = []

        for b in range(B):
            new_indices = torch.where(new_mask[b])[0].tolist()
            disappear_indices = torch.where(disappear_mask[b])[0].tolist()
            phrases = []

            # —— ① 有解剖区域名的情况 ——
            if region_names is not None:
                if new_indices:
                    new_regions = [region_names[i] for i in new_indices]
                    phrases.append(f"New changes appear in: {', '.join(new_regions)}.")
                if disappear_indices:
                    dis_regions = [region_names[i] for i in disappear_indices]
                    phrases.append(f"Previous findings have resolved in: {', '.join(dis_regions)}.")
            # —— ② 无解剖区域名 → 直接列 patch 索引 ——
            else:
                if new_indices:
                    phrases.append(f"New abnormal patches at indices: {new_indices}.")
                if disappear_indices:
                    phrases.append(f"Resolved abnormal patches from previous scan at indices: {disappear_indices}.")

            prompt = (
                "Compared to the previous scan, " + " ".join(phrases)
                if phrases else "No significant change detected."
            )
            prompts.append(prompt)
        return prompts

    def forward(self, current_embeds, history_embeds):
        """
        全流程执行：
        输入：
            current_embeds: [B, N, D]
            history_embeds: [B, N, D]
        输出：
            prompts: List[str]
            new_mask: [B, N]
            disappear_mask: [B, N]
        """
        plan_p2n, sinkhorn_loss_p2n = self.compute_sinkhorn_transport(history_embeds, current_embeds)
        plan_n2p, sinkhorn_loss_n2p = self.compute_sinkhorn_transport(current_embeds, history_embeds)

        new_mask, disappear_mask = self.extract_change_masks(plan_p2n, plan_n2p)

        # # 动态阈值下使用
        # # 得到两个方向的匹配强度
        # C_now2past, C_past2now = self._get_patch_mass(plan_p2n, plan_n2p)
        #
        # # 通过动态阈值模块得到 mask
        # new_mask, disappear_mask = self.thresh(C_now2past, C_past2now)

        prompts = self.build_prompt(new_mask, disappear_mask)

        return prompts, new_mask, disappear_mask, sinkhorn_loss_p2n, sinkhorn_loss_n2p