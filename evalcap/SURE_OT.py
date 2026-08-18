"""SURE-OT modules for longitudinal radiology report generation.

This file implements:
1. entropic unbalanced optimal transport with relaxed marginals;
2. birth/resolution residual maps from unmatched transport mass;
3. continuous evolution tokens for LLM prompting;
4. temporal-swap consistency without additional dataset annotations.
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class RoleAdapter(nn.Module):
    """Small residual adapter that encodes the temporal role of a feature."""

    def __init__(self, dim: int, rank: int) -> None:
        super().__init__()
        rank = max(8, min(rank, dim))
        self.norm = nn.LayerNorm(dim)
        self.down = nn.Linear(dim, rank, bias=False)
        self.act = nn.GELU()
        self.up = nn.Linear(rank, dim, bias=False)
        nn.init.normal_(self.up.weight, mean=0.0, std=1e-3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.up(self.act(self.down(self.norm(x))))


class BottleneckProjector(nn.Module):
    """Parameter-efficient projector used for persistent/uncertain evidence."""

    def __init__(self, dim: int, rank: int) -> None:
        super().__init__()
        rank = max(8, min(rank, dim))
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, rank, bias=False),
            nn.GELU(),
            nn.Linear(rank, dim, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(self.norm(x))


class UnbalancedSinkhorn(nn.Module):
    """Log-domain entropic unbalanced Sinkhorn solver.

    The relaxed marginal exponent is rho=tau/(tau+epsilon). Setting
    ``balanced=True`` recovers the standard balanced Sinkhorn updates.
    """

    def __init__(
        self,
        epsilon: float = 0.07,
        tau: float = 0.7,
        num_iters: int = 40,
        spatial_weight: float = 0.05,
        balanced: bool = False,
    ) -> None:
        super().__init__()
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        if tau <= 0:
            raise ValueError("tau must be positive")
        self.epsilon = float(epsilon)
        self.tau = float(tau)
        self.num_iters = int(num_iters)
        self.spatial_weight = float(spatial_weight)
        self.balanced = bool(balanced)

    @staticmethod
    def _coordinates(
        length: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        side = int(round(math.sqrt(length)))
        if side * side == length:
            axis = torch.linspace(0.0, 1.0, side, device=device, dtype=dtype)
            yy, xx = torch.meshgrid(axis, axis, indexing="ij")
            return torch.stack((yy.reshape(-1), xx.reshape(-1)), dim=-1)

        axis = torch.linspace(0.0, 1.0, length, device=device, dtype=dtype)
        return torch.stack((axis, torch.zeros_like(axis)), dim=-1)

    def _cost(
        self, source: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        source_f = F.normalize(source.float(), dim=-1)
        target_f = F.normalize(target.float(), dim=-1)
        semantic = 1.0 - torch.einsum("bnd,bmd->bnm", source_f, target_f)
        semantic = semantic.clamp_min(0.0)

        if self.spatial_weight <= 0:
            return semantic

        source_xy = self._coordinates(
            source.shape[1], source.device, source_f.dtype
        )
        target_xy = self._coordinates(
            target.shape[1], target.device, target_f.dtype
        )
        spatial = torch.cdist(source_xy, target_xy, p=2).pow(2)
        return semantic + self.spatial_weight * spatial.unsqueeze(0)

    @staticmethod
    def _kl_divergence(
        marginal: torch.Tensor, reference: torch.Tensor
    ) -> torch.Tensor:
        marginal_safe = marginal.clamp_min(1e-12)
        reference_safe = reference.clamp_min(1e-12)
        return (
            marginal_safe * (marginal_safe.log() - reference_safe.log())
            - marginal_safe
            + reference_safe
        ).sum(dim=-1)

    def forward(
        self, source: torch.Tensor, target: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        if source.ndim != 3 or target.ndim != 3:
            raise ValueError("source and target must have shape [B, N, D]")
        if source.shape[0] != target.shape[0]:
            raise ValueError("source and target batch sizes must match")
        if source.shape[-1] != target.shape[-1]:
            raise ValueError("source and target feature dimensions must match")

        batch_size, source_len, _ = source.shape
        target_len = target.shape[1]
        cost = self._cost(source, target)
        dtype = cost.dtype
        device = cost.device

        log_a = torch.full(
            (batch_size, source_len),
            -math.log(source_len),
            device=device,
            dtype=dtype,
        )
        log_b = torch.full(
            (batch_size, target_len),
            -math.log(target_len),
            device=device,
            dtype=dtype,
        )
        log_kernel = -cost / self.epsilon
        log_u = torch.zeros_like(log_a)
        log_v = torch.zeros_like(log_b)
        rho = 1.0 if self.balanced else self.tau / (self.tau + self.epsilon)

        for _ in range(self.num_iters):
            log_u = rho * (
                log_a
                - torch.logsumexp(
                    log_kernel + log_v.unsqueeze(1), dim=2
                )
            )
            log_v = rho * (
                log_b
                - torch.logsumexp(
                    log_kernel.transpose(1, 2) + log_u.unsqueeze(1), dim=2
                )
            )

        log_plan = (
            log_u.unsqueeze(2) + log_kernel + log_v.unsqueeze(1)
        )
        plan = torch.exp(log_plan.clamp(min=-60.0, max=20.0))
        row_mass = plan.sum(dim=2)
        col_mass = plan.sum(dim=1)

        a = log_a.exp()
        b = log_b.exp()
        entropy_term = (
            plan * (plan.clamp_min(1e-12).log() - 1.0)
        ).sum(dim=(1, 2))
        objective = (plan * cost).sum(dim=(1, 2))
        objective = objective + self.epsilon * entropy_term
        if not self.balanced:
            objective = objective + self.tau * (
                self._kl_divergence(row_mass, a)
                + self._kl_divergence(col_mass, b)
            )

        return {
            "plan": plan,
            "cost": cost,
            "row_mass": row_mass,
            "col_mass": col_mass,
            "source_mass": a,
            "target_mass": b,
            "objective": objective.mean(),
        }


class ResidualEvolutionTokenizer(nn.Module):
    """Pool soft residual maps into continuous evolution prompt tokens."""

    NEW = 0
    RESOLVED = 1
    PERSISTENT = 2
    UNCERTAIN = 3
    NUM_TYPES = 4

    def __init__(
        self,
        dim: int,
        num_tokens: int = 2,
        prior_strength: float = 1.0,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.num_tokens = int(num_tokens)
        self.prior_strength = float(prior_strength)

        self.queries = nn.Parameter(
            torch.empty(self.NUM_TYPES, self.num_tokens, self.dim)
        )
        self.type_embeddings = nn.Parameter(
            torch.empty(self.NUM_TYPES, self.num_tokens, self.dim)
        )
        self.output_norm = nn.LayerNorm(self.dim)
        nn.init.normal_(self.queries, std=0.02)
        nn.init.normal_(self.type_embeddings, std=0.02)

    def _pool(
        self,
        features: torch.Tensor,
        prior: torch.Tensor,
        type_index: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        feature_norm = F.normalize(features, dim=-1)
        queries = self.queries[type_index].to(dtype=features.dtype)
        query_norm = F.normalize(queries, dim=-1)
        logits = torch.einsum(
            "bnd,kd->bkn", feature_norm, query_norm
        ) / math.sqrt(self.dim)

        prior_f = prior.float().clamp_min(1e-6)
        prior_f = prior_f / prior_f.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        logits = logits.float() + self.prior_strength * prior_f.log().unsqueeze(1)
        attention = logits.softmax(dim=-1).to(dtype=features.dtype)
        tokens = torch.bmm(attention, features)
        tokens = tokens + self.type_embeddings[type_index].to(
            dtype=features.dtype
        ).unsqueeze(0)
        return self.output_norm(tokens), attention

    def forward(
        self,
        new_features: torch.Tensor,
        new_prior: torch.Tensor,
        resolved_features: torch.Tensor,
        resolved_prior: torch.Tensor,
        persistent_features: torch.Tensor,
        persistent_prior: torch.Tensor,
        uncertain_features: torch.Tensor,
        uncertain_prior: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        new_tokens, new_attention = self._pool(
            new_features, new_prior, self.NEW
        )
        resolved_tokens, resolved_attention = self._pool(
            resolved_features, resolved_prior, self.RESOLVED
        )
        persistent_tokens, persistent_attention = self._pool(
            persistent_features, persistent_prior, self.PERSISTENT
        )
        uncertain_tokens, uncertain_attention = self._pool(
            uncertain_features, uncertain_prior, self.UNCERTAIN
        )

        tokens = torch.cat(
            (
                new_tokens,
                resolved_tokens,
                persistent_tokens,
                uncertain_tokens,
            ),
            dim=1,
        )
        return {
            "tokens": tokens,
            "new_tokens": new_tokens,
            "resolved_tokens": resolved_tokens,
            "persistent_tokens": persistent_tokens,
            "uncertain_tokens": uncertain_tokens,
            "new_attention": new_attention,
            "resolved_attention": resolved_attention,
            "persistent_attention": persistent_attention,
            "uncertain_attention": uncertain_attention,
        }


class SUREOTModule(nn.Module):
    """Swap-consistent Unbalanced Residual Evolution prompting module."""

    def __init__(
        self,
        dim: int,
        num_tokens: int = 2,
        epsilon: float = 0.07,
        tau: float = 0.7,
        num_iters: int = 40,
        spatial_weight: float = 0.05,
        adapter_rank: int = 128,
        prior_strength: float = 1.0,
        residual_threshold: float = 0.25,
        balanced: bool = False,
        use_role_adapters: bool = True,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.num_tokens = int(num_tokens)
        self.residual_threshold = float(residual_threshold)
        self.use_role_adapters = bool(use_role_adapters)

        self.current_role = (
            RoleAdapter(dim, adapter_rank)
            if self.use_role_adapters
            else nn.Identity()
        )
        self.history_role = (
            RoleAdapter(dim, adapter_rank)
            if self.use_role_adapters
            else nn.Identity()
        )
        self.transport = UnbalancedSinkhorn(
            epsilon=epsilon,
            tau=tau,
            num_iters=num_iters,
            spatial_weight=spatial_weight,
            balanced=balanced,
        )
        self.persistent_projector = BottleneckProjector(dim, adapter_rank)
        self.uncertain_projector = BottleneckProjector(dim, adapter_rank)
        self.tokenizer = ResidualEvolutionTokenizer(
            dim=dim,
            num_tokens=num_tokens,
            prior_strength=prior_strength,
        )

    @staticmethod
    def _total_variation(score: torch.Tensor) -> torch.Tensor:
        batch_size, length = score.shape
        side = int(round(math.sqrt(length)))
        if side * side == length:
            grid = score.reshape(batch_size, side, side)
            vertical = (grid[:, 1:, :] - grid[:, :-1, :]).abs().mean()
            horizontal = (grid[:, :, 1:] - grid[:, :, :-1]).abs().mean()
            return vertical + horizontal
        if length <= 1:
            return score.new_zeros(())
        return (score[:, 1:] - score[:, :-1]).abs().mean()

    @staticmethod
    def _cosine_loss(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        return 1.0 - F.cosine_similarity(first, second, dim=-1).mean()

    @staticmethod
    def _normalise_plan(plan: torch.Tensor) -> torch.Tensor:
        return plan / plan.sum(dim=(1, 2), keepdim=True).clamp_min(1e-8)

    def _directional(
        self, current: torch.Tensor, history: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        current_role = self.current_role(current)
        history_role = self.history_role(history)

        current_to_history = self.transport(current_role, history_role)
        history_to_current = self.transport(history_role, current_role)

        current_reference = current_to_history["source_mass"]
        history_reference = history_to_current["source_mass"]

        new_score = (
            (current_reference - current_to_history["row_mass"])
            / current_reference.clamp_min(1e-8)
        ).clamp(0.0, 1.0)
        resolved_score = (
            (history_reference - history_to_current["row_mass"])
            / history_reference.clamp_min(1e-8)
        ).clamp(0.0, 1.0)

        current_sent = (
            current_to_history["row_mass"]
            / current_reference.clamp_min(1e-8)
        ).clamp(0.0, 1.0)
        current_received = (
            history_to_current["col_mass"]
            / current_reference.clamp_min(1e-8)
        ).clamp(0.0, 1.0)
        persistent_score = (0.5 * (current_sent + current_received)).clamp(
            0.0, 1.0
        )

        row_distribution = current_to_history["plan"] / current_to_history[
            "row_mass"
        ].unsqueeze(-1).clamp_min(1e-8)
        row_entropy = -(
            row_distribution
            * row_distribution.clamp_min(1e-8).log()
        ).sum(dim=-1)
        entropy_normaliser = math.log(max(history.shape[1], 2))
        row_entropy = (row_entropy / entropy_normaliser).clamp(0.0, 1.0)
        directional_disagreement = (current_sent - current_received).abs()
        uncertain_score = (
            0.5 * row_entropy * persistent_score
            + 0.5 * directional_disagreement
        ).clamp(0.0, 1.0)

        aligned_history = torch.bmm(
            row_distribution, history_role.float()
        ).to(dtype=current_role.dtype)
        persistent_features = self.persistent_projector(
            0.5 * (current_role + aligned_history)
        )
        uncertain_features = self.uncertain_projector(
            current_role - aligned_history
        )

        token_output = self.tokenizer(
            new_features=current_role,
            new_prior=new_score,
            resolved_features=history_role,
            resolved_prior=resolved_score,
            persistent_features=persistent_features,
            persistent_prior=persistent_score,
            uncertain_features=uncertain_features,
            uncertain_prior=uncertain_score,
        )

        residual_sparsity = new_score.mean() + resolved_score.mean()
        spatial_smoothness = self._total_variation(
            new_score
        ) + self._total_variation(resolved_score)
        regularization_loss = 0.1 * residual_sparsity + spatial_smoothness

        result = {
            **token_output,
            "new_score": new_score,
            "resolved_score": resolved_score,
            "persistent_score": persistent_score,
            "uncertain_score": uncertain_score,
            "new_mask": new_score > self.residual_threshold,
            "resolved_mask": resolved_score > self.residual_threshold,
            "plan_current_to_history": current_to_history["plan"],
            "plan_history_to_current": history_to_current["plan"],
            "transport_loss": (
                current_to_history["objective"]
                + history_to_current["objective"]
            ),
            "regularization_loss": regularization_loss,
        }
        return result

    def forward(
        self,
        current: torch.Tensor,
        history: torch.Tensor,
        compute_swap: bool = True,
    ) -> Dict[str, torch.Tensor]:
        forward_output = self._directional(current, history)
        zero = forward_output["tokens"].new_zeros((), dtype=torch.float32)

        swap_map_loss = zero
        swap_token_loss = zero
        swap_plan_loss = zero

        if compute_swap:
            reverse_output = self._directional(history, current)

            swap_map_loss = F.smooth_l1_loss(
                forward_output["new_score"],
                reverse_output["resolved_score"],
            ) + F.smooth_l1_loss(
                forward_output["resolved_score"],
                reverse_output["new_score"],
            )

            swap_token_loss = (
                self._cosine_loss(
                    forward_output["new_tokens"],
                    reverse_output["resolved_tokens"],
                )
                + self._cosine_loss(
                    forward_output["resolved_tokens"],
                    reverse_output["new_tokens"],
                )
                + self._cosine_loss(
                    forward_output["persistent_tokens"],
                    reverse_output["persistent_tokens"],
                )
                + self._cosine_loss(
                    forward_output["uncertain_tokens"],
                    reverse_output["uncertain_tokens"],
                )
            )

            forward_plan = self._normalise_plan(
                forward_output["plan_current_to_history"]
            )
            reverse_plan = self._normalise_plan(
                reverse_output["plan_current_to_history"]
            ).transpose(1, 2)
            swap_plan_loss = F.smooth_l1_loss(forward_plan, reverse_plan)

        swap_loss = swap_map_loss + swap_token_loss + swap_plan_loss
        forward_output.update(
            {
                "swap_loss": swap_loss,
                "swap_map_loss": swap_map_loss,
                "swap_token_loss": swap_token_loss,
                "swap_plan_loss": swap_plan_loss,
            }
        )
        return forward_output
