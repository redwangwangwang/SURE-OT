import torch

from evalcap.SURE_OT import SUREOTModule


def test_output_shapes_and_finite_values():
    torch.manual_seed(7)
    module = SUREOTModule(
        dim=32,
        num_tokens=2,
        epsilon=0.1,
        tau=0.5,
        num_iters=30,
        spatial_weight=0.05,
        adapter_rank=16,
    )
    current = torch.randn(2, 49, 32)
    history = torch.randn(2, 49, 32)

    output = module(current, history, compute_swap=True)

    assert output["tokens"].shape == (2, 8, 32)
    assert output["new_score"].shape == (2, 49)
    assert output["resolved_score"].shape == (2, 49)
    assert output["new_mask"].dtype == torch.bool
    assert output["resolved_mask"].dtype == torch.bool

    for name in (
        "tokens",
        "new_score",
        "resolved_score",
        "persistent_score",
        "uncertain_score",
        "transport_loss",
        "regularization_loss",
        "swap_loss",
    ):
        assert torch.isfinite(output[name]).all(), name


def test_unmatched_current_patch_receives_birth_residual():
    torch.manual_seed(11)
    module = SUREOTModule(
        dim=16,
        num_tokens=2,
        epsilon=0.05,
        tau=0.1,
        num_iters=80,
        spatial_weight=0.0,
        adapter_rank=8,
        use_role_adapters=False,
    )
    history = torch.randn(1, 16, 16)
    current = history.clone()
    current[:, -1] = -4.0 * history[:, -1]

    output = module(current, history, compute_swap=False)
    changed_patch_score = output["new_score"][0, -1]
    unchanged_patch_median = output["new_score"][0, :-1].median()

    assert changed_patch_score > unchanged_patch_median + 0.1


def test_balanced_ablation_has_nearly_zero_marginal_residual():
    torch.manual_seed(13)
    module = SUREOTModule(
        dim=16,
        num_tokens=2,
        epsilon=0.1,
        tau=1.0,
        num_iters=100,
        spatial_weight=0.0,
        adapter_rank=8,
        balanced=True,
        use_role_adapters=False,
    )
    features = torch.randn(2, 16, 16)
    output = module(features, features, compute_swap=True)

    assert output["new_score"].max() < 5e-3
    assert output["resolved_score"].max() < 5e-3


def test_gradients_reach_evolution_queries_and_role_adapters():
    torch.manual_seed(17)
    module = SUREOTModule(
        dim=24,
        num_tokens=2,
        epsilon=0.08,
        tau=0.4,
        num_iters=30,
        spatial_weight=0.02,
        adapter_rank=12,
    )
    current = torch.randn(2, 25, 24, requires_grad=True)
    history = torch.randn(2, 25, 24, requires_grad=True)

    output = module(current, history, compute_swap=True)
    loss = (
        output["tokens"].square().mean()
        + 0.1 * output["swap_loss"]
        + 0.01 * output["transport_loss"]
        + 0.01 * output["regularization_loss"]
    )
    loss.backward()

    assert current.grad is not None
    assert history.grad is not None
    assert torch.isfinite(current.grad).all()
    assert torch.isfinite(history.grad).all()
    assert module.tokenizer.queries.grad is not None
    assert module.current_role.up.weight.grad is not None
    assert module.history_role.up.weight.grad is not None
