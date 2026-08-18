# SURE-OT

**SURE-OT: Swap-Consistent Unbalanced Residual Evolution Prompting for Longitudinal Radiology Report Generation**

This repository is a method-level extension of [BiOTPrompt](https://github.com/TengfeiLiu966/BiOTPrompt). It keeps the original Longitudinal-MIMIC data interface unchanged and replaces hard patch-index prompting with differentiable birth/resolution residual transport and continuous evolution tokens.

## What changed

SURE-OT adds four components:

1. **Unbalanced optimal transport (UOT).** Relaxed marginals let current or historical patches remain partially unmatched instead of forcing every patch into a correspondence.
2. **Birth/resolution residual maps.** Missing current-to-history mass indicates newly emerged evidence; missing history-to-current mass indicates resolved evidence.
3. **Continuous evolution tokens.** Learnable queries pool new, resolved, persistent, and uncertain visual evidence directly into LLM-space prompt tokens.
4. **Temporal swap consistency.** Reversing the current/history order provides annotation-free constraints: new in the forward direction should correspond to resolved in the reverse direction.

The original BiOTPrompt path remains available with `--use_sure_ot False`.

## Installation

```bash
git clone https://github.com/redwangwangwang/SURE-OT.git
cd SURE-OT
pip install -r requirements.txt
```

Prepare Longitudinal-MIMIC exactly as required by the upstream BiOTPrompt repository. No new bounding boxes, masks, progression classes, or modified dataset files are required.

## Training

Edit the dataset/model paths in `scripts/7-1.sure_ot_run.sh`, then run:

```bash
bash scripts/7-1.sure_ot_run.sh
```

## Testing

Set `delta_file` in `scripts/7-2.sure_ot_test.sh`, then run:

```bash
bash scripts/7-2.sure_ot_test.sh
```

## Main options

```text
--use_sure_ot True
--sure_ot_num_tokens 2
--sure_ot_epsilon 0.07
--sure_ot_tau 0.7
--sure_ot_iters 40
--sure_ot_spatial_weight 0.05
--sure_ot_use_swap True
--lambda_sure_ot_swap 0.1
--lambda_sure_ot_transport 0.01
--lambda_sure_ot_reg 0.01
```

Useful ablations:

```bash
# Original BiOTPrompt
--use_sure_ot False

# Balanced-OT control
--sure_ot_balanced True

# Remove temporal role adapters
--sure_ot_use_role_adapters False

# Remove swap consistency
--sure_ot_use_swap False
```

## Verification

The repository includes focused unit tests for numerical stability, unmatched-patch residual behavior, balanced-OT marginal degeneration, and gradient propagation:

```bash
pytest -q tests/test_sure_ot.py
```

These tests validate the implementation mechanics. They do not substitute for full training on Longitudinal-MIMIC, and this repository does not claim unrun benchmark results.

## Method details

See [`docs/METHOD.md`](docs/METHOD.md).

## Acknowledgements and license

The codebase is derived from BiOTPrompt and R2GenGPT. The upstream BSD 3-Clause license is retained in [`LICENSE`](LICENSE).
