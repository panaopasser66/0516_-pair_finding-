# TASK: Dipole Pair Detector v3

## Scope

Dipole Pair Detector v3 builds on v2 without rewriting the detector pipeline. It keeps the same CLI, preprocessing, blob detection, target-vector logic, and pair matching, then improves candidate quality scoring so top and high-confidence rows are more reliable.

## CLI Compatibility

The script at `scripts/dipole_pair_detector_v3.py` keeps these v2 flags:

- `--residual`
- `--seed-pairs`
- `--target-dx`
- `--target-dy`
- `--auto-vector`
- `--out`

## V3 Pair Metrics

Each pair row preserves `score_v2` and `confidence_v2`, then adds:

- `template_score`
- `isolation_score`
- `isolation_score_clipped`
- `confidence_v3`
- `score_v3`
- `rank_v3`
- compatibility aliases for `dx`, `dy`, partner counts, and aspect ratios

## V3 Score

`score_v3 = score_v2 + 4.0*template_score + 1.5*clip(isolation_score, 0, 3) - 0.5*partner_ambiguity_penalty`

## Confidence Categories

- `high_confidence`: vector error <= 6, amplitude balance >= 0.30, both peaks >= 3.5, template score >= 0.35, isolation score >= 1.5, and both aspect ratios < 3.5.
- `medium_confidence`: vector error <= 9, amplitude balance >= 0.20, template score >= 0.20, and isolation score >= 1.2.
- `low_confidence`: all other accepted pairs.

## Review Package

Review artifacts are exported to:

`docs/review_runs/dipole_pair_detector_v3/`

The README reports manual and auto target vectors, matched counts, v3 confidence counts, seed-pair ranks, v2-vs-v3 high-confidence changes, and whether manual seed remains the main mode.
