# TASK: Dipole Pair Detector v2

## Scope

Dipole Pair Detector v2 builds on v1 without rewriting the detector pipeline. It keeps the same CLI and preprocessing/blob/pair matching flow, then improves pair scoring, confidence classification, and review outputs.

## CLI Compatibility

The script at `scripts/dipole_pair_detector_v2.py` keeps these v1 flags:

- `--residual`
- `--seed-pairs`
- `--target-dx`
- `--target-dy`
- `--auto-vector`
- `--out`

## V2 Pair Metrics

Each pair row adds:

- `pos_peak_z`
- `neg_peak_abs_z`
- `amplitude_balance`
- `vector_error`
- `distance`
- `area_ratio`
- `shape_penalty`
- `pos_num_possible_partners`
- `neg_num_possible_partners`
- `num_possible_partners`
- `local_contrast_score`
- `confidence_category`
- `score_v2`

## V2 Score

`score_v2 = pos_peak_z + neg_peak_abs_z + 2.0*amplitude_balance - 0.5*vector_error - 0.5*size_mismatch_penalty - 0.3*partner_ambiguity_penalty - elongation_penalty`

## Confidence Categories

- `high_confidence`: vector error <= 5, amplitude balance >= 0.35, both peaks >= 3.5, and both aspect ratios < 3.
- `medium_confidence`: vector error <= 8 and amplitude balance >= 0.2.
- `low_confidence`: all other accepted pairs.

## Review Package

Review artifacts are exported to:

`docs/review_runs/dipole_pair_detector_v2/`

The README compares v1 and v2 matched counts, confidence counts, seed-pair ranks, and auto-vector confidence.
