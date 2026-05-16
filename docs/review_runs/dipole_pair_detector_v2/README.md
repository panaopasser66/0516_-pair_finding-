# Dipole Pair Detector v2 Review Package

This directory contains v2 human-review artifacts exported by `scripts/dipole_pair_detector_v2.py`.

## Scope

v2 keeps the v1 detection pipeline and improves scoring, confidence classification, and review grouping. It does not claim true defect identity.

## Input File

- `Result_DIVDE.tif` (raw TIFF is not committed)
- Coordinate convention: `x,y`, where `x` is column and `y` is row

## V1 / V2 Comparison

| Metric | v1 | v2 |
| --- | ---: | ---: |
| Manual matched count | 32 | 32 |
| Auto matched count | 31 | 31 |
| Manual target dx,dy | (28.500,-0.500) | (28.500,-0.500) |
| Auto target dx,dy | (28.999,0.335) | (28.999,0.335) |

## V2 Confidence Counts

- Manual: `high=12, medium=9, low=11`
- Auto: `high=10, medium=10, low=11`

## Seed Pair Rank Check

- Seed pair 1: rank `1`, near top 10: `True`, center error `0.942`
- Seed pair 2: rank `3`, near top 10: `True`, center error `4.026`

## Auto Vector Confidence

- v1 confidence: `0.16145833333333334`
- v2 confidence: `0.16145833333333334`
- Improved: `False`
- Warning: Dominant unconstrained vector cluster confidence is low; used the provided target vector as a weak prior. Manual seed calibration is still recommended.

## Blob Counts

- Positive blobs: `196`
- Negative blobs: `202`

## V2 Scoring And Classification

V2 adds `pos_peak_z`, `neg_peak_abs_z`, `amplitude_balance`, `area_ratio`, `shape_penalty`, partner ambiguity counts, and `local_contrast_score` to each pair row.

`score_v2 = pos_peak_z + neg_peak_abs_z + 2.0*amplitude_balance - 0.5*vector_error - 0.5*size_mismatch_penalty - 0.3*partner_ambiguity_penalty - elongation_penalty`.

Confidence categories:

- `high_confidence`: vector error <= 5, amplitude balance >= 0.35, both peaks >= 3.5, and both aspect ratios < 3.
- `medium_confidence`: vector error <= 8 and amplitude balance >= 0.2.
- `low_confidence`: all other accepted pairs.

## Artifacts

- `pairs_manual_v2.csv` and `pairs_auto_v2.csv`
- `pair_overlay_manual_v2.jpg` and `pair_overlay_auto_v2.jpg`
- `contact_sheet_manual_v2.jpg` and `contact_sheet_auto_v2.jpg`
- `high_confidence_contact_sheet.jpg` and `medium_confidence_contact_sheet.jpg`
- `crops_manual_v2/`, `crops_auto_v2/`, `crops_high_confidence/`, and `crops_medium_confidence/`

## Interpretation Warning

This run is only a dipole candidate probe. Matched candidates are not proof of true defects and require expert review.
