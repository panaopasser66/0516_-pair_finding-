# Dipole Pair Detector v3 Review Package

This directory contains v3 human-review artifacts exported by `scripts/dipole_pair_detector_v3.py`.

## V3 Changes From V2

V3 keeps the v2 preprocessing, blob detection, vector matching, and CLI intact. It adds candidate-quality scoring terms instead of trying to find more pairs.

- `template_score`: normalized correlation against a local positive/negative Gaussian dipole template placed on the candidate vector.
- `isolation_score`: center peak/valley response divided by surrounding annulus response to down-rank textured regions.
- `partner_ambiguity_penalty`: retains the v2 possible-partner count idea and penalizes dense opposite-polarity neighborhoods.
- `score_v3 = score_v2 + 4.0*template_score + 1.5*clip(isolation_score,0,3) - 0.5*partner_ambiguity_penalty`.

## Run Summary

| Metric | Manual v3 | Auto v3 |
| --- | ---: | ---: |
| Target vector dx,dy | (28.500, -0.500) | (28.999, 0.335) |
| Matched count | 32 | 31 |
| V3 confidence counts | high=2, medium=11, low=19 | high=2, medium=11, low=18 |
| V2 high confidence in same rows | 12 | 10 |
| V3 high confidence | 2 | 2 |

## V2 vs V3 High-Confidence Change

- Previous v2 manual high-confidence count: `12`
- Current v3 manual high-confidence count: `2`
- Previous v2 auto high-confidence count: `10`
- Current v3 auto high-confidence count: `2`

## Seed Pair Rank Check

- Seed pair 1: rank `1`, near top 10: `True`, center error `0.942`
- Seed pair 2: rank `2`, near top 10: `True`, center error `4.026`

## Conclusion

- Manual seed is still the main mode: `yes`
- Positive blobs: `196`
- Negative blobs: `202`

## V3 Confidence Rules

- `high_confidence`: vector error <= 6, amplitude balance >= 0.30, both peaks >= 3.5, template score >= 0.35, isolation score >= 1.5, and both aspect ratios < 3.5.
- `medium_confidence`: vector error <= 9, amplitude balance >= 0.20, template score >= 0.20, and isolation score >= 1.2.
- `low_confidence`: all other accepted pairs.

## Artifacts

- `pairs_manual_v3.csv` and `pairs_auto_v3.csv`
- `pair_overlay_manual_v3.jpg` and `pair_overlay_auto_v3.jpg`
- `contact_sheet_manual_v3.jpg` and `contact_sheet_auto_v3.jpg`
- `high_confidence_contact_sheet_manual_v3.jpg` and `medium_confidence_contact_sheet_manual_v3.jpg`
- `high_confidence_contact_sheet_auto_v3.jpg` and `medium_confidence_contact_sheet_auto_v3.jpg`
- Top crop JPGs under `crops_manual_v3/`, `crops_auto_v3/`, `crops_high_confidence/`, and `crops_medium_confidence/`

## Interpretation Warning

This run is only a dipole candidate probe. Matched candidates are not proof of true defects and require expert review.
