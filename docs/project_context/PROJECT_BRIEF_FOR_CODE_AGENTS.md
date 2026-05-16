# Project Brief For Code Agents

## Objective

Build and maintain a minimal Dipole Pair Detector v1 for `Result_DIVDE.tif`. The detector should automatically find compact candidate pairs made of one positive/bright blob and one negative/dark blob in a divide/residual image.

## Important Domain Assumptions

- Coordinates use `x,y`: `x` is column, `y` is row.
- Expected pair separation is approximately 30 px in the x direction.
- A few pixels of y offset are acceptable.
- Polarity order is not fixed; both positive-to-negative and negative-to-positive pairs are valid.
- The detector output is for candidate review only and does not establish true defect identity.

## Known Manual Seed Pairs

- `(1519,1290)->(1537,1290)`
- `(1638,688)->(1664,680)`

These are approximate user annotations and should be refined to nearby local extrema before estimating the manual target vector.

## Review Package

Human-reviewable outputs should be collected in:

`docs/review_runs/dipole_pair_detector_v1/`

Include CSVs, vector summaries, overlays, contact sheets, and top candidate crops.
