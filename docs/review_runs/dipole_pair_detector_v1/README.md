# Dipole Pair Detector v1 Review Package

This directory contains human-review artifacts exported by `scripts/dipole_pair_detector_v1.py`.

## Input File

- `Result_DIVDE.tif` (raw TIFF is not committed)
- Coordinate convention: `x,y`, where `x` is column and `y` is row

## Seed Pairs

- `(1519,1290)->(1537,1290)`
- `(1638,688)->(1664,680)`

## Thresholds

- Positive blobs: `z > +3.5`
- Negative blobs: `z < -3.5`
- Blob area: `3-200 px`
- Equivalent diameter: `2-20 px`
- Eccentricity: `< 0.95`
- Aspect ratio: `< 4`
- Pair distance: `15-45 px`
- Pair vector error: `<= 10 px`

## Blob Counts

- Positive blobs: `196`
- Negative blobs: `202`

## Manual Seed Calibration

- Manual estimated target vector: `dx=28.500, dy=-0.500`
- Manual matched pair count: `32`
- Refined seed centers:
  - pair 1: (1521.0,1290.0) positive -> (1548.0,1290.0) negative; vector=(27.0,0.0)
  - pair 2: (1636.0,686.0) negative -> (1666.0,685.0) positive; vector=(30.0,-1.0)

## Auto Vector Discovery

- Auto estimated target vector: `dx=28.999, dy=0.335`
- Auto matched pair count: `31`
- Candidate vector votes: `192`
- Dominant cluster confidence: `0.16145833333333334`
- Warning: Dominant unconstrained vector cluster confidence is low; used the provided target vector as a weak prior. Manual seed calibration is still recommended.

## Pair Matching Criteria

For each positive blob and negative blob, the detector evaluates `vector = negative_center - positive_center`. A pair is accepted when:

- The vector is close to the estimated target vector or its negative.
- `min(norm(vector-target), norm(vector+target)) <= 10 px`.
- Pair distance is between `15` and `45` px.
- Score is `pos_max_z + abs(neg_min_z) - 0.5*vector_error - size_mismatch_penalty - elongation_penalty`.

## Artifacts

- `pairs_manual.csv` and `pairs_auto.csv`
- `positive_blobs.csv` and `negative_blobs.csv`
- `refined_seed_pairs.csv`
- `auto_vector_summary.json`
- `residual_z_preview.jpg`
- `positive_negative_overlay.jpg`
- `pair_overlay_manual.jpg` and `pair_overlay_auto.jpg`
- `vector_vote_heatmap.jpg`
- `contact_sheet_manual.jpg` and `contact_sheet_auto.jpg`
- `crops_manual/` and `crops_auto/` top candidate crop JPGs

## Interpretation Warning

This run is only a dipole candidate probe. Matched candidates are not proof of true defects and require expert review.
