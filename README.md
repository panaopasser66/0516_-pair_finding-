# Dipole Pair Detector v1

This repository contains a minimal, reviewable detector for finding compact black/white dipole-like pairs in a divide/residual TIFF image.

## Input

- Expected residual/divide image: `Result_DIVDE.tif`
- Coordinate convention: `x,y`, where `x` is column and `y` is row
- Raw TIFF files and generated runtime outputs are intentionally ignored by Git.

## Goal

Detect candidate compact defect-like pairs containing one positive/bright blob and one negative/dark blob. The expected pair vector is roughly horizontal with about 30 px separation, but the polarity order may be either bright-to-dark or dark-to-bright.

## Usage

Manual seed calibration:

```powershell
python scripts/dipole_pair_detector_v1.py ^
  --residual Result_DIVDE.tif ^
  --seed-pairs "1519,1290,1537,1290;1638,688,1664,680" ^
  --out outputs/dipole_pair_detector_v1_manual
```

Automatic target-vector discovery:

```powershell
python scripts/dipole_pair_detector_v1.py ^
  --residual Result_DIVDE.tif ^
  --target-dx 30 ^
  --target-dy 0 ^
  --auto-vector ^
  --out outputs/dipole_pair_detector_v1_auto
```

## Outputs

Each run writes CSV files, overlays, top-pair crops, and a contact sheet under the selected `outputs/<run_name>/` directory. A shareable review package is also updated under:

```text
docs/review_runs/dipole_pair_detector_v1/
```

## Detector Summary

The v1 detector:

1. Reads the TIFF as `float32` and records image statistics.
2. Applies log-ratio or median-centered preprocessing, high-pass filtering, smoothing, and robust MAD z-scoring.
3. Finds compact positive and negative connected components with size and shape filters.
4. Estimates a target vector from manual seeds or automatic positive-negative vector voting.
5. Matches positive/negative blobs whose vector is close to the target or its opposite.
6. Exports review images and CSVs for human inspection.

This is a dipole candidate probe only. Its matches are not proof of true physical defects.
