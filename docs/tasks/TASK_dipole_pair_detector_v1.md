# TASK: Dipole Pair Detector v1

## Scope

Create a minimal Python CLI detector at `scripts/dipole_pair_detector_v1.py` with two supported operating modes:

1. Manual seed calibration using approximate annotated dipole pairs.
2. Automatic target-vector estimation from positive-negative blob pair voting.

## Required CLI Examples

Manual seed mode:

```powershell
python scripts/dipole_pair_detector_v1.py ^
  --residual Result_DIVDE.tif ^
  --seed-pairs "1519,1290,1537,1290;1638,688,1664,680" ^
  --out outputs/dipole_pair_detector_v1_manual
```

Auto-vector mode:

```powershell
python scripts/dipole_pair_detector_v1.py ^
  --residual Result_DIVDE.tif ^
  --target-dx 30 ^
  --target-dy 0 ^
  --auto-vector ^
  --out outputs/dipole_pair_detector_v1_auto
```

## Required Processing

- Read TIFF with `tifffile.imread` as `float32`.
- Record shape, dtype, min, max, mean, and std.
- If all image values are positive, use `log(image / median(image))`; otherwise subtract the median.
- Apply high-pass filtering with Gaussian sigma 15, smooth with sigma 1.0, and robust MAD z-scoring.
- Detect positive blobs at `z > +3.5` and negative blobs at `z < -3.5`.
- Filter connected components by area, equivalent diameter, eccentricity, and aspect ratio.
- Estimate manual target vectors from refined seed extrema or auto target vectors from dominant vector voting.
- Match positive-negative pairs with distance 15-45 px and vector error <= 10 px.
- Export CSVs, overlays, top-50 crops, contact sheets, and review package artifacts.

## Required Git Hygiene

Do not commit raw data, generated runtime output directories, NumPy arrays, model files, or checkpoints.
