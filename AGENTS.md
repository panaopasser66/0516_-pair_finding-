# Project Agent Notes

This repository is for Dipole Pair Detector experiments on divide/residual microscopy-style TIFF images.

## Data And Outputs
- Do not commit raw TIFF images, model/checkpoint files, NumPy arrays, `data/`, or `outputs/`.
- Review artifacts intended for sharing live under `docs/review_runs/`.
- Coordinates are always `x,y`, where `x` is image column and `y` is image row.

## Detector Context
- The detector searches for compact bright/dark dipole-like blob pairs in `Result_DIVDE.tif`.
- Expected horizontal separation is approximately 30 px, with a small `y` offset allowed.
- Polarity order is not fixed: bright-to-dark and dark-to-bright are both valid.

## Verification
- Run `python -m compileall src scripts` before handoff when possible.
- Run the detector in both manual seed and auto-vector modes for v1 review packages.
- Report blob counts, matched pair counts, vectors, top pair coordinates, review path, and git status.
