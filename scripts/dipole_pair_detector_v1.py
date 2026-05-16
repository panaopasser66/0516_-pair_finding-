#!/usr/bin/env python3
"""Dipole Pair Detector v1.

Find compact positive/negative dipole-like blob pairs in a residual/divide TIFF.
Coordinates are reported as x,y where x is image column and y is image row.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from scipy import ndimage as ndi
from skimage import measure


POSITIVE_THRESHOLD = 3.5
NEGATIVE_THRESHOLD = -3.5
HIGHPASS_SIGMA = 15.0
SMOOTH_SIGMA = 1.0
SEED_REFINE_RADIUS = 40
PAIR_DISTANCE_MIN = 15.0
PAIR_DISTANCE_MAX = 45.0
PAIR_VECTOR_ERROR_MAX = 10.0
TOP_CROP_COUNT = 50
CROP_HALF_SIZE = 42
REVIEW_DIR = Path("docs/review_runs/dipole_pair_detector_v1")


@dataclass
class Blob:
    blob_id: int
    polarity: str
    x: float
    y: float
    area: float
    equivalent_diameter: float
    eccentricity: float
    aspect_ratio: float
    min_z: float
    max_z: float
    mean_z: float
    bbox_min_x: int
    bbox_min_y: int
    bbox_max_x: int
    bbox_max_y: int


@dataclass
class PairMatch:
    pair_id: int
    pos_blob_id: int
    neg_blob_id: int
    pos_x: float
    pos_y: float
    neg_x: float
    neg_y: float
    dx_pos_to_neg: float
    dy_pos_to_neg: float
    distance: float
    vector_error: float
    pos_max_z: float
    neg_min_z: float
    size_mismatch_penalty: float
    elongation_penalty: float
    score: float


@dataclass
class RefinedSeedPair:
    seed_pair_id: int
    seed1_x: float
    seed1_y: float
    seed2_x: float
    seed2_y: float
    refined1_x: float
    refined1_y: float
    refined2_x: float
    refined2_y: float
    polarity1: str
    polarity2: str
    vector_dx: float
    vector_dy: float
    distance: float
    peak1_z: float
    peak2_z: float


@dataclass
class AutoVectorSummary:
    initial_target_dx: float
    initial_target_dy: float
    estimated_target_dx: float
    estimated_target_dy: float
    candidate_vector_count: int
    dominant_bin_count: int
    neighboring_bin_count: int
    confidence: float
    confidence_warning: str
    bin_size_px: float
    distance_min_px: float
    distance_max_px: float


class DetectorError(RuntimeError):
    pass


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def to_builtin(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def write_json(path: Path, data: Dict) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False, default=to_builtin)
        file.write("\n")


def write_csv(path: Path, rows: Sequence, fieldnames: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            if hasattr(row, "__dataclass_fields__"):
                row_dict = asdict(row)
            else:
                row_dict = dict(row)
            writer.writerow({key: row_dict.get(key, "") for key in fieldnames})


def read_residual(path: Path) -> Tuple[np.ndarray, Dict]:
    raw = tifffile.imread(str(path))
    original_shape = tuple(int(v) for v in np.shape(raw))
    original_dtype = str(getattr(raw, "dtype", "unknown"))
    image = np.asarray(raw)
    image = np.squeeze(image)
    if image.ndim != 2:
        if image.ndim == 3 and image.shape[-1] <= 4:
            image = np.mean(image, axis=-1)
        elif image.ndim == 3:
            image = image[0]
        else:
            raise DetectorError("Residual image must be 2D or reducible to 2D.")
    image = image.astype(np.float32, copy=False)
    finite_values = image[np.isfinite(image)]
    if finite_values.size == 0:
        raise DetectorError("Residual image contains no finite values.")
    fill_value = float(np.nanmedian(finite_values))
    image = np.nan_to_num(
        image,
        nan=fill_value,
        posinf=float(np.nanmax(finite_values)),
        neginf=float(np.nanmin(finite_values)),
    )
    stats = {
        "input_path": str(path),
        "original_shape": original_shape,
        "loaded_shape": tuple(int(v) for v in image.shape),
        "original_dtype": original_dtype,
        "working_dtype": str(image.dtype),
        "min": float(np.min(image)),
        "max": float(np.max(image)),
        "mean": float(np.mean(image)),
        "std": float(np.std(image)),
    }
    return image, stats


def robust_mad_scale(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 1.0
    med = float(np.median(finite))
    mad = float(np.median(np.abs(finite - med)))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale < 1e-6:
        std = float(np.std(finite))
        scale = std if np.isfinite(std) and std > 1e-6 else 1.0
    return scale


def preprocess(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    finite = image[np.isfinite(image)]
    median_value = float(np.median(finite))
    min_value = float(np.min(finite))
    safe_image = np.nan_to_num(
        image,
        nan=median_value,
        posinf=float(np.max(finite)),
        neginf=float(np.min(finite)),
    )
    if min_value > 0 and median_value > 0:
        mode = "log_ratio_to_median"
        work = np.log(np.maximum(safe_image, np.finfo(np.float32).tiny) / median_value)
    else:
        mode = "median_centered"
        work = safe_image - median_value
    work = np.nan_to_num(work, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    background = ndi.gaussian_filter(work, sigma=HIGHPASS_SIGMA)
    highpass = work - background
    smoothed = ndi.gaussian_filter(highpass, sigma=SMOOTH_SIGMA)
    scale = robust_mad_scale(smoothed)
    z = smoothed / scale
    z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    stats = {
        "preprocess_mode": mode,
        "image_median": median_value,
        "highpass_sigma": HIGHPASS_SIGMA,
        "smooth_sigma": SMOOTH_SIGMA,
        "robust_mad_scale": float(scale),
        "z_min": float(np.min(z)),
        "z_max": float(np.max(z)),
        "z_mean": float(np.mean(z)),
        "z_std": float(np.std(z)),
    }
    return work, highpass.astype(np.float32, copy=False), z, stats


def component_aspect_ratio(region) -> float:
    min_row, min_col, max_row, max_col = region.bbox
    height = max(1, int(max_row - min_row))
    width = max(1, int(max_col - min_col))
    return float(max(width / height, height / width))


def detect_blobs(z: np.ndarray, polarity: str) -> List[Blob]:
    if polarity == "positive":
        mask = z > POSITIVE_THRESHOLD
    elif polarity == "negative":
        mask = z < NEGATIVE_THRESHOLD
    else:
        raise ValueError("polarity must be positive or negative")

    labels = measure.label(mask, connectivity=2)
    blobs: List[Blob] = []
    for region in measure.regionprops(labels, intensity_image=z):
        area = float(region.area)
        equivalent_diameter = float(region.equivalent_diameter)
        eccentricity = float(region.eccentricity)
        aspect_ratio = component_aspect_ratio(region)
        if not (3.0 <= area <= 200.0):
            continue
        if not (2.0 <= equivalent_diameter <= 20.0):
            continue
        if eccentricity >= 0.95:
            continue
        if aspect_ratio >= 4.0:
            continue
        coords = region.coords
        values = z[coords[:, 0], coords[:, 1]]
        min_row, min_col, max_row, max_col = region.bbox
        centroid_y, centroid_x = region.weighted_centroid
        if not np.isfinite(centroid_y) or not np.isfinite(centroid_x):
            centroid_y, centroid_x = region.centroid
        blobs.append(
            Blob(
                blob_id=len(blobs) + 1,
                polarity=polarity,
                x=float(centroid_x),
                y=float(centroid_y),
                area=area,
                equivalent_diameter=equivalent_diameter,
                eccentricity=eccentricity,
                aspect_ratio=aspect_ratio,
                min_z=float(np.min(values)),
                max_z=float(np.max(values)),
                mean_z=float(np.mean(values)),
                bbox_min_x=int(min_col),
                bbox_min_y=int(min_row),
                bbox_max_x=int(max_col),
                bbox_max_y=int(max_row),
            )
        )
    return blobs


def parse_seed_pairs(text: str) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
    pairs = []
    if not text:
        return pairs
    for item in text.split(";"):
        item = item.strip()
        if not item:
            continue
        parts = [float(part.strip()) for part in item.split(",")]
        if len(parts) != 4:
            raise DetectorError(f"Invalid seed pair '{item}', expected x1,y1,x2,y2.")
        pairs.append(((parts[0], parts[1]), (parts[2], parts[3])))
    return pairs


def seed_extrema(z: np.ndarray, x: float, y: float, radius: int = SEED_REFINE_RADIUS) -> Dict[str, Dict]:
    height, width = z.shape
    cx = int(round(x))
    cy = int(round(y))
    x0 = max(0, cx - radius)
    x1 = min(width, cx + radius + 1)
    y0 = max(0, cy - radius)
    y1 = min(height, cy + radius + 1)
    patch = z[y0:y1, x0:x1]
    if patch.size == 0:
        raise DetectorError(f"Seed point ({x},{y}) is outside image bounds.")
    max_index = np.unravel_index(int(np.argmax(patch)), patch.shape)
    min_index = np.unravel_index(int(np.argmin(patch)), patch.shape)
    max_y, max_x = max_index
    min_y, min_x = min_index
    return {
        "positive": {
            "x": float(x0 + max_x),
            "y": float(y0 + max_y),
            "polarity": "positive",
            "peak_z": float(patch[max_index]),
        },
        "negative": {
            "x": float(x0 + min_x),
            "y": float(y0 + min_y),
            "polarity": "negative",
            "peak_z": float(patch[min_index]),
        },
    }


def local_extremum(z: np.ndarray, x: float, y: float, radius: int = SEED_REFINE_RADIUS) -> Dict:
    extrema = seed_extrema(z, x, y, radius)
    positive = extrema["positive"]
    negative = extrema["negative"]
    if abs(positive["peak_z"]) >= abs(negative["peak_z"]):
        return positive
    return negative


def refine_seed_pair_joint(z: np.ndarray, seed1: Tuple[float, float], seed2: Tuple[float, float]) -> Tuple[Dict, Dict]:
    extrema1 = seed_extrema(z, seed1[0], seed1[1])
    extrema2 = seed_extrema(z, seed2[0], seed2[1])
    options = [
        (extrema1["positive"], extrema2["negative"]),
        (extrema1["negative"], extrema2["positive"]),
    ]
    expected_dx = seed2[0] - seed1[0]
    expected_dy = seed2[1] - seed1[1]

    def option_cost(option: Tuple[Dict, Dict]) -> float:
        first, second = option
        vector_cost = math.hypot((second["x"] - first["x"]) - expected_dx, (second["y"] - first["y"]) - expected_dy)
        first_motion = math.hypot(first["x"] - seed1[0], first["y"] - seed1[1])
        second_motion = math.hypot(second["x"] - seed2[0], second["y"] - seed2[1])
        strength_reward = 0.05 * (abs(first["peak_z"]) + abs(second["peak_z"]))
        return vector_cost + 0.25 * (first_motion + second_motion) - strength_reward

    return min(options, key=option_cost)


def canonical_vector(dx: float, dy: float) -> Tuple[float, float]:
    if dx < 0:
        return -dx, -dy
    return dx, dy


def refine_seed_pairs(z: np.ndarray, seed_pairs: Sequence[Tuple[Tuple[float, float], Tuple[float, float]]]) -> Tuple[List[RefinedSeedPair], Tuple[float, float]]:
    refined_pairs: List[RefinedSeedPair] = []
    vectors: List[Tuple[float, float]] = []
    for index, (seed1, seed2) in enumerate(seed_pairs, start=1):
        refined1, refined2 = refine_seed_pair_joint(z, seed1, seed2)
        if refined1["polarity"] == refined2["polarity"]:
            raise DetectorError(
                "Seed pair {0} refined to same polarity ({1}). Use better seeds or inspect z preview.".format(index, refined1["polarity"])
            )
        raw_dx = float(refined2["x"] - refined1["x"])
        raw_dy = float(refined2["y"] - refined1["y"])
        dx, dy = canonical_vector(raw_dx, raw_dy)
        distance = float(math.hypot(dx, dy))
        vectors.append((dx, dy))
        refined_pairs.append(
            RefinedSeedPair(
                seed_pair_id=index,
                seed1_x=float(seed1[0]),
                seed1_y=float(seed1[1]),
                seed2_x=float(seed2[0]),
                seed2_y=float(seed2[1]),
                refined1_x=float(refined1["x"]),
                refined1_y=float(refined1["y"]),
                refined2_x=float(refined2["x"]),
                refined2_y=float(refined2["y"]),
                polarity1=str(refined1["polarity"]),
                polarity2=str(refined2["polarity"]),
                vector_dx=dx,
                vector_dy=dy,
                distance=distance,
                peak1_z=float(refined1["peak_z"]),
                peak2_z=float(refined2["peak_z"]),
            )
        )
    if not vectors:
        raise DetectorError("Manual seed mode requires at least one seed pair.")
    target_dx = float(np.median([vector[0] for vector in vectors]))
    target_dy = float(np.median([vector[1] for vector in vectors]))
    target_dx, target_dy = canonical_vector(target_dx, target_dy)
    return refined_pairs, (target_dx, target_dy)

def estimate_auto_vector(
    positive_blobs: Sequence[Blob],
    negative_blobs: Sequence[Blob],
    initial_target: Tuple[float, float],
    out_dir: Path,
) -> Tuple[Tuple[float, float], AutoVectorSummary, np.ndarray]:
    vectors: List[Tuple[float, float]] = []
    for pos in positive_blobs:
        for neg in negative_blobs:
            raw_dx = neg.x - pos.x
            raw_dy = neg.y - pos.y
            distance = math.hypot(raw_dx, raw_dy)
            if PAIR_DISTANCE_MIN <= distance <= PAIR_DISTANCE_MAX:
                vectors.append(canonical_vector(raw_dx, raw_dy))
    vector_array = np.asarray(vectors, dtype=np.float32) if vectors else np.empty((0, 2), dtype=np.float32)
    bin_size = 2.0
    if vector_array.size == 0:
        summary = AutoVectorSummary(
            initial_target_dx=float(initial_target[0]),
            initial_target_dy=float(initial_target[1]),
            estimated_target_dx=float(initial_target[0]),
            estimated_target_dy=float(initial_target[1]),
            candidate_vector_count=0,
            dominant_bin_count=0,
            neighboring_bin_count=0,
            confidence=0.0,
            confidence_warning="No candidate positive-negative vectors in the accepted distance range; falling back to initial target vector and manual seeds are recommended.",
            bin_size_px=bin_size,
            distance_min_px=PAIR_DISTANCE_MIN,
            distance_max_px=PAIR_DISTANCE_MAX,
        )
        plot_vector_heatmap(vector_array, initial_target, initial_target, out_dir / "vector_vote_heatmap.png")
        return initial_target, summary, vector_array

    dx_edges = np.arange(0.0, PAIR_DISTANCE_MAX + bin_size * 2, bin_size)
    dy_limit = PAIR_DISTANCE_MAX
    dy_edges = np.arange(-dy_limit, dy_limit + bin_size * 2, bin_size)
    hist, dx_edges, dy_edges = np.histogram2d(vector_array[:, 0], vector_array[:, 1], bins=[dx_edges, dy_edges])
    dominant_index = np.unravel_index(int(np.argmax(hist)), hist.shape)
    dominant_count = int(hist[dominant_index])
    dx_center = float((dx_edges[dominant_index[0]] + dx_edges[dominant_index[0] + 1]) / 2.0)
    dy_center = float((dy_edges[dominant_index[1]] + dy_edges[dominant_index[1] + 1]) / 2.0)
    distances_to_center = np.hypot(vector_array[:, 0] - dx_center, vector_array[:, 1] - dy_center)
    cluster_mask = distances_to_center <= 4.0
    neighboring_count = int(np.sum(cluster_mask))
    if neighboring_count > 0:
        estimated_dx = float(np.median(vector_array[cluster_mask, 0]))
        estimated_dy = float(np.median(vector_array[cluster_mask, 1]))
    else:
        estimated_dx = dx_center
        estimated_dy = dy_center
    estimated_dx, estimated_dy = canonical_vector(estimated_dx, estimated_dy)
    confidence = float(neighboring_count / max(1, len(vector_array)))
    warning = ""
    if neighboring_count < 5 or confidence < 0.05:
        prior_distances = np.hypot(vector_array[:, 0] - initial_target[0], vector_array[:, 1] - initial_target[1])
        prior_mask = prior_distances <= PAIR_VECTOR_ERROR_MAX
        prior_count = int(np.sum(prior_mask))
        if prior_count >= 5:
            estimated_dx = float(np.median(vector_array[prior_mask, 0]))
            estimated_dy = float(np.median(vector_array[prior_mask, 1]))
            estimated_dx, estimated_dy = canonical_vector(estimated_dx, estimated_dy)
            neighboring_count = prior_count
            confidence = float(prior_count / max(1, len(vector_array)))
            warning = "Dominant unconstrained vector cluster confidence is low; used the provided target vector as a weak prior. Manual seed calibration is still recommended."
        else:
            warning = "Dominant vector cluster confidence is low; manual seed calibration is recommended."
    summary = AutoVectorSummary(
        initial_target_dx=float(initial_target[0]),
        initial_target_dy=float(initial_target[1]),
        estimated_target_dx=estimated_dx,
        estimated_target_dy=estimated_dy,
        candidate_vector_count=int(len(vector_array)),
        dominant_bin_count=dominant_count,
        neighboring_bin_count=neighboring_count,
        confidence=confidence,
        confidence_warning=warning,
        bin_size_px=bin_size,
        distance_min_px=PAIR_DISTANCE_MIN,
        distance_max_px=PAIR_DISTANCE_MAX,
    )
    plot_vector_heatmap(vector_array, initial_target, (estimated_dx, estimated_dy), out_dir / "vector_vote_heatmap.png")
    return (estimated_dx, estimated_dy), summary, vector_array


def match_pairs(positive_blobs: Sequence[Blob], negative_blobs: Sequence[Blob], target_vector: Tuple[float, float]) -> List[PairMatch]:
    target = np.asarray(target_vector, dtype=np.float32)
    matches: List[PairMatch] = []
    for pos in positive_blobs:
        for neg in negative_blobs:
            vector = np.asarray([neg.x - pos.x, neg.y - pos.y], dtype=np.float32)
            distance = float(np.linalg.norm(vector))
            if not (PAIR_DISTANCE_MIN <= distance <= PAIR_DISTANCE_MAX):
                continue
            vector_error = float(min(np.linalg.norm(vector - target), np.linalg.norm(vector + target)))
            if vector_error > PAIR_VECTOR_ERROR_MAX:
                continue
            size_mismatch_penalty = abs(pos.area - neg.area) / max(pos.area, neg.area, 1.0)
            elongation_penalty = max(0.0, pos.aspect_ratio - 1.0) + max(0.0, neg.aspect_ratio - 1.0)
            score = pos.max_z + abs(neg.min_z) - 0.5 * vector_error - size_mismatch_penalty - elongation_penalty
            matches.append(
                PairMatch(
                    pair_id=0,
                    pos_blob_id=pos.blob_id,
                    neg_blob_id=neg.blob_id,
                    pos_x=float(pos.x),
                    pos_y=float(pos.y),
                    neg_x=float(neg.x),
                    neg_y=float(neg.y),
                    dx_pos_to_neg=float(vector[0]),
                    dy_pos_to_neg=float(vector[1]),
                    distance=distance,
                    vector_error=vector_error,
                    pos_max_z=float(pos.max_z),
                    neg_min_z=float(neg.min_z),
                    size_mismatch_penalty=float(size_mismatch_penalty),
                    elongation_penalty=float(elongation_penalty),
                    score=float(score),
                )
            )
    matches.sort(key=lambda item: item.score, reverse=True)
    for index, match in enumerate(matches, start=1):
        match.pair_id = index
    return matches


def percentile_limits(image: np.ndarray, lower: float = 1.0, upper: float = 99.0) -> Tuple[float, float]:
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        return 0.0, 1.0
    vmin, vmax = np.percentile(finite, [lower, upper])
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        return float(np.min(finite)), float(np.max(finite) + 1e-6)
    return float(vmin), float(vmax)


def save_z_preview(z: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 10), dpi=150)
    im = ax.imshow(z, cmap="coolwarm", vmin=-6, vmax=6)
    ax.set_title("Robust z-score preview")
    ax.set_axis_off()
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_blob_overlay(z: np.ndarray, positive_blobs: Sequence[Blob], negative_blobs: Sequence[Blob], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 11), dpi=150)
    ax.imshow(z, cmap="gray", vmin=-5, vmax=5)
    if positive_blobs:
        ax.scatter([blob.x for blob in positive_blobs], [blob.y for blob in positive_blobs], s=12, facecolors="none", edgecolors="red", linewidths=0.7, label="positive")
    if negative_blobs:
        ax.scatter([blob.x for blob in negative_blobs], [blob.y for blob in negative_blobs], s=12, facecolors="none", edgecolors="cyan", linewidths=0.7, label="negative")
    ax.set_title("Positive / negative compact blobs")
    ax.set_axis_off()
    if positive_blobs or negative_blobs:
        ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_pair_overlay(z: np.ndarray, matches: Sequence[PairMatch], target_vector: Tuple[float, float], path: Path, title_suffix: str) -> None:
    fig, ax = plt.subplots(figsize=(11, 11), dpi=150)
    ax.imshow(z, cmap="gray", vmin=-5, vmax=5)
    top_matches = matches[:200]
    if top_matches:
        ax.scatter([pair.pos_x for pair in top_matches], [pair.pos_y for pair in top_matches], s=18, c="red", marker="+", linewidths=0.8, label="positive")
        ax.scatter([pair.neg_x for pair in top_matches], [pair.neg_y for pair in top_matches], s=18, c="cyan", marker="x", linewidths=0.8, label="negative")
        for pair in top_matches:
            ax.plot([pair.pos_x, pair.neg_x], [pair.pos_y, pair.neg_y], color="yellow", linewidth=0.35, alpha=0.65)
        ax.legend(loc="upper right")
    ax.set_title("Pair overlay {0} | target=({1:.2f},{2:.2f}) | top {3}".format(title_suffix, target_vector[0], target_vector[1], len(top_matches)))
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_vector_heatmap(vector_array: np.ndarray, initial_target: Tuple[float, float], estimated_target: Tuple[float, float], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    if vector_array.size:
        h = ax.hist2d(vector_array[:, 0], vector_array[:, 1], bins=[np.arange(0, 48, 2), np.arange(-46, 48, 2)], cmap="magma")
        fig.colorbar(h[3], ax=ax, label="votes")
    else:
        ax.text(0.5, 0.5, "No candidate vectors", transform=ax.transAxes, ha="center", va="center")
    ax.scatter([initial_target[0]], [initial_target[1]], c="white", edgecolors="black", marker="o", s=50, label="initial")
    ax.scatter([estimated_target[0]], [estimated_target[1]], c="lime", edgecolors="black", marker="*", s=90, label="estimated")
    ax.set_xlabel("canonical dx (px)")
    ax.set_ylabel("canonical dy (px)")
    ax.set_title("Auto-vector vote heatmap")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)

def crop_bounds(shape: Tuple[int, int], center_x: float, center_y: float, half_size: int = CROP_HALF_SIZE) -> Tuple[int, int, int, int]:
    height, width = shape
    cx = int(round(center_x))
    cy = int(round(center_y))
    x0 = max(0, cx - half_size)
    x1 = min(width, cx + half_size + 1)
    y0 = max(0, cy - half_size)
    y1 = min(height, cy + half_size + 1)
    return x0, x1, y0, y1


def save_pair_crop(image: np.ndarray, z: np.ndarray, pair: PairMatch, path: Path) -> None:
    center_x = (pair.pos_x + pair.neg_x) / 2.0
    center_y = (pair.pos_y + pair.neg_y) / 2.0
    x0, x1, y0, y1 = crop_bounds(z.shape, center_x, center_y)
    residual_crop = image[y0:y1, x0:x1]
    z_crop = z[y0:y1, x0:x1]
    pos_x = pair.pos_x - x0
    pos_y = pair.pos_y - y0
    neg_x = pair.neg_x - x0
    neg_y = pair.neg_y - y0
    rvmin, rvmax = percentile_limits(residual_crop, 1, 99)
    fig, axes = plt.subplots(1, 3, figsize=(8, 3), dpi=120)
    axes[0].imshow(residual_crop, cmap="gray", vmin=rvmin, vmax=rvmax)
    axes[0].set_title("residual")
    axes[1].imshow(z_crop, cmap="coolwarm", vmin=-6, vmax=6)
    axes[1].set_title("z map")
    axes[2].imshow(z_crop, cmap="gray", vmin=-5, vmax=5)
    axes[2].scatter([pos_x], [pos_y], c="red", marker="+", s=70, linewidths=1.2)
    axes[2].scatter([neg_x], [neg_y], c="cyan", marker="x", s=70, linewidths=1.2)
    axes[2].plot([pos_x, neg_x], [pos_y, neg_y], color="yellow", linewidth=1.0)
    axes[2].set_title("overlay")
    for ax in axes:
        ax.set_axis_off()
    fig.suptitle("pair {0:04d} score={1:.2f}".format(pair.pair_id, pair.score), fontsize=9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def create_crops_and_contact_sheet(image: np.ndarray, z: np.ndarray, matches: Sequence[PairMatch], out_dir: Path) -> List[Path]:
    crops_dir = out_dir / "crops"
    ensure_dir(crops_dir)
    crop_paths: List[Path] = []
    for pair in matches[:TOP_CROP_COUNT]:
        crop_path = crops_dir / "pair_{0:04d}.png".format(pair.pair_id)
        save_pair_crop(image, z, pair, crop_path)
        crop_paths.append(crop_path)
    contact_path = out_dir / "reports" / "contact_sheet.png"
    ensure_dir(contact_path.parent)
    if not crop_paths:
        fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
        ax.text(0.5, 0.5, "No matched pairs", transform=ax.transAxes, ha="center", va="center")
        ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(contact_path)
        plt.close(fig)
        return crop_paths

    thumb_images = []
    for crop_path in crop_paths:
        image_array = plt.imread(str(crop_path))
        thumb_images.append(image_array)
    columns = 5
    rows = int(math.ceil(len(thumb_images) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(columns * 3.0, rows * 1.4), dpi=120)
    axes_array = np.atleast_1d(axes).ravel()
    for ax, thumb, pair in zip(axes_array, thumb_images, matches[:TOP_CROP_COUNT]):
        ax.imshow(thumb)
        ax.set_title("#{0} s={1:.1f}".format(pair.pair_id, pair.score), fontsize=8)
        ax.set_axis_off()
    for ax in axes_array[len(thumb_images) :]:
        ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(contact_path)
    plt.close(fig)
    return crop_paths


def copy_image_as_jpg(source: Path, destination: Path) -> None:
    ensure_dir(destination.parent)
    try:
        from PIL import Image

        with Image.open(source) as image_file:
            image_file.convert("RGB").save(destination, quality=92)
    except Exception:
        shutil.copyfile(source, destination.with_suffix(source.suffix))


def copy_csv(source: Path, destination: Path) -> None:
    ensure_dir(destination.parent)
    if source.exists():
        shutil.copyfile(source, destination)


def write_run_summary(
    out_dir: Path,
    mode: str,
    image_stats: Dict,
    preprocess_stats: Dict,
    target_vector: Tuple[float, float],
    positive_count: int,
    negative_count: int,
    pair_count: int,
    refined_pairs: Sequence[RefinedSeedPair],
    auto_summary: Optional[AutoVectorSummary],
) -> None:
    summary = {
        "mode": mode,
        "image_stats": image_stats,
        "preprocess_stats": preprocess_stats,
        "thresholds": {
            "positive_z": POSITIVE_THRESHOLD,
            "negative_z": NEGATIVE_THRESHOLD,
            "blob_area_range_px": [3, 200],
            "equivalent_diameter_range_px": [2, 20],
            "eccentricity_max": 0.95,
            "aspect_ratio_max": 4,
            "pair_distance_range_px": [PAIR_DISTANCE_MIN, PAIR_DISTANCE_MAX],
            "pair_vector_error_max_px": PAIR_VECTOR_ERROR_MAX,
        },
        "target_vector": {"dx": float(target_vector[0]), "dy": float(target_vector[1])},
        "positive_blob_count": int(positive_count),
        "negative_blob_count": int(negative_count),
        "matched_pair_count": int(pair_count),
        "refined_seed_pairs": [asdict(item) for item in refined_pairs],
        "auto_vector_summary": asdict(auto_summary) if auto_summary else None,
    }
    write_json(out_dir / "run_summary.json", summary)


def build_review_package(
    mode: str,
    out_dir: Path,
    image_stats: Dict,
    preprocess_stats: Dict,
    target_vector: Tuple[float, float],
    positive_count: int,
    negative_count: int,
    pair_count: int,
    refined_pairs: Sequence[RefinedSeedPair],
    auto_summary: Optional[AutoVectorSummary],
    seed_pairs_text: str,
) -> None:
    review_dir = REVIEW_DIR
    ensure_dir(review_dir)
    copy_csv(out_dir / "positive_blobs.csv", review_dir / "positive_blobs.csv")
    copy_csv(out_dir / "negative_blobs.csv", review_dir / "negative_blobs.csv")
    if mode == "manual":
        copy_csv(out_dir / "pairs.csv", review_dir / "pairs_manual.csv")
        copy_csv(out_dir / "refined_seed_pairs.csv", review_dir / "refined_seed_pairs.csv")
        copy_image_as_jpg(out_dir / "pair_overlay.png", review_dir / "pair_overlay_manual.jpg")
        copy_image_as_jpg(out_dir / "reports" / "contact_sheet.png", review_dir / "contact_sheet_manual.jpg")
        crop_destination = review_dir / "crops_manual"
    else:
        copy_csv(out_dir / "pairs.csv", review_dir / "pairs_auto.csv")
        if (out_dir / "auto_vector_summary.json").exists():
            shutil.copyfile(out_dir / "auto_vector_summary.json", review_dir / "auto_vector_summary.json")
        copy_image_as_jpg(out_dir / "pair_overlay.png", review_dir / "pair_overlay_auto.jpg")
        copy_image_as_jpg(out_dir / "reports" / "contact_sheet.png", review_dir / "contact_sheet_auto.jpg")
        if (out_dir / "vector_vote_heatmap.png").exists():
            copy_image_as_jpg(out_dir / "vector_vote_heatmap.png", review_dir / "vector_vote_heatmap.jpg")
        crop_destination = review_dir / "crops_auto"

    copy_image_as_jpg(out_dir / "residual_z_preview.png", review_dir / "residual_z_preview.jpg")
    copy_image_as_jpg(out_dir / "positive_negative_overlay.png", review_dir / "positive_negative_overlay.jpg")
    ensure_dir(crop_destination)
    for old_crop in crop_destination.glob("*.jpg"):
        old_crop.unlink()
    for crop in sorted((out_dir / "crops").glob("pair_*.png"))[:TOP_CROP_COUNT]:
        copy_image_as_jpg(crop, crop_destination / crop.with_suffix(".jpg").name)

    write_json(
        review_dir / ("summary_{0}.json".format(mode)),
        {
            "mode": mode,
            "input_file": image_stats.get("input_path"),
            "seed_pairs": seed_pairs_text if seed_pairs_text else None,
            "target_vector": {"dx": float(target_vector[0]), "dy": float(target_vector[1])},
            "positive_blob_count": int(positive_count),
            "negative_blob_count": int(negative_count),
            "matched_pair_count": int(pair_count),
            "image_stats": image_stats,
            "preprocess_stats": preprocess_stats,
            "refined_seed_pairs": [asdict(item) for item in refined_pairs],
            "auto_vector_summary": asdict(auto_summary) if auto_summary else None,
        },
    )
    update_review_readme(review_dir)

def load_json_if_exists(path: Path) -> Optional[Dict]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def csv_count(path: Path) -> Optional[int]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as file:
        return max(0, sum(1 for _ in file) - 1)


def update_review_readme(review_dir: Path) -> None:
    manual = load_json_if_exists(review_dir / "summary_manual.json")
    auto = load_json_if_exists(review_dir / "summary_auto.json")
    positive_count = csv_count(review_dir / "positive_blobs.csv")
    negative_count = csv_count(review_dir / "negative_blobs.csv")

    lines = [
        "# Dipole Pair Detector v1 Review Package",
        "",
        "This directory contains human-review artifacts exported by `scripts/dipole_pair_detector_v1.py`.",
        "",
        "## Input File",
        "",
        "- `Result_DIVDE.tif` (raw TIFF is not committed)",
        "- Coordinate convention: `x,y`, where `x` is column and `y` is row",
        "",
        "## Seed Pairs",
        "",
        "- `(1519,1290)->(1537,1290)`",
        "- `(1638,688)->(1664,680)`",
        "",
        "## Thresholds",
        "",
        f"- Positive blobs: `z > +{POSITIVE_THRESHOLD}`",
        f"- Negative blobs: `z < {NEGATIVE_THRESHOLD}`",
        "- Blob area: `3-200 px`",
        "- Equivalent diameter: `2-20 px`",
        "- Eccentricity: `< 0.95`",
        "- Aspect ratio: `< 4`",
        f"- Pair distance: `{PAIR_DISTANCE_MIN:g}-{PAIR_DISTANCE_MAX:g} px`",
        f"- Pair vector error: `<= {PAIR_VECTOR_ERROR_MAX:g} px`",
        "",
        "## Blob Counts",
        "",
        f"- Positive blobs: `{positive_count if positive_count is not None else 'not available'}`",
        f"- Negative blobs: `{negative_count if negative_count is not None else 'not available'}`",
        "",
    ]

    if manual:
        lines.extend(["## Manual Seed Calibration", ""])
        lines.append(
            "- Manual estimated target vector: `dx={0:.3f}, dy={1:.3f}`".format(
                manual["target_vector"]["dx"], manual["target_vector"]["dy"]
            )
        )
        lines.append("- Manual matched pair count: `{0}`".format(manual.get("matched_pair_count", "not available")))
        lines.append("- Refined seed centers:")
        for item in manual.get("refined_seed_pairs", []):
            lines.append(
                "  - pair {0}: ({1:.1f},{2:.1f}) {3} -> ({4:.1f},{5:.1f}) {6}; vector=({7:.1f},{8:.1f})".format(
                    item["seed_pair_id"],
                    item["refined1_x"],
                    item["refined1_y"],
                    item["polarity1"],
                    item["refined2_x"],
                    item["refined2_y"],
                    item["polarity2"],
                    item["vector_dx"],
                    item["vector_dy"],
                )
            )
        lines.append("")

    if auto:
        summary = auto.get("auto_vector_summary") or {}
        lines.extend(["## Auto Vector Discovery", ""])
        lines.append(
            "- Auto estimated target vector: `dx={0:.3f}, dy={1:.3f}`".format(
                auto["target_vector"]["dx"], auto["target_vector"]["dy"]
            )
        )
        lines.append("- Auto matched pair count: `{0}`".format(auto.get("matched_pair_count", "not available")))
        lines.append("- Candidate vector votes: `{0}`".format(summary.get("candidate_vector_count", "not available")))
        lines.append("- Dominant cluster confidence: `{0}`".format(summary.get("confidence", "not available")))
        warning = summary.get("confidence_warning")
        if warning:
            lines.append("- Warning: {0}".format(warning))
        lines.append("")

    lines.extend(
        [
            "## Pair Matching Criteria",
            "",
            "For each positive blob and negative blob, the detector evaluates `vector = negative_center - positive_center`. A pair is accepted when:",
            "",
            "- The vector is close to the estimated target vector or its negative.",
            f"- `min(norm(vector-target), norm(vector+target)) <= {PAIR_VECTOR_ERROR_MAX:g} px`.",
            f"- Pair distance is between `{PAIR_DISTANCE_MIN:g}` and `{PAIR_DISTANCE_MAX:g}` px.",
            "- Score is `pos_max_z + abs(neg_min_z) - 0.5*vector_error - size_mismatch_penalty - elongation_penalty`.",
            "",
            "## Artifacts",
            "",
            "- `pairs_manual.csv` and `pairs_auto.csv`",
            "- `positive_blobs.csv` and `negative_blobs.csv`",
            "- `refined_seed_pairs.csv`",
            "- `auto_vector_summary.json`",
            "- `residual_z_preview.jpg`",
            "- `positive_negative_overlay.jpg`",
            "- `pair_overlay_manual.jpg` and `pair_overlay_auto.jpg`",
            "- `vector_vote_heatmap.jpg`",
            "- `contact_sheet_manual.jpg` and `contact_sheet_auto.jpg`",
            "- `crops_manual/` and `crops_auto/` top candidate crop JPGs",
            "",
            "## Interpretation Warning",
            "",
            "This run is only a dipole candidate probe. Matched candidates are not proof of true defects and require expert review.",
        ]
    )
    (review_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def blob_fieldnames() -> List[str]:
    return list(Blob.__dataclass_fields__.keys())


def pair_fieldnames() -> List[str]:
    return list(PairMatch.__dataclass_fields__.keys())


def seed_fieldnames() -> List[str]:
    return list(RefinedSeedPair.__dataclass_fields__.keys())


def run_detector(args: argparse.Namespace) -> Dict:
    residual_path = Path(args.residual)
    out_dir = Path(args.out)
    ensure_dir(out_dir)
    ensure_dir(out_dir / "reports")
    ensure_dir(out_dir / "crops")

    image, image_stats = read_residual(residual_path)
    _work, _highpass, z, preprocess_stats = preprocess(image)

    positive_blobs = detect_blobs(z, "positive")
    negative_blobs = detect_blobs(z, "negative")
    write_csv(out_dir / "positive_blobs.csv", positive_blobs, blob_fieldnames())
    write_csv(out_dir / "negative_blobs.csv", negative_blobs, blob_fieldnames())

    save_z_preview(z, out_dir / "residual_z_preview.png")
    plot_blob_overlay(z, positive_blobs, negative_blobs, out_dir / "positive_negative_overlay.png")

    refined_pairs: List[RefinedSeedPair] = []
    auto_summary: Optional[AutoVectorSummary] = None
    initial_target = canonical_vector(float(args.target_dx), float(args.target_dy))
    if args.auto_vector:
        mode = "auto"
        target_vector, auto_summary, _vector_array = estimate_auto_vector(positive_blobs, negative_blobs, initial_target, out_dir)
        write_json(out_dir / "auto_vector_summary.json", asdict(auto_summary))
    else:
        mode = "manual"
        seed_pairs = parse_seed_pairs(args.seed_pairs or "")
        refined_pairs, target_vector = refine_seed_pairs(z, seed_pairs)
        write_csv(out_dir / "refined_seed_pairs.csv", refined_pairs, seed_fieldnames())

    matches = match_pairs(positive_blobs, negative_blobs, target_vector)
    write_csv(out_dir / "pairs.csv", matches, pair_fieldnames())
    plot_pair_overlay(z, matches, target_vector, out_dir / "pair_overlay.png", mode)
    create_crops_and_contact_sheet(image, z, matches, out_dir)

    write_run_summary(
        out_dir=out_dir,
        mode=mode,
        image_stats=image_stats,
        preprocess_stats=preprocess_stats,
        target_vector=target_vector,
        positive_count=len(positive_blobs),
        negative_count=len(negative_blobs),
        pair_count=len(matches),
        refined_pairs=refined_pairs,
        auto_summary=auto_summary,
    )
    build_review_package(
        mode=mode,
        out_dir=out_dir,
        image_stats=image_stats,
        preprocess_stats=preprocess_stats,
        target_vector=target_vector,
        positive_count=len(positive_blobs),
        negative_count=len(negative_blobs),
        pair_count=len(matches),
        refined_pairs=refined_pairs,
        auto_summary=auto_summary,
        seed_pairs_text=args.seed_pairs or "",
    )

    result = {
        "mode": mode,
        "out_dir": str(out_dir),
        "review_dir": str(REVIEW_DIR),
        "image_stats": image_stats,
        "preprocess_stats": preprocess_stats,
        "target_vector": {"dx": float(target_vector[0]), "dy": float(target_vector[1])},
        "positive_blob_count": len(positive_blobs),
        "negative_blob_count": len(negative_blobs),
        "matched_pair_count": len(matches),
        "refined_seed_pairs": [asdict(item) for item in refined_pairs],
        "auto_vector_summary": asdict(auto_summary) if auto_summary else None,
        "top10_pairs": [asdict(item) for item in matches[:10]],
    }
    print(json.dumps(result, indent=2, ensure_ascii=False, default=to_builtin))
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Dipole Pair Detector v1")
    parser.add_argument("--residual", required=True, help="Input residual/divide TIFF path")
    parser.add_argument("--seed-pairs", default="", help="Manual seed pairs as 'x1,y1,x2,y2;x1,y1,x2,y2'")
    parser.add_argument("--target-dx", type=float, default=30.0, help="Initial or fixed target dx in pixels")
    parser.add_argument("--target-dy", type=float, default=0.0, help="Initial or fixed target dy in pixels")
    parser.add_argument("--auto-vector", action="store_true", help="Estimate target vector automatically from blob vector votes")
    parser.add_argument("--out", required=True, help="Output directory")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        run_detector(args)
    except DetectorError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
