"""Preprocessing stage: decode camera images, resize, normalize; load lidar sweeps."""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import v2 as transforms_v2

DEFAULT_IMAGE_SIZE = (450, 800)  # (height, width)
LIDAR_POINT_DIM = 5  # x, y, z, intensity, ring — nuScenes LIDAR_TOP .bin layout

_TO_TENSOR = transforms_v2.Compose(
    [
        transforms_v2.ToImage(),
        transforms_v2.ToDtype(torch.float32, scale=True),
    ]
)


def decode_camera_image(path: str) -> Image.Image:
    """Decode a camera JPEG into an RGB PIL image."""
    return Image.open(path).convert("RGB")


def resize_and_normalize(
    image: Image.Image, size: tuple[int, int] = DEFAULT_IMAGE_SIZE
) -> torch.Tensor:
    """Resize to `size` and scale to float32 [0, 1], channel-first (C, H, W).

    Deliberately skips ImageNet mean/std normalization: the torchvision detection
    models used in the inference stage apply their own internal normalization
    (`GeneralizedRCNNTransform`) on [0, 1] input, so normalizing here too would
    double-normalize.
    """
    resized = transforms_v2.functional.resize(image, list(size))
    return _TO_TENSOR(resized)


def load_lidar_points(path: str) -> np.ndarray:
    """Load a LIDAR_TOP .bin sweep as an (N, 5) array of x, y, z, intensity, ring."""
    return np.fromfile(path, dtype=np.float32).reshape(-1, LIDAR_POINT_DIM)


def summarize_lidar_points(points: np.ndarray) -> dict[str, float | int]:
    """Reduce a lidar sweep to flat scalar features.

    Summarizing here lets the pipeline drop the raw sweep (~700KB/record) before the
    inference stage, which never reads it — only these derived features reach the output.
    """
    if points.shape[0] == 0:
        return {
            "lidar_num_points": 0,
            "lidar_mean_range_m": 0.0,
            "lidar_max_range_m": 0.0,
            "lidar_mean_intensity": 0.0,
            "lidar_z_min_m": 0.0,
            "lidar_z_max_m": 0.0,
        }

    xyz = points[:, :3]
    ranges = np.linalg.norm(xyz, axis=1)
    return {
        "lidar_num_points": int(points.shape[0]),
        "lidar_mean_range_m": float(ranges.mean()),
        "lidar_max_range_m": float(ranges.max()),
        "lidar_mean_intensity": float(points[:, 3].mean()),
        "lidar_z_min_m": float(xyz[:, 2].min()),
        "lidar_z_max_m": float(xyz[:, 2].max()),
    }


LIDAR_FEATURE_COLUMNS = (
    "lidar_num_points",
    "lidar_mean_range_m",
    "lidar_max_range_m",
    "lidar_mean_intensity",
    "lidar_z_min_m",
    "lidar_z_max_m",
)


def preprocess_batch(batch: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Ray Data `map_batches` entry point: decode/resize/normalize images, summarize lidar."""
    images = []
    lidar_features: dict[str, list] = {column: [] for column in LIDAR_FEATURE_COLUMNS}
    for cam_path, lidar_path in zip(batch["cam_front_path"], batch["lidar_top_path"], strict=True):
        image = decode_camera_image(cam_path)
        images.append(resize_and_normalize(image).numpy())

        summary = summarize_lidar_points(load_lidar_points(lidar_path))
        for column in LIDAR_FEATURE_COLUMNS:
            lidar_features[column].append(summary[column])

    return {
        "sample_token": batch["sample_token"],
        "timestamp": batch["timestamp"],
        "image": np.stack(images),
        **{column: np.array(values) for column, values in lidar_features.items()},
    }
