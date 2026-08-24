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


def preprocess_batch(batch: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Ray Data `map_batches` entry point: decode/resize/normalize images, load lidar points."""
    images = []
    lidar_points = np.empty(len(batch["cam_front_path"]), dtype=object)
    for i, (cam_path, lidar_path) in enumerate(
        zip(batch["cam_front_path"], batch["lidar_top_path"], strict=True)
    ):
        image = decode_camera_image(cam_path)
        images.append(resize_and_normalize(image).numpy())
        lidar_points[i] = load_lidar_points(lidar_path)

    return {
        "sample_token": batch["sample_token"],
        "timestamp": batch["timestamp"],
        "image": np.stack(images),
        "lidar_points": lidar_points,
    }
