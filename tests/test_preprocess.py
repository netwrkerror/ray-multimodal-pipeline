import numpy as np
import pytest
import torch
from PIL import Image

from ray_multimodal_pipeline.preprocess import (
    DEFAULT_IMAGE_SIZE,
    LIDAR_FEATURE_COLUMNS,
    decode_camera_image,
    load_lidar_points,
    preprocess_batch,
    resize_and_normalize,
    summarize_lidar_points,
)


def _write_camera_image(path, size: tuple[int, int] = (1600, 900)) -> None:
    Image.new("RGB", size, color=(128, 64, 32)).save(path)


def _write_lidar_sweep(path, num_points: int = 10) -> np.ndarray:
    points = np.random.default_rng(0).random((num_points, 5)).astype(np.float32)
    points.tofile(path)
    return points


def test_decode_camera_image_returns_rgb(tmp_path) -> None:
    path = tmp_path / "cam.jpg"
    _write_camera_image(path)

    image = decode_camera_image(str(path))

    assert image.mode == "RGB"
    assert image.size == (1600, 900)


def test_resize_and_normalize_shapes_and_scales(tmp_path) -> None:
    path = tmp_path / "cam.jpg"
    _write_camera_image(path)
    image = decode_camera_image(str(path))

    tensor = resize_and_normalize(image)

    assert tensor.shape == (3, *DEFAULT_IMAGE_SIZE)
    assert tensor.dtype == torch.float32
    assert tensor.min() >= 0.0
    assert tensor.max() <= 1.0


def test_load_lidar_points_roundtrip(tmp_path) -> None:
    path = tmp_path / "lidar.bin"
    points = _write_lidar_sweep(path, num_points=7)

    loaded = load_lidar_points(str(path))

    assert loaded.shape == (7, 5)
    np.testing.assert_allclose(loaded, points)


def test_preprocess_batch_produces_expected_columns(tmp_path) -> None:
    cam_path = tmp_path / "cam.jpg"
    lidar_path = tmp_path / "lidar.bin"
    _write_camera_image(cam_path)
    _write_lidar_sweep(lidar_path, num_points=5)

    batch = {
        "sample_token": np.array(["tok-0", "tok-1"]),
        "timestamp": np.array([100, 200]),
        "cam_front_path": np.array([str(cam_path), str(cam_path)]),
        "lidar_top_path": np.array([str(lidar_path), str(lidar_path)]),
    }

    out = preprocess_batch(batch)

    assert list(out["sample_token"]) == ["tok-0", "tok-1"]
    assert out["image"].shape == (2, 3, *DEFAULT_IMAGE_SIZE)
    assert list(out["lidar_num_points"]) == [5, 5]
    for column in LIDAR_FEATURE_COLUMNS:
        assert len(out[column]) == 2


def test_preprocess_batch_drops_raw_lidar_points(tmp_path) -> None:
    """Raw sweeps are ~700KB/record and unread downstream — only summaries propagate."""
    cam_path = tmp_path / "cam.jpg"
    lidar_path = tmp_path / "lidar.bin"
    _write_camera_image(cam_path)
    _write_lidar_sweep(lidar_path, num_points=5)

    out = preprocess_batch(
        {
            "sample_token": np.array(["tok-0"]),
            "timestamp": np.array([100]),
            "cam_front_path": np.array([str(cam_path)]),
            "lidar_top_path": np.array([str(lidar_path)]),
        }
    )

    assert "lidar_points" not in out


def test_summarize_lidar_points_computes_ranges() -> None:
    points = np.array(
        [
            [3.0, 4.0, 1.0, 10.0, 0.0],  # range 5 (3-4-5 triangle with z=0 plane)
            [0.0, 0.0, 2.0, 20.0, 0.0],  # range 2
        ],
        dtype=np.float32,
    )

    summary = summarize_lidar_points(points)

    assert summary["lidar_num_points"] == 2
    assert summary["lidar_max_range_m"] == pytest.approx(np.sqrt(26.0))
    assert summary["lidar_mean_intensity"] == pytest.approx(15.0)
    assert summary["lidar_z_min_m"] == pytest.approx(1.0)
    assert summary["lidar_z_max_m"] == pytest.approx(2.0)


def test_summarize_lidar_points_handles_empty_sweep() -> None:
    summary = summarize_lidar_points(np.empty((0, 5), dtype=np.float32))

    assert summary["lidar_num_points"] == 0
    assert summary["lidar_max_range_m"] == 0.0
