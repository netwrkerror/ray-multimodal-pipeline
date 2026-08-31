import pyarrow as pa
import pytest
import torch

from ray_multimodal_pipeline.inference import (
    DETECTIONS_TYPE,
    filter_detections,
    resolve_device,
)

CATEGORIES = ["__background__", "person", "bicycle", "car"]


def _prediction(boxes: list[list[float]], labels: list[int], scores: list[float]) -> dict:
    return {
        "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
        "labels": torch.tensor(labels, dtype=torch.int64),
        "scores": torch.tensor(scores, dtype=torch.float32),
    }


def test_filter_detections_drops_low_confidence() -> None:
    prediction = _prediction(
        boxes=[[0, 0, 10, 10], [5, 5, 20, 20], [1, 1, 2, 2]],
        labels=[3, 1, 2],
        scores=[0.9, 0.4, 0.75],
    )

    detections = filter_detections(prediction, CATEGORIES, score_threshold=0.5)

    assert [d["label_name"] for d in detections] == ["car", "bicycle"]
    assert [pytest.approx(d["score"], rel=1e-6) for d in detections] == [0.9, 0.75]
    assert detections[0]["x1"] == 0.0 and detections[0]["y2"] == 10.0


def test_filter_detections_handles_no_surviving_detections() -> None:
    prediction = _prediction(boxes=[[0, 0, 10, 10]], labels=[1], scores=[0.1])

    assert filter_detections(prediction, CATEGORIES, score_threshold=0.5) == []


def test_filter_detections_threshold_is_inclusive() -> None:
    prediction = _prediction(boxes=[[0, 0, 1, 1]], labels=[1], scores=[0.5])

    assert len(filter_detections(prediction, CATEGORIES, score_threshold=0.5)) == 1


def test_detections_match_declared_arrow_schema() -> None:
    prediction = _prediction(boxes=[[0, 0, 10, 10]], labels=[3], scores=[0.9])
    detections = filter_detections(prediction, CATEGORIES, score_threshold=0.5)

    array = pa.array([detections], type=DETECTIONS_TYPE)

    assert array.type == DETECTIONS_TYPE
    assert array[0][0]["label_name"].as_py() == "car"


def test_all_empty_block_still_matches_declared_schema() -> None:
    """Inferred types would collapse to list<null> here and desync the Parquet schema."""
    array = pa.array([[], [], []], type=DETECTIONS_TYPE)

    assert array.type == DETECTIONS_TYPE
    assert len(array) == 3


def test_resolve_device_honours_explicit_request() -> None:
    assert resolve_device("cpu") == "cpu"


def test_resolve_device_falls_back_to_available_hardware() -> None:
    assert resolve_device() in {"cuda", "mps", "cpu"}
