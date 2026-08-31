"""Batch-inference stage: pretrained object detection over preprocessed camera images.

Exposed as a callable class rather than a plain function so Ray Data runs it on an
actor pool: model weights (~160MB) are loaded once per worker in `__init__` instead
of once per batch, which is the whole reason this stage is stateful.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import torch
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_Weights,
    fasterrcnn_resnet50_fpn,
)

DEFAULT_SCORE_THRESHOLD = 0.5

DETECTION_STRUCT = pa.struct(
    [
        ("x1", pa.float32()),
        ("y1", pa.float32()),
        ("x2", pa.float32()),
        ("y2", pa.float32()),
        ("label", pa.int64()),
        ("label_name", pa.string()),
        ("score", pa.float32()),
    ]
)
DETECTIONS_TYPE = pa.list_(DETECTION_STRUCT)


def resolve_device(requested: str | None = None) -> str:
    """Pick an inference device, preferring an explicit request then the best available."""
    if requested is not None:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def filter_detections(
    prediction: dict[str, torch.Tensor],
    categories: list[str],
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
) -> list[dict]:
    """Drop low-confidence detections, returning one flat dict per surviving box."""
    scores = prediction["scores"].detach().cpu().numpy()
    keep = scores >= score_threshold

    boxes = prediction["boxes"].detach().cpu().numpy()[keep]
    labels = prediction["labels"].detach().cpu().numpy()[keep]
    return [
        {
            "x1": float(box[0]),
            "y1": float(box[1]),
            "x2": float(box[2]),
            "y2": float(box[3]),
            "label": int(label),
            "label_name": categories[label],
            "score": float(score),
        }
        for box, label, score in zip(boxes, labels, scores[keep], strict=True)
    ]


class DetectionPredictor:
    """Ray Data actor: runs a pretrained Faster R-CNN over batches of camera tensors."""

    def __init__(
        self,
        device: str | None = None,
        score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    ) -> None:
        self.device = resolve_device(device)
        self.score_threshold = score_threshold

        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        self.categories = weights.meta["categories"]
        self.model = fasterrcnn_resnet50_fpn(weights=weights).eval().to(self.device)

    def __call__(self, batch: dict[str, np.ndarray]) -> pa.Table:
        images = torch.from_numpy(batch["image"]).to(self.device)
        with torch.inference_mode():
            predictions = self.model(list(images))

        detections = [
            filter_detections(prediction, self.categories, self.score_threshold)
            for prediction in predictions
        ]

        # The image tensor is deliberately not propagated: it is ~4.3MB/record, nothing
        # downstream reads it, and persisting it would dominate the Parquet output.
        columns = {name: pa.array(values) for name, values in batch.items() if name != "image"}
        # DETECTIONS_TYPE is declared rather than inferred: a block in which every record
        # has zero detections gives Arrow nothing to infer a struct from, yielding
        # list<null> and a Parquet dataset whose files no longer share one schema.
        columns["detections"] = pa.array(detections, type=DETECTIONS_TYPE)
        columns["num_detections"] = pa.array(
            [len(record) for record in detections], type=pa.int64()
        )
        return pa.table(columns)
