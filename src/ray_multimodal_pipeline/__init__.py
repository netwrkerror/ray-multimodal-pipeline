"""Distributed multimodal sensor-data pipeline built on Ray.

Reads camera + lidar + metadata records, preprocesses them with Ray Data,
runs pretrained-model batch inference, and writes structured (Parquet) output.
"""

__version__ = "0.1.0"
