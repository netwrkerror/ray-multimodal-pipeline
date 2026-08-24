import os

from ray_multimodal_pipeline.manifest import build_manifest


class _FakeNuScenes:
    def __init__(self, dataroot: str = "/data/nuscenes") -> None:
        self.dataroot = dataroot
        self.sample = [
            {
                "token": "sample-0",
                "timestamp": 1000,
                "data": {"CAM_FRONT": "cam-0", "LIDAR_TOP": "lidar-0"},
            },
            {
                "token": "sample-1",
                "timestamp": 2000,
                "data": {"CAM_FRONT": "cam-1", "LIDAR_TOP": "lidar-1"},
            },
        ]
        self._sample_data = {
            "cam-0": {"filename": "samples/CAM_FRONT/0.jpg"},
            "lidar-0": {"filename": "samples/LIDAR_TOP/0.bin"},
            "cam-1": {"filename": "samples/CAM_FRONT/1.jpg"},
            "lidar-1": {"filename": "samples/LIDAR_TOP/1.bin"},
        }

    def get(self, table_name: str, token: str) -> dict:
        assert table_name == "sample_data"
        return self._sample_data[token]


def test_build_manifest_maps_samples_to_paths() -> None:
    manifest = build_manifest(_FakeNuScenes())

    assert manifest == [
        {
            "sample_token": "sample-0",
            "cam_front_path": "/data/nuscenes/samples/CAM_FRONT/0.jpg",
            "lidar_top_path": "/data/nuscenes/samples/LIDAR_TOP/0.bin",
            "timestamp": 1000,
        },
        {
            "sample_token": "sample-1",
            "cam_front_path": "/data/nuscenes/samples/CAM_FRONT/1.jpg",
            "lidar_top_path": "/data/nuscenes/samples/LIDAR_TOP/1.bin",
            "timestamp": 2000,
        },
    ]


def test_build_manifest_absolutizes_relative_dataroot() -> None:
    """Ray workers run from their own cwd, so manifest paths must not stay relative."""
    manifest = build_manifest(_FakeNuScenes(dataroot="data/nuscenes"))

    for record in manifest:
        assert os.path.isabs(record["cam_front_path"])
        assert os.path.isabs(record["lidar_top_path"])
    assert manifest[0]["cam_front_path"] == os.path.abspath("data/nuscenes/samples/CAM_FRONT/0.jpg")
