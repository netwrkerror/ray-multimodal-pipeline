from ray_multimodal_pipeline.ingest import build_ingest_dataset


def test_build_ingest_dataset_preserves_records() -> None:
    manifest = [
        {
            "sample_token": "a",
            "cam_front_path": "/x/a.jpg",
            "lidar_top_path": "/x/a.bin",
            "timestamp": 1,
        },
        {
            "sample_token": "b",
            "cam_front_path": "/x/b.jpg",
            "lidar_top_path": "/x/b.bin",
            "timestamp": 2,
        },
    ]

    dataset = build_ingest_dataset(manifest)

    assert dataset.count() == 2
    rows = sorted(dataset.take_all(), key=lambda r: r["sample_token"])
    assert rows[0]["sample_token"] == "a"
    assert rows[1]["timestamp"] == 2
