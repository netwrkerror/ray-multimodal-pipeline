import os

from ray_multimodal_pipeline.pipeline import resolve_output_uri


def test_resolve_output_uri_absolutizes_relative_local_path() -> None:
    """Ray write tasks run from their own cwd, so a relative path escapes the project."""
    resolved = resolve_output_uri("data/outputs/detections")

    assert os.path.isabs(resolved)
    assert resolved == os.path.abspath("data/outputs/detections")


def test_resolve_output_uri_leaves_absolute_path_unchanged() -> None:
    assert resolve_output_uri("/var/data/detections") == "/var/data/detections"


def test_resolve_output_uri_passes_remote_uris_through() -> None:
    for uri in ("s3://bucket/detections", "gs://bucket/detections"):
        assert resolve_output_uri(uri) == uri
