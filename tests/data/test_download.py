from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from claim_detector.data.download import IntegrityError, acquire_all, load_sources


def write_manifest(tmp_path: Path, payload: bytes) -> Path:
    source = tmp_path / "source.txt"
    source.write_bytes(payload)
    manifest = tmp_path / "sources.json"
    manifest.write_text(
        json.dumps(
            {
                "files": [
                    {
                        "relative_path": "nested/result.txt",
                        "url": source.as_uri(),
                        "bytes": len(payload),
                        "sha256": hashlib.sha256(payload).hexdigest(),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    return manifest


def test_acquisition_downloads_then_verifies(tmp_path: Path) -> None:
    manifest = write_manifest(tmp_path, b"real bytes")
    output = tmp_path / "output"

    assert acquire_all(manifest, output) == ["downloaded nested/result.txt"]
    assert acquire_all(manifest, output) == ["verified   nested/result.txt"]
    assert (output / "nested" / "result.txt").read_bytes() == b"real bytes"


def test_existing_corrupt_file_fails_closed(tmp_path: Path) -> None:
    manifest = write_manifest(tmp_path, b"expected")
    output_file = tmp_path / "output" / "nested" / "result.txt"
    output_file.parent.mkdir(parents=True)
    output_file.write_bytes(b"corrupt")

    with pytest.raises(IntegrityError, match="Integrity check failed"):
        acquire_all(manifest, tmp_path / "output")


def test_manifest_rejects_parent_traversal(tmp_path: Path) -> None:
    manifest = tmp_path / "sources.json"
    manifest.write_text(
        json.dumps(
            {
                "files": [
                    {
                        "relative_path": "../escape",
                        "url": "file:///unused",
                        "bytes": 0,
                        "sha256": hashlib.sha256(b"").hexdigest(),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Unsafe source path"):
        load_sources(manifest)
