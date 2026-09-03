"""Download pinned upstream research data with integrity verification."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CHUNK_SIZE = 1024 * 1024
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MANIFEST = Path(__file__).with_name("sources.json")
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "raw"


class IntegrityError(RuntimeError):
    """Raised when an acquired file does not match its pinned metadata."""


@dataclass(frozen=True)
class SourceFile:
    relative_path: Path
    url: str
    bytes: int
    sha256: str

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> SourceFile:
        relative_path = Path(str(value["relative_path"]))
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"Unsafe source path: {relative_path}")
        return cls(
            relative_path=relative_path,
            url=str(value["url"]),
            bytes=int(value["bytes"]),
            sha256=str(value["sha256"]),
        )


def load_sources(manifest_path: Path) -> list[SourceFile]:
    with manifest_path.open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("Source manifest must contain a non-empty files list")
    return [SourceFile.from_dict(item) for item in files]


def digest_file(file_path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with file_path.open("rb") as stream:
        while chunk := stream.read(CHUNK_SIZE):
            size += len(chunk)
            digest.update(chunk)
    return size, digest.hexdigest()


def verify_file(file_path: Path, source: SourceFile) -> None:
    actual_size, actual_digest = digest_file(file_path)
    if actual_size != source.bytes or actual_digest != source.sha256:
        raise IntegrityError(
            f"Integrity check failed for {file_path}: "
            f"expected {source.bytes} bytes/{source.sha256}, "
            f"got {actual_size} bytes/{actual_digest}"
        )


def download_file(source: SourceFile, output_dir: Path, *, force: bool = False) -> str:
    destination = output_dir / source.relative_path
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists() and not force:
        verify_file(destination, source)
        return "verified"

    partial = destination.with_suffix(f"{destination.suffix}.partial")
    if partial.exists():
        partial.unlink()

    request = urllib.request.Request(
        source.url,
        headers={"User-Agent": "claim-detector-research/0.1"},
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response, partial.open("wb") as output:
            shutil.copyfileobj(response, output, length=CHUNK_SIZE)
        verify_file(partial, source)
        partial.replace(destination)
    except Exception:
        partial.unlink(missing_ok=True)
        raise
    return "downloaded"


def acquire_all(manifest_path: Path, output_dir: Path, *, force: bool = False) -> list[str]:
    results = []
    for source in load_sources(manifest_path):
        status = download_file(source, output_dir, force=force)
        results.append(f"{status:10} {source.relative_path}")
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace existing files and verify the replacement",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    for result in acquire_all(args.manifest, args.output_dir, force=args.force):
        print(result)


if __name__ == "__main__":
    main()
