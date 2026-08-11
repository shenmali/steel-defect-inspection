"""Bounded upload reads and local annotation retention."""

from datetime import timedelta
from pathlib import Path
from time import time
from uuid import uuid4

import cv2
import numpy as np
from fastapi import UploadFile

from steel_inspection.inference.annotate import annotate_mask
from steel_inspection.inference.types import PredictionResult


UPLOAD_CHUNK_BYTES = 1024 * 1024
ANNOTATION_TTL = timedelta(hours=24)
MAX_ANNOTATION_BYTES = 1024 * 1024 * 1024


class UploadTooLargeError(ValueError):
    """Raised when an upload exceeds its allowed body size."""


class AnnotationStorageError(RuntimeError):
    """Raised when an annotation cannot be retained safely."""


async def read_limited_upload(upload: UploadFile, limit: int) -> bytes:
    """Read an upload in bounded chunks without retaining an over-limit body."""
    chunks: list[bytes] = []
    size = 0
    while chunk := await upload.read(min(UPLOAD_CHUNK_BYTES, limit + 1 - size)):
        size += len(chunk)
        if size > limit:
            raise UploadTooLargeError
        chunks.append(chunk)
    return b"".join(chunks)


def store_annotation(
    source_bgr: np.ndarray,
    result: PredictionResult,
    *,
    results_dir: Path,
    enabled: bool = False,
    ttl: timedelta = ANNOTATION_TTL,
    maximum_total_bytes: int = MAX_ANNOTATION_BYTES,
) -> Path | None:
    """Optionally persist an annotated PNG while enforcing local retention limits."""
    if not enabled:
        return None

    annotated = annotate_mask(source_bgr, result.mask)
    success, encoded = cv2.imencode(".png", annotated)
    if not success:
        raise AnnotationStorageError("Unable to encode annotated PNG")
    png_bytes = encoded.tobytes()

    try:
        resolved_results_dir = _resolve_results_dir(results_dir)
        _purge_expired_pngs(resolved_results_dir, ttl)
        _make_room_for_png(resolved_results_dir, len(png_bytes), maximum_total_bytes)
        filename = f"{uuid4()}.png"
        (resolved_results_dir / filename).write_bytes(png_bytes)
    except OSError as error:
        raise AnnotationStorageError("Unable to store annotated PNG") from error

    return results_dir / filename


def _resolve_results_dir(results_dir: Path) -> Path:
    """Create and resolve the single directory allowed to hold retained annotations."""
    results_dir.mkdir(parents=True, exist_ok=True)
    resolved = results_dir.resolve(strict=True)
    if not resolved.is_dir():
        raise OSError("Annotation results path is not a directory")
    return resolved


def _regular_png_files(results_dir: Path) -> list[Path]:
    """Return only regular PNG files immediately inside the resolved results directory."""
    return [
        path
        for path in results_dir.iterdir()
        if path.suffix.lower() == ".png"
        and path.is_file()
        and not path.is_symlink()
        and path.resolve().parent == results_dir
    ]


def _purge_expired_pngs(results_dir: Path, ttl: timedelta) -> None:
    """Delete only expired regular PNG annotations from the configured results directory."""
    expires_before = time() - ttl.total_seconds()
    for path in _regular_png_files(results_dir):
        if path.stat().st_mtime < expires_before:
            path.unlink()


def _make_room_for_png(results_dir: Path, incoming_size: int, maximum_total_bytes: int) -> None:
    """Trim oldest regular PNGs until the incoming annotation fits the total cap."""
    if maximum_total_bytes < 0:
        raise ValueError("Annotation capacity must not be negative")

    files = _regular_png_files(results_dir)
    sizes = {path: path.stat().st_size for path in files}
    total_size = sum(sizes.values())
    for path in sorted(files, key=lambda candidate: (candidate.stat().st_mtime, candidate.name)):
        if total_size + incoming_size <= maximum_total_bytes:
            break
        path.unlink()
        total_size -= sizes[path]

    if total_size + incoming_size > maximum_total_bytes:
        raise OSError("Annotated PNG exceeds the configured retention capacity")
