"""
Extract frames from video files with optional on-disk cache.

Usage:
  python video_to_frames.py data/mmexport1736520716793.mp4
  python video_to_frames.py data/mmexport1736520716793.mp4 --frame-interval 10
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
DEFAULT_CACHE_DIR = Path(".cache/frames")
META_FILENAME = "meta.json"


@dataclass
class ExtractedFrames:
    """Frames extracted from a video, stored in cache or held in memory."""

    video_path: Path
    frame_paths: list[Path]
    fps: float
    frame_interval: int
    cache_dir: Path | None
    frames: list["np.ndarray"] | None = None

    def __len__(self) -> int:
        return len(self.frame_paths) if self.frame_paths else len(self.frames or [])


def find_videos(data_dir: Path) -> list[Path]:
    """Return video files directly under *data_dir* (non-recursive)."""
    if not data_dir.is_dir():
        return []
    return sorted(
        path
        for path in data_dir.iterdir()
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )


def _cache_key(video_path: Path, frame_interval: int) -> str:
    stat = video_path.stat()
    payload = f"{video_path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}:{frame_interval}"
    digest = hashlib.md5(payload.encode(), usedforsecurity=False).hexdigest()[:12]
    return f"{video_path.stem}_{digest}"


def _meta_path(cache_dir: Path) -> Path:
    return cache_dir / META_FILENAME


def _load_meta(cache_dir: Path) -> dict | None:
    meta_path = _meta_path(cache_dir)
    if not meta_path.is_file():
        return None
    with open(meta_path, encoding="utf-8") as f:
        return json.load(f)


def _save_meta(cache_dir: Path, meta: dict) -> None:
    with open(_meta_path(cache_dir), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def _cache_is_valid(cache_dir: Path, video_path: Path, frame_interval: int) -> bool:
    meta = _load_meta(cache_dir)
    if meta is None:
        return False

    expected_count = int(meta.get("saved_count", -1))
    frame_paths = sorted(cache_dir.glob("frame_*.jpg"))
    if len(frame_paths) != expected_count or expected_count <= 0:
        return False

    stat = video_path.stat()
    return (
        meta.get("video_path") == str(video_path.resolve())
        and meta.get("frame_interval") == frame_interval
        and meta.get("video_size") == stat.st_size
        and meta.get("video_mtime_ns") == stat.st_mtime_ns
    )


def video_to_frames(
    video_path: Path | str,
    cache_dir: Path | str | None = DEFAULT_CACHE_DIR,
    frame_interval: int = 1,
    use_cache: bool = True,
    load_into_memory: bool = False,
) -> ExtractedFrames:
    """
    Extract frames from a video, optionally caching them under ``.cache/frames``.

    Returns frame paths when cached on disk, or in-memory arrays when
    ``load_into_memory=True`` and no cache directory is used.
    """
    import cv2

    video_path = Path(video_path)
    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if frame_interval < 1:
        raise ValueError("frame_interval must be >= 1")

    resolved_cache: Path | None = None
    if cache_dir is not None:
        base_cache = Path(cache_dir)
        resolved_cache = base_cache / _cache_key(video_path, frame_interval)

    if use_cache and resolved_cache is not None and _cache_is_valid(
        resolved_cache, video_path, frame_interval
    ):
        meta = _load_meta(resolved_cache)
        frame_paths = sorted(resolved_cache.glob("frame_*.jpg"))
        fps = float(meta.get("fps", 30.0))
        frames = None
        if load_into_memory:
            frames = []
            for frame_path in frame_paths:
                image = cv2.imread(str(frame_path))
                if image is None:
                    raise ValueError(f"Could not read cached frame: {frame_path}")
                frames.append(image)
        print(f"Using cached frames: {resolved_cache} ({len(frame_paths)} frames)")
        return ExtractedFrames(
            video_path=video_path,
            frame_paths=frame_paths,
            fps=fps,
            frame_interval=frame_interval,
            cache_dir=resolved_cache,
            frames=frames,
        )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    frame_paths: list[Path] = []
    frames: list = []
    frame_count = 0
    saved_count = 0

    if resolved_cache is not None:
        resolved_cache.mkdir(parents=True, exist_ok=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            if resolved_cache is not None:
                save_path = resolved_cache / f"frame_{saved_count:06d}.jpg"
                cv2.imwrite(str(save_path), frame)
                frame_paths.append(save_path)
            elif load_into_memory:
                frames.append(frame.copy())
            saved_count += 1

        frame_count += 1

    cap.release()

    if resolved_cache is not None:
        stat = video_path.stat()
        _save_meta(
            resolved_cache,
            {
                "video_path": str(video_path.resolve()),
                "video_size": stat.st_size,
                "video_mtime_ns": stat.st_mtime_ns,
                "frame_interval": frame_interval,
                "fps": fps,
                "saved_count": saved_count,
            },
        )
        print(f"Cached {saved_count} frame(s) -> {resolved_cache}")
    else:
        print(f"Extracted {saved_count} frame(s) into memory")

    return ExtractedFrames(
        video_path=video_path,
        frame_paths=frame_paths,
        fps=fps,
        frame_interval=frame_interval,
        cache_dir=resolved_cache,
        frames=frames if load_into_memory and resolved_cache is None else None,
    )


def video_to_frames_opencv(
    video_path: str | Path,
    output_folder: str | Path | None = None,
    frame_interval: int = 1,
    cache_dir: str | Path | None = DEFAULT_CACHE_DIR,
) -> ExtractedFrames:
    """Backward-compatible wrapper around :func:`video_to_frames`."""
    video_path = Path(video_path)
    if output_folder is not None:
        return video_to_frames(
            video_path,
            cache_dir=output_folder,
            frame_interval=frame_interval,
            use_cache=False,
        )
    return video_to_frames(
        video_path,
        cache_dir=cache_dir,
        frame_interval=frame_interval,
        use_cache=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract frames from a video file.")
    parser.add_argument("video", type=Path, help="Path to the input video.")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help=f"Cache directory (default: {DEFAULT_CACHE_DIR}).",
    )
    parser.add_argument(
        "--frame-interval",
        type=int,
        default=1,
        help="Save one frame every N frames (default: 1).",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Extract into memory without writing cache files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    video_path = args.video.resolve()

    if not video_path.is_file():
        print(f"Error: video not found: {video_path}", file=sys.stderr)
        sys.exit(1)
    if args.frame_interval < 1:
        print("Error: --frame-interval must be >= 1.", file=sys.stderr)
        sys.exit(1)

    try:
        extracted = video_to_frames(
            video_path,
            cache_dir=None if args.no_cache else args.cache_dir,
            frame_interval=args.frame_interval,
            use_cache=not args.no_cache,
            load_into_memory=args.no_cache,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Video: {extracted.video_path}")
    print(f"Frames: {len(extracted)}")
    print(f"FPS: {extracted.fps}")
    if extracted.cache_dir is not None:
        print(f"Cache: {extracted.cache_dir}")


if __name__ == "__main__":
    main()
