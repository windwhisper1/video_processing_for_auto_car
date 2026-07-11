"""
Assemble image frames into an MP4 video (inverse of video_to_frames.py).

Usage:
  python frames_to_video.py path/to/frames --output output.mp4 --fps 30
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import cv2

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
FRAME_RE = re.compile(r"frame_(\d+)", re.IGNORECASE)


def _frame_sort_key(path: Path) -> tuple[int, int, str]:
    match = FRAME_RE.search(path.name)
    if match:
        return (0, int(match.group(1)), path.name.lower())
    return (1, 0, path.name.lower())


def list_frame_paths(frames_dir: Path) -> list[Path]:
    paths = [
        path
        for path in frames_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return sorted(paths, key=_frame_sort_key)


def frames_to_video(frames_dir: Path, output_path: Path, fps: float) -> int:
    frame_paths = list_frame_paths(frames_dir)
    if not frame_paths:
        raise FileNotFoundError(f"No image frames found in: {frames_dir}")

    first = cv2.imread(str(frame_paths[0]))
    if first is None:
        raise ValueError(f"Could not read frame: {frame_paths[0]}")

    height, width = first.shape[:2]
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for: {output_path}")

    written = 0
    try:
        for frame_path in frame_paths:
            frame = cv2.imread(str(frame_path))
            if frame is None:
                print(
                    f"Warning: skipping unreadable frame: {frame_path}",
                    file=sys.stderr,
                )
                continue

            if frame.shape[1] != width or frame.shape[0] != height:
                frame = cv2.resize(
                    frame,
                    (width, height),
                    interpolation=cv2.INTER_AREA,
                )

            writer.write(frame)
            written += 1
    finally:
        writer.release()

    if written == 0:
        raise ValueError("No frames were written to the output video.")

    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Assemble image frames into an MP4 video."
    )
    parser.add_argument("frames_dir", type=Path, help="Directory containing frame images.")
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output .mp4 path (default: <parent>/<frames_dir_name>.mp4).",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Output video frame rate (default: 30).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frames_dir = args.frames_dir.resolve()

    if not frames_dir.is_dir():
        print(f"Error: not a directory: {frames_dir}", file=sys.stderr)
        sys.exit(1)

    if args.fps <= 0:
        print("Error: --fps must be greater than 0.", file=sys.stderr)
        sys.exit(1)

    output_path = (
        args.output.resolve()
        if args.output is not None
        else frames_dir.parent / f"{frames_dir.name}.mp4"
    )

    try:
        written = frames_to_video(frames_dir, output_path, args.fps)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Frames directory: {frames_dir}")
    print(f"Output video: {output_path}")
    print(f"FPS: {args.fps}")
    print(f"Written frames: {written}")


if __name__ == "__main__":
    main()
