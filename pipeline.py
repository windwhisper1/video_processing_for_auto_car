"""
End-to-end perception pipeline:

  1. Extract video frames from ``data/`` (cached under ``.cache/frames``)
  2. Segment lane dashes (polygon outlines, no boxes)
  3. Detect obstacles (bounding boxes + confidence)
  4. Detect QR codes (polygon outline + decoded text)
  5. Combine overlays and HUD text (lane count + QR status)
  6. Assemble annotated frames into an output MP4

Usage:
  python pipeline.py
  python pipeline.py --frame-interval 10
  python pipeline.py --video data/mmexport1736520716793.mp4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

from detect_QR_code import (
    detect_qr_codes_from_image,
    qr_status_text,
    render_prediction_overlay as render_qr_overlay,
)
from frames_to_video import frames_to_video
from model.lane_lines import (
    ensure_opencv_headless,
    infer_frame as infer_lanes,
    load_yolo_model as load_lane_model,
    render_prediction_overlay as render_lane_overlay,
    resolve_weights as resolve_lane_weights,
)
from model.obstacles import (
    infer_frame as infer_obstacles,
    load_yolo_model as load_obstacle_model,
    render_prediction_overlay as render_obstacle_overlay,
    resolve_weights as resolve_obstacle_weights,
)
from video_to_frames import DEFAULT_CACHE_DIR, ExtractedFrames, find_videos, video_to_frames

DATA_DIR = Path("data")
OUTPUT_DIR = Path("output")
HUD_COLOR = (0, 255, 0)
# Avoid import-time failure when cv2 bindings are partially loaded on some servers.
HUD_FONT = getattr(cv2, "FONT_HERSHEY_SIMPLEX", 0)
HUD_LINE_TYPE = getattr(cv2, "LINE_AA", 16)


def read_frame(extracted: ExtractedFrames, index: int):
    if extracted.frames is not None:
        return extracted.frames[index].copy()

    frame_path = extracted.frame_paths[index]
    image = cv2.imread(str(frame_path))
    if image is None:
        raise ValueError(f"Could not read frame: {frame_path}")
    return image


def render_status_overlay(
    image_bgr,
    lane_count: int,
    qr_text: str,
) -> None:
    """Draw HUD text in the top-left corner."""
    lines = [f"detected lanes: {lane_count}", qr_text]
    y = 24
    for line in lines:
        cv2.putText(
            image_bgr,
            line,
            (12, y),
            HUD_FONT,
            0.65,
            HUD_COLOR,
            2,
            HUD_LINE_TYPE,
        )
        y += 28


def combine_frame(
    image_bgr,
    lane_polygons,
    obstacle_detections,
    qr_results,
):
    """Merge lane, obstacle, and QR overlays plus HUD text onto one frame."""
    overlay = render_lane_overlay(
        image_bgr,
        lane_polygons,
        show_labels=False,
    )
    overlay = render_obstacle_overlay(overlay, obstacle_detections)
    overlay = render_qr_overlay(overlay, qr_results)
    render_status_overlay(
        overlay,
        lane_count=len(lane_polygons),
        qr_text=qr_status_text(qr_results),
    )
    return overlay


def process_video(
    video_path: Path,
    *,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    frame_interval: int = 1,
    output_dir: Path = OUTPUT_DIR,
    lane_model=None,
    obstacle_model=None,
    device: str | None = None,
    lane_conf: float = 0.25,
    obstacle_conf: float = 0.25,
    iou: float = 0.5,
    use_cache: bool = True,
) -> Path:
    """Run the full pipeline for a single video and return the output MP4 path."""
    extracted = video_to_frames(
        video_path,
        cache_dir=cache_dir,
        frame_interval=frame_interval,
        use_cache=use_cache,
    )

    annotated_frames_dir = output_dir / "annotated_frames" / video_path.stem
    annotated_frames_dir.mkdir(parents=True, exist_ok=True)

    if lane_model is None:
        lane_model = load_lane_model(resolve_lane_weights(None))
    if obstacle_model is None:
        obstacle_model = load_obstacle_model(resolve_obstacle_weights(None))

    total = len(extracted)
    print(f"Processing {total} frame(s) from {video_path.name}")
    for index in range(total):
        frame = read_frame(extracted, index)
        lane_polygons = infer_lanes(
            lane_model,
            frame,
            device=device,
            conf=lane_conf,
            iou=iou,
        )
        obstacle_detections = infer_obstacles(
            obstacle_model,
            frame,
            device=device,
            conf=obstacle_conf,
            iou=iou,
        )
        qr_results = detect_qr_codes_from_image(frame)
        annotated = combine_frame(
            frame,
            lane_polygons,
            obstacle_detections,
            qr_results,
        )

        out_path = annotated_frames_dir / f"frame_{index:06d}.jpg"
        cv2.imwrite(str(out_path), annotated)

        if (index + 1) % 25 == 0 or index + 1 == total:
            print(
                f"[{video_path.name}] processed {index + 1}/{total} frame(s)",
                flush=True,
            )

    output_fps = extracted.fps / extracted.frame_interval
    output_video = output_dir / f"{video_path.stem}_annotated.mp4"
    written = frames_to_video(annotated_frames_dir, output_video, output_fps)

    print(f"Annotated frames: {annotated_frames_dir}")
    print(f"Output video: {output_video}")
    print(f"Written frames: {written}")
    print(f"Output FPS: {output_fps:.3f}")
    return output_video


def run_pipeline(
    data_dir: Path = DATA_DIR,
    output_dir: Path = OUTPUT_DIR,
    videos: list[Path] | None = None,
    **kwargs,
) -> list[Path]:
    """Process all videos under ``data_dir`` (or an explicit list)."""
    ensure_opencv_headless()
    output_dir.mkdir(parents=True, exist_ok=True)

    video_paths = videos or find_videos(data_dir)
    if not video_paths:
        raise FileNotFoundError(
            f"No video files found in {data_dir.resolve()}. "
            f"Place .mp4 files directly under the data folder."
        )

    lane_model = load_lane_model(resolve_lane_weights(kwargs.pop("lane_weights", None)))
    obstacle_model = load_obstacle_model(
        resolve_obstacle_weights(kwargs.pop("obstacle_weights", None))
    )

    output_videos: list[Path] = []
    for video_path in video_paths:
        print(f"Processing video: {video_path}")
        output_videos.append(
            process_video(
                video_path,
                output_dir=output_dir,
                lane_model=lane_model,
                obstacle_model=obstacle_model,
                **kwargs,
            )
        )
    return output_videos


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run lane segmentation, obstacle detection, and QR decoding on videos in data/."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help=f"Directory containing input videos (default: {DATA_DIR}).",
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=None,
        help="Process a single video instead of every video in --data-dir.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help=f"Directory for annotated frames and videos (default: {OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help=f"Frame cache directory (default: {DEFAULT_CACHE_DIR}).",
    )
    parser.add_argument(
        "--frame-interval",
        type=int,
        default=1,
        help="Process one frame every N source frames (default: 1).",
    )
    parser.add_argument(
        "--lane-weights",
        type=Path,
        default=None,
        help="Optional lane-line weights path.",
    )
    parser.add_argument(
        "--obstacle-weights",
        type=Path,
        default=None,
        help="Optional obstacle weights path.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device for inference (default: cuda:0 if available else cpu).",
    )
    parser.add_argument(
        "--lane-conf",
        type=float,
        default=0.25,
        help="Lane segmentation confidence threshold.",
    )
    parser.add_argument(
        "--obstacle-conf",
        type=float,
        default=0.25,
        help="Obstacle detection confidence threshold.",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.5,
        help="NMS IoU threshold for both models.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Re-extract frames without reading or writing the cache.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.frame_interval < 1:
        print("Error: --frame-interval must be >= 1.", file=sys.stderr)
        sys.exit(1)

    videos = [args.video.resolve()] if args.video is not None else None
    if args.video is not None and not args.video.is_file():
        print(f"Error: video not found: {args.video}", file=sys.stderr)
        sys.exit(1)

    try:
        run_pipeline(
            data_dir=args.data_dir.resolve(),
            output_dir=args.output_dir.resolve(),
            videos=videos,
            cache_dir=args.cache_dir.resolve(),
            frame_interval=args.frame_interval,
            lane_weights=str(args.lane_weights) if args.lane_weights else None,
            obstacle_weights=str(args.obstacle_weights) if args.obstacle_weights else None,
            device=args.device,
            lane_conf=args.lane_conf,
            obstacle_conf=args.obstacle_conf,
            iou=args.iou,
            use_cache=not args.no_cache,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
