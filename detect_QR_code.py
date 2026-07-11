"""
Detect and decode QR codes in an image (OpenCV + zxing-cpp fallback).

Usage:
  python detect_QR_code.py path/to/image.jpg
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

import cv2
import zxingcpp

QR_COLOR = (0, 255, 0)
QR_THICKNESS = 2
DISPLAY_PREFIX = "code"
# Small QR codes in robot-camera frames need upscaling for reliable decoding.
ZXING_UPSCALE_FACTORS = (2.0, 2.5)


def _normalize_corners(corners) -> list[tuple[int, int]]:
    """Convert OpenCV corner output to integer (x, y) points."""
    import numpy as np

    if corners is None:
        return []

    points = np.asarray(corners, dtype=np.float32).reshape(-1, 2)
    return [(int(x), int(y)) for x, y in points]


def _detect_with_opencv(image) -> list[dict]:
    """Try OpenCV's built-in QR detector (fast when it succeeds)."""
    detector = cv2.QRCodeDetector()
    results: list[dict] = []

    found, decoded_list, points, _ = detector.detectAndDecodeMulti(image)
    if found and decoded_list is not None and points is not None:
        for data, corners in zip(decoded_list, points, strict=False):
            if not data:
                continue
            results.append(
                {
                    "data": data,
                    "corners": _normalize_corners(corners),
                }
            )

    if not results:
        data, corners, _ = detector.detectAndDecode(image)
        if data:
            results.append(
                {
                    "data": data,
                    "corners": _normalize_corners(corners),
                }
            )

    return results


def _corners_from_zxing_position(position, scale: float) -> list[tuple[int, int]]:
    """Map zxing-cpp corner points back to the original image scale."""
    corners: list[tuple[int, int]] = []
    for corner_name in ("top_left", "top_right", "bottom_right", "bottom_left"):
        point = getattr(position, corner_name)
        corners.append((int(point.x / scale), int(point.y / scale)))
    return corners


def _detect_with_zxing(image) -> list[dict]:
    """Fallback decoder for small or perspective-distorted QR codes."""
    results: list[dict] = []
    seen: set[str] = set()

    for scale in ZXING_UPSCALE_FACTORS:
        if scale == 1.0:
            scaled = image
        else:
            scaled = cv2.resize(
                image,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_CUBIC,
            )

        for barcode in zxingcpp.read_barcodes(scaled):
            data = barcode.text
            if not data or data in seen:
                continue
            seen.add(data)
            results.append(
                {
                    "data": data,
                    "corners": _corners_from_zxing_position(barcode.position, scale),
                }
            )

        if results:
            break

    return results


def detect_qr_codes_from_image(image) -> list[dict]:
    """Detect and decode QR codes in a BGR image array."""
    results = _detect_with_opencv(image)
    if not results:
        results = _detect_with_zxing(image)
    return results


def detect_qr_codes(image_path: Path) -> list[dict]:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    return detect_qr_codes_from_image(image)


def qr_status_text(results: list[dict]) -> str:
    """Return a short HUD string for QR detection results."""
    if not results:
        return "no code"
    return results[0]["data"]


def render_prediction_overlay(
    image_bgr: "np.ndarray",
    qr_results: list[dict],
    label_prefix: str = DISPLAY_PREFIX,
) -> "np.ndarray":
    """Draw green QR outlines and decoded text for each detection."""
    import numpy as np

    overlay = image_bgr.copy()
    font = getattr(cv2, "FONT_HERSHEY_SIMPLEX", 0)
    line_type = getattr(cv2, "LINE_AA", 16)

    for result in qr_results:
        corners = result.get("corners") or []
        if len(corners) < 4:
            continue

        pts = np.asarray(corners, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(overlay, [pts], isClosed=True, color=QR_COLOR, thickness=QR_THICKNESS)

        data = result.get("data", "")
        if not data:
            continue

        xs = [point[0] for point in corners]
        ys = [point[1] for point in corners]
        text_x = min(xs)
        text_y = max(min(ys) - 6, 14)
        text = f"{label_prefix}: {data}"
        cv2.putText(
            overlay,
            text,
            (text_x, text_y),
            font,
            0.55,
            QR_COLOR,
            2,
            line_type,
        )

    return overlay


def print_results(image_path: Path, results: list[dict]) -> None:
    print(f"Image: {image_path}")
    if not results:
        print("No QR code detected.")
        return

    print(f"Found {len(results)} QR code(s):")
    for index, result in enumerate(results, start=1):
        print(f"  [{index}] Data: {result['data']}")
        if result["corners"]:
            print(f"       Corners: {result['corners']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect and decode QR codes in an image."
    )
    parser.add_argument("image", type=Path, help="Path to the input image.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_path = args.image.resolve()

    if not image_path.is_file():
        print(f"Error: file not found: {image_path}", file=sys.stderr)
        sys.exit(1)

    try:
        results = detect_qr_codes(image_path)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    print_results(image_path, results)
    sys.exit(0 if results else 1)


if __name__ == "__main__":
    main()
