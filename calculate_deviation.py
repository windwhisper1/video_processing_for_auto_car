"""
Lane-center deviation from YOLO-seg dash polygons (``model.lane_lines``).

  1. Mask centroids
  2. Quadratic fit ``x = a y^2 + b y + c``
  3. Reference point ``M = (x_ref, y_ref)`` with ``y_ref = H``
  4. Absolute lateral deviation ``|x_ref - W / 2|`` plus left/right side
  5. Draw ``M`` and HUD text (below other top-left status lines)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import cv2
import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

# Match pipeline HUD layout: first line at y=24, then +28 per line.
HUD_COLOR = (0, 255, 0)
HUD_FONT = getattr(cv2, "FONT_HERSHEY_SIMPLEX", 0)
HUD_LINE_TYPE = getattr(cv2, "LINE_AA", 16)
HUD_TEXT_X = 12
HUD_FONT_SCALE = 0.65
HUD_THICKNESS = 2
HUD_LINE_HEIGHT = 28

MIDPOINT_COLOR = (0, 255, 0)
MIDPOINT_RADIUS = 8
MIDPOINT_CROSS = 12
# Keep the full M marker on-screen when y_ref is at the bottom edge (y = H).
MIDPOINT_DRAW_MARGIN = MIDPOINT_RADIUS + 2


def _visible_draw_xy(
    x_ref: float,
    y_ref: float,
    width: int,
    height: int,
) -> tuple[int, int]:
    """Clamp draw coords so the filled M circle stays fully inside the frame."""
    x = int(np.clip(round(x_ref), MIDPOINT_DRAW_MARGIN, max(MIDPOINT_DRAW_MARGIN, width - 1 - MIDPOINT_DRAW_MARGIN)))
    y = int(np.clip(round(y_ref), MIDPOINT_DRAW_MARGIN, max(MIDPOINT_DRAW_MARGIN, height - 1 - MIDPOINT_DRAW_MARGIN)))
    return x, y


SideLabel = Literal["left", "right", "center"]


@dataclass(frozen=True)
class DeviationResult:
    """Lane midpoint and lateral offset for one frame."""

    centers: "NDArray[np.floating]"
    coeffs: "NDArray[np.floating] | None"
    x_ref: float
    y_ref: float
    deviation: float
    """Absolute pixel offset ``|x_ref - W/2|``."""
    signed_offset: float
    """``x_ref - W/2``: negative → ``right``, positive → ``left`` (car vs lane)."""
    side: SideLabel
    image_width: int
    image_height: int


def side_from_offset(signed_offset: float, *, eps: float = 0.5) -> SideLabel:
    """
    Map signed offset to a left/right/center label.

    ``signed_offset = x_ref - W/2``. Positive means M is right of image center,
    so the car (at center) is left of the lane → label ``left`` (and vice versa).
    """
    if signed_offset > eps:
        return "left"
    if signed_offset < -eps:
        return "right"
    return "center"


def mask_centers(polygons: list) -> "NDArray[np.floating]":
    """Return (N, 2) array of centroid ``(x, y)`` for each valid polygon mask."""
    centers: list[list[float]] = []
    for polygon in polygons:
        if polygon is None:
            continue
        pts = np.asarray(polygon, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[0] < 1 or pts.shape[1] < 2:
            continue
        centers.append([float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1]))])
    if not centers:
        return np.zeros((0, 2), dtype=np.float64)
    return np.asarray(centers, dtype=np.float64)


def fit_quadratic_x_of_y(centers: "NDArray[np.floating]") -> "NDArray[np.floating] | None":
    """
    Fit ``x = a y^2 + b y + c`` through mask centers.

    Falls back to linear (1–2 points) or constant (1 point). Returns ``None``
    if there are no centers.
    """
    if centers is None or len(centers) == 0:
        return None

    ys = centers[:, 1]
    xs = centers[:, 0]
    n = len(centers)

    if n == 1:
        # x = 0*y^2 + 0*y + c
        return np.array([0.0, 0.0, float(xs[0])], dtype=np.float64)

    if n == 2:
        # x = 0*y^2 + b*y + c
        b, c = np.polyfit(ys, xs, deg=1)
        return np.array([0.0, float(b), float(c)], dtype=np.float64)

    a, b, c = np.polyfit(ys, xs, deg=2)
    return np.array([float(a), float(b), float(c)], dtype=np.float64)


def eval_x_of_y(coeffs: "NDArray[np.floating]", y: float) -> float:
    """Evaluate ``x = a y^2 + b y + c``."""
    a, b, c = coeffs
    return float(a * y * y + b * y + c)


def compute_deviation(
    polygons: list,
    image_shape: tuple[int, ...],
) -> DeviationResult | None:
    """
    Compute midpoint ``M=(x_ref, y_ref)`` and ``deviation = |x_ref - W/2|``.

    ``y_ref = H``. Returns ``None`` when no lane masks are available.
    """
    height, width = int(image_shape[0]), int(image_shape[1])
    centers = mask_centers(polygons)
    coeffs = fit_quadratic_x_of_y(centers)
    if coeffs is None:
        return None

    y_ref = float(height)
    x_ref = eval_x_of_y(coeffs, y_ref)
    signed_offset = x_ref - width / 2.0
    deviation = abs(signed_offset)
    side = side_from_offset(signed_offset)

    return DeviationResult(
        centers=centers,
        coeffs=coeffs,
        x_ref=x_ref,
        y_ref=y_ref,
        deviation=deviation,
        signed_offset=signed_offset,
        side=side,
        image_width=width,
        image_height=height,
    )


def draw_midpoint(image_bgr, x_ref: float, y_ref: float) -> None:
    """Draw point ``M`` as a filled circle with a small crosshair (always fully visible)."""
    height, width = image_bgr.shape[:2]
    x, y = _visible_draw_xy(x_ref, y_ref, width, height)

    cv2.circle(image_bgr, (x, y), MIDPOINT_RADIUS, MIDPOINT_COLOR, thickness=-1)
    cv2.circle(image_bgr, (x, y), MIDPOINT_RADIUS + 2, MIDPOINT_COLOR, thickness=2)
    cv2.line(
        image_bgr,
        (x - MIDPOINT_CROSS, y),
        (x + MIDPOINT_CROSS, y),
        MIDPOINT_COLOR,
        2,
        HUD_LINE_TYPE,
    )
    cv2.line(
        image_bgr,
        (x, y - MIDPOINT_CROSS),
        (x, y + MIDPOINT_CROSS),
        MIDPOINT_COLOR,
        2,
        HUD_LINE_TYPE,
    )
    # Place label above the marker when near the bottom edge (y_ref = H).
    near_bottom = y >= height - 1 - MIDPOINT_DRAW_MARGIN - 2
    label_y = y - MIDPOINT_RADIUS - 2 if near_bottom else y + MIDPOINT_RADIUS + 16
    label_y = int(np.clip(label_y, 8, height - 1))
    cv2.putText(
        image_bgr,
        "M",
        (x + MIDPOINT_RADIUS + 4, label_y),
        HUD_FONT,
        0.6,
        MIDPOINT_COLOR,
        2,
        HUD_LINE_TYPE,
    )


def draw_deviation_text(
    image_bgr,
    deviation: float,
    side: SideLabel,
    text_y: int,
) -> int:
    """
    Draw deviation with left/right side on the top-left HUD at ``text_y``.

    Returns the next free baseline ``y`` so callers can stack more lines.
    """
    text = f"deviation: {deviation:.1f} px {side}"
    cv2.putText(
        image_bgr,
        text,
        (HUD_TEXT_X, text_y),
        HUD_FONT,
        HUD_FONT_SCALE,
        HUD_COLOR,
        HUD_THICKNESS,
        HUD_LINE_TYPE,
    )
    return text_y + HUD_LINE_HEIGHT


def annotate_deviation(
    image_bgr,
    polygons: list,
    *,
    text_y: int,
) -> int:
    """
    Fit lane center, draw midpoint ``M``, and show deviation at ``text_y``.

    Does nothing (returns ``text_y`` unchanged) when no masks are present.
    Returns the next free HUD baseline after drawing.
    """
    result = compute_deviation(polygons, image_bgr.shape)
    if result is None:
        return text_y

    next_y = draw_deviation_text(image_bgr, result.deviation, result.side, text_y)
    # Draw M after HUD text so the marker stays visible on top.
    draw_midpoint(image_bgr, result.x_ref, result.y_ref)
    return next_y
