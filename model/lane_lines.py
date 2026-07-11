"""
Train and run YOLO11-seg for lane-dash instance segmentation.

Uses SAM pseudo-labels from data/*/lane_line_labels as training targets.
Each lane dash is a single class ("lane_dash") with polygon masks converted to
YOLO segmentation format.

Typical workflow:
  python -m model.lane_lines prepare
  python -m model.lane_lines train
  python -m model.lane_lines val
  python -m model.lane_lines predict
  python -m model.lane_lines predict --source path/to/images

Saved predictions use green mask contours and a "lane line" label (no boxes).
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


def _opencv_package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for pkg in ("opencv-python", "opencv-python-headless", "opencv-contrib-python"):
        try:
            versions[pkg] = version(pkg)
        except PackageNotFoundError:
            versions[pkg] = "NOT_INSTALLED"
    return versions


def _cv2_paths_on_disk() -> list[str]:
    import site
    import sysconfig

    roots = list(dict.fromkeys(site.getsitepackages() + [sysconfig.get_path("purelib")]))
    paths: list[str] = []
    for root in roots:
        cv2_dir = Path(root) / "cv2"
        if cv2_dir.is_dir():
            paths.append(str(cv2_dir))
    return paths


def ensure_opencv_headless() -> None:
    """Fail fast on headless servers when OpenCV cannot be imported."""
    packages = _opencv_package_versions()
    cv2_paths = _cv2_paths_on_disk()

    if (
        packages.get("opencv-python") != "NOT_INSTALLED"
        and packages.get("opencv-python-headless") != "NOT_INSTALLED"
    ):
        raise SystemExit(
            "Both opencv-python and opencv-python-headless are installed. "
            "The GUI build requires libGL.so.1 on headless servers. Fix with:\n"
            "  pip uninstall -y opencv-python\n"
            '  pip install "opencv-python-headless>=4.8.0,<5" --force-reinstall'
        )

    try:
        import cv2  # noqa: F401
    except (ImportError, OSError, ModuleNotFoundError) as exc:
        error_text = str(exc)
        headless_only = (
            packages.get("opencv-python") == "NOT_INSTALLED"
            and packages.get("opencv-python-headless") != "NOT_INSTALLED"
        )
        missing_cv2_module = "No module named 'cv2'" in error_text

        if headless_only and missing_cv2_module and not cv2_paths:
            raise SystemExit(
                "opencv-python-headless is listed in pip, but the cv2 module files "
                "are missing. This commonly happens after uninstalling opencv-python "
                "without reinstalling headless. Fix with:\n"
                '  pip install "opencv-python-headless>=4.8.0,<5" --force-reinstall'
            ) from exc

        if headless_only and "libGL" in error_text:
            raise SystemExit(
                "opencv-python-headless is installed but its native cv2 wheel still "
                f"requires libGL.so.1 ({error_text}). Fix with:\n"
                '  pip install "opencv-python-headless>=4.8.0,<5" --force-reinstall'
            ) from exc

        raise SystemExit(
            "Failed to import cv2.\n"
            f"  error: {error_text}\n"
            "On headless servers, reinstall OpenCV headless 4.x with:\n"
            '  pip install "opencv-python-headless>=4.8.0,<5" --force-reinstall'
        ) from exc

    import cv2

    if not callable(getattr(cv2, "imread", None)):
        raise SystemExit(
            "cv2 imported but OpenCV bindings look broken (missing imread).\n"
            "Check for a local file or folder named cv2.py / cv2/ shadowing OpenCV, "
            "then reinstall with:\n"
            '  pip install "opencv-python-headless>=4.8.0,<5" --force-reinstall'
        )


import yaml

# YOLO11-seg: latest Ultralytics seg head; s variant balances speed and accuracy
# on small, thin lane dashes with a modest dataset.
DEFAULT_MODEL = "yolo11s-seg.pt"
CLASS_NAME = "lane_dash"
CLASS_ID = 0
DISPLAY_LABEL = "lane line"
CONTOUR_COLOR = (0, 255, 0)
CONTOUR_THICKNESS = 2

DATASET_ROOT = Path("data/yolo_lane_lines")
DATA_YAML = DATASET_ROOT / "data.yaml"
WEIGHTS_DIR = Path("weights/lane_lines")
RUNS_DIR = Path("runs/lane_lines")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CAMERA_IMAGE_RE = re.compile(r"camera_image_(\d+)_", re.IGNORECASE)
FRAME_RE = re.compile(r"frame_(\d+)", re.IGNORECASE)

TRAIN_RAW_LABEL_PAIRS = (
    ("data/train/raw/images", "data/train/lane_line_labels/images"),
    ("data/train/raw/video_frames", "data/train/lane_line_labels/video_frames"),
)
VALID_RAW_LABEL_PAIRS = (
    ("data/valid/raw/images", "data/valid/lane_line_labels/images"),
)
DEFAULT_PREDICT_SOURCE = VALID_RAW_LABEL_PAIRS[0][0]


def image_sort_key(path: Path) -> tuple:
    name = path.name
    camera_match = CAMERA_IMAGE_RE.search(name)
    if camera_match:
        return (0, int(camera_match.group(1)), name)
    frame_match = FRAME_RE.search(name)
    if frame_match:
        return (1, int(frame_match.group(1)), name)
    return (2, 0, name)


def list_images(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    files = [
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return sorted(files, key=image_sort_key)


def polygon_to_yolo_line(
    polygon: list[list[int]],
    image_width: int,
    image_height: int,
) -> str | None:
    if len(polygon) < 3:
        return None

    coords: list[str] = []
    for x, y in polygon:
        nx = max(0.0, min(1.0, x / image_width))
        ny = max(0.0, min(1.0, y / image_height))
        coords.extend([f"{nx:.6f}", f"{ny:.6f}"])

    return f"{CLASS_ID} " + " ".join(coords)


def load_label_annotation(json_path: Path) -> dict:
    with open(json_path, encoding="utf-8") as f:
        return json.load(f)


def collect_samples(
    raw_label_pairs: tuple[tuple[str, str], ...],
) -> list[tuple[Path, Path]]:
    samples: list[tuple[Path, Path]] = []
    for raw_rel, label_rel in raw_label_pairs:
        raw_dir = Path(raw_rel)
        label_dir = Path(label_rel)
        for image_path in list_images(raw_dir):
            json_path = label_dir / f"{image_path.stem}.json"
            if json_path.is_file():
                samples.append((image_path, json_path))
    return sorted(samples, key=lambda item: image_sort_key(item[0]))


def write_yolo_sample(
    image_path: Path,
    json_path: Path,
    images_dir: Path,
    labels_dir: Path,
) -> tuple[int, int]:
    """Copy image and write YOLO-seg label file. Returns (num_instances, num_skipped)."""
    annotation = load_label_annotation(json_path)
    width = int(annotation["image_width"])
    height = int(annotation["image_height"])
    instances = annotation.get("instances", [])

    lines: list[str] = []
    skipped = 0
    for instance in instances:
        polygon = instance.get("polygon", [])
        line = polygon_to_yolo_line(polygon, width, height)
        if line is None:
            skipped += 1
            continue
        lines.append(line)

    dest_image = images_dir / image_path.name
    dest_label = labels_dir / f"{image_path.stem}.txt"

    shutil.copy2(image_path, dest_image)
    dest_label.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return len(lines), skipped


def prepare_dataset(force: bool = False) -> Path:
    """Export pseudo-labels into an Ultralytics YOLO11-seg dataset."""
    if force and DATASET_ROOT.exists():
        shutil.rmtree(DATASET_ROOT)

    train_images = DATASET_ROOT / "images" / "train"
    train_labels = DATASET_ROOT / "labels" / "train"
    val_images = DATASET_ROOT / "images" / "val"
    val_labels = DATASET_ROOT / "labels" / "val"
    for folder in (train_images, train_labels, val_images, val_labels):
        folder.mkdir(parents=True, exist_ok=True)

    train_samples = collect_samples(TRAIN_RAW_LABEL_PAIRS)
    val_samples = collect_samples(VALID_RAW_LABEL_PAIRS)
    if not train_samples:
        raise SystemExit("No training samples found. Run generate_pseudo_labels.lane_lines first.")

    train_instances = 0
    val_instances = 0
    for image_path, json_path in train_samples:
        count, _ = write_yolo_sample(image_path, json_path, train_images, train_labels)
        train_instances += count
    for image_path, json_path in val_samples:
        count, _ = write_yolo_sample(image_path, json_path, val_images, val_labels)
        val_instances += count

    data = {
        "path": str(DATASET_ROOT.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {CLASS_ID: CLASS_NAME},
    }
    with open(DATA_YAML, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)

    print(
        f"Prepared YOLO dataset at {DATASET_ROOT}\n"
        f"  train: {len(train_samples)} images, {train_instances} instances\n"
        f"  val:   {len(val_samples)} images, {val_instances} instances\n"
        f"  config: {DATA_YAML}"
    )
    return DATA_YAML


def resolve_device(device: str | None) -> str | int:
    if device:
        return device
    try:
        import torch

        return 0 if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def resolve_weights(weights: str | None) -> Path:
    if weights:
        path = Path(weights)
        if not path.is_file():
            raise SystemExit(f"Weights not found: {path}")
        return path

    candidates = [
        WEIGHTS_DIR / "best.pt",
        RUNS_DIR / "train" / "weights" / "best.pt",
    ]
    for path in candidates:
        if path.is_file():
            return path

    raise SystemExit(
        "No trained weights found. Run `python -m model.lane_lines train` first "
        f"or pass --weights PATH."
    )


def load_yolo_model(weights: str | Path):
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "ultralytics is required. Install with: pip install ultralytics"
        ) from exc
    return YOLO(str(weights))


def train(
    model_name: str = DEFAULT_MODEL,
    epochs: int = 150,
    imgsz: int = 640,
    batch: int = 16,
    device: str | None = None,
    project: str | Path = RUNS_DIR,
    name: str = "train",
    resume: bool = False,
) -> Path:
    """Fine-tune YOLO11-seg on pseudo lane-dash labels."""
    ensure_opencv_headless()
    if not DATA_YAML.is_file():
        prepare_dataset()

    yolo = load_yolo_model(model_name)
    resolved_device = resolve_device(device)

    results = yolo.train(
        data=str(DATA_YAML),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=resolved_device,
        project=str(project),
        name=name,
        exist_ok=True,
        resume=resume,
        # Small thin objects: keep mosaic/copy-paste moderate, train longer.
        mosaic=1.0,
        copy_paste=0.15,
        close_mosaic=15,
        patience=40,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        weight_decay=0.0005,
        warmup_epochs=3,
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.4,
        degrees=2.0,
        translate=0.08,
        scale=0.35,
        fliplr=0.0,
        flipud=0.0,
        # Segmentation quality matters more than box mAP for lane dashes.
        overlap_mask=True,
        mask_ratio=2,
        single_cls=True,
        plots=True,
        save=True,
    )

    best_weights = Path(results.save_dir) / "weights" / "best.pt"
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    if best_weights.is_file():
        shutil.copy2(best_weights, WEIGHTS_DIR / "best.pt")
        print(f"Copied best weights -> {WEIGHTS_DIR / 'best.pt'}")

    print(f"Training finished. Best weights: {best_weights}")
    return best_weights


def validate(weights: str | None = None, device: str | None = None) -> None:
    ensure_opencv_headless()
    if not DATA_YAML.is_file():
        prepare_dataset()

    yolo = load_yolo_model(resolve_weights(weights))
    yolo.val(
        data=str(DATA_YAML),
        device=resolve_device(device),
        split="val",
        imgsz=640,
        plots=True,
    )


def extract_lane_polygons(result) -> list["np.ndarray"]:
    """Return polygon contours from a single Ultralytics segmentation result."""
    if result.masks is None:
        return []
    return list(result.masks.xy)


def infer_frame(
    model,
    image_bgr: "np.ndarray",
    device: str | int | None = None,
    conf: float = 0.25,
    iou: float = 0.5,
) -> list["np.ndarray"]:
    """Run lane-dash segmentation on one BGR frame."""
    results = model.predict(
        source=image_bgr,
        device=resolve_device(device),
        conf=conf,
        iou=iou,
        imgsz=640,
        save=False,
        verbose=False,
        retina_masks=True,
    )
    return extract_lane_polygons(results[0])


def render_prediction_overlay(
    image_bgr: "np.ndarray",
    polygons: list["np.ndarray"],
    label: str = DISPLAY_LABEL,
    show_labels: bool = True,
) -> "np.ndarray":
    """Draw mask edges and optional lane-line labels per instance (no boxes)."""
    import cv2
    import numpy as np

    overlay = image_bgr.copy()
    for polygon in polygons:
        if polygon is None or len(polygon) < 3:
            continue

        pts = np.asarray(polygon, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(overlay, [pts], isClosed=True, color=CONTOUR_COLOR, thickness=CONTOUR_THICKNESS)

        if show_labels:
            cx = int(np.mean(polygon[:, 0]))
            cy = int(np.mean(polygon[:, 1]))
            height, width = overlay.shape[:2]
            cx = int(np.clip(cx, 0, width - 1))
            cy = int(np.clip(cy, 8, height - 1))
            cv2.putText(
                overlay,
                label,
                (cx, cy),
                getattr(cv2, "FONT_HERSHEY_SIMPLEX", 0),
                0.5,
                CONTOUR_COLOR,
                1,
                getattr(cv2, "LINE_AA", 16),
            )

    return overlay


def predict(
    source: str | Path | None = None,
    weights: str | None = None,
    device: str | None = None,
    conf: float = 0.25,
    iou: float = 0.5,
    save: bool = True,
    project: str | Path = RUNS_DIR,
    name: str = "predict",
) -> None:
    ensure_opencv_headless()
    import cv2

    resolved_source = Path(source) if source else Path(DEFAULT_PREDICT_SOURCE)
    if not resolved_source.exists():
        raise SystemExit(f"Source not found: {resolved_source}")

    yolo = load_yolo_model(resolve_weights(weights))
    print(f"Predicting on {resolved_source}")
    results = yolo.predict(
        source=str(resolved_source),
        device=resolve_device(device),
        conf=conf,
        iou=iou,
        imgsz=640,
        save=False,
        project=str(project),
        name=name,
        exist_ok=True,
        retina_masks=True,
    )

    if not save:
        for result in results:
            count = 0 if result.masks is None else len(result.masks)
            print(f"[predict] {Path(result.path).name}: {count} lane line(s)")
        return

    output_dir = Path(project) / name
    output_dir.mkdir(parents=True, exist_ok=True)

    for result in results:
        image_bgr = result.orig_img.copy()
        polygons = extract_lane_polygons(result)
        vis = render_prediction_overlay(image_bgr, polygons)
        out_path = output_dir / Path(result.path).name
        cv2.imwrite(str(out_path), vis)
        print(f"[ok] {Path(result.path).name} -> {len(polygons)} lane line(s) -> {out_path}")

    print(f"Saved {len(results)} visualization(s) to {output_dir}")


def export(weights: str | None = None, fmt: str = "onnx") -> None:
    ensure_opencv_headless()
    yolo = load_yolo_model(resolve_weights(weights))
    yolo.export(format=fmt, imgsz=640, simplify=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="YOLO11-seg lane-dash detection (train / val / predict)."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser(
        "prepare",
        help="Convert pseudo-label JSON polygons into a YOLO11-seg dataset.",
    )
    prepare_parser.add_argument(
        "--force",
        action="store_true",
        help="Delete and rebuild data/yolo_lane_lines.",
    )

    train_parser = subparsers.add_parser("train", help="Train YOLO11-seg.")
    train_parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Base checkpoint (default: {DEFAULT_MODEL}).",
    )
    train_parser.add_argument("--epochs", type=int, default=150)
    train_parser.add_argument("--imgsz", type=int, default=640)
    train_parser.add_argument("--batch", type=int, default=16)
    train_parser.add_argument("--device", type=str, default=None)
    train_parser.add_argument("--resume", action="store_true")

    val_parser = subparsers.add_parser("val", help="Validate trained weights.")
    val_parser.add_argument("--weights", type=str, default=None)
    val_parser.add_argument("--device", type=str, default=None)

    predict_parser = subparsers.add_parser("predict", help="Run inference.")
    predict_parser.add_argument(
        "--source",
        type=str,
        default=None,
        help=f"Image path, folder, or glob (default: {DEFAULT_PREDICT_SOURCE}).",
    )
    predict_parser.add_argument("--weights", type=str, default=None)
    predict_parser.add_argument("--device", type=str, default=None)
    predict_parser.add_argument("--conf", type=float, default=0.25)
    predict_parser.add_argument("--iou", type=float, default=0.5)
    predict_parser.add_argument("--no-save", action="store_true", help="Run inference without writing images.")

    export_parser = subparsers.add_parser("export", help="Export trained weights.")
    export_parser.add_argument("--weights", type=str, default=None)
    export_parser.add_argument(
        "--format",
        type=str,
        default="onnx",
        help="Export format supported by Ultralytics (onnx, torchscript, ...).",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.command == "prepare":
        prepare_dataset(force=args.force)
    elif args.command == "train":
        train(
            model_name=args.model,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            resume=args.resume,
        )
    elif args.command == "val":
        validate(weights=args.weights, device=args.device)
    elif args.command == "predict":
        predict(
            source=args.source,
            weights=args.weights,
            device=args.device,
            conf=args.conf,
            iou=args.iou,
            save=not args.no_save,
        )
    elif args.command == "export":
        export(weights=args.weights, fmt=args.format)
    else:
        raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
