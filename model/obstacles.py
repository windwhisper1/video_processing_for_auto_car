"""
Train and run YOLO11 detect for indoor obstacle detection (cones, furniture, etc.).

Uses bbox annotations from data/*/obstacle_labels or per-image .boxes.json files.
Each obstacle is a single class ("obstacle") in standard YOLO detection format.

Typical workflow:
  python -m model.obstacles prepare
  python -m model.obstacles train
  python -m model.obstacles val
  python -m model.obstacles predict
  python -m model.obstacles predict --source path/to/images

Saved predictions use green bounding boxes and an "obstacle 0.90" style label.
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

import yaml

# YOLO11 detect: s variant balances speed and accuracy for real-time robot cameras
# on compact objects (cones, chairs) in noisy indoor frames.
DEFAULT_MODEL = "yolo11s.pt"
CLASS_NAME = "obstacle"
CLASS_ID = 0
DISPLAY_LABEL = "obstacle"
BOX_COLOR = (0, 255, 0)
BOX_THICKNESS = 2

DATASET_ROOT = Path("data/yolo_obstacles")
DATA_YAML = DATASET_ROOT / "data.yaml"
WEIGHTS_DIR = Path("weights/obstacles")
RUNS_DIR = Path("runs/obstacles")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CAMERA_IMAGE_RE = re.compile(r"camera_image_(\d+)_", re.IGNORECASE)
FRAME_RE = re.compile(r"frame_(\d+)", re.IGNORECASE)

TRAIN_RAW_LABEL_PAIRS = (
    ("data/train/raw/images", "data/train/obstacle_labels/images"),
    ("data/train/raw/video_frames", "data/train/obstacle_labels/video_frames"),
)
VALID_RAW_LABEL_PAIRS = (
    ("data/valid/raw/images", "data/valid/obstacle_labels/images"),
)
DEFAULT_PREDICT_SOURCE = VALID_RAW_LABEL_PAIRS[0][0]


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


def bbox_to_yolo_line(
    bbox: list[int | float],
    image_width: int,
    image_height: int,
) -> str | None:
    if len(bbox) != 4:
        return None

    x0, y0, x1, y1 = (float(v) for v in bbox)
    x0 = max(0.0, min(float(image_width - 1), x0))
    y0 = max(0.0, min(float(image_height - 1), y0))
    x1 = max(0.0, min(float(image_width - 1), x1))
    y1 = max(0.0, min(float(image_height - 1), y1))
    if x1 <= x0 or y1 <= y0:
        return None

    cx = (x0 + x1) / 2.0 / image_width
    cy = (y0 + y1) / 2.0 / image_height
    w = (x1 - x0) / image_width
    h = (y1 - y0) / image_height
    return f"{CLASS_ID} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def load_label_annotation(json_path: Path) -> dict:
    with open(json_path, encoding="utf-8") as f:
        return json.load(f)


def boxes_json_path_for(image_path: Path) -> Path:
    return image_path.with_suffix(".boxes.json")


def resolve_annotation(
    image_path: Path,
    json_path: Path | None = None,
) -> tuple[dict | None, Path | None]:
    """Load obstacle annotation from label JSON or a sidecar .boxes.json file."""
    if json_path is not None and json_path.is_file():
        return load_label_annotation(json_path), json_path

    boxes_path = boxes_json_path_for(image_path)
    if not boxes_path.is_file():
        return None, None

    import cv2

    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        return None, boxes_path

    height, width = image_bgr.shape[:2]
    with open(boxes_path, encoding="utf-8") as f:
        data = json.load(f)

    raw_boxes = data.get("boxes", data.get("instances", []))
    instances: list[dict] = []
    for item in raw_boxes:
        if isinstance(item, dict):
            bbox = item.get("bbox")
        else:
            bbox = item
        if bbox and len(bbox) == 4:
            instances.append({"bbox": [int(v) for v in bbox]})

    annotation = {
        "image_width": width,
        "image_height": height,
        "instances": instances,
    }
    return annotation, boxes_path


def collect_samples(
    raw_label_pairs: tuple[tuple[str, str], ...],
    include_boxes_json: bool = True,
) -> list[tuple[Path, Path | None]]:
    samples: list[tuple[Path, Path | None]] = []
    for raw_rel, label_rel in raw_label_pairs:
        raw_dir = Path(raw_rel)
        label_dir = Path(label_rel)
        for image_path in list_images(raw_dir):
            json_path = label_dir / f"{image_path.stem}.json"
            if json_path.is_file():
                samples.append((image_path, json_path))
                continue
            if include_boxes_json and boxes_json_path_for(image_path).is_file():
                samples.append((image_path, None))
    return sorted(samples, key=lambda item: image_sort_key(item[0]))


def write_yolo_sample(
    image_path: Path,
    json_path: Path | None,
    images_dir: Path,
    labels_dir: Path,
) -> tuple[int, int]:
    """Copy image and write YOLO-detect label file. Returns (num_instances, num_skipped)."""
    annotation, _ = resolve_annotation(image_path, json_path)
    if annotation is None:
        return 0, 0

    width = int(annotation["image_width"])
    height = int(annotation["image_height"])
    instances = annotation.get("instances", [])

    lines: list[str] = []
    skipped = 0
    for instance in instances:
        bbox = instance.get("bbox")
        if bbox is None and instance.get("polygon"):
            polygon = instance["polygon"]
            xs = [p[0] for p in polygon]
            ys = [p[1] for p in polygon]
            bbox = [min(xs), min(ys), max(xs), max(ys)]

        line = bbox_to_yolo_line(bbox or [], width, height)
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
    """Export obstacle bbox annotations into an Ultralytics YOLO11-detect dataset."""
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
        raise SystemExit(
            "No training samples found.\n"
            "Add obstacle labels under data/train/obstacle_labels/ "
            "(JSON with instances[].bbox) or place .boxes.json next to raw images."
        )

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
        f"  train: {len(train_samples)} images, {train_instances} boxes\n"
        f"  val:   {len(val_samples)} images, {val_instances} boxes\n"
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
        "No trained weights found. Run `python -m model.obstacles train` first "
        "or pass --weights PATH."
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
    """Fine-tune YOLO11-detect on obstacle bbox labels."""
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
        # Indoor obstacles: moderate mosaic, scale for near/far objects, no vertical flip.
        mosaic=1.0,
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
        degrees=5.0,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        flipud=0.0,
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


def extract_obstacle_detections(result) -> list[tuple[tuple[int, int, int, int], float]]:
    """Return obstacle boxes and confidences from one Ultralytics detect result."""
    detections: list[tuple[tuple[int, int, int, int], float]] = []
    if result.boxes is None or len(result.boxes) == 0:
        return detections

    boxes_xyxy = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()
    for box, box_conf in zip(boxes_xyxy, confs):
        x0, y0, x1, y1 = (int(v) for v in box)
        detections.append(((x0, y0, x1, y1), float(box_conf)))
    return detections


def infer_frame(
    model,
    image_bgr: "np.ndarray",
    device: str | int | None = None,
    conf: float = 0.25,
    iou: float = 0.5,
) -> list[tuple[tuple[int, int, int, int], float]]:
    """Run obstacle detection on one BGR frame."""
    results = model.predict(
        source=image_bgr,
        device=resolve_device(device),
        conf=conf,
        iou=iou,
        imgsz=640,
        save=False,
        verbose=False,
    )
    return extract_obstacle_detections(results[0])


def render_prediction_overlay(
    image_bgr: "np.ndarray",
    detections: list[tuple[tuple[int, int, int, int], float]],
    label: str = DISPLAY_LABEL,
) -> "np.ndarray":
    """Draw green boxes and an 'obstacle 0.90' style label per detection."""
    import cv2

    overlay = image_bgr.copy()
    for (x0, y0, x1, y1), conf in detections:
        cv2.rectangle(overlay, (x0, y0), (x1, y1), BOX_COLOR, BOX_THICKNESS)
        text = f"{label} {conf:.2f}"
        text_y = max(y0 - 6, 14)
        cv2.putText(
            overlay,
            text,
            (x0, text_y),
            getattr(cv2, "FONT_HERSHEY_SIMPLEX", 0),
            0.55,
            BOX_COLOR,
            2,
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
    )

    if not save:
        for result in results:
            count = 0 if result.boxes is None else len(result.boxes)
            print(f"[predict] {Path(result.path).name}: {count} obstacle(s)")
        return

    output_dir = Path(project) / name
    output_dir.mkdir(parents=True, exist_ok=True)

    for result in results:
        image_bgr = result.orig_img.copy()
        detections = extract_obstacle_detections(result)
        vis = render_prediction_overlay(image_bgr, detections)
        out_path = output_dir / Path(result.path).name
        cv2.imwrite(str(out_path), vis)
        print(
            f"[ok] {Path(result.path).name} -> {len(detections)} obstacle(s) -> {out_path}"
        )

    print(f"Saved {len(results)} visualization(s) to {output_dir}")


def export(weights: str | None = None, fmt: str = "onnx") -> None:
    ensure_opencv_headless()
    yolo = load_yolo_model(resolve_weights(weights))
    yolo.export(format=fmt, imgsz=640, simplify=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="YOLO11-detect obstacle detection (train / val / predict)."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser(
        "prepare",
        help="Convert obstacle bbox JSON into a YOLO11-detect dataset.",
    )
    prepare_parser.add_argument(
        "--force",
        action="store_true",
        help="Delete and rebuild data/yolo_obstacles.",
    )

    train_parser = subparsers.add_parser("train", help="Train YOLO11-detect.")
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
    predict_parser.add_argument(
        "--no-save",
        action="store_true",
        help="Run inference without writing images.",
    )

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
