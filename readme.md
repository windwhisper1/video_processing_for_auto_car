# Perception Pipeline

End-to-end video pipeline for indoor robot perception: lane-dash segmentation, obstacle detection, and QR code decoding. Annotated frames and an output MP4 are produced automatically.

## 1. Environment configuration

### Requirements

- **Python** 3.12
- **GPU** optional — CUDA is used when available; CPU fallback is supported
- Trained weights under `weights/` (see [Project layout](#project-layout))

### Install dependencies

From the project root (`pipeline/`):

```bash
conda create -n pipeline python=3.12

pip install -r requirements.txt
```

### Headless / cloud servers

Use the headless OpenCV build only. Do **not** install both `opencv-python` and `opencv-python-headless` at the same time.

If `cv2` import fails on a headless Linux machine:

```bash
pip uninstall -y opencv-python
pip install "opencv-python-headless>=4.8.0,<5" --force-reinstall
```

---



## 2. Acceptable data



### For inference (`python pipeline.py`)


| Type            | Formats                                 | Description                                                                               |
| --------------- | --------------------------------------- | ----------------------------------------------------------------------------------------- |
| **Input video** | `.mp4`, `.avi`, `.mov`, `.mkv`, `.webm` | Robot-camera footage showing lane dashes, obstacles (e.g. cones), and optionally QR codes |


Videos should be shot from a low, forward-facing camera in an indoor environment similar to the training data.

---



## 3. Where data should be stored

```
pipeline/
├── data/
│   └── example.mp4     # input video(s) — place .mp4 files here
├── weights/
│   ├── lane_lines/best.pt            # required for inference
│   └── obstacles/best.pt             # required for inference
└── .cache/frames/                    # auto-created frame cache (do not edit)
```



### Rules

- **Inference videos** go directly inside `data/` (not in subfolders). The pipeline scans only the top level of `data/`.
- **One or more videos** are supported; each file is processed independently.
- **Weights** must exist before running the pipeline. The default paths are `weights/lane_lines/best.pt` and `weights/obstacles/best.pt`.
- **Frame cache** is written to `.cache/frames/` to avoid re-extracting frames on repeated runs.

---



## 4. How to generate output



### Basic usage

Process every video in `data/`:

```bash
python pipeline.py
```

Process a single video:

```bash
python pipeline.py --video data/example.mp4
```



### Useful options


| Option                    | Default                      | Description                                                            |
| ------------------------- | ---------------------------- | ---------------------------------------------------------------------- |
| `--frame-interval N`      | `1`                          | Process one frame every N source frames (use `10` for faster previews) |
| `--output-dir PATH`       | `output`                     | Where annotated frames and videos are saved                            |
| `--device cuda:0`         | auto                         | Force inference device                                                 |
| `--lane-conf`             | `0.25`                       | Lane segmentation confidence threshold                                 |
| `--obstacle-conf`         | `0.25`                       | Obstacle detection confidence threshold                                |
| `--lane-weights PATH`     | `weights/lane_lines/best.pt` | Custom lane model weights                                              |
| `--obstacle-weights PATH` | `weights/obstacles/best.pt`  | Custom obstacle model weights                                          |
| `--no-cache`              | off                          | Re-extract frames without using the cache                              |




### Example (faster preview)

```bash
python pipeline.py --frame-interval 10
```



### Pipeline steps

1. Extract frames from video(s) (cached under `.cache/frames/`)
2. Segment lane dashes — green polygon outlines, no boxes
3. Detect obstacles — green bounding boxes with confidence scores
4. Detect QR codes — green outline + decoded text (HUD shows `no code` if none found)
5. Combine all overlays and status text (`detected lanes: N`)
6. Assemble annotated frames into an output MP4

---



## 5. Where to find the output

After a successful run:


| Output               | Path                                                       |
| -------------------- | ---------------------------------------------------------- |
| **Annotated video**  | `output/<video_name>_annotated.mp4`                        |
| **Annotated frames** | `output/annotated_frames/<video_name>/frame_000000.jpg`, … |


Example for `data/example.mp4`:

```
output/
├── example_annotated.mp4
└── annotated_frames/
    └── example/
        ├── frame_000000.jpg
        ├── frame_000001.jpg
        └── ...
```

Each annotated frame shows:

- Green polygon outlines around lane dashes
- Green bounding boxes labeled `obstacle 0.XX`
- Green QR code outline and decoded text (when present)
- Top-left HUD: `detected lanes: N` and QR status (`no code` or decoded value)

---



## Project layout

```
pipeline/
├── pipeline.py              # main entry point
├── video_to_frames.py       # frame extraction + cache
├── frames_to_video.py       # assemble frames into MP4
├── detect_QR_code.py        # QR detection and overlay
├── model/
│   ├── lane_lines.py        # YOLO11-seg lane-dash model
│   └── obstacles.py         # YOLO11-detect obstacle model
├── data/                    # input videos (+ optional training data)
├── weights/                 
│   ├── lane_lines/
│   │   └──best.pt           # YOLO11-seg trained model checkpoints
│   └── obstacles/
│       └──best.pt           # YOLO11-detect trained model checkpoints
├── output/                  # generated videos and frames
├── .cache/frames/           # extracted frame cache
└── requirements.txt
```

