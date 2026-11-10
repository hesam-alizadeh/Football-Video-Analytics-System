# Football Video Analytics

A modular Python computer vision project for analyzing football match footage through object detection, multi-object tracking, team assignment, and top-down pitch visualization.

The project connects frame-level visual predictions with geometric reasoning to make player and ball positions easier to inspect in both the broadcast view and a tactical pitch view.

## Overview

Football footage presents several challenges: players overlap, the ball is small and fast, jersey colors vary with lighting, and camera perspective distorts positions. This project combines detection, tracking, appearance-based grouping, and homography to address these problems within a modular pipeline.

## Core Capabilities

- **Object detection:** YOLO-based detection of relevant objects in match footage.
- **Multi-object tracking:** ByteTrack integration through Supervision to associate detections across frames and maintain track IDs.
- **Team assignment:** Visual grouping of players using jersey appearance and scikit-learn clustering.
- **Pitch projection:** Homography-based transformation of player and ball positions into top-down pitch coordinates.
- **Visualization:** Annotated match footage and top-down pitch representations for inspecting results.
- **Geometry tests:** pytest tests for bounding-box geometry and coordinate projection.

The object categories available in a run depend on the detection model and its class mapping. Tracking IDs represent temporal associations; they do not identify named players.

## Processing Pipeline

1. **Read video:** Load frames and retain the frame rate and image dimensions.
2. **Detect objects:** Extract bounding boxes, class labels, and confidence scores using YOLO.
3. **Track detections:** Associate objects over time with ByteTrack.
4. **Assign teams:** Estimate jersey appearance and group players into team clusters.
5. **Map positions:** Match image-space pitch reference points to pitch-space coordinates and estimate a homography.
6. **Render results:** Draw object annotations and projected positions, then save the output visualization.

For player projection, the bottom-center of the bounding box approximates the point of contact with the ground. Projection quality depends on accurate pitch correspondences and a valid transformation for the current camera view.

## Technology Stack

| Component | Technology | Purpose |
| --- | --- | --- |
| Language | Python | Pipeline implementation |
| Detection | Ultralytics / YOLO | Object detection |
| Tracking | Supervision / ByteTrack | Association across frames |
| Video and geometry | OpenCV | Video I/O, image processing, homography, and rendering |
| Numerical computing | NumPy | Arrays and coordinate operations |
| Team assignment | scikit-learn | Color clustering |
| Testing | pytest | Geometry and projection validation |

## Getting Started

> These setup instructions follow the documented project design. Repository filenames, dependency versions, configuration fields, and the entry point should be checked against the source code before running.

### 1. Create a virtual environment

Run these commands from the repository root:

```bash
python -m venv .venv
```

Activate it on Linux or macOS:

```bash
source .venv/bin/activate
```

Or on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

If the repository includes a dependency file, use it to preserve the intended package versions:

```bash
python -m pip install -r requirements.txt
```

Otherwise, the documented core packages are:

```bash
python -m pip install numpy opencv-python ultralytics supervision scikit-learn pytest
```

Package versions are not pinned here because the source environment has not been verified. In particular, the Ultralytics and Supervision APIs must be compatible with the implementation.

### 3. Prepare inputs and model weights

- Provide a football match video readable by OpenCV.
- Provide detection model weights with the required football object classes.
- Provide the pitch reference points or keypoint model required by the transformation module.
- Set input, output, and model paths in the repository configuration.
- Check confidence thresholds, object-class mappings, and pitch coordinate conventions.

Model weights and example footage are separate assets; this README does not supply them.

### 4. Run the pipeline

For a repository using the assignment's documented entry point:

```bash
python main.py
```

Configure paths before execution. No command-line flags are assumed here.

## Module Organization

The assignment recommends the following separation of responsibilities. Exact paths may differ in the implementation.

| Suggested path | Responsibility |
| --- | --- |
| `main.py` | Coordinate the processing stages |
| `config.py` | Store paths, thresholds, and visualization settings |
| `tracking/tracker.py` | Load detections and maintain tracks |
| `team_assignment/team_assigner.py` | Extract appearance features and assign team labels |
| `transformation/transformer.py` | Estimate homography and project coordinates |
| `visualization/renderer.py` | Draw annotations and tactical pitch views |
| `utils/video_utils.py` | Video I/O and reusable geometry helpers |
| `models/` | Store model weights |
| `data/` | Store local input assets |

## Outputs

The visual outputs support inspection of:

- Detected objects and their tracking identities in the original footage.
- Estimated team membership through consistent colors or labels.
- Player and ball positions in a top-down pitch representation.

Output filenames, encoding, and whether views are saved separately or combined depend on the renderer and configuration. No numerical accuracy or processing-speed benchmark is claimed in this README.

## Testing and Validation

The project includes geometry and projection tests. From the repository root, run:

```bash
python -m pytest
```

Deterministic tests help validate coordinate calculations, but they do not establish detector accuracy or tracking quality. Inspect a short processed clip for ID switches, incorrect team labels, missing ball detections, and implausible projected positions before processing longer footage.

## Limitations

- Occlusion, motion blur, and small objects can cause missed detections and tracking errors.
- Similar jerseys, shadows, and background pixels can reduce team-assignment reliability.
- Camera cuts, zoom, and movement can invalidate a previously estimated homography.
- Inaccurate or insufficient pitch reference points can produce unstable projections.
- A homography models the ground plane; an airborne ball may not project to its true ground position.
- Performance depends on video resolution, model size, hardware, and inference settings.

## Possible Extensions

These are future directions rather than claims about implemented features:

- More robust ball tracking and recovery after missed detections.
- Temporal smoothing of team labels and pitch transformations.
- Camera-cut detection and automatic homography reinitialization.
- Player trajectories, spatial heatmaps, and team-shape analysis.
- Evaluation on labeled clips using detection, tracking, and projection metrics.
- Speed and distance estimation after validating pitch scale and camera calibration.

## Author

**Hesam Alizadeh Zarnagh**  
Computer Science, K. N. Toosi University of Technology  
[GitHub](https://github.com/hesam-alizadeh)

## Academic Context

Developed in the context of the *Applied Neural Network* football video analytics assignment at K. N. Toosi University of Technology, Spring 2026, instructed by Maryam Abdolali.

## License and Third-Party Assets

Consult the repository's `LICENSE` file for reuse terms, if one is provided. Model weights, datasets, and match footage may have separate licenses or usage conditions.
