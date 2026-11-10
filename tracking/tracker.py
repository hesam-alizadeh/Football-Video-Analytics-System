from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv

import config



@dataclass
class TrackResult:
    frame_index: int
    detections: sv.Detections


class Tracker:
    """
    Football detector + multi-object tracker.

    Uses ONLY the trained Players model (best.pt) for detection,
    then applies ByteTrack to get stable track IDs.
    """

    def __init__(
        self,
        model_path: Path | str = "models/players/best.pt",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        track_buffer: int = 30,
        match_threshold: float = 0.8,
    ):
        self.model_path = Path(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        if not self.model_path.exists():
            raise FileNotFoundError(f"YOLO model not found: {self.model_path}")

        self.model = YOLO(str(self.model_path))

        self.tracker = sv.ByteTrack()


    def _frame_to_detections(self, frame: np.ndarray) -> sv.Detections:
        """
        Run YOLO inference on a frame and convert results to supervision.Detections.
        """
        result = self.model.predict(
            source=frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False
        )[0]

        if result.boxes is None or len(result.boxes) == 0:
            return sv.Detections.empty()

        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)

        return sv.Detections(
            xyxy=boxes,
            confidence=confidences,
            class_id=class_ids,
        )

    def _filter_relevant_detections(self, detections: sv.Detections) -> sv.Detections:
        """
        Keep only relevant football classes (must match your Players model labels.yaml).
        """
        if len(detections) == 0:
            return detections

        relevant_ids = {
            config.ProjectConfig.PLAYER_CLASS_ID,
            config.ProjectConfig.REFEREE_CLASS_ID,
            config.ProjectConfig.GOALKEEPER_CLASS_ID,
            config.ProjectConfig.BALL_CLASS_ID,
        }

        mask = np.array([cls in relevant_ids for cls in detections.class_id], dtype=bool)
        return detections[mask]

    def detect_and_track_frame(self, frame: np.ndarray) -> sv.Detections:
        """
        Detect objects in one frame and attach track IDs.
        """
        detections = self._frame_to_detections(frame)
        detections = self._filter_relevant_detections(detections)

        if len(detections) == 0:
            return detections

        tracked = self.tracker.update_with_detections(detections)
        return tracked

    def process_video(
        self,
        video_path: str | Path,
        cache_path: Optional[str | Path] = None,
        frame_stride: int = 1,
    ) -> List[TrackResult]:
        """
        Process the entire video and return tracking results per frame.
        Optionally cache results to disk.
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        if cache_path is not None:
            cache_path = Path(cache_path)
            if cache_path.exists():
                with open(cache_path, "rb") as f:
                    return pickle.load(f)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        results: List[TrackResult] = []
        frame_index = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_index % frame_stride != 0:
                frame_index += 1
                continue

            tracked_detections = self.detect_and_track_frame(frame)
            results.append(
                TrackResult(
                    frame_index=frame_index,
                    detections=tracked_detections,
                )
            )
            frame_index += 1

        cap.release()

        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "wb") as f:
                pickle.dump(results, f)

        return results

    @staticmethod
    def group_by_class(detections: sv.Detections) -> Dict[int, sv.Detections]:
        """
        Split detections by class ID.
        """
        if len(detections) == 0 or detections.class_id is None:
            return {}

        groups: Dict[int, List[int]] = {}
        for idx, cls_id in enumerate(detections.class_id):
            groups.setdefault(int(cls_id), []).append(idx)

        return {cls_id: detections[np.array(idxs, dtype=int)] for cls_id, idxs in groups.items()}

    @staticmethod
    def get_track_centers(detections: sv.Detections) -> np.ndarray:
        """
        Return center points of detections as Nx2 array.
        """
        if len(detections) == 0:
            return np.empty((0, 2), dtype=np.float32)

        xyxy = detections.xyxy
        centers = np.column_stack(
            ((xyxy[:, 0] + xyxy[:, 2]) / 2.0, (xyxy[:, 1] + xyxy[:, 3]) / 2.0)
        )
        return centers.astype(np.float32)
