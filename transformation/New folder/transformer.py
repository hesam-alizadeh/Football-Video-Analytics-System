from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass
class PitchDimensions:
    length: float = 105.0  # meters
    width: float = 68.0    # meters


class Transformer:
    """
    Transform points from image space to pitch space using homography.
    """

    def __init__(self, pitch_dimensions: PitchDimensions = PitchDimensions()):
        self.pitch_dimensions = pitch_dimensions
        self.homography: Optional[np.ndarray] = None

    def compute_homography(
        self,
        image_points: np.ndarray,
        pitch_points: np.ndarray,
    ) -> Optional[np.ndarray]:
        """
        Compute homography matrix H mapping image_points -> pitch_points.

        image_points: Nx2
        pitch_points: Nx2
        """
        if image_points is None or pitch_points is None:
            return None

        if len(image_points) < 4 or len(pitch_points) < 4:
            raise ValueError("At least 4 point correspondences are required for homography")

        image_points = np.asarray(image_points, dtype=np.float32)
        pitch_points = np.asarray(pitch_points, dtype=np.float32)

        H, status = cv2.findHomography(image_points, pitch_points, method=cv2.RANSAC)
        self.homography = H
        return H

    def transform_points(self, points: np.ndarray) -> Optional[np.ndarray]:
        """
        Apply homography to Nx2 points.
        Returns transformed Nx2 points.
        """
        if self.homography is None:
            return None

        if points is None or len(points) == 0:
            return np.empty((0, 2), dtype=np.float32)

        pts = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
        transformed = cv2.perspectiveTransform(pts, self.homography)
        return transformed.reshape(-1, 2)

    def transform_bbox_center(self, bbox_xyxy: np.ndarray) -> Optional[np.ndarray]:
        """
        Transform the center point of a bounding box.
        """
        if bbox_xyxy is None:
            return None

        x1, y1, x2, y2 = bbox_xyxy.astype(np.float32)
        center = np.array([[[(x1 + x2) / 2.0, (y1 + y2) / 2.0]]], dtype=np.float32)

        if self.homography is None:
            return None

        transformed = cv2.perspectiveTransform(center, self.homography)
        return transformed[0, 0]

    def warp_frame(self, frame: np.ndarray, output_size: tuple[int, int]) -> np.ndarray:
        """
        Warp a frame into top-down pitch view using the current homography.
        output_size: (width, height)
        """
        if self.homography is None:
            raise ValueError("Homography has not been computed yet")

        return cv2.warpPerspective(frame, self.homography, output_size)

    def create_pitch_canvas(self, scale: int = 10) -> np.ndarray:
        """
        Create a blank top-down pitch canvas for visualization.
        scale = pixels per meter
        """
        width_px = int(self.pitch_dimensions.length * scale)
        height_px = int(self.pitch_dimensions.width * scale)

        canvas = np.zeros((height_px, width_px, 3), dtype=np.uint8)
        canvas[:] = (0, 128, 0)  # green field
        return canvas
