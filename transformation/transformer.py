from __future__ import annotations

from dataclasses import dataclass, field
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

    def __init__(self, pitch_dimensions: PitchDimensions = None):
        self.pitch_dimensions = pitch_dimensions or PitchDimensions()
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

        image_points = np.asarray(image_points, dtype=np.float32)
        pitch_points = np.asarray(pitch_points, dtype=np.float32)

        if image_points.ndim != 2 or pitch_points.ndim != 2:
            raise ValueError("image_points and pitch_points must be Nx2 arrays")

        if image_points.shape[0] < 4 or pitch_points.shape[0] < 4:
            raise ValueError("At least 4 point correspondences are required for homography")

        if image_points.shape[0] != pitch_points.shape[0]:
            raise ValueError("image_points and pitch_points must have the same number of points")

        H, status = cv2.findHomography(image_points, pitch_points, method=cv2.RANSAC)

        if H is None:
            self.homography = None
            return None

        self.homography = H
        return H

    def transform_points(self, points: np.ndarray) -> Optional[np.ndarray]:
        """
        Apply homography to Nx2 points.
        Returns transformed Nx2 points.
        """
        if self.homography is None:
            return None

        if points is None:
            return None

        points = np.asarray(points, dtype=np.float32)

        if points.size == 0:
            return np.empty((0, 2), dtype=np.float32)

        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("points must be an Nx2 array")

        pts = points.reshape(-1, 1, 2)
        transformed = cv2.perspectiveTransform(pts, self.homography)
        return transformed.reshape(-1, 2)

    def transform_bbox_center(self, bbox_xyxy: np.ndarray) -> Optional[np.ndarray]:
        """
        Transform the center point of a bounding box.
        """
        if self.homography is None or bbox_xyxy is None:
            return None

        bbox_xyxy = np.asarray(bbox_xyxy, dtype=np.float32)
        if bbox_xyxy.shape[0] != 4:
            raise ValueError("bbox_xyxy must have 4 values: [x1, y1, x2, y2]")

        x1, y1, x2, y2 = bbox_xyxy
        center = np.array([[[ (x1 + x2) / 2.0, (y1 + y2) / 2.0 ]]], dtype=np.float32)

        transformed = cv2.perspectiveTransform(center, self.homography)
        return transformed[0, 0]








# این زیر اضافه شد تا دقیق نقطه ی اتصال پای بازیکن با زمین شناخته شود تا لوکیشن آن
# بازیکن دقیق دربیاد وگرنه از مرکز بازیکن نمیشه کار را درآورد 




    def transform_bbox_to_foot(self, bbox_xyxy: np.ndarray) -> Optional[np.ndarray]:
        """
        Get the bottom-center point of a bounding box in image space.
        This approximates the player's foot position on the ground.
        خروجی فقط نقطه تصویر است، هنوز تبدیل هموگرافی روی آن اعمال نشده.
        """
        if bbox_xyxy is None:
            return None

        bbox_xyxy = np.asarray(bbox_xyxy, dtype=np.float32)
        if bbox_xyxy.shape[0] != 4:
            raise ValueError("bbox_xyxy must have 4 values: [x1, y1, x2, y2]")

        x1, y1, x2, y2 = bbox_xyxy
        foot_point = np.array([(x1 + x2) / 2.0, y2], dtype=np.float32)
        return foot_point
    

    def transform_point(self, point: np.ndarray) -> Optional[np.ndarray]:
        """
        Transform a single 2D point from image space to pitch space.
        point: [x, y]
        """
        if self.homography is None or point is None:
            return None

        point = np.asarray(point, dtype=np.float32)

        if point.shape != (2,):
            raise ValueError("point must have shape (2,)")

        point_reshaped = point.reshape(1, 1, 2)
        transformed = cv2.perspectiveTransform(point_reshaped, self.homography)
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
        canvas[:] = (0, 128, 0)
        return canvas
