from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO


class PitchKeypointDetector:
    """
    Detect pitch keypoints from a football frame using a trained YOLO pose/keypoint model.

    نکته مهم:
    خروجی detect دیگر keypointها را فیلتر و فشرده نمی‌کند.
    یعنی index اصلی keypointهای مدل حفظ می‌شود.

    خروجی detect:
        np.ndarray با shape برابر با:

        اگر مدل confidence بدهد:
            (K, 3) => [x, y, conf]

        اگر فقط مختصات بدهد:
            (K, 2) => [x, y]

    این طراحی برای Homography ضروری است، چون mapping بین index keypoint و مختصات واقعی زمین
    نباید خراب شود.
    """

    def __init__(
        self,
        model_path: str = "models/field/best.pt",
        conf_threshold: float = 0.25,
        keypoint_conf_threshold: float = 0.3,
    ):
        self.model_path = Path(model_path)
        self.conf_threshold = conf_threshold
        self.keypoint_conf_threshold = keypoint_conf_threshold

        if not self.model_path.exists():
            raise FileNotFoundError(f"Field keypoint model not found: {self.model_path}")

        self.model = YOLO(str(self.model_path))

                # استاندارد ۵۸ نقطه‌ای برای زمین ۱۰۵x۶۸ متر
        self.PITCH_COORDINATES = {
            # خط وسط و دایره مرکز
            8:  [52.5, 0.0],   # Halfway line - Top
            9:  [52.5, 68.0],  # Halfway line - Bottom
            10: [52.5, 34.0],  # Center spot
            
            # محوطه جریمه سمت چپ (Left Penalty Area)
            4:  [0.0, 20.15],  # Left Pen Area - Top Corner
            5:  [0.0, 47.85],  # Left Pen Area - Bottom Corner
            13: [16.5, 20.15], # Left Pen Area - Top Inner
            14: [16.5, 47.85], # Left Pen Area - Bottom Inner
            
            # محوطه جریمه سمت راست (Right Penalty Area)
            6:  [105.0, 20.15], # Right Pen Area - Top Corner
            7:  [105.0, 47.85], # Right Pen Area - Bottom Corner
            19: [88.5, 20.15],  # Right Pen Area - Top Inner
            20: [88.5, 47.85],  # Right Pen Area - Bottom Inner
            
            # گوشه‌های زمین (Corners)
            0: [0.0, 0.0],     # Top Left
            3: [0.0, 68.0],    # Bottom Left
            1: [105.0, 0.0],   # Top Right
            2: [105.0, 68.0],  # Bottom Right
            
            # نقاط کمکی خطوط محوطه (Six Yard Box)
            11: [0.0, 24.85],  # Left 6-yard - Top
            12: [0.0, 43.15],  # Left 6-yard - Bottom
            17: [105.0, 24.85],# Right 6-yard - Top
            18: [105.0, 43.15] # Right 6-yard - Bottom
        }




    def get_valid_point_pairs(self, detected_kpts):
        image_points = []
        pitch_points = []

        for idx, kp in enumerate(detected_kpts):
            if idx in self.PITCH_COORDINATES:
                x, y = kp[0], kp[1]
                image_points.append([x, y])
                pitch_points.append(self.PITCH_COORDINATES[idx])

        return np.array(image_points, dtype=np.float32), np.array(pitch_points, dtype=np.float32)








    def detect(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect pitch keypoints.

        Returns:
            Optional[np.ndarray]

            اگر keypoint پیدا شود:
                shape: (K, 3) => [x, y, conf]
                یا در حالت fallback:
                shape: (K, 2) => [x, y]

            اگر چیزی پیدا نشود:
                None

        نکته:
            این متد عمداً keypointهای کم‌اعتماد را حذف نمی‌کند.
            چون حذف کردن آن‌ها باعث خراب شدن indexها می‌شود.
            فیلتر confidence داخل get_corresponding_points انجام می‌شود.
        """

        if frame is None:
            return None

        results = self.model.predict(
            source=frame,
            conf=self.conf_threshold,
            verbose=False,
        )

        if not results:
            return None

        result = results[0]

        if result.keypoints is None or result.keypoints.data is None:
            return None

        keypoints = result.keypoints.data.cpu().numpy()

        if keypoints is None or keypoints.size == 0:
            return None

        # معمولاً shape این است:
        # (N, K, 3)
        # N = تعداد instanceهای تشخیص داده شده
        # K = تعداد keypointها
        # 3 = x, y, confidence
        if keypoints.ndim != 3:
            return None

        if keypoints.shape[0] == 0:
            return None

        # اگر چند instance تشخیص داده شد، بهترین instance را انتخاب می‌کنیم.
        # برای مدل field معمولاً فقط یک instance داریم، اما این روش امن‌تر است.
        best_instance_index = self._select_best_keypoint_instance(keypoints)

        pts = keypoints[best_instance_index]

        if pts is None or pts.size == 0:
            return None

        # اگر مدل x,y,conf داده باشد
        if pts.shape[1] >= 3:
            pts_out = pts[:, :3].astype(np.float32)

            # بعضی مدل‌ها برای keypoint نامعتبر x=y=0 می‌دهند.
            # ما اینجا حذف نمی‌کنیم، فقط همان را نگه می‌داریم تا index حفظ شود.
            return pts_out

        # fallback اگر فقط x,y وجود داشته باشد
        if pts.shape[1] >= 2:
            pts_out = pts[:, :2].astype(np.float32)
            return pts_out

        return None

    def _select_best_keypoint_instance(self, keypoints: np.ndarray) -> int:
        """
        اگر مدل چند instance خروجی داد، بهترین instance را انتخاب می‌کند.

        معیار:
            میانگین confidence keypointها.

        keypoints shape:
            (N, K, 3)
        """

        if keypoints.shape[0] == 1:
            return 0

        # اگر confidence وجود داشته باشد
        if keypoints.shape[2] >= 3:
            confs = keypoints[:, :, 2]

            # میانگین confidence هر instance
            # اگر همه صفر باشند هم argmax عددی برمی‌گرداند.
            mean_confs = np.nanmean(confs, axis=1)

            if mean_confs.size > 0:
                return int(np.nanargmax(mean_confs))

        return 0

    def draw_keypoints(
        self,
        frame: np.ndarray,
        keypoints: Optional[np.ndarray],
        draw_only_valid: bool = False,
    ) -> np.ndarray:
        """
        Draw detected keypoints on frame.

        Args:
            frame:
                تصویر ورودی.

            keypoints:
                خروجی detect، یعنی آرایه Kx3 یا Kx2.

            draw_only_valid:
                اگر True باشد، فقط نقاطی با confidence کافی رسم می‌شوند.
                اگر False باشد، همه نقاط دارای مختصات رسم می‌شوند.

        نکته:
            عددی که کنار هر نقطه نوشته می‌شود، index واقعی keypoint مدل است.
        """

        annotated = frame.copy()

        if keypoints is None or len(keypoints) == 0:
            return annotated

        for idx, kp in enumerate(keypoints):
            if len(kp) < 2:
                continue

            x = float(kp[0])
            y = float(kp[1])

            conf = None
            if len(kp) >= 3:
                conf = float(kp[2])

            # نقطه نامعتبر
            if not np.isfinite(x) or not np.isfinite(y):
                continue

            # خیلی از مدل‌ها برای نقاط نامعتبر x=y=0 می‌دهند.
            if x <= 0 and y <= 0:
                continue

            if draw_only_valid and conf is not None:
                if conf < self.keypoint_conf_threshold:
                    continue

            if conf is not None:
                if conf >= self.keypoint_conf_threshold:
                    color = (0, 255, 0)      # green
                else:
                    color = (0, 165, 255)    # orange
            else:
                color = (0, 0, 255)          # red fallback

            cv2.circle(
                annotated,
                (int(x), int(y)),
                5,
                color,
                -1,
            )

            label = str(idx)
            if conf is not None:
                label = f"{idx}:{conf:.2f}"

            cv2.putText(
                annotated,
                label,
                (int(x) + 6, int(y) - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        return annotated

    def get_corresponding_points(
        self,
        detected_kpts: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert detected keypoints to corresponding image and pitch points.

        Args:
            detected_kpts:
                خروجی متد detect.
                شکل مورد انتظار:
                    Kx3 => [x, y, conf]
                    یا
                    Kx2 => [x, y]

        Returns:
            image_points:
                مختصات keypointها در تصویر، shape برابر Nx2.

            pitch_points:
                مختصات متناظر روی زمین واقعی، shape برابر Nx2.

        نکته مهم:
            اینجا با enumerate روی آرایه اصلی حرکت می‌کنیم.
            چون detect دیگر نقاط کم‌اعتماد را حذف نمی‌کند، indexها سالم می‌مانند.
        """

        if detected_kpts is None or len(detected_kpts) == 0:
            return (
                np.empty((0, 2), dtype=np.float32),
                np.empty((0, 2), dtype=np.float32),
            )

        image_points = []
        pitch_points = []

        for kp_index, kp in enumerate(detected_kpts):
            # اگر این index در mapping واقعی زمین تعریف نشده، رد می‌کنیم
            if kp_index not in self.PITCH_COORDINATES:
                continue

            if len(kp) < 2:
                continue

            x = float(kp[0])
            y = float(kp[1])

            # validation مختصات
            if not np.isfinite(x) or not np.isfinite(y):
                continue

            # خیلی از مدل‌ها برای keypoint نامعتبر x=y=0 می‌دهند
            if x <= 0 and y <= 0:
                continue

            # اگر confidence داریم، فیلتر را اینجا اعمال می‌کنیم
            if len(kp) >= 3:
                conf = float(kp[2])

                if not np.isfinite(conf):
                    continue

                if conf < self.keypoint_conf_threshold:
                    continue

            image_points.append([x, y])
            pitch_points.append(self.PITCH_COORDINATES[kp_index])

        if len(image_points) == 0:
            return (
                np.empty((0, 2), dtype=np.float32),
                np.empty((0, 2), dtype=np.float32),
            )

        return (
            np.asarray(image_points, dtype=np.float32),
            np.asarray(pitch_points, dtype=np.float32),
        )

    def get_valid_keypoint_count(self, detected_kpts: Optional[np.ndarray]) -> int:
        """
        تعداد keypointهای معتبر و قابل استفاده برای Homography را برمی‌گرداند.
        این helper برای debug و log مفید است.
        """

        image_points, pitch_points = self.get_corresponding_points(detected_kpts)
        return min(len(image_points), len(pitch_points))







