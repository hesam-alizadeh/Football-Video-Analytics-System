from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import cv2
import numpy as np
from sklearn.cluster import KMeans


@dataclass
class TeamColorInfo:
    team_id: int
    color_bgr: np.ndarray


class TeamAssigner:
    """
    Assign football players to teams using jersey color clustering.

    Main idea:
    - crop player bounding boxes
    - use upper-body region for jersey color
    - compute representative color
    - cluster into 2 teams with KMeans
    - cache assignment per track_id
    """

    def __init__(
        self,
        n_teams: int = 2,
        upper_body_ratio: float = 0.55,
        sample_scale: float = 0.5,
        random_state: int = 42,
    ):
        self.n_teams = n_teams
        self.upper_body_ratio = upper_body_ratio
        self.sample_scale = sample_scale
        self.random_state = random_state

        self.track_id_to_team: Dict[int, int] = {}
        self.team_centers: Optional[np.ndarray] = None

    def _get_player_crop(self, frame: np.ndarray, bbox_xyxy: np.ndarray) -> np.ndarray:
        x1, y1, x2, y2 = bbox_xyxy.astype(int)
        h, w = frame.shape[:2]

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        if x2 <= x1 or y2 <= y1:
            return np.empty((0, 0, 3), dtype=np.uint8)

        crop = frame[y1:y2, x1:x2]
        return crop

    def _get_upper_body_crop(self, crop: np.ndarray) -> np.ndarray:
        if crop.size == 0:
            return crop

        h = crop.shape[0]
        upper_h = max(1, int(h * self.upper_body_ratio))
        return crop[:upper_h, :]

    def _preprocess_crop(self, crop: np.ndarray) -> np.ndarray:
        """
        Reduce noise and optionally downsample for faster color extraction.
        """
        if crop.size == 0:
            return crop

        if self.sample_scale != 1.0:
            crop = cv2.resize(
                crop,
                None,
                fx=self.sample_scale,
                fy=self.sample_scale,
                interpolation=cv2.INTER_AREA,
            )
        return crop

    def extract_dominant_color(self, frame: np.ndarray, bbox_xyxy: np.ndarray) -> np.ndarray:
        """
        Extract a representative jersey color from the upper-body crop.
        Returns BGR color as [b, g, r].
        """
        crop = self._get_player_crop(frame, bbox_xyxy)
        crop = self._get_upper_body_crop(crop)
        crop = self._preprocess_crop(crop)

        if crop.size == 0:
            return np.array([0, 0, 0], dtype=np.float32)

        # Flatten pixels to Nx3
        pixels = crop.reshape(-1, 3).astype(np.float32)

        # Remove very dark pixels and very bright pixels if needed
        brightness = pixels.mean(axis=1)
        mask = (brightness > 30) & (brightness < 245)
        pixels = pixels[mask]

        if len(pixels) == 0:
            return np.array([0, 0, 0], dtype=np.float32)

        # Use KMeans to find dominant jersey-like color
        n_clusters = min(3, len(pixels))
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=self.random_state,
            n_init=10
        )
        labels = kmeans.fit_predict(pixels)
        centers = kmeans.cluster_centers_

        # Pick the largest cluster
        counts = np.bincount(labels)
        dominant_idx = np.argmax(counts)
        dominant_color = centers[dominant_idx]

        return dominant_color.astype(np.float32)






    def fit_teams_from_tracks(self, frame: np.ndarray, tracks) -> None:
        """
        Fit KMeans on all player tracks in the current frame and assign them to n_teams.
        'tracks' is expected to be a supervision.Detections object with:
            - tracks.xyxy
            - tracks.tracker_id
            - tracks.class_id
        """
        if tracks is None or tracks.xyxy is None or len(tracks.xyxy) == 0:
            return

        if tracks.tracker_id is None:
            print("WARNING: tracks.tracker_id is None, cannot fit teams")
            return

        colors = []
        valid_track_ids = []

        for bbox, track_id, class_id in zip(tracks.xyxy, tracks.tracker_id, tracks.class_id):
            if track_id is None:
                continue

            # فقط روی بازیکن و دروازه‌بان کام کنید، توپ/داور را نادیده بگیر
            # اگر config را اینجا import کرده‌ای:
            # if class_id not in [config.PLAYER_CLASS_ID, config.GOALKEEPER_CLASS_ID]:
            #     continue

            color = self.extract_dominant_color(frame, bbox)
            if np.all(color == 0):
                continue

            colors.append(color)
            valid_track_ids.append(int(track_id))

        if len(colors) < 2:
            # داده کافی برای ۲ تیم نداریم
            return

        colors = np.array(colors, dtype=np.float32)

        kmeans = KMeans(
            n_clusters=2, # self.n_teams, برای دیباگ 
            random_state=self.random_state,
            n_init=10
        )
        team_labels = kmeans.fit_predict(colors)
        self.team_centers = kmeans.cluster_centers_

        for track_id, team_id in zip(valid_track_ids, team_labels):
            self.track_id_to_team[track_id] = int(team_id)

















    def _extract_xyxy_and_id(self, tr):
        # حالت 1: اگر tr تاپل باشد
        if isinstance(tr, (list, tuple)) and len(tr) >= 2:
            xyxy = tr[0]
            track_id = tr[1]
        
        # حالت 2: اگر آبجکت supervision باشد
        elif hasattr(tr, "xyxy") and hasattr(tr, "tracker_id"):
            xyxy = tr.xyxy
            track_id = tr.tracker_id
        
        # حالت 3: اگر دیکشنری باشد
        elif isinstance(tr, dict):
            xyxy = tr.get("bbox") or tr.get("xyxy")
            track_id = tr.get("track_id") or tr.get("id")
        
        else:
            return None, None

        # ---- FIX ----
        # flatten bbox to (4,) no matter what shape it had
        xyxy = np.array(xyxy, dtype=np.float32).reshape(-1)[:4]

        return xyxy, track_id







    def assign(self, tracks, frame):
        team_assignment = {}
        if tracks is None or tracks.xyxy is None:
            return team_assignment

        for i in range(len(tracks.xyxy)):
            bbox = tracks.xyxy[i]
            track_id = int(tracks.tracker_id[i]) if tracks.tracker_id is not None else None
            
            if track_id is None:
                continue

            # اگر قبلاً برای این ID تیمی تعیین شده، همان را برگردان
            if track_id in self.track_id_to_team:
                team_assignment[track_id] = self.track_id_to_team[track_id]
                continue

            # استخراج رنگ
            color = self.extract_dominant_color(frame, bbox)
            
            # اگر مراکز تیم‌ها (KMeans) آماده است، دقیق تخصیص بده
            if self.team_centers is not None:
                team_id = self.assign_team(frame, bbox, track_id)
                self.track_id_to_team[track_id] = team_id
                team_assignment[track_id] = team_id
            else:
                # اگر هنوز فیت نشده، موقتاً همه را تیم 0 بگذار تا کرش نکند
                team_assignment[track_id] = 0 
                
        return team_assignment



    def assign_team(self, frame: np.ndarray, bbox_xyxy: np.ndarray, track_id: int) -> int:
        """
        Assign team for a single player.
        If cached, return cached label.
        Otherwise estimate from jersey color and compare with fitted team centers.
        """
        if track_id in self.track_id_to_team:
            return self.track_id_to_team[track_id]

        color = self.extract_dominant_color(frame, bbox_xyxy)

        if self.team_centers is not None:
            distances = np.linalg.norm(self.team_centers - color[None, :], axis=1)
            team_id = int(np.argmin(distances))
        else:
            # fallback: simple rule if teams not fitted yet
            team_id = 0 if color.mean() < 128 else 1

        self.track_id_to_team[track_id] = team_id
        return team_id




    def get_team_color_info(self) -> Dict[int, TeamColorInfo]:
        """
        Return team centroid colors for visualization.
        """
        if self.team_centers is None:
            return {}

        return {
            i: TeamColorInfo(team_id=i, color_bgr=self.team_centers[i])
            for i in range(len(self.team_centers))
        }
