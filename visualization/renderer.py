from typing import List, Dict, Any, Optional, Tuple

import cv2
import numpy as np
import supervision as sv



class MockConfig:
    PLAYER_CLASS_ID = 2
    BALL_CLASS_ID = 0
    REFEREE_CLASS_ID = 3
    GOALKEEPER_CLASS_ID = 1


config = MockConfig()


class Renderer:
    """
    Responsible for rendering:
      1. Normal frame annotations: bbox, ellipse, track id, trace
      2. Tactical pitch map from already-transformed world coordinates
      3. Overlaying the tactical pitch map on video frames

    Important architecture:
      - Renderer does NOT compute homography.
      - Renderer expects transformed world/pitch coordinates:
            foot_point_world = [x_meters, y_meters]
      - Homography should be computed in Transformer/main.py.
    """

    def __init__(
        self,
        renderer_params: Dict[str, Any],
        team_assignment: Optional[Dict[int, Any]] = None,
        homography_matrix: Optional[np.ndarray] = None,
        pitch_points: Optional[np.ndarray] = None,
        frame_width: Optional[int] = None,
        frame_height: Optional[int] = None,
    ):
        self.renderer_params = renderer_params or {}
        self.team_assignment = team_assignment or {}

        # Kept for compatibility, but not used for tactical map rendering anymore.
        self.homography_matrix = homography_matrix
        self.pitch_points = pitch_points
        self.frame_width = frame_width
        self.frame_height = frame_height

        # ------------------------------------------------------------------
        # Colors
        # ------------------------------------------------------------------
        colors = self.renderer_params.get("colors", {})

        self.player_color = tuple(colors.get("player", (0, 255, 0)))
        self.ball_color = tuple(colors.get("ball", (0, 165, 255)))
        self.referee_color = tuple(colors.get("referee", (255, 255, 0)))
        self.goalkeeper_color = tuple(colors.get("goalkeeper", (255, 0, 255)))
        self.default_color = tuple(colors.get("default", (255, 255, 255)))

        # Team colors in BGR format for OpenCV.
        # اگر رنگ‌ها در config شما RGB هستند، باید تبدیل شوند.
        self.team_color_map = {
            0: tuple(colors.get("team_0", (0, 0, 255))),      # Red in BGR
            1: tuple(colors.get("team_1", (255, 0, 0))),      # Blue in BGR
        }

        # ------------------------------------------------------------------
        # Annotators
        # ------------------------------------------------------------------
        # Some versions of supervision have slightly different APIs.
        # This setup is intentionally simple.
        self.box_annotator = sv.BoxAnnotator()

        self.trace_annotator = sv.TraceAnnotator(
            thickness=self.renderer_params.get("trace_thickness", 2),
            trace_length=self.renderer_params.get("trace_length", 30)
        )

        # ------------------------------------------------------------------
        # Tactical pitch setup
        # ------------------------------------------------------------------
        # Format: (height, width)
        # A 105x68 pitch is wider than it is tall, so (300, 450) is reasonable.
        self.pitch_canvas = None
        self.pitch_canvas_size = self.renderer_params.get(
            "pitch_canvas_size",
            (300, 450)
        )

        # Real pitch dimensions in meters
        self.pitch_length_m = float(self.renderer_params.get("pitch_length_m", 105.0))
        self.pitch_width_m = float(self.renderer_params.get("pitch_width_m", 68.0))

    # ------------------------------------------------------------------
    # Color utilities
    # ------------------------------------------------------------------
    def _get_color_for_track(self, track_id: int, class_id: int) -> sv.Color:
        """
        Returns sv.Color for a given track/class.
        This is mostly used for normal frame annotations.
        Tactical map may use raw BGR tuples directly.
        """

        if class_id == config.BALL_CLASS_ID:
            return sv.Color.from_bgr_tuple(self.ball_color)

        if class_id == config.REFEREE_CLASS_ID:
            return sv.Color.from_bgr_tuple(self.referee_color)

        if class_id == config.GOALKEEPER_CLASS_ID:
            return sv.Color.from_bgr_tuple(self.goalkeeper_color)

        if class_id == config.PLAYER_CLASS_ID:
            team_id = self.team_assignment.get(track_id, None)

            if isinstance(team_id, int) and team_id in self.team_color_map:
                return sv.Color.from_bgr_tuple(self.team_color_map[team_id])

            return sv.Color.from_bgr_tuple(self.player_color)

        return sv.Color.from_bgr_tuple(self.default_color)

    def _get_bgr_color_for_track(self, track_id: int, class_id: int) -> Tuple[int, int, int]:
        """
        Same logic as _get_color_for_track, but returns raw BGR tuple.
        Useful for cv2 drawing.
        """

        if class_id == config.BALL_CLASS_ID:
            return self.ball_color

        if class_id == config.REFEREE_CLASS_ID:
            return self.referee_color

        if class_id == config.GOALKEEPER_CLASS_ID:
            return self.goalkeeper_color

        if class_id == config.PLAYER_CLASS_ID:
            team_id = self.team_assignment.get(track_id, None)

            if isinstance(team_id, int) and team_id in self.team_color_map:
                return self.team_color_map[team_id]

            return self.player_color

        return self.default_color

    def _normalize_team_id(self, team_id: Any) -> Optional[int]:
        """
        Converts possible team values to 0/1 if possible.

        Supports:
          0, 1
          "0", "1"
          "team0", "team1"
          "team_0", "team_1"
          "red", "blue", etc. as simple fallbacks
        """

        if team_id is None:
            return None

        if isinstance(team_id, int):
            return team_id

        if isinstance(team_id, np.integer):
            return int(team_id)

        if isinstance(team_id, str):
            normalized = team_id.strip().lower()

            if normalized.isdigit():
                return int(normalized)

            if normalized in ["team0", "team_0", "home", "red"]:
                return 0

            if normalized in ["team1", "team_1", "away", "blue", "white"]:
                return 1

        return None

    # ------------------------------------------------------------------
    # Normal video frame annotations
    # ------------------------------------------------------------------
    def render_frame_annotations(
        self,
        frame: np.ndarray,
        tracks_data: List[Dict[str, Any]]
    ) -> np.ndarray:
        """
        Draws normal annotations on the video frame.

        Expected tracks_data item format, approximately:
        {
            "track_id": int,
            "class_id": int,
            "xyxy": [x1, y1, x2, y2],
            ...
        }
        """

        if frame is None:
            return frame

        if not tracks_data:
            return frame

        output_frame = frame.copy()

        for track in tracks_data:
            if "xyxy" not in track:
                continue

            xyxy = np.asarray(track["xyxy"], dtype=np.float32)
            if xyxy.shape[0] < 4:
                continue

            x1, y1, x2, y2 = xyxy[:4].astype(int)

            track_id = int(track.get("track_id", -1))
            class_id = int(track.get("class_id", -1))

            color_bgr = self._get_bgr_color_for_track(track_id, class_id)

            # Draw bbox
            cv2.rectangle(
                output_frame,
                (x1, y1),
                (x2, y2),
                color_bgr,
                self.renderer_params.get("box_thickness", 2)
            )

            # Label
            label = f"#{track_id}"

            cv2.putText(
                output_frame,
                label,
                (x1, max(0, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color_bgr,
                2,
                cv2.LINE_AA
            )

            # Draw ellipse under player/goalkeeper
            if class_id in [config.PLAYER_CLASS_ID, config.GOALKEEPER_CLASS_ID]:
                foot_x = int((x1 + x2) / 2)
                foot_y = int(y2)

                ellipse_width = max(10, int((x2 - x1) * 0.45))
                ellipse_height = max(4, int(ellipse_width * 0.35))

                cv2.ellipse(
                    output_frame,
                    center=(foot_x, foot_y),
                    axes=(ellipse_width, ellipse_height),
                    angle=0,
                    startAngle=0,
                    endAngle=360,
                    color=color_bgr,
                    thickness=2
                )

            # Draw ball as small circle
            if class_id == config.BALL_CLASS_ID:
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                cv2.circle(
                    output_frame,
                    (cx, cy),
                    5,
                    self.ball_color,
                    -1
                )

        return output_frame

    # ------------------------------------------------------------------
    # Tactical pitch drawing utilities
    # ------------------------------------------------------------------
    def _world_to_canvas(
        self,
        point_world: Any,
        canvas_width: int,
        canvas_height: int,
        clamp: bool = True
    ) -> Optional[Tuple[int, int]]:
        """
        Converts real pitch coordinates in meters to tactical canvas pixels.

        Input:
            point_world = [x_meters, y_meters]

        Coordinate convention:
            x: 0..105 along pitch length
            y: 0..68 along pitch width

        Output:
            x_px, y_px on canvas
        """

        if point_world is None:
            return None

        point_world = np.asarray(point_world, dtype=np.float32).reshape(-1)

        if point_world.shape[0] < 2:
            return None

        x_m = float(point_world[0])
        y_m = float(point_world[1])

        if not np.isfinite(x_m) or not np.isfinite(y_m):
            return None

        if clamp:
            x_m = max(0.0, min(self.pitch_length_m, x_m))
            y_m = max(0.0, min(self.pitch_width_m, y_m))
        else:
            if not (0.0 <= x_m <= self.pitch_length_m):
                return None
            if not (0.0 <= y_m <= self.pitch_width_m):
                return None

        x_px = int((x_m / self.pitch_length_m) * (canvas_width - 1))
        y_px = int((y_m / self.pitch_width_m) * (canvas_height - 1))

        return x_px, y_px

    def _draw_pitch_lines(self, canvas: np.ndarray) -> np.ndarray:
        """
        Draws standard football pitch lines on the tactical canvas.
        """

        h, w = canvas.shape[:2]

        white = (255, 255, 255)

        # Outer border
        cv2.rectangle(
            canvas,
            (0, 0),
            (w - 1, h - 1),
            white,
            2
        )

        # Halfway line
        cv2.line(
            canvas,
            (w // 2, 0),
            (w // 2, h - 1),
            white,
            2
        )

        # Center circle
        center_x = w // 2
        center_y = h // 2

        center_circle_radius = int((9.15 / self.pitch_width_m) * h)

        cv2.circle(
            canvas,
            (center_x, center_y),
            center_circle_radius,
            white,
            1
        )

        # Center spot
        cv2.circle(
            canvas,
            (center_x, center_y),
            2,
            white,
            -1
        )

        # Penalty areas
        penalty_depth_px = int((16.5 / self.pitch_length_m) * w)
        penalty_half_height_px = int((40.32 / self.pitch_width_m) * h / 2)

        cv2.rectangle(
            canvas,
            (0, center_y - penalty_half_height_px),
            (penalty_depth_px, center_y + penalty_half_height_px),
            white,
            1
        )

        cv2.rectangle(
            canvas,
            (w - penalty_depth_px, center_y - penalty_half_height_px),
            (w - 1, center_y + penalty_half_height_px),
            white,
            1
        )

        # Goal areas
        goal_depth_px = int((5.5 / self.pitch_length_m) * w)
        goal_half_height_px = int((18.32 / self.pitch_width_m) * h / 2)

        cv2.rectangle(
            canvas,
            (0, center_y - goal_half_height_px),
            (goal_depth_px, center_y + goal_half_height_px),
            white,
            1
        )

        cv2.rectangle(
            canvas,
            (w - goal_depth_px, center_y - goal_half_height_px),
            (w - 1, center_y + goal_half_height_px),
            white,
            1
        )

        # Penalty spots
        left_penalty_spot = self._world_to_canvas(
            [11.0, self.pitch_width_m / 2],
            canvas_width=w,
            canvas_height=h
        )

        right_penalty_spot = self._world_to_canvas(
            [self.pitch_length_m - 11.0, self.pitch_width_m / 2],
            canvas_width=w,
            canvas_height=h
        )

        if left_penalty_spot is not None:
            cv2.circle(canvas, left_penalty_spot, 2, white, -1)

        if right_penalty_spot is not None:
            cv2.circle(canvas, right_penalty_spot, 2, white, -1)

        return canvas

    # ------------------------------------------------------------------
    # Tactical pitch map
    # ------------------------------------------------------------------
    def render_tactical_map(
        self,
        transformed_players: Optional[List[Dict[str, Any]]] = None,
        transformed_balls: Optional[List[Dict[str, Any]]] = None
    ) -> np.ndarray:
        """
        Creates a top-down tactical pitch map.

        Expected transformed_players item:
        {
            "track_id": int,
            "class_id": int,
            "team": int or str, optional
            "foot_point_world": [x_meters, y_meters]
        }

        Expected transformed_balls item:
        {
            "track_id": int, optional
            "class_id": int, optional
            "point_world": [x_meters, y_meters]
        }

        Important:
            This function expects coordinates already transformed by homography.
            It does not transform image coordinates.
        """

        transformed_players = transformed_players or []
        transformed_balls = transformed_balls or []

        h, w = self.pitch_canvas_size

        # Background
        tactical_pitch = np.zeros((h, w, 3), dtype=np.uint8)
        tactical_pitch[:] = tuple(
            self.renderer_params.get("pitch_background_color", (34, 139, 34))
        )

        # Draw field lines
        tactical_pitch = self._draw_pitch_lines(tactical_pitch)

        # --------------------------------------------------------------
        # Draw players
        # --------------------------------------------------------------
        for player in transformed_players:
            point_world = (
                player.get("foot_point_world")
                or player.get("point_world")
                or player.get("world_point")
                or player.get("point")
            )

            canvas_pos = self._world_to_canvas(
                point_world,
                canvas_width=w,
                canvas_height=h,
                clamp=True
            )

            if canvas_pos is None:
                continue

            x_map, y_map = canvas_pos

            track_id = int(player.get("track_id", -1))
            class_id = int(player.get("class_id", config.PLAYER_CLASS_ID))

            # Prefer team inside player dict.
            # Fallback to self.team_assignment.
            team_id = player.get("team", None)

            if team_id is None:
                team_id = self.team_assignment.get(track_id, None)

            team_id = self._normalize_team_id(team_id)

            if team_id is not None and team_id in self.team_color_map:
                color_bgr = self.team_color_map[team_id]
            else:
                color_bgr = self._get_bgr_color_for_track(track_id, class_id)

            # Player dot
            cv2.circle(
                tactical_pitch,
                (x_map, y_map),
                self.renderer_params.get("tactical_player_radius", 6),
                color_bgr,
                -1
            )

            # Black outline
            cv2.circle(
                tactical_pitch,
                (x_map, y_map),
                self.renderer_params.get("tactical_player_radius", 6) + 1,
                (0, 0, 0),
                1
            )

            # Track id
            if self.renderer_params.get("show_tactical_track_id", True):
                cv2.putText(
                    tactical_pitch,
                    str(track_id),
                    (x_map + 7, y_map - 7),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA
                )

        # --------------------------------------------------------------
        # Draw balls
        # --------------------------------------------------------------
        for ball in transformed_balls:
            point_world = (
                ball.get("point_world")
                or ball.get("foot_point_world")
                or ball.get("world_point")
                or ball.get("point")
            )

            canvas_pos = self._world_to_canvas(
                point_world,
                canvas_width=w,
                canvas_height=h,
                clamp=True
            )

            if canvas_pos is None:
                continue

            x_map, y_map = canvas_pos

            cv2.circle(
                tactical_pitch,
                (x_map, y_map),
                self.renderer_params.get("tactical_ball_radius", 4),
                self.ball_color,
                -1
            )

            cv2.circle(
                tactical_pitch,
                (x_map, y_map),
                self.renderer_params.get("tactical_ball_radius", 4) + 1,
                (255, 255, 255),
                1
            )

        return tactical_pitch

    # ------------------------------------------------------------------
    # Overlay tactical map on frame
    # ------------------------------------------------------------------
    def overlay_tactical_map(
        self,
        frame: np.ndarray,
        tactical_map: np.ndarray,
        position: str = "top_right",
        padding: int = 10,
        alpha: float = 0.75
    ) -> np.ndarray:
        """
        Overlays tactical_map onto frame.

        position options:
            "top_right"
            "top_left"
            "bottom_right"
            "bottom_left"
        """

        if frame is None:
            return frame

        if tactical_map is None:
            return frame

        output = frame.copy()

        frame_h, frame_w = output.shape[:2]
        map_h, map_w = tactical_map.shape[:2]

        if map_h <= 0 or map_w <= 0:
            return output

        # Resize map if it is too large for the frame.
        max_w = frame_w - 2 * padding
        max_h = frame_h - 2 * padding

        if max_w <= 0 or max_h <= 0:
            return output

        if map_w > max_w or map_h > max_h:
            scale = min(max_w / map_w, max_h / map_h)
            new_w = max(1, int(map_w * scale))
            new_h = max(1, int(map_h * scale))

            tactical_map = cv2.resize(
                tactical_map,
                (new_w, new_h),
                interpolation=cv2.INTER_AREA
            )

            map_h, map_w = tactical_map.shape[:2]

        # Position
        if position == "top_left":
            x = padding
            y = padding

        elif position == "bottom_left":
            x = padding
            y = frame_h - map_h - padding

        elif position == "bottom_right":
            x = frame_w - map_w - padding
            y = frame_h - map_h - padding

        else:
            # default: top_right
            x = frame_w - map_w - padding
            y = padding

        # Safety check
        if x < 0 or y < 0:
            return output

        if x + map_w > frame_w or y + map_h > frame_h:
            return output

        roi = output[y:y + map_h, x:x + map_w]

        try:
            blended = cv2.addWeighted(
                roi,
                1.0 - alpha,
                tactical_map,
                alpha,
                0
            )

            output[y:y + map_h, x:x + map_w] = blended

        except cv2.error as e:
            print(f"[Renderer] cv2.addWeighted failed: {e}")
            return output

        return output

    # ------------------------------------------------------------------
    # Main tactical overlay wrapper
    # ------------------------------------------------------------------
    def render_tactical_pitch_overlay(
        self,
        frame: np.ndarray,
        transformed_players: Optional[List[Dict[str, Any]]] = None,
        transformed_balls: Optional[List[Dict[str, Any]]] = None,
        position: str = "top_right"
    ) -> np.ndarray:
        """
        Builds a tactical map from transformed world coordinates and overlays it.

        Expected call:

            annotated_frame = renderer.render_tactical_pitch_overlay(
                frame=annotated_frame,
                transformed_players=transformed_players,
                transformed_balls=transformed_balls,
                position="top_right"
            )

        Important:
            transformed_players must NOT be raw tracks_data.
            It must contain foot_point_world values in pitch meters.
        """

        tactical_map = self.render_tactical_map(
            transformed_players=transformed_players,
            transformed_balls=transformed_balls
        )

        return self.overlay_tactical_map(
            frame=frame,
            tactical_map=tactical_map,
            position=position,
            padding=self.renderer_params.get("tactical_padding", 10),
            alpha=self.renderer_params.get("tactical_alpha", 0.75)
        )

    # ------------------------------------------------------------------
    # Legacy compatibility helpers
    # ------------------------------------------------------------------
    def _setup_tactical_pitch(self):
        """
        Legacy method.

        This method is intentionally not used in the new architecture.
        Homography should be handled by Transformer/main.py, not Renderer.
        """

        print(
            "[Renderer] _setup_tactical_pitch is deprecated. "
            "Use render_tactical_map with transformed world coordinates instead."
        )

    def _transform_points_to_pitch_canvas(self, points: np.ndarray) -> Optional[np.ndarray]:
        """
        Legacy method.

        Kept only to avoid breaking old code accidentally.
        New code should not use this.
        """

        print(
            "[Renderer] _transform_points_to_pitch_canvas is deprecated. "
            "Use Transformer to create foot_point_world first."
        )

        return None

    @staticmethod
    def get_class_id_map() -> Dict[int, str]:
        """
        Returns class id to class name mapping.
        """

        return {
            config.BALL_CLASS_ID: "ball",
            config.GOALKEEPER_CLASS_ID: "goalkeeper",
            config.PLAYER_CLASS_ID: "player",
            config.REFEREE_CLASS_ID: "referee",
        }




