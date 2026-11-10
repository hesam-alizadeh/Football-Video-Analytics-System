
import cv2
import numpy as np
import pickle
import os
import time
from typing import Dict, Any, Optional

from config import load_config, ProjectConfig
from tracking.tracker import Tracker
from transformation.transformer import Transformer, PitchDimensions
from pitch_key_points.pitch_keypoints import PitchKeypointDetector
from team_assignment.team_assigner import TeamAssigner
from visualization.renderer import Renderer
from utils.video_utils import VideoWriter


# -------------------------
# Helper functions
# -------------------------
def setup_directories(config: ProjectConfig):
    output_dir = os.path.dirname(config.output_video)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    if getattr(config, "use_cache", False):
        os.makedirs(config.cache_dir, exist_ok=True)


def load_cached_data(frame_index: int, cache_dir: str) -> Optional[Dict[str, Any]]:
    cache_file = os.path.join(cache_dir, f"frame_{frame_index:06d}.pkl")
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "rb") as f:
                print(f"Loading data from cache: {cache_file}")
                return pickle.load(f)
        except Exception as e:
            print(f"Error loading cache file {cache_file}: {e}")
            return None
    return None


def save_cached_data(frame_index: int, cache_dir: str, data: Dict[str, Any]):
    cache_file = os.path.join(cache_dir, f"frame_{frame_index:06d}.pkl")
    try:
        with open(cache_file, "wb") as f:
            pickle.dump(data, f)
    except Exception as e:
        print(f"Error saving cache file {cache_file}: {e}")


def generate_final_report(
    start_time: float,
    frame_count: int,
    config: ProjectConfig,
    timing_info: Dict[str, float],
):
    end_time = time.time()
    total_time = end_time - start_time
    fps = frame_count / total_time if total_time > 0 else 0

    report = f"""
## Final Project Report

**Input Video:** {config.input_video}
**Output Video:** {config.output_video}
**Total Frames Processed:** {frame_count}
**Total Processing Time:** {total_time:.2f} seconds
**Average FPS:** {fps:.2f}

### Timing Breakdown:
"""
    for step, duration in timing_info.items():
        report += f"- {step}: {duration:.2f}s\n"

    report += """
### Notes:
- Detection, tracking & homography depend on your modules.
"""
    print(report)

    report_path = os.path.join(os.path.dirname(config.output_video), "project_report.txt")
    try:
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)
    except Exception as e:
        print(f"Could not write report file: {e}")


def _extract_xyxy_and_id(tr):
    # --- Case 1: supervision Detections object ---
    if hasattr(tr, "xyxy") and hasattr(tr, "tracker_id"):
        if tr.xyxy is not None and len(tr.xyxy) > 0:
            xyxy = np.asarray(tr.xyxy[0]).flatten()[:4]
            track_id = int(tr.tracker_id[0]) if tr.tracker_id is not None else None
            return xyxy, track_id

    # --- Case 2: tuple format ---
    if isinstance(tr, tuple) and len(tr) >= 5:
        xyxy = np.asarray(tr[0]).flatten()[:4]
        track_id = tr[4]
        return xyxy, track_id

    return None, None


def _extract_class_id(track):
    if isinstance(track, dict):
        return track.get("class_id", track.get("cls", None))
    return getattr(track, "class_id", getattr(track, "cls", None))


def main():
    start_time = time.time()

    config = load_config()
    setup_directories(config)

    # -------------------------
    # INIT MODULES
    # -------------------------
    tracker = Tracker(
        model_path=config.PLAYERS_MODEL_PATH,
        track_buffer=config.tracker_params.get("track_buffer", 30),
        match_threshold=config.tracker_params.get("match_thresh", 0.8)
    )

    team_assigner = TeamAssigner(n_teams=2)

    pitch_dims = getattr(config, "pitch_dimensions", None)
    if isinstance(pitch_dims, dict):
        transformer = Transformer(
            PitchDimensions(
                length=pitch_dims.get("length", 105.0),
                width=pitch_dims.get("width", 68.0),
            )
        )
    else:
        transformer = Transformer()

    # -------------------------
    # STATIC HOMOGRAPHY SETUP FROM CONFIG
    # -------------------------
    # اگر config.image_points و config.pitch_points درست باشند،
    # این homography اولیه استفاده می‌شود.
    image_points = np.array(config.image_points, dtype=np.float32)
    pitch_points = np.array(config.pitch_points, dtype=np.float32)

    homography_matrix = transformer.compute_homography(image_points, pitch_points)
    if homography_matrix is None:
        print("Error computing initial homography from config points.")
        return

    # -------------------------
    # VIDEO I/O
    # -------------------------
    cap = cv2.VideoCapture(config.input_video)
    if not cap.isOpened():
        print("Cannot open input video.")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    output_path = config.output_video
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        (frame_width, frame_height),
    )

    # -------------------------
    # INIT RENDERER
    # -------------------------
    renderer = Renderer(
        renderer_params=config.renderer_params,
        team_assignment={},
        homography_matrix=homography_matrix,   # فقط برای compatibility
        pitch_points=pitch_points,             # فقط برای compatibility
        frame_width=frame_width,
        frame_height=frame_height
    )

    frame_index = 0
    timing_info = {
        "Detection": 0.0,
        "Tracking": 0.0,
        "Team Assignment": 0.0,
        "Homography/Projection": 0.0,
        "Rendering": 0.0,
        "Cache I/O": 0.0,
    }

    keypoint_detector = PitchKeypointDetector(
        model_path=config.FIELD_MODEL_PATH,
        conf_threshold=0.25,
        keypoint_conf_threshold=0.3
    )

    # اگر در config class id واقعی داری، از آن‌ها استفاده کن
    PLAYER_CLASS_ID = getattr(config, "PLAYER_CLASS_ID", 2)
    GOALKEEPER_CLASS_ID = getattr(config, "GOALKEEPER_CLASS_ID", 1)
    BALL_CLASS_ID = getattr(config, "BALL_CLASS_ID", 0)
    REFEREE_CLASS_ID = getattr(config, "REFEREE_CLASS_ID", 3)








    # در خارج از حلقه (قبل از شروع حلقه پردازش فریم‌ها)
    previous_H = None
    alpha = 0.2  # ضریب نرم‌سازی (بین 0 و 1 - هرچه کمتر باشد، پرش کمتر اما تأخیر بیشتر)




    prev_H = None
    H = None




    # -------------------------
    # MAIN LOOP
    # -------------------------
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("No more frames or invalid frame, stopping.")
            break

        tracks = []
        team_assignment = {}

        # -------------------------
        # DETECT + TRACK
        # -------------------------
        t0 = time.time()
        tracks = tracker.detect_and_track_frame(frame)
        elapsed = time.time() - t0
        timing_info["Detection"] += elapsed
        timing_info["Tracking"] += elapsed

        # -------------------------
        # TEAM FIT (first frames)
        # -------------------------
        if team_assigner.team_centers is None:
            try:
                team_assigner.fit_teams_from_tracks(frame, tracks)
            except Exception as e:
                print(f"[Warning] team fitting failed on frame {frame_index}: {e}")

        # -------------------------
        # TEAM ASSIGNMENT
        # -------------------------
        t0 = time.time()
        try:
            team_assignment = team_assigner.assign(tracks, frame)
        except Exception as e:
            print(f"[Warning] team assignment failed on frame {frame_index}: {e}")
            team_assignment = {}

        timing_info["Team Assignment"] += time.time() - t0
        renderer.team_assignment = team_assignment

        # -------------------------
        # PREPARE RAW TRACK DATA FOR NORMAL FRAME ANNOTATION
        # -------------------------
        tracks_data_for_renderer = []

        if hasattr(tracks, "xyxy") and tracks.xyxy is not None and len(tracks.xyxy) > 0:
            for i in range(len(tracks.xyxy)):
                xyxy = tracks.xyxy[i]
                track_id = int(tracks.tracker_id[i]) if getattr(tracks, "tracker_id", None) is not None else i
                class_id = int(tracks.class_id[i]) if getattr(tracks, "class_id", None) is not None else -1

                tracks_data_for_renderer.append({
                    "track_id": track_id,
                    "xyxy": xyxy.tolist() if hasattr(xyxy, "tolist") else list(xyxy),
                    "class_id": class_id,
                    "team": team_assignment.get(track_id, "unknown"),
                })

        # -------------------------
        # HOMOGRAPHY / PROJECTION
        # -------------------------
        t0 = time.time()






    # ۱. گرفتن کی‌پوینت‌ها از مدل مخصوص زمین
        detected_kpts = keypoint_detector.detect(frame)

        if detected_kpts is not None:
            # ۲. استخراج نقاط متناظر (تصویر -> زمین)
            # این تابع را قبلاً در pitch_keypoints.py نوشتیم
            image_points, pitch_points = keypoint_detector.get_valid_point_pairs(detected_kpts)
            
            if len(image_points) >= 4:
                # ۳. محاسبه هموگرافی واقعی بر اساس فریم جاری
                H_new, _ = cv2.findHomography(image_points, pitch_points, cv2.RANSAC, 5.0)
                
                # ۴. اعمال Smoothing (همان کدی که قبلاً گفتم)
                if H_new is not None:
                    if prev_H is None: prev_H = H_new
                    H = 0.8 * prev_H + 0.2 * H_new
                    prev_H = H
                    
                    # ۵. حالا از این H برای تبدیل مختصات بازیکنان استفاده کنید
                    # transformed_players = ...
            else:
                print("Warning: Not enough keypoints detected for this frame!")


























        transformed_players = []
        transformed_balls = []

        # تبدیل trackها به world coordinates
        if (
            hasattr(tracks, "xyxy")
            and tracks.xyxy is not None
            and len(tracks.xyxy) > 0
            and transformer.homography is not None
        ):
            for i in range(len(tracks.xyxy)):
                bbox = tracks.xyxy[i]
                track_id = int(tracks.tracker_id[i]) if getattr(tracks, "tracker_id", None) is not None else i
                class_id = int(tracks.class_id[i]) if getattr(tracks, "class_id", None) is not None else -1

                # 1) player / goalkeeper -> use foot point
                if class_id in [PLAYER_CLASS_ID, GOALKEEPER_CLASS_ID]:
                    foot_point = transformer.transform_bbox_to_foot(bbox)
                    transformed_point = transformer.transform_point(foot_point)

                    if transformed_point is None:
                        continue

                    team_id = team_assignment.get(track_id, None)

                    transformed_players.append({
                        "track_id": track_id,
                        "class_id": class_id,
                        "team": team_id,
                        "foot_point_world": (
                            transformed_point.tolist()
                            if hasattr(transformed_point, "tolist")
                            else list(transformed_point)
                        )
                    })

                # 2) ball -> use bbox center or bottom-center
                elif class_id == BALL_CLASS_ID:
                    transformed_point = transformer.transform_bbox_center(bbox)

                    if transformed_point is None:
                        continue

                    transformed_balls.append({
                        "track_id": track_id,
                        "class_id": class_id,
                        "point_world": (
                            transformed_point.tolist()
                            if hasattr(transformed_point, "tolist")
                            else list(transformed_point)
                        )
                    })

                # referee / other classes are ignored for tactical map
                else:
                    continue





                # --- قبل از شروع حلقه while در main.py ---
        prev_H = None





        # --- قبل از cv2.findHomography ---
        print("DEBUG: Mapping Check:")
        for i in range(len(image_points)):
            # این پرینت به شما نشان می‌دهد که آیا نقطه درستی به نقطه درستی وصل شده
            print(f"Index {i}: {image_points[i]} -> {pitch_points[i]}")

        # بعد از مشاهده خروجی در کنسول:
        # اگر مثلاً Index 0 در تصویر (گوشه) به یک نقطه وسط در dst_points وصل شده،
        # باید فایل pitch_keypoints.py را باز کنید و شماره اندیس آن را در PITCH_COORDINATES اصلاح کنید.


        # --- داخل حلقه پردازش ---
        # فرض کنیم H_new همان ماتریسی است که cv2.findHomography برمی‌گرداند
        H_new, _ = cv2.findHomography(image_points, pitch_points, cv2.RANSAC, 5.0)

        if H_new is not None:
            if prev_H is not None:
                # 0.8 وزن ماتریس قبلی و 0.2 وزن ماتریس جدید (پایداری عالی)
                H = 0.8 * prev_H + 0.2 * H_new
            else:
                H = H_new
            prev_H = H








        timing_info["Homography/Projection"] += time.time() - t0

        # -------------------------
        # RENDER
        # -------------------------
        t0 = time.time()

        if frame is not None:
            frame_to_render = frame.copy()

            # 1) normal frame annotations
            annotated_frame = renderer.render_frame_annotations(
                frame=frame_to_render,
                tracks_data=tracks_data_for_renderer
            )

            # 2) tactical map overlay
            annotated_frame = renderer.render_tactical_pitch_overlay(
                frame=annotated_frame,
                transformed_players=transformed_players,
                transformed_balls=transformed_balls,
                position="top_right"
            )

            timing_info["Rendering"] += time.time() - t0

            # -------------------------
            # WRITE
            # -------------------------
            video_writer.write(annotated_frame)

        frame_index += 1
        if frame_index % 30 == 0:
            print(
                f"Processed {frame_index} frames | "
                f"players on map: {len(transformed_players)} | "
                f"balls on map: {len(transformed_balls)}"
            )

    cap.release()
    video_writer.release()

    generate_final_report(start_time, frame_index, config, timing_info)


if __name__ == "__main__":
    main()































