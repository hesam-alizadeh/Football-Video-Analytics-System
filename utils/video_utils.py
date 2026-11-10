

# utils/video_utils.py

from pathlib import Path
import cv2


def read_video(video_path: Path):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()

    return frames, fps, width, height


def save_video(output_path: Path, frames, fps, width, height):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (width, height)
    )

    for frame in frames:
        writer.write(frame)

    writer.release()



def blend_frames(frame_a, frame_b, alpha=0.5):
    """
    Blends two frames together using given alpha value.
    """
    return cv2.addWeighted(frame_a, alpha, frame_b, 1 - alpha, 0)



# ------------------------------------------------------------
# NEW: Simple VideoWriter class for compatibility with main.py
# ------------------------------------------------------------
class VideoWriter:
    """
    Simple wrapper class around OpenCV's VideoWriter
    so main.py can use VideoWriter(...) without changes.
    """

    def __init__(self, output_path: Path, fps: float, width: int, height: int):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(
            str(output_path),
            fourcc,
            fps,
            (width, height)
        )

        if not self.writer.isOpened():
            raise ValueError(f"Cannot open VideoWriter for: {output_path}")

    def write(self, frame):
        self.writer.write(frame)

    def release(self):
        self.writer.release()

    # Optional: enable usage with "with VideoWriter(...)"
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()



