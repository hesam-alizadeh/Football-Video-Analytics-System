

from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any, List

@dataclass
class ProjectConfig:

    BALL_CLASS_ID = 0
    GOALKEEPER_CLASS_ID = 1
    PLAYER_CLASS_ID = 2
    REFEREE_CLASS_ID = 3



    
    # =========================
    # Project paths
    # =========================
    PROJECT_ROOT = Path(__file__).resolve().parent
    MODEL_DIR = PROJECT_ROOT / "models"
    DATA_DIR = PROJECT_ROOT / "data"
    OUTPUT_DIR = PROJECT_ROOT / "outputs"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # =========================
    # Models (use best.pt)
    # =========================
    # Players detection model
    PLAYERS_MODEL_PATH = MODEL_DIR / "players" / "best.pt"
    # Field / Pitch keypoints (or segmentation) model
    FIELD_MODEL_PATH = MODEL_DIR / "field" / "best.pt"
    # Default detector used by Tracker
    YOLO_MODEL_PATH = PLAYERS_MODEL_PATH

    # =========================
    # Detection thresholds
    # =========================
    DETECTION_CONF_THRESHOLD = 0.25
    DETECTION_IOU_THRESHOLD = 0.45

    # =========================
    # Tracking (ByteTrack)
    # =========================
    TRACKING_FRAME_STRIDE = 1  # 1 = process every frame
    BYTE_TRACK_TRACK_BUFFER = 30
    BYTE_TRACK_MATCH_THRESHOLD = 0.8


    # مسیرها
    input_video: str
    output_video: str
    
    # تنظیمات کش
    use_cache: bool = True
    cache_dir: str = "outputs/cache"
    
    # پارامترهای دتکتور و ترکر
    detection_threshold: float = 0.25
    detector_params: Dict[str, Any] = field(default_factory=lambda: {
        "model_path": "models/players/best.pt",
        "conf": 0.25
    })
    
    tracker_params: Dict[str, Any] = field(default_factory=lambda: {
        "track_buffer": 30,
        "match_thresh": 0.8
    })
    
    team_assignment_params: Dict[str, Any] = field(default_factory=lambda: {
        "method": "color_clustering"
    })
    
    renderer_params: Dict[str, Any] = field(default_factory=lambda: {
        "colors": {"player": (0, 255, 0), "referee": (0, 165, 255), "goalkeeper": (255, 0, 0)}
    })
    
    # تنظیمات زمین و هوموگرافی
    pitch_dimensions: Dict[str, float] = field(default_factory=lambda: {
        "length": 105.0, 
        "width": 68.0
    })
    
    # نقطه های متناظر (باید از قبل استخراج شده باشند - اینجا مقادیر نمونه)
    # این نقاط را با نقاط واقعی ویدئوی خود جایگزین کنید
    image_points: List[List[float]] = field(default_factory=lambda: [
        [100, 200], [500, 200], [100, 800], [500, 800]
    ])
    pitch_points: List[List[float]] = field(default_factory=lambda: [
        [0, 0], [105, 0], [0, 68], [105, 68]
    ])

def load_config() -> ProjectConfig:
    """بارگذاری تنظیمات پروژه."""
    # در اینجا می‌توانید مسیرها را از متغیرهای قدیمی هم بخوانید
    return ProjectConfig(
        input_video="08fd33_4.mp4",
        output_video="outputs/annotated_output.mp4"
    )







