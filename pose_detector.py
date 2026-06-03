
import cv2
import mediapipe as mp
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict
from utils import PointSmoother
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    PoseLandmarker, PoseLandmarkerOptions,
    drawing_utils,
)
from mediapipe.tasks.python.vision.drawing_utils import DrawingSpec
from mediapipe.tasks.python.vision.pose_landmarker import (
    PoseLandmark, PoseLandmarksConnections,
)
from mediapipe.tasks.python.vision.core.vision_task_running_mode import (
    VisionTaskRunningMode,
)

KEY_LANDMARKS = {
    "left_shoulder":  PoseLandmark.LEFT_SHOULDER,
    "right_shoulder": PoseLandmark.RIGHT_SHOULDER,
    "left_hip":       PoseLandmark.LEFT_HIP,
    "right_hip":      PoseLandmark.RIGHT_HIP,
    "left_knee":      PoseLandmark.LEFT_KNEE,
    "right_knee":     PoseLandmark.RIGHT_KNEE,
    "left_ankle":     PoseLandmark.LEFT_ANKLE,
    "right_ankle":    PoseLandmark.RIGHT_ANKLE,
    "left_ear":       PoseLandmark.LEFT_EAR,
    "right_ear":      PoseLandmark.RIGHT_EAR,
}

MIN_VISIBILITY = 0.5

POSE_CONNECTIONS = PoseLandmarksConnections.POSE_LANDMARKS


@dataclass
class PoseKeypoints:
    landmarks: Dict[str, np.ndarray] = field(default_factory=dict)
    visibility: Dict[str, float] = field(default_factory=dict)
    valid: bool = False

    def get(self, name: str) -> Optional[np.ndarray]:
        return self.landmarks.get(name)

    def has(self, *names: str) -> bool:
        return all(n in self.landmarks for n in names)


class PoseDetector:

    def __init__(
        self,
        model_path: str = "pose_landmarker_lite.task",
        num_poses: int = 1,
        min_pose_detection_confidence: float = 0.5,
        min_pose_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        smoothing_window: int = 5,
    ):
        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionTaskRunningMode.IMAGE,
            num_poses=num_poses,
            min_pose_detection_confidence=min_pose_detection_confidence,
            min_pose_presence_confidence=min_pose_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.pose = PoseLandmarker.create_from_options(options)

        self._smoothers: Dict[str, PointSmoother] = {
            name: PointSmoother(window=smoothing_window, dims=3)
            for name in KEY_LANDMARKS
        }

        self._last_landmark_list = None
        self._last_connections = None

    def process(self, frame: np.ndarray) -> PoseKeypoints:
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.pose.detect(mp_image)

        kp = PoseKeypoints()

        if not result.pose_landmarks:
            self._last_landmark_list = None
            self._last_connections = None
            return kp

        landmark_list = result.pose_landmarks[0]
        self._last_landmark_list = landmark_list
        self._last_connections = POSE_CONNECTIONS

        drawing_utils.draw_landmarks(
            frame,
            landmark_list,
            connections=POSE_CONNECTIONS,
            landmark_drawing_spec=DrawingSpec(color=(255, 50, 255), thickness=2, circle_radius=3),
            connection_drawing_spec=DrawingSpec(color=(255, 255, 50), thickness=2),
        )

        for name, idx in KEY_LANDMARKS.items():
            lm = landmark_list[idx]
            vis = lm.visibility if lm.visibility is not None else 0.0
            kp.visibility[name] = float(vis)

            if vis >= MIN_VISIBILITY:
                raw = np.array([lm.x * w, lm.y * h, lm.z * w], dtype=float)
                smoothed = self._smoothers[name].update(raw)
                kp.landmarks[name] = smoothed

        critical = [
            "left_hip", "right_hip",
            "left_knee", "right_knee",
            "left_ankle", "right_ankle",
        ]
        kp.valid = all(n in kp.landmarks for n in critical)
        return kp

    def draw_cached(self, frame: np.ndarray) -> None:
        if self._last_landmark_list is None:
            return
        drawing_utils.draw_landmarks(
            frame,
            self._last_landmark_list,
            connections=self._last_connections,
            landmark_drawing_spec=DrawingSpec(color=(255, 50, 255), thickness=2, circle_radius=3),
            connection_drawing_spec=DrawingSpec(color=(255, 255, 50), thickness=2),
        )

    def reset_smoothers(self) -> None:
        for s in self._smoothers.values():
            s.reset()
        self._last_landmark_list = None
        self._last_connections = None

    def release(self) -> None:
        self.pose.close()
