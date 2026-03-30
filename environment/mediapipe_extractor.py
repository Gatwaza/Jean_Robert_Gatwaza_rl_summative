"""
MediaPipe CPR Landmark Extractor
==================================
Extracts 17-keypoint pose landmarks from three sources (in priority order):

  1. CPR-Coach video (downloaded from HuggingFace or local file)
  2. Live webcam stream
  3. Synthetic CPR pose generator (always available as fallback)

Output: numpy array of shape (N_frames, 51) — 17 keypoints × (x, y, visibility)
Each value normalised to [0, 1].

Usage:
    extractor = LandmarkExtractor(source="video", path="cpr_sample.mp4")
    for frame_landmarks in extractor.stream():
        obs = env.set_landmarks(frame_landmarks)

    extractor = LandmarkExtractor(source="webcam")
    extractor = LandmarkExtractor(source="synthetic")
"""

import os
import sys
import time
import numpy as np
import logging
from typing import Generator, Optional

log = logging.getLogger("MediaPipe")

# ── MediaPipe availability check ────────────────────────────────────────────
# mp.solutions is accessed lazily inside methods — on Apple Silicon with
# newer mediapipe builds, solutions isn't available at import time.
try:
    import mediapipe as mp
    import cv2
    MP_AVAILABLE = True
except ImportError:
    mp = None
    cv2 = None
    MP_AVAILABLE = False
    log.warning("mediapipe / opencv not installed. Using synthetic fallback only.")

def _get_mp_pose():
    """Lazy accessor — safe on all mediapipe versions and platforms."""
    if not MP_AVAILABLE:
        return None, None
    try:
        return mp.solutions.pose, mp.solutions.drawing_utils
    except AttributeError:
        import mediapipe.python.solutions.pose as _pose
        import mediapipe.python.solutions.drawing_utils as _draw
        return _pose, _draw

# MediaPipe uses 33 landmarks; we map the first 17 (upper body + hips)
# matching the COCO-17 layout used in the environment
MP_TO_COCO17 = list(range(17))  # MediaPipe indices 0-16 = nose → right wrist


class LandmarkExtractor:
    """
    Unified landmark extraction interface.

    Parameters
    ----------
    source : str
        'video'     — from a video file (CPR-Coach or any MP4/AVI)
        'webcam'    — from the default camera device
        'synthetic' — procedurally generated supine CPR poses (always works)
    path : str, optional
        Path to video file (only for source='video').
    loop : bool
        Whether to loop the video (default True).
    noise_std : float
        Gaussian noise added to synthetic landmarks (default 0.015).
    """

    def __init__(
        self,
        source: str = "synthetic",
        path: Optional[str] = None,
        loop: bool = True,
        noise_std: float = 0.015,
    ):
        self.source = source
        self.path = path
        self.loop = loop
        self.noise_std = noise_std
        self._cap = None
        self._pose = None
        self._frame_cache = []  # pre-extracted frames for video replay
        self._cache_idx = 0

        if source in ("video", "webcam") and not MP_AVAILABLE:
            log.warning(
                f"Source '{source}' requested but mediapipe is not installed. "
                "Falling back to synthetic landmarks."
            )
            self.source = "synthetic"

    # ── Public API ───────────────────────────────────────────────────────────

    def stream(self) -> Generator[np.ndarray, None, None]:
        """
        Yields landmark arrays of shape (51,) indefinitely.
        Loops video/synthetic; webcam runs until camera disconnects.
        """
        if self.source == "synthetic":
            yield from self._synthetic_stream()
        elif self.source == "video":
            yield from self._video_stream()
        elif self.source == "webcam":
            yield from self._webcam_stream()

    def extract_all_frames(self, max_frames: int = 500) -> np.ndarray:
        """
        Extract up to max_frames from the source.
        Returns array of shape (N, 51).
        """
        frames = []
        for lm in self.stream():
            frames.append(lm)
            if len(frames) >= max_frames:
                break
        return np.array(frames, dtype=np.float32)

    def close(self):
        if self._cap is not None:
            self._cap.release()
        if self._pose is not None:
            self._pose.close()

    # ── Internal streams ─────────────────────────────────────────────────────

    def _synthetic_stream(self) -> Generator[np.ndarray, None, None]:
        """
        Generates physically plausible supine CPR poses with smooth animation.
        Simulates: patient lying still → compressions → head tilt → recovery.
        """
        rng = np.random.default_rng(42)
        t = 0.0

        # Base supine pose (x, y, visibility) for 17 keypoints
        base = np.array([
            # nose
            0.500, 0.820, 0.95,
            # left eye chain
            0.530, 0.800, 0.90,
            0.560, 0.790, 0.88,
            0.580, 0.780, 0.85,
            # right eye chain
            0.470, 0.800, 0.90,
            0.440, 0.790, 0.88,
            0.420, 0.780, 0.85,
            # left ear, right ear
            0.600, 0.810, 0.80,
            0.400, 0.810, 0.80,
            # mouth left, right
            0.540, 0.840, 0.90,
            0.460, 0.840, 0.90,
            # shoulders
            0.620, 0.720, 0.95,
            0.380, 0.720, 0.95,
            # elbows
            0.680, 0.600, 0.90,
            0.320, 0.600, 0.90,
            # wrists
            0.650, 0.500, 0.85,
            0.350, 0.500, 0.85,
        ], dtype=np.float32)

        while True:
            pose = base.copy()

            # --- CPR animation cycle (period ~5 seconds at 10Hz) ---
            # Phase 1: compression (wrists move toward chest)
            comp_phase = 0.5 * (1 + np.sin(2 * np.pi * t / 0.5))
            comp_depth = 0.08 * comp_phase
            pose[15 * 3 + 1] = base[15 * 3 + 1] - comp_depth  # left wrist y
            pose[16 * 3 + 1] = base[16 * 3 + 1] - comp_depth  # right wrist y

            # Phase 2: gentle head tilt oscillation
            head_tilt = 0.02 * np.sin(2 * np.pi * t / 8.0)
            pose[0 * 3 + 1] = base[0 * 3 + 1] + head_tilt  # nose y

            # Small natural body sway
            sway = 0.005 * np.sin(2 * np.pi * t / 3.0)
            for i in range(17):
                pose[i * 3] = np.clip(base[i * 3] + sway, 0.0, 1.0)

            # Sensor noise
            noise = rng.normal(0, self.noise_std, size=pose.shape).astype(np.float32)
            yield np.clip(pose + noise, 0.0, 1.0)

            t += 0.1  # 10 Hz

    def _video_stream(self) -> Generator[np.ndarray, None, None]:
        """Extract landmarks from a video file using MediaPipe Pose."""
        if not os.path.exists(self.path):
            log.error(f"Video not found: {self.path}. Falling back to synthetic.")
            yield from self._synthetic_stream()
            return

        # Use cache if already extracted
        if self._frame_cache:
            yield from self._replay_cache()
            return

        log.info(f"Extracting landmarks from video: {self.path}")
        self._cap = cv2.VideoCapture(self.path)
        _mp_pose_mod, _mp_draw_mod = _get_mp_pose()
        self._pose = _mp_pose_mod.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        prev_landmarks = None
        frames_extracted = 0

        while True:
            ret, frame = self._cap.read()
            if not ret:
                if self.loop and frames_extracted > 0:
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    log.info("Video looped")
                    yield from self._replay_cache()
                    return
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self._pose.process(rgb)

            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark
                landmarks = np.array([
                    [lm[i].x, lm[i].y, lm[i].visibility]
                    for i in MP_TO_COCO17
                ], dtype=np.float32).flatten()
                landmarks = np.clip(landmarks, 0.0, 1.0)
                prev_landmarks = landmarks
            else:
                # Use previous frame if detection fails
                landmarks = prev_landmarks if prev_landmarks is not None else \
                            next(self._synthetic_stream())

            self._frame_cache.append(landmarks.copy())
            frames_extracted += 1
            yield landmarks

        self.close()
        log.info(f"Extracted {frames_extracted} frames from video")

    def _webcam_stream(self) -> Generator[np.ndarray, None, None]:
        """Real-time landmark extraction from webcam."""
        self._cap = cv2.VideoCapture(0)
        if not self._cap.isOpened():
            log.error("Cannot open webcam. Falling back to synthetic.")
            yield from self._synthetic_stream()
            return

        _mp_pose_mod, _mp_draw_mod = _get_mp_pose()
        self._pose = _mp_pose_mod.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        log.info("Webcam stream started. Press Q to quit.")
        prev_landmarks = None

        while True:
            ret, frame = self._cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self._pose.process(rgb)

            # Draw skeleton on frame (optional display)
            if results.pose_landmarks:
                _mp_draw_mod.draw_landmarks(frame, results.pose_landmarks,
                                        _mp_pose_mod.POSE_CONNECTIONS)
                lm = results.pose_landmarks.landmark
                landmarks = np.array([
                    [lm[i].x, lm[i].y, lm[i].visibility]
                    for i in MP_TO_COCO17
                ], dtype=np.float32).flatten()
                landmarks = np.clip(landmarks, 0.0, 1.0)
                prev_landmarks = landmarks
            else:
                landmarks = prev_landmarks if prev_landmarks is not None else \
                            next(self._synthetic_stream())

            # Show camera feed with skeleton
            cv2.imshow("CPR — Webcam Pose Extraction (Q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            yield landmarks

        self.close()
        cv2.destroyAllWindows()

    def _replay_cache(self) -> Generator[np.ndarray, None, None]:
        """Replay cached frames, looping indefinitely."""
        while True:
            for lm in self._frame_cache:
                yield lm.copy()
            if not self.loop:
                break


# ── CPR-Coach Dataset Downloader ────────────────────────────────────────────

def download_cpr_coach_sample(output_dir: str = "data/cpr_coach") -> Optional[str]:
    """
    Attempt to download a CPR-Coach video sample from HuggingFace.

    Returns the path to the downloaded video, or None if unavailable.

    The CPR-Coach dataset (CVPR 2024, Wang et al.) contains 4,544 CPR
    procedure videos across 14 action classes and 74 composite error classes.
    It requires HuggingFace authentication for gated access.

    Fallback: The extractor will use synthetic CPR poses if download fails.
    """
    os.makedirs(output_dir, exist_ok=True)
    video_path = os.path.join(output_dir, "cpr_sample.mp4")

    if os.path.exists(video_path):
        log.info(f"CPR-Coach sample already exists: {video_path}")
        return video_path

    log.info("Attempting to download CPR-Coach sample from HuggingFace...")

    try:
        from huggingface_hub import hf_hub_download, list_repo_files
        import requests

        # The dataset is at ShunliWang/CPR-Coach on HuggingFace
        # It may require authentication (set HF_TOKEN env var)
        hf_token = os.environ.get("HF_TOKEN", None)

        files = list(list_repo_files(
            "ShunliWang/CPR-Coach",
            repo_type="dataset",
            token=hf_token,
        ))

        # Find first video file
        video_files = [f for f in files if f.endswith((".mp4", ".avi"))]
        if not video_files:
            log.warning("No video files found in CPR-Coach dataset")
            return None

        target = video_files[0]
        log.info(f"Downloading: {target}")

        downloaded = hf_hub_download(
            repo_id="ShunliWang/CPR-Coach",
            filename=target,
            repo_type="dataset",
            token=hf_token,
            local_dir=output_dir,
        )

        import shutil
        shutil.copy(downloaded, video_path)
        log.info(f"CPR-Coach sample saved to: {video_path}")
        return video_path

    except ImportError:
        log.warning("huggingface_hub not installed. Run: pip install huggingface_hub")
    except Exception as e:
        log.warning(
            f"CPR-Coach download failed: {e}\n"
            "Options:\n"
            "  1. Set HF_TOKEN environment variable for gated access\n"
            "  2. Manually download from https://huggingface.co/datasets/ShunliWang/CPR-Coach\n"
            "  3. System will use synthetic CPR poses as fallback"
        )

    return None


def get_best_extractor(prefer_video: bool = True) -> LandmarkExtractor:
    """
    Factory: returns the best available extractor in this priority order:
      CPR-Coach video → local video file → webcam → synthetic
    """
    if prefer_video:
        # Try CPR-Coach
        video_path = download_cpr_coach_sample()
        if video_path:
            log.info("Using CPR-Coach video for landmark extraction")
            return LandmarkExtractor(source="video", path=video_path)

        # Try any local video
        for candidate in ["data/cpr.mp4", "data/cpr.avi", "cpr_sample.mp4"]:
            if os.path.exists(candidate):
                log.info(f"Using local video: {candidate}")
                return LandmarkExtractor(source="video", path=candidate)

    log.info("Using synthetic CPR pose generator")
    return LandmarkExtractor(source="synthetic")