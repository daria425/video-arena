import cv2
from typing import List, Dict
import numpy as np
from video_judge.models import VideoFrame


def get_video_metadata(video_path: str) -> dict:
    """Get fps, duration, total_frames"""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_s = total_frames / fps if fps > 0 else 0
    cap.release()
    return {"fps": fps, "total_frames": total_frames, "duration_s": duration_s}


def sample_frames(video_path: str, num_frames: int = 8) -> List[VideoFrame]:
    """Sample exactly num_frames uniformly, including first/last"""
    meta = get_video_metadata(video_path)
    fps, total = meta["fps"], meta["total_frames"]
    if num_frames >= 2:
        middle = np.linspace(1, total - 2, num_frames - 2, dtype=int).tolist()
        indices = [0] + middle + [total - 1]
    else:
        indices = [0]

    cap = cv2.VideoCapture(video_path)
    frames = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            success, buffer = cv2.imencode('.png', frame_rgb)
            if success:
                frames.append(VideoFrame(
                    idx=idx,
                    timestamp_s=idx / fps,
                    image=buffer.tobytes()
                ))

    cap.release()
    return frames
