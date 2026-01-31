from typing import Dict, Any, List, Optional, Literal
from pydantic import BaseModel, Field
from datetime import datetime


class VideoMetadata(BaseModel):
    generated_at: datetime
    prompt: str
    file_size: int
    seed: int


class VideoInfo(BaseModel):
    video_url: str
    saved_path: str
    metadata: VideoMetadata


class VideoFrame(BaseModel):
    idx: int
    image: bytes
    timestamp_s: float


class JudgeEval(BaseModel):
    score: float
    reason: str


class Report(BaseModel):
    input: Dict[str, Any]
    scores: Dict[str, float]
    verdict: str
    details: List[Dict]
    total_attempts: int = 1


class InterceptedVideoData(BaseModel):
    new_prompt: Optional[str] = None
    new_video_path: Optional[str] = None


class TemporalCorruptionConfig(BaseModel):
    """Config for temporal consistency breaking"""
    reverse_video: bool = True
    jumble_frames: bool = True
    max_frames: Optional[int] = None
    frame_interval: int = 1
    fps: Optional[int] = None


class InterceptorConfig(BaseModel):
    attribute: Literal["temporal", "alignment", "both"]
    temporal_corruption_config: Optional[TemporalCorruptionConfig] = None
