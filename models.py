from typing import Dict, Any, List
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
