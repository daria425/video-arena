from typing import Dict, Any, List, Optional, Literal
from pydantic import BaseModel, Field
from datetime import datetime


class VideoMetadata(BaseModel):
    generated_at: datetime
    prompt: str
    file_size: int
    seed: Optional[int] = None


class VideoInfo(BaseModel):
    video_url: Optional[str] = None
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
    details: List[Dict]


class ArenaRun(BaseModel):
    model: str
    report: Report


class ArenaReport(BaseModel):
    prompt: str
    results: List[ArenaRun]
    winner: str


class ArenaRunFailure(BaseModel):
    model: str
    error: str
    error_type: str


class VideoGenModelConfig(BaseModel):
    provider: Literal["fal", "openai"]
    model_id: str
