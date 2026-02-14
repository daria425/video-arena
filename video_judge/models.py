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


class Evidence(BaseModel):
    frame: int
    timestamp: float
    finding: str = Field(..., description="Description of the evidence found in the frame, relevant to overall evaluation reasoning")


class JudgeEval(BaseModel):
    score: float
    reason: str
    evidence: List[Evidence]


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
    rankings: List[str]


class ArenaRunFailure(BaseModel):
    model: str
    error: str
    error_type: str


class VideoGenModelConfig(BaseModel):
    provider: Literal["fal", "openai", "google"]
    model_id: str


class PromptDecomposition(BaseModel):
    entities: List[str]  # ["sleek sci-fi rocketship"]
    actions: List[str]   # ["launching vertically"]
    locations: List[str]  # ["lavender field"]
    time_of_day: Optional[str]  # "sunset"
    style_attributes: List[str]  # ["cinematic", "epic scale"]
