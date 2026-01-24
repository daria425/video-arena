from pydantic import BaseModel
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
