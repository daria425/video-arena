from pydantic import BaseModel
from datetime import datetime


class VideoMetadata(BaseModel):
    generated_at: datetime
    prompt: str
    file_size: int
    seed: int
    # duration_seconds: float


class VideoInfo(BaseModel):
    video_url: str
    saved_path: str
    metadata: VideoMetadata
