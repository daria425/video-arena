from pathlib import Path
import requests
import tempfile
from moviepy.editor import VideoFileClip


def get_video(video_url: str) -> bytes:
    """Fetch video content from a URL."""
    response = requests.get(video_url)
    response.raise_for_status()
    return response.content


def time_video(video_content: bytes) -> float:
    """Get the duration of a video in seconds from video bytes (not used for now)."""
    # Write bytes to a temporary file
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
        temp_file.write(video_content)
        temp_path = temp_file.name

    try:
        # Load video and get duration
        with VideoFileClip(temp_path) as clip:
            duration = clip.duration
        return duration
    finally:
        # Clean up temp file
        Path(temp_path).unlink(missing_ok=True)


def download_video(video_content: bytes, output_path: str) -> str:

    with open(output_path, 'wb') as f:
        f.write(video_content)

    print(f"Video downloaded to: {output_path}")
    return output_path
