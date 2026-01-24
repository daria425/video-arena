import io
from config.genai_client import client as google_client
import requests
from google.genai import types


def get_video(video_url: str) -> bytes:
    """Fetch video content from a URL."""
    response = requests.get(video_url, timeout=30)
    response.raise_for_status()
    return response.content


def download_video(video_content: bytes, output_path: str) -> str:

    with open(output_path, 'wb') as f:
        f.write(video_content)

    print(f"Video downloaded to: {output_path}")
    return output_path


def create_image_input(image_bytes: bytes):
    max_bytes = 15 * 1024 * 1024  # 15 MB
    size = len(image_bytes or b"")
    if size > max_bytes:
        image_file = google_client.files.upload(file=io.BytesIO(image_bytes))
        return image_file
    else:
        image_file = types.Part.from_bytes(
            data=image_bytes, mime_type="image/jpeg")
    return image_file
