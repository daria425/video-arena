import io
import subprocess
import tempfile
import shutil
from video_judge.ai_api_client import google_client
from video_judge.config.logger import logger
import requests
import glob
import random
from google.genai import types


def jumble_video(video_path, max_frames=None, frame_interval=1, fps=None):
    """
    Jumble video frames randomly.

    Args:
        video_path: Path to the video file
        max_frames: Maximum number of frames to use (None for all)
        frame_interval: Extract every nth frame (default 1 for all frames)
        fps: Frame rate to extract at (None to extract all frames at original rate)
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # Extract frames
        ffmpeg_cmd = ['ffmpeg', '-i', video_path]

        # Add fps filter if specified
        if fps:
            ffmpeg_cmd.extend(['-vf', f'fps={fps}'])

        ffmpeg_cmd.append(f'{tmpdir}/frame_%04d.png')

        subprocess.run(ffmpeg_cmd, check=True, capture_output=True)

        frames = sorted(glob.glob(f'{tmpdir}/frame_*.png'))

        # Apply frame interval sampling
        if frame_interval > 1:
            frames = frames[::frame_interval]

        # Limit to max_frames if specified
        if max_frames and len(frames) > max_frames:
            frames = frames[:max_frames]

        random.shuffle(frames)

        # Create file list for concatenation
        with open(f'{tmpdir}/filelist.txt', 'w') as f:
            for frame in frames:
                f.write(f"file '{frame}'\nduration 0.042\n")  # ~24fps

        # Reassemble
        temp_out = video_path + ".tmp.mp4"
        subprocess.run([
            'ffmpeg', '-f', 'concat', '-safe', '0',
            '-i', f'{tmpdir}/filelist.txt',
            '-vf', 'fps=24',
            '-y', temp_out
        ], check=True, capture_output=True)

        shutil.move(temp_out, video_path)
        logger.info(f"Shuffled {len(frames)} frames using ffmpeg")

    return video_path


def reverse_video(video_path: str):
    temp_out = video_path + ".tmp.mp4"
    cmd = [
        'ffmpeg', '-i', video_path,
        '-vf', 'reverse',
        '-af', 'areverse',
        '-y',  # Overwrite
        temp_out
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    shutil.move(temp_out, video_path)
    logger.info("Reversed video using ffmpeg")
    return video_path


def get_video(video_url: str) -> bytes:
    """Fetch video content from a URL."""
    response = requests.get(video_url, timeout=30)
    response.raise_for_status()
    return response.content


def download_video(video_content: bytes, output_path: str) -> str:

    with open(output_path, 'wb') as f:
        f.write(video_content)

    logger.info(f"Video downloaded to: {output_path}")
    return output_path


def create_image_input(image_bytes: bytes):
    max_bytes = 15 * 1024 * 1024  # 15 MB
    size = len(image_bytes or b"")
    if size > max_bytes:
        image_file = google_client.client.files.upload(
            file=io.BytesIO(image_bytes))
        return image_file
    else:
        image_file = types.Part.from_bytes(
            data=image_bytes, mime_type="image/jpeg")
    return image_file


if __name__ == "__main__":
    vid_path = "./output/videos/test_2.mp4"
    reverse_video(vid_path)
