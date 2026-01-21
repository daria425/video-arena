import time
from datetime import datetime
from typing import Optional
import fal_client
from fal_client.client import Completed
from dotenv import load_dotenv
import requests
from pathlib import Path
load_dotenv()


def submit_request(prompt: str, model: Optional[str] = "fal-ai/bytedance/seedance/v1/pro/fast/text-to-video") -> str:
    handler = fal_client.submit(
        model,
        arguments={
            "prompt": prompt
        },
    )
    request_id = handler.request_id
    return request_id


def fetch_status(request_id: str, model: str) -> str:
    status = fal_client.status(model, request_id, with_logs=True)
    return status


def get_result(request_id: str, model: str):
    while True:
        status = fetch_status(
            request_id, model)
        print(f"Current status: {status}")
        if isinstance(status, Completed):
            result = fal_client.result(
                model, request_id)
            return result
        time.sleep(1)


def download_video(video_url: str, output_path: str) -> str:
    """Download video from URL to local path."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    response = requests.get(video_url, stream=True)
    response.raise_for_status()

    with open(output_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    print(f"Video downloaded to: {output_path}")
    return output_path


if __name__ == "__main__":
    download_path = f"./output/generated_video_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
    model_name = "fal-ai/bytedance/seedance/v1/pro/fast/text-to-video"
    prompt = """A sleek sci-fi rocketship launching vertically from the center of a vast lavender field at sunset. 
Endless rows of blooming purple lavender stretch toward the horizon, gently swaying from the rocket’s exhaust. 
The sky is filled with soft purple and pink clouds, glowing with warm golden sunset light. 
The rocket emits a bright white-violet flame and glowing thrusters, creating swirling dust and petals near the ground. 
Cinematic wide shot, epic scale, fantasy sci-fi atmosphere, soft volumetric lighting, shallow haze near the horizon, high detail, smooth motion, dramatic yet serene mood."""
    request_id = submit_request(prompt, model_name)
    print(f"Submitted request with ID: {request_id}")

    result = get_result(request_id, model_name)
    print("Video generation completed", result)
    video_url = result["video"]["url"]

    # Download the video
    local_path = download_video(video_url, download_path)
    print(f"Video saved at: {local_path}")
