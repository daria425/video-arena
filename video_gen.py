import time
from datetime import datetime
from typing import Optional
import fal_client
from fal_client.client import Completed
from dotenv import load_dotenv
from utils.file_utils import download_video, get_video
from config.logger import logger
from models import VideoInfo
from abc import ABC, abstractmethod
load_dotenv()


class BaseVideoGenerator(ABC):
    @abstractmethod
    def run_video_gen(self, prompt: str, download_path: str = f"./output/videos/generated_video_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"):
        pass


class FalVideoGenerator(BaseVideoGenerator):
    def __init__(self, model: Optional[str] = "fal-ai/bytedance/seedance/v1/pro/fast/text-to-video"):
        self.model = model
        self.request_id = None

    def submit_request(self, prompt: str) -> str:
        handler = fal_client.submit(
            self.model,
            arguments={
                "prompt": prompt
            },
        )
        request_id = handler.request_id
        self.request_id = request_id

    def fetch_status(self) -> str:
        status = fal_client.status(self.model, self.request_id, with_logs=True)
        return status

    def get_result(self, timeout: int = 600):
        start_time = time.time()
        while True:
            if time.time() - start_time > timeout:
                raise TimeoutError(
                    f"Video generation failed after {timeout} seconds")
            status = self.fetch_status()
            logger.info(f"Current status: {status}")
            if isinstance(status, Completed):
                result = fal_client.result(
                    self.model, self.request_id)
                return result
            time.sleep(1)

    def run_video_gen(self, prompt: str, download_path: str = f"./output/videos/generated_video_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4") -> VideoInfo:
        self.submit_request(prompt)
        result = self.get_result()
        logger.info("Video generation completed")
        video_url = result["video"]["url"]
        file_size = result["video"]["file_size"]
        generated_at = datetime.now()
        seed = result.get("seed", "")
        video_content = get_video(video_url)
        local_path = download_video(video_content, download_path)
        info = VideoInfo(
            video_url=video_url,
            saved_path=local_path,
            metadata={
                "generated_at": generated_at,
                "prompt": prompt,
                "file_size": file_size,
                "seed": seed}

        )
        return info
