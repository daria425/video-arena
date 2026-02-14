import time
import uuid
from google.genai import types
from datetime import datetime
from typing import Optional
import fal_client
from fal_client.client import Completed
from dotenv import load_dotenv
from video_judge.utils.file_utils import download_video, get_video
from video_judge.ai_api_client import openai_client
from video_judge.config.logger import logger
from video_judge.models import VideoInfo
from video_judge.ai_api_client import openai_client, google_client
from abc import ABC, abstractmethod
load_dotenv()


class BaseVideoGenerator(ABC):
    def __init__(self, model: str):
        self.model = model
        self._request_id = None

    @abstractmethod
    def run_video_gen(self, prompt: str, download_path: Optional[str] = None):
        pass


class FalVideoGenerator(BaseVideoGenerator):
    def __init__(self, model: Optional[str] = "fal-ai/bytedance/seedance/v1/pro/fast/text-to-video"):
        super().__init__(model)
        self._request_id = None

    def submit_request(self, prompt: str):
        handler = fal_client.submit(
            self.model,
            arguments={
                "prompt": prompt
            },
        )
        request_id = handler.request_id
        self._request_id = request_id

    def fetch_status(self) -> str:
        status = fal_client.status(
            self.model, self._request_id, with_logs=True)
        return status

    def get_result(self, timeout: int = 600):
        start_time = time.time()
        while True:
            if time.time() - start_time > timeout:
                raise TimeoutError(
                    f"{self.__class__.__name__}: Video generation failed after {timeout} seconds")
            status = self.fetch_status()
            logger.info(f"Current status: {status}")
            if isinstance(status, Completed):
                result = fal_client.result(
                    self.model, self._request_id)
                return result
            time.sleep(1)

    def run_video_gen(self, prompt: str, download_path: Optional[str] = None) -> VideoInfo:
        if download_path is None:
            unique_id = uuid.uuid4().hex[:8]
            download_path = f"./output/videos/generated_video_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{unique_id}.mp4"
        self.submit_request(prompt)
        result = self.get_result()
        logger.info("Video generation completed")
        video_url = result["video"]["url"]
        file_size = result["video"]["file_size"]
        generated_at = datetime.now()
        seed = result.get("seed", "")
        video_content = get_video(video_url)
        local_path = download_video(video_content, download_path)
        if seed == "":  # some return empty string instead of omitting the field when seed is not provided, handle both cases
            logger.warning("No seed returned from video generation API")
            seed = None
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


class OpenAIVideoGenerator(BaseVideoGenerator):
    def __init__(self, model: str = "sora-2"):
        super().__init__(model)
        self._request_id = None

    def submit_request(self, prompt: str):
        video_request = openai_client.client.videos.create(
            model=self.model, prompt=prompt)
        self._request_id = video_request.id

    def fetch_status(self):
        response = openai_client.client.videos.retrieve(self._request_id)
        return response

    def get_result(self, timeout: int = 900):
        start_time = time.time()
        while True:
            if time.time() - start_time > timeout:
                raise TimeoutError(
                    f"{self.__class__.__name__}: Video generation failed after {timeout} seconds")
            job = self.fetch_status()
            logger.info(
                f"Current status: {job.status}, progress: {job.progress}")
            if job.status == "completed":
                content = openai_client.client.videos.download_content(
                    self._request_id)
                video_content = content.read()
                return {
                    "video": {
                        "content": video_content,
                        "file_size": len(video_content),
                    },
                    "seed": None
                }
            elif job.status == "failed":
                raise RuntimeError(
                    f"{self.__class__.__name__}: Video generation failed with error {job.error}")
            time.sleep(5)

    def run_video_gen(self, prompt: str, download_path: Optional[str] = None) -> VideoInfo:
        if download_path is None:
            unique_id = uuid.uuid4().hex[:8]
            download_path = f"./output/videos/generated_video_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{unique_id}.mp4"

        self.submit_request(prompt)
        result = self.get_result()
        logger.info("Video generation completed")

        video_content = result["video"]["content"]
        local_path = download_video(video_content, download_path)

        return VideoInfo(
            saved_path=local_path,
            metadata={
                "generated_at": datetime.now(),
                "prompt": prompt,
                "file_size": result["video"]["file_size"],
            }
        )


class GoogleVideoGenerator(BaseVideoGenerator):
    def __init__(self, model: str = "veo-3.1-fast-generate-preview"):
        super().__init__(model)
        self._operation = None

    def submit_request(self, prompt: str):
        operation = google_client.client.models.generate_videos(
            model=self.model,
            prompt=prompt
        )
        self._operation = operation

    def fetch_status(self) -> types.GenerateVideosOperation:
        operation = google_client.client.operations.get(self._operation)
        self._operation = operation

    def get_result(self, timeout: int = 900):
        start_time = time.time()
        while True:
            if time.time() - start_time > timeout:
                raise TimeoutError(
                    f"{self.__class__.__name__}: Video generation failed after {timeout} seconds")

            time.sleep(5)

            self.fetch_status()
            logger.info(
                f"Completion status:{self._operation.done is not None}")
            if self._operation.done:
                video = self._operation.response.generated_videos[0]
                video_bytes = google_client.client.files.download(
                    file=video.video)
                return {
                    "video": {
                        "content": video_bytes,
                        "file_size": len(video_bytes),
                    },
                    "seed": None
                }
            elif self._operation.error:
                raise RuntimeError(
                    f"{self.__class__.__name__}: Video generation failed with error {self._operation.error}")

    def run_video_gen(self, prompt: str, download_path: Optional[str] = None) -> VideoInfo:
        if download_path is None:
            unique_id = uuid.uuid4().hex[:8]
            download_path = f"./output/videos/generated_video_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{unique_id}.mp4"

        self.submit_request(prompt)
        result = self.get_result()
        logger.info("Video generation completed")

        video_content = result["video"]["content"]
        local_path = download_video(video_content, download_path)

        return VideoInfo(
            saved_path=local_path,
            metadata={
                "generated_at": datetime.now(),
                "prompt": prompt,
                "file_size": result["video"]["file_size"],
            }
        )
