from typing import Optional
from utils.file_utils import jumble_video, reverse_video
from models import InterceptedVideoData, InterceptorConfig


class VideoInterceptor:
    def __init__(self, video_path: str, original_prompt: str, interceptor_config: Optional[InterceptorConfig] = None):
        self.video_path = video_path
        self.config = interceptor_config
        self.video_prompt = original_prompt

    def break_temporal_consistency(self):
        reversed_video = reverse_video(self.video_path)
        broken_video = jumble_video(reversed_video)
        return broken_video

    def break_brompt_alignment(self):
        # TO-DO replace with llm logic later
        return "Some different prompt"

    def intercept(self) -> InterceptedVideoData:
        if not self.config or self.config.attribute == "both":
            # use both by default
            return InterceptedVideoData(
                new_prompt=self.break_brompt_alignment(),
                new_video_path=self.break_temporal_consistency()
            )
        elif self.config.attribute == "alignment":
            return InterceptedVideoData(
                new_prompt=self.break_brompt_alignment(),
                new_video_path=self.video_path  # original
            )
        else:
            return InterceptedVideoData(
                new_prompt=self.video_prompt,  # original
                new_video_path=self.break_temporal_consistency()
            )
