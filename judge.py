from abc import abstractmethod, ABC
from typing import List
from utils.llm_utils import _call_gemini_with_image_list
from models import JudgeEval


class BaseJudge(ABC):
    """Base class for video judges."""

    @abstractmethod
    def evaluate(self, images: List[bytes], user_prompts: List[str], system_prompt: str, **kwargs) -> JudgeEval:
        """Evaluate images using prompts."""
        pass


class GeminiJudge(BaseJudge):
    """Gemini-based judge implementation."""

    def evaluate(self, images: List[bytes], user_prompts: List[str], system_prompt: str, **kwargs) -> JudgeEval:
        """Evaluate images using Gemini API."""
        response: JudgeEval = _call_gemini_with_image_list(
            image_bytes_list=images,
            user_prompt_list=user_prompts,
            system_instruction=system_prompt,
            response_schema=JudgeEval,
            ** kwargs
        )
        return response
