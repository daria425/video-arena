from abc import ABC, abstractmethod
from typing import Optional
from video_judge.input_builders import (
    build_claude_input_with_text,
    build_gemini_input_with_text,
    build_openai_input_with_text,
)
from video_judge.utils.format import format_prompt
from video_judge.models import PromptDecomposition
from video_judge.config.logger import logger


class BaseDecomposer(ABC):
    """Base class for prompt decomposition using LLMs.

    Subclasses implement _call_api() to use provider-specific APIs.
    """
    default_model: str

    def decompose(self, user_prompt: str, model: Optional[str] = None) -> PromptDecomposition:
        """Decompose a video generation prompt into structured criteria.

        Args:
            user_prompt: The video generation prompt to decompose
            model: Optional model override (uses default_model if not specified)

        Returns:
            PromptDecomposition with entities, actions, locations, etc.
        """
        system_prompt = format_prompt("./prompts/decompose.txt")
        formatted_prompt = (
            "Decompose the following prompt into structured, verifiable criteria "
            "that judges can check against sampled video frames:\n\n"
            f"{user_prompt}"
        )
        response = self._call_api(
            user_prompt=formatted_prompt,
            system_instruction=system_prompt,
            model=model or self.default_model,
        )
        logger.info(f"Decomposition result: {response}")
        return response

    @abstractmethod
    def _call_api(
        self, user_prompt: str, system_instruction: str, model: str
    ) -> PromptDecomposition:
        """Call provider-specific API with structured output."""
        pass


class ClaudeDecomposer(BaseDecomposer):
    """Decompose prompts using Anthropic Claude."""
    default_model = "claude-sonnet-3-5"

    def _call_api(
        self, user_prompt: str, system_instruction: str, model: str
    ) -> PromptDecomposition:
        return build_claude_input_with_text(
            user_prompt=user_prompt,
            system_instruction=system_instruction,
            model=model,
            response_schema=PromptDecomposition,
        )


class GeminiDecomposer(BaseDecomposer):
    """Decompose prompts using Google Gemini."""
    default_model = "gemini-2.5-pro"

    def _call_api(
        self, user_prompt: str, system_instruction: str, model: str
    ) -> PromptDecomposition:
        return build_gemini_input_with_text(
            user_prompt=user_prompt,
            system_instruction=system_instruction,
            model=model,
            response_schema=PromptDecomposition,
        )


class OpenAIDecomposer(BaseDecomposer):
    """Decompose prompts using OpenAI."""
    default_model = "gpt-4o"

    def _call_api(
        self, user_prompt: str, system_instruction: str, model: str
    ) -> PromptDecomposition:
        return build_openai_input_with_text(
            user_prompt=user_prompt,
            system_instruction=system_instruction,
            model=model,
            response_schema=PromptDecomposition,
        )
