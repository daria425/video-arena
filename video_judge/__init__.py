"""Video Generation Arena - LLM-as-a-Judge for video quality evaluation."""

from video_judge.arena import VideoGenArena
from video_judge.judge import BaseJudge, GeminiJudge, OpenAIJudge
from video_judge.decomposer import GeminiDecomposer, ClaudeDecomposer, OpenAIDecomposer, BaseDecomposer
from video_judge.orchestrator import VideoEvaluationOrchestrator
from video_judge.models import (
    VideoGenModelConfig,
    ArenaReport,
    ArenaRun,
    Report,
    JudgeEval,
)

__version__ = "0.1.0"

__all__ = [
    "VideoGenArena",
    "BaseJudge",
    "GeminiJudge",
    "OpenAIJudge",
    "VideoEvaluationOrchestrator",
    "VideoGenModelConfig",
    "ArenaReport",
    "ArenaRun",
    "Report",
    "JudgeEval",
    "BaseDecomposer",
    "GeminiDecomposer",
    "ClaudeDecomposer",
    "OpenAIDecomposer",
]
