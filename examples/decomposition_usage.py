"""Example: Using prompt decomposition for structured evaluation.

This shows how to pass structured criteria to judges for more rigorous evaluation.
"""

from video_judge import (
    VideoEvaluationOrchestrator,
    OpenAIJudge,
    VideoGenModelConfig,
    VideoGenArena,
)
from video_judge.models import PromptDecomposition
from video_judge.config.logger import setup_default_logging

setup_default_logging(level=20)

# Full generation prompt
prompt = "A sleek sci-fi rocketship launching vertically from the center of a vast lavender field at sunset"

# Decompose prompt into structured criteria
# In production, you'd use a decomposer agent to generate this automatically
decomposition = PromptDecomposition(
    entities=["sleek sci-fi rocketship"],
    actions=["launching vertically"],
    locations=["vast lavender field"],
    time_of_day="sunset",
    style_attributes=["cinematic", "epic scale"],
)

# Create orchestrator with decomposition
orchestrator = VideoEvaluationOrchestrator(
    video_gen_prompt=prompt,
    prompt_decomposition=decomposition,  # NEW: structured criteria
    existing_video_path="./output/videos/your_video.mp4",  # or None to generate
)

judge = OpenAIJudge()

# What the judge sees in user_prompts:
# [
#   "Frame 0 at 0.00s",
#   "Frame 1 at 0.50s",
#   ...
#   "Original prompt: A sleek sci-fi rocketship launching...",
#   """Key criteria - ensure these align with the generated video:
#     - Entity: sleek sci-fi rocketship
#     - Action: launching vertically
#     - Location: vast lavender field
#     - Time: sunset
#     - Style: cinematic
#     - Style: epic scale"""
# ]

# Run evaluation (same API, enhanced prompting)
report = orchestrator.run(judge=judge, video_generator=None)  # None when using existing video

print(f"Overall score: {report.scores['overall']:.3f}")
print(f"Prompt alignment: {report.scores['prompt_alignment']:.3f}")
print(f"Temporal consistency: {report.scores['temporal_consistency']:.3f}")

# Also works in arena mode
configs = [
    VideoGenModelConfig(provider="openai", model_id="sora-2"),
    VideoGenModelConfig(provider="google", model_id="veo-3.1-fast-generate-preview"),
]

arena = VideoGenArena(model_configs=configs, judge=judge)
arena_result = arena.fight(orchestrator)

print(f"\nWinner: {arena_result.winner}")
for result in arena_result.results:
    print(f"  {result.model}: {result.report.scores['overall']:.3f}")
