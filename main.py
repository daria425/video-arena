"""Example script showing programmatic usage of video-judge."""
import json
from video_judge import (
    VideoEvaluationOrchestrator,
    OpenAIJudge,
    VideoGenModelConfig,
    VideoGenArena,
    OpenAIDecomposer
)
from video_judge.config.logger import setup_default_logging
from datetime import datetime

setup_default_logging(level=20)
prompt = "A sleek sci-fi rocketship launching vertically from the center of a vast lavender field at sunset. Endless rows of blooming purple lavender stretch toward the horizon, gently swaying from the rocket’s exhaust. The sky is filled with soft purple and pink clouds, glowing with warm golden sunset light. The rocket emits a bright white-violet flame and glowing thrusters, creating swirling dust and petals near the ground. Cinematic wide shot, epic scale, fantasy sci-fi atmosphere, soft volumetric lighting, shallow haze near the horizon, high detail, smooth motion, dramatic yet serene mood."
with open("model_config.json", "r") as f:
    model_config_data = json.load(f)

decomposer = OpenAIDecomposer()
prompt_decomposition = decomposer.decompose(user_prompt=prompt)
judge = OpenAIJudge()

# configs = [VideoGenModelConfig(provider="openai", model_id="sora-2"), VideoGenModelConfig(
#     provider="fal", model_id="fal-ai/bytedance/seedance/v1/pro/fast/text-to-video")]
configs = [
    VideoGenModelConfig(provider=model_config["provider"], model_id=model_config["model_id"]) for model_config in model_config_data["models"]
]
arena = VideoGenArena(model_configs=configs, judge=judge)
result = arena.fight(video_gen_prompt=prompt,
                     prompt_decomposition=prompt_decomposition,
                     existing_video_path=None
                     )
with open(f"output/arena_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", "w") as f:
    f.write(result.model_dump_json(indent=2))
