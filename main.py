from orchestrator import VideoEvaluationOrchestrator
from judge import OpenAIJudge
from models import VideoGenModelConfig
from arena import VideoGenArena
from config.logger import setup_default_logging
from datetime import datetime
setup_default_logging(level=20)
prompt = "A sleek sci-fi rocketship launching vertically from the center of a vast lavender field at sunset. Endless rows of blooming purple lavender stretch toward the horizon, gently swaying from the rocket’s exhaust. The sky is filled with soft purple and pink clouds, glowing with warm golden sunset light. The rocket emits a bright white-violet flame and glowing thrusters, creating swirling dust and petals near the ground. Cinematic wide shot, epic scale, fantasy sci-fi atmosphere, soft volumetric lighting, shallow haze near the horizon, high detail, smooth motion, dramatic yet serene mood."
video_eval_orchestrator = VideoEvaluationOrchestrator(
    video_gen_prompt=prompt, existing_video_path="./output/videos/generated_video_20260207_190028.mp4")
judge = OpenAIJudge()

configs = [VideoGenModelConfig(provider="openai", model_id="sora-2"), VideoGenModelConfig(
    provider="fal", model_id="fal-ai/bytedance/seedance/v1/pro/fast/text-to-video")]
arena = VideoGenArena(model_configs=configs, judge=judge)
result = arena.fight(video_eval_orchestrator)
with open(f"output/arena_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", "w") as f:
    f.write(result.model_dump_json(indent=2))
