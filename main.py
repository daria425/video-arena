from orchestrator import VideoEvaluationOrchestrator
from judge import GeminiJudge
from video_gen import VideoGenerator
from arena import VideoGenArena
from config.logger import setup_default_logging

setup_default_logging(level=20)
prompt = "A sleek sci-fi rocketship launching vertically from the center of a vast lavender field at sunset. Endless rows of blooming purple lavender stretch toward the horizon, gently swaying from the rocket’s exhaust. The sky is filled with soft purple and pink clouds, glowing with warm golden sunset light. The rocket emits a bright white-violet flame and glowing thrusters, creating swirling dust and petals near the ground. Cinematic wide shot, epic scale, fantasy sci-fi atmosphere, soft volumetric lighting, shallow haze near the horizon, high detail, smooth motion, dramatic yet serene mood."
video_eval_orchestrator = VideoEvaluationOrchestrator(
    video_gen_prompt=prompt)
judge = GeminiJudge()
models = [
    "fal-ai/bytedance/seedance/v1/pro/fast/text-to-video", "fal-ai/kandinsky5-pro/text-to-video"
]
arena = VideoGenArena(models=models, judge=judge)
result = arena.fight(video_eval_orchestrator)
print(result)
