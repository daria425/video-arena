from pathlib import Path
from process import sample_frames
from video_gen import VideoGenerator
from models import JudgeEval
from config.constants import EVAL_CRITERIA
from utils.llm_utils import _call_gemini_with_image_list
from utils.calculate import calculate_overall_score
if __name__ == "__main__":
    video_gen = VideoGenerator()
    prompt = "A sleek sci-fi rocketship launching vertically from the center of a vast lavender field at sunset. Endless rows of blooming purple lavender stretch toward the horizon, gently swaying from the rocket’s exhaust. The sky is filled with soft purple and pink clouds, glowing with warm golden sunset light. The rocket emits a bright white-violet flame and glowing thrusters, creating swirling dust and petals near the ground. Cinematic wide shot, epic scale, fantasy sci-fi atmosphere, soft volumetric lighting, shallow haze near the horizon, high detail, smooth motion, dramatic yet serene mood."
    video_info = video_gen.run(prompt)
    print(f"Generated video saved at: {video_info.saved_path}")
    frames = sample_frames(video_info.saved_path)
    image_bytes_list = [img.image for img in frames]
    user_prompts = [
        f"Frame {f.idx} at {f.timestamp_s:.2f}s"
        for f in frames
    ]
    # Add generation prompt at end
    user_prompts.append(f"Original prompt: {prompt}")
    video_id = Path(video_info.saved_path).stem
    input_data = {
        "prompt": prompt,
        "video_id": video_id
        # add duration, num frames, fps etc later
    }
    details = []
    scores = {}
    for eval_criterion in EVAL_CRITERIA:
        response: JudgeEval = _call_gemini_with_image_list(
            image_bytes_list=image_bytes_list,
            user_prompt_list=user_prompts,
            system_prompt_path=f"prompts/{eval_criterion}.txt",
            response_schema=JudgeEval
        )
        detail_item = {
            "criteria": eval_criterion,
            "score": response.score,
            "reasoning": response.reason
        }
        details.append(detail_item)
        scores[eval_criterion] = response.score
    overall_score = calculate_overall_score(
        scores=[s for s in list(scores.values())])
    scores["overall"] = overall_score
    report = {
        "input": input_data,
        "scores": scores,
        "details": details
    }
    print(report)
