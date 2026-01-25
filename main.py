from process import sample_frames
from video_gen import VideoGenerator
from models import JudgeEval
from utils.llm_utils import _call_gemini_with_image_list

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
    print(f"DEBUG: Sampled {len(frames)} frames")  # Add this
    print(f"DEBUG: image_bytes_list length: {len(image_bytes_list)}")
    print(f"DEBUG: user_prompts length: {len(user_prompts)}")
    _call_gemini_with_image_list(
        image_bytes_list=image_bytes_list,
        user_prompt_list=user_prompts,
        system_prompt_path="./prompts/alignment.txt",
        response_schema=JudgeEval
    )
