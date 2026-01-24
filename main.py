from process import sample_frames
from video_gen import VideoGenerator

if __name__ == "__main__":
    video_gen = VideoGenerator()
    prompt = "A sleek sci-fi rocketship launching vertically from the center of a vast lavender field at sunset. Endless rows of blooming purple lavender stretch toward the horizon, gently swaying from the rocket’s exhaust. The sky is filled with soft purple and pink clouds, glowing with warm golden sunset light. The rocket emits a bright white-violet flame and glowing thrusters, creating swirling dust and petals near the ground. Cinematic wide shot, epic scale, fantasy sci-fi atmosphere, soft volumetric lighting, shallow haze near the horizon, high detail, smooth motion, dramatic yet serene mood."
    video_info = video_gen.run(prompt)
    print(f"Generated video saved at: {video_info.saved_path}")
    frames = sample_frames(video_info.saved_path)
    for f in frames:
        img_path = f"output/frames/frame_{f.idx}_t{f.timestamp_s:.2f}s.png"
        print(f"Frame image saved at: {img_path}")
        with open(img_path, 'wb') as file:
            file.write(f.image)
