#!/bin/bash
# Arena competition: Sora vs Seedance on rocketship prompt

video-judge arena \
  --prompt "A sleek sci-fi rocketship launching vertically from the center of a vast lavender field at sunset. Endless rows of blooming purple lavender stretch toward the horizon, gently swaying from the rocket's exhaust. The sky is filled with soft purple and pink clouds, glowing with warm golden sunset light. The rocket emits a bright white-violet flame and glowing thrusters, creating swirling dust and petals near the ground. Cinematic wide shot, epic scale, fantasy sci-fi atmosphere, soft volumetric lighting, shallow haze near the horizon, high detail, smooth motion, dramatic yet serene mood." \
  --models openai:sora-2 \
  --models fal:fal-ai/bytedance/seedance/v1/pro/fast/text-to-video \
  --judge openai \
  --verbose
