# Video Generation Arena

**LLM-as-a-Judge evaluation framework for text-to-video models**

Arena-style competitive benchmarking for video generation models using LLM as a judge evaluation.

---

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/video-judge.git
cd video-judge

# Install in editable mode
pip install -e .

# Or install with dev dependencies
pip install -e ".[dev]"
```

### CLI Usage

**Run an arena competition:**

```bash
video-judge arena \
  --prompt "A rocket launching from a lavender field at sunset" \
  --models openai:sora-2 fal:seedance \
  --judge openai \
  --output results.json
```

**Evaluate a single video:**

```bash
video-judge eval \
  --video output/my_video.mp4 \
  --prompt "A rocket launching" \
  --judge gemini
```

### Programmatic Usage

```python
from video_judge import VideoGenArena, OpenAIJudge, VideoGenModelConfig

# Define models to compete
configs = [
    VideoGenModelConfig(provider="openai", model_id="sora-2"),
    VideoGenModelConfig(provider="fal", model_id="fal-ai/bytedance/seedance/v1/pro/fast/text-to-video")
]

# Run arena
arena = VideoGenArena(model_configs=configs, judge=OpenAIJudge())
result = arena.fight(prompt="A rocket launching from a field")

print(f"Winner: {result.winner}")
print(f"Score: {result.results[0].report.scores['overall']}")
```

See `main.py` for a complete example.

## CLI Commands

### `video-judge arena`

Run arena competition between multiple video generation models.

**Required:**

- `--prompt, -p`: Video generation prompt
- `--models, -m`: Model specs (format: `provider:model_id`)

**Optional:**

- `--judge, -j`: Judge to use (default: `openai`)
- `--output, -o`: Output path for results JSON
- `--existing-video, -e`: Use existing video for testing
- `--verbose, -v`: Enable verbose logging

**Example:**

```bash
video-judge arena \
  -p "A cinematic shot of a rocket launch" \
  -m openai:sora-2 \
  -m fal:seedance \
  -j openai \
  -o my_results.json
```

### `video-judge eval`

Evaluate a single existing video.

**Required:**

- `--video, -v`: Path to video file
- `--prompt, -p`: Original generation prompt

**Optional:**

- `--judge, -j`: Judge to use (default: `openai`)
- `--output, -o`: Output path for results JSON
- `--verbose`: Enable verbose logging

**Example:**

```bash
video-judge eval \
  -v output/my_video.mp4 \
  -p "A rocket launching" \
  -j gemini
```

---

## Environment Variables

Create a `.env` file with your API keys:

```bash
# Required for Gemini judge
GEMINI_API_KEY=your_key_here

# Required for OpenAI judge or Sora generation
OPENAI_API_KEY=your_key_here

# Required for FAL generation
FAL_KEY=your_key_here

# Optional for Anthropic/Claude
ANTHROPIC_API_KEY=your_key_here
```

---

## Output Format

### Arena Report

```json
{
  "prompt": "A rocket launching from a field",
  "results": [
    {
      "model": "sora-2",
      "report": {
        "input": {
          "prompt": "...",
          "video_id": "..."
        },
        "scores": {
          "prompt_alignment": 0.9,
          "temporal_consistency": 0.85,
          "aesthetic_quality": 0.8,
          "technical_quality": 0.9,
          "overall": 0.875
        },
        "details": [...]
      }
    }
  ],
  "winner": "sora-2"
}
```
