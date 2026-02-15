# Video Generation Arena

**LLM-as-a-Judge evaluation framework for text-to-video models**

Arena-style competitive benchmarking for video generation models using multi-criteria LLM evaluation.

## Installation

```bash
# Clone the repository
git clone https://github.com/daria425/video-judge.git
cd video-judge

# Install dependencies
pip install -e .

# Or with dev dependencies
pip install -e ".[dev]"
```

---

## Quick Start

```python
from video_judge import (
    VideoGenArena,
    VideoEvaluationOrchestrator,
    OpenAIJudge,
    VideoGenModelConfig,
    OpenAIDecomposer
)

# Define prompt
prompt = "A sleek sci-fi rocketship launching from a lavender field at sunset"

# Decompose prompt for structured evaluation
decomposer = OpenAIDecomposer()
decomposition = decomposer.decompose(prompt)

# Setup orchestrator
orchestrator = VideoEvaluationOrchestrator(
    video_gen_prompt=prompt,
    prompt_decomposition=decomposition
)

# Define competing models
configs = [
    VideoGenModelConfig(provider="openai", model_id="sora-2"),
    VideoGenModelConfig(provider="google", model_id="veo-3.1-fast-generate-preview"),
    VideoGenModelConfig(provider="fal", model_id="fal-ai/bytedance/seedance/v1/pro/fast/text-to-video")
]

# Run arena
arena = VideoGenArena(model_configs=configs, judge=OpenAIJudge())
result = arena.fight(orchestrator)

# View results
print(f"🏆 Winner: {result.winner}")
for run in result.results:
    print(f"{run.model}: {run.report.scores['overall']:.3f}")
```

See `main.py` for a complete example with config loading.

---

## Supported Models

### OpenAI

- `sora-2-pro` - Premium quality with synced audio
- `sora-2` - High quality, lower cost

### Google Veo

- `veo-3.1-generate-preview` - Advanced cinematic generation
- `veo-3.1-fast-generate-preview` - 2x faster, 70% cheaper

### FAL (ByteDance, Kuaishou, Luma, MiniMax)

- `fal-ai/kling-video/o3/standard/text-to-video` - Kling 3.0 multi-shot
- `fal-ai/bytedance/seedance/v1/pro/fast/text-to-video` - Seedance Pro Fast
- `fal-ai/minimax/hailuo-2.3/pro/text-to-video` - Hailuo 2.3 cinematic
- `fal-ai/luma-dream-machine/ray-2` - Luma Ray 2 motion

See `model_config.json` for full list and benchmark presets.

---

## Configuration

Create a `.env` file with your API keys:

```bash
# Required for video generation
OPENAI_API_KEY=your_key_here
GEMINI_API_KEY=your_key_here
FAL_KEY=your_key_here

# Optional for Claude judge/decomposer
ANTHROPIC_API_KEY=your_key_here
```

Model configuration in `model_config.json`:

```json
{
  "models": [
    {
      "provider": "google",
      "model_id": "veo-3.1-fast-generate-preview",
      "tier": "balanced"
    }
  ],
  "benchmark_configs": {
    "quick_benchmark": {
      "models": ["sora-2", "veo-3.1-fast", "seedance"]
    }
  }
}
```

---

## Evaluation Criteria

**Prompt Alignment (50% weight)** - Entities, actions, attributes match prompt
**Temporal Consistency (30% weight)** - Stable identity, no jumps or morphing
**Aesthetic Quality (10% weight)** - Composition, lighting, cinematography
**Technical Quality (10% weight)** - No artifacts, deformations, or glitches

Each criterion scored 0.0-1.0 with frame-level evidence and reasoning.

---

## Output Format

```json
{
  "prompt": "A rocket launching...",
  "results": [
    {
      "model": "veo-3.1-fast-generate-preview",
      "report": {
        "scores": {
          "prompt_alignment": 0.92,
          "temporal_consistency": 0.88,
          "aesthetic_quality": 0.85,
          "technical_quality": 0.9,
          "overall": 0.9
        },
        "details": [
          {
            "criteria": "prompt_alignment",
            "score": 0.92,
            "reasoning": "All entities present...",
            "evidence": [
              {
                "frame": 0,
                "timestamp": 0.0,
                "finding": "Rocket visible in center"
              }
            ]
          }
        ]
      }
    }
  ],
  "winner": "veo-3.1-fast-generate-preview",
  "rankings": ["veo-3.1-fast-generate-preview", "sora-2", "seedance"]
}
```

---
