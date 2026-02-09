"""CLI for Video Generation Arena."""

import typer
from typing import List, Optional
from pathlib import Path
from datetime import datetime
import json

from video_judge.arena import VideoGenArena
from video_judge.judge import GeminiJudge, OpenAIJudge, BaseJudge
from video_judge.orchestrator import VideoEvaluationOrchestrator
from video_judge.models import VideoGenModelConfig
from video_judge.config.logger import setup_default_logging, logger

app = typer.Typer(help="Video Generation Arena - LLM-as-a-Judge for video models")


def parse_model_spec(spec: str) -> VideoGenModelConfig:
    """Parse model specification like 'openai:sora-2' or 'fal:seedance'."""
    if ":" not in spec:
        raise ValueError(
            f"Invalid model spec '{spec}'. Format: 'provider:model_id' (e.g., 'openai:sora-2')"
        )
    provider, model_id = spec.split(":", 1)

    if provider not in ["openai", "fal"]:
        raise ValueError(
            f"Unknown provider '{provider}'. Supported: openai, fal"
        )

    return VideoGenModelConfig(provider=provider, model_id=model_id)


def get_judge(judge_name: str) -> BaseJudge:
    """Get judge instance by name."""
    judges = {
        "gemini": GeminiJudge,
        "openai": OpenAIJudge,
    }

    judge_class = judges.get(judge_name.lower())
    if not judge_class:
        raise ValueError(
            f"Unknown judge '{judge_name}'. Supported: {', '.join(judges.keys())}"
        )

    return judge_class()


@app.command()
def arena(
    prompt: str = typer.Option(..., "--prompt", "-p", help="Video generation prompt"),
    models: List[str] = typer.Option(
        ...,
        "--models",
        "-m",
        help="Model specs (format: provider:model_id, e.g., 'openai:sora-2')"
    ),
    judge: str = typer.Option("openai", "--judge", "-j", help="Judge to use (gemini, openai)"),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output path for results JSON (default: auto-generated)"
    ),
    existing_video: Optional[Path] = typer.Option(
        None,
        "--existing-video",
        "-e",
        help="Use existing video instead of generating new ones (for testing)"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging"),
):
    """
    Run arena competition between video generation models.

    Example:
        video-judge arena \\
            --prompt "A rocket launching from a field" \\
            --models openai:sora-2 fal:seedance \\
            --judge openai
    """
    # Setup logging
    log_level = 10 if verbose else 20  # DEBUG if verbose, INFO otherwise
    setup_default_logging(level=log_level)

    logger.info(f"Starting arena with {len(models)} models")

    # Parse model configs
    try:
        model_configs = [parse_model_spec(spec) for spec in models]
    except ValueError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)

    # Get judge
    try:
        judge_instance = get_judge(judge)
    except ValueError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)

    # Setup orchestrator
    existing_path = str(existing_video) if existing_video else None
    orchestrator = VideoEvaluationOrchestrator(
        video_gen_prompt=prompt,
        existing_video_path=existing_path
    )

    # Run arena
    arena_instance = VideoGenArena(model_configs=model_configs, judge=judge_instance)

    try:
        result = arena_instance.fight(orchestrator)
    except Exception as e:
        logger.error(f"Arena failed: {e}")
        typer.echo(f"Error: Arena failed - {e}", err=True)
        raise typer.Exit(1)

    # Determine output path
    if output is None:
        output = Path(f"output/arena_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")

    # Ensure output directory exists
    output.parent.mkdir(parents=True, exist_ok=True)

    # Write results
    with open(output, "w") as f:
        f.write(result.model_dump_json(indent=2))

    # Print summary
    typer.echo("\n" + "="*60)
    typer.echo(f"🏆 WINNER: {result.winner}")
    typer.echo("="*60)
    typer.echo(f"\nResults saved to: {output}")
    typer.echo(f"\nRankings:")
    for i, run in enumerate(result.results, 1):
        score = run.report.scores.get("overall", 0.0)
        typer.echo(f"  {i}. {run.model}: {score:.3f}")
    typer.echo("")


@app.command()
def eval(
    video: Path = typer.Option(..., "--video", "-v", help="Path to video file"),
    prompt: str = typer.Option(..., "--prompt", "-p", help="Original generation prompt"),
    judge: str = typer.Option("openai", "--judge", "-j", help="Judge to use (gemini, openai)"),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output path for results JSON (default: auto-generated)"
    ),
    verbose: bool = typer.Option(False, "--verbose", help="Enable verbose logging"),
):
    """
    Evaluate a single existing video.

    Example:
        video-judge eval \\
            --video output/my_video.mp4 \\
            --prompt "A rocket launching" \\
            --judge openai
    """
    # Setup logging
    log_level = 10 if verbose else 20
    setup_default_logging(level=log_level)

    if not video.exists():
        typer.echo(f"Error: Video file not found: {video}", err=True)
        raise typer.Exit(1)

    logger.info(f"Evaluating video: {video}")

    # Get judge
    try:
        judge_instance = get_judge(judge)
    except ValueError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)

    # Setup orchestrator
    orchestrator = VideoEvaluationOrchestrator(
        video_gen_prompt=prompt,
        existing_video_path=str(video)
    )

    # Run evaluation
    try:
        from video_judge.video_gen import BaseVideoGenerator
        # Create a dummy generator (won't be used since we have existing_video_path)
        class DummyGenerator(BaseVideoGenerator):
            def run_video_gen(self, prompt, download_path=None):
                pass

        report = orchestrator.run(judge=judge_instance, video_generator=DummyGenerator("dummy"))
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        typer.echo(f"Error: Evaluation failed - {e}", err=True)
        raise typer.Exit(1)

    # Determine output path
    if output is None:
        output = Path(f"output/eval_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")

    # Ensure output directory exists
    output.parent.mkdir(parents=True, exist_ok=True)

    # Write results
    with open(output, "w") as f:
        json.dump(report.model_dump(), f, indent=2)

    # Print summary
    typer.echo("\n" + "="*60)
    typer.echo(f"Evaluation Results")
    typer.echo("="*60)
    typer.echo(f"\nOverall Score: {report.scores.get('overall', 0.0):.3f}")
    typer.echo(f"\nBreakdown:")
    for detail in report.details:
        typer.echo(f"  {detail['criteria']}: {detail['score']:.3f}")
    typer.echo(f"\nFull results saved to: {output}\n")


def main():
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
