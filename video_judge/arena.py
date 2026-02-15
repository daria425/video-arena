import asyncio
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor
from video_judge.judge import BaseJudge
from video_judge.video_gen import FalVideoGenerator, BaseVideoGenerator, OpenAIVideoGenerator, GoogleVideoGenerator
from video_judge.orchestrator import VideoEvaluationOrchestrator
from video_judge.config.logger import logger
from video_judge.models import ArenaRun, ArenaReport, ArenaRunFailure, VideoGenModelConfig, PromptDecomposition


class VideoGenArena:
    def __init__(self, model_configs: List[VideoGenModelConfig], judge: BaseJudge):
        self.model_config_list = model_configs
        self.judge = judge

    def _video_generator_factory(self) -> List[BaseVideoGenerator]:
        video_generators = []
        for config in self.model_config_list:
            if config.provider == "openai":
                video_generators.append(
                    OpenAIVideoGenerator(model=config.model_id))
            elif config.provider == "fal":
                video_generators.append(
                    FalVideoGenerator(model=config.model_id))
            elif config.provider == "google":
                video_generators.append(
                    GoogleVideoGenerator(model=config.model_id))
            else:
                raise NotImplementedError(
                    "Providers other than openai and fal not supported yet")
        return video_generators

    def _evaluate_model(self, generator: BaseVideoGenerator, judge: BaseJudge,
                        prompt: str, existing_video_path: Optional[str] = None, prompt_decomposition: Optional[PromptDecomposition] = None) -> ArenaRun:
        """Run a single model's full pipeline (generation + evaluation).

        Creates a fresh orchestrator per model to avoid shared state.
        """
        orchestrator = VideoEvaluationOrchestrator(
            video_gen_prompt=prompt,
            existing_video_path=existing_video_path,
            prompt_decomposition=prompt_decomposition
        )
        logger.info(f"Starting evaluation run for model: {generator.model}")
        report = orchestrator.run(judge=judge, video_generator=generator)
        logger.debug(f"Report for model {generator.model}: {report}")
        return ArenaRun(model=generator.model, report=report)

    async def _fight_async(self, prompt: str, existing_video_path: Optional[str] = None, prompt_decomposition: Optional[PromptDecomposition] = None):
        """Run all models in parallel using thread pool."""
        generators = self._video_generator_factory()
        loop = asyncio.get_event_loop()

        with ThreadPoolExecutor(max_workers=len(generators)) as pool:
            tasks = [
                loop.run_in_executor(
                    pool,
                    self._evaluate_model,
                    gen, self.judge, prompt, existing_video_path, prompt_decomposition
                )
                for gen in generators
            ]
            raw_results = await asyncio.gather(*tasks, return_exceptions=True)

        results = []
        failures = []
        for gen, result in zip(generators, raw_results):
            if isinstance(result, Exception):
                logger.error(
                    f"Model {gen.model} failed: {type(result).__name__}: {result}")
                failures.append(ArenaRunFailure(
                    model=gen.model, error=str(result), error_type=type(result).__name__))
            else:
                logger.info(f"Model {gen.model} completed successfully.")
                results.append(result)

        if not results:
            raise RuntimeError(f"All models failed. Failures: {failures}")

        ranked = sorted(
            results, key=lambda x: x.report.scores["overall"], reverse=True)
        model_rankings = [run.model for run in ranked]
        return ArenaReport(prompt=prompt, results=ranked, winner=ranked[0].model, rankings=model_rankings)

    def fight(self, video_gen_prompt: str, existing_video_path: Optional[str] = None, prompt_decomposition: Optional[PromptDecomposition] = None):
        """Begins a video generation competition among text-to-video models.

        Runs all models in parallel. Each model gets its own orchestrator
        instance to avoid shared mutable state.
        """
        return asyncio.run(self._fight_async(
            prompt=video_gen_prompt,
            existing_video_path=existing_video_path,
            prompt_decomposition=prompt_decomposition
        ))
