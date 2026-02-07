from typing import List
from judge import BaseJudge
from video_gen import FalVideoGenerator, BaseVideoGenerator, OpenAIVideoGenerator
from orchestrator import VideoEvaluationOrchestrator
from config.logger import logger
from models import ArenaRun, ArenaReport, ArenaRunFailure, VideoGenModelConfig


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
            else:
                raise NotImplementedError(
                    "Providers other than openai and fal not supported yet")
        return video_generators

    def fight(self, orchestrator: VideoEvaluationOrchestrator):
        """Begins a video generation competition among text-to-video models."""
        generators = self._video_generator_factory()
        results = []
        failures = []
        for generator in generators:
            try:
                logger.info(
                    f"Starting evaluation run for model: {generator.model}")
                report = orchestrator.run(
                    judge=self.judge, video_generator=generator)
                logger.debug(f"Report for model {generator.model}: {report}")
                results.append(ArenaRun(model=generator.model, report=report))
            except Exception as e:
                logger.error(
                    f"Model {generator.model} failed: {type(e).__name__}: {e}")
                failures.append(ArenaRunFailure(
                    model=generator.model, error=str(e), error_type=type(e).__name__))
        if not results:
            raise RuntimeError(f"All models failed. Failures: {failures}")
        ranked = sorted(
            results, key=lambda x: x.report.scores["overall"], reverse=True)
        return ArenaReport(prompt=orchestrator.video_gen_prompt, results=ranked, winner=ranked[0].model)
