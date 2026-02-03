from typing import List
from judge import BaseJudge
from video_gen import VideoGenerator
from orchestrator import VideoEvaluationOrchestrator
from config.logger import logger
from models import ArenaRun, ArenaReport, ArenaRunFailure


class VideoGenArena:
    def __init__(self, models: List[str], judge: BaseJudge):
        self.models = models
        self.judge = judge

    def _video_generator_factory(self) -> List[VideoGenerator]:
        return [VideoGenerator(model=model) for model in self.models]

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
