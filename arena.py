from typing import List
from judge import BaseJudge
from video_gen import VideoGenerator
from orchestrator import VideoEvaluationOrchestrator
from config.logger import logger
from models import ArenaRun, ArenaReport


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
        for generator in generators:
            logger.info(f"Starting evaluation for model: {generator.model}")
            report = orchestrator.run(
                judge=self.judge, video_generator=generator)
            logger.debug(f"Report for model {generator.model}: {report}")
            results.append(ArenaRun(model=generator.model, report=report))
        ranked = sorted(
            results, key=lambda x: x.report.scores["overall"], reverse=True)
        return ArenaReport(prompt=orchestrator.video_gen_prompt, results=ranked, winner=ranked[0].model)
