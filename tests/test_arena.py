import pytest
from unittest.mock import MagicMock, patch
from video_judge.arena import VideoGenArena
from video_judge.models import VideoGenModelConfig, Report, ArenaReport


def _make_report(overall: float) -> Report:
    return Report(
        input={"prompt": "test", "video_id": "v1"},
        scores={
            "prompt_alignment": overall,
            "temporal_consistency": overall,
            "aesthetic_quality": overall,
            "technical_quality": overall,
            "overall": overall,
        },
        details=[],
    )


class TestVideoGeneratorFactory:
    def test_creates_openai_generator(self):
        configs = [VideoGenModelConfig(provider="openai", model_id="sora-2")]
        arena = VideoGenArena(model_configs=configs, judge=MagicMock())
        generators = arena._video_generator_factory()
        assert len(generators) == 1
        assert generators[0].model == "sora-2"

    def test_creates_fal_generator(self):
        configs = [VideoGenModelConfig(provider="fal", model_id="seedance")]
        arena = VideoGenArena(model_configs=configs, judge=MagicMock())
        generators = arena._video_generator_factory()
        assert len(generators) == 1
        assert generators[0].model == "seedance"

    def test_multiple_generators(self):
        configs = [
            VideoGenModelConfig(provider="openai", model_id="sora-2"),
            VideoGenModelConfig(provider="fal", model_id="seedance"),
        ]
        arena = VideoGenArena(model_configs=configs, judge=MagicMock())
        generators = arena._video_generator_factory()
        assert len(generators) == 2


class TestArenaFight:
    def test_ranks_by_overall_score(self):
        """Higher overall score should be ranked first (winner)."""
        mock_judge = MagicMock()
        configs = [
            VideoGenModelConfig(provider="openai", model_id="model-a"),
            VideoGenModelConfig(provider="fal", model_id="model-b"),
        ]
        arena = VideoGenArena(model_configs=configs, judge=mock_judge)

        orchestrator = MagicMock()
        orchestrator.video_gen_prompt = "test prompt"
        orchestrator.existing_video_path = None

        reports = [_make_report(0.6), _make_report(0.9)]
        report_iter = iter(reports)

        def mock_orch_run(judge, video_generator):
            return next(report_iter)

        with patch.object(arena, "_video_generator_factory") as factory, \
             patch("video_judge.arena.VideoEvaluationOrchestrator") as MockOrch:
            gen_a = MagicMock()
            gen_a.model = "model-a"
            gen_b = MagicMock()
            gen_b.model = "model-b"
            factory.return_value = [gen_a, gen_b]

            mock_orch_instance = MagicMock()
            mock_orch_instance.run.side_effect = mock_orch_run
            MockOrch.return_value = mock_orch_instance

            result = arena.fight(orchestrator)

        assert result.winner == "model-b"
        assert result.results[0].report.scores["overall"] == 0.9

    def test_model_failure_doesnt_crash_arena(self):
        """If one model fails, the other should still produce a result."""
        mock_judge = MagicMock()
        configs = [
            VideoGenModelConfig(provider="openai", model_id="bad-model"),
            VideoGenModelConfig(provider="fal", model_id="good-model"),
        ]
        arena = VideoGenArena(model_configs=configs, judge=mock_judge)

        orchestrator = MagicMock()
        orchestrator.video_gen_prompt = "test"
        orchestrator.existing_video_path = None

        def mock_orch_run(judge, video_generator):
            if video_generator.model == "bad-model":
                raise RuntimeError("API down")
            return _make_report(0.75)

        with patch.object(arena, "_video_generator_factory") as factory, \
             patch("video_judge.arena.VideoEvaluationOrchestrator") as MockOrch:
            gen_bad = MagicMock()
            gen_bad.model = "bad-model"
            gen_good = MagicMock()
            gen_good.model = "good-model"
            factory.return_value = [gen_bad, gen_good]

            mock_orch_instance = MagicMock()
            mock_orch_instance.run.side_effect = mock_orch_run
            MockOrch.return_value = mock_orch_instance

            result = arena.fight(orchestrator)

        assert result.winner == "good-model"
        assert len(result.results) == 1

    def test_all_models_fail_raises(self):
        """If every model fails, arena should raise RuntimeError."""
        mock_judge = MagicMock()
        configs = [VideoGenModelConfig(provider="openai", model_id="bad")]
        arena = VideoGenArena(model_configs=configs, judge=mock_judge)

        orchestrator = MagicMock()
        orchestrator.video_gen_prompt = "test"
        orchestrator.existing_video_path = None

        with patch.object(arena, "_video_generator_factory") as factory, \
             patch("video_judge.arena.VideoEvaluationOrchestrator") as MockOrch:
            gen = MagicMock()
            gen.model = "bad"
            factory.return_value = [gen]

            mock_orch_instance = MagicMock()
            mock_orch_instance.run.side_effect = RuntimeError("fail")
            MockOrch.return_value = mock_orch_instance

            with pytest.raises(RuntimeError, match="All models failed"):
                arena.fight(orchestrator)
