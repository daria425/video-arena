import pytest
from datetime import datetime
from pydantic import ValidationError
from video_judge.models import (
    VideoMetadata,
    VideoInfo,
    VideoFrame,
    Evidence,
    JudgeEval,
    Report,
    ArenaRun,
    ArenaReport,
    ArenaRunFailure,
    VideoGenModelConfig,
)


class TestVideoGenModelConfig:
    def test_valid_providers(self):
        openai = VideoGenModelConfig(provider="openai", model_id="sora-2")
        assert openai.provider == "openai"

        fal = VideoGenModelConfig(provider="fal", model_id="seedance")
        assert fal.provider == "fal"

    def test_invalid_provider_rejected(self):
        with pytest.raises(ValidationError):
            VideoGenModelConfig(provider="runway", model_id="gen3")


class TestVideoMetadata:
    def test_required_fields(self):
        meta = VideoMetadata(
            generated_at=datetime.now(),
            prompt="a cat",
            file_size=1024,
        )
        assert meta.seed is None

    def test_missing_prompt_raises(self):
        with pytest.raises(ValidationError):
            VideoMetadata(generated_at=datetime.now(), file_size=100)


class TestJudgeEval:
    def test_valid_eval(self):
        ev = JudgeEval(
            score=0.85,
            reason="looks good",
            evidence=[Evidence(frame=0, timestamp=0.0, finding="sharp")],
        )
        assert ev.score == 0.85
        assert len(ev.evidence) == 1

    def test_missing_evidence_raises(self):
        with pytest.raises(ValidationError):
            JudgeEval(score=0.5, reason="ok")


class TestReport:
    def test_report_roundtrip(self):
        report = Report(
            input={"prompt": "rocket", "video_id": "abc"},
            scores={"overall": 0.7},
            details=[{"criteria": "prompt_alignment", "score": 0.7, "reasoning": "good"}],
        )
        data = report.model_dump()
        assert data["scores"]["overall"] == 0.7


class TestArenaReport:
    def test_arena_report_structure(self):
        report = Report(
            input={"prompt": "test"},
            scores={"overall": 0.8},
            details=[],
        )
        arena = ArenaReport(
            prompt="test",
            results=[ArenaRun(model="sora-2", report=report)],
            winner="sora-2",
        )
        assert arena.winner == "sora-2"
        assert len(arena.results) == 1


class TestArenaRunFailure:
    def test_failure_captures_error_type(self):
        f = ArenaRunFailure(model="bad-model", error="boom", error_type="RuntimeError")
        assert f.error_type == "RuntimeError"
