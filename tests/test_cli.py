import pytest
from video_judge.cli import parse_model_spec, get_judge
from video_judge.judge import GeminiJudge, OpenAIJudge


class TestParseModelSpec:
    def test_valid_openai_spec(self):
        config = parse_model_spec("openai:sora-2")
        assert config.provider == "openai"
        assert config.model_id == "sora-2"

    def test_valid_fal_spec(self):
        config = parse_model_spec("fal:seedance-v1")
        assert config.provider == "fal"
        assert config.model_id == "seedance-v1"

    def test_colon_in_model_id_preserved(self):
        config = parse_model_spec("fal:fal-ai/bytedance/seedance/v1/pro/fast/text-to-video")
        assert config.model_id == "fal-ai/bytedance/seedance/v1/pro/fast/text-to-video"

    def test_missing_colon_raises(self):
        with pytest.raises(ValueError, match="Invalid model spec"):
            parse_model_spec("openai-sora-2")

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown provider"):
            parse_model_spec("runway:gen3")


class TestGetJudge:
    def test_gemini_judge(self):
        judge = get_judge("gemini")
        assert isinstance(judge, GeminiJudge)

    def test_openai_judge(self):
        judge = get_judge("openai")
        assert isinstance(judge, OpenAIJudge)

    def test_case_insensitive(self):
        judge = get_judge("OPENAI")
        assert isinstance(judge, OpenAIJudge)

    def test_unknown_judge_raises(self):
        with pytest.raises(ValueError, match="Unknown judge"):
            get_judge("claude")
