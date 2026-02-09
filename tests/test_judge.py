from unittest.mock import patch, MagicMock
from video_judge.judge import GeminiJudge, OpenAIJudge
from video_judge.models import JudgeEval, Evidence


def _fake_eval():
    return JudgeEval(
        score=0.8,
        reason="good",
        evidence=[Evidence(frame=0, timestamp=0.0, finding="ok")],
    )


class TestGeminiJudge:
    @patch("video_judge.judge.build_gemini_input_with_image_list")
    def test_evaluate_delegates_to_builder(self, mock_builder):
        mock_builder.return_value = _fake_eval()
        judge = GeminiJudge()
        result = judge.evaluate(
            images=[b"img"],
            user_prompts=["frame 0"],
            system_prompt="evaluate",
        )
        assert result.score == 0.8
        mock_builder.assert_called_once()


class TestOpenAIJudge:
    @patch("video_judge.judge.build_openai_input_with_image_list")
    def test_evaluate_delegates_to_builder(self, mock_builder):
        mock_builder.return_value = _fake_eval()
        judge = OpenAIJudge()
        result = judge.evaluate(
            images=[b"img"],
            user_prompts=["frame 0"],
            system_prompt="evaluate",
        )
        assert result.score == 0.8
        mock_builder.assert_called_once()
