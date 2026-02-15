from unittest.mock import MagicMock, patch
from video_judge.orchestrator import VideoEvaluationOrchestrator
from video_judge.models import JudgeEval, Evidence, PromptDecomposition


def _mock_judge_eval(score: float) -> JudgeEval:
    return JudgeEval(
        score=score,
        reason="mock reason",
        evidence=[Evidence(frame=0, timestamp=0.0, finding="mock finding")],
    )


class TestRunNodes:
    def test_report_has_all_criteria_and_overall(self):
        """run_nodes should produce scores for all 4 criteria + overall."""
        orch = VideoEvaluationOrchestrator(video_gen_prompt="test prompt")
        orch.input_data = {"prompt": "test prompt", "video_id": "v1"}
        orch.existing_video_path = "/fake/video.mp4"

        mock_judge = MagicMock()
        mock_judge.evaluate.side_effect = [
            _mock_judge_eval(0.9),  # alignment
            _mock_judge_eval(0.8),  # temporal
            _mock_judge_eval(0.7),  # aesthetic
            _mock_judge_eval(0.6),  # technical
        ]

        with patch("video_judge.orchestrator.format_prompt", return_value="system prompt"):
            report = orch.run_nodes(
                images=[b"fake"], user_prompts=["frame 0"], judge=mock_judge
            )

        assert "prompt_alignment" in report.scores
        assert "temporal_consistency" in report.scores
        assert "aesthetic_quality" in report.scores
        assert "technical_quality" in report.scores
        assert "overall" in report.scores
        assert len(report.details) == 4

    def test_overall_uses_correct_weights(self):
        """overall = 0.5*alignment + 0.3*temporal + 0.1*aesthetic + 0.1*technical."""
        orch = VideoEvaluationOrchestrator(video_gen_prompt="test")
        orch.input_data = {"prompt": "test", "video_id": "v1"}
        orch.existing_video_path = "/fake/video.mp4"

        mock_judge = MagicMock()
        mock_judge.evaluate.side_effect = [
            _mock_judge_eval(1.0),  # alignment
            _mock_judge_eval(1.0),  # temporal
            _mock_judge_eval(0.0),  # aesthetic
            _mock_judge_eval(0.0),  # technical
        ]

        with patch("video_judge.orchestrator.format_prompt", return_value="prompt"):
            report = orch.run_nodes(
                images=[b"fake"], user_prompts=["f0"], judge=mock_judge
            )

        # 1.0*0.5 + 1.0*0.3 + 0.0*0.1 + 0.0*0.1 = 0.8
        assert abs(report.scores["overall"] - 0.8) < 1e-9


class TestCreateJudgeInput:
    def test_user_prompts_include_frame_labels_and_original_prompt(self):
        """user_prompts should have per-frame labels + the original prompt appended."""
        orch = VideoEvaluationOrchestrator(
            video_gen_prompt="a rocket",
            existing_video_path="/fake/video.mp4",
        )

        from video_judge.models import VideoFrame

        fake_frames = [
            VideoFrame(idx=0, image=b"img0", timestamp_s=0.0),
            VideoFrame(idx=10, image=b"img1", timestamp_s=1.0),
        ]

        with patch("video_judge.orchestrator.sample_frames", return_value=fake_frames):
            images, user_prompts = orch.create_judge_input_from_video()

        assert len(images) == 2
        assert len(user_prompts) == 3  # 2 frame labels + 1 original prompt
        assert "Frame 0" in user_prompts[0]
        assert "Original prompt: a rocket" in user_prompts[-1]

    def test_user_prompts_include_decomposition_when_provided(self):
        """When prompt_decomposition is provided, it should be formatted and appended."""
        decomposition = PromptDecomposition(
            entities=["rocket"],
            actions=["launching"],
            locations=["lavender field"],
            time_of_day="sunset",
            style_attributes=["cinematic"],
        )

        orch = VideoEvaluationOrchestrator(
            video_gen_prompt="a rocket launching",
            existing_video_path="/fake/video.mp4",
            prompt_decomposition=decomposition,
        )

        from video_judge.models import VideoFrame

        fake_frames = [VideoFrame(idx=0, image=b"img0", timestamp_s=0.0)]

        with patch("video_judge.orchestrator.sample_frames", return_value=fake_frames):
            images, user_prompts = orch.create_judge_input_from_video()

        # Should have: 1 frame label + original prompt + decomposition
        assert len(user_prompts) == 3
        assert "Frame 0" in user_prompts[0]
        assert "Original prompt:" in user_prompts[1]
        assert "Key criteria" in user_prompts[2]
        assert "Entity: rocket" in user_prompts[2]
        assert "Action: launching" in user_prompts[2]
        assert "Location: lavender field" in user_prompts[2]
        assert "Time: sunset" in user_prompts[2]
        assert "Style: cinematic" in user_prompts[2]

    def test_no_decomposition_when_not_provided(self):
        """When prompt_decomposition is None, user_prompts should not include criteria."""
        orch = VideoEvaluationOrchestrator(
            video_gen_prompt="a rocket",
            existing_video_path="/fake/video.mp4",
            prompt_decomposition=None,
        )

        from video_judge.models import VideoFrame

        fake_frames = [VideoFrame(idx=0, image=b"img0", timestamp_s=0.0)]

        with patch("video_judge.orchestrator.sample_frames", return_value=fake_frames):
            images, user_prompts = orch.create_judge_input_from_video()

        # Should only have frame label + original prompt (no decomposition)
        assert len(user_prompts) == 2
        assert "Key criteria" not in user_prompts[-1]


class TestNodeRouting:
    def test_node_calls_judge_with_correct_prompt_file(self):
        """Each node method should load the right prompt file."""
        orch = VideoEvaluationOrchestrator(video_gen_prompt="test")
        mock_judge = MagicMock()
        mock_judge.evaluate.return_value = _mock_judge_eval(0.5)

        with patch("video_judge.orchestrator.format_prompt", return_value="sys") as mock_fmt:
            orch.alignment_node(images=[b"x"], user_prompts=["f0"], judge=mock_judge)
            mock_fmt.assert_called_with("./prompts/prompt_alignment.txt")

        with patch("video_judge.orchestrator.format_prompt", return_value="sys") as mock_fmt:
            orch.temporal_consistency_node(images=[b"x"], user_prompts=["f0"], judge=mock_judge)
            mock_fmt.assert_called_with("./prompts/temporal_consistency.txt")
