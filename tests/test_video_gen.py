import pytest
from unittest.mock import patch, MagicMock, PropertyMock
from video_judge.video_gen import FalVideoGenerator, OpenAIVideoGenerator, GoogleVideoGenerator


class TestFalVideoGenerator:
    def test_init_default_model(self):
        gen = FalVideoGenerator()
        assert "fal-ai" in gen.model

    def test_init_custom_model(self):
        gen = FalVideoGenerator(model="fal-ai/custom-model")
        assert gen.model == "fal-ai/custom-model"

    @patch("video_judge.video_gen.fal_client")
    def test_submit_request_stores_request_id(self, mock_fal):
        mock_handler = MagicMock()
        mock_handler.request_id = "req-123"
        mock_fal.submit.return_value = mock_handler

        gen = FalVideoGenerator()
        gen.submit_request("a cat")

        assert gen._request_id == "req-123"
        mock_fal.submit.assert_called_once()


class TestOpenAIVideoGenerator:
    def test_init_default_model(self):
        gen = OpenAIVideoGenerator()
        assert gen.model == "sora-2"

    def test_init_custom_model(self):
        gen = OpenAIVideoGenerator(model="sora-3")
        assert gen.model == "sora-3"

    @patch("video_judge.video_gen.openai_client")
    def test_submit_request_stores_request_id(self, mock_openai):
        mock_response = MagicMock()
        mock_response.id = "vid-456"
        mock_openai.client.videos.create.return_value = mock_response

        gen = OpenAIVideoGenerator()
        gen.submit_request("a rocket")

        assert gen._request_id == "vid-456"


class TestGoogleVideoGenerator:
    def test_init_default_model(self):
        gen = GoogleVideoGenerator()
        assert gen.model == "veo-3.1-fast-generate-preview"

    def test_init_custom_model(self):
        gen = GoogleVideoGenerator(model="veo-2")
        assert gen.model == "veo-2"

    @patch("video_judge.video_gen.google_client")
    def test_submit_request_stores_operation(self, mock_google):
        mock_operation = MagicMock()
        mock_google.client.models.generate_videos.return_value = mock_operation

        gen = GoogleVideoGenerator()
        gen.submit_request("a rocket")

        assert gen._operation == mock_operation
        mock_google.client.models.generate_videos.assert_called_once_with(
            model=gen.model,
            prompt="a rocket"
        )

    @patch("video_judge.video_gen.google_client")
    def test_fetch_status_updates_operation(self, mock_google):
        initial_operation = MagicMock()
        updated_operation = MagicMock()
        mock_google.client.operations.get.return_value = updated_operation

        gen = GoogleVideoGenerator()
        gen._operation = initial_operation
        gen.fetch_status()

        assert gen._operation == updated_operation
        mock_google.client.operations.get.assert_called_once_with(
            initial_operation)

    @patch("video_judge.video_gen.google_client")
    @patch("video_judge.video_gen.get_video")
    @patch("video_judge.video_gen.time.sleep")
    def test_get_result_success(self, mock_sleep, mock_get_video, mock_google):
        mock_video_content = b"fake_video_content"
        mock_get_video.return_value = mock_video_content

        # Mock the operation to be done on first check
        mock_operation = MagicMock()
        mock_operation.done = True
        mock_operation.error = None
        mock_operation.response.generated_videos = [MagicMock()]
        mock_operation.response.generated_videos[0].video.uri = "gs://bucket/video.mp4"

        mock_google.client.operations.get.return_value = mock_operation

        gen = GoogleVideoGenerator()
        gen._operation = MagicMock()
        result = gen.get_result()

        assert result["video"]["content"] == mock_video_content
        assert result["video"]["file_size"] == len(mock_video_content)
        assert result["seed"] is None
        mock_get_video.assert_called_once_with("gs://bucket/video.mp4")

    @patch("video_judge.video_gen.google_client")
    @patch("video_judge.video_gen.time.sleep")
    def test_get_result_with_error(self, mock_sleep, mock_google):
        # Mock the operation to have an error
        mock_operation = MagicMock()
        mock_operation.done = False
        mock_operation.error = "Some error occurred"

        mock_google.client.operations.get.return_value = mock_operation

        gen = GoogleVideoGenerator()
        gen._operation = MagicMock()

        with pytest.raises(RuntimeError, match="Video generation failed with error"):
            gen.get_result()

    @patch("video_judge.video_gen.google_client")
    @patch("video_judge.video_gen.time.sleep")
    @patch("video_judge.video_gen.time.time")
    def test_get_result_timeout(self, mock_time, mock_sleep, mock_google):
        # Mock time to exceed timeout
        mock_time.side_effect = [0, 1000]  # Start time, then past timeout

        mock_operation = MagicMock()
        mock_operation.done = False
        mock_operation.error = None
        mock_google.client.operations.get.return_value = mock_operation

        gen = GoogleVideoGenerator()
        gen._operation = MagicMock()

        with pytest.raises(TimeoutError, match="Video generation failed after"):
            gen.get_result(timeout=900)
