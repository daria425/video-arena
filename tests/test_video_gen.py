import pytest
from unittest.mock import patch, MagicMock, PropertyMock
from video_judge.video_gen import FalVideoGenerator, OpenAIVideoGenerator


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
