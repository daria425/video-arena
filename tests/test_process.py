from unittest.mock import patch, MagicMock
import numpy as np
from video_judge.process import sample_frames, get_video_metadata


class TestGetVideoMetadata:
    def test_returns_fps_total_frames_duration(self):
        mock_cap = MagicMock()
        mock_cap.get.side_effect = lambda prop: {
            # cv2.CAP_PROP_FPS = 5, cv2.CAP_PROP_FRAME_COUNT = 7
            5: 30.0,
            7: 300,
        }.get(prop, 0)

        with patch("video_judge.process.cv2.VideoCapture", return_value=mock_cap):
            meta = get_video_metadata("/fake/video.mp4")

        assert meta["fps"] == 30.0
        assert meta["total_frames"] == 300
        assert abs(meta["duration_s"] - 10.0) < 1e-6


class TestSampleFrames:
    def test_returns_correct_number_of_frames(self):
        mock_cap = MagicMock()
        mock_cap.get.side_effect = lambda prop: {5: 24.0, 7: 240}.get(prop, 0)
        # Every read succeeds
        mock_cap.read.return_value = (True, np.zeros((100, 100, 3), dtype=np.uint8))

        with patch("video_judge.process.cv2.VideoCapture", return_value=mock_cap):
            with patch("video_judge.process.cv2.cvtColor", side_effect=lambda f, _: f):
                with patch(
                    "video_judge.process.cv2.imencode",
                    return_value=(True, np.array([1, 2, 3], dtype=np.uint8)),
                ):
                    frames = sample_frames("/fake/video.mp4", num_frames=8)

        assert len(frames) == 8

    def test_first_and_last_frame_indices(self):
        total = 240
        mock_cap = MagicMock()
        mock_cap.get.side_effect = lambda prop: {5: 24.0, 7: total}.get(prop, 0)
        mock_cap.read.return_value = (True, np.zeros((100, 100, 3), dtype=np.uint8))

        with patch("video_judge.process.cv2.VideoCapture", return_value=mock_cap):
            with patch("video_judge.process.cv2.cvtColor", side_effect=lambda f, _: f):
                with patch(
                    "video_judge.process.cv2.imencode",
                    return_value=(True, np.array([1], dtype=np.uint8)),
                ):
                    frames = sample_frames("/fake/video.mp4", num_frames=4)

        indices = [f.idx for f in frames]
        assert indices[0] == 0
        assert indices[-1] == total - 1

    def test_single_frame_request(self):
        mock_cap = MagicMock()
        mock_cap.get.side_effect = lambda prop: {5: 30.0, 7: 100}.get(prop, 0)
        mock_cap.read.return_value = (True, np.zeros((50, 50, 3), dtype=np.uint8))

        with patch("video_judge.process.cv2.VideoCapture", return_value=mock_cap):
            with patch("video_judge.process.cv2.cvtColor", side_effect=lambda f, _: f):
                with patch(
                    "video_judge.process.cv2.imencode",
                    return_value=(True, np.array([0], dtype=np.uint8)),
                ):
                    frames = sample_frames("/fake/video.mp4", num_frames=1)

        assert len(frames) == 1
        assert frames[0].idx == 0
