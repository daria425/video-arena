import pytest
from video_judge.utils.calculate import calculate_overall_score


class TestCalculateOverallScore:
    def test_weighted_score(self):
        scores = [0.9, 0.8, 0.7, 0.6]
        weights = [0.5, 0.3, 0.1, 0.1]
        result = calculate_overall_score(scores, weights)
        expected = 0.9 * 0.5 + 0.8 * 0.3 + 0.7 * 0.1 + 0.6 * 0.1
        assert abs(result - expected) < 1e-9

    def test_equal_weights_when_none(self):
        scores = [0.8, 0.6]
        result = calculate_overall_score(scores, weights=None)
        assert abs(result - 0.7) < 1e-9

    def test_empty_scores_returns_zero(self):
        assert calculate_overall_score([]) == 0.0

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError, match="same length"):
            calculate_overall_score([0.5, 0.5], [0.5])

    def test_weights_not_summing_to_one_raises(self):
        with pytest.raises(ValueError, match="sum to 1.0"):
            calculate_overall_score([0.5, 0.5], [0.3, 0.3])
