import os
import tempfile
from video_judge.utils.format import format_prompt


class TestFormatPrompt:
    def test_replaces_placeholders(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Evaluate {{criteria}} for {{model}}")
            f.flush()
            result = format_prompt(f.name, criteria="alignment", model="sora")
        os.unlink(f.name)
        assert result == "Evaluate alignment for sora"

    def test_no_placeholders_returns_as_is(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Plain text prompt")
            f.flush()
            result = format_prompt(f.name)
        os.unlink(f.name)
        assert result == "Plain text prompt"
