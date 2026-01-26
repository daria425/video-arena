from typing import List
from pathlib import Path
from functools import wraps
from utils.format import format_prompt
from utils.calculate import calculate_overall_score
from judge import BaseJudge
from models import JudgeEval, Report
from process import sample_frames
from video_gen import VideoGenerator


def retry_on_failure(max_attempts=3, pass_threshold=0.8):
    """Decorator to retry video generation if evaluation fails"""
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            for attempt in range(1, max_attempts + 1):
                print(f"\n{'=' * 50}")
                print(f"Attempt {attempt}/{max_attempts}")
                print(f"{'=' * 50}")

                result: Report = func(self, *args, **kwargs)

                print(
                    f"Overall: {result.scores['overall']:.2f} | Verdict: {result.verdict}")

                # Check if passed
                if result.verdict == "pass":
                    print(f"✓ Passed on attempt {attempt}!")
                    result.input["total_attempts"] = attempt
                    return result

                # Log failures
                print("Evaluation failed:")
                for detail in result.details:
                    if detail["score"] < pass_threshold:
                        print(
                            f"  - {detail['criteria']}: {detail['score']:.2f}")

                # Decide to retry or stop
                if attempt < max_attempts:
                    print("Regenerating video...")
                else:
                    print("Max attempts reached.")
                    result.input["total_attempts"] = attempt
                    return result

            return result
        return wrapper
    return decorator


class VideoEvaluationOrchestrator:
    def __init__(self, video_gen_prompt: str):
        self.video_gen_prompt = video_gen_prompt
        self.input_data = {}

    def node(self, images: List[bytes], user_prompts: List[str], judge: BaseJudge, prompt_criterion: str):
        system_prompt = format_prompt(f"./prompts/{prompt_criterion}.txt")
        return judge.evaluate(images=images, user_prompts=user_prompts, system_prompt=system_prompt)

    def alignment_node(self, images: List[bytes], user_prompts: List[str], judge: BaseJudge) -> JudgeEval:
        return self.node(images=images, user_prompts=user_prompts, judge=judge, prompt_criterion="prompt_alignment")

    def temporal_consistency_node(self, images: List[bytes], user_prompts: List[str], judge: BaseJudge):
        return self.node(images=images, user_prompts=user_prompts, judge=judge, prompt_criterion="temporal_consistency")

    def create_judge_input(self, video_generator: VideoGenerator) -> tuple:
        video_info = video_generator.run(self.video_gen_prompt)
        video_id = Path(video_info.saved_path).stem
        self.input_data = {
            "prompt": self.video_gen_prompt,
            "video_id": video_id
            # add duration, num frames, fps etc later
        }
        frames = sample_frames(video_info.saved_path)
        image_bytes_list = [img.image for img in frames]
        user_prompts = [
            f"Frame {f.idx} at {f.timestamp_s:.2f}s"
            for f in frames
        ]
        # Add generation prompt at end
        user_prompts.append(f"Original prompt: {self.video_gen_prompt}")
        return (image_bytes_list, user_prompts)

    @retry_on_failure()
    def run(self, judge: BaseJudge, video_generator: VideoGenerator) -> Report:
        images, user_prompts = self.create_judge_input(
            video_generator=video_generator)
        details = []
        scores = {}
        alignment_response = self.alignment_node(
            images=images, user_prompts=user_prompts, judge=judge)
        details.append(
            {
                "criteria": "prompt_alignment",
                "score": alignment_response.score,
                "reasoning": alignment_response.reason
            }
        )
        scores["prompt_alignment"] = alignment_response.score
        temporal_response = self.temporal_consistency_node(
            images=images, user_prompts=user_prompts, judge=judge)
        scores["temporal_consistency"] = temporal_response.score
        details.append(
            {
                "criteria": "temporal_consistency",
                "score": temporal_response.score,
                "reasoning": temporal_response.reason
            }
        )
        overall = calculate_overall_score(
            scores=[scores["prompt_alignment"], scores["temporal_consistency"]
                    ], weights=[0.6, 0.4])
        scores["overall"] = overall
        if overall >= 0.8:
            verdict = "pass"
        elif overall >= 0.6:
            verdict = "needs_review"
        else:
            verdict = "fail"

        return Report(input=self.input_data, scores=scores, verdict=verdict, details=details)
