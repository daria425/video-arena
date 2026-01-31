from typing import List, Optional
import json
from pathlib import Path
from functools import wraps
from datetime import datetime
from utils.format import format_prompt
from utils.calculate import calculate_overall_score
from config.logger import logger
from judge import BaseJudge
from models import JudgeEval, Report
from process import sample_frames
from video_gen import VideoGenerator
from corruption import VideoInterceptor
from models import InterceptorConfig, Report


def retry_on_failure(max_attempts=3, pass_threshold=0.8):
    """Decorator to retry video generation if evaluation fails"""
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            for attempt in range(1, max_attempts + 1):
                logger.info(f"Attempt {attempt}/{max_attempts}")

                result: Report = func(self, *args, **kwargs)

                logger.info(
                    f"Overall: {result.scores['overall']:.2f} | Verdict: {result.verdict}")

                # Check if passed
                if result.verdict == "pass":
                    logger.info(f"Passed on attempt {attempt}!")
                    result.input["total_attempts"] = attempt
                    return result

                # Log failures
                logger.warning("Evaluation failed:")
                for detail in result.details:
                    if detail["score"] < pass_threshold:
                        logger.warning(
                            f"  - {detail['criteria']}: {detail['score']:.2f}")

                # Decide to retry or stop
                if attempt < max_attempts:
                    logger.debug("Regenerating video...")
                else:
                    logger.error("Max attempts reached.")
                    result.input["total_attempts"] = attempt
                    return result

            return result
        return wrapper
    return decorator


class VideoEvaluationOrchestrator:
    def __init__(self, video_gen_prompt: str, intercept_video: Optional[bool] = False):
        self.video_gen_prompt = video_gen_prompt
        self.input_data = {}
        self.use_interceptor = intercept_video

    def node(self, images: List[bytes], user_prompts: List[str], judge: BaseJudge, prompt_criterion: str):
        system_prompt = format_prompt(f"./prompts/{prompt_criterion}.txt")
        return judge.evaluate(images=images, user_prompts=user_prompts, system_prompt=system_prompt)

    def alignment_node(self, images: List[bytes], user_prompts: List[str], judge: BaseJudge) -> JudgeEval:
        return self.node(images=images, user_prompts=user_prompts, judge=judge, prompt_criterion="prompt_alignment")

    def temporal_consistency_node(self, images: List[bytes], user_prompts: List[str], judge: BaseJudge):
        return self.node(images=images, user_prompts=user_prompts, judge=judge, prompt_criterion="temporal_consistency")

    def create_judge_input(self, video_generator: VideoGenerator, interceptor_config: Optional[InterceptorConfig] = None) -> tuple:
        video_info = video_generator.run(self.video_gen_prompt)
        video_id = Path(video_info.saved_path).stem
        video_prompt = self.video_gen_prompt
        video_path = video_info.saved_path
        if self.use_interceptor:
            interceptor = VideoInterceptor(
                video_path, video_prompt, interceptor_config)
            new_data = interceptor.intercept()
            video_prompt = new_data.new_prompt
            video_path = new_data.new_video_path
        self.input_data = {
            "prompt": video_prompt,
            "video_id": video_id
            # add duration, num frames, fps etc later
        }
        frames = sample_frames(video_path)
        image_bytes_list = [img.image for img in frames]
        user_prompts = [
            f"Frame {f.idx} at {f.timestamp_s:.2f}s"
            for f in frames
        ]
        # Add generation prompt at end for llm
        user_prompts.append(f"Original prompt: {video_prompt}")
        return (image_bytes_list, user_prompts)

    @retry_on_failure()
    def run(self, judge: BaseJudge, video_generator: VideoGenerator, interceptor_config: Optional[InterceptorConfig] = None) -> Report:
        images, user_prompts = self.create_judge_input(
            video_generator=video_generator, interceptor_config=interceptor_config)
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
        report = Report(input=self.input_data, scores=scores,
                        verdict=verdict, details=details)
        with open(f"output/evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", "w") as f:
            f.write(report.model_dump_json(indent=2))
        return report
