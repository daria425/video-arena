from typing import List, Optional
from pathlib import Path
from datetime import datetime
from utils.format import format_prompt
from utils.calculate import calculate_overall_score
from config.logger import logger
from judge import BaseJudge
from models import JudgeEval, Report, VideoInfo
from process import sample_frames
from video_gen import BaseVideoGenerator
from corruption import VideoInterceptor
from models import InterceptorConfig, Report


class VideoEvaluationOrchestrator:
    def __init__(self, video_gen_prompt: str, existing_video_path: Optional[str] = None, intercept_video: Optional[bool] = False):
        self.video_gen_prompt = video_gen_prompt
        self.input_data = {}
        self.use_interceptor = intercept_video
        self.existing_video_path = existing_video_path

    def node(self, images: List[bytes], user_prompts: List[str], judge: BaseJudge, prompt_criterion: str):
        system_prompt = format_prompt(f"./prompts/{prompt_criterion}.txt")
        logger.info(f"Evaluating {prompt_criterion}")
        return judge.evaluate(images=images, user_prompts=user_prompts, system_prompt=system_prompt)

    def alignment_node(self, images: List[bytes], user_prompts: List[str], judge: BaseJudge) -> JudgeEval:
        return self.node(images=images, user_prompts=user_prompts, judge=judge, prompt_criterion="prompt_alignment")

    def temporal_consistency_node(self, images: List[bytes], user_prompts: List[str], judge: BaseJudge):
        return self.node(images=images, user_prompts=user_prompts, judge=judge, prompt_criterion="temporal_consistency")

    def aesthetic_quality_node(self, images: List[bytes], user_prompts: List[str], judge: BaseJudge) -> JudgeEval:
        return self.node(images=images, user_prompts=user_prompts, judge=judge, prompt_criterion="aesthetic_quality")

    def technical_quality_node(self, images: List[bytes], user_prompts: List[str], judge: BaseJudge):
        return self.node(images=images, user_prompts=user_prompts, judge=judge, prompt_criterion="technical_quality")

    def _judge_input_from_video_generator(self, video_generator: BaseVideoGenerator) -> VideoInfo:
        return video_generator.run_video_gen(self.video_gen_prompt)

    def create_judge_input_from_generator(self, video_generator: BaseVideoGenerator, interceptor_config: Optional[InterceptorConfig] = None) -> tuple:
        video_info = video_generator.run_video_gen(self.video_gen_prompt)
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

    def create_judge_input_from_video(self, interceptor_config: Optional[InterceptorConfig] = None):
        video_id = Path(self.existing_video_path).stem
        video_prompt = self.video_gen_prompt
        video_path = self.existing_video_path
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
        frames = sample_frames(self.existing_video_path)
        image_bytes_list = [img.image for img in frames]
        user_prompts = [
            f"Frame {f.idx} at {f.timestamp_s:.2f}s"
            for f in frames
        ]
        # Add generation prompt at end for llm
        user_prompts.append(f"Original prompt: {video_prompt}")
        return (image_bytes_list, user_prompts)

    def run_nodes(self, images: List[bytes], user_prompts: List[str], judge: BaseJudge) -> Report:
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
        aesthetic_quality_response = self.aesthetic_quality_node(
            images=images, user_prompts=user_prompts, judge=judge)
        details.append(
            {
                "criteria": "aesthetic_quality",
                "score": aesthetic_quality_response.score,
                "reasoning": aesthetic_quality_response.reason
            }
        )
        scores["aesthetic_quality"] = aesthetic_quality_response.score
        technical_quality_response = self.technical_quality_node(
            images=images, user_prompts=user_prompts, judge=judge)
        scores["technical_quality"] = technical_quality_response.score
        details.append(
            {
                "criteria": "technical_quality",
                "score": technical_quality_response.score,
                "reasoning": technical_quality_response.reason
            }
        )
        overall = calculate_overall_score(
            scores=[scores["prompt_alignment"], scores["temporal_consistency"], scores["aesthetic_quality"], scores["technical_quality"]
                    ], weights=[0.5, 0.3, 0.1, 0.1])
        scores["overall"] = overall
        if overall >= 0.8:
            verdict = "pass"
        elif overall >= 0.6:
            verdict = "needs_review"
        else:
            verdict = "fail"
        return Report(input=self.input_data, scores=scores,
                      verdict=verdict, details=details)

    def run(self, judge: BaseJudge, video_generator: BaseVideoGenerator, interceptor_config: Optional[InterceptorConfig] = None) -> Report:
        if self.existing_video_path:
            images, user_prompts = self.create_judge_input_from_video(
                interceptor_config)
        else:
            # generate new
            images, user_prompts = self.create_judge_input_from_generator(
                video_generator=video_generator, interceptor_config=interceptor_config)
        report = self.run_nodes(
            images=images, user_prompts=user_prompts, judge=judge)
        with open(f"output/evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", "w") as f:
            f.write(report.model_dump_json(indent=2))
        return report
