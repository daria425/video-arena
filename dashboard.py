import streamlit as st
import json
from video_judge import VideoGenArena, VideoGenModelConfig, OpenAIJudge, OpenAIDecomposer, ClaudeDecomposer, ClaudeJudge, GeminiDecomposer, GeminiJudge, BaseDecomposer, BaseJudge
from video_judge.config.logger import setup_default_logging
JUDGES = {
    "OpenAI": OpenAIJudge,
    "Gemini": GeminiJudge,
    "Claude": ClaudeJudge,
}
DECOMPOSERS = {
    "OpenAI": OpenAIDecomposer,
    "Gemini": GeminiDecomposer,
    "Claude": ClaudeDecomposer,
}
setup_default_logging(level=20)
st.set_page_config(page_title="Video Generation Arena", layout="wide")
st.title("Video Generation Arena")
st.caption("A comparison of text-to-video models across multiple evaluation criteria using automated LLM-as-a-judge evals.")
with open("model_config.json", "r") as f:
    model_config_data = json.load(f)
available_models = {
    m["model_id"]: m for m in model_config_data["models"]
}
selected_models = st.multiselect(options=list(
    available_models.keys()), label="Select video generation models to compare")

st.subheader("Evaluation Setup")
judge_selection = st.selectbox(label="LLM Judge Model", options=[
                               "OpenAI", "Claude", "Gemini"])

decomposer_selection = st.selectbox(label="Prompt Decomposer Model (model used to extract key details from the prompt)", options=[
    "OpenAI", "Claude", "Gemini"])
prompt = st.text_area(label="Video Generation Prompt", value="A sleek sci-fi rocketship launching vertically from the center of a vast lavender field at sunset. Endless rows of blooming purple lavender stretch toward the horizon, gently swaying from the rocket’s exhaust. The sky is filled with soft purple and pink clouds, glowing with warm golden sunset light. The rocket emits a bright white-violet flame and glowing thrusters, creating swirling dust and petals near the ground. Cinematic wide shot, epic scale, fantasy sci-fi atmosphere, soft volumetric lighting, shallow haze near the horizon, high detail, smooth motion, dramatic yet serene mood.")
if st.button("Fight!"):
    with st.spinner("Generating videos and running evals..."):
        judge: BaseJudge = JUDGES[judge_selection]()
        decomposer: BaseDecomposer = DECOMPOSERS[decomposer_selection]()
        decomposition = decomposer.decompose(user_prompt=prompt)
        # orchestrator = VideoEvaluationOrchestrator(
        #     video_gen_prompt=prompt, prompt_decomposition=decomposition)
        configs = [
            VideoGenModelConfig(provider=available_models[m]["provider"], model_id=m) for m in selected_models
        ]
        arena = VideoGenArena(model_configs=configs, judge=judge)
        result = arena.fight(
            video_gen_prompt=prompt, existing_video_path=None, prompt_decomposition=decomposition)
        st.session_state["latest_result"] = result
    st.success(f"Evaluation complete! Winner: {result.winner}")
    st.rerun()

if "latest_result" in st.session_state:
    result = st.session_state["latest_result"]
    st.subheader("Final Rankings")
    for i, model in enumerate(result.rankings, 1):
        score = next(r.report.scores['overall']
                     for r in result.results if r.model == model)
        st.write(f"**{model}** — {score:.3f}")
    cols = st.columns(len(result.results))

    for col, run in zip(cols, result.results):
        with col:
            st.subheader(run.model)
            st.video(run.report.video_path, width=300)
            st.metric("Overall Score", f"{run.report.scores['overall']:.2f}")

            with st.expander("Breakdown"):
                for criteria in ["prompt_alignment", "temporal_consistency",
                                 "aesthetic_quality", "technical_quality"]:
                    score = run.report.scores[criteria]
                    st.write(
                        f"**{criteria.replace('_', ' ').title()}**: {score:.2f}")

            # Evidence
            with st.expander("Evidence"):
                for detail in run.report.details:
                    st.write(f"**{detail['criteria']}**")
                    st.caption(detail['reasoning'])
                    if 'evidence' in detail:
                        for ev in detail['evidence']:
                            st.write(f"- Frame {ev['frame']}: {ev['finding']}")
