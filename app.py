"""Streamlit UI for the Auto Video Montage system."""
from __future__ import annotations
import os
import json
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="Auto Video Montage",
    page_icon="🎬",
    layout="wide",
)

st.title("🎬 Auto Video Montage")
st.caption("AI-powered video montage using multi-agent orchestration")


# ── Sidebar config ────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")

    api_key = st.text_input(
        "Anthropic API Key",
        value=os.environ.get("ANTHROPIC_API_KEY", ""),
        type="password",
        help="Your Anthropic API key. Will be set as environment variable.",
    )
    if api_key:
        os.environ["ANTHROPIC_API_KEY"] = api_key

    max_iter = st.slider(
        "Max revision iterations",
        min_value=1,
        max_value=5,
        value=3,
        help="How many times the CRITIC→REVISION→SCENARIO loop can run",
    )

    quality_threshold = st.slider(
        "Quality threshold",
        min_value=0.5,
        max_value=0.95,
        value=0.70,
        step=0.05,
        help="Minimum critic score to accept the montage (0.0–1.0)",
    )


# ── Main area ─────────────────────────────────────────────────────────────────
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1. Upload Rush Files")
    uploaded_files = st.file_uploader(
        "Drop your video rushes here",
        accept_multiple_files=True,
        type=["mp4", "mov", "avi", "mkv"],
        help="Upload one or more raw video files to montage",
    )

    if uploaded_files:
        st.success(f"{len(uploaded_files)} file(s) uploaded")
        for uf in uploaded_files:
            st.text(f"  • {uf.name} ({uf.size / 1024 / 1024:.1f} MB)")

with col2:
    st.subheader("2. Generate Montage")

    run_button = st.button(
        "🚀 Generate Montage",
        type="primary",
        disabled=not uploaded_files or not api_key,
    )

    if not api_key:
        st.warning("Set your Anthropic API key in the sidebar to continue.")
    if not uploaded_files:
        st.info("Upload at least one video file to get started.")


# ── Pipeline execution ────────────────────────────────────────────────────────
if run_button and uploaded_files and api_key:
    # Save uploads to a persistent dir (not tempdir) so moviepy doesn't lock files
    # that Windows can't delete while still open
    import uuid
    session_dir = Path("output") / f"session_{uuid.uuid4().hex[:8]}"
    session_dir.mkdir(parents=True, exist_ok=True)

    rush_paths = []
    for uf in uploaded_files:
        dest = session_dir / uf.name
        dest.write_bytes(uf.read())
        rush_paths.append(str(dest))

    # Override config threshold
    import src.config as cfg
    cfg.QUALITY_THRESHOLD = quality_threshold

    from src.orchestrator import Orchestrator

    orchestrator = Orchestrator(max_iterations=max_iter)

    # Live progress display
    st.divider()
    st.subheader("3. Pipeline Progress")

    log_lines = []
    log_placeholder = st.empty()

    node_colors = {
        "START": "🟢", "ANALYZE": "🔵", "SCENARIO": "🟣", "EDIT": "🟡",
        "CRITIC": "🔴", "REVISION": "🟠", "QUALITY": "🟤", "END": "⚪",
    }

    def progress_callback(node: str, message: str):
        icon = node_colors.get(node, "⚫")
        log_lines.append(f"{icon} **{node}** — {message}")
        log_placeholder.markdown("\n\n".join(log_lines))

    with st.spinner("Running AI agents..."):
        state = orchestrator.run(rush_paths, progress_callback=progress_callback)

    # ── Results ───────────────────────────────────────────────────────────
    st.divider()
    st.subheader("4. Results")

    if state.final_output_path and Path(state.final_output_path).exists():
        score_str = f"{state.critic_feedback.score:.2f}" if state.critic_feedback else "N/A"
        st.success(f"✅ Montage complete! Score: {score_str}")

        with open(state.final_output_path, "rb") as f:
            video_bytes = f.read()
        st.video(video_bytes)

        dl_cols = st.columns([2, 1, 1])
        with dl_cols[0]:
            st.download_button(
                label="⬇️ Download Montage (.mp4)",
                data=video_bytes,
                file_name="montage.mp4",
                mime="video/mp4",
                use_container_width=True,
            )
        with dl_cols[1]:
            if state.exports.get("edl") and Path(state.exports["edl"]).exists():
                st.download_button(
                    label="⬇️ EDL (DaVinci)",
                    data=Path(state.exports["edl"]).read_bytes(),
                    file_name=Path(state.exports["edl"]).name,
                    mime="text/plain",
                    use_container_width=True,
                    help="CMX 3600 EDL — import in DaVinci Resolve via File > Import Timeline",
                )
        with dl_cols[2]:
            if state.exports.get("fcpxml") and Path(state.exports["fcpxml"]).exists():
                st.download_button(
                    label="⬇️ FCPXML (DaVinci)",
                    data=Path(state.exports["fcpxml"]).read_bytes(),
                    file_name=Path(state.exports["fcpxml"]).name,
                    mime="application/xml",
                    use_container_width=True,
                    help="Final Cut Pro XML — import in DaVinci Resolve via File > Import Timeline",
                )

        with st.expander("📋 Edit Plan Details"):
            if state.edit_plan:
                st.markdown(f"**Title:** {state.edit_plan.title}")
                st.markdown(f"**Narrative arc:** {state.edit_plan.narrative_arc}")
                st.markdown(f"**Duration:** {state.edit_plan.total_duration:.1f}s")
                st.markdown(f"**Cuts:** {len(state.edit_plan.edits)}")
                if state.edit_plan.music_suggestion:
                    st.markdown(f"**Music suggestion:** {state.edit_plan.music_suggestion}")

        with st.expander("🎬 Critic Feedback"):
            if state.critic_feedback:
                score = state.critic_feedback.score
                st.progress(score, text=f"Quality Score: {score:.0%}")
                st.markdown(f"**Pacing:** {state.critic_feedback.pacing_notes}")
                st.markdown(f"**Narrative:** {state.critic_feedback.narrative_notes}")
                st.markdown(f"**Technical:** {state.critic_feedback.technical_notes}")

        with st.expander("📊 Full Orchestration Log"):
            st.json(state.model_dump(exclude_none=True))

    elif state.error:
        st.error(f"❌ Pipeline failed: {state.error}")
        with st.expander("Error details"):
            st.code(state.error)
    else:
        st.warning("Pipeline ended without output. Check the logs above.")


# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Powered by Claude (Anthropic) · moviepy · ffmpeg · "
    "Agents: ANALYZER → SCENARIO → EDITOR → CRITIC → REVISION → QUALITY"
)
