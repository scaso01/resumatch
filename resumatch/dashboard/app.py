"""
ResuMatch Dashboard - Streamlit Application
=============================================
Upload resumes, view scores, feedback, and JD matching results.
"""

import os

import httpx
import streamlit as st

from resumatch.dashboard.components import (
    render_feedback_list,
    render_improvements,
    render_jd_match,
    render_radar_chart,
    render_score_card,
)

API_URL = os.getenv("RESUMATCH_API_URL", "http://localhost:8510")
MAX_FILE_SIZE_MB = 10


def _css():
    css_path = os.path.join(os.path.dirname(__file__), "style.css")
    if os.path.exists(css_path):
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def main():
    st.set_page_config(
        page_title="ResuMatch",
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _css()

    st.title("ResuMatch")
    st.caption("Open-source resume scoring engine")

    # Sidebar
    with st.sidebar:
        st.header("Upload Resume")
        uploaded_file = st.file_uploader(
            "Choose a PDF or DOCX file",
            type=["pdf", "docx"],
            help=f"Max {MAX_FILE_SIZE_MB}MB",
        )

        st.divider()
        st.header("Job Description (Optional)")
        jd_text = st.text_area(
            "Paste the job description",
            height=200,
            placeholder="Paste a job description here to see how well the resume matches...",
        )

        analyze_btn = st.button(
            "Analyze Resume",
            type="primary",
            use_container_width=True,
            disabled=uploaded_file is None,
        )

    # Main content
    if analyze_btn and uploaded_file is not None:
        if uploaded_file.size > MAX_FILE_SIZE_MB * 1024 * 1024:
            st.error(f"File exceeds {MAX_FILE_SIZE_MB}MB limit.")
            return

        with st.spinner("Analyzing resume..."):
            try:
                result = _call_api(uploaded_file, jd_text)
            except httpx.ConnectError:
                st.error(f"Cannot connect to API at {API_URL}. Is the server running?")
                return
            except httpx.HTTPStatusError as e:
                st.error(f"API error: {e.response.status_code} — {e.response.text}")
                return
            except Exception as e:
                st.error(f"Unexpected error: {e}")
                return

        st.session_state["result"] = result
        st.session_state["filename"] = uploaded_file.name
        st.session_state["improve_result"] = None

    # Display results
    if "result" in st.session_state:
        result = st.session_state["result"]
        score = result["score"]
        feedback = result["feedback"]

        st.subheader(f"Results for {st.session_state.get('filename', 'resume')}")

        # Score overview row
        col1, col2 = st.columns([1, 2])

        with col1:
            render_score_card(score)

        with col2:
            render_radar_chart(score)

        st.divider()

        # Improve button
        if st.button("Improve Resume", type="secondary", use_container_width=True):
            with st.spinner("Identifying weak bullets and generating improvements..."):
                try:
                    improve_resp = httpx.post(
                        f"{API_URL}/api/v1/improve",
                        json=result["resume"],
                        timeout=120,
                    )
                    improve_resp.raise_for_status()
                    st.session_state["improve_result"] = improve_resp.json()
                except Exception as e:
                    st.error(f"Improve failed: {e}")

        # Feedback and JD match tabs
        tabs = ["Feedback"]
        if result.get("jd_match"):
            tabs.append("JD Match")
        if st.session_state.get("improve_result"):
            tabs.append("Improvements")
        tabs.append("Parsed Resume")

        tab_objects = st.tabs(tabs)

        with tab_objects[0]:
            render_feedback_list(feedback)

        if result.get("jd_match"):
            with tab_objects[1]:
                render_jd_match(result["jd_match"])

        if st.session_state.get("improve_result") and "Improvements" in tabs:
            imp_idx = tabs.index("Improvements")
            with tab_objects[imp_idx]:
                render_improvements(st.session_state["improve_result"])

        with tab_objects[-1]:
            resume = result.get("resume", {})
            if resume.get("contact"):
                st.json(resume["contact"])
            if resume.get("sections"):
                for section in resume["sections"]:
                    with st.expander(section.get("heading", section.get("name", "Section"))):
                        if section.get("bullets"):
                            for b in section["bullets"]:
                                st.markdown(f"- {b}")
                        elif section.get("content"):
                            st.text(section["content"])

    elif not analyze_btn:
        st.info("Upload a resume in the sidebar and click **Analyze Resume** to get started.")


def _call_api(uploaded_file, jd_text: str) -> dict:
    """Call the ResuMatch API."""
    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}

    if jd_text.strip():
        endpoint = f"{API_URL}/api/v1/analyze-with-jd"
        resp = httpx.post(endpoint, files=files, data={"jd_text": jd_text}, timeout=60)
    else:
        endpoint = f"{API_URL}/api/v1/analyze"
        resp = httpx.post(endpoint, files=files, timeout=60)

    resp.raise_for_status()
    return resp.json()


if __name__ == "__main__":
    main()
