"""Streamlit UI for the RAG Email Reply Generator."""
import logging
import os
from pathlib import Path

import requests
import streamlit as st

logger = logging.getLogger(__name__)

# Configuration
API_URL = os.getenv("API_URL", "http://localhost:8000")
CSS_PATH = Path(__file__).parent / "static" / "style.css"


def load_css() -> None:
    """Load custom CSS styling."""
    if CSS_PATH.exists():
        with open(CSS_PATH, "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def check_api_health(api_url: str) -> bool:
    """Check if the backend API is reachable."""
    try:
        response = requests.get(f"{api_url}/api/v1/health", timeout=5)
        return response.status_code == 200
    except Exception:
        return False


def generate_reply(
    api_url: str,
    subject: str,
    body: str,
    sender: str,
    tone: str,
    top_k: int,
) -> dict:
    """Send email to API and get generated reply."""
    response = requests.post(
        f"{api_url}/api/v1/generate_reply",
        json={
            "subject": subject,
            "body": body,
            "sender": sender if sender else None,
            "tone": tone,
        },
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


def main() -> None:
    """Main Streamlit application."""
    st.set_page_config(
        page_title="RAG Email Reply Generator",
        page_icon="✉️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    load_css()

    # Sidebar
    with st.sidebar:
        st.title("⚙️ Settings")
        st.divider()

        api_url = st.text_input(
            "API URL",
            value=API_URL,
            help="Backend API URL",
        )

        tone = st.selectbox(
            "Reply Tone",
            options=["formal", "friendly", "concise"],
            index=0,
            help="Controls the tone and style of the generated reply",
        )

        top_k = st.slider(
            "Retrieved Contexts",
            min_value=1,
            max_value=10,
            value=5,
            help="Number of similar past emails to retrieve as context",
        )

        st.divider()

        # API Health Status
        if st.button("🔍 Check API Status"):
            if check_api_health(api_url):
                st.success("✅ API is online")
            else:
                st.error("❌ API is offline. Start the backend with:\nuvicorn backend.main:app --host 0.0.0.0 --port 8000")

    # Main Content
    st.title("✉️ RAG Email Reply Generator")
    st.markdown(
        "Generate **context-aware, professional email replies** powered by "
        "Retrieval-Augmented Generation (RAG)."
    )

    # Input Form
    st.subheader("📧 Compose Incoming Email")
    col1, col2 = st.columns([2, 1])

    with col1:
        subject = st.text_input(
            "Subject *",
            placeholder="e.g., Product refund request",
            key="subject",
        )

    with col2:
        sender_email = st.text_input(
            "From (optional)",
            placeholder="customer@example.com",
            key="sender",
        )

    body = st.text_area(
        "Email Body *",
        placeholder="Paste the email content here...",
        height=200,
        key="body",
    )

    generate_btn = st.button("🚀 Generate Reply", type="primary", use_container_width=True)

    # Handle Generation
    if generate_btn:
        if not subject or not body:
            st.error("Please provide both the Subject and Email Body fields.")
        else:
            with st.spinner("Generating reply using RAG pipeline..."):
                try:
                    result = generate_reply(
                        api_url=api_url,
                        subject=subject,
                        body=body,
                        sender=sender_email,
                        tone=tone,
                        top_k=top_k,
                    )

                    st.success("✅ Reply generated successfully!")

                    # Generated Reply Section
                    st.subheader("📝 Generated Reply")
                    reply_text = st.text_area(
                        "Reply (editable)",
                        value=result.get("reply_text", ""),
                        height=300,
                        key="reply_output",
                    )

                    # Copy to clipboard hint
                    st.caption("✏️ You can edit the reply above before sending.")

                    # Metadata Row
                    col_m1, col_m2, col_m3 = st.columns(3)
                    with col_m1:
                        intent = result.get("intent", "unknown")
                        st.metric("🎯 Intent", intent.replace("_", " ").title())
                    with col_m2:
                        confidence = result.get("confidence_score", 0)
                        st.metric("📊 Confidence", f"{confidence:.0%}")
                    with col_m3:
                        num_ctx = len(result.get("retrieved_contexts", []))
                        st.metric("📚 Contexts Used", num_ctx)

                    # Retrieved Context
                    contexts = result.get("retrieved_contexts", [])
                    if contexts:
                        st.subheader("🔍 Retrieved Context Documents")
                        for i, ctx in enumerate(contexts, 1):
                            score = ctx.get("similarity_score", 0)
                            source = ctx.get("source", "unknown")
                            with st.expander(
                                f"Context {i} — Score: {score:.2%} | Source: {source}"
                            ):
                                st.text(ctx.get("content", ""))
                                meta = ctx.get("metadata", {})
                                if meta:
                                    st.json(meta)

                    # Additional Metadata
                    meta = result.get("metadata", {})
                    if meta:
                        with st.expander("📊 Pipeline Metadata"):
                            col_e1, col_e2 = st.columns(2)
                            with col_e1:
                                if "elapsed_ms" in meta:
                                    st.metric("⏱️ Generation Time", f"{meta['elapsed_ms']}ms")
                                if "llm_provider" in meta:
                                    st.info(f"LLM: **{meta['llm_provider']}** / **{meta.get('model', 'N/A')}**")
                            with col_e2:
                                if "entities" in meta:
                                    entities = meta["entities"]
                                    if any(entities.values()):
                                        st.write("**Extracted Entities:**")
                                        for key, vals in entities.items():
                                            if vals:
                                                st.write(f"- {key}: {', '.join(str(v) for v in vals[:3])}")

                except requests.exceptions.ConnectionError:
                    st.error(
                        "❌ Cannot connect to the API. Please start the backend:\n\n"
                        "```bash\nuvicorn backend.main:app --host 0.0.0.0 --port 8000\n```"
                    )
                except requests.exceptions.HTTPError as e:
                    st.error(f"❌ API error: {e.response.status_code} — {e.response.text}")
                except Exception as exc:
                    st.error(f"❌ Unexpected error: {exc}")

    # Footer
    st.divider()
    st.markdown(
        "<div style='text-align: center; color: gray; font-size: 0.8em;'>"
        "RAG Email Reply Generator — Powered by Retrieval-Augmented Generation"
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
