"""
Streamlit frontend for the RAG Email Reply Generator.
Provides an interactive UI for composing emails and generating AI-powered replies.
"""

import time
from typing import Any, Dict, Optional

import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="RAG Email Reply Generator",
    layout="wide",
    page_icon="📧",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1a73e8;
        margin-bottom: 0.25rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #5f6368;
        margin-bottom: 1.5rem;
    }
    .card {
        background: #f8f9fa;
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid #e0e0e0;
        margin-bottom: 1rem;
    }
    .intent-badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
        margin-bottom: 0.5rem;
    }
    .context-card {
        background: #fff;
        border-left: 4px solid #1a73e8;
        padding: 0.8rem 1rem;
        margin-bottom: 0.5rem;
        border-radius: 0 8px 8px 0;
        font-size: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Intent colour mapping
# ---------------------------------------------------------------------------

INTENT_COLORS: Dict[str, str] = {
    "inquiry": "#1a73e8",
    "complaint": "#d93025",
    "request": "#f9ab00",
    "follow_up": "#1e8e3e",
    "thank_you": "#a142f4",
    "scheduling": "#e8710a",
    "introduction": "#00897b",
    "other": "#5f6368",
}

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## ⚙️ Settings")
    backend_url = st.text_input(
        "Backend URL", value="http://localhost:8000", help="FastAPI backend address"
    )
    tone = st.selectbox(
        "Reply Tone",
        options=["formal", "friendly", "professional"],
        index=0,
        help="Tone of the generated email reply",
    )
    top_k = st.slider(
        "Top-K Retrieved Contexts",
        min_value=1,
        max_value=10,
        value=5,
        help="Number of similar past emails to retrieve for context",
    )

    st.markdown("---")
    st.markdown("## ℹ️ About")
    st.markdown(
        """
        **RAG Email Reply Generator** uses
        *Retrieval-Augmented Generation* to produce
        context-aware, professional email replies.

        1. Your email is embedded with Sentence-BERT.
        2. Similar past emails are retrieved from ChromaDB.
        3. An LLM generates a reply grounded in that context.
        """
    )

    # Backend health check
    st.markdown("---")
    if st.button("🔍 Check Backend"):
        try:
            r = requests.get(f"{backend_url}/", timeout=5)
            if r.status_code == 200:
                st.success("✅ Backend is running")
            else:
                st.error(f"⚠️ Backend returned {r.status_code}")
        except requests.exceptions.ConnectionError:
            st.error("❌ Cannot connect to backend")

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------

st.markdown('<p class="main-header">📧 RAG Email Reply Generator</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Generate context-aware, professional email replies '
    "powered by Retrieval-Augmented Generation.</p>",
    unsafe_allow_html=True,
)

left_col, right_col = st.columns([1, 1], gap="large")

# ---------------------------------------------------------------------------
# Input column
# ---------------------------------------------------------------------------

with left_col:
    st.markdown("### ✉️ Compose Email")

    subject = st.text_input("Subject *", placeholder="e.g. Question about pricing plans")
    body = st.text_area(
        "Body *",
        height=200,
        placeholder="Write your email body here…",
    )
    sender = st.text_input(
        "Sender Email (optional)", placeholder="sender@example.com"
    )

    generate_button = st.button("🚀 Generate Reply", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# Output column
# ---------------------------------------------------------------------------

with right_col:
    st.markdown("### 💬 Generated Reply")

    if generate_button:
        if not subject or not body:
            st.warning("⚠️ Please fill in both **Subject** and **Body** fields.")
        else:
            with st.spinner("Retrieving context and generating reply…"):
                start = time.time()
                try:
                    response = requests.post(
                        f"{backend_url}/api/v1/generate_reply",
                        json={
                            "subject": subject,
                            "body": body,
                            "sender": sender or None,
                            "tone": tone,
                        },
                        timeout=30,
                    )
                    elapsed = time.time() - start

                    if response.status_code == 200:
                        data: Dict[str, Any] = response.json()

                        # Intent badge
                        intent = data.get("intent", "other")
                        badge_color = INTENT_COLORS.get(intent, "#5f6368")
                        st.markdown(
                            f'<span class="intent-badge" style="background:{badge_color};color:white;">'
                            f"🏷️ {intent.replace('_', ' ').title()}</span>",
                            unsafe_allow_html=True,
                        )

                        # Confidence
                        confidence = data.get("confidence_score", 0.0)
                        st.markdown(f"**Confidence:** {confidence:.1%}")
                        st.progress(confidence)

                        # Reply text
                        reply_text = data.get("reply_text", "")
                        st.text_area(
                            "Generated Reply (editable)",
                            value=reply_text,
                            height=280,
                            key="reply_output",
                        )

                        # Retrieved contexts
                        contexts = data.get("retrieved_contexts", [])
                        if contexts:
                            with st.expander(
                                f"📚 Retrieved Context ({len(contexts)} emails)", expanded=False
                            ):
                                for i, ctx in enumerate(contexts, 1):
                                    meta = ctx.get("metadata", {})
                                    score = ctx.get("score", 0.0)
                                    subject_ctx = meta.get("subject", "N/A")
                                    intent_ctx = meta.get("intent", "N/A")
                                    st.markdown(
                                        f'<div class="context-card">'
                                        f"<strong>#{i} — {subject_ctx}</strong> "
                                        f"<em>({intent_ctx})</em> "
                                        f"· score: {score:.3f}<br>"
                                        f"{ctx.get('content', '')[:200]}…"
                                        f"</div>",
                                        unsafe_allow_html=True,
                                    )

                        # Metadata
                        with st.expander("🔧 Metadata", expanded=False):
                            metadata = data.get("metadata", {})
                            st.json(
                                {
                                    **metadata,
                                    "api_response_time_seconds": round(elapsed, 3),
                                }
                            )
                    else:
                        st.error(
                            f"❌ Backend returned {response.status_code}: "
                            f"{response.text[:300]}"
                        )

                except requests.exceptions.ConnectionError:
                    st.error(
                        "❌ Could not connect to the backend. "
                        f"Is the server running at **{backend_url}**?"
                    )
                except requests.exceptions.Timeout:
                    st.error("⏳ Request timed out. The backend may be overloaded.")
                except Exception as exc:  # noqa: BLE001
                    st.error(f"❌ Unexpected error: {exc}")
    else:
        st.info("👈 Fill in the email details and click **Generate Reply**.")
