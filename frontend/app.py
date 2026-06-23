"""
CampusGPT — Production Streamlit Frontend
Portfolio-quality UI for RGPV RAG Teaching Assistant.
"""

import os
import time
from datetime import datetime

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(
    page_title="CampusGPT",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ──────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .stApp { background: linear-gradient(160deg, #0a0f1e 0%, #111827 50%, #1a1040 100%); color: #f1f5f9; }
    .hero-title {
        font-size: 2.8rem; font-weight: 700; text-align: center;
        background: linear-gradient(90deg, #6366f1, #a855f7, #ec4899);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 4px;
    }
    .hero-sub { text-align: center; color: #94a3b8; font-size: 1.05rem; margin-bottom: 28px; }
    .metric-card {
        background: rgba(30,41,59,0.5); border: 1px solid rgba(99,102,241,0.2);
        border-radius: 14px; padding: 20px; backdrop-filter: blur(10px);
    }
    .metric-val { font-size: 2rem; font-weight: 700; color: #818cf8; }
    .metric-label { color: #94a3b8; font-size: 0.85rem; text-transform: uppercase; }
    .user-bubble {
        background: rgba(99,102,241,0.15); border-left: 4px solid #6366f1;
        padding: 14px 18px; border-radius: 12px; margin: 8px 0;
    }
    .assistant-bubble {
        background: rgba(30,41,59,0.7); border-left: 4px solid #a855f7;
        padding: 14px 18px; border-radius: 12px; margin: 8px 0;
    }
    .source-card {
        background: rgba(139,92,246,0.1); border: 1px solid rgba(139,92,246,0.25);
        border-radius: 8px; padding: 10px 14px; margin: 6px 0; font-size: 0.88rem;
    }
    .typing-indicator {
        display: inline-block; animation: pulse 1.2s infinite;
        color: #a855f7; font-style: italic;
    }
    @keyframes pulse { 0%,100%{opacity:.4} 50%{opacity:1} }
    section[data-testid="stSidebar"] { background: #0b0f19; border-right: 1px solid rgba(99,102,241,0.1); }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Session State ─────────────────────────────────────────────────────────────
def _init_state():
    defaults = {
        "token": None,
        "user": None,
        "session_id": f"session_{int(datetime.now().timestamp())}",
        "messages": [],
        "sources": [],
        "follow_ups": [],
        "confidence": 0.0,
        "last_q": "",
        "last_a": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()


def api_headers():
    h = {}
    if st.session_state.token:
        h["Authorization"] = f"Bearer {st.session_state.token}"
    return h


def api_get(path, **kwargs):
    return requests.get(f"{BACKEND_URL}{path}", headers=api_headers(), timeout=kwargs.pop("timeout", 30), **kwargs)


def api_post(path, **kwargs):
    return requests.post(f"{BACKEND_URL}{path}", headers=api_headers(), timeout=kwargs.pop("timeout", 120), **kwargs)


def api_delete(path, **kwargs):
    return requests.delete(f"{BACKEND_URL}{path}", headers=api_headers(), timeout=kwargs.pop("timeout", 30), **kwargs)


# ── Auth UI ───────────────────────────────────────────────────────────────────
def render_auth():
    st.sidebar.markdown("### 🔐 Account")
    if st.session_state.user:
        st.sidebar.success(f"Logged in as **{st.session_state.user.get('username', 'user')}**")
        if st.sidebar.button("Logout", use_container_width=True):
            try:
                api_post("/auth/logout")
            except Exception:
                pass
            st.session_state.token = None
            st.session_state.user = None
            st.rerun()
        return True

    tab_login, tab_reg = st.sidebar.tabs(["Login", "Register"])
    with tab_login:
        email = st.text_input("Email", key="login_email")
        password = st.text_input("Password", type="password", key="login_pass")
        if st.button("Sign In", use_container_width=True, key="btn_login"):
            try:
                r = api_post("/auth/login", json={"email": email, "password": password})
                if r.status_code == 200:
                    st.session_state.token = r.json()["access_token"]
                    me = api_get("/auth/me")
                    st.session_state.user = me.json()
                    st.rerun()
                else:
                    st.error(r.json().get("detail", "Login failed"))
            except Exception as e:
                st.error(f"Cannot reach backend: {e}")

    with tab_reg:
        reg_email = st.text_input("Email", key="reg_email")
        reg_user = st.text_input("Username", key="reg_user")
        reg_name = st.text_input("Full Name", key="reg_name")
        reg_pass = st.text_input("Password", type="password", key="reg_pass")
        if st.button("Create Account", use_container_width=True, key="btn_reg"):
            try:
                r = api_post(
                    "/auth/register",
                    json={
                        "email": reg_email,
                        "username": reg_user,
                        "password": reg_pass,
                        "full_name": reg_name,
                    },
                )
                if r.status_code == 201:
                    st.success("Account created! Please login.")
                else:
                    st.error(r.json().get("detail", "Registration failed"))
            except Exception as e:
                st.error(f"Cannot reach backend: {e}")
    return False


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("<div class='hero-title'>🎓 CampusGPT</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='hero-sub'>Production RAG Teaching Assistant for RGPV Students · Data Science · AI/ML · CS</div>",
    unsafe_allow_html=True,
)

logged_in = render_auth()

menu = st.sidebar.radio(
    "Navigation",
    [
        "💬 Chat",
        "📊 Analytics",
        "📤 Upload",
        "👍 Feedback",
        "🎯 RAG Performance",
        "⚙️ Admin",
    ],
    label_visibility="collapsed",
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Session ID**")
sid = st.text_input("Session", value=st.session_state.session_id, label_visibility="collapsed")
if sid != st.session_state.session_id:
    st.session_state.session_id = sid
    st.session_state.messages = []

if st.sidebar.button("🔄 Reset Chat", use_container_width=True):
    try:
        api_delete(f"/chat/history/{st.session_state.session_id}")
    except Exception:
        pass
    st.session_state.messages = []
    st.session_state.sources = []
    st.rerun()


# ── Pages ─────────────────────────────────────────────────────────────────────
if menu == "💬 Chat":
    if not logged_in:
        st.warning("Please login or register in the sidebar to use chat.")
    else:
        c1, c2 = st.columns([1, 3])
        with c1:
            st.markdown("#### ⚙️ Options")
            mode = st.selectbox(
                "Mode",
                ["Default RAG Chat", "Exam Prep", "Viva", "Quiz", "Assignment", "Important Questions"],
            )
            mode_map = {
                "Default RAG Chat": "chat",
                "Exam Prep": "exam_prep",
                "Viva": "viva",
                "Quiz": "quiz",
                "Assignment": "assignment",
                "Important Questions": "important_questions",
            }
            subject = st.text_input("Subject")
            semester = st.text_input("Semester")
            unit = st.text_input("Unit")
            topic = st.text_input("Topic")
            doc_type = st.selectbox("Doc Type", ["", "notes", "lecture", "assignment", "lab_manual", "question_paper"])
            use_mmr = st.checkbox("Diverse Retrieval (MMR)")
            top_k = st.slider("Top K", 1, 15, 5)
            if mode in ("Quiz", "Viva"):
                num_qs = st.slider("Num Questions", 1, 25, 10)
                difficulty = st.selectbox("Difficulty", ["easy", "medium", "hard"])
            else:
                num_qs, difficulty = 10, "medium"

        with c2:
            chat_box = st.container(height=480)
            with chat_box:
                for msg in st.session_state.messages:
                    cls = "user-bubble" if msg["role"] == "user" else "assistant-bubble"
                    icon = "🗣️" if msg["role"] == "user" else "🤖"
                    label = "You" if msg["role"] == "user" else "CampusGPT"
                    st.markdown(f"<div class='{cls}'>{icon} <b>{label}:</b><br>{msg['content']}</div>", unsafe_allow_html=True)

                if st.session_state.get("typing"):
                    st.markdown("<div class='typing-indicator'>CampusGPT is thinking...</div>", unsafe_allow_html=True)

            if st.session_state.sources:
                st.markdown("#### 📚 Sources")
                for src in st.session_state.sources:
                    label = src.get("citation_label") or f"{src.get('source','?')} | Page {src.get('page_number','?')}"
                    subj = src.get("subject", "")
                    sem = src.get("semester", "")
                    with st.expander(f"📄 {label}"):
                        st.markdown(f"**Subject:** {subj or 'N/A'} · **Semester:** {sem or 'N/A'} · **Topic:** {src.get('topic','N/A')}")
                        st.markdown(f"**Relevance:** {src.get('similarity_score', 0):.2f}")
                        if src.get("text_preview"):
                            st.caption(src["text_preview"])

            if st.session_state.confidence > 0:
                st.progress(st.session_state.confidence, text=f"Confidence: {st.session_state.confidence*100:.0f}%")

            # Feedback on last answer
            if st.session_state.last_a:
                fb1, fb2, _ = st.columns([1, 1, 4])
                with fb1:
                    if st.button("👍 Helpful"):
                        api_post("/feedback", json={
                            "question": st.session_state.last_q,
                            "answer": st.session_state.last_a,
                            "feedback": "helpful",
                            "session_id": st.session_state.session_id,
                        })
                        st.toast("Thanks for your feedback!")
                with fb2:
                    if st.button("👎 Not Helpful"):
                        api_post("/feedback", json={
                            "question": st.session_state.last_q,
                            "answer": st.session_state.last_a,
                            "feedback": "not_helpful",
                            "session_id": st.session_state.session_id,
                        })
                        st.toast("Feedback recorded.")

            query = st.text_input("Ask a question", placeholder="e.g. What is the data science lifecycle?")
            if st.button("Send", use_container_width=True) and query.strip():
                st.session_state.messages.append({"role": "user", "content": query})
                st.session_state.typing = True
                payload = {
                    "question": query,
                    "subject": subject,
                    "semester": semester,
                    "unit": unit,
                    "topic": topic,
                    "document_type": doc_type,
                    "session_id": st.session_state.session_id,
                    "top_k": top_k,
                    "use_mmr": use_mmr,
                    "mode": mode_map[mode],
                    "difficulty": difficulty,
                    "num_questions": num_qs,
                }
                try:
                    r = api_post("/chat", json=payload)
                    st.session_state.typing = False
                    if r.status_code == 200:
                        data = r.json()
                        ans = data["answer"]
                        st.session_state.messages.append({"role": "assistant", "content": ans})
                        st.session_state.sources = data.get("sources", [])
                        st.session_state.follow_ups = data.get("follow_up_questions", [])
                        st.session_state.confidence = data.get("confidence_score", 0)
                        st.session_state.last_q = query
                        st.session_state.last_a = ans
                        st.rerun()
                    else:
                        st.error(f"Error {r.status_code}: {r.text}")
                except Exception as e:
                    st.session_state.typing = False
                    st.error(f"Backend error: {e}")

elif menu == "📊 Analytics":
    if not logged_in:
        st.warning("Login required.")
    else:
        st.markdown("### 📊 System Analytics")
        try:
            r = api_get("/metrics")
            if r.status_code == 200:
                m = r.json()
                cols = st.columns(4)
                for col, (lbl, key) in zip(cols, [
                    ("Documents", "total_documents"), ("Chunks", "total_chunks"),
                    ("Queries", "total_queries"), ("Avg Latency", "avg_response_time"),
                ]):
                    val = m.get(key, 0)
                    col.markdown(f"<div class='metric-card'><div class='metric-val'>{val}</div><div class='metric-label'>{lbl}</div></div>", unsafe_allow_html=True)

                fb = m.get("feedback", {})
                if fb.get("total", 0) > 0:
                    st.markdown("#### 👍 User Satisfaction")
                    fcol1, fcol2 = st.columns(2)
                    fcol1.metric("Helpful", fb.get("helpful", 0))
                    fcol2.metric("Satisfaction Rate", f"{fb.get('satisfaction_rate', 0)*100:.1f}%")

                if m.get("topics_searched"):
                    df = pd.DataFrame(list(m["topics_searched"].items()), columns=["Topic", "Count"])
                    fig = px.bar(df, x="Topic", y="Count", color="Count", color_continuous_scale="Viridis")
                    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#f1f5f9")
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("Failed to load metrics")
        except Exception as e:
            st.error(f"Backend unreachable: {e}")

elif menu == "📤 Upload":
    if not logged_in:
        st.warning("Login required.")
    else:
        st.markdown("### 📤 Upload Course Material")
        with st.form("upload"):
            f = st.file_uploader("File", type=["pdf", "docx", "pptx", "txt"])
            c1, c2 = st.columns(2)
            with c1:
                subj = st.text_input("Subject *")
                sem = st.selectbox("Semester", ["", "1", "2", "3", "4", "5", "6", "7", "8"])
                unit = st.text_input("Unit")
            with c2:
                top = st.text_input("Topic")
                dtype = st.selectbox("Type", ["notes", "lecture", "assignment", "lab_manual", "question_paper", "ppt"])
                course = st.text_input("Course")
            if st.form_submit_button("Ingest", use_container_width=True):
                if f and subj:
                    with st.spinner("Ingesting & embedding..."):
                        try:
                            r = requests.post(
                                f"{BACKEND_URL}/ingest",
                                headers=api_headers(),
                                files={"file": (f.name, f.getvalue(), f.type)},
                                data={"subject": subj, "semester": sem, "unit": unit, "topic": top, "document_type": dtype, "course": course},
                                timeout=300,
                            )
                            if r.status_code == 200:
                                res = r.json()
                                st.success(f"Ingested {res['chunks']} chunks from {res['source']}")
                            else:
                                st.error(r.text)
                        except Exception as e:
                            st.error(str(e))
                else:
                    st.error("File and subject required.")

elif menu == "👍 Feedback":
    if not logged_in:
        st.warning("Login required.")
    else:
        st.markdown("### 👍 Feedback Analytics")
        try:
            r = api_get("/feedback/stats")
            if r.status_code == 200:
                s = r.json()
                c1, c2, c3 = st.columns(3)
                c1.metric("👍 Helpful", s["helpful"])
                c2.metric("👎 Not Helpful", s["not_helpful"])
                c3.metric("Satisfaction", f"{s['satisfaction_rate']*100:.1f}%")
                if s["total"] > 0:
                    fig = px.pie(names=["Helpful", "Not Helpful"], values=[s["helpful"], s["not_helpful"]], hole=0.45)
                    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#f1f5f9")
                    st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(str(e))

elif menu == "🎯 RAG Performance":
    if not logged_in:
        st.warning("Login required.")
    else:
        st.markdown("### 🎯 RAG Performance (RAGAS)")
        q = st.text_input("Evaluation Question", placeholder="What is overfitting in machine learning?")
        if st.button("Run Evaluation", use_container_width=True) and q:
            with st.spinner("Running RAGAS evaluation..."):
                try:
                    r = api_post("/evaluation", json={"question": q})
                    if r.status_code == 200:
                        e = r.json()
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("Faithfulness", f"{e['faithfulness']:.2f}")
                        c2.metric("Context Precision", f"{e['context_precision']:.2f}")
                        c3.metric("Answer Relevancy", f"{e['answer_relevancy']:.2f}")
                        c4.metric("Latency", f"{e['latency_seconds']:.2f}s")
                        st.info(f"Method: {e.get('method', 'ragas')}")
                        with st.expander("Generated Answer"):
                            st.write(e["answer"])
                    else:
                        st.error(r.text)
                except Exception as e:
                    st.error(str(e))

        st.markdown("#### Recent Evaluations")
        try:
            hist = api_get("/evaluation/history")
            if hist.status_code == 200:
                runs = hist.json().get("runs", [])
                if runs:
                    st.dataframe(pd.DataFrame(runs), use_container_width=True)
        except Exception:
            pass

elif menu == "⚙️ Admin":
    if not logged_in:
        st.warning("Login required.")
    else:
        st.markdown("### ⚙️ Knowledge Base Admin")
        try:
            r = api_get("/documents")
            if r.status_code == 200:
                docs = r.json().get("documents", [])
                if docs:
                    st.dataframe(pd.DataFrame(docs), use_container_width=True)
                    to_del = st.selectbox("Delete document", [d["source"] for d in docs])
                    if st.button("Delete", type="primary"):
                        resp = requests.request("DELETE", f"{BACKEND_URL}/documents", headers=api_headers(), json={"source": to_del})
                        if resp.status_code == 200:
                            st.success("Deleted")
                            st.rerun()
                else:
                    st.info("No documents ingested yet.")
        except Exception as e:
            st.error(str(e))
