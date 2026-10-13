import html

import streamlit as st

from rag_eval import run_eval
from rag_engine import DATA_DIR, answer_question, build_index, get_vectorstore

st.set_page_config(page_title="RAG with Sources + Evaluation", page_icon="📚", layout="centered")

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@500;700;800&family=Inter:wght@400;500;600&display=swap');
html, body, .stMarkdown, .stMarkdown p, label, input, textarea, button { font-family: 'Inter', sans-serif; }
[data-testid="stIconMaterial"], .material-symbols-rounded, .material-icons { font-family: 'Material Symbols Rounded' !important; }
h1, h2, h3 { font-family: 'Plus Jakarta Sans', sans-serif !important; letter-spacing: -0.02em; }

[data-testid="stHeader"] { background: transparent; }
[data-testid="stAppViewContainer"] {
    background-color: #F4F0FA;
    background-image: linear-gradient(rgba(123,94,167,.07) 1px, transparent 1px),
                      linear-gradient(90deg, rgba(123,94,167,.07) 1px, transparent 1px);
    background-size: 56px 56px;
}
.block-container { padding-top: 2rem; max-width: 820px; }

/* hero */
.hero { background: linear-gradient(135deg, #E7DDF6, #F6E6F0); color: #2D2540; border: 1px solid #DDD3EC; border-radius: 26px; padding: 30px 30px 26px; margin-bottom: 22px;
        box-shadow: 0 24px 50px -30px rgba(90,70,130,.35); }
.hero h1 { color: #2D2540; font-size: 2.1rem; margin: 0 0 6px; font-weight: 800; }
.hero p { color: #615A74; margin: 0 0 16px; font-size: 1.02rem; }
.chip { display: inline-block; font-family: ui-monospace, Menlo, Consolas, monospace; font-size: .78rem;
        padding: 4px 12px; margin: 0 8px 6px 0; border-radius: 999px; border: 1px solid rgba(123,94,167,.4); color: #7B5EA7; }

/* sidebar */
[data-testid="stSidebar"] { background: #FBF9FE; border-right: 1px solid #DDD3EC; }
[data-testid="stSidebar"] h2 { font-size: 1.25rem; }

/* buttons */
.stButton > button { background: #7B5EA7; border: 0; border-radius: 14px; padding: .6rem 1.2rem; font-weight: 600; }
.stButton > button p { color: #fff !important; }
.stButton > button:hover { background: #644A8C; }

/* tabs */
[data-baseweb="tab-list"] { gap: 6px; }
[data-baseweb="tab"] { border-radius: 12px 12px 0 0; padding: 10px 18px; font-weight: 600; }

/* chat */
[data-testid="stChatMessage"] { background: #FBF9FE; border: 1px solid #DDD3EC; border-radius: 20px; padding: 14px 18px; margin-bottom: 12px; }
[data-testid="stChatInput"] { border-radius: 16px; }
[data-testid="stExpander"] { border: 1px solid #DDD3EC; border-radius: 14px; background: #FFFFFF; }

/* source cards */
.src { border: 1px solid #DDD3EC; border-left: 4px solid #7B5EA7; border-radius: 12px; padding: 10px 14px; margin: 8px 0; background: #fff; }
.src b { color: #7B5EA7; }
.src span { display: block; color: #615A74; font-size: .9rem; margin-top: 4px; }

/* metrics */
[data-testid="stMetric"] { background: #FBF9FE; border: 1px solid #DDD3EC; border-radius: 18px; padding: 16px 18px; }
[data-testid="stMetricValue"] { color: #7B5EA7; font-family: 'Plus Jakarta Sans', sans-serif; font-weight: 800; }

/* evaluation rows */
.row { border: 1px solid #DDD3EC; border-radius: 14px; padding: 12px 16px; margin: 8px 0; background: #FBF9FE; }
.row.ok { border-left: 5px solid #2E7D68; }
.row.bad { border-left: 5px solid #C0556F; }
.row small { display: block; color: #615A74; margin-top: 4px; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

st.markdown(
    """
<div class="hero">
  <h1>📚 RAG with Sources + Evaluation</h1>
  <p>Ask questions about your documents. Every answer shows where it came from.</p>
  <span class="chip">LangChain</span><span class="chip">Chroma</span><span class="chip">Groq</span><span class="chip">Streamlit</span>
</div>
""",
    unsafe_allow_html=True,
)

@st.cache_resource(show_spinner="Loading the AI model (only the first time)...")
def warm_up():
    get_vectorstore()
    return True


try:
    warm_up()
except ValueError as e:
    st.warning(str(e))

with st.sidebar:
    st.header("Knowledge base")
    uploads = st.file_uploader("Add .txt, .md or .pdf files", type=["txt", "md", "pdf"], accept_multiple_files=True)
    if st.button("Save files and rebuild index"):
        DATA_DIR.mkdir(exist_ok=True)
        for f in uploads or []:
            (DATA_DIR / f.name).write_bytes(f.getbuffer())
        with st.spinner("Building index..."):
            try:
                docs, chunks = build_index()
                st.success(f"Indexed {docs} document pages into {chunks} chunks.")
            except ValueError as e:
                st.error(str(e))

chat_tab, eval_tab = st.tabs(["💬 Chat", "📊 Evaluation"])


def show_sources(sources):
    with st.expander("Sources"):
        for s in sources:
            page = f", page {s['page']}" if s["page"] else ""
            st.markdown(
                f"<div class='src'><b>{html.escape(s['file'])}{page}</b><span>{html.escape(s['snippet'])}</span></div>",
                unsafe_allow_html=True,
            )


with chat_tab:
    if "history" not in st.session_state:
        st.session_state.history = []
    if not st.session_state.history:
        st.info("Try: How long does standard delivery take?")
    for msg in st.session_state.history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg.get("sources"):
                show_sources(msg["sources"])
            if msg.get("timing"):
                t = msg["timing"]
                st.caption(f"search {t['search']}s, answer {t['llm']}s")

    question = st.chat_input("Ask a question about your documents")
    if question:
        st.session_state.history.append({"role": "user", "content": question})
        with st.spinner("Thinking..."):
            out = answer_question(question)
        st.session_state.history.append({"role": "assistant", "content": out["answer"], "sources": out["sources"], "timing": out["timing"]})
        st.rerun()

with eval_tab:
    st.write("Runs the questions in `eval/test_set.json` and measures how many are answered correctly.")
    if st.button("Run evaluation"):
        bar = st.progress(0.0)
        summary, results = run_eval(progress=bar.progress)
        c1, c2, c3 = st.columns(3)
        c1.metric("Questions", summary["total_questions"])
        c2.metric("Answer accuracy", f"{summary['answer_accuracy'] * 100:.0f}%")
        c3.metric("Retrieval hit rate", f"{summary['retrieval_hit_rate'] * 100:.0f}%")
        for r in results:
            cls = "ok" if r["answer_ok"] else "bad"
            mark = "✅" if r["answer_ok"] else "❌"
            st.markdown(
                f"<div class='row {cls}'>{mark} <b>{html.escape(r['question'])}</b>"
                f"<small>{html.escape(r['answer'])}</small></div>",
                unsafe_allow_html=True,
            )