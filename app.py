import streamlit as st
from src.rag_pipeline import RAGPipeline

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="CrediTrust AI Complaint Assistant",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# CUSTOM CSS
# =====================================================

st.markdown("""
<style>

.main {
    padding-top: 1rem;
}

.stApp {
    background-color: #f5f7fb;
}

.hero {
    background: linear-gradient(135deg,#1e3a8a,#2563eb);
    padding: 2rem;
    border-radius: 18px;
    color: white;
    margin-bottom: 20px;
}

.answer-card {
    background-color: white;
    padding: 20px;
    border-radius: 12px;
    border-left: 6px solid #2563eb;
    box-shadow: 0px 2px 10px rgba(0,0,0,0.08);
}

.metric-card {
    background-color: white;
    padding: 15px;
    border-radius: 12px;
}

.footer {
    text-align:center;
    color:gray;
    margin-top:30px;
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# LOAD RAG PIPELINE
# =====================================================

@st.cache_resource
def load_rag():
    return RAGPipeline()

rag = load_rag()

# =====================================================
# SESSION STATE
# =====================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

# =====================================================
# SIDEBAR
# =====================================================

with st.sidebar:

    st.title("🏦 CrediTrust AI")

    st.markdown("---")

    st.subheader("Product Filter")

    product_filter = st.selectbox(
        "Select Product",
        [
            "All Products",
            "Credit Card",
            "Savings Account",
            "Money Transfer",
            "Personal Loan"
        ]
    )

    st.markdown("---")

    st.subheader("Dataset Overview")

    st.metric("Complaints", "445K+")
    st.metric("Vector Chunks", "34K")
    st.metric("Embedding Model", "MiniLM-L6-v2")

    st.markdown("---")

    st.subheader("Example Questions")

    st.caption("Why are customers unhappy with credit cards?")
    st.caption("What fraud complaints are common?")
    st.caption("What billing issues do customers report?")
    st.caption("What problems exist with savings accounts?")
    st.caption("What complaints occur with money transfers?")

    st.markdown("---")

    st.subheader("Technology Stack")

    st.write("""
    - Streamlit
    - ChromaDB
    - LangChain
    - Sentence Transformers
    - Qwen 2.5
    """)

# =====================================================
# HERO SECTION
# =====================================================

st.markdown("""
<div class="hero">

<h1>🏦 CrediTrust Complaint Analysis Assistant</h1>

<p>
AI-powered Retrieval-Augmented Generation (RAG) system designed to help
Customer Support, Product Managers, and Compliance teams analyze customer complaints.
</p>

</div>
""", unsafe_allow_html=True)

# =====================================================
# CHAT HISTORY
# =====================================================

for message in st.session_state.messages:

    with st.chat_message("user"):
        st.write(message["question"])

    with st.chat_message("assistant"):
        st.write(message["answer"])

# =====================================================
# QUESTION INPUT
# =====================================================

question = st.chat_input(
    "Ask a question about customer complaints..."
)

# =====================================================
# RUN QUERY
# =====================================================

if question:

    with st.chat_message("user"):
        st.write(question)

    with st.spinner("Analyzing complaint data..."):

        try:

            result = rag.run(question)

            answer = result["answer"]

            sources = result["sources"]

        except Exception as e:

            answer = f"Error: {str(e)}"

            sources = []

    # =================================================
    # STORE CHAT
    # =================================================

    st.session_state.messages.append(
        {
            "question": question,
            "answer": answer
        }
    )

    # =================================================
    # DISPLAY ANSWER
    # =================================================

    with st.chat_message("assistant"):

        st.subheader("📌 Analysis")

        placeholder = st.empty()

        streamed_text = ""

        for word in answer.split():

            streamed_text += word + " "

            placeholder.markdown(
                f"""
                <div class="answer-card">
                {streamed_text}
                </div>
                """,
                unsafe_allow_html=True
            )

    # =================================================
    # RETRIEVAL METRICS
    # =================================================

    st.markdown("### 📊 Retrieval Summary")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Retrieved Chunks",
            len(sources)
        )

    with col2:

        unique_products = len(
            set(
                doc.metadata.get(
                    "product_category",
                    "Unknown"
                )
                for doc in sources
            )
        )

        st.metric(
            "Product Categories",
            unique_products
        )

    with col3:
        st.metric(
            "LLM",
            "Qwen 2.5"
        )

    # =================================================
    # SOURCES
    # =================================================

    st.markdown("### 📚 Supporting Evidence")

    st.info(
        "These complaint excerpts were retrieved from the vector database and used by the model to generate the answer."
    )

    for i, doc in enumerate(sources, start=1):

        with st.expander(
            f"Source {i} | {doc.metadata.get('product_category','Unknown')}"
        ):

            st.markdown(
                f"**Complaint ID:** {doc.metadata.get('complaint_id','N/A')}"
            )

            st.markdown("---")

            st.write(doc.page_content)

# =====================================================
# CLEAR BUTTON
# =====================================================

col1, col2, col3 = st.columns([1,1,4])

with col1:

    if st.button("🗑 Clear Chat"):

        st.session_state.messages = []

        st.rerun()

# =====================================================
# FOOTER
# =====================================================

st.markdown("---")

st.markdown(
    """
    <div class="footer">

    Intelligent Complaint Analysis for Financial Services<br>

    Built with Retrieval-Augmented Generation (RAG)

    </div>
    """,
    unsafe_allow_html=True
)