import streamlit as st
import os, time, re, gc, torch
import pymupdf4llm
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder
from openai import OpenAI

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="PH Family Planning Handbook AI",
    page_icon="🇵🇭",
    layout="centered"
)

# ── Load API keys from Streamlit secrets ──────────────────────
HF_TOKEN   = st.secrets["HF_TOKEN"]
GROQ_KEY   = st.secrets["GROQ_API_KEY"]

os.environ["HF_TOKEN"]    = HF_TOKEN
os.environ["GROQ_API_KEY"] = GROQ_KEY

# ── Model IDs ─────────────────────────────────────────────────
QWEN_MODEL_ID  = "Qwen/Qwen3-4B-Instruct-2507"
JUDGE_MODEL_ID = "llama-3.1-8b-instant"
RERANK_ID      = "BAAI/bge-reranker-base"
EMBED_ID       = "nomic-ai/nomic-embed-text-v1.5"
PDF_PATH       = "phfphandbook-compressed.pdf"

# ── Clients ───────────────────────────────────────────────────
hf_client   = OpenAI(base_url="https://router.huggingface.co/v1", api_key=HF_TOKEN)
groq_client = OpenAI(base_url="https://api.groq.com/openai/v1",   api_key=GROQ_KEY)

# ── Cache heavy resources so they only load once ──────────────
@st.cache_resource
def load_pipeline():
    md_pages = pymupdf4llm.to_markdown(PDF_PATH, page_chunks=True)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=4000, chunk_overlap=1000,
        separators=["\n## ", "\n### ", "\n\n", "\n", " "]
    )
    splits = []
    for page_data in md_pages:
        text     = page_data.get("text", "")
        page_num = page_data.get("metadata", {}).get("page", 0) + 1
        for chunk in splitter.split_text(text):
            splits.append(Document(page_content=chunk, metadata={"page": page_num}))

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_ID,
        model_kwargs={"device": "cpu", "trust_remote_code": True},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 16}
    )
    faiss_ret = FAISS.from_documents(splits, embeddings).as_retriever(search_kwargs={"k": 10})
    bm25_ret  = BM25Retriever.from_documents(splits)
    bm25_ret.k = 10
    hybrid    = EnsembleRetriever(retrievers=[bm25_ret, faiss_ret], weights=[0.6, 0.4])
    reranker  = CrossEncoder(RERANK_ID, device="cpu")
    return splits, hybrid, reranker

splits, hybrid_retriever, reranker = load_pipeline()

# ── Retrieval function ────────────────────────────────────────
def retrieve_and_rerank(query):
    docs   = hybrid_retriever.invoke(query)
    scores = reranker.predict([[query, d.page_content] for d in docs])
    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    top    = [d for d, s in ranked[:3]]
    expanded = []
    for doc in top:
        try:
            idx = next(i for i, d in enumerate(splits) if d.page_content == doc.page_content)
            for offset in [-1, 0, 1]:
                n = idx + offset
                if 0 <= n < len(splits) and splits[n] not in expanded:
                    expanded.append(splits[n])
        except StopIteration:
            if doc not in expanded:
                expanded.append(doc)
    expanded.sort(key=lambda x: x.metadata.get("page", 0))
    return "\n\n---\n\n".join([d.page_content for d in expanded])

# ── Generation function ───────────────────────────────────────
QA_SYSTEM = """You are a highly capable and intelligent clinical extraction AI.
STRICT RULES:
1. USE ONLY the information provided in the [Context] section.
2. INTENT & SYNONYM MATCHING: Handle imprecise user wording by matching underlying concepts.
3. PARTIAL ANSWERS ONLY — NO INFERENCE: Present only what is explicitly stated.
4. If the topic is missing from context, respond: This information was not found in the provided context."""

CONDENSE_SYSTEM = """Rewrite the follow-up question as a standalone search query based on chat history.
Output ONLY the rewritten query, nothing else."""

def generate_answer(context, question, history_str):
    prompt = f"[Chat History]\n{history_str}\n\n[Context]\n{context}\n\n[Question]\n{question}"
    response = hf_client.chat.completions.create(
        model=QWEN_MODEL_ID,
        messages=[
            {"role": "system", "content": QA_SYSTEM},
            {"role": "user",   "content": prompt}
        ],
        max_tokens=2048,
    )
    return response.choices[0].message.content.strip()

# ── Streamlit UI ──────────────────────────────────────────────
st.title("🇵🇭 Philippine Family Planning Handbook AI")
st.caption("Answers grounded on the Philippine Family Planning Handbook 2023 Edition")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask a family planning question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Reading the handbook..."):
            history = st.session_state.messages[-6:]
            hist_str = "\n".join([
                f"{'User' if m['role']=='user' else 'Bot'}: {m['content']}"
                for m in history[:-1]
            ]) or "No history yet."

            search_query = prompt
            if len(st.session_state.messages) > 1:
                condense_prompt = f"[Chat History]\n{hist_str}\n\n[New Question]\n{prompt}\n\nStandalone Query:"
                search_query = hf_client.chat.completions.create(
                    model=QWEN_MODEL_ID,
                    messages=[
                        {"role": "system", "content": CONDENSE_SYSTEM},
                        {"role": "user",   "content": condense_prompt}
                    ],
                    max_tokens=50,
                ).choices[0].message.content.strip()

            context  = retrieve_and_rerank(search_query)
            answer   = generate_answer(context, prompt, hist_str)

        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
```

---

## Step 6 — Add the PDF to Your Repository

Upload `phfphandbook-compressed.pdf` directly to the root of your GitHub repository alongside `app.py`. Streamlit Community Cloud will have access to it since it clones the full repository on deployment.

---

## Step 7 — Push Everything to GitHub

Your repository should look like this before deploying:
```
fp-handbook-chatbot/
├── app.py
├── requirements.txt
└── phfphandbook-compressed.pdf