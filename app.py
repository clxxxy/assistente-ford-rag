import os
import time
import shutil
import threading
import streamlit as st
from index import index_from_upload, load_vector_store
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA


# =============================
# config inicial do streamlit, modelos e ensure state
# =============================

# --- config inicial ---
st.set_page_config(
    page_title="Assistente de Manual Ford",
    layout="centered",
)
st.markdown(f"<style>{open('style.css').read()}</style>", unsafe_allow_html=True)

# --- modelos: embedding e LLM ---
DEFAULT_EMBED = "paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_LLM = "qwen2:1.5b"

# --- ensure state ---
def ensure_state():
    st.session_state.setdefault("stage", "upload")
    st.session_state.setdefault("emb_model", DEFAULT_EMBED)
    st.session_state.setdefault("llm_model", DEFAULT_LLM)
    st.session_state.setdefault("persist_dir", None)
    st.session_state.setdefault("collection", None)
    st.session_state.setdefault("last_doc_name", None)
    st.session_state.setdefault("last_doc_path", None)
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("is_thinking", False)
    st.session_state.setdefault("last_query", None)
    st.session_state.setdefault("error_message", None)
ensure_state()


# =============================
# funções auxiliares
# =============================

# --- prompt ---
PROMPT_TEMPLATE = open("prompt.txt", 'r', encoding='utf-8').read()

# --- carrega LLM ---
def _get_llm(model_name: str):
    key = f"_llm_{model_name}"
    if key not in st.session_state:
        llm = OllamaLLM(model=model_name)
        _ = llm.invoke("ping")
        st.session_state[key] = llm
    return st.session_state[key]

# --- carrega vector store ---
def _get_vs(persist_dir: str, collection_name: str, emb_model: str):
    key = f"_vs_{persist_dir}_{collection_name}_{emb_model}"
    if key not in st.session_state:
        st.session_state[key] = load_vector_store(persist_dir, collection_name, emb_model)
    return st.session_state[key]

# --- cria a chain (recuperação) ---
def _get_chain():
    emb = st.session_state.get("emb_model", DEFAULT_EMBED)
    llm_model = st.session_state.get("llm_model", DEFAULT_LLM)
    pd = st.session_state.get("persist_dir")
    coll = st.session_state.get("collection")

    key = f"_chain_{pd}_{coll}_{emb}_{llm_model}"
    if key not in st.session_state:
        vs = _get_vs(pd, coll, emb)
        llm = _get_llm(llm_model)
        retriever = vs.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 8, "fetch_k": 40, "lambda_mult": 0.8},
        )
        st.session_state[key] = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": PromptTemplate.from_template(PROMPT_TEMPLATE)},
        )
    return st.session_state[key]

# --- limpa estado para novo manual ---
def _cleanup_current_manual():
    pd = st.session_state.get("persist_dir")
    pfile = st.session_state.get("last_doc_path")

    for k in list(st.session_state.keys()):
        if k.startswith(("_llm_", "_vs_", "_chain_")):
            del st.session_state[k]

    if pd and os.path.exists(pd):
        shutil.rmtree(pd, ignore_errors=True)
    if pfile and os.path.exists(pfile):
        try:
            os.remove(pfile)
        except Exception:
            pass

    st.session_state.update({
        "persist_dir": None,
        "collection": None,
        "last_doc_name": None,
        "last_doc_path": None,
        "messages": [],
        "stage": "upload",
        "error_message": None,
    })

# --- processo de indexação do PDF ---
def _start_indexing(file_bytes: bytes, file_name: str):
    result = {"ok": False, "data": None, "err": None}
    emb = st.session_state.get("emb_model", DEFAULT_EMBED)

    def worker():
        try:
            info = index_from_upload(file_bytes, file_name, emb_model_name=emb)
            result["ok"] = True
            result["data"] = info
        except Exception as e:
            result["ok"] = False
            result["err"] = str(e)

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    return t, result


# =============================
# código principal
# =============================

# =============================
# UI - cabeçalho
# =============================
st.markdown(
    """
    <div style='text-align: center;'>
        <img src='https://images.icon-icons.com/2402/PNG/512/ford_logo_icon_145825.png' width='150'/>
    </div>
    """,
    unsafe_allow_html=True
)
st.markdown(
    """
    <style>
    .headerlink, .stMarkdown h1 a, .stMarkdown h2 a, .stMarkdown h3 a, .stMarkdown h4 a, .stMarkdown h5 a, .stMarkdown h6 a {
        display: none !important;
        visibility: hidden !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.markdown("<h1 style='text-align:center; '>Assistente de Manual Ford</h1>", unsafe_allow_html=True)

# --- frases de carregamento ---
LOADING_PHRASES = [
    "Você sabia que a Ford foi a primeira fabricante de automóveis a chegar ao Brasil? em 1º de maio de 1919",
    "O Ford GT40 venceu Le Mans quatro vezes seguidas",
    "Quase lá, prometo ser mais rápido que o trânsito das 18h...",
    "Enquanto você espera, já pensou em quantas histórias você já viveu com seu Ford?",
    "Nosso amado Ford Mustang foi o primeiro Muscle Car da história! sabia dessa?",
    "Só um instante, estou decifrando o que o engenheiro quis dizer...",
    "Sabe aquele icônico carro de polícia em filmes dos anos 90? É nosso! Ford Crown Victoria!",
]


# =============================
# estágio 1: upload do manual
# =============================
if st.session_state.get("stage") == "upload":
    with st.container():
        st.markdown("<h3 style='text-align:center; '>Envie o manual do seu veículo</h3>", unsafe_allow_html=True)
        st.markdown("<p style='text-align:center; '><a href='https://www.ford.com.br/support/owner-manuals/' target='_blank' style='color:#0072c6; text-decoration:underline;'>Consulte o modelo do seu veículo neste link</a> e baixe o manual em PDF</p>", unsafe_allow_html=True)
        pdf_file = st.file_uploader("Arraste ou selecione o arquivo (PDF)", type=["pdf"], accept_multiple_files=False) 

        if pdf_file is not None:
            if st.session_state.get("persist_dir") or st.session_state.get("last_doc_path"):
                _cleanup_current_manual()
            
            st.session_state["stage"] = "loading"
            st.session_state["_upload_bytes"] = pdf_file.getvalue()
            st.session_state["_upload_name"] = pdf_file.name
            st.rerun()

# =============================
# estágio 2: carregando e indexando manual
# =============================
elif st.session_state.get("stage") == "loading":
    loading_ui = st.empty()
    st.progress(0)

    if "_loader_thread" not in st.session_state:
        t, result = _start_indexing(st.session_state["_upload_bytes"], st.session_state["_upload_name"])
        st.session_state["_loader_thread"] = t
        st.session_state["_loader_result"] = result

    i = 0
    while st.session_state["_loader_thread"].is_alive():
        dynamic_text = LOADING_PHRASES[i % len(LOADING_PHRASES)]
        loading_ui.markdown(
            f"""
            <div class="loading-container">
                <p style='font-size: 1.1rem;'>Estou estudando seu veículo...</p>
                <div class="spinner"></div>
                <p class='loading-text'>{dynamic_text}</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        i += 1
        time.sleep(5)

    loading_ui.empty()
    res = st.session_state["_loader_result"]
    st.session_state.pop("_loader_thread", None)
    st.session_state.pop("_loader_result", None)
    st.session_state.pop("_upload_bytes", None)
    st.session_state.pop("_upload_name", None)
    
    if res["ok"]:
        info = res["data"]
        st.session_state.update({
            "persist_dir": info["persist_directory"],
            "collection": info["collection_name"],
            "last_doc_name": os.path.basename(info["pdf_path"]) or st.session_state.get("_upload_name", "Arquivo"),
            "last_doc_path": info["pdf_path"],
            "stage": "success_transition",
        })
        st.rerun()
    
    # --- log de erro geral ---
    else:
        st.session_state.update({
            "stage": "error_display",
            "error_message": res["err"] or "Sem detalhes."
        })
        st.rerun()

elif st.session_state.get("stage") == "success_transition":
    time.sleep(0.1)
    st.session_state["stage"] = "chat"
    st.rerun()

# --- log de erro no arquivo ---
elif st.session_state.get("stage") == "error_display":
    st.markdown(
        """
        <div class="custom-message-container">
            <h3>Não foi possível indexar o PDF.</h3>
        </div>
        """,
        unsafe_allow_html=True
    )
    with st.expander("Ver detalhes técnicos"):
        st.code(st.session_state.get("error_message", "Sem detalhes."))
    
    if st.button("Tentar com outro arquivo", use_container_width=True):
        _cleanup_current_manual()
        st.rerun()


# =============================
# estágio 3: chat de perguntas
# =============================

# --- cabeçalho: manual carregado e opção de alterar manual ---
elif st.session_state.get("stage") == "chat":
    with st.container():
        name = st.session_state.get("last_doc_name") or "seu arquivo"
        st.markdown(
            f"""
            <div style='text-align: center; margin-bottom: 0.5rem; color: #555;'>
                Manual carregado: <strong>{name}</strong>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        _, btn_col, _ = st.columns([1, 2, 1])
        with btn_col:
            if st.button("Alterar manual", type="secondary", use_container_width=True):
                _cleanup_current_manual()
                st.rerun()

    # --- input de Pergunta ---
    with st.container():
        with st.form("ask_form", clear_on_submit=True):
            q = st.text_input("O que você quer saber sobre seu veículo hoje?", placeholder="O que você quer saber sobre seu veículo hoje?")
            submitted = st.form_submit_button("Enviar", use_container_width=True, type="primary")
            if submitted and q.strip():
                st.session_state["messages"].append({"role": "user", "text": q.strip()})
                st.session_state["is_thinking"] = True
                st.session_state["last_query"] = q.strip()
                st.rerun()

    # --- exibição visual das respostas ---
    st.markdown("<h3 style='text-align: left;'>Histórico</h3>", unsafe_allow_html=True)

    if st.session_state.get("is_thinking"):
        st.markdown(
            """
            <div class="small-spinner-container">
                <div class="small-spinner"></div>
                <span>Pensando...</span>
            </div>
            """,
            unsafe_allow_html=True
        )

    messages = st.session_state.get("messages", [])
    qa_pairs = []
    i = 0
    while i < len(messages) - 1:
        if messages[i]["role"] == "user" and messages[i+1]["role"] == "assistant":
            qa_pairs.append((messages[i], messages[i+1]))
            i += 2
        else:
            i += 1 

    for user_msg, assistant_msg in reversed(qa_pairs):
        with st.container():
            st.markdown('<div class="qa-card">', unsafe_allow_html=True)
            st.markdown(f'<div class="question-header">Sua Pergunta:</div>', unsafe_allow_html=True)
            st.markdown(user_msg["text"])
            st.markdown('<div class="answer-body">', unsafe_allow_html=True)
            st.markdown(assistant_msg["text"])
            if assistant_msg.get("sources"):

                # --- fontes das respostas ---
                with st.expander("De onde tirei isso"):
                    for i, d in enumerate(assistant_msg["sources"][:5], start=1):
                        meta = getattr(d, "metadata", {}) or {}
                        page = meta.get("page", "?")
                        st.markdown(f"**Trecho {i}** — página aprox. {page}")
                        snippet = (getattr(d, "page_content", "") or "")[:700]
                        st.code(snippet + ("…" if len(getattr(d, "page_content", "")) > 700 else ""))
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

    # --- processamento geral: pergunta -> resposta ---
    if st.session_state.get("is_thinking"):
        query = st.session_state.pop("last_query", None)
        if query:
            try:
                qa = _get_chain()
                out = qa.invoke({"query": query})
                answer = (out.get("result") or "").strip()
                srcs = out.get("source_documents", []) or []
                st.session_state["messages"].append({"role": "assistant", "text": answer, "sources": srcs})
            except Exception as e:
                st.session_state["messages"].append({"role": "assistant", "text": f"Ops, não consegui responder agora.\n\nDetalhes técnicos: {e}"})
        
        st.session_state["is_thinking"] = False
        st.rerun()