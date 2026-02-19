"""
RAG Application con CodeCarbon integrato correttamente.

Funzionalità:
1. Carica uno o più PDF
2. Estrae testo, crea chunk e genera embeddings (all-MiniLM-L6-v2)
3. Memorizza i chunk in un vector store Chroma persistente
4. Risponde alle domande riportando i chunk più pertinenti
5. Traccia le emissioni CO2 con CodeCarbon (OfflineEmissionsTracker)
6. Mostra una dashboard delle emissioni con grafici nella sidebar
7. Permette di azzerare le emissioni tracciate
"""

import os
import streamlit as st
import pandas as pd
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from codecarbon import OfflineEmissionsTracker

# ── Costanti ───────────────────────────────────────────────────────────────────
CHROMA_PERSIST_DIR = "./chroma_db"
EMISSIONS_DIR      = "."
EMISSIONS_FILE     = "emissions.csv"
EMISSIONS_PATH     = os.path.join(EMISSIONS_DIR, EMISSIONS_FILE)

# Colonne rilevanti per la dashboard (subset di quelle scritte da CodeCarbon)
DASHBOARD_COLS = [
    "timestamp", "project_name", "duration", "emissions",
    "energy_consumed", "cpu_power", "gpu_power", "ram_power",
    "cpu_count", "gpu_count", "country_name",
]

# ── Configurazione pagina ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG App",
    page_icon="📄",
    layout="wide",
)

# ── Inizializzazione sessione ──────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "total_chunks" not in st.session_state:
    st.session_state.total_chunks = 0

# ── Titolo ─────────────────────────────────────────────────────────────────────
st.title("📄 RAG Application")
st.markdown(
    "Carica uno o più documenti **PDF** dalla sidebar, poi fai domande "
    "nella chat. L'app restituirà i chunk più pertinenti dal tuo documento, "
    "tracciando le emissioni CO₂ con **CodeCarbon**."
)

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — prima parte (upload + configurazione)
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.header("📁 Carica i tuoi PDF")
    uploaded_files = st.file_uploader(
        "Seleziona uno o più file PDF",
        type=["pdf"],
        accept_multiple_files=True,
    )
    process_btn = st.button(
        "⚙️ Elabora Documenti", type="primary", use_container_width=True
    )

    st.divider()

    st.header("🔍 Configurazione Retrieval")
    top_k = st.slider(
        "Chunk da mostrare (k)",
        min_value=1, max_value=10, value=3, step=1,
        help="Quanti chunk mostrare in risposta alla domanda.",
    )
    chunk_size = st.slider(
        "Dimensione chunk (caratteri)",
        min_value=200, max_value=2000, value=1000, step=100,
        help="Grandezza massima di ogni chunk di testo.",
    )
    chunk_overlap = st.slider(
        "Overlap tra chunk",
        min_value=0, max_value=400, value=200, step=50,
        help="Caratteri sovrapposti tra chunk adiacenti.",
    )


# ══════════════════════════════════════════════════════════════════════════════
# FUNZIONI
# ══════════════════════════════════════════════════════════════════════════════

def extract_text_from_pdf(pdf_file) -> str:
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text


def split_text_into_chunks(text: str, size: int, overlap: int) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        length_function=len,
    )
    return splitter.split_text(text)


@st.cache_resource(show_spinner=False)
def get_embedding_model():
    """Caricato una sola volta e cachato tra i rerun."""
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def create_vector_store(chunks: list[str], embeddings) -> Chroma:
    return Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PERSIST_DIR,
        collection_name="pdf_collection",
    )


def make_tracker(project_name: str) -> OfflineEmissionsTracker:
    """
    Crea un OfflineEmissionsTracker configurato per Streamlit:
    - offline → nessuna chiamata a API esterne, più stabile
    - country_iso_code="ITA" → intensità carbonica italiana
    - allow_multiple_runs=True → evita eccezioni ai rerun di Streamlit
    - measure_power_secs=5 → campionamento frequente anche per task brevi
    - save_to_file=True → CSV aggiornato ad ogni stop()
    """
    return OfflineEmissionsTracker(
        project_name=project_name,
        country_iso_code="ITA",
        measure_power_secs=5,
        save_to_file=True,
        output_dir=EMISSIONS_DIR,
        output_file=EMISSIONS_FILE,
        log_level="error",
        allow_multiple_runs=True,
    )


def load_emissions_df() -> pd.DataFrame | None:
    """Legge il CSV di CodeCarbon; ritorna None se assente o vuoto."""
    if not os.path.isfile(EMISSIONS_PATH):
        return None
    try:
        df = pd.read_csv(EMISSIONS_PATH)
        if df.empty:
            return None
        cols = [c for c in DASHBOARD_COLS if c in df.columns]
        return df[cols]
    except Exception:
        return None


def fmt_emissions(kg: float) -> str:
    if kg >= 1:
        return f"{kg:.6f} kg CO₂eq"
    elif kg >= 0.001:
        return f"{kg * 1000:.4f} g CO₂eq"
    else:
        return f"{kg * 1_000_000:.2f} µg CO₂eq"


def fmt_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f} s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f} min"
    else:
        return f"{seconds / 3600:.2f} h"


# ══════════════════════════════════════════════════════════════════════════════
# ELABORAZIONE PDF
# ══════════════════════════════════════════════════════════════════════════════
if uploaded_files and process_btn:
    with st.spinner("📖 Lettura e chunking dei PDF..."):
        all_text = ""
        file_summary = []
        for pdf_file in uploaded_files:
            text = extract_text_from_pdf(pdf_file)
            all_text += text
            file_summary.append({
                "File": pdf_file.name,
                "Caratteri": f"{len(text):,}",
                "Pagine": len(PdfReader(pdf_file).pages),
            })
        chunks = split_text_into_chunks(all_text, chunk_size, chunk_overlap)

    with st.spinner("🔢 Creazione embeddings e vector store (tracking CO₂)..."):
        embedding_model = get_embedding_model()
        tracker = make_tracker("rag_embedding")
        try:
            tracker.start()
            st.session_state.vector_store = create_vector_store(
                chunks, embedding_model
            )
        finally:
            # stop() scrive la riga nel CSV anche in caso di errore
            tracker.stop()

        st.session_state.total_chunks = len(chunks)

    st.success(
        f"✅ Indicizzati **{len(chunks)} chunk** da **{len(uploaded_files)} file**."
    )
    st.dataframe(
        pd.DataFrame(file_summary), use_container_width=True, hide_index=True
    )


# ══════════════════════════════════════════════════════════════════════════════
# INTERFACCIA CHAT
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.vector_store is not None:
    st.divider()
    st.subheader("💬 Chatta con i tuoi documenti")

    if st.session_state.total_chunks:
        st.caption(
            f"Vector store attivo — {st.session_state.total_chunks} chunk indicizzati."
        )

    # Storico messaggi
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Input utente
    if user_question := st.chat_input("Fai una domanda sui tuoi PDF…"):
        st.session_state.messages.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)

        with st.chat_message("assistant"):
            with st.spinner("🔍 Ricerca chunk pertinenti (tracking CO₂)..."):
                docs = []
                tracker = make_tracker("rag_retrieval")
                try:
                    tracker.start()
                    retriever = st.session_state.vector_store.as_retriever(
                        search_type="similarity",
                        search_kwargs={"k": top_k},
                    )
                    docs = retriever.invoke(user_question)
                finally:
                    tracker.stop()

            if not docs:
                response = "⚠️ Nessun chunk rilevante trovato nei documenti caricati."
            else:
                parts = []
                for i, doc in enumerate(docs, start=1):
                    parts.append(
                        f"**📌 Chunk {i} (su {len(docs)}):**\n\n{doc.page_content}"
                    )
                response = "\n\n---\n\n".join(parts)

            st.markdown(response)
            st.session_state.messages.append(
                {"role": "assistant", "content": response}
            )

else:
    st.info(
        "👈 Carica uno o più PDF dalla sidebar e clicca **Elabora Documenti** per iniziare."
    )


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — seconda parte: Dashboard Emissioni + Reset
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.divider()
    st.header("🌱 Dashboard Emissioni CO₂")

    df_em = load_emissions_df()

    if df_em is not None:
        total_emissions_kg = df_em["emissions"].sum()
        total_energy_kwh   = df_em["energy_consumed"].sum()
        total_duration_s   = df_em["duration"].sum()
        n_runs             = len(df_em)

        # ── Metriche principali ────────────────────────────────────────────────
        col1, col2 = st.columns(2)
        col1.metric("🌍 Emissioni totali", fmt_emissions(total_emissions_kg))
        col2.metric("⚡ Energia",          f"{total_energy_kwh:.5f} kWh")

        col3, col4 = st.columns(2)
        col3.metric("⏱ Durata totale", fmt_duration(total_duration_s))
        col4.metric("🔁 Esecuzioni",   str(n_runs))

        # ── Potenze medie hardware ─────────────────────────────────────────────
        hw_cols = {
            "cpu_power": ("🖥 CPU media", "W"),
            "ram_power": ("💾 RAM media", "W"),
            "gpu_power": ("🎮 GPU media", "W"),
        }
        for col_name, (label, unit) in hw_cols.items():
            if col_name in df_em.columns and df_em[col_name].notna().any():
                avg_val = df_em[col_name].mean()
                if avg_val > 0:
                    st.metric(label, f"{avg_val:.2f} {unit}")

        # Paese rilevato da CodeCarbon
        if "country_name" in df_em.columns and df_em["country_name"].notna().any():
            country = df_em["country_name"].iloc[-1]
            st.caption(f"🗺 Paese rilevato: **{country}**")

        # ── Grafico emissioni per esecuzione ──────────────────────────────────
        if n_runs > 1:
            chart_df = df_em[["project_name", "emissions"]].copy()
            chart_df["emissions_mg"] = chart_df["emissions"] * 1_000_000
            chart_df.index = [
                f"{row['project_name']} #{i+1}"
                for i, row in chart_df.iterrows()
            ]
            st.markdown("**📊 Emissioni per run (µg CO₂eq)**")
            st.bar_chart(chart_df["emissions_mg"], use_container_width=True)

        # ── Tabella dettagliata + download ────────────────────────────────────
        with st.expander("📋 Dettaglio esecuzioni"):
            st.dataframe(df_em, use_container_width=True)
            st.download_button(
                label="⬇️ Scarica emissions.csv",
                data=df_em.to_csv(index=False).encode("utf-8"),
                file_name="emissions.csv",
                mime="text/csv",
                use_container_width=True,
            )

    else:
        st.caption(
            "Nessun dato disponibile. Le emissioni saranno tracciate "
            "durante l'elaborazione dei PDF e le query."
        )

    # ── Bottone reset ──────────────────────────────────────────────────────────
    st.divider()
    st.markdown("**⚠️ Azzera emissioni**")
    st.caption("Elimina definitivamente il file emissions.csv e azzera la dashboard.")
    if st.button(
        "🗑️ Cancella storico emissioni",
        use_container_width=True,
        type="secondary",
    ):
        if os.path.isfile(EMISSIONS_PATH):
            os.remove(EMISSIONS_PATH)
            st.success("✅ Storico emissioni azzerato.")
            st.rerun()
        else:
            st.info("Nessun file da cancellare.")