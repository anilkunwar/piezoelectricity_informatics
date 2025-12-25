import arxiv
import fitz  # PyMuPDF
import pandas as pd
import streamlit as st
import os
import re
import sqlite3
from datetime import datetime
import logging
import tempfile
import requests
import zipfile
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

# ==============================
# ENVIRONMENT & PATH SETUP
# ==============================
def is_streamlit_cloud():
    """Detect Streamlit Cloud runtime."""
    return (
        os.getenv("HOME") == "/home/appuser" or
        os.getenv("IS_STREAMLIT_CLOUD", "false").lower() == "true"
    )

IS_CLOUD = is_streamlit_cloud()

# Use a guaranteed-writable directory
BASE_DIR = Path(tempfile.gettempdir()) if IS_CLOUD else Path(__file__).parent.resolve()
METADATA_DB_FILE = BASE_DIR / "piezoelectricity_metadata.db"
UNIVERSE_DB_FILE = BASE_DIR / "piezoelectricity_universe.db"
LOG_FILE = BASE_DIR / "piezoelectricity_query.log"

# Ensure BASE_DIR exists
BASE_DIR.mkdir(exist_ok=True)

# Logging setup
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# ==============================
# STREAMLIT PAGE CONFIG
# ==============================
st.set_page_config(page_title="Piezoelectricity in PVDF Query Tool", layout="wide")
st.title("Piezoelectricity in PVDF Query Tool with SciBERT")
st.markdown("""
This tool queries arXiv for papers on **piezoelectricity in PVDF with dopants like SnO₂**, focusing on **alpha and beta phase fractions**, **electrospun nanofibers**, **efficiency**, **electricity generation**, **mechanical force**, and related factors for piezoelectric studies. It uses SciBERT with attention mechanism to prioritize relevant abstracts (>30% relevance) and stores metadata in `piezoelectricity_metadata.db` and full PDF text in `piezoelectricity_universe.db` for fallback searches.
""")

if IS_CLOUD:
    st.info("ℹ️ **Running on Streamlit Cloud**: To respect arXiv's terms, PDFs are **only downloaded when you click 'Download PDF'**. Metadata is saved securely.")

# ==============================
# SESSION STATE INITIALIZATION
# ==============================
if "log_buffer" not in st.session_state:
    st.session_state.log_buffer = []
if "downloaded_pdfs" not in st.session_state:
    st.session_state.downloaded_pdfs = {}  # {paper_id: pdf_bytes}

def update_log(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.log_buffer.append(f"[{timestamp}] {message}")
    if len(st.session_state.log_buffer) > 30:
        st.session_state.log_buffer.pop(0)
    logging.info(message)

# ==============================
# SCIBERT LOADING
# ==============================
@st.cache_resource
def load_scibert():
    update_log("Loading SciBERT model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
    model = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased")
    model.eval()
    return tokenizer, model

try:
    scibert_tokenizer, scibert_model = load_scibert()
except Exception as e:
    st.error(f"Failed to load SciBERT: {e}. Install: `pip install transformers torch`")
    st.stop()

# ==============================
# TEXT NORMALIZATION & PATTERNS
# ==============================
@st.cache_data
def normalize_text(text):
    # Replace Greek letters with Latin equivalents
    greek_to_latin = {
        'α': 'alpha', 'β': 'beta', 'γ': 'gamma', 'δ': 'delta', 'ε': 'epsilon',
        'Α': 'alpha', 'Β': 'beta', 'Γ': 'gamma', 'Δ': 'delta', 'Ε': 'epsilon'
    }
    for g, l in greek_to_latin.items():
        text = text.replace(g, l)
    # Replace subscripts
    subscripts = {
        '₀': '0', '₁': '1', '₂': '2', '₃': '3', '₄': '4',
        '₅': '5', '₆': '6', '₇': '7', '₈': '8', '₉': '9'
    }
    for s, d in subscripts.items():
        text = text.replace(s, d)
    # Replace superscripts
    superscripts = {
        '⁰': '0', '¹': '1', '²': '2', '³': '3', '⁴': '4',
        '⁵': '5', '⁶': '6', '⁷': '7', '⁸': '8', '⁹': '9'
    }
    for s, d in superscripts.items():
        text = text.replace(s, d)
    return text.lower()

# Full key terms and patterns as in original
KEY_TERMS = [
    "piezoelectricity", "piezoelectric effect", "piezoelectric performance", "piezoelectric properties",
    "electrospun nanofibers", "electrospun fibers", "piezoelectric nanofibers", "nanofibrous membranes",
    "PVDF", "polyvinylidene fluoride", "poly(vinylidene fluoride)", "PVdF", "P(VDF-TrFE)",
    "alpha phase", "beta phase", "gamma phase", "delta phase",
    "efficiency", "piezoelectric efficiency",
    "electricity generation", "electrical power generation", "power output", "voltage output",
    "mechanical force", "mechanical stress", "mechanical deformation", "mechanical energy",
    "SnO2", "tin oxide", "tin dioxide", "stannic oxide",
    "dopants", "doped", "doping",
    "doped PVDF", "doped polyvinylidene fluoride",
    "piezoelectrics", "piezoelectric polymer", "piezoelectric materials",
    "phase fraction", "phase content", "fraction of phase", "crystalline phase",
    "energy harvesting", "nanogenerators", "scavenging mechanical energy",
    "nanofiber mats", "nanofibrous mats",
    "doping effects", "dopant effects",
    "polarization", "ferroelectric polarization", "pyroelectric",
    "ferroelectricity", "ferroelectric properties",
    "current density",
    "power density",
    "crystallinity", "semicrystalline"
]

KEY_PATTERNS = [
    r'\bpiezoelectric(?:ity| effect| performance| properties| coefficient| constant| polymer| materials)?\b',
    r'\belectrospun (?:nano)?fibers?|nanofiber mats|nanofibrous membranes?\b',
    r'\bpvdf|polyvinylidene fluoride|poly\s*\(?\s*vinylidene fluoride\s*\)?|pvd?f\b',
    r'\b(alpha|beta|gamma|delta|epsilon)\s*(?:phase|polymorph|crystal|crystals?|crystalline phase)\b',
    r'\befficiency|piezoelectric efficiency\b',
    r'\belectricity generation|electrical power generation|power output|voltage output\b',
    r'\bmechanical (?:force|stress|deformation|energy)\b',
    r'\bsno2|tin oxide|tin dioxide|stannic oxide\b',
    r'\bdopants?|doped|doping effects?\b',
    r'\bdoped pvdf\b',
    r'\bpiezoelectrics\b',
    r'\b(?:beta|alpha|gamma|delta|epsilon|phase) fraction|phase content|fraction of phase\b',
    r'\benergy harvesting|nanogenerators?|scavenging mechanical energy\b',
    r'\bpolarization|ferroelectric polarization|pyroelectric\b',
    r'\bferroelectric(?:ity| properties)?\b',
    r'\bcurrent density\b',
    r'\bpower density\b',
    r'\bcrystallinity|semicrystalline\b',
    r'\bpyroelectric properties?|pyroelectric coefficient\b',
    r'\bdielectric properties?|dielectric constant|permittivity\b',
    r'\bd33|d31|g33\b',  # Piezoelectric coefficients
    r'\bpvdf-trfe|pvdf-hfp|pvdf-ctfe|p\(vdf-co-hfp\)|p\(vdf-co-trfe\)\b',  # Copolymers
    r'\bbatio3|barium titanate\b',
    r'\bzno|zinc oxide\b',
    r'\btio2|titanium dioxide\b',
    r'\bcnt|carbon nanotubes?\b',
    r'\bgraphene(?: oxide)?\b',
    r'\bcofe2o4|fe3o4|magnetic nanoparticles?\b',
    r'\bnanocomposites?|composites?\b',
    r'\bpoling|annealing|stretching\b'  # Processing methods
]

@st.cache_data
def compile_patterns():
    return [re.compile(pat, re.IGNORECASE) for pat in KEY_PATTERNS]

COMPILED_PATTERNS = compile_patterns()

# ==============================
# SCIBERT SCORING WITH ATTENTION BOOST
# ==============================
@st.cache_data
def score_abstract_with_scibert(abstract):
    try:
        inputs = scibert_tokenizer(
            abstract,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
            return_attention_mask=True
        )
        with torch.no_grad():
            outputs = scibert_model(**inputs, output_attentions=True)
        abstract_normalized = normalize_text(abstract)
        num_matched = sum(1 for pat in COMPILED_PATTERNS if pat.search(abstract_normalized))
        relevance_prob = np.sqrt(num_matched) / np.sqrt(len(KEY_PATTERNS))
        
        # Attention boost
        tokens = scibert_tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        keyword_indices = [
            i for i, token in enumerate(tokens)
            if any(kw in token.lower() for kw in ['pvdf', 'piezo', 'phase', 'beta', 'alpha'])
        ]
        if keyword_indices:
            attentions = outputs.attentions[-1][0, 0].numpy()  # last layer, first head
            attn_score = np.sum(attentions[keyword_indices, :]) / len(keyword_indices)
            if attn_score > 0.1:
                relevance_prob = min(relevance_prob + 0.2 * (len(keyword_indices) / len(tokens)), 1.0)
        update_log(f"SciBERT (attention-boosted) scored abstract: {relevance_prob:.3f} (patterns matched: {num_matched})")
        return relevance_prob
    except Exception as e:
        update_log(f"SciBERT scoring failed: {str(e)}")
        # Fallback
        abstract_normalized = normalize_text(abstract)
        num_matched = sum(1 for pat in COMPILED_PATTERNS if pat.search(abstract_normalized))
        relevance_prob = np.sqrt(num_matched) / np.sqrt(len(KEY_PATTERNS))
        update_log(f"Fallback scoring: {relevance_prob:.3f}")
        return relevance_prob

# ==============================
# DATABASE FUNCTIONS
# ==============================
def init_metadata_db():
    conn = sqlite3.connect(METADATA_DB_FILE)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS papers (
            id TEXT PRIMARY KEY,
            title TEXT,
            authors TEXT,
            year INTEGER,
            categories TEXT,
            abstract TEXT,
            pdf_url TEXT,
            matched_terms TEXT,
            relevance_prob REAL
        )
    """)
    conn.commit()
    conn.close()
    update_log(f"Initialized metadata database at {METADATA_DB_FILE}")

def save_papers_to_db(papers_df):
    init_metadata_db()
    conn = sqlite3.connect(METADATA_DB_FILE)
    papers_df.to_sql("papers", conn, if_exists="replace", index=False)
    conn.close()
    update_log(f"Saved {len(papers_df)} papers to {METADATA_DB_FILE}")

# ==============================
# ARXIV QUERY FUNCTION
# ==============================
@st.cache_data
def query_arxiv_api(query, categories, max_results, start_year, end_year):
    try:
        client = arxiv.Client()
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance,
            sort_order=arxiv.SortOrder.Descending
        )
        papers = []
        query_terms = [t.strip() for t in query.split(' OR ')]
        query_words = {t.strip('"').lower() for t in query_terms}
        for result in client.results(search):
            if not (start_year <= result.published.year <= end_year):
                continue
            if not any(cat in result.categories for cat in categories):
                continue
            abstract_lower = result.summary.lower()
            title_lower = result.title.lower()
            matched_terms = [term for term in query_words if term in abstract_lower or term in title_lower]
            if not matched_terms:
                continue
            relevance_prob = score_abstract_with_scibert(result.summary)
            # Highlight matched terms in abstract
            abstract_highlighted = result.summary
            for term in matched_terms:
                abstract_highlighted = re.sub(
                    r'\b' + re.escape(term) + r'\b',
                    f'<b style="color: orange">{term}</b>',
                    abstract_highlighted,
                    flags=re.IGNORECASE
                )
            papers.append({
                "id": result.entry_id.split('/')[-1],
                "title": result.title,
                "authors": ", ".join([author.name for author in result.authors]),
                "year": result.published.year,
                "categories": ", ".join(result.categories),
                "abstract": result.summary,
                "abstract_highlighted": abstract_highlighted,
                "pdf_url": result.pdf_url,
                "matched_terms": ", ".join(matched_terms) if matched_terms else "None",
                "relevance_prob": round(relevance_prob * 100, 2)
            })
            if len(papers) >= max_results:
                break
        papers = sorted(papers, key=lambda x: x["relevance_prob"], reverse=True)
        update_log(f"Found {len(papers)} papers")
        return papers
    except Exception as e:
        update_log(f"arXiv query failed: {str(e)}")
        st.error(f"Error querying arXiv: {str(e)}. Try simplifying the query.")
        return []

# ==============================
# PDF DOWNLOAD HANDLING (ON-DEMAND)
# ==============================
def download_pdf_bytes(pdf_url):
    """Download PDF as bytes with proper headers."""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Piezoelectricity Research Tool; Academic Use) AppleWebKit/537.36'
    }
    response = requests.get(pdf_url, headers=headers, timeout=30)
    response.raise_for_status()
    return response.content

def handle_pdf_download(paper_id, pdf_url, paper_metadata):
    try:
        pdf_bytes = download_pdf_bytes(pdf_url)
        # Store in session state
        st.session_state.downloaded_pdfs[paper_id] = pdf_bytes
        # Optional: save full text to universe DB (only if not on Cloud to avoid disk issues)
        if not IS_CLOUD:
            try:
                doc = fitz.open(stream=pdf_bytes, filetype="pdf")
                text = ""
                for page in doc:
                    text += page.get_text()
                doc.close()
                # Save to universe DB
                conn = sqlite3.connect(UNIVERSE_DB_FILE)
                cur = conn.cursor()
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS papers (
                        id TEXT PRIMARY KEY,
                        title TEXT,
                        authors TEXT,
                        year INTEGER,
                        content TEXT
                    )
                """)
                cur.execute("""
                    INSERT OR REPLACE INTO papers (id, title, authors, year, content)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    paper_id,
                    paper_metadata.get("title", ""),
                    paper_metadata.get("authors", "Unknown"),
                    paper_metadata.get("year", 0),
                    text
                ))
                conn.commit()
                conn.close()
                update_log(f"Saved full text for {paper_id} to universe DB")
            except Exception as e:
                update_log(f"Failed to save full text for {paper_id}: {e}")
        update_log(f"✅ PDF downloaded for {paper_id}")
        return True
    except Exception as e:
        error_msg = f"❌ Failed to download {paper_id}: {str(e)}"
        update_log(error_msg)
        st.error(error_msg)
        return False

def create_zip_of_downloaded_pdfs():
    if not st.session_state.downloaded_pdfs:
        return None
    zip_path = BASE_DIR / "piezoelectricity_pdfs.zip"
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for paper_id, pdf_bytes in st.session_state.downloaded_pdfs.items():
            zipf.writestr(f"{paper_id}.pdf", pdf_bytes)
    return zip_path

# ==============================
# MAIN APP UI
# ==============================
st.header("arXiv Query for Piezoelectricity in Doped PVDF")
st.markdown("Search for abstracts on **piezoelectricity**, **electrospun nanofibers**, **PVDF**, **alpha/beta phases**, **SnO₂ dopants**, **efficiency**, **electricity generation**, **mechanical force** using SciBERT with attention mechanism.")

log_container = st.empty()
def display_logs():
    log_container.text_area("Processing Logs", "\n".join(st.session_state.log_buffer), height=200)

with st.sidebar:
    st.subheader("Search Parameters")
    query = st.text_area(
        "Query",
        value=' OR '.join([f'"{term}"' for term in KEY_TERMS]),
        height=150
    )
    default_categories = ["cond-mat.mtrl-sci", "physics.app-ph", "physics.chem-ph"]
    categories = st.multiselect(
        "Categories",
        default_categories,
        default=default_categories
    )
    max_results = st.slider("Max Papers", min_value=1, max_value=500, value=50)
    current_year = datetime.now().year
    col1, col2 = st.columns(2)
    with col1:
        start_year = st.number_input("Start Year", min_value=1990, max_value=current_year, value=2010)
    with col2:
        end_year = st.number_input("End Year", min_value=start_year, max_value=current_year, value=current_year)
    output_formats = st.multiselect(
        "Output Formats",
        ["CSV", "JSON"],
        default=["CSV"]
    )
    search_button = st.button("🔍 Search arXiv")

if search_button:
    if not query.strip():
        st.error("Enter a valid query.")
    elif not categories:
        st.error("Select at least one category.")
    elif start_year > end_year:
        st.error("Start year must be ≤ end year.")
    else:
        with st.spinner("Querying arXiv..."):
            papers = query_arxiv_api(query, categories, max_results, start_year, end_year)
        
        if not papers:
            st.warning("No papers found. Broaden query or categories.")
        else:
            st.success(f"Found **{len(papers)}** papers. Filtering for relevance > 30%...")
            relevant_papers = [p for p in papers if p["relevance_prob"] > 30.0]
            if not relevant_papers:
                st.warning("No papers with relevance > 30%. Broaden query or check logs.")
            else:
                st.success(f"✅ **{len(relevant_papers)}** papers with relevance > 30%.")
                df = pd.DataFrame(relevant_papers)
                
                # Save metadata to SQLite
                save_papers_to_db(df)
                st.info(f"Metadata saved to: `{METADATA_DB_FILE}`")
                if not IS_CLOUD:
                    st.info(f"Full-text DB (on download): `{UNIVERSE_DB_FILE}`")

                # Display papers
                for _, paper in df.iterrows():
                    with st.expander(f"📄 {paper['title']} ({paper['year']}) - {paper['relevance_prob']}%"):
                        st.markdown(f"**Authors**: {paper['authors']}")
                        st.markdown(f"**Categories**: {paper['categories']}")
                        st.markdown(f"**Matched Terms**: {paper['matched_terms']}")
                        st.markdown(paper['abstract_highlighted'], unsafe_allow_html=True)
                        
                        col1, col2 = st.columns([1, 3])
                        with col1:
                            if st.button("📥 Download PDF", key=f"btn_{paper['id']}"):
                                with st.spinner("Downloading PDF..."):
                                    success = handle_pdf_download(
                                        paper["id"],
                                        paper["pdf_url"],
                                        paper.to_dict()
                                    )
                                    if success:
                                        st.success("Downloaded!")
                        with col2:
                            abs_url = paper['pdf_url'].replace('/pdf/', '/abs/')
                            st.markdown(f"[🌐 View on arXiv]({abs_url}) | [📄 View PDF]({paper['pdf_url']})")
                
                # ZIP download for all manually downloaded PDFs
                if st.session_state.downloaded_pdfs:
                    zip_path = create_zip_of_downloaded_pdfs()
                    if zip_path and zip_path.exists():
                        with open(zip_path, "rb") as f:
                            st.download_button(
                                label="📦 Download All Downloaded PDFs as ZIP",
                                data=f,
                                file_name="piezoelectricity_pdfs.zip",
                                mime="application/zip"
                            )
                
                # Export metadata
                if "CSV" in output_formats:
                    csv = df.drop(columns=["abstract_highlighted"]).to_csv(index=False)
                    st.download_button(
                        label="📥 Download Paper Metadata (CSV)",
                        data=csv,
                        file_name="piezoelectricity_papers.csv",
                        mime="text/csv"
                    )
                if "JSON" in output_formats:
                    json_data = df.drop(columns=["abstract_highlighted"]).to_json(orient="records")
                    st.download_button(
                        label="📥 Download Paper Metadata (JSON)",
                        data=json_data,
                        file_name="piezoelectricity_papers.json",
                        mime="application/json"
                    )
                
                display_logs()
