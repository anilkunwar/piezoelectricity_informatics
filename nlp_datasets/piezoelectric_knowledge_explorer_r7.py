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
import io
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
import base64
from streamlit.components.v1 import html
import time
import json

# ==============================
# ENVIRONMENT & PATH SETUP
# ==============================
def is_streamlit_cloud():
    """Detect if running on Streamlit Cloud."""
    return (
        os.getenv("HOME") == "/home/appuser" or
        "streamlitapp.com" in os.getenv("HOSTNAME", "") or
        os.getenv("IS_STREAMLIT_CLOUD", "false").lower() == "true"
    )

IS_CLOUD = is_streamlit_cloud()

# Use a guaranteed-writable temporary directory
BASE_DIR = Path(tempfile.gettempdir())
METADATA_DB_FILE = BASE_DIR / "piezoelectricity_metadata.db"
UNIVERSE_DB_FILE = BASE_DIR / "piezoelectricity_universe.db"
LOG_FILE = BASE_DIR / "piezoelectricity_query.log"
SESSION_STATE_FILE = BASE_DIR / "piezoelectricity_session.json"

# Ensure the base directory exists
BASE_DIR.mkdir(exist_ok=True)

# Set up logging
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# ==============================
# CUSTOM CSS FOR AESTHETIC IMPROVEMENT
# ==============================
def inject_custom_css():
    """Inject custom CSS for better aesthetics."""
    st.markdown("""
    <style>
    /* Main container styling */
    .main {
        padding: 2rem;
    }
    
    /* Header styling */
    .stTitle {
        color: #1E3A8A;
        font-weight: 800 !important;
        padding-bottom: 1rem;
        border-bottom: 3px solid #3B82F6;
    }
    
    /* Card-like containers */
    .card {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid #dee2e6;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        transition: transform 0.3s ease;
    }
    
    .card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.1);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #3B82F6 0%, #1D4ED8 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #1D4ED8 0%, #1E3A8A 100%);
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4);
    }
    
    /* Download button special styling */
    .download-btn {
        background: linear-gradient(135deg, #10B981 0%, #059669 100%) !important;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #1E3A8A 0%, #3B82F6 100%);
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #1E3A8A 0%, #3B82F6 100%);
    }
    
    /* Metric cards */
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        border-left: 5px solid #3B82F6;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    /* Progress bar styling */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #3B82F6 0%, #8B5CF6 100%);
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #f1f5f9 0%, #e2e8f0 100%);
        border-radius: 8px;
        font-weight: 600;
    }
    
    /* Table styling */
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
    }
    
    /* Success/Error/Warning messages */
    .stAlert {
        border-radius: 10px;
        border-left: 5px solid;
    }
    
    /* Custom highlight for matched terms */
    .term-highlight {
        background: linear-gradient(120deg, #FEF3C7 0%, #FDE68A 100%);
        padding: 2px 6px;
        border-radius: 4px;
        font-weight: 600;
        color: #92400E;
    }
    
    /* Animation for new items */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .fade-in {
        animation: fadeIn 0.5s ease-out;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .main {
            padding: 1rem;
        }
    }
    </style>
    """, unsafe_allow_html=True)

# ==============================
# SESSION STATE PERSISTENCE UTILITIES — FIXED FOR BINARY DATA
# ==============================
def save_session_state():
    """Save critical session state to disk, encoding PDF bytes as base64."""
    try:
        # Encode PDF bytes to base64 strings for JSON serialization
        downloaded_pdfs_b64 = {}
        for paper_id, pdf_bytes in st.session_state.get('downloaded_pdfs', {}).items():
            if isinstance(pdf_bytes, bytes):
                downloaded_pdfs_b64[paper_id] = base64.b64encode(pdf_bytes).decode('utf-8')
            else:
                # In case of unexpected type, skip
                logging.warning(f"Skipping non-bytes PDF for {paper_id}")
        
        session_data = {
            'downloaded_pdfs_b64': downloaded_pdfs_b64,
            'log_buffer': st.session_state.get('log_buffer', []),
            'papers_df': st.session_state.get('papers_df', None),
            'search_performed': st.session_state.get('search_performed', False),
            'universe_db_updated': st.session_state.get('universe_db_updated', False),
            'last_update': datetime.now().isoformat()
        }
        
        # Convert DataFrames to dict for JSON serialization
        if session_data['papers_df'] is not None:
            session_data['papers_df'] = session_data['papers_df'].to_dict()
            
        with open(SESSION_STATE_FILE, 'w') as f:
            json.dump(session_data, f)
            
        logging.info(f"Session state saved: {len(downloaded_pdfs_b64)} PDFs, {len(session_data['log_buffer'])} logs")
    except Exception as e:
        logging.error(f"Error saving session state: {e}")

def load_session_state():
    """Load session state from disk after crash/restart, decoding base64 PDFs."""
    try:
        if SESSION_STATE_FILE.exists():
            with open(SESSION_STATE_FILE, 'r') as f:
                session_data = json.load(f)
            
            # Decode base64 PDFs back to bytes
            downloaded_pdfs = {}
            for paper_id, b64_str in session_data.get('downloaded_pdfs_b64', {}).items():
                try:
                    pdf_bytes = base64.b64decode(b64_str)
                    downloaded_pdfs[paper_id] = pdf_bytes
                except Exception as e:
                    logging.warning(f"Failed to decode PDF {paper_id}: {e}")
            
            st.session_state.downloaded_pdfs = downloaded_pdfs
            st.session_state.log_buffer = session_data.get('log_buffer', [])
            st.session_state.search_performed = session_data.get('search_performed', False)
            st.session_state.universe_db_updated = session_data.get('universe_db_updated', False)
            
            # Convert dict back to DataFrame
            if session_data.get('papers_df'):
                st.session_state.papers_df = pd.DataFrame(session_data['papers_df'])
            
            logging.info(f"Session state restored: {len(st.session_state.downloaded_pdfs)} PDFs, {len(st.session_state.log_buffer)} logs")
            return True
    except Exception as e:
        logging.error(f"Error loading session state: {e}")
    return False

def clear_session_state():
    """Clear session state and remove session file."""
    st.session_state.clear()
    if SESSION_STATE_FILE.exists():
        SESSION_STATE_FILE.unlink()
    logging.info("Session state cleared")
    st.success("✅ Session state cleared successfully!")
    time.sleep(1)
    st.rerun()

# ==============================
# EARLY DEFINITION OF update_log — CRITICAL FOR INITIALIZATION
# ==============================
def update_log(message):
    """Add a timestamped message to the log buffer and file. Defined early for safe use during setup."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_msg = f"[{timestamp}] {message}"
    
    # Ensure log buffer exists
    if 'log_buffer' not in st.session_state:
        st.session_state.log_buffer = []
    
    st.session_state.log_buffer.append(full_msg)
    if len(st.session_state.log_buffer) > 100:
        st.session_state.log_buffer.pop(0)
    
    logging.info(message)
    
    # Auto-save with protection
    try:
        if st.session_state.get('auto_save', True):
            save_session_state()
    except Exception as e:
        logging.warning(f"Auto-save during logging failed: {e}")

# ==============================
# STREAMLIT PAGE CONFIGURATION
# ==============================
st.set_page_config(
    page_title="Piezoelectricity in PVDF Knowledge Explorer",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🔬"
)

# Inject custom CSS
inject_custom_css()

# ==============================
# HEADER WITH ANIMATED ELEMENTS
# ==============================
st.markdown("""
<div class="fade-in">
    <h1 style="text-align: center; color: #1E3A8A; margin-bottom: 1rem;">
        🔬 Piezoelectricity in PVDF Knowledge Explorer
    </h1>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="card fade-in">
    <h3 style="color: #1E3A8A; margin-top: 0;">📚 About This Tool</h3>
    <p>This advanced tool queries <strong>arXiv</strong> for scientific literature on <strong>piezoelectricity in PVDF-based nanocomposites</strong>, with a focus on:</p>
    <ul style="columns: 2; column-gap: 2rem;">
        <li><strong>Phase fractions</strong> (α, β, γ)</li>
        <li><strong>Dopants</strong> (SnO₂, BaTiO₃, ZnO, etc.)</li>
        <li><strong>Electrospun nanofibers</strong></li>
        <li><strong>Energy harvesting efficiency</strong></li>
        <li><strong>Mechanical-to-electrical conversion</strong></li>
    </ul>
    <p>It uses <strong>SciBERT with attention-aware relevance scoring</strong> (>30% threshold) and stores:</p>
    <ul>
        <li><strong>Metadata</strong> in <code>piezoelectricity_metadata.db</code></li>
        <li><strong>Full extracted text</strong> in <code>piezoelectricity_universe.db</code></li>
    </ul>
    <div style="background: linear-gradient(135deg, #DCFCE7 0%, #BBF7D0 100%); padding: 1rem; border-radius: 8px; border-left: 4px solid #10B981;">
        <strong>🔒 Session Persistence:</strong> Your downloaded PDFs and search results are preserved across app restarts.
    </div>
</div>
""", unsafe_allow_html=True)

if IS_CLOUD:
    st.warning("""
    ☁️ **Streamlit Cloud Mode**: Files are stored temporarily. 
    Use download buttons before your session expires!
    """)

# ==============================
# SESSION STATE INITIALIZATION — NOW SAFE
# ==============================
if "initialized" not in st.session_state:
    st.session_state.log_buffer = []
    
    # Attempt to restore previous session
    if not load_session_state():
        # Fresh session
        st.session_state.downloaded_pdfs = {}
        st.session_state.papers_df = None
        st.session_state.search_performed = False
        st.session_state.universe_db_updated = False
        st.session_state.download_queued = {}
        st.session_state.auto_save = True
        update_log("Intialized new session (no prior state)")
    else:
        update_log("Restored session from disk")
    
    st.session_state.initialized = True
    update_log("Session initialized")

# ==============================
# SCIBERT MODEL LOADING
# ==============================
@st.cache_resource(show_spinner=False)
def load_scibert():
    """Load and cache the SciBERT tokenizer and model."""
    update_log("Loading SciBERT model and tokenizer from Hugging Face...")
    try:
        tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
        model = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased")
        model.eval()
        update_log("✅ SciBERT loaded successfully")
        return tokenizer, model
    except Exception as e:
        update_log(f"❌ Failed to load SciBERT: {e}")
        raise e

try:
    with st.spinner("🔄 Loading AI model..."):
        scibert_tokenizer, scibert_model = load_scibert()
except Exception as e:
    st.error(f"""
    ❌ Failed to load SciBERT: {e}
    
    Please install required packages:
    ```bash
    pip install transformers torch
    ```
    """)
    st.stop()

# ==============================
# TEXT NORMALIZATION AND PATTERN DEFINITIONS
# ==============================
@st.cache_data(show_spinner=False)
def normalize_text(text):
    """Normalize text by replacing Greek letters, subscripts, and superscripts."""
    greek_to_latin = {
        'α': 'alpha', 'β': 'beta', 'γ': 'gamma', 'δ': 'delta', 'ε': 'epsilon',
        'Α': 'alpha', 'Β': 'beta', 'Γ': 'gamma', 'Δ': 'delta', 'Ε': 'epsilon'
    }
    for greek, latin in greek_to_latin.items():
        text = text.replace(greek, latin)
    
    subscripts = {
        '₀': '0', '₁': '1', '₂': '2', '₃': '3', '₄': '4',
        '₅': '5', '₆': '6', '₇': '7', '₈': '8', '₉': '9'
    }
    for sub, digit in subscripts.items():
        text = text.replace(sub, digit)
    
    return text.lower()

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
    r'\bd33|d31|g33\b',
    r'\bpvdf-trfe|pvdf-hfp|pvdf-ctfe|p\(vdf-co-hfp\)|p\(vdf-co-trfe\)\b',
    r'\bbatio3|barium titanate\b',
    r'\bzno|zinc oxide\b',
    r'\btio2|titanium dioxide\b',
    r'\bcnt|carbon nanotubes?\b',
    r'\bgraphene(?: oxide)?\b',
    r'\bcofe2o4|fe3o4|magnetic nanoparticles?\b',
    r'\bnanocomposites?|composites?\b',
    r'\bpoling|annealing|stretching\b'
]

@st.cache_data(show_spinner=False)
def compile_patterns():
    """Compile regex patterns for efficiency."""
    return [re.compile(pattern, re.IGNORECASE) for pattern in KEY_PATTERNS]

COMPILED_PATTERNS = compile_patterns()

# ==============================
# SCIBERT-BASED ABSTRACT SCORING
# ==============================
@st.cache_data(show_spinner=False)
def score_abstract_with_scibert(abstract):
    """Score abstract relevance using SciBERT and regex pattern matching."""
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
        
        tokens = scibert_tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        keyword_indices = [
            i for i, token in enumerate(tokens)
            if any(kw in token.lower() for kw in ['pvdf', 'piezo', 'phase', 'beta', 'alpha', 'sn', 'oxide'])
        ]
        if keyword_indices:
            attentions = outputs.attentions[-1][0, 0].numpy()
            attn_score = np.sum(attentions[keyword_indices, :]) / len(keyword_indices)
            if attn_score > 0.1:
                boost = 0.2 * (len(keyword_indices) / len(tokens))
                relevance_prob = min(relevance_prob + boost, 1.0)
        
        update_log(f"SciBERT scored abstract: {relevance_prob:.3f} (patterns matched: {num_matched})")
        return relevance_prob
    except Exception as e:
        update_log(f"SciBERT scoring failed: {str(e)}")
        abstract_normalized = normalize_text(abstract)
        num_matched = sum(1 for pat in COMPILED_PATTERNS if pat.search(abstract_normalized))
        relevance_prob = np.sqrt(num_matched) / np.sqrt(len(KEY_PATTERNS))
        update_log(f"Fallback scoring: {relevance_prob:.3f}")
        return relevance_prob

# ==============================
# DATABASE INITIALIZATION
# ==============================
def init_metadata_db():
    """Initialize the metadata SQLite database."""
    try:
        conn = sqlite3.connect(METADATA_DB_FILE)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS papers (
                id TEXT PRIMARY KEY,
                title TEXT,
                authors TEXT,
                year INTEGER,
                categories TEXT,
                abstract TEXT,
                pdf_url TEXT,
                matched_terms TEXT,
                relevance_prob REAL,
                downloaded INTEGER DEFAULT 0
            )
        """)
        conn.commit()
        conn.close()
        update_log(f"Initialized metadata database at {METADATA_DB_FILE}")
    except Exception as e:
        update_log(f"Failed to initialize metadata DB: {e}")

def init_universe_db():
    """Initialize the full-text universe SQLite database."""
    try:
        conn = sqlite3.connect(UNIVERSE_DB_FILE)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS papers (
                id TEXT PRIMARY KEY,
                title TEXT,
                authors TEXT,
                year INTEGER,
                content TEXT,
                download_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        conn.close()
        update_log(f"Initialized universe database at {UNIVERSE_DB_FILE}")
    except Exception as e:
        update_log(f"Failed to initialize universe DB: {e}")

# ==============================
# ARXIV QUERY FUNCTION
# ==============================
@st.cache_data(show_spinner=False)
def query_arxiv_api(query, categories, max_results, start_year, end_year):
    """Query arXiv and return relevant papers."""
    try:
        client = arxiv.Client()
        search = arxiv.Search(
            query=query,
            max_results=max_results * 2,
            sort_by=arxiv.SortCriterion.Relevance,
            sort_order=arxiv.SortOrder.Descending
        )
        papers = []
        query_terms = [t.strip() for t in query.split(' OR ')]
        query_words = {t.strip('"').lower() for t in query_terms}
        seen_ids = set()
        
        for result in client.results(search):
            if not (start_year <= result.published.year <= end_year):
                continue
            if not any(cat in result.categories for cat in categories):
                continue
            
            paper_id = result.get_short_id()
            if paper_id in seen_ids:
                continue
            seen_ids.add(paper_id)
            
            abstract_lower = result.summary.lower()
            title_lower = result.title.lower()
            matched_terms = [term for term in query_words if term in abstract_lower or term in title_lower]
            if not matched_terms:
                continue
            
            relevance_prob = score_abstract_with_scibert(result.summary)
            
            abstract_highlighted = result.summary
            for term in matched_terms:
                abstract_highlighted = re.sub(
                    r'\b' + re.escape(term) + r'\b',
                    f'<span class="term-highlight">{term}</span>',
                    abstract_highlighted,
                    flags=re.IGNORECASE
                )
            
            papers.append({
                "id": paper_id,
                "title": result.title,
                "authors": ", ".join([author.name for author in result.authors]),
                "year": result.published.year,
                "categories": ", ".join(result.categories),
                "abstract": result.summary,
                "abstract_highlighted": abstract_highlighted,
                "pdf_url": result.pdf_url,
                "matched_terms": ", ".join(matched_terms) if matched_terms else "None",
                "relevance_prob": round(relevance_prob * 100, 2),
                "downloaded": paper_id in st.session_state.downloaded_pdfs
            })
            
            if len(papers) >= max_results:
                break
        
        papers = sorted(papers, key=lambda x: x["relevance_prob"], reverse=True)
        update_log(f"Query returned {len(papers)} unique papers")
        return papers
    except Exception as e:
        update_log(f"arXiv query failed: {str(e)}")
        st.error(f"Error querying arXiv: {str(e)}")
        return []

# ==============================
# PDF DOWNLOAD AND PROCESSING
# ==============================
def download_pdf_bytes(pdf_url):
    """Download a PDF as bytes with proper headers."""
    headers = {
        'User-Agent': 'Mozilla/5.0 (compatible; Piezoelectricity Research Tool/1.0)'
    }
    response = requests.get(pdf_url, headers=headers, timeout=30)
    response.raise_for_status()
    return response.content

def handle_pdf_download(paper_id, pdf_url, paper_metadata):
    """Download a PDF, extract text, and update databases."""
    try:
        pdf_bytes = download_pdf_bytes(pdf_url)
        st.session_state.downloaded_pdfs[paper_id] = pdf_bytes
        update_log(f"Downloaded PDF for {paper_id} ({len(pdf_bytes)} bytes)")
        
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        full_text = ""
        for page in doc:
            full_text += page.get_text()
        doc.close()
        update_log(f"Extracted {len(full_text)} characters from {paper_id}")
        
        init_universe_db()
        conn = sqlite3.connect(UNIVERSE_DB_FILE)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO papers (id, title, authors, year, content)
            VALUES (?, ?, ?, ?, ?)
        """, (
            paper_id,
            paper_metadata.get("title", ""),
            paper_metadata.get("authors", "Unknown"),
            paper_metadata.get("year", 0),
            full_text
        ))
        conn.commit()
        conn.close()
        
        init_metadata_db()
        conn = sqlite3.connect(METADATA_DB_FILE)
        cursor = conn.cursor()
        cursor.execute("UPDATE papers SET downloaded = 1 WHERE id = ?", (paper_id,))
        conn.commit()
        conn.close()
        
        st.session_state.universe_db_updated = True
        save_session_state()
        update_log(f"Updated databases with {paper_id}")
        return True
    except Exception as e:
        error_msg = f"PDF download/extraction failed for {paper_id}: {str(e)}"
        update_log(error_msg)
        return False

# ==============================
# FILE CREATION UTILITIES
# ==============================
def create_zip_of_downloaded_pdfs():
    """Create a ZIP file of all downloaded PDFs in memory."""
    if not st.session_state.downloaded_pdfs:
        return None
    
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for paper_id, pdf_bytes in st.session_state.downloaded_pdfs.items():
            zipf.writestr(f"{paper_id}.pdf", pdf_bytes)
    
    zip_buffer.seek(0)
    update_log(f"Created ZIP with {len(st.session_state.downloaded_pdfs)} PDFs")
    return zip_buffer.getvalue()

def get_db_as_bytes(db_path):
    """Read a SQLite database file as bytes."""
    if not db_path.exists():
        return None
    with open(db_path, "rb") as f:
        return f.read()

# ==============================
# DOWNLOAD MANAGER COMPONENT
# ==============================
def download_manager():
    """Display download options for all available files."""
    st.markdown("### 📥 Download Manager")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.session_state.downloaded_pdfs:
            zip_data = create_zip_of_downloaded_pdfs()
            if zip_data:
                st.download_button(
                    label="📦 All PDFs (ZIP)",
                    data=zip_data,
                    file_name="piezoelectricity_pdfs.zip",
                    mime="application/zip",
                    key="zip_download",
                    use_container_width=True
                )
                st.caption(f"{len(st.session_state.downloaded_pdfs)} PDFs")
        else:
            st.button("📦 All PDFs (ZIP)", disabled=True, help="No PDFs downloaded", use_container_width=True)
    
    with col2:
        metadata_bytes = get_db_as_bytes(METADATA_DB_FILE)
        if metadata_bytes:
            st.download_button(
                label="🗃️ Metadata DB",
                data=metadata_bytes,
                file_name="piezoelectricity_metadata.db",
                mime="application/x-sqlite3",
                key="metadata_download",
                use_container_width=True
            )
            size_mb = len(metadata_bytes) / (1024 * 1024)
            st.caption(f"{size_mb:.1f} MB")
        else:
            st.button("🗃️ Metadata DB", disabled=True, help="Search first", use_container_width=True)
    
    with col3:
        universe_bytes = get_db_as_bytes(UNIVERSE_DB_FILE)
        if universe_bytes and st.session_state.universe_db_updated:
            st.download_button(
                label="🔍 Full-Text DB",
                data=universe_bytes,
                file_name="piezoelectricity_universe.db",
                mime="application/x-sqlite3",
                key="universe_download",
                use_container_width=True
            )
            size_mb = len(universe_bytes) / (1024 * 1024)
            st.caption(f"{size_mb:.1f} MB")
        else:
            st.button("🔍 Full-Text DB", disabled=True, help="Download PDFs first", use_container_width=True)
    
    with col4:
        session_data = {
            'downloaded_pdfs_count': len(st.session_state.downloaded_pdfs),
            'search_performed': st.session_state.search_performed,
            'last_update': datetime.now().isoformat()
        }
        session_json = json.dumps(session_data, indent=2)
        st.download_button(
            label="💾 Session Info",
            data=session_json,
            file_name="session_info.json",
            mime="application/json",
            key="session_download",
            use_container_width=True
        )
        st.caption("Session metadata")

# ==============================
# DATABASE INSPECTION
# ==============================
def inspect_metadata_db():
    """Display the metadata database contents."""
    if not METADATA_DB_FILE.exists():
        st.warning("Metadata database not found.")
        return
    
    conn = sqlite3.connect(METADATA_DB_FILE)
    df = pd.read_sql("SELECT * FROM papers ORDER BY relevance_prob DESC", conn)
    conn.close()
    
    st.subheader("🗃️ Metadata Database")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Papers", len(df))
    with col2:
        downloaded = df['downloaded'].sum() if 'downloaded' in df.columns else 0
        st.metric("PDFs Downloaded", downloaded)
    with col3:
        avg_relevance = df['relevance_prob'].mean() if len(df) > 0 else 0
        st.metric("Avg Relevance", f"{avg_relevance:.1f}%")
    with col4:
        latest_year = df['year'].max() if len(df) > 0 else 0
        st.metric("Latest Year", latest_year)
    
    st.dataframe(df, use_container_width=True, hide_index=True)

def inspect_universe_db():
    """Display the universe database contents."""
    if not st.session_state.universe_db_updated or not UNIVERSE_DB_FILE.exists():
        st.warning("Full-text database not available. Download at least one PDF first.")
        return
    
    st.subheader("🔍 Full-Text Database")
    
    search_term = st.text_input("🔎 Search in full text:", key="universe_search")
    if search_term:
        conn = sqlite3.connect(UNIVERSE_DB_FILE)
        query = """
        SELECT id, title, authors, year, 
               substr(content, max(1, instr(lower(content), lower(?)) - 100), 200) as snippet
        FROM papers 
        WHERE lower(content) LIKE ?
        """
        df_results = pd.read_sql_query(query, conn, params=(search_term, f"%{search_term.lower()}%"))
        conn.close()
        
        if not df_results.empty:
            st.success(f"Found {len(df_results)} results for '{search_term}':")
            for _, row in df_results.iterrows():
                with st.expander(f"📄 {row['title']} ({row['year']})"):
                    st.markdown(f"**Authors**: {row['authors']}")
                    st.markdown(f"**Snippet**: ...{row['snippet']}...")
        else:
            st.info("No matches found.")
    
    conn = sqlite3.connect(UNIVERSE_DB_FILE)
    df = pd.read_sql("SELECT id, title, authors, year, download_time FROM papers", conn)
    conn.close()
    st.dataframe(df, use_container_width=True, hide_index=True)

# ==============================
# SIDEBAR CONFIGURATION
# ==============================
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h2 style="color: white; margin: 0;">⚙️ Configuration</h2>
        <p style="color: #E0F2FE; opacity: 0.9;">Customize your search</p>
    </div>
    """, unsafe_allow_html=True)
    
    query_mode = st.radio("**Query Mode**", ["Auto (Recommended)", "Custom"], horizontal=True)
    if query_mode == "Auto":
        query = ' OR '.join([f'"{term}"' for term in KEY_TERMS[:10]])
        st.text_area("**Generated Query**", value=query, height=100, disabled=True)
    else:
        query = st.text_area("**Custom Query**", value=' OR '.join([f'"{term}"' for term in KEY_TERMS[:5]]), height=100)
    
    default_categories = ["cond-mat.mtrl-sci", "physics.app-ph", "physics.chem-ph"]
    categories = st.multiselect(
        "**arXiv Categories**",
        options=default_categories + ["cond-mat.soft", "cond-mat.other", "physics.ins-det"],
        default=default_categories,
        help="Select at least one category"
    )
    
    max_results = st.slider("**Max Results**", min_value=1, max_value=200, value=30)
    current_year = datetime.now().year
    col1, col2 = st.columns(2)
    with col1:
        start_year = st.number_input("**Start Year**", min_value=1990, max_value=current_year, value=2010)
    with col2:
        end_year = st.number_input("**End Year**", min_value=start_year, max_value=current_year, value=current_year)
    
    col1, col2 = st.columns(2)
    with col1:
        search_button = st.button("🚀 Execute Search", type="primary", use_container_width=True)
    with col2:
        clear_button = st.button("🔄 Clear Session", use_container_width=True)
    
    if clear_button:
        clear_session_state()
    
    st.markdown("---")
    st.markdown("### 💾 Session Info")
    st.metric("PDFs Downloaded", len(st.session_state.downloaded_pdfs))
    st.metric("Log Entries", len(st.session_state.log_buffer))
    
    st.session_state.auto_save = st.toggle("Auto-save session", value=True)

# ==============================
# MAIN APPLICATION LOGIC
# ==============================
if search_button:
    if not categories:
        st.error("⚠️ Please select at least one arXiv category.")
    elif start_year > end_year:
        st.error("⚠️ Start year cannot be greater than end year.")
    else:
        st.session_state.search_performed = True
        with st.spinner("📡 Querying arXiv API..."):
            papers = query_arxiv_api(query, categories, max_results, start_year, end_year)
        
        if not papers:
            st.warning("📭 No papers found. Try broadening your query or categories.")
        else:
            relevant_papers = [p for p in papers if p["relevance_prob"] > 30.0]
            st.success(f"""
            ✅ Found **{len(relevant_papers)}** relevant papers (relevance > 30%)
            
            *Total papers retrieved: {len(papers)}*
            """)
            
            if not relevant_papers:
                st.warning("📭 No papers above 30% relevance threshold.")
            else:
                df = pd.DataFrame(relevant_papers)
                st.session_state.papers_df = df
                
                init_metadata_db()
                conn = sqlite3.connect(METADATA_DB_FILE)
                df_to_save = df.drop(columns=["abstract_highlighted", "downloaded"], errors='ignore')
                df_to_save.to_sql("papers", conn, if_exists="replace", index=False)
                conn.close()
                update_log(f"Saved {len(df)} papers to metadata DB")
                
                st.subheader("📚 Relevant Papers")
                
                for idx, paper in df.iterrows():
                    with st.expander(f"""
                    📄 **{paper['title']}** 
                    *({paper['year']})* — **{paper['relevance_prob']}%** relevance
                    """):
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.markdown(f"**Authors**: {paper['authors']}")
                            st.markdown(f"**Categories**: `{paper['categories']}`")
                            st.markdown(f"**Matched Terms**: `{paper['matched_terms']}`")
                        with col2:
                            st.metric("Relevance", f"{paper['relevance_prob']}%")
                        
                        st.markdown("### Abstract")
                        st.markdown(paper["abstract_highlighted"], unsafe_allow_html=True)
                        
                        col_dl, col_view, col_save = st.columns(3)
                        with col_dl:
                            if st.button("📥 Download & Index", key=f"dl_{paper['id']}", use_container_width=True):
                                with st.spinner("Downloading..."):
                                    success = handle_pdf_download(paper["id"], paper["pdf_url"], paper)
                                    if success:
                                        st.success("✅ Downloaded!")
                                        st.rerun()
                        
                        with col_view:
                            abs_url = paper['pdf_url'].replace('/pdf/', '/abs/')
                            st.link_button("🌐 View on arXiv", abs_url, use_container_width=True)
                        
                        with col_save:
                            if paper["id"] in st.session_state.downloaded_pdfs:
                                st.download_button(
                                    "💾 Save PDF",
                                    data=st.session_state.downloaded_pdfs[paper["id"]],
                                    file_name=f"{paper['id']}.pdf",
                                    mime="application/pdf",
                                    key=f"save_{paper['id']}",
                                    use_container_width=True
                                )

# ==============================
# DOWNLOAD AND INSPECTION SECTION
# ==============================
st.markdown("---")

if st.session_state.search_performed or st.session_state.downloaded_pdfs:
    download_manager()
    
    st.markdown("### 🔍 Database Inspection")
    tab1, tab2 = st.tabs(["📊 Metadata Database", "🔍 Full-Text Database"])
    
    with tab1:
        inspect_metadata_db()
    
    with tab2:
        inspect_universe_db()

# ==============================
# LOG DISPLAY
# ==============================
st.markdown("---")
st.subheader("📝 Activity Log")

col1, col2 = st.columns([3, 1])
with col1:
    log_display = st.select_slider(
        "Show last N logs:",
        options=[10, 25, 50, 100],
        value=25
    )
with col2:
    if st.button("Clear Logs"):
        st.session_state.log_buffer = []
        st.rerun()

log_container = st.container()
with log_container:
    logs_to_show = st.session_state.log_buffer[-log_display:] if st.session_state.log_buffer else ["No logs yet"]
    log_text = "\n".join(logs_to_show)
    st.text_area("", value=log_text, height=200, label_visibility="collapsed", key="log_display")

# Final auto-save
if st.session_state.get('auto_save', True):
    save_session_state()

# Footer
st.markdown("---")
st.markdown(f"""
<div style="text-align: center; color: #6B7280; font-size: 0.9rem; padding: 1rem;">
    <p>🔬 <strong>Piezoelectricity in PVDF Knowledge Explorer</strong> | 
    Session persisted: <strong>{"✅" if st.session_state.auto_save else "❌"}</strong> | 
    PDFs stored: <strong>{len(st.session_state.downloaded_pdfs)}</strong></p>
    <p style="opacity: 0.7;">Data is automatically saved. Use download buttons to export files.</p>
</div>
""", unsafe_allow_html=True)
