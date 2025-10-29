import arxiv
import fitz  # PyMuPDF
import pandas as pd
import streamlit as st
import urllib.request
import os
import re
import sqlite3
from datetime import datetime
import logging
import time
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from tenacity import retry, stop_after_attempt, wait_fixed
import zipfile
import concurrent.futures
import random
from pathlib import Path

# Define database directory and files
DB_DIR = os.path.dirname(os.path.abspath(__file__))
METADATA_DB_FILE = os.path.join(DB_DIR, "piezoelectricity_metadata.db")
UNIVERSE_DB_FILE = os.path.join(DB_DIR, "piezoelectricity_universe.db")

# Initialize logging
logging.basicConfig(filename=os.path.join(DB_DIR, 'piezoelectricity_query.log'), level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize Streamlit app
st.set_page_config(page_title="Piezoelectricity in PVDF Query Tool", layout="wide")
st.title("Piezoelectricity in PVDF Query Tool with SciBERT")
st.markdown("""
This tool queries arXiv for papers on **piezoelectricity in PVDF with dopants like SnO2**, focusing on **alpha and beta phase fractions**, **electrospun nanofibers**, **efficiency**, **electricity generation**, **mechanical force**, and related factors for piezoelectric studies. It uses SciBERT with attention mechanism to prioritize relevant abstracts (>30% relevance) and stores metadata in `piezoelectricity_metadata.db` and full PDF text in `piezoelectricity_universe.db` for fallback searches.
""")

# Dependency check
st.sidebar.header("Setup")
st.sidebar.markdown("""
**Dependencies**:
- `arxiv`, `pymupdf`, `pandas`, `streamlit`, `transformers`, `torch`, `numpy`, `tenacity`
- Install: `pip install arxiv pymupdf pandas streamlit transformers torch numpy tenacity`
""")

# Cache the SciBERT model and tokenizer
@st.cache_resource
def load_scibert():
    tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
    model = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased")
    model.eval()
    return tokenizer, model

try:
    scibert_tokenizer, scibert_model = load_scibert()
except Exception as e:
    st.error(f"Failed to load SciBERT: {e}. Install: `pip install transformers torch`")
    st.stop()

# Create PDFs directory
pdf_dir = os.path.join(DB_DIR, "pdfs")
if not os.path.exists(pdf_dir):
    os.makedirs(pdf_dir)
    st.info(f"Created directory: {pdf_dir}")

# Initialize session state for logs
if "log_buffer" not in st.session_state:
    st.session_state.log_buffer = []

def update_log(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.log_buffer.append(f"[{timestamp}] {message}")
    if len(st.session_state.log_buffer) > 30:
        st.session_state.log_buffer.pop(0)
    logging.info(message)

# Define normalization function with caching
@st.cache_data
def normalize_text(text):
    greek_to_latin = {
        'α': 'alpha', 'β': 'beta', 'γ': 'gamma', 'δ': 'delta', 'ε': 'epsilon',
        'Α': 'alpha', 'Β': 'beta', 'Γ': 'gamma', 'Δ': 'delta', 'Ε': 'epsilon'
    }
    for g, l in greek_to_latin.items():
        text = text.replace(g, l)
    subscripts = {
        '₀': '0', '₁': '1', '₂': '2', '₃': '3', '₄': '4',
        '₅': '5', '₆': '6', '₇': '7', '₈': '8', '₉': '9'
    }
    for s, d in subscripts.items():
        text = text.replace(s, d)
    superscripts = {
        '⁰': '0', '¹': '1', '²': '2', '³': '3', '⁴': '4',
        '⁵': '5', '⁶': '6', '⁷': '7', '⁸': '8', '⁹': '9'
    }
    for s, d in superscripts.items():
        text = text.replace(s, d)
    return text.lower()

# Define key terms for query
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

# Define key patterns as regex
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

# Compile patterns
@st.cache_data
def compile_patterns():
    return [re.compile(pat, re.IGNORECASE) for pat in KEY_PATTERNS]

COMPILED_PATTERNS = compile_patterns()

# SciBERT scoring
@st.cache_data
def score_abstract_with_scibert(abstract):
    try:
        inputs = scibert_tokenizer(abstract, return_tensors="pt", truncation=True, max_length=512, padding=True, return_attention_mask=True)
        with torch.no_grad():
            outputs = scibert_model(**inputs, output_attentions=True)
        abstract_normalized = normalize_text(abstract)
        num_matched = sum(1 for pat in COMPILED_PATTERNS if pat.search(abstract_normalized))
        relevance_prob = np.sqrt(num_matched) / np.sqrt(len(KEY_PATTERNS))
        
        tokens = scibert_tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        keyword_indices = [i for i, token in enumerate(tokens) if any(kw in token.lower() for kw in ['pvdf', 'piezo', 'phase', 'beta', 'alpha'])]
        if keyword_indices:
            attentions = outputs.attentions[-1][0, 0].numpy()
            attn_score = np.sum(attentions[keyword_indices, :]) / len(keyword_indices)
            if attn_score > 0.1:
                relevance_prob = min(relevance_prob + 0.2 * (len(keyword_indices) / len(tokens)), 1.0)
        update_log(f"SciBERT scored: {relevance_prob:.3f} (matched: {num_matched})")
        return relevance_prob
    except Exception as e:
        update_log(f"SciBERT failed: {str(e)}")
        abstract_normalized = normalize_text(abstract)
        num_matched = sum(1 for pat in COMPILED_PATTERNS if pat.search(abstract_normalized))
        relevance_prob = np.sqrt(num_matched) / np.sqrt(len(KEY_PATTERNS))
        update_log(f"Fallback score: {relevance_prob:.3f}")
        return relevance_prob

# Extract text from PDF
@st.cache_data
def extract_text_from_pdf(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text
    except Exception as e:
        update_log(f"PDF extraction failed for {pdf_path}: {str(e)}")
        return f"Error: {str(e)}"

# Initialize database
@st.cache_resource
def get_db_connection(db_file):
    conn = sqlite3.connect(db_file)
    return conn

def initialize_db(db_file):
    try:
        conn = get_db_connection(db_file)
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
                download_status TEXT,
                matched_terms TEXT,
                relevance_prob REAL,
                pdf_path TEXT,
                content TEXT
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS parameters (
                paper_id TEXT,
                entity_text TEXT,
                entity_label TEXT,
                value REAL,
                unit TEXT,
                context TEXT,
                phase TEXT,
                score REAL,
                co_occurrence BOOLEAN,
                FOREIGN KEY (paper_id) REFERENCES papers(id)
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_paper_id ON parameters(paper_id)")
        conn.commit()
        update_log(f"Initialized database schema for {db_file}")
    except Exception as e:
        update_log(f"Failed to initialize {db_file}: {str(e)}")
        st.error(f"Failed to initialize {db_file}: {str(e)}")

# Create universe DB
@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def create_universe_db(paper, db_file=UNIVERSE_DB_FILE):
    try:
        conn = get_db_connection(db_file)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS papers (
                id TEXT PRIMARY KEY,
                title TEXT,
                authors TEXT,
                year INTEGER,
                content TEXT
            )
        """)
        cursor.execute("""
            INSERT OR REPLACE INTO papers (id, title, authors, year, content)
            VALUES (?, ?, ?, ?, ?)
        """, (
            paper["id"],
            paper.get("title", ""),
            paper.get("authors", "Unknown"),
            paper.get("year", 0),
            paper.get("content", "No text extracted")
        ))
        conn.commit()
        update_log(f"Updated {db_file} with paper {paper['id']}")
        return db_file
    except Exception as e:
        update_log(f"Error updating {db_file}: {str(e)}")
        raise

# Save to SQLite
def save_to_sqlite(papers_df, params_list, metadata_db_file=METADATA_DB_FILE):
    try:
        initialize_db(metadata_db_file)
        conn = get_db_connection(metadata_db_file)
        papers_df.to_sql("papers", conn, if_exists="replace", index=False)
        params_df = pd.DataFrame(params_list)
        if not params_df.empty:
            params_df.to_sql("parameters", conn, if_exists="append", index=False)
        update_log(f"Saved {len(papers_df)} papers and {len(params_list)} parameters to {metadata_db_file}")
        return f"Saved to {metadata_db_file}"
    except Exception as e:
        update_log(f"SQLite save failed: {str(e)}")
        return f"Failed to save to SQLite: {str(e)}"

# Query arXiv
@st.cache_data
def query_arxiv(query, categories, max_results, start_year, end_year):
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
            if any(cat in result.categories for cat in categories) and start_year <= result.published.year <= end_year:
                abstract_lower = result.summary.lower()
                title_lower = result.title.lower()
                matched_terms = [term for term in query_words if term in abstract_lower or term in title_lower]
                if not matched_terms:
                    continue
                relevance_prob = score_abstract_with_scibert(result.summary)
                abstract_highlighted = result.summary
                for term in matched_terms:
                    abstract_highlighted = re.sub(r'\b' + re.escape(term) + r'\b', f'<b style="color: orange">{term}</b>', abstract_highlighted, flags=re.IGNORECASE)
                
                papers.append({
                    "id": result.entry_id.split('/')[-1],
                    "title": result.title,
                    "authors": ", ".join([author.name for author in result.authors]),
                    "year": result.published.year,
                    "categories": ", ".join(result.categories),
                    "abstract": result.summary,
                    "abstract_highlighted": abstract_highlighted,
                    "pdf_url": result.pdf_url,
                    "download_status": "Not downloaded",
                    "matched_terms": ", ".join(matched_terms) if matched_terms else "None",
                    "relevance_prob": round(relevance_prob * 100, 2),
                    "pdf_path": None,
                    "content": None
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

# FIXED: Download PDF and extract text (NO @st.cache_data)
@retry(stop=stop_after_attempt(4), wait=wait_fixed(2))
def _download_single_pdf(pdf_url: str, dest_path: Path) -> float:
    req = urllib.request.Request(pdf_url, headers={"User-Agent": "Mozilla/5.0 (compatible; PiezoelectricityTool/1.0)"})
    with urllib.request.urlopen(req, timeout=30) as response, open(dest_path, "wb") as out_file:
        data = response.read()
        out_file.write(data)
    size_kb = len(data) / 1024.0
    if size_kb < 1:
        raise RuntimeError("Downloaded file is empty")
    return size_kb

def download_pdf_and_extract(pdf_url: str, paper_id: str, paper_metadata: dict):
    pdf_path = Path(pdf_dir) / f"{paper_id}.pdf"
    try:
        size_kb = _download_single_pdf(pdf_url, pdf_path)
        time.sleep(random.uniform(0.3, 0.8))  # Be kind to arXiv

        text = extract_text_from_pdf(str(pdf_path))
        if text.startswith("Error"):
            raise RuntimeError(text)

        paper_data = {
            "id": paper_id,
            "title": paper_metadata.get("title", ""),
            "authors": paper_metadata.get("authors", "Unknown"),
            "year": paper_metadata.get("year", 0),
            "content": text,
        }
        create_universe_db(paper_data)

        status = f"Downloaded ({size_kb:.1f} KB)"
        update_log(f"Success {paper_id}: {status}")
        return status, str(pdf_path), text

    except Exception as exc:
        msg = f"Failed: {exc}"
        update_log(f"Download error {paper_id}: {exc}")
        if pdf_path.exists() and pdf_path.stat().st_size == 0:
            pdf_path.unlink(missing_ok=True)
        return msg, None, f"Error: {exc}"

# Concurrent download wrapper
def download_paper(paper):
    if paper["pdf_url"]:
        status, pdf_path, content = download_pdf_and_extract(paper["pdf_url"], paper["id"], paper)
        paper["download_status"] = status
        paper["pdf_path"] = pdf_path
        paper["content"] = content
    update_log(f"Processed: {paper['title']}")

# Create ZIP
@st.cache_data
def create_pdf_zip(pdf_paths):
    zip_path = os.path.join(DB_DIR, "piezoelectricity_pdfs.zip")
    try:
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for pdf in pdf_paths:
                if pdf and os.path.exists(pdf):
                    zipf.write(pdf, os.path.basename(pdf))
        update_log(f"Created ZIP: {zip_path}")
        return zip_path
    except Exception as e:
        update_log(f"ZIP creation failed: {str(e)}")
        return None

# Main UI
st.header("arXiv Query for Piezoelectricity in Doped PVDF")
st.markdown("Search for abstracts on **piezoelectricity**, **electrospun nanofibers**, **PVDF**, **alpha/beta phases**, **SnO2 dopants**, **efficiency**, **electricity generation**, **mechanical force** using SciBERT.")

log_container = st.empty()
def display_logs():
    log_container.text_area("Processing Logs", "\n".join(st.session_state.log_buffer), height=200)

with st.sidebar:
    st.subheader("Search Parameters")
    query = st.text_input("Query", value=' OR '.join([f'"{term}"' for term in KEY_TERMS]))
    default_categories = ["cond-mat.mtrl-sci", "physics.app-ph", "physics.chem-ph"]
    categories = st.multiselect("Categories", default_categories, default=default_categories)
    max_results = st.slider("Max Papers", min_value=1, max_value=500, value=10)
    current_year = datetime.now().year
    col1, col2 = st.columns(2)
    with col1:
        start_year = st.number_input("Start Year", min_value=1990, max_value=current_year, value=2010)
    with col2:
        end_year = st.number_input("End Year", min_value=start_year, max_value=current_year, value=current_year)
    output_formats = st.multiselect("Output Formats", ["SQLite (.db)", "CSV", "JSON"], default=["SQLite (.db)"])
    search_button = st.button("Search arXiv")

if search_button:
    if not query.strip():
        st.error("Enter a valid query.")
    elif not categories:
        st.error("Select at least one category.")
    elif start_year > end_year:
        st.error("Start year must be ≤ end year.")
    else:
        with st.spinner("Querying arXiv..."):
            papers = query_arxiv(query, categories, max_results, start_year, end_year)
        
        if not papers:
            st.warning("No papers found. Broaden query or categories.")
        else:
            st.success(f"Found **{len(papers)}** papers. Filtering for relevance > 30%...")
            relevant_papers = [p for p in papers if p["relevance_prob"] > 30.0]
            if not relevant_papers:
                st.warning("No papers with relevance > 30%. Check logs.")
            else:
                st.success(f"**{len(relevant_papers)}** papers with relevance > 30%. Downloading PDFs...")
                progress_bar = st.progress(0)
                with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                    futures = [executor.submit(download_paper, paper) for paper in relevant_papers]
                    for i, future in enumerate(concurrent.futures.as_completed(futures)):
                        future.result()
                        progress_bar.progress((i + 1) / len(relevant_papers))
                        time.sleep(0.3)
                
                df = pd.DataFrame(relevant_papers)
                st.subheader("Papers (Relevance > 30%)")
                df_display = df[["id", "title", "year", "categories", "abstract_highlighted", "matched_terms", "relevance_prob", "download_status"]].copy()
                df_display["PDF Link"] = [f"[View PDF]({url})" for url in df["pdf_url"]]
                st.dataframe(df_display, use_container_width=True, unsafe_allow_html=True)
                
                # ZIP Download
                valid_paths = [p['pdf_path'] for p in relevant_papers if p['pdf_path'] and os.path.exists(p['pdf_path'])]
                zip_path = create_pdf_zip(tuple(valid_paths))
                if zip_path and os.path.exists(zip_path):
                    with open(zip_path, 'rb') as f:
                        st.download_button(
                            label="Download PDFs as ZIP",
                            data=f,
                            file_name="piezoelectricity_pdfs.zip",
                            mime="application/zip"
                        )
                
                # Output formats
                if "SQLite (.db)" in output_formats:
                    sqlite_status = save_to_sqlite(df.drop(columns=["abstract_highlighted"]), [])
                    st.info(sqlite_status)
                
                if "CSV" in output_formats:
                    csv = df.drop(columns=["abstract_highlighted"]).to_csv(index=False)
                    st.download_button("Download CSV", data=csv, file_name="piezoelectricity_papers.csv", mime="text/csv")
                
                if "JSON" in output_formats:
                    json_data = df.drop(columns=["abstract_highlighted"]).to_json(orient="records", lines=True)
                    st.download_button("Download JSON", data=json_data, file_name="piezoelectricity_papers.json", mime="application/json")
                
                display_logs()
