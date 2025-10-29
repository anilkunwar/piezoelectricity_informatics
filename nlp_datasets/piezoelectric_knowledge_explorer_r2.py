import arxiv
import fitz  # PyMuPDF
import pandas as pd
import streamlit as st
import os
import re
import sqlite3
from datetime import datetime
import logging
import time
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from collections import Counter
from tenacity import retry, stop_after_attempt, wait_fixed
import zipfile
import gc
import psutil
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import concurrent.futures

# ===== CLOUD-OPTIMIZED CONFIGURATION =====
if os.path.exists("/tmp"):  # Cloud environments
    DB_DIR = "/tmp"
else:
    DB_DIR = os.path.join(os.path.expanduser("~"), "Desktop")

os.makedirs(DB_DIR, exist_ok=True)
pdf_dir = os.path.join(DB_DIR, "pdfs")
os.makedirs(pdf_dir, exist_ok=True)

METADATA_DB_FILE = os.path.join(DB_DIR, "piezoelectricity_metadata.db")
UNIVERSE_DB_FILE = os.path.join(DB_DIR, "piezoelectricity_universe.db")

# Initialize logging
logging.basicConfig(filename=os.path.join(DB_DIR, 'piezoelectricity_query.log'), 
                   level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize Streamlit app
st.set_page_config(page_title="Piezoelectricity in PVDF Query Tool", layout="wide")
st.title("Piezoelectricity in PVDF Query Tool with SciBERT")
st.markdown("""
This tool queries arXiv for papers on **piezoelectricity in PVDF with dopants like SnO2**, focusing on **alpha/beta phase fractions**, **electrospun nanofibers**, **efficiency**, **electricity generation**, **mechanical force**, and related factors. It uses SciBERT with attention mechanism to prioritize relevant abstracts and stores metadata in `piezoelectricity_metadata.db` and full PDF text in `piezoelectricity_universe.db`. PDFs and databases can be downloaded as ZIP.
""")

# Dependency check
st.sidebar.header("Setup")
st.sidebar.markdown("""
**Dependencies**:
- `arxiv`, `pymupdf`, `pandas`, `streamlit`, `transformers`, `torch`, `numpy`, `tenacity`, `requests`, `psutil`
- Install: `pip install arxiv pymupdf pandas streamlit transformers torch numpy tenacity requests psutil`
""")

# ===== RESOURCE MANAGEMENT =====
def check_memory_usage():
    try:
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024  # MB
    except:
        return 0

def cleanup_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def system_health_check():
    try:
        memory_usage = check_memory_usage()
        disk_usage = psutil.disk_usage(DB_DIR)
        disk_free_gb = disk_usage.free / (1024**3)
        
        update_log(f"Health - Memory: {memory_usage:.1f}MB, Disk free: {disk_free_gb:.1f}GB")
        
        if memory_usage > 1500:
            st.warning(f"High memory ({memory_usage:.1f}MB), processing may be slow")
            cleanup_memory()
        if disk_free_gb < 0.5:
            st.error(f"Low disk space ({disk_free_gb:.1f}GB)")
            return False
        return True
    except Exception as e:
        update_log(f"Health check warning: {str(e)}")
        return True

def create_retry_session():
    session = requests.Session()
    retry_strategy = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

def limit_pdf_processing(papers, max_pdfs=10):
    if len(papers) > max_pdfs and os.path.exists("/tmp"):
        st.warning(f"Cloud: Limiting to {max_pdfs} PDFs")
        return papers[:max_pdfs]
    return papers

# ===== SESSION STATE MANAGEMENT =====
if "log_buffer" not in st.session_state:
    st.session_state.log_buffer = []
if "processing" not in st.session_state:
    st.session_state.processing = False
if "download_files" not in st.session_state:
    st.session_state.download_files = {"pdf_paths": [], "zip_path": None}
if "search_results" not in st.session_state:
    st.session_state.search_results = None
if "relevant_papers" not in st.session_state:
    st.session_state.relevant_papers = None
if "relevance_threshold" not in st.session_state:
    st.session_state.relevance_threshold = 30

def reset_processing():
    st.session_state.processing = False

def reset_downloads():
    st.session_state.download_files = {"pdf_paths": [], "zip_path": None}
    st.session_state.search_results = None
    st.session_state.relevant_papers = None
    query_arxiv.clear()
    cleanup_memory()
    update_log("Downloads reset")

def update_log(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {message}"
    st.session_state.log_buffer.append(log_entry)
    if len(st.session_state.log_buffer) > 30:
        st.session_state.log_buffer.pop(0)
    logging.info(message)

# ===== MODEL LOADING =====
@st.cache_resource
def load_scibert_model():
    try:
        tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
        model = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased")
        model.eval()
        update_log("SciBERT loaded")
        return tokenizer, model
    except Exception as e:
        st.error(f"SciBERT load failed: {e}")
        st.stop()

scibert_tokenizer, scibert_model = load_scibert_model()

# ===== KEY TERMS AND SCORING =====
KEY_TERMS = [
    "piezoelectricity", "electrospun nanofibers", "PVDF", "alpha phase", "beta phase",
    "SnO2", "dopants", "efficiency", "electricity generation", "mechanical force",
    "nanogenerators", "d33", "energy harvesting", "doped PVDF"
]

KEY_PATTERNS = [
    r'\bpiezoelectric(?:ity| effect| performance| properties| coefficient| constant| polymer| materials| harvester| generator)?\b',
    r'\belectrospun (?:nano)?fibers?|nanofiber mats|electrospinning\b',
    r'\bpvdf|polyvinylidene fluoride|pvd?f|p\(vdf-trfe\)|p\(vdf-hfp\)\b',
    r'\b(alpha|beta|gamma|delta)\s*(?:phase|polymorph)\b',
    r'\bsno2|tin oxide|tin dioxide\b',
    r'\bdopants?|doped|doping|additives?\b',
    r'\befficiency|conversion efficiency\b',
    r'\belectricity generation|power output|voltage output\b',
    r'\bmechanical (?:force|stress|deformation|energy)\b',
    r'\bd33|d31|g33\b',
    r'\benergy harvesting|nanogenerators?\b'
]

@st.cache_data
def compile_patterns():
    return [re.compile(pat, re.IGNORECASE) for pat in KEY_PATTERNS]

COMPILED_PATTERNS = compile_patterns()

def score_abstract_with_scibert(abstract):
    try:
        inputs = scibert_tokenizer(abstract, return_tensors="pt", truncation=True, max_length=512, padding=True)
        with torch.no_grad():
            outputs = scibert_model(**inputs, output_attentions=True)
        
        abstract_norm = abstract.lower()
        num_matched = sum(1 for pat in COMPILED_PATTERNS if pat.search(abstract_norm))
        relevance_prob = np.sqrt(num_matched) / np.sqrt(len(KEY_PATTERNS))

        tokens = scibert_tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        keyword_indices = [i for i, token in enumerate(tokens) if any(kw in token.lower() for kw in KEY_TERMS)]
        if keyword_indices:
            attentions = outputs.attentions[-1][0, 0].numpy()
            attn_score = np.sum(attentions[keyword_indices, :]) / len(keyword_indices)
            if attn_score > 0.1:
                relevance_prob = min(relevance_prob + 0.2 * len(keyword_indices) / len(tokens), 1.0)
        
        update_log(f"SciBERT score: {relevance_prob:.3f}")
        return relevance_prob
    except Exception as e:
        update_log(f"SciBERT failed: {str(e)}")
        abstract_norm = abstract.lower()
        num_matched = sum(1 for pat in COMPILED_PATTERNS if pat.search(abstract_norm))
        return np.sqrt(num_matched) / np.sqrt(len(KEY_PATTERNS))

# ===== PDF PROCESSING =====
def extract_text_from_pdf(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text
    except Exception as e:
        update_log(f"PDF extract failed: {str(e)}")
        return f"Error: {str(e)}"

def update_db_content(db_file, paper_id, content):
    try:
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        cursor.execute("UPDATE papers SET content = ? WHERE id = ?", (content, paper_id))
        if cursor.rowcount == 0 and 'universe' in db_file:
            cursor.execute("INSERT INTO papers (id, title, authors, year, content) VALUES (?, ?, ?, ?, ?)",
                           (paper_id, "Unknown", "Unknown", 0, content))
        conn.commit()
        conn.close()
    except Exception as e:
        update_log(f"DB update failed: {str(e)}")

# ===== BATCH PROCESSING =====
def batch_convert_pdfs():
    pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith('.pdf')]
    if not pdf_files:
        update_log("No PDFs to convert.")
        return
    
    if not system_health_check():
        st.error("Health check failed.")
        return
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, filename in enumerate(pdf_files):
        pdf_path = os.path.join(pdf_dir, filename)
        paper_id = filename[:-4]
        status_text.text(f"Converting {i+1}/{len(pdf_files)}: {filename}")
        
        text = extract_text_from_pdf(pdf_path)
        if not text.startswith("Error"):
            update_db_content(METADATA_DB_FILE, paper_id, text)
            update_db_content(UNIVERSE_DB_FILE, paper_id, text)
        
        progress_bar.progress((i + 1) / len(pdf_files))
        time.sleep(0.1)
        if i % 5 == 0:
            cleanup_memory()
    
    status_text.empty()
    cleanup_memory()

# ===== DATABASE (with @retry) =====
@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def create_universe_db(paper, db_file=UNIVERSE_DB_FILE):
    try:
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS papers (
                id TEXT PRIMARY KEY, title TEXT, authors TEXT, year INTEGER, content TEXT
            )
        """)
        cursor.execute("""
            INSERT OR REPLACE INTO papers (id, title, authors, year, content)
            VALUES (?, ?, ?, ?, ?)
        """, (paper["id"], paper.get("title", ""), paper.get("authors", "Unknown"), paper.get("year", 0), paper.get("content", "")))
        conn.commit()
        conn.close()
        update_log(f"Universe DB updated: {paper['id']}")
    except Exception as e:
        update_log(f"Universe DB error: {str(e)}")
        raise

# ===== DATA STORAGE =====
def save_to_sqlite(papers_df, params_list, metadata_db_file=METADATA_DB_FILE):
    try:
        conn = sqlite3.connect(metadata_db_file)
        papers_df.to_sql("papers", conn, if_exists="replace", index=False)
        if params_list:
            pd.DataFrame(params_list).to_sql("parameters", conn, if_exists="append", index=False)
        conn.close()
        update_log(f"Saved {len(papers_df)} papers")
        return "Saved to DB"
    except Exception as e:
        update_log(f"SQLite save failed: {str(e)}")
        return f"Failed: {str(e)}"

# ===== ARXIV QUERY (no @retry) =====
@st.cache_data(ttl=3600)
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
        query_words = {w.strip('"').lower() for w in query.split() if w.strip('"')}
        
        for result in client.results(search):
            if any(cat in result.categories for cat in categories) and start_year <= result.published.year <= end_year:
                abstract_lower = result.summary.lower()
                title_lower = result.title.lower()
                matched_terms = [w for w in query_words if w in abstract_lower or w in title_lower]
                if not matched_terms:
                    continue
                
                relevance_prob = score_abstract_with_scibert(result.summary)
                abstract_highlighted = result.summary
                for term in matched_terms:
                    abstract_highlighted = re.sub(rf'\b{re.escape(term)}\b', f'<b style="color: orange">{term}</b>', abstract_highlighted, flags=re.IGNORECASE)
                
                papers.append({
                    "id": result.entry_id.split('/')[-1],
                    "title": result.title,
                    "authors": ", ".join([a.name for a in result.authors]),
                    "year": result.published.year,
                    "categories": ", ".join(result.categories),
                    "abstract": result.summary,
                    "abstract_highlighted": abstract_highlighted,
                    "pdf_url": result.pdf_url,
                    "download_status": "Not downloaded",
                    "matched_terms": ", ".join(matched_terms),
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
        st.error(f"arXiv error: {str(e)}")
        return []

# ===== PDF DOWNLOAD (no @retry) =====
def download_pdf_and_extract(pdf_url, paper_id, paper_metadata):
    pdf_path = os.path.join(pdf_dir, f"{paper_id}.pdf")
    session = create_retry_session()
    try:
        response = session.get(pdf_url, timeout=30)
        response.raise_for_status()
        
        with open(pdf_path, 'wb') as f:
            f.write(response.content)
            
        file_size = os.path.getsize(pdf_path) / 1024
        text = extract_text_from_pdf(pdf_path)
        
        if not text.startswith("Error"):
            paper_data = {
                "id": paper_id,
                "title": paper_metadata.get("title", ""),
                "authors": paper_metadata.get("authors", "Unknown"),
                "year": paper_metadata.get("year", 0),
                "content": text
            }
            create_universe_db(paper_data)
            update_db_content(METADATA_DB_FILE, paper_id, text)
            update_log(f"Downloaded {paper_id} ({file_size:.1f} KB)")
            return f"Downloaded ({file_size:.1f} KB)", pdf_path, text
        else:
            return f"Failed: {text}", None, text
    except Exception as e:
        update_log(f"Download failed {paper_id}: {str(e)}")
        return f"Failed: {str(e)}", None, f"Error: {str(e)}"
    finally:
        session.close()

def download_paper(paper):
    if not paper["pdf_url"]:
        return
    for attempt in range(3):
        try:
            status, pdf_path, content = download_pdf_and_extract(paper["pdf_url"], paper["id"], paper)
            paper["download_status"] = status
            paper["pdf_path"] = pdf_path
            paper["content"] = content
            return
        except Exception as e:
            update_log(f"Attempt {attempt+1}/3 failed for {paper['id']}: {e}")
            time.sleep(2)
    paper["download_status"] = "Failed after 3 attempts"

# ===== FILE MANAGEMENT =====
def create_pdf_zip(pdf_paths):
    zip_path = os.path.join(DB_DIR, "piezoelectricity_pdfs.zip")
    try:
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for pdf_path in pdf_paths:
                if pdf_path and os.path.exists(pdf_path):
                    zipf.write(pdf_path, os.path.basename(pdf_path))
        update_log(f"ZIP created: {zip_path}")
        return zip_path
    except Exception as e:
        update_log(f"ZIP failed: {str(e)}")
        return None

def read_file_for_download(file_path):
    try:
        with open(file_path, "rb") as f:
            return f.read()
    except Exception as e:
        update_log(f"Read failed {file_path}: {str(e)}")
        return None

# ===== STREAMLIT UI =====
st.header("arXiv Query for Piezoelectricity in Doped PVDF")
st.markdown("Search for **PVDF**, **SnO2 dopants**, **beta phase**, **nanofibers**, **energy harvesting** using SciBERT.")

log_container = st.empty()
def display_logs():
    log_container.text_area("Processing Logs", "\n".join(st.session_state.log_buffer), height=200)

# Sidebar
with st.sidebar:
    st.subheader("Search Parameters")
    query = st.text_input("Query", value=' OR '.join([f'"{t}"' for t in KEY_TERMS]), key="query_input")
    default_categories = ["cond-mat.mtrl-sci", "physics.app-ph"]
    categories = st.multiselect("Categories", default_categories, default=default_categories, key="categories_select")
    max_results = st.slider("Max Papers", 1, 200, 10, key="max_results_slider")
    current_year = datetime.now().year
    col1, col2 = st.columns(2)
    with col1:
        start_year = st.number_input("Start Year", 1990, current_year, 2010, key="start_year_input")
    with col2:
        end_year = st.number_input("End Year", start_year, current_year, current_year, key="end_year_input")
    
    st.session_state.relevance_threshold = st.slider(
        "Relevance Threshold (%)", 0, 100, st.session_state.relevance_threshold, key="relevance_slider"
    )
    
    output_formats = st.multiselect("Output Formats", ["SQLite (.db)", "CSV", "JSON"], default=["SQLite (.db)"], key="output_formats_select")
    
    st.subheader("Cloud Settings")
    enable_cloud_limits = st.checkbox("Enable Cloud Optimization", value=os.path.exists("/tmp"), key="cloud_optimization_checkbox")
    max_pdf_downloads = st.slider("Max PDF Downloads", 1, 200, 10, key="max_pdf_downloads_slider")
    
    search_button = st.button("Search arXiv", key="search_button")
    convert_button = st.button("Update DBs from Existing PDFs", key="convert_button")
    reset_downloads_button = st.button("Reset Downloads", key="reset_downloads_button")

# Reset
if reset_downloads_button:
    reset_downloads()
    st.success("Downloads reset.")

# Batch convert
if convert_button:
    if st.session_state.processing:
        st.warning("Processing in progress...")
    else:
        st.session_state.processing = True
        with st.spinner("Converting PDFs to DB..."):
            batch_convert_pdfs()
        display_logs()
        st.success("DB update complete.")
        st.session_state.processing = False

# Restore results
if st.session_state.search_results and st.session_state.relevant_papers:
    df = pd.DataFrame(st.session_state.relevant_papers)
    st.subheader(f"Papers (Relevance > {st.session_state.relevance_threshold}%)")
    st.dataframe(df[["id", "title", "year", "relevance_prob", "download_status"]], use_container_width=True)
    
    if "SQLite (.db)" in output_formats:
        st.info(save_to_sqlite(df.drop(columns=["abstract_highlighted"]), []))

# Search
if search_button:
    if st.session_state.processing:
        st.warning("Processing in progress...")
        st.stop()
    
    if not query.strip() or not categories or start_year > end_year:
        st.error("Invalid input.")
    else:
        st.session_state.processing = True
        st.session_state.download_files = {"pdf_paths": [], "zip_path": None}
        
        try:
            if not system_health_check():
                st.error("Health check failed.")
                reset_processing()
                st.stop()
            
            with st.spinner("Querying arXiv..."):
                papers = query_arxiv(query, categories, max_results, start_year, end_year)
            
            if not papers:
                st.warning("No papers found.")
            else:
                st.success(f"Found {len(papers)} papers.")
                relevant_papers = [p for p in papers if p["relevance_prob"] > st.session_state.relevance_threshold]
                
                if not relevant_papers:
                    st.warning(f"No papers above {st.session_state.relevance_threshold}% relevance.")
                else:
                    if enable_cloud_limits:
                        relevant_papers = limit_pdf_processing(relevant_papers, max_pdf_downloads)
                    
                    st.success(f"Downloading {len(relevant_papers)} PDFs...")
                    progress_bar = st.progress(0)
                    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                        futures = [executor.submit(download_paper, p) for p in relevant_papers]
                        for i, f in enumerate(concurrent.futures.as_completed(futures)):
                            f.result()
                            progress_bar.progress((i + 1) / len(relevant_papers))
                            time.sleep(0.5)
                    
                    pdf_paths = [p["pdf_path"] for p in relevant_papers if p["pdf_path"]]
                    st.session_state.search_results = papers
                    st.session_state.relevant_papers = relevant_papers
                    st.session_state.download_files["pdf_paths"] = pdf_paths
                    
                    df = pd.DataFrame(relevant_papers)
                    st.subheader(f"Papers (Relevance > {st.session_state.relevance_threshold}%)")
                    st.dataframe(df[["id", "title", "year", "relevance_prob", "download_status"]], use_container_width=True)
                    
                    if "SQLite (.db)" in output_formats:
                        st.info(save_to_sqlite(df.drop(columns=["abstract_highlighted"]), []))
                    
                    if pdf_paths:
                        st.subheader("Individual PDF Downloads")
                        for idx, path in enumerate(pdf_paths):
                            data = read_file_for_download(path)
                            if data:
                                st.download_button(f"Download {os.path.basename(path)}", data, os.path.basename(path), "application/pdf", key=f"pdf_{idx}_{time.time()}")
                        
                        zip_path = create_pdf_zip(pdf_paths)
                        if zip_path:
                            st.session_state.download_files["zip_path"] = zip_path
                            data = read_file_for_download(zip_path)
                            if data:
                                st.download_button("Download All PDFs as ZIP", data, "piezoelectricity_pdfs.zip", "application/zip", key=f"zip_{time.time()}")
                    
                    st.subheader("Database Downloads")
                    for db_file, name in [(METADATA_DB_FILE, "Metadata DB"), (UNIVERSE_DB_FILE, "Universe DB")]:
                        if os.path.exists(db_file):
                            data = read_file_for_download(db_file)
                            if data:
                                st.download_button(f"Download {name}", data, os.path.basename(db_file), "application/octet-stream", key=f"db_{name}_{time.time()}")
                        else:
                            st.warning(f"{name} not found.")
                
                display_logs()
                
        except Exception as e:
            st.error(f"Error: {str(e)}")
            update_log(f"Processing error: {str(e)}")
        finally:
            reset_processing()
            cleanup_memory()

# System info
with st.sidebar:
    st.subheader("System Info")
    st.write(f"Memory: {check_memory_usage():.1f} MB")
    st.write(f"DB Dir: {DB_DIR}")
    if st.button("Clear Memory Cache"):
        cleanup_memory()
        st.success("Cache cleared")

if st.session_state.log_buffer:
    with st.sidebar:
        st.subheader("Recent Logs")
        for log in st.session_state.log_buffer[-5:]:
            st.text(log)

st.markdown("---")
st.markdown("*Cloud-optimized • No RetryError • Ready for Deployment*")
