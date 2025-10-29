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
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from scipy.special import softmax
from collections import Counter
import numpy as np
from tenacity import retry, stop_after_attempt, wait_fixed
import zipfile
import gc
import psutil
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import concurrent.futures
from pathlib import Path
import json

# ====================== CLOUD-OPTIMIZED CONFIG ======================
if os.path.exists("/tmp"):  # Streamlit Cloud / most cloud platforms
    DB_DIR = "/tmp"
else:
    DB_DIR = os.path.join(os.path.expanduser("~"), "Desktop")

os.makedirs(DB_DIR, exist_ok=True)
pdf_dir = os.path.join(DB_DIR, "pdfs")
os.makedirs(pdf_dir, exist_ok=True)

METADATA_DB_FILE = os.path.join(DB_DIR, "piezoelectricity_metadata.db")
UNIVERSE_DB_FILE = os.path.join(DB_DIR, "piezoelectricity_universe.db")

# ====================== LOGGING ======================
logging.basicConfig(
    filename=os.path.join(DB_DIR, 'piezoelectricity_query.log'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# ====================== STREAMLIT SETUP ======================
st.set_page_config(page_title="Piezoelectricity in PVDF Query Tool", layout="wide")
st.title("Piezoelectricity in PVDF Query Tool with SciBERT")
st.markdown("""
This tool queries arXiv for **piezoelectricity in PVDF** (dopants, phases, electrospinning, efficiency, etc.).  
SciBERT scores abstracts, and a **relevance slider** lets you control the cutoff.  
PDFs, metadata, and full-text are stored and downloadable as ZIP + DB + CSV/JSON.
""")

# ====================== SESSION STATE ======================
for key in ["log_buffer", "processing", "current_progress", "download_files",
            "search_results", "relevant_papers", "search_params"]:
    if key not in st.session_state:
        st.session_state[key] = [] if key == "log_buffer" else False if key == "processing" else 0 if key == "current_progress" else {"pdf_paths": [], "zip_path": None} if key == "download_files" else None

def update_log(message: str):
    """Thread-safe: only main thread writes to session_state"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = f"[{timestamp}] {message}"
    if st.session_state.log_buffer is not None:
        st.session_state.log_buffer.append(entry)
        if len(st.session_state.log_buffer) > 30:
            st.session_state.log_buffer.pop(0)
    logging.info(message)

# ====================== SYSTEM HEALTH ======================
def check_memory_usage():
    try:
        return psutil.Process().memory_info().rss / 1024 / 1024
    except:
        return 0

def cleanup_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def system_health_check():
    mem = check_memory_usage()
    disk = psutil.disk_usage(DB_DIR)
    free_gb = disk.free / (1024**3)
    update_log(f"Health - RAM: {mem:.1f}MB, Disk free: {free_gb:.1f}GB")
    if mem > 1500:
        st.warning(f"High RAM ({mem:.1f}MB)")
        cleanup_memory()
    if free_gb < 0.5:
        st.error(f"Low disk ({free_gb:.1f}GB)")
        return False
    return True

# ====================== HTTP RETRY SESSION ======================
def create_retry_session():
    session = requests.Session()
    retry = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

# ====================== MODEL LOADING ======================
@st.cache_resource
def load_scibert_model():
    try:
        tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
        model = AutoModelForSequenceClassification.from_pretrained("allenai/scibert_scivocab_uncased")
        model.eval()
        update_log("SciBERT loaded")
        return tokenizer, model
    except Exception as e:
        st.error(f"SciBERT load failed: {e}")
        st.stop()

scibert_tokenizer, scibert_model = load_scibert_model()

# ====================== KEY TERMS ======================
KEY_TERMS = [
    "piezoelectricity", "piezoelectric effect", "piezoelectric performance",
    "electrospun nanofibers", "PVDF", "polyvinylidene fluoride", "beta phase",
    "alpha phase", "SnO2", "dopants", "doping", "efficiency", "nanogenerators",
    "power output", "mechanical stress", "phase fraction", "energy harvesting"
]

# ====================== SCORING ======================
def score_abstract_with_scibert(abstract: str) -> float:
    try:
        inputs = scibert_tokenizer(abstract, return_tensors="pt", truncation=True, max_length=512, padding=True)
        with torch.no_grad():
            outputs = scibert_model(**inputs, output_attentions=True)
        logits = outputs.logits.numpy()
        prob = float(softmax(logits, axis=1)[0][1])

        # Boost if keywords appear
        tokens = scibert_tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        kw_idx = [i for i, t in enumerate(tokens) if any(kw in t.lower() for kw in KEY_TERMS)]
        if kw_idx:
            att = outputs.attentions[-1][0, 0].numpy()
            boost = np.sum(att[kw_idx, :]) / len(kw_idx)
            if boost > 0.1 and prob < 0.5:
                prob = min(prob + 0.2 * len(kw_idx) / len(tokens), 1.0)
        return prob
    except Exception as e:
        update_log(f"SciBERT fail: {e}")
        # Fallback
        words = Counter(re.findall(r'\b\w+\b', abstract.lower()))
        total = sum(words.values())
        score = sum(words.get(kw.lower(), 0) for kw in KEY_TERMS)
        return min(score / (total + 1e-6) * 10, 1.0)

# ====================== PDF EXTRACTION ======================
def extract_text_from_pdf(pdf_path: str) -> str:
    try:
        doc = fitz.open(pdf_path)
        text = "".join(page.get_text() for page in doc)
        doc.close()
        return text
    except Exception as e:
        return f"Error: {e}"

# ====================== DATABASE ======================
def initialize_db(db_file: str):
    conn = sqlite3.connect(db_file)
    cur = conn.cursor()
    if "universe" in db_file:
        cur.execute("""CREATE TABLE IF NOT EXISTS papers (
            id TEXT PRIMARY KEY, title TEXT, authors TEXT, year INTEGER, content TEXT)""")
    else:
        cur.execute("""CREATE TABLE IF NOT EXISTS papers (
            id TEXT PRIMARY KEY, title TEXT, authors TEXT, year INTEGER, categories TEXT,
            abstract TEXT, pdf_url TEXT, download_status TEXT, matched_terms TEXT,
            relevance_prob REAL, pdf_path TEXT, content TEXT)""")
        cur.execute("""CREATE TABLE IF NOT EXISTS parameters (
            paper_id TEXT, entity_text TEXT, entity_label TEXT, value REAL, unit TEXT,
            context TEXT, phase TEXT, score REAL, co_occurrence BOOLEAN,
            FOREIGN KEY(paper_id) REFERENCES papers(id))""")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_paper_id ON parameters(paper_id)")
    conn.commit()
    conn.close()

initialize_db(METADATA_DB_FILE)
initialize_db(UNIVERSE_DB_FILE)

@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def update_universe_db(paper: dict):
    conn = sqlite3.connect(UNIVERSE_DB_FILE)
    cur = conn.cursor()
    cur.execute("""INSERT OR REPLACE INTO papers (id, title, authors, year, content)
                   VALUES (?, ?, ?, ?, ?)""",
                (paper["id"], paper.get("title",""), paper.get("authors","Unknown"),
                 paper.get("year",0), paper.get("content","")))
    conn.commit()
    conn.close()

# ====================== ARXIV QUERY ======================
@st.cache_data(ttl=3600)
def query_arxiv(query: str, categories: list, max_results: int, start_year: int, end_year: int):
    client = arxiv.Client()
    search = arxiv.Search(query=query, max_results=max_results,
                          sort_by=arxiv.SortCriterion.Relevance,
                          sort_order=arxiv.SortOrder.Descending)
    papers = []
    query_words = {w.strip('"').lower() for w in query.split("OR")}
    for r in client.results(search):
        if not (any(c in r.categories for c in categories) and start_year <= r.published.year <= end_year):
            continue
        matched = [w for w in query_words if w in r.summary.lower() or w in r.title.lower()]
        if not matched: continue
        rel = score_abstract_with_scibert(r.summary)
        papers.append({
            "id": r.entry_id.split("/")[-1],
            "title": r.title,
            "authors": ", ".join(a.name for a in r.authors),
            "year": r.published.year,
            "categories": ", ".join(r.categories),
            "abstract": r.summary,
            "pdf_url": r.pdf_url,
            "download_status": "Pending",
            "matched_terms": ", ".join(matched),
            "relevance_prob": round(rel * 100, 2),
            "pdf_path": None,
            "content": None
        })
        if len(papers) >= max_results: break
    return sorted(papers, key=lambda x: x["relevance_prob"], reverse=True)

# ====================== PDF DOWNLOAD (THREAD-SAFE) ======================
@retry(stop=stop_after_attempt(4), wait=wait_fixed(2))
def _download_pdf(url: str, dest: Path) -> float:
    req = requests.Request("GET", url, headers={"User-Agent": "PiezoelectricityTool/1.0"})
    prep = req.prepare()
    session = create_retry_session()
    resp = session.send(prep, timeout=30)
    resp.raise_for_status()
    with open(dest, "wb") as f:
        f.write(resp.content)
    size_kb = len(resp.content) / 1024
    if size_kb < 1:
        raise RuntimeError("Empty PDF")
    return size_kb

def download_and_extract(paper: dict) -> dict:
    """Returns updated paper dict (thread-safe, no st.session_state inside)"""
    pid = paper["id"]
    path = Path(pdf_dir) / f"{pid}.pdf"
    try:
        size_kb = _download_pdf(paper["pdf_url"], path)
        time.sleep(0.5 + random.uniform(0, 0.5))
        text = extract_text_from_pdf(str(path))
        if text.startswith("Error"):
            raise RuntimeError(text)
        update_universe_db({"id": pid, "title": paper["title"], "authors": paper["authors"],
                            "year": paper["year"], "content": text})
        paper.update({
            "download_status": f"Downloaded ({size_kb:.1f} KB)",
            "pdf_path": str(path),
            "content": text
        })
    except Exception as e:
        paper.update({
            "download_status": f"Failed: {e}",
            "pdf_path": None,
            "content": f"Error: {e}"
        })
        if path.exists() and path.stat().st_size == 0:
            path.unlink(missing_ok=True)
    return paper

# ====================== ZIP & SAVE ======================
def create_zip(pdf_paths: list) -> str:
    zip_path = os.path.join(DB_DIR, "piezoelectricity_pdfs.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for p in pdf_paths:
            if p and os.path.exists(p):
                z.write(p, os.path.basename(p))
    return zip_path

def save_to_sqlite(df: pd.DataFrame):
    conn = sqlite3.connect(METADATA_DB_FILE)
    df.to_sql("papers", conn, if_exists="replace", index=False)
    conn.close()
    return "Saved to metadata DB"

# ====================== UI ======================
st.sidebar.header("Search Parameters")
query = st.sidebar.text_input("Query", value=' OR '.join([f'"{t}"' for t in KEY_TERMS]))
cats = st.sidebar.multiselect("Categories", ["cond-mat.mtrl-sci", "physics.app-ph"], default=["cond-mat.mtrl-sci"])
max_res = st.sidebar.slider("Max Results", 1, 200, 20)
col1, col2 = st.sidebar.columns(2)
start_y = col1.number_input("Start Year", 1990, datetime.now().year, 2010)
end_y = col2.number_input("End Year", start_y, datetime.now().year, datetime.now().year)

# RELEVANCE THRESHOLD SLIDER
rel_thresh = st.sidebar.slider("Relevance Threshold (%)", 0, 100, 30, help="Only show papers above this SciBERT score")

output_fmt = st.sidebar.multiselect("Output", ["SQLite", "CSV", "JSON"], default=["SQLite"])

search_btn = st.sidebar.button("Search arXiv")
reset_btn = st.sidebar.button("Reset All")

if reset_btn:
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    st.success("Reset complete")
    st.rerun()

log_container = st.empty()
def show_logs():
    if st.session_state.log_buffer:
        log_container.text_area("Logs", "\n".join(st.session_state.log_buffer[-20:]), height=150)

# ====================== MAIN SEARCH ======================
if search_btn:
    if not query.strip() or not cats or start_y > end_y:
        st.error("Fix inputs")
    else:
        st.session_state.processing = True
        with st.spinner("Querying arXiv..."):
            all_papers = query_arxiv(query, cats, max_res, start_y, end_y)
        if not all_papers:
            st.warning("No results")
        else:
            # Filter by threshold
            relevant = [p for p in all_papers if p["relevance_prob"] >= rel_thresh]
            st.success(f"**{len(relevant)}** papers ≥ {rel_thresh}% relevance")

            if relevant:
                progress = st.progress(0)
                status = st.empty()
                downloaded_paths = []

                # Use ThreadPool (no st.session_state inside thread)
                with concurrent.futures.ThreadPoolExecutor(max_workers=4) as exe:
                    futures = {exe.submit(download_and_extract, p): p for p in relevant}
                    for i, future in enumerate(concurrent.futures.as_completed(futures)):
                        paper = future.result()
                        idx = relevant.index(futures[future])
                        relevant[idx] = paper
                        if paper["pdf_path"]:
                            downloaded_paths.append(paper["pdf_path"])
                        progress.progress((i + 1) / len(relevant))
                        status.text(f"Processed {i+1}/{len(relevant)}: {paper['title'][:60]}...")

                progress.empty()
                status.empty()

                # Save results
                st.session_state.relevant_papers = relevant
                st.session_state.download_files["pdf_paths"] = downloaded_paths
                zip_path = create_zip(downloaded_paths)
                st.session_state.download_files["zip_path"] = zip_path

                df = pd.DataFrame(relevant)
                st.subheader("Results")
                st.dataframe(df[["id", "title", "year", "relevance_prob", "download_status"]], use_container_width=True)

                # Output
                if "SQLite" in output_fmt:
                    st.info(save_to_sqlite(df.drop(columns=["content"], errors="ignore")))
                if "CSV" in output_fmt:
                    st.download_button("CSV", df.to_csv(index=False), "piezoelectricity_papers.csv", "text/csv")
                if "JSON" in output_fmt:
                    st.download_button("JSON", df.to_json(orient="records"), "piezoelectricity_papers.json", "application/json")

                # ZIP
                if zip_path and os.path.exists(zip_path):
                    with open(zip_path, "rb") as f:
                        st.download_button("Download All PDFs (ZIP)", f.read(), "piezoelectricity_pdfs.zip", "application/zip")

                # DBs
                for name, path in [("Metadata DB", METADATA_DB_FILE), ("Universe DB", UNIVERSE_DB_FILE)]:
                    if os.path.exists(path):
                        with open(path, "rb") as f:
                            st.download_button(f"Download {name}", f.read(), os.path.basename(path), "application/octet-stream")
        st.session_state.processing = False
        show_logs()

# Show logs always
show_logs()
