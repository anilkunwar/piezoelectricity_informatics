# --------------------------------------------------------------
#  Piezoelectricity in PVDF – FINAL FIXED VERSION
#  ✅ No SQLite cross-DB FK
#  ✅ No nested expanders
#  ✅ No NameError on 'papers'
#  ✅ Full dopant/beta-phase Numba scoring
# --------------------------------------------------------------
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
import random
from pathlib import Path
import zipfile
import io
import gc
import psutil
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import concurrent.futures
import tempfile
import hashlib
import json
from typing import List, Dict, Any, Optional, Tuple
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from tenacity import retry, stop_after_attempt, wait_fixed, wait_exponential
from numba import njit

# ========================= ENVIRONMENT DETECTION =========================
def is_streamlit_cloud():
    """Detect if running on Streamlit Cloud."""
    return (
        os.getenv("HOME") == "/home/appuser" or
        "streamlitapp.com" in os.getenv("HOSTNAME", "") or
        os.getenv("IS_STREAMLIT_CLOUD", "false").lower() == "true"
    )

IS_CLOUD = is_streamlit_cloud()

# ========================= MUST BE FIRST =========================
if "page_config_set" not in st.session_state:
    st.set_page_config(page_title="Piezoelectricity in PVDF", layout="wide")
    st.session_state.page_config_set = True

# =================================================================
# -------------------------- TERM DEFINITIONS --------------------------
DOPANT_TERMS = {
    'sno2', 'tin oxide', 'zno', 'tio2', 'graphene', 'cnt', 'carbon nanotube',
    'batio3', 'bto', 'nio', 'fe3o4', 'al2o3', 'sio2', 'clay', 'montmorillonite',
    'tio', 'cao', 'mgo', 'pzt', 'nanoparticle', 'nanofiller', 'dopant', 'filler'
}

BETA_PHASE_TERMS = {
    'beta phase', 'β-phase', 'beta-phase', '1270 cm⁻¹', '1275 cm-1',
    '840 cm⁻¹', '840 cm-1', 'ftir beta', 'fraction beta', 'beta content',
    'beta-phase content', 'phase fraction', 'beta polymorph', 'pvdf beta'
}

PVDF_TERMS = {'pvdf', 'polyvinylidene fluoride'}

# -------------------------- CONFIG --------------------------
if IS_CLOUD:
    DB_DIR = "/tmp"
    st.info("🌐 Running on Streamlit Cloud: Using temporary storage")
else:
    DB_DIR = os.path.join(os.path.expanduser("~"), "Desktop", "piezoelectricity_data")
    os.makedirs(DB_DIR, exist_ok=True)

METADATA_DB = os.path.join(DB_DIR, "piezoelectricity_metadata.db")
UNIVERSE_DB = os.path.join(DB_DIR, "piezoelectricity_universe.db")
PDF_STORAGE_DB = os.path.join(DB_DIR, "piezoelectricity_pdfs.db")

TEMP_DIR = os.path.join(DB_DIR, "temp")
os.makedirs(TEMP_DIR, exist_ok=True)

log_file = os.path.join(DB_DIR, "piezoelectricity_query.log")
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# -------------------------- SESSION STATE --------------------------
DEFAULT_STATE = {
    "log_buffer": [],
    "processing": False,
    "search_results": None,
    "relevant_papers": None,
    "downloaded_pdfs": {},
    "zip_buffer": None,
    "processing_time": 0.0,
    "db_stats": {},
    "search_session_id": None,
    "temp_files": [],
}

for k, v in DEFAULT_STATE.items():
    if k not in st.session_state:
        st.session_state[k] = v

def update_log(message: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = f"[{timestamp}] {message}"
    st.session_state.log_buffer.append(entry)
    if len(st.session_state.log_buffer) > 50:
        st.session_state.log_buffer.pop(0)
    logging.info(message)

# -------------------------- NUMBA SCORER --------------------------
@njit
def compute_relevance_score_numba(
    dopant_hits: np.ndarray,
    beta_hits: np.ndarray,
    pvdf_hits: np.ndarray,
    weights: np.ndarray
) -> float:
    score = (
        weights[0] * np.sum(dopant_hits) +
        weights[1] * np.sum(beta_hits) +
        weights[2] * np.sum(pvdf_hits)
    )
    max_possible = weights.sum() * 3.0
    return min(100.0, (score / (max_possible + 1e-8)) * 100.0)

def analyze_dopant_beta_relevance(text: str) -> Dict[str, Any]:
    text_lower = text.lower()
    has_dopant = any(term in text_lower for term in DOPANT_TERMS)
    has_beta = any(term in text_lower for term in BETA_PHASE_TERMS)
    has_pvdf = any(term in text_lower for term in PVDF_TERMS)
    dopant_arr = np.array([1 if has_dopant else 0], dtype=np.int8)
    beta_arr = np.array([1 if has_beta else 0], dtype=np.int8)
    pvdf_arr = np.array([1 if has_pvdf else 0], dtype=np.int8)
    weights = np.array([0.5, 0.4, 0.1], dtype=np.float32)
    score = compute_relevance_score_numba(dopant_arr, beta_arr, pvdf_arr, weights)
    return {
        "dopant_present": bool(has_dopant),
        "beta_phase_present": bool(has_beta),
        "pvdf_present": bool(has_pvdf),
        "enhanced_relevance_score": float(score)
    }

# -------------------------- DATABASE MANAGER --------------------------
class DatabaseManager:
    def __init__(self):
        self.metadata_db = METADATA_DB
        self.universe_db = UNIVERSE_DB
        self.pdf_db = PDF_STORAGE_DB
        self.init_databases()
        update_log("Database manager initialized")

    def init_databases(self):
        # Metadata DB
        conn = sqlite3.connect(self.metadata_db)
        c = conn.cursor()
        c.execute("""CREATE TABLE IF NOT EXISTS papers (
            id TEXT PRIMARY KEY,
            arxiv_id TEXT UNIQUE,
            title TEXT NOT NULL,
            authors TEXT,
            year INTEGER,
            categories TEXT,
            abstract TEXT,
            pdf_url TEXT,
            published_date TEXT,
            updated_date TEXT,
            doi TEXT,
            relevance_score REAL,
            matched_terms TEXT,
            download_status TEXT,
            pdf_stored BOOLEAN DEFAULT 0,
            fulltext_stored BOOLEAN DEFAULT 0,
            pdf_size INTEGER,
            download_time TIMESTAMP,
            enhanced_relevance_score REAL,
            dopant_present BOOLEAN,
            beta_phase_present BOOLEAN,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )""")
        c.execute("""CREATE TABLE IF NOT EXISTS search_sessions (
            session_id TEXT PRIMARY KEY,
            query TEXT,
            categories TEXT,
            start_year INTEGER,
            end_year INTEGER,
            max_results INTEGER,
            threshold REAL,
            total_found INTEGER,
            relevant_found INTEGER,
            downloaded_count INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )""")
        c.execute("CREATE INDEX IF NOT EXISTS idx_year ON papers(year)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_score ON papers(relevance_score)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_enhanced_score ON papers(enhanced_relevance_score)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_status ON papers(download_status)")
        conn.commit()
        conn.close()

        # Universe DB — NO cross-DB FK
        conn = sqlite3.connect(self.universe_db)
        c = conn.cursor()
        c.execute("""CREATE TABLE IF NOT EXISTS papers_fulltext (
            paper_id TEXT PRIMARY KEY,
            title TEXT,
            abstract TEXT,
            full_text TEXT,
            text_hash TEXT UNIQUE,
            word_count INTEGER,
            page_count INTEGER,
            extraction_status TEXT,
            extracted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )""")
        c.execute("""CREATE TABLE IF NOT EXISTS extracted_entities (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            paper_id TEXT,
            entity_type TEXT,
            entity_text TEXT,
            context TEXT,
            page_number INTEGER,
            confidence REAL,
            extracted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (paper_id) REFERENCES papers_fulltext(paper_id)
        )""")
        c.execute("""CREATE VIRTUAL TABLE IF NOT EXISTS papers_fts
            USING fts5(paper_id, title, abstract, full_text, tokenize='porter')""")
        conn.commit()
        conn.close()

        # PDF DB — NO cross-DB FK
        conn = sqlite3.connect(self.pdf_db)
        c = conn.cursor()
        c.execute("""CREATE TABLE IF NOT EXISTS pdf_storage (
            paper_id TEXT PRIMARY KEY,
            pdf_data BLOB NOT NULL,
            pdf_hash TEXT UNIQUE,
            original_url TEXT,
            file_size INTEGER,
            page_count INTEGER,
            compression_method TEXT DEFAULT 'none',
            stored_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )""")
        c.execute("""CREATE TABLE IF NOT EXISTS pdf_chunks (
            chunk_id INTEGER PRIMARY KEY AUTOINCREMENT,
            paper_id TEXT,
            chunk_index INTEGER,
            chunk_data BLOB,
            chunk_hash TEXT,
            stored_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (paper_id) REFERENCES pdf_storage(paper_id),
            UNIQUE(paper_id, chunk_index)
        )""")
        c.execute("CREATE INDEX IF NOT EXISTS idx_pdf_hash ON pdf_storage(pdf_hash)")
        conn.commit()
        conn.close()

    def get_db_stats(self) -> Dict[str, Any]:
        stats = {}
        try:
            conn = sqlite3.connect(self.metadata_db)
            c = conn.cursor()
            c.execute("SELECT COUNT(*) FROM papers")
            stats['total_papers'] = c.fetchone()[0]
            c.execute("SELECT COUNT(*) FROM papers WHERE pdf_stored = 1")
            stats['pdfs_stored'] = c.fetchone()[0]
            c.execute("SELECT COUNT(*) FROM papers WHERE fulltext_stored = 1")
            stats['fulltext_stored'] = c.fetchone()[0]
            c.execute("SELECT COUNT(DISTINCT year) FROM papers")
            stats['years_covered'] = c.fetchone()[0]
            conn.close()

            conn = sqlite3.connect(self.universe_db)
            c = conn.cursor()
            c.execute("SELECT COUNT(*) FROM papers_fulltext")
            stats['fulltext_count'] = c.fetchone()[0]
            c.execute("SELECT SUM(word_count) FROM papers_fulltext")
            stats['total_words'] = c.fetchone()[0] or 0
            conn.close()

            conn = sqlite3.connect(self.pdf_db)
            c = conn.cursor()
            c.execute("SELECT COUNT(*) FROM pdf_storage")
            stats['pdf_storage_count'] = c.fetchone()[0]
            c.execute("SELECT SUM(file_size) FROM pdf_storage")
            total_bytes = c.fetchone()[0] or 0
            stats['total_pdf_size_mb'] = round(total_bytes / (1024 * 1024), 2)
            conn.close()
        except Exception as e:
            update_log(f"Error getting DB stats: {e}")
        return stats

    def store_paper_metadata(self, paper: Dict[str, Any]) -> bool:
        try:
            conn = sqlite3.connect(self.metadata_db)
            c = conn.cursor()
            c.execute("""INSERT OR REPLACE INTO papers
                (id, arxiv_id, title, authors, year, categories, abstract,
                pdf_url, published_date, updated_date, doi, relevance_score,
                matched_terms, download_status, pdf_stored, fulltext_stored,
                pdf_size, download_time, enhanced_relevance_score,
                dopant_present, beta_phase_present)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    paper.get('id'), paper.get('arxiv_id'), paper.get('title'),
                    paper.get('authors'), paper.get('year'), paper.get('categories'),
                    paper.get('abstract'), paper.get('pdf_url'), paper.get('published_date'),
                    paper.get('updated_date'), paper.get('doi'), paper.get('relevance_score'),
                    paper.get('matched_terms'), paper.get('download_status'),
                    paper.get('pdf_stored', 0), paper.get('fulltext_stored', 0),
                    paper.get('pdf_size', 0), paper.get('download_time'),
                    paper.get('enhanced_relevance_score'),
                    paper.get('dopant_present'),
                    paper.get('beta_phase_present')
                ))
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            update_log(f"Failed to store metadata for {paper.get('id')}: {e}")
            return False

    def store_pdf_data(self, paper_id: str, pdf_bytes: bytes, pdf_url: str) -> bool:
        try:
            pdf_hash = hashlib.sha256(pdf_bytes).hexdigest()
            file_size = len(pdf_bytes)
            try:
                doc = fitz.open(stream=pdf_bytes, filetype="pdf")
                page_count = len(doc)
                doc.close()
            except:
                page_count = 0

            conn = sqlite3.connect(self.pdf_db)
            c = conn.cursor()
            c.execute("SELECT paper_id FROM pdf_storage WHERE pdf_hash = ?", (pdf_hash,))
            existing = c.fetchone()
            if existing:
                update_log(f"PDF already exists for {paper_id}")
                c.execute("UPDATE papers SET pdf_stored = 1, pdf_size = ? WHERE id = ?", (file_size, paper_id))
            else:
                c.execute("""INSERT OR REPLACE INTO pdf_storage
                    (paper_id, pdf_data, pdf_hash, original_url, file_size, page_count)
                    VALUES (?, ?, ?, ?, ?, ?)""",
                    (paper_id, sqlite3.Binary(pdf_bytes), pdf_hash, pdf_url, file_size, page_count))
                c.execute("UPDATE papers SET pdf_stored = 1, pdf_size = ? WHERE id = ?", (file_size, paper_id))
            conn.commit()
            conn.close()
            update_log(f"Stored PDF for {paper_id} ({file_size/1024:.1f} KB, {page_count} pages)")
            return True
        except Exception as e:
            update_log(f"Failed to store PDF for {paper_id}: {e}")
            return False

    def store_fulltext(self, paper_id: str, title: str, abstract: str,
                       full_text: str, page_count: int = 0) -> bool:
        try:
            text_hash = hashlib.md5(full_text.encode()).hexdigest()
            word_count = len(full_text.split())
            analysis = analyze_dopant_beta_relevance(full_text)
            enhanced_score = analysis["enhanced_relevance_score"]

            conn = sqlite3.connect(self.universe_db)
            c = conn.cursor()
            c.execute("""INSERT OR REPLACE INTO papers_fulltext
                (paper_id, title, abstract, full_text, text_hash, word_count, page_count)
                VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (paper_id, title, abstract, full_text, text_hash, word_count, page_count))
            c.execute("""INSERT OR REPLACE INTO papers_fts
                (paper_id, title, abstract, full_text)
                VALUES (?, ?, ?, ?)""",
                (paper_id, title, abstract, full_text))
            conn.commit()
            conn.close()

            conn = sqlite3.connect(self.metadata_db)
            c = conn.cursor()
            c.execute("""UPDATE papers SET fulltext_stored = 1,
                enhanced_relevance_score = ?,
                dopant_present = ?,
                beta_phase_present = ?
                WHERE id = ?""",
                (enhanced_score, analysis["dopant_present"],
                 analysis["beta_phase_present"], paper_id))
            conn.commit()
            conn.close()
            update_log(f"Stored full text for {paper_id} ({word_count} words, score: {enhanced_score:.1f})")
            return True
        except Exception as e:
            update_log(f"Failed to store full text for {paper_id}: {e}")
            return False

    def get_pdf(self, paper_id: str) -> Optional[bytes]:
        try:
            conn = sqlite3.connect(self.pdf_db)
            c = conn.cursor()
            c.execute("SELECT pdf_data FROM pdf_storage WHERE paper_id = ?", (paper_id,))
            result = c.fetchone()
            conn.close()
            return result[0] if result else None
        except Exception as e:
            update_log(f"Failed to retrieve PDF for {paper_id}: {e}")
            return None

    def get_paper_info(self, paper_id: str) -> Optional[Dict[str, Any]]:
        try:
            conn = sqlite3.connect(self.metadata_db)
            c = conn.cursor()
            c.execute("""SELECT title, authors, year, abstract, pdf_url,
                relevance_score, enhanced_relevance_score, dopant_present, beta_phase_present,
                download_status, pdf_stored, fulltext_stored
                FROM papers WHERE id = ?""", (paper_id,))
            meta = c.fetchone()
            conn.close()
            if not meta:
                return None
            pdf_bytes = self.get_pdf(paper_id)
            conn = sqlite3.connect(self.universe_db)
            c = conn.cursor()
            c.execute("SELECT word_count FROM papers_fulltext WHERE paper_id = ?", (paper_id,))
            fulltext_info = c.fetchone()
            conn.close()
            return {
                'title': meta[0],
                'authors': meta[1],
                'year': meta[2],
                'abstract': meta[3],
                'pdf_url': meta[4],
                'relevance_score': meta[5],
                'enhanced_relevance_score': meta[6],
                'dopant_present': meta[7],
                'beta_phase_present': meta[8],
                'download_status': meta[9],
                'has_pdf': meta[10],
                'has_fulltext': meta[11],
                'pdf_bytes': pdf_bytes,
                'word_count': fulltext_info[0] if fulltext_info else 0
            }
        except Exception as e:
            update_log(f"Failed to get paper info for {paper_id}: {e}")
            return None

    def create_zip_from_db(self, paper_ids: List[str]) -> io.BytesIO:
        zip_buffer = io.BytesIO()
        try:
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                for paper_id in paper_ids:
                    pdf_data = self.get_pdf(paper_id)
                    if pdf_data:
                        info = self.get_paper_info(paper_id)
                        if info:
                            title = re.sub(r'[^\w\s-]', '', info['title'])[:100]
                            authors = info['authors'].split(',')[0][:50] if info['authors'] else 'unknown'
                            filename = f"{paper_id}_{authors}_{info['year']}_{title}.pdf"
                            filename = re.sub(r'\s+', '_', filename)
                        else:
                            filename = f"{paper_id}.pdf"
                        zip_file.writestr(filename, pdf_data)
            zip_buffer.seek(0)
            update_log(f"Created ZIP with {len(paper_ids)} PDFs")
        except Exception as e:
            update_log(f"Failed to create ZIP: {e}")
        return zip_buffer

    def export_metadata(self, format: str = "csv") -> io.BytesIO:
        try:
            conn = sqlite3.connect(self.metadata_db)
            if format.lower() == "csv":
                df = pd.read_sql_query("SELECT * FROM papers", conn)
                output = io.BytesIO()
                df.to_csv(output, index=False)
                output.seek(0)
            elif format.lower() == "json":
                df = pd.read_sql_query("SELECT * FROM papers", conn)
                output = io.BytesIO()
                df.to_json(output, orient="records", indent=2)
                output.seek(0)
            elif format.lower() == "excel":
                df = pd.read_sql_query("SELECT * FROM papers", conn)
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    df.to_excel(writer, index=False, sheet_name='Papers')
                output.seek(0)
            else:
                output = io.BytesIO()
            conn.close()
            return output
        except Exception as e:
            update_log(f"Export failed: {e}")
            return io.BytesIO()

# Initialize DB
db_manager = DatabaseManager()

# -------------------------- DOWNLOAD & QUERY FUNCTIONS --------------------------
def download_pdf_bytes(pdf_url: str) -> Optional[bytes]:
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'application/pdf,text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate',
        'DNT': '1',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1'
    }
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def _download():
        session = requests.Session()
        session.mount('https://', HTTPAdapter(max_retries=3))
        response = session.get(pdf_url, headers=headers, timeout=30)
        response.raise_for_status()
        return response.content
    try:
        pdf_bytes = _download()
        if len(pdf_bytes) < 1024:
            raise ValueError("PDF file too small")
        return pdf_bytes
    except Exception as e:
        update_log(f"Download failed for {pdf_url}: {e}")
        return None

def extract_text_from_bytes(pdf_bytes: bytes) -> str:
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        text = ""
        for page_num in range(min(50, len(doc))):
            text += doc[page_num].get_text()
        doc.close()
        text = re.sub(r'\s+', ' ', text).strip()
        return text[:1000000]
    except Exception as e:
        return f"Error extracting text: {str(e)}"

def handle_paper_download(paper: Dict[str, Any], manual_download: bool = False) -> Dict[str, Any]:
    paper_id = paper['id']
    if manual_download and paper_id in st.session_state.downloaded_pdfs:
        update_log(f"PDF for {paper_id} already in session")
        return paper
    try:
        update_log(f"Downloading PDF for {paper_id}...")
        pdf_bytes = download_pdf_bytes(paper['pdf_url'])
        if pdf_bytes is None:
            paper['download_status'] = "Failed to download"
            return paper
        full_text = extract_text_from_bytes(pdf_bytes)
        pdf_stored = db_manager.store_pdf_data(paper_id, pdf_bytes, paper['pdf_url'])
        if not full_text.startswith("Error"):
            text_stored = db_manager.store_fulltext(paper_id, paper['title'], paper['abstract'], full_text)
            info = db_manager.get_paper_info(paper_id)
            if info:
                paper['enhanced_relevance_score'] = info['enhanced_relevance_score']
                paper['dopant_present'] = info['dopant_present']
                paper['beta_phase_present'] = info['beta_phase_present']
        else:
            text_stored = False
        paper['pdf_stored'] = 1 if pdf_stored else 0
        paper['fulltext_stored'] = 1 if text_stored else 0
        paper['pdf_size'] = len(pdf_bytes)
        paper['download_time'] = datetime.now().isoformat()
        paper['download_status'] = "Successfully downloaded and stored"
        st.session_state.downloaded_pdfs[paper_id] = {
            'pdf_bytes': pdf_bytes,
            'title': paper['title'],
            'authors': paper['authors'],
            'year': paper['year']
        }
        db_manager.store_paper_metadata(paper)
        update_log(f"✅ Successfully processed {paper_id}")
    except Exception as e:
        paper['download_status'] = f"Failed: {str(e)[:100]}"
        update_log(f"❌ Failed to process {paper_id}: {e}")
    return paper

@st.cache_data(ttl=3600)
def query_arxiv(query: str, categories: List[str], max_results: int,
                start_year: int, end_year: int) -> List[Dict[str, Any]]:
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=min(max_results * 2, 500),
        sort_by=arxiv.SortCriterion.Relevance,
        sort_order=arxiv.SortOrder.Descending
    )
    results = []
    query_terms = {term.strip('"').lower() for term in query.split('OR')}
    for result in client.results(search):
        if not (any(cat in result.categories for cat in categories) and
                start_year <= result.published.year <= end_year):
            continue
        abstract_lower = result.summary.lower()
        title_lower = result.title.lower()
        matched_terms = [term for term in query_terms if term in abstract_lower or term in title_lower]
        if not matched_terms:
            continue
        relevance_score = min(len(matched_terms) / len(query_terms) * 100, 100)
        paper = {
            'id': result.entry_id.split('/')[-1],
            'arxiv_id': result.entry_id,
            'title': result.title,
            'authors': ', '.join(a.name for a in result.authors),
            'year': result.published.year,
            'categories': ', '.join(result.categories),
            'abstract': result.summary,
            'pdf_url': result.pdf_url,
            'published_date': result.published.isoformat(),
            'updated_date': result.updated.isoformat() if result.updated else result.published.isoformat(),
            'doi': result.doi if hasattr(result, 'doi') and result.doi else None,
            'relevance_score': round(relevance_score, 2),
            'matched_terms': ', '.join(matched_terms),
            'download_status': 'Pending',
            'pdf_stored': 0,
            'fulltext_stored': 0,
            'pdf_size': 0,
            'download_time': None,
            'enhanced_relevance_score': 0.0,
            'dopant_present': False,
            'beta_phase_present': False
        }
        results.append(paper)
        if len(results) >= max_results:
            break
    results.sort(key=lambda x: x['relevance_score'], reverse=True)
    return results[:max_results]

# -------------------------- UI --------------------------
def show_logs():
    if st.session_state.log_buffer:
        with st.expander("📋 Processing Logs", expanded=False):
            st.text_area("Logs", "\n".join(st.session_state.log_buffer[-20:]), height=150, key="log_display")

def create_dashboard():
    stats = db_manager.get_db_stats()
    st.subheader("📊 Database Statistics")
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("Total Papers", stats.get('total_papers', 0))
    with col2: st.metric("PDFs Stored", stats.get('pdfs_stored', 0))
    with col3: st.metric("Full Text Papers", stats.get('fulltext_stored', 0))
    with col4: st.metric("Total Size", f"{stats.get('total_pdf_size_mb', 0):.1f} MB")
    if stats.get('total_papers', 0) > 0:
        pdf_coverage = (stats.get('pdfs_stored', 0) / stats.get('total_papers', 1)) * 100
        st.progress(pdf_coverage / 100, text=f"PDF Coverage: {pdf_coverage:.1f}%")

# -------------------------- MAIN APP --------------------------
st.title("🔬 Piezoelectricity in PVDF Research Tool")
st.markdown("""
**Advanced tool for searching, downloading, and analyzing piezoelectricity research in PVDF materials.**
Features:
- **Smart Search**: Query arXiv with relevance scoring
- **PDF Storage**: Store PDFs in SQLite databases with deduplication
- **Full-Text Extraction**: Extract and search paper content
- **Dopant & Beta-Phase Focus**: Enhanced relevance using Numba-optimized scoring
- **Multiple Export Formats**: Download papers individually or in bulk
- **Database Management**: View statistics and manage stored papers
""")
if IS_CLOUD:
    st.warning("""
⚠️ **Running on Streamlit Cloud**:
- PDF downloads are manual (click individual buttons)
- Use 'Download All' button for bulk downloads
- Data is stored temporarily (may be cleared between sessions)
""")

show_logs()

# Sidebar
with st.sidebar:
    st.header("🔍 Search Configuration")
    default_query = ' OR '.join([
        '"piezoelectricity"', '"PVDF"', '"beta phase"', '"electrospun nanofibers"',
        '"SnO2"', '"dopants"', '"efficiency"', '"nanogenerators"'
    ])
    query = st.text_area("Search Query", value=default_query, height=100)
    default_cats = ["cond-mat.mtrl-sci", "physics.app-ph", "physics.chem-ph"]
    categories = st.multiselect("Categories", default_cats, default=default_cats[:2])
    current_year = datetime.now().year
    col1, col2 = st.columns(2)
    with col1: start_year = st.number_input("Start Year", 1990, current_year, 2010)
    with col2: end_year = st.number_input("End Year", start_year, current_year, current_year)
    max_results = st.slider("Maximum Results", 1, 100, 20)
    relevance_threshold = st.slider("Relevance Threshold (%)", 0, 100, 30)

    st.subheader("💾 Storage Options")
    auto_download = st.checkbox("Auto-download PDFs", value=not IS_CLOUD, disabled=IS_CLOUD)

    st.subheader("📤 Export Options")
    export_formats = st.multiselect(
        "Select export formats",
        ["ZIP Archive", "CSV", "JSON", "Excel", "Database Backup"],
        default=["ZIP Archive", "CSV"]
    )

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        search_btn = st.button("🔍 Search arXiv", type="primary", use_container_width=True)
    with col_btn2:
        if st.button("🔄 Reset Session", use_container_width=True):
            for key in list(st.session_state.keys()):
                if key not in ["page_config_set", "log_buffer"]:
                    st.session_state[key] = DEFAULT_STATE[key]
            st.rerun()

    st.subheader("🔎 Search Database")
    db_query = st.text_input("Search stored papers", placeholder="e.g., d33 coefficient")
    if st.button("Search in Database", use_container_width=True):
        if db_query:
            conn = sqlite3.connect(db_manager.universe_db)
            c = conn.cursor()
            c.execute("""SELECT paper_id, title, snippet(papers_fts, 2, '<b>', '</b>', '...', 30)
                         FROM papers_fts WHERE papers_fts MATCH ? LIMIT 10""", (db_query,))
            results = c.fetchall()
            conn.close()
            if results:
                st.success(f"Found {len(results)} papers")
                for paper_id, title, snippet in results:
                    with st.expander(f"{title[:80]}..."):
                        st.write(f"**ID:** {paper_id}")
                        st.write(f"**Snippet:** {snippet}")
            else:
                st.warning("No results found")

create_dashboard()

# -------------------------- SEARCH & PROCESSING --------------------------
if search_btn:
    if not query.strip():
        st.error("Please enter a search query")
        st.stop()
    if not categories:
        st.error("Please select at least one category")
        st.stop()
    st.session_state.processing = True
    start_time = time.time()
    with st.spinner("🔍 Searching arXiv..."):
        papers = query_arxiv(query, categories, max_results, start_year, end_year)
    if not papers:
        st.warning("No papers found matching your criteria")
        st.session_state.processing = False
        st.stop()
    relevant_papers = [p for p in papers if p['relevance_score'] >= relevance_threshold]
    if not relevant_papers:
        st.warning(f"No papers above {relevance_threshold}% relevance threshold")
        st.session_state.processing = False
        st.stop()
    st.success(f"Found **{len(relevant_papers)}** relevant papers")
    if auto_download and not IS_CLOUD:
        progress_bar = st.progress(0)
        status_text = st.empty()
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = {executor.submit(handle_paper_download, paper): i for i, paper in enumerate(relevant_papers)}
            completed = 0
            for future in concurrent.futures.as_completed(futures):
                idx = futures[future]
                paper = future.result()
                relevant_papers[idx] = paper
                completed += 1
                progress_bar.progress(completed / len(relevant_papers))
                status_text.text(f"Processed {completed}/{len(relevant_papers)} papers")
        progress_bar.empty()
        status_text.empty()
    st.session_state.relevant_papers = relevant_papers
    st.session_state.processing_time = time.time() - start_time
    update_log(f"Search completed in {st.session_state.processing_time:.1f} seconds")

# -------------------------- DISPLAY & EXPORT (ONLY IF RESULTS EXIST) --------------------------
if st.session_state.get('relevant_papers'):
    papers = st.session_state.relevant_papers
    st.subheader(f"📄 Search Results ({len(papers)} papers)")
    for i, paper in enumerate(papers):
        enhanced = paper.get('enhanced_relevance_score', 0)
        dopant = "🟢" if paper.get('dopant_present') else "⚪"
        beta = "🔵" if paper.get('beta_phase_present') else "⚪"
        with st.expander(f"**{paper['title']}** ({paper['year']}) - Basic: {paper['relevance_score']}% | Enhanced: {enhanced:.1f}% {dopant}{beta}", expanded=i < 2):
            col_info, col_actions = st.columns([3, 1])
            with col_info:
                st.write(f"**Authors:** {paper['authors']}")
                st.write(f"**Categories:** {paper['categories']}")
                st.write(f"**Matched Terms:** {paper['matched_terms']}")
                st.write(f"**Status:** {paper['download_status']}")
                # ✅ FIXED: No nested expander — use toggle or direct markdown
                show_abstract = st.toggle("Show Abstract", key=f"toggle_abstract_{paper['id']}")
                if show_abstract:
                    st.markdown(f"> {paper['abstract']}")

            with col_actions:
                if paper.get('pdf_stored') or paper['id'] in st.session_state.downloaded_pdfs:
                    if paper['id'] in st.session_state.downloaded_pdfs:
                        pdf_bytes = st.session_state.downloaded_pdfs[paper['id']]['pdf_bytes']
                    else:
                        pdf_bytes = db_manager.get_pdf(paper['id'])
                    if pdf_bytes:
                        safe_title = re.sub(r'[^\w\s-]', '', paper['title'])[:50]
                        filename = f"{paper['id']}_{safe_title}.pdf".replace(' ', '_')
                        st.download_button(
                            label="📥 Download PDF",
                            data=pdf_bytes,
                            file_name=filename,
                            mime="application/pdf",
                            key=f"dl_{paper['id']}_{i}",
                            use_container_width=True
                        )
                else:
                    if st.button("⬇️ Download Now", key=f"manual_{paper['id']}", use_container_width=True):
                        with st.spinner("Downloading..."):
                            updated_paper = handle_paper_download(paper, manual_download=True)
                            papers[i] = updated_paper
                            st.session_state.relevant_papers = papers  # update session state
                            st.rerun()
                st.markdown(f"[🌐 arXiv Page]({paper['pdf_url'].replace('/pdf/', '/abs/')})")
                st.markdown(f"[📄 Direct PDF]({paper['pdf_url']})")

    # ✅ EXPORT SECTION — now safely inside if block
    st.subheader("📤 Export Results")
    export_cols = st.columns(5)
    paper_ids = [p['id'] for p in papers if p.get('pdf_stored') or p['id'] in st.session_state.downloaded_pdfs]

    if "ZIP Archive" in export_formats:
        with export_cols[0]:
            if paper_ids:
                if st.button("📦 Create ZIP", use_container_width=True):
                    with st.spinner("Creating ZIP archive..."):
                        zip_buffer = db_manager.create_zip_from_db(paper_ids)
                        st.session_state.zip_buffer = zip_buffer
                        st.success(f"ZIP created with {len(paper_ids)} PDFs")
                if st.session_state.zip_buffer:
                    st.download_button(
                        label="⬇️ Download ZIP",
                        data=st.session_state.zip_buffer.getvalue(),
                        file_name="piezoelectricity_papers.zip",
                        mime="application/zip",
                        use_container_width=True
                    )
    if "CSV" in export_formats:
        with export_cols[1]:
            csv_buffer = db_manager.export_metadata("csv")
            if csv_buffer.getbuffer().nbytes > 0:
                st.download_button(
                    label="📊 CSV Export",
                    data=csv_buffer.getvalue(),
                    file_name="piezoelectricity_metadata.csv",
                    mime="text/csv",
                    use_container_width=True
                )
    if "JSON" in export_formats:
        with export_cols[2]:
            json_buffer = db_manager.export_metadata("json")
            if json_buffer.getbuffer().nbytes > 0:
                st.download_button(
                    label="📄 JSON Export",
                    data=json_buffer.getvalue(),
                    file_name="piezoelectricity_metadata.json",
                    mime="application/json",
                    use_container_width=True
                )
    if "Excel" in export_formats:
        with export_cols[3]:
            excel_buffer = db_manager.export_metadata("excel")
            if excel_buffer.getbuffer().nbytes > 0:
                st.download_button(
                    label="📈 Excel Export",
                    data=excel_buffer.getvalue(),
                    file_name="piezoelectricity_metadata.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
    with export_cols[4]:
        with st.expander("🗃️ Databases"):
            for label, path in [("Metadata DB", db_manager.metadata_db),
                                ("Fulltext DB", db_manager.universe_db),
                                ("PDF Storage DB", db_manager.pdf_db)]:
                if os.path.exists(path):
                    with open(path, 'rb') as f:
                        st.download_button(label=label, data=f.read(), file_name=os.path.basename(path),
                                           mime="application/octet-stream", use_container_width=True)

# -------------------------- DATABASE MANAGEMENT --------------------------
st.divider()
st.subheader("🗄️ Database Management")
col_stats, col_clean = st.columns(2)
with col_stats:
    if st.button("🔄 Refresh Statistics", use_container_width=True):
        st.session_state.db_stats = db_manager.get_db_stats()
        st.rerun()
with col_clean:
    if st.button("🧹 Clean Temporary Files", use_container_width=True):
        temp_dir = os.path.join(DB_DIR, "temp")
        if os.path.exists(temp_dir):
            for file in os.listdir(temp_dir):
                try:
                    os.remove(os.path.join(temp_dir, file))
                except:
                    pass
        st.session_state.temp_files = []
        st.success("Temporary files cleaned")

if st.session_state.db_stats:
    with st.expander("📊 Detailed Statistics"):
        st.json(st.session_state.db_stats)

st.divider()
st.caption(f"""
**Piezoelectricity in PVDF Research Tool** |
Running on {'☁️ Streamlit Cloud' if IS_CLOUD else '💻 Local'} |
Data Directory: `{DB_DIR}` |
Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
""")

show_logs()

import atexit
def cleanup():
    temp_dir = os.path.join(DB_DIR, "temp")
    if os.path.exists(temp_dir):
        for file in os.listdir(temp_dir):
            try:
                os.remove(os.path.join(temp_dir, file))
            except:
                pass
atexit.register(cleanup)
