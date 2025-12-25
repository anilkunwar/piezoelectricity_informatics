# --------------------------------------------------------------
#  Piezoelectricity in PVDF – FINAL, PRODUCTION-READY with SQLite PDF Storage
# --------------------------------------------------------------
import arxiv
import fitz
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
import gc
import psutil
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import concurrent.futures
import io
import hashlib
import json
from typing import List, Dict, Any, Optional, Tuple, BinaryIO

from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from tenacity import retry, stop_after_attempt, wait_fixed

# ========================= Numba JIT =========================
try:
    from numba import jit, njit, prange, objmode
    from numba.typed import List as NumbaList
    import numba
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    st.warning("Numba not installed. Install with: pip install numba")

# ========================= MUST BE FIRST =========================
if "page_config_set" not in st.session_state:
    st.set_page_config(page_title="Piezoelectricity in PVDF", layout="wide")
    st.session_state.page_config_set = True
# =================================================================

# -------------------------- CONFIG --------------------------
DB_DIR = "/tmp" if os.path.exists("/tmp") else os.path.join(os.path.expanduser("~"), "Desktop")
os.makedirs(DB_DIR, exist_ok=True)

# No longer need PDF directory since we're storing in SQLite
METADATA_DB = os.path.join(DB_DIR, "piezoelectricity_metadata.db")
UNIVERSE_DB = os.path.join(DB_DIR, "piezoelectricity_universe.db")
PDF_STORAGE_DB = os.path.join(DB_DIR, "piezoelectricity_pdfs.db")  # Separate DB for PDFs

logging.basicConfig(
    filename=os.path.join(DB_DIR, "piezoelectricity_query.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# -------------------------- SESSION STATE --------------------------
DEFAULT_STATE = {
    "log_buffer": [],
    "processing": False,
    "search_results": None,
    "relevant_papers": None,
    "pdf_paths": [],
    "zip_buffer": None,
    "processing_time": 0.0,
    "speed_metrics": {},
    "db_stats": {"metadata": 0, "universe": 0, "pdfs": 0},
}
for k, v in DEFAULT_STATE.items():
    if k not in st.session_state:
        st.session_state[k] = v

def update_log(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = f"[{ts}] {msg}"
    st.session_state.log_buffer.append(entry)
    if len(st.session_state.log_buffer) > 50:
        st.session_state.log_buffer.pop(0)
    logging.info(msg)

# -------------------------- DATABASE MANAGER --------------------------
class DatabaseManager:
    """Manager for SQLite databases with PDF storage"""
    
    def __init__(self):
        self.metadata_db = METADATA_DB
        self.universe_db = UNIVERSE_DB
        self.pdf_db = PDF_STORAGE_DB
        self.init_databases()
    
    def init_databases(self):
        """Initialize all SQLite databases with proper schema"""
        # Metadata Database (small, fast queries)
        conn = sqlite3.connect(self.metadata_db)
        c = conn.cursor()
        c.execute("""CREATE TABLE IF NOT EXISTS papers (
                     id TEXT PRIMARY KEY,
                     title TEXT,
                     authors TEXT,
                     year INTEGER,
                     categories TEXT,
                     abstract TEXT,
                     pdf_url TEXT,
                     arxiv_id TEXT,
                     published_date TEXT,
                     updated_date TEXT,
                     doi TEXT,
                     matched_terms TEXT,
                     relevance_prob REAL,
                     has_pdf BOOLEAN DEFAULT 0,
                     has_fulltext BOOLEAN DEFAULT 0,
                     pdf_size INTEGER,
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
                     total_papers INTEGER,
                     relevant_papers INTEGER,
                     created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                     )""")
        
        c.execute("""CREATE INDEX IF NOT EXISTS idx_year ON papers(year)""")
        c.execute("""CREATE INDEX IF NOT EXISTS idx_relevance ON papers(relevance_prob)""")
        c.execute("""CREATE INDEX IF NOT EXISTS idx_has_pdf ON papers(has_pdf)""")
        conn.commit()
        conn.close()
        
        # Universe Database (full text storage)
        conn = sqlite3.connect(self.universe_db)
        c = conn.cursor()
        c.execute("""CREATE TABLE IF NOT EXISTS papers_fulltext (
                     paper_id TEXT PRIMARY KEY,
                     title TEXT,
                     authors TEXT,
                     year INTEGER,
                     abstract TEXT,
                     full_text TEXT,
                     text_hash TEXT,
                     word_count INTEGER,
                     section_count INTEGER,
                     extracted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                     FOREIGN KEY (paper_id) REFERENCES metadata.papers(id)
                     )""")
        
        c.execute("""CREATE TABLE IF NOT EXISTS extracted_entities (
                     id INTEGER PRIMARY KEY AUTOINCREMENT,
                     paper_id TEXT,
                     entity_type TEXT,
                     entity_text TEXT,
                     context TEXT,
                     score REAL,
                     FOREIGN KEY (paper_id) REFERENCES papers_fulltext(paper_id)
                     )""")
        
        c.execute("""CREATE VIRTUAL TABLE IF NOT EXISTS papers_fts USING fts5(
                     paper_id, title, abstract, full_text,
                     tokenize="porter"
                     )""")
        
        c.execute("""CREATE INDEX IF NOT EXISTS idx_entities_paper ON extracted_entities(paper_id)""")
        c.execute("""CREATE INDEX IF NOT EXISTS idx_entities_type ON extracted_entities(entity_type)""")
        conn.commit()
        conn.close()
        
        # PDF Storage Database (binary storage with compression)
        conn = sqlite3.connect(self.pdf_db)
        c = conn.cursor()
        c.execute("""CREATE TABLE IF NOT EXISTS pdf_storage (
                     paper_id TEXT PRIMARY KEY,
                     pdf_data BLOB,
                     pdf_hash TEXT UNIQUE,
                     original_filename TEXT,
                     file_size INTEGER,
                     page_count INTEGER,
                     compression_ratio REAL,
                     stored_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                     FOREIGN KEY (paper_id) REFERENCES metadata.papers(id)
                     )""")
        
        c.execute("""CREATE TABLE IF NOT EXISTS pdf_chunks (
                     chunk_id INTEGER PRIMARY KEY AUTOINCREMENT,
                     paper_id TEXT,
                     chunk_index INTEGER,
                     chunk_data BLOB,
                     chunk_hash TEXT,
                     FOREIGN KEY (paper_id) REFERENCES pdf_storage(paper_id),
                     UNIQUE(paper_id, chunk_index)
                     )""")
        
        c.execute("""CREATE INDEX IF NOT EXISTS idx_pdf_hash ON pdf_storage(pdf_hash)""")
        c.execute("""CREATE INDEX IF NOT EXISTS idx_pdf_size ON pdf_storage(file_size)""")
        conn.commit()
        conn.close()
        
        update_log("All databases initialized")
    
    def get_db_stats(self) -> Dict[str, int]:
        """Get statistics for all databases"""
        stats = {}
        
        # Metadata DB stats
        conn = sqlite3.connect(self.metadata_db)
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM papers")
        stats['metadata'] = c.fetchone()[0]
        c.execute("SELECT COUNT(*) FROM papers WHERE has_pdf = 1")
        stats['has_pdf'] = c.fetchone()[0]
        conn.close()
        
        # Universe DB stats
        conn = sqlite3.connect(self.universe_db)
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM papers_fulltext")
        stats['universe'] = c.fetchone()[0]
        c.execute("SELECT SUM(word_count) FROM papers_fulltext")
        stats['total_words'] = c.fetchone()[0] or 0
        conn.close()
        
        # PDF DB stats
        conn = sqlite3.connect(self.pdf_db)
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM pdf_storage")
        stats['pdfs'] = c.fetchone()[0]
        c.execute("SELECT SUM(file_size) FROM pdf_storage")
        total_size = c.fetchone()[0] or 0
        stats['total_pdf_size_mb'] = round(total_size / (1024 * 1024), 2)
        conn.close()
        
        return stats
    
    def store_pdf(self, paper_id: str, pdf_path: str) -> bool:
        """Store PDF file in database with deduplication"""
        try:
            if not os.path.exists(pdf_path):
                return False
            
            with open(pdf_path, 'rb') as f:
                pdf_data = f.read()
            
            # Calculate hash for deduplication
            pdf_hash = hashlib.sha256(pdf_data).hexdigest()
            
            # Check if PDF already exists (deduplication)
            conn = sqlite3.connect(self.pdf_db)
            c = conn.cursor()
            c.execute("SELECT paper_id FROM pdf_storage WHERE pdf_hash = ?", (pdf_hash,))
            existing = c.fetchone()
            
            if existing:
                # PDF already exists, link to this paper
                update_log(f"PDF already exists for {paper_id} (duplicate)")
                c.execute("""UPDATE metadata.papers 
                            SET has_pdf = 1, pdf_size = (SELECT file_size FROM pdf_storage WHERE pdf_hash = ?)
                            WHERE id = ?""", (pdf_hash, paper_id))
                conn.commit()
                conn.close()
                return True
            
            # Get PDF metadata
            try:
                doc = fitz.open(pdf_path)
                page_count = len(doc)
                doc.close()
            except:
                page_count = 0
            
            file_size = len(pdf_data)
            
            # Store in database
            c.execute("""INSERT OR REPLACE INTO pdf_storage 
                         (paper_id, pdf_data, pdf_hash, original_filename, file_size, page_count, compression_ratio)
                         VALUES (?, ?, ?, ?, ?, ?, ?)""",
                     (paper_id, sqlite3.Binary(pdf_data), pdf_hash, 
                      f"{paper_id}.pdf", file_size, page_count, 1.0))
            
            # Update metadata
            c.execute("""UPDATE metadata.papers 
                        SET has_pdf = 1, pdf_size = ?, updated_at = CURRENT_TIMESTAMP
                        WHERE id = ?""", (file_size, paper_id))
            
            conn.commit()
            conn.close()
            
            update_log(f"Stored PDF for {paper_id} ({file_size/1024:.1f} KB, {page_count} pages)")
            return True
            
        except Exception as e:
            update_log(f"Failed to store PDF for {paper_id}: {e}")
            return False
    
    def store_fulltext(self, paper_id: str, title: str, authors: str, year: int, 
                      abstract: str, full_text: str) -> bool:
        """Store full text in universe database"""
        try:
            text_hash = hashlib.md5(full_text.encode()).hexdigest()
            word_count = len(full_text.split())
            section_count = len(full_text.split('\n\n'))
            
            conn = sqlite3.connect(self.universe_db)
            c = conn.cursor()
            
            # Store full text
            c.execute("""INSERT OR REPLACE INTO papers_fulltext 
                         (paper_id, title, authors, year, abstract, full_text, text_hash, word_count, section_count)
                         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                     (paper_id, title, authors, year, abstract, full_text, text_hash, word_count, section_count))
            
            # Update FTS table
            c.execute("""INSERT OR REPLACE INTO papers_fts 
                         (paper_id, title, abstract, full_text)
                         VALUES (?, ?, ?, ?)""",
                     (paper_id, title, abstract, full_text))
            
            # Update metadata
            c.execute("""UPDATE metadata.papers 
                        SET has_fulltext = 1, updated_at = CURRENT_TIMESTAMP
                        WHERE id = ?""", (paper_id,))
            
            conn.commit()
            conn.close()
            
            update_log(f"Stored full text for {paper_id} ({word_count} words)")
            return True
            
        except Exception as e:
            update_log(f"Failed to store full text for {paper_id}: {e}")
            return False
    
    def get_pdf(self, paper_id: str) -> Optional[bytes]:
        """Retrieve PDF from database"""
        try:
            conn = sqlite3.connect(self.pdf_db)
            c = conn.cursor()
            c.execute("SELECT pdf_data FROM pdf_storage WHERE paper_id = ?", (paper_id,))
            result = c.fetchone()
            conn.close()
            
            if result:
                return result[0]
            return None
        except Exception as e:
            update_log(f"Failed to get PDF for {paper_id}: {e}")
            return None
    
    def get_fulltext(self, paper_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve full text from database"""
        try:
            conn = sqlite3.connect(self.universe_db)
            c = conn.cursor()
            c.execute("""SELECT title, authors, year, abstract, full_text, word_count 
                         FROM papers_fulltext WHERE paper_id = ?""", (paper_id,))
            result = c.fetchone()
            conn.close()
            
            if result:
                return {
                    "title": result[0],
                    "authors": result[1],
                    "year": result[2],
                    "abstract": result[3],
                    "full_text": result[4],
                    "word_count": result[5]
                }
            return None
        except Exception as e:
            update_log(f"Failed to get full text for {paper_id}: {e}")
            return None
    
    def search_fulltext(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Search full text using FTS"""
        try:
            conn = sqlite3.connect(self.universe_db)
            c = conn.cursor()
            
            # Search in FTS table
            c.execute("""SELECT paper_id, title, abstract, 
                        snippet(papers_fts, 2, '<b>', '</b>', '...', 30) as snippet
                         FROM papers_fts 
                         WHERE papers_fts MATCH ? 
                         ORDER BY rank 
                         LIMIT ?""", (query, limit))
            
            results = []
            for row in c.fetchall():
                results.append({
                    "paper_id": row[0],
                    "title": row[1],
                    "abstract": row[2],
                    "snippet": row[3]
                })
            
            conn.close()
            return results
        except Exception as e:
            update_log(f"Full text search failed: {e}")
            return []
    
    def create_zip_buffer(self, paper_ids: List[str]) -> io.BytesIO:
        """Create ZIP file in memory from database PDFs"""
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for paper_id in paper_ids:
                pdf_data = self.get_pdf(paper_id)
                if pdf_data:
                    # Get paper info for filename
                    conn = sqlite3.connect(self.metadata_db)
                    c = conn.cursor()
                    c.execute("SELECT title, authors, year FROM papers WHERE id = ?", (paper_id,))
                    paper_info = c.fetchone()
                    conn.close()
                    
                    if paper_info:
                        title, authors, year = paper_info
                        # Create safe filename
                        safe_title = re.sub(r'[^\w\s-]', '', title)[:50]
                        safe_authors = re.sub(r'[^\w\s-]', '', authors.split(',')[0])[:30]
                        filename = f"{paper_id}_{safe_authors}_{year}_{safe_title}.pdf"
                        filename = filename.replace(' ', '_')
                    else:
                        filename = f"{paper_id}.pdf"
                    
                    zip_file.writestr(filename, pdf_data)
        
        zip_buffer.seek(0)
        return zip_buffer
    
    def export_metadata(self, format: str = "csv") -> bytes:
        """Export metadata in various formats"""
        conn = sqlite3.connect(self.metadata_db)
        
        if format.lower() == "csv":
            df = pd.read_sql_query("SELECT * FROM papers", conn)
            output = df.to_csv(index=False).encode()
        elif format.lower() == "json":
            df = pd.read_sql_query("SELECT * FROM papers", conn)
            output = df.to_json(orient="records", indent=2).encode()
        elif format.lower() == "sql":
            # Dump SQL
            output = b""
            for line in conn.iterdump():
                output += line.encode() + b'\n'
        else:
            output = b""
        
        conn.close()
        return output
    
    def backup_databases(self) -> io.BytesIO:
        """Create backup of all databases"""
        backup_buffer = io.BytesIO()
        
        with zipfile.ZipFile(backup_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # Backup metadata DB
            with open(self.metadata_db, 'rb') as f:
                zip_file.writestr("metadata.db", f.read())
            
            # Backup universe DB
            with open(self.universe_db, 'rb') as f:
                zip_file.writestr("universe.db", f.read())
            
            # Backup PDF DB
            with open(self.pdf_db, 'rb') as f:
                zip_file.writestr("pdf_storage.db", f.read())
            
            # Add stats
            stats = self.get_db_stats()
            zip_file.writestr("stats.json", json.dumps(stats, indent=2))
        
        backup_buffer.seek(0)
        return backup_buffer

# Initialize database manager
db_manager = DatabaseManager()

# -------------------------- HEALTH & SPEED METRICS --------------------------
def health_check() -> bool:
    mem = psutil.Process().memory_info().rss / 1024 / 1024
    free_gb = psutil.disk_usage(DB_DIR).free / (1024**3)
    update_log(f"RAM {mem:.1f} MB | Disk free {free_gb:.1f} GB")
    if mem > 1500:
        st.warning(f"High RAM ({mem:.1f} MB)")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if free_gb < 0.5:
        st.error("Low disk space")
        return False
    return True

# -------------------------- HTTP RETRY --------------------------
def retry_session():
    s = requests.Session()
    r = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    s.mount("http://", HTTPAdapter(max_retries=r))
    s.mount("https://", HTTPAdapter(max_retries=r))
    return s

# -------------------------- KEYWORD MATCHER --------------------------
class KeywordMatcher:
    """Optimized keyword matching"""
    
    def __init__(self):
        self.patterns = KEY_PATTERNS
        self.compiled_regex = [re.compile(p, re.IGNORECASE) for p in self.patterns]
    
    def score_abstract(self, abstract: str) -> float:
        """Fast scoring without SciBERT"""
        n = sum(bool(p.search(abstract.lower())) for p in self.compiled_regex)
        return np.sqrt(n) / np.sqrt(len(self.patterns))

keyword_matcher = KeywordMatcher()

# -------------------------- SciBERT --------------------------
@st.cache_resource
def load_scibert():
    tok = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
    mdl = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased")
    mdl.eval()
    
    if torch.cuda.is_available():
        mdl = mdl.to('cuda')
        update_log("SciBERT loaded on GPU")
    else:
        update_log("SciBERT loaded on CPU")
    
    return tok, mdl

scibert_tok, scibert_mdl = load_scibert()

def score_with_scibert(abstract: str) -> float:
    """Score abstract with SciBERT attention"""
    try:
        enc = scibert_tok(abstract, return_tensors="pt", truncation=True, max_length=512, padding=True)
        
        if torch.cuda.is_available():
            enc = {k: v.to('cuda') for k, v in enc.items()}
        
        with torch.no_grad():
            out = scibert_mdl(**enc, output_attentions=True)
        
        # Combine keyword score with attention
        kw_score = keyword_matcher.score_abstract(abstract)
        
        # Get attention boost
        tokens = scibert_tok.convert_ids_to_tokens(enc["input_ids"][0])
        kw_idx = [i for i, t in enumerate(tokens) if any(k in t.lower() for k in ["pvdf","piezo","phase","beta","alpha"])]
        
        if kw_idx:
            att = out.attentions[-1][0,0].cpu().numpy()
            boost = np.mean(att[kw_idx, :])
            if boost > 0.1:
                kw_score = min(kw_score + 0.2 * len(kw_idx)/len(tokens), 1.0)
        
        return kw_score
        
    except Exception as e:
        update_log(f"SciBERT error: {e}")
        return keyword_matcher.score_abstract(abstract)

# -------------------------- PDF PROCESSING --------------------------
def extract_pdf_text(pdf_path: str, max_pages: int = 50) -> str:
    """Extract text from PDF with error handling"""
    try:
        doc = fitz.open(pdf_path)
        texts = []
        
        for i in range(min(len(doc), max_pages)):
            page = doc[i]
            text = page.get_text()
            if text.strip():
                texts.append(text)
        
        doc.close()
        
        if not texts:
            return "No text extracted"
        
        full_text = "\n".join(texts)
        # Clean up text
        full_text = re.sub(r'\s+', ' ', full_text)
        return full_text[:1000000]  # Limit to 1MB
        
    except Exception as e:
        return f"Error extracting text: {str(e)}"

# -------------------------- ARXIV QUERY --------------------------
@st.cache_data(ttl=3600)
def query_arxiv(_query: str, cats: list, max_res: int, sy: int, ey: int):
    """Query arXiv for papers"""
    client = arxiv.Client()
    search = arxiv.Search(
        query=_query,
        max_results=max_res,
        sort_by=arxiv.SortCriterion.Relevance,
        sort_order=arxiv.SortOrder.Descending
    )
    
    out = []
    qwords = {w.strip('"').lower() for w in _query.split("OR")}
    
    for r in client.results(search):
        if not (any(c in r.categories for c in cats) and sy <= r.published.year <= ey):
            continue
        
        matched = [w for w in qwords if w in r.summary.lower() or w in r.title.lower()]
        if not matched:
            continue
        
        rel = score_with_scibert(r.summary)
        
        out.append({
            "id": r.entry_id.split("/")[-1],
            "arxiv_id": r.entry_id,
            "title": r.title,
            "authors": ", ".join(a.name for a in r.authors),
            "year": r.published.year,
            "categories": ", ".join(r.categories),
            "abstract": r.summary,
            "pdf_url": r.pdf_url,
            "published_date": r.published.isoformat(),
            "updated_date": r.updated.isoformat() if r.updated else r.published.isoformat(),
            "doi": r.doi if hasattr(r, 'doi') and r.doi else None,
            "download_status": "Pending",
            "matched_terms": ", ".join(matched),
            "relevance_prob": round(rel * 100, 2),
            "has_pdf": 0,
            "has_fulltext": 0,
            "pdf_size": 0
        })
        
        if len(out) >= max_res:
            break
    
    return sorted(out, key=lambda x: x["relevance_prob"], reverse=True)

# -------------------------- DOWNLOAD AND STORE --------------------------
@retry(stop=stop_after_attempt(4), wait=wait_fixed(2))
def download_and_store(paper: dict, temp_dir: str) -> dict:
    """Download PDF and store in database"""
    pid = paper["id"]
    temp_path = os.path.join(temp_dir, f"{pid}.pdf")
    
    try:
        # Download PDF
        s = retry_session()
        resp = s.get(paper["pdf_url"], timeout=30, headers={"User-Agent": "arXiv-PDF-Downloader/1.0"})
        resp.raise_for_status()
        
        with open(temp_path, "wb") as f:
            f.write(resp.content)
        
        time.sleep(random.uniform(0.3, 0.7))
        
        # Extract text
        full_text = extract_pdf_text(temp_path)
        
        # Store in databases
        # 1. Store PDF in PDF storage DB
        pdf_stored = db_manager.store_pdf(pid, temp_path)
        
        # 2. Store full text in universe DB
        if not full_text.startswith("Error"):
            text_stored = db_manager.store_fulltext(
                pid, paper["title"], paper["authors"], paper["year"],
                paper["abstract"], full_text
            )
        else:
            text_stored = False
        
        # Update paper status
        paper.update({
            "download_status": "✓ Downloaded and stored in DB",
            "has_pdf": 1 if pdf_stored else 0,
            "has_fulltext": 1 if text_stored else 0,
            "pdf_size": os.path.getsize(temp_path) if os.path.exists(temp_path) else 0
        })
        
        update_log(f"Success: {pid} - PDF:{'✓' if pdf_stored else '✗'} Text:{'✓' if text_stored else '✗'}")
        
    except Exception as e:
        paper.update({
            "download_status": f"✗ Failed: {str(e)[:100]}",
            "has_pdf": 0,
            "has_fulltext": 0
        })
        update_log(f"Failed {pid}: {e}")
    
    finally:
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)
    
    return paper

# -------------------------- UI --------------------------
st.title("📚 Piezoelectricity in PVDF – SQLite PDF Storage")
st.markdown("""
**Advanced scientific paper search with SQLite database storage**
- **PDF Storage**: All PDFs stored in SQLite database with deduplication
- **Full-Text Search**: Full papers stored in separate database
- **Multiple Downloads**: Individual PDFs, ZIP archives, database exports
- **Cloud Ready**: All data stored in SQLite files (portable)
""")

# ---------- DATABASE STATS ----------
with st.expander("📊 Database Statistics", expanded=True):
    stats = db_manager.get_db_stats()
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Metadata Papers", stats['metadata'])
    with col2:
        st.metric("PDFs Stored", stats['pdfs'])
    with col3:
        st.metric("Full Text Papers", stats['universe'])
    with col4:
        st.metric("Total PDF Size", f"{stats.get('total_pdf_size_mb', 0)} MB")
    
    if stats['metadata'] > 0:
        st.progress(stats['pdfs'] / stats['metadata'], text=f"PDF Coverage: {stats['pdfs']}/{stats['metadata']}")

# ---------- LOG AREA ----------
log_placeholder = st.empty()
def show_logs():
    if st.session_state.log_buffer:
        with st.expander("📝 Processing Logs", expanded=False):
            log_placeholder.text_area(
                "Logs",
                "\n".join(st.session_state.log_buffer[-20:]),
                height=150,
                key="log_area",
                label_visibility="collapsed"
            )
    else:
        log_placeholder.empty()

show_logs()

# ---------- SIDEBAR ----------
with st.sidebar:
    st.header("🔍 Search Configuration")
    
    # Search options
    q = st.text_input(
        "Query",
        value=' OR '.join(f'"{t}"' for t in [
            "piezoelectricity", "PVDF", "beta phase", "electrospun nanofibers",
            "SnO2", "dopants", "efficiency", "nanogenerators"
        ]),
        help="Use OR for multiple terms, quotes for exact phrases"
    )
    
    cats = st.multiselect(
        "Categories",
        ["cond-mat.mtrl-sci", "physics.app-ph", "cond-mat", "physics"],
        default=["cond-mat.mtrl-sci", "physics.app-ph"]
    )
    
    col1, col2 = st.columns(2)
    max_res = col1.slider("Max Results", 1, 100, 30)
    rel_thr = col2.slider("Threshold %", 0, 100, 30)
    
    col3, col4 = st.columns(2)
    sy = col3.number_input("Start Year", 2000, datetime.now().year, 2015)
    ey = col4.number_input("End Year", sy, datetime.now().year, datetime.now().year)
    
    # Download options
    st.subheader("💾 Storage Options")
    store_pdfs = st.checkbox("Store PDFs in Database", value=True, 
                            help="Store PDF files in SQLite database")
    store_text = st.checkbox("Extract Full Text", value=True,
                            help="Extract and store full text from PDFs")
    
    # Output formats
    st.subheader("📤 Export Options")
    export_formats = st.multiselect(
        "Export Formats",
        ["SQLite", "CSV", "JSON", "ZIP", "Backup"],
        default=["SQLite", "ZIP"]
    )
    
    # Buttons
    col_btn1, col_btn2 = st.columns(2)
    search_btn = col_btn1.button("🔍 Search & Download", type="primary", use_container_width=True)
    reset_btn = col_btn2.button("🔄 Reset", use_container_width=True)
    
    # Full-text search
    st.subheader("🔎 Full-Text Search")
    ft_query = st.text_input("Search in stored papers", placeholder="e.g., piezoelectric coefficient d33")
    if st.button("Search Database", key="ft_search"):
        if ft_query:
            with st.spinner("Searching..."):
                results = db_manager.search_fulltext(ft_query)
                if results:
                    st.success(f"Found {len(results)} papers")
                    for r in results:
                        with st.expander(f"{r['title'][:80]}..."):
                            st.write(f"**Snippet:** {r['snippet']}")
                else:
                    st.warning("No results found")

if reset_btn:
    for k in list(st.session_state.keys()):
        if k not in ["page_config_set", "log_buffer"]:
            st.session_state[k] = DEFAULT_STATE[k]
    st.rerun()

# -------------------------- MAIN PROCESSING --------------------------
if search_btn:
    if not q.strip() or not cats:
        st.error("Please enter a query and select categories")
        st.stop()
    
    if not health_check():
        st.stop()
    
    st.session_state.processing = True
    start_time = time.time()
    
    # Create temp directory for downloads
    temp_dir = os.path.join(DB_DIR, "temp_pdfs")
    os.makedirs(temp_dir, exist_ok=True)
    
    with st.spinner("🔍 Searching arXiv..."):
        all_papers = query_arxiv(q, cats, max_res, sy, ey)
    
    if not all_papers:
        st.warning("No papers found matching your criteria")
        st.session_state.processing = False
        st.stop()
    
    # Filter by relevance
    relevant = [p for p in all_papers if p["relevance_prob"] >= rel_thr]
    
    if not relevant:
        st.warning(f"No papers above {rel_thr}% relevance threshold")
        st.session_state.processing = False
        st.stop()
    
    st.success(f"Found **{len(relevant)}** relevant papers")
    
    # Download and store papers
    if relevant and (store_pdfs or store_text):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for paper in relevant:
                if store_pdfs:
                    futures.append(executor.submit(download_and_store, paper, temp_dir))
                else:
                    # Just store metadata if PDF storage is disabled
                    futures.append(executor.submit(lambda p: p, paper))
            
            completed = 0
            for future in concurrent.futures.as_completed(futures):
                completed += 1
                progress_bar.progress(completed / len(futures))
                status_text.text(f"Downloaded {completed}/{len(futures)} papers")
        
        progress_bar.empty()
        status_text.empty()
    
    # Store metadata in database
    if relevant:
        conn = sqlite3.connect(db_manager.metadata_db)
        for paper in relevant:
            # Convert to tuple for insertion
            paper_tuple = (
                paper["id"], paper["title"], paper["authors"], paper["year"],
                paper["categories"], paper["abstract"], paper["pdf_url"],
                paper["arxiv_id"], paper["published_date"], paper["updated_date"],
                paper["doi"], paper["matched_terms"], paper["relevance_prob"],
                paper.get("has_pdf", 0), paper.get("has_fulltext", 0),
                paper.get("pdf_size", 0)
            )
            
            conn.execute("""INSERT OR REPLACE INTO papers 
                          (id, title, authors, year, categories, abstract, pdf_url,
                           arxiv_id, published_date, updated_date, doi, matched_terms,
                           relevance_prob, has_pdf, has_fulltext, pdf_size)
                          VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""", 
                        paper_tuple)
        
        conn.commit()
        conn.close()
        
        update_log(f"Metadata stored for {len(relevant)} papers")
    
    # Create ZIP buffer if requested
    if "ZIP" in export_formats and store_pdfs:
        paper_ids = [p["id"] for p in relevant if p.get("has_pdf")]
        if paper_ids:
            with st.spinner("Creating ZIP archive..."):
                st.session_state.zip_buffer = db_manager.create_zip_buffer(paper_ids)
                update_log(f"Created ZIP with {len(paper_ids)} PDFs")
    
    # Update session state
    st.session_state.relevant_papers = relevant
    st.session_state.processing_time = time.time() - start_time
    st.session_state.processing = False
    
    # Clean up temp directory
    import shutil
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    
    # Show completion message
    st.balloons()
    st.success(f"✅ Processing complete in {st.session_state.processing_time:.1f} seconds")

# -------------------------- RESULTS DISPLAY --------------------------
if st.session_state.relevant_papers:
    papers = st.session_state.relevant_papers
    df = pd.DataFrame(papers)
    
    st.subheader(f"📄 Results ({len(papers)} papers)")
    
    # Display papers in an interactive table
    for i, paper in enumerate(papers):
        with st.expander(f"📑 {paper['title'][:100]}...", expanded=i < 3):
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                st.write(f"**Authors:** {paper['authors']}")
                st.write(f"**Year:** {paper['year']} | **Categories:** {paper['categories']}")
                st.write(f"**Relevance:** {paper['relevance_prob']}%")
                st.write(f"**Status:** {paper['download_status']}")
                
                if paper.get('has_pdf'):
                    st.success("✓ PDF stored in database")
                if paper.get('has_fulltext'):
                    st.success("✓ Full text extracted")
            
            with col2:
                # Individual PDF download
                if paper.get('has_pdf'):
                    pdf_data = db_manager.get_pdf(paper['id'])
                    if pdf_data:
                        st.download_button(
                            label="📥 Download PDF",
                            data=pdf_data,
                            file_name=f"{paper['id']}.pdf",
                            mime="application/pdf",
                            key=f"pdf_{paper['id']}_{i}",
                            use_container_width=True
                        )
            
            with col3:
                # Full text view
                if paper.get('has_fulltext'):
                    if st.button("📖 View Full Text", key=f"view_{paper['id']}"):
                        fulltext = db_manager.get_fulltext(paper['id'])
                        if fulltext:
                            with st.expander("Full Text", expanded=True):
                                st.write(fulltext['full_text'][:5000] + "..." if len(fulltext['full_text']) > 5000 else fulltext['full_text'])
    
    # -------------------------- EXPORT SECTION --------------------------
    st.subheader("📤 Export Options")
    
    # Create columns for export buttons
    export_cols = st.columns(5)
    
    # 1. ZIP Export
    if "ZIP" in export_formats and st.session_state.zip_buffer:
        with export_cols[0]:
            st.download_button(
                label="📦 Download ZIP",
                data=st.session_state.zip_buffer.getvalue() if hasattr(st.session_state.zip_buffer, 'getvalue') else st.session_state.zip_buffer,
                file_name="piezoelectricity_papers.zip",
                mime="application/zip",
                use_container_width=True
            )
    
    # 2. CSV Export
    if "CSV" in export_formats:
        with export_cols[1]:
            csv_data = db_manager.export_metadata("csv")
            st.download_button(
                label="📊 Download CSV",
                data=csv_data,
                file_name="piezoelectricity_metadata.csv",
                mime="text/csv",
                use_container_width=True
            )
    
    # 3. JSON Export
    if "JSON" in export_formats:
        with export_cols[2]:
            json_data = db_manager.export_metadata("json")
            st.download_button(
                label="📄 Download JSON",
                data=json_data,
                file_name="piezoelectricity_metadata.json",
                mime="application/json",
                use_container_width=True
            )
    
    # 4. SQLite Backup
    if "Backup" in export_formats:
        with export_cols[3]:
            if st.button("💾 Backup Databases", use_container_width=True):
                with st.spinner("Creating backup..."):
                    backup_buffer = db_manager.backup_databases()
                    st.download_button(
                        label="📥 Download Backup",
                        data=backup_buffer.getvalue(),
                        file_name="piezoelectricity_backup.zip",
                        mime="application/zip",
                        use_container_width=True,
                        key="backup_download"
                    )
    
    # 5. Individual Database Downloads
    with export_cols[4]:
        with st.expander("🗃️ Database Files"):
            col_db1, col_db2, col_db3 = st.columns(3)
            
            with col_db1:
                if os.path.exists(db_manager.metadata_db):
                    with open(db_manager.metadata_db, 'rb') as f:
                        st.download_button(
                            label="Metadata DB",
                            data=f.read(),
                            file_name="piezoelectricity_metadata.db",
                            mime="application/octet-stream",
                            use_container_width=True
                        )
            
            with col_db2:
                if os.path.exists(db_manager.universe_db):
                    with open(db_manager.universe_db, 'rb') as f:
                        st.download_button(
                            label="Universe DB",
                            data=f.read(),
                            file_name="piezoelectricity_universe.db",
                            mime="application/octet-stream",
                            use_container_width=True
                        )
            
            with col_db3:
                if os.path.exists(db_manager.pdf_db):
                    with open(db_manager.pdf_db, 'rb') as f:
                        st.download_button(
                            label="PDF Storage DB",
                            data=f.read(),
                            file_name="piezoelectricity_pdfs.db",
                            mime="application/octet-stream",
                            use_container_width=True
                        )
    
    # -------------------------- DATABASE MANAGEMENT --------------------------
    st.subheader("🗄️ Database Management")
    
    col_mgmt1, col_mgmt2 = st.columns(2)
    
    with col_mgmt1:
        if st.button("🔄 Refresh Statistics", use_container_width=True):
            st.session_state.db_stats = db_manager.get_db_stats()
            st.rerun()
    
    with col_mgmt2:
        if st.button("🧹 Clean Temporary Files", use_container_width=True):
            # Clean temp directory
            temp_dir = os.path.join(DB_DIR, "temp_pdfs")
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
                os.makedirs(temp_dir, exist_ok=True)
            st.success("Temporary files cleaned")
    
    # Show database info
    with st.expander("📊 Database Information", expanded=False):
        stats = db_manager.get_db_stats()
        st.json(stats)
        
        # Sample queries
        st.write("**Sample Queries:**")
        
        query_col1, query_col2 = st.columns(2)
        
        with query_col1:
            if st.button("Top 10 Most Relevant"):
                conn = sqlite3.connect(db_manager.metadata_db)
                df_top = pd.read_sql_query(
                    "SELECT title, authors, year, relevance_prob FROM papers ORDER BY relevance_prob DESC LIMIT 10",
                    conn
                )
                conn.close()
                st.dataframe(df_top, use_container_width=True)
        
        with query_col2:
            if st.button("Papers with PDFs"):
                conn = sqlite3.connect(db_manager.metadata_db)
                df_pdfs = pd.read_sql_query(
                    "SELECT COUNT(*) as count, AVG(pdf_size)/1024 as avg_size_kb FROM papers WHERE has_pdf = 1",
                    conn
                )
                conn.close()
                st.metric("PDFs Stored", df_pdfs.iloc[0]['count'])
                st.metric("Avg PDF Size", f"{df_pdfs.iloc[0]['avg_size_kb']:.1f} KB")

# -------------------------- FOOTER --------------------------
st.divider()
st.caption("""
**Piezoelectricity in PVDF Research Tool** | 
PDFs stored in SQLite databases | 
Metadata: `piezoelectricity_metadata.db` | 
Full Text: `piezoelectricity_universe.db` | 
PDF Storage: `piezoelectricity_pdfs.db`
""")

# Always show logs
show_logs()
