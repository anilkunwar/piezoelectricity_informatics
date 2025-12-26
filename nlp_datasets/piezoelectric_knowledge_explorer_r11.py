# --------------------------------------------------------------
#  Piezoelectricity in PVDF – ENHANCED BATCH DOWNLOAD VERSION
#  ✅ No SQLite cross-DB FK
#  ✅ No nested expanders
#  ✅ No NameError on 'papers'
#  ✅ Full dopant/beta-phase Numba scoring
#  ✅ Batch PDF downloads with single click
#  ✅ Universe database streaming download
#  ✅ Guaranteed widget key uniqueness
#  ✅ Memory-optimized large file handling
# --------------------------------------------------------------
import arxiv
import fitz  # PyMuPDF
import pandas as pd
import streamlit as st
import os
import re
import sqlite3
from datetime import datetime, timedelta
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
from typing import List, Dict, Any, Optional, Tuple, Generator, Callable
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from tenacity import retry, stop_after_attempt, wait_fixed, wait_exponential, retry_if_exception_type
from numba import njit, prange
import threading
from queue import Queue, Empty
import traceback
import sys
import math
from collections import defaultdict, Counter
import textwrap
from io import BytesIO
import base64
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
    st.set_page_config(
        page_title="Piezoelectricity in PVDF Research Hub",
        layout="wide",
        page_icon="🔬",
        menu_items={
            'Get Help': 'https://github.com/pvdf-research/piezoelectricity-tool',
            'Report a bug': "https://github.com/pvdf-research/piezoelectricity-tool/issues",
            'About': """
            Advanced research tool for piezoelectricity studies in PVDF materials.
            Features batch downloads, database management, and AI-powered analysis.
            """
        }
    )
    st.session_state.page_config_set = True

# =================================================================
# -------------------------- TERM DEFINITIONS --------------------------
DOPANT_TERMS = {
    'sno2', 'tin oxide', 'zno', 'zinc oxide', 'tio2', 'titanium dioxide', 'graphene', 'cnt', 'carbon nanotube',
    'batio3', 'barium titanate', 'bto', 'nio', 'nickel oxide', 'fe3o4', 'magnetite', 'al2o3', 'alumina', 'sio2', 'silica',
    'clay', 'montmorillonite', 'tio', 'titanium monoxide', 'cao', 'calcium oxide', 'mgo', 'magnesium oxide',
    'pzt', 'lead zirconate titanate', 'nanoparticle', 'nanofiller', 'dopant', 'filler', 'nanocomposite',
    'quantum dot', 'mxene', 'boron nitride', 'cellulose nanocrystals', 'halloysite', 'reduced graphene oxide',
    'rgo', 'cnts', 'carbon nanotubes', 'metal oxide', 'ceramic filler', 'polymer blend'
}

BETA_PHASE_TERMS = {
    'beta phase', 'β-phase', 'beta-phase', '1270 cm⁻¹', '1275 cm-1', '1270cm⁻¹', '1275cm-1',
    '840 cm⁻¹', '840 cm-1', '840cm⁻¹', '840cm-1', 'ftir beta', 'fraction beta', 'beta content',
    'beta-phase content', 'phase fraction', 'beta polymorph', 'pvdf beta', 'β polymorph',
    'stretching ratio', 'poling', 'electrospinning', 'mechanical stretching', 'annealing temperature',
    'crystallinity', 'crystalline phase', 'd-phase', 'all-trans conformation', 'planar zigzag'
}

PVDF_TERMS = {'pvdf', 'polyvinylidene fluoride', 'poly(vinylidene fluoride)', 'pvdf polymer', 'fluro polymer'}

# -------------------------- CONFIG --------------------------
if IS_CLOUD:
    DB_DIR = "/tmp/pvdf_research"
    st.info("🌐 Running on Streamlit Cloud: Using temporary storage")
else:
    DB_DIR = os.path.join(os.path.expanduser("~"), "Desktop", "piezoelectricity_data")
    os.makedirs(DB_DIR, exist_ok=True)

# Create all necessary subdirectories
os.makedirs(os.path.join(DB_DIR, "temp"), exist_ok=True)
os.makedirs(os.path.join(DB_DIR, "backups"), exist_ok=True)
os.makedirs(os.path.join(DB_DIR, "downloads"), exist_ok=True)

METADATA_DB = os.path.join(DB_DIR, "piezoelectricity_metadata.db")
UNIVERSE_DB = os.path.join(DB_DIR, "piezoelectricity_universe.db")
PDF_STORAGE_DB = os.path.join(DB_DIR, "piezoelectricity_pdfs.db")
ANALYTICS_DB = os.path.join(DB_DIR, "piezoelectricity_analytics.db")

# Configure logging with rotation and detailed formatting
log_file = os.path.join(DB_DIR, "piezoelectricity_query.log")
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    filemode='a'
)
logger = logging.getLogger(__name__)

# -------------------------- WIDGET KEY MANAGEMENT --------------------------
class WidgetKeyManager:
    """Manages widget keys to ensure uniqueness across the application."""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(WidgetKeyManager, cls).__new__(cls)
            cls._instance._used_keys = set()
            cls._instance._namespace_counters = defaultdict(int)
        return cls._instance
    
    def generate_unique_key(self, base_key: str, namespace: str = "default") -> str:
        """
        Generate a unique widget key with namespace management.
        
        Args:
            base_key: The base key name
            namespace: The namespace for this key (prevents collisions across different sections)
        
        Returns:
            A unique key string
        """
        # Clean the base key
        clean_base = re.sub(r'[^a-zA-Z0-9_-]', '_', base_key).lower()
        
        # Create namespace-specific key
        namespaced_key = f"{namespace}_{clean_base}"
        
        # Check if this key is already used
        if namespaced_key in self._used_keys:
            # Generate a unique variant
            counter = self._namespace_counters[namespace]
            self._namespace_counters[namespace] += 1
            unique_key = f"{namespaced_key}_{counter}"
        else:
            unique_key = namespaced_key
        
        # Register the key
        self._used_keys.add(unique_key)
        return unique_key
    
    def reset_namespace(self, namespace: str):
        """Reset the counter for a specific namespace."""
        self._namespace_counters[namespace] = 0
    
    def clear_all_keys(self):
        """Clear all registered keys (use with caution)."""
        self._used_keys.clear()
        self._namespace_counters.clear()

# Initialize the key manager
key_manager = WidgetKeyManager()

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
    "bulk_download_status": "",
    "bulk_download_progress": 0.0,
    "bulk_download_complete": False,
    "universe_db_buffer": None,
    "memory_usage": {},
    "last_cleanup": None,
    "active_namespace": "main",
    "download_queue": [],
}

for k, v in DEFAULT_STATE.items():
    if k not in st.session_state:
        st.session_state[k] = v

def update_log(message: str):
    """Update log with timestamped message and rotate buffer."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = f"[{timestamp}] {message}"
    st.session_state.log_buffer.append(entry)
    if len(st.session_state.log_buffer) > 200:
        st.session_state.log_buffer.pop(0)
    logger.info(message)
    return entry

def cleanup_memory(force: bool = False):
    """Clean up memory by clearing caches and removing temporary files."""
    try:
        # Only clean if it's been more than 5 minutes or force is True
        last_cleanup = st.session_state.get('last_cleanup')
        if force or not last_cleanup or (datetime.now() - last_cleanup).total_seconds() > 300:
            # Clear large objects from session state
            for key in list(st.session_state.keys()):
                if key in ['zip_buffer', 'universe_db_buffer']:
                    if st.session_state[key] is not None:
                        st.session_state[key] = None
            
            # Remove temp files
            temp_dir = os.path.join(DB_DIR, "temp")
            cleaned_count = 0
            if os.path.exists(temp_dir):
                for file in os.listdir(temp_dir):
                    try:
                        file_path = os.path.join(temp_dir, file)
                        if os.path.isfile(file_path):
                            os.remove(file_path)
                            cleaned_count += 1
                    except Exception as e:
                        logger.warning(f"Error removing temp file {file}: {e}")
            
            # Clear temp files list
            st.session_state.temp_files = []
            
            # Force garbage collection
            gc.collect()
            
            # Update memory usage
            process = psutil.Process(os.getpid())
            st.session_state.memory_usage = {
                'rss_mb': process.memory_info().rss / 1024 / 1024,
                'vms_mb': process.memory_info().vms / 1024 / 1024,
                'last_update': datetime.now()
            }
            
            st.session_state.last_cleanup = datetime.now()
            update_log(f"Memory cleanup completed. Removed {cleaned_count} temp files. RSS: {st.session_state.memory_usage['rss_mb']:.1f}MB")
    except Exception as e:
        logger.error(f"Error during memory cleanup: {e}")
        update_log(f"Memory cleanup failed: {e}")

# -------------------------- NUMBA SCORER --------------------------
@njit(parallel=True)
def compute_relevance_score_numba(
    dopant_hits: np.ndarray,
    beta_hits: np.ndarray,
    pvdf_hits: np.ndarray,
    weights: np.ndarray
) -> float:
    """Numba-optimized relevance scoring with parallel execution."""
    score = (
        weights[0] * np.sum(dopant_hits) +
        weights[1] * np.sum(beta_hits) +
        weights[2] * np.sum(pvdf_hits)
    )
    max_possible = weights.sum() * 3.0
    return min(100.0, (score / (max_possible + 1e-8)) * 100.0)

def analyze_dopant_beta_relevance(text: str) -> Dict[str, Any]:
    """Enhanced analysis with context awareness and term weighting."""
    text_lower = text.lower()
    
    # Weighted term detection
    dopant_weight = 0.0
    for term in DOPANT_TERMS:
        if term in text_lower:
            # Give higher weight to more specific terms
            if len(term.split()) > 1:  # Multi-word terms are more specific
                dopant_weight += 0.7
            else:
                dopant_weight += 0.3
    
    beta_weight = 0.0
    for term in BETA_PHASE_TERMS:
        if term in text_lower:
            if 'cm⁻¹' in term or 'cm-1' in term:  # FTIR peaks are high confidence
                beta_weight += 0.8
            elif 'fraction' in term or 'content' in term:  # Quantitative terms
                beta_weight += 0.6
            else:
                beta_weight += 0.4
    
    pvdf_weight = 0.0
    for term in PVDF_TERMS:
        if term in text_lower:
            pvdf_weight += 0.5
    
    # Convert to binary arrays for Numba (but keep weights for context)
    has_dopant = dopant_weight > 0
    has_beta = beta_weight > 0
    has_pvdf = pvdf_weight > 0
    
    dopant_arr = np.array([1 if has_dopant else 0], dtype=np.int8)
    beta_arr = np.array([1 if has_beta else 0], dtype=np.int8)
    pvdf_arr = np.array([1 if has_pvdf else 0], dtype=np.int8)
    
    weights = np.array([0.5, 0.4, 0.1], dtype=np.float32)
    
    base_score = compute_relevance_score_numba(dopant_arr, beta_arr, pvdf_arr, weights)
    
    # Apply context boosts
    context_boost = 0.0
    if has_dopant and has_beta:
        context_boost += 0.15  # Strong boost for papers discussing both
    if has_pvdf and (has_dopant or has_beta):
        context_boost += 0.1
    
    enhanced_score = min(100.0, base_score * (1 + context_boost))
    
    return {
        "dopant_present": bool(has_dopant),
        "beta_phase_present": bool(has_beta),
        "pvdf_present": bool(has_pvdf),
        "enhanced_relevance_score": float(enhanced_score),
        "dopant_weight": float(dopant_weight),
        "beta_weight": float(beta_weight),
        "context_boost": float(context_boost)
    }

# -------------------------- DATABASE MANAGER WITH BATCH OPERATIONS --------------------------
class DatabaseManager:
    def __init__(self):
        self.metadata_db = METADATA_DB
        self.universe_db = UNIVERSE_DB
        self.pdf_db = PDF_STORAGE_DB
        self.analytics_db = ANALYTICS_DB
        self.init_databases()
        self.init_analytics_database()
        update_log("Database manager initialized with batch operations support")
    
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
            word_count INTEGER DEFAULT 0,
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
            extracted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )""")
        c.execute("""CREATE VIRTUAL TABLE IF NOT EXISTS papers_fts
            USING fts5(paper_id, title, abstract, full_text, content='papers_fulltext', tokenize='porter')""")
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
            UNIQUE(paper_id, chunk_index)
        )""")
        c.execute("CREATE INDEX IF NOT EXISTS idx_pdf_hash ON pdf_storage(pdf_hash)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_file_size ON pdf_storage(file_size)")
        conn.commit()
        conn.close()

    def init_analytics_database(self):
        """Initialize analytics database for storing research insights."""
        conn = sqlite3.connect(self.analytics_db)
        c = conn.cursor()
        
        # Store research trends over time
        c.execute("""CREATE TABLE IF NOT EXISTS research_trends (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            year INTEGER,
            dopant_mentions INTEGER,
            beta_phase_mentions INTEGER,
            total_papers INTEGER,
            avg_relevance_score REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )""")
        
        # Store keyword co-occurrence
        c.execute("""CREATE TABLE IF NOT EXISTS keyword_cooccurrence (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            keyword1 TEXT,
            keyword2 TEXT,
            cooccurrence_count INTEGER,
            year INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )""")
        
        conn.commit()
        conn.close()
    
    def get_db_stats(self) -> Dict[str, Any]:
        """Get comprehensive database statistics."""
        stats = {}
        try:
            # Metadata stats
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
            c.execute("SELECT AVG(enhanced_relevance_score) FROM papers WHERE enhanced_relevance_score > 0")
            avg_score = c.fetchone()[0]
            stats['avg_relevance_score'] = round(avg_score, 2) if avg_score else 0
            c.execute("SELECT COUNT(*) FROM papers WHERE dopant_present = 1")
            stats['dopant_papers'] = c.fetchone()[0]
            c.execute("SELECT COUNT(*) FROM papers WHERE beta_phase_present = 1")
            stats['beta_phase_papers'] = c.fetchone()[0]
            conn.close()

            # Universe DB stats
            conn = sqlite3.connect(self.universe_db)
            c = conn.cursor()
            c.execute("SELECT COUNT(*) FROM papers_fulltext")
            stats['fulltext_count'] = c.fetchone()[0]
            c.execute("SELECT SUM(word_count) FROM papers_fulltext")
            stats['total_words'] = c.fetchone()[0] or 0
            c.execute("SELECT AVG(word_count) FROM papers_fulltext WHERE word_count > 0")
            avg_words = c.fetchone()[0]
            stats['avg_words_per_paper'] = round(avg_words, 1) if avg_words else 0
            conn.close()

            # PDF DB stats
            conn = sqlite3.connect(self.pdf_db)
            c = conn.cursor()
            c.execute("SELECT COUNT(*) FROM pdf_storage")
            stats['pdf_storage_count'] = c.fetchone()[0]
            c.execute("SELECT SUM(file_size) FROM pdf_storage")
            total_bytes = c.fetchone()[0] or 0
            stats['total_pdf_size_mb'] = round(total_bytes / (1024 * 1024), 2)
            c.execute("SELECT AVG(file_size) FROM pdf_storage WHERE file_size > 0")
            avg_size = c.fetchone()[0]
            stats['avg_pdf_size_kb'] = round((avg_size / 1024) if avg_size else 0, 1)
            c.execute("SELECT AVG(page_count) FROM pdf_storage WHERE page_count > 0")
            avg_pages = c.fetchone()[0]
            stats['avg_pages_per_pdf'] = round(avg_pages, 1) if avg_pages else 0
            conn.close()

            # Calculate coverage percentages
            if stats['total_papers'] > 0:
                stats['pdf_coverage'] = round((stats['pdfs_stored'] / stats['total_papers']) * 100, 1)
                stats['fulltext_coverage'] = round((stats['fulltext_stored'] / stats['total_papers']) * 100, 1)
            else:
                stats['pdf_coverage'] = 0
                stats['fulltext_coverage'] = 0

            # Calculate growth rate
            if stats['years_covered'] > 1 and stats['total_papers'] > 0:
                stats['avg_papers_per_year'] = round(stats['total_papers'] / stats['years_covered'], 1)
            else:
                stats['avg_papers_per_year'] = 0

        except Exception as e:
            logger.error(f"Error getting DB stats: {e}")
            stats = {'error': str(e)}
        return stats

    def store_paper_metadata(self, paper: Dict[str, Any]) -> bool:
        """Store paper metadata with enhanced fields."""
        try:
            conn = sqlite3.connect(self.metadata_db)
            c = conn.cursor()
            c.execute("""INSERT OR REPLACE INTO papers
                (id, arxiv_id, title, authors, year, categories, abstract,
                pdf_url, published_date, updated_date, doi, relevance_score,
                matched_terms, download_status, pdf_stored, fulltext_stored,
                pdf_size, download_time, enhanced_relevance_score,
                dopant_present, beta_phase_present, word_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
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
                    paper.get('beta_phase_present'),
                    paper.get('word_count', 0)
                ))
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            logger.error(f"Failed to store metadata for {paper.get('id')}: {e}")
            return False

    def store_pdf_data(self, paper_id: str, pdf_bytes: bytes, pdf_url: str) -> bool:
        """Store PDF data with memory optimization and chunking for large files."""
        try:
            pdf_hash = hashlib.sha256(pdf_bytes).hexdigest()
            file_size = len(pdf_bytes)
            
            # Extract page count safely
            page_count = 0
            try:
                with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
                    page_count = len(doc)
            except Exception as e:
                logger.warning(f"Error extracting page count for {paper_id}: {e}")

            conn = sqlite3.connect(self.pdf_db)
            c = conn.cursor()
            
            # Check if PDF already exists
            c.execute("SELECT paper_id FROM pdf_storage WHERE pdf_hash = ?", (pdf_hash,))
            existing = c.fetchone()
            if existing:
                logger.info(f"PDF already exists for {paper_id}, updating reference")
                c.execute("UPDATE papers SET pdf_stored = 1, pdf_size = ? WHERE id = ?", (file_size, paper_id))
                conn.commit()
                conn.close()
                return True

            # Store PDF in chunks if large (over 10MB)
            if file_size > 10 * 1024 * 1024:  # 10MB
                logger.info(f"Large PDF detected ({file_size/1024/1024:.1f}MB), storing in chunks")
                chunk_size = 5 * 1024 * 1024  # 5MB chunks
                chunks = []
                
                for i in range(0, file_size, chunk_size):
                    chunk = pdf_bytes[i:i+chunk_size]
                    chunk_hash = hashlib.sha256(chunk).hexdigest()
                    chunks.append((paper_id, len(chunks), sqlite3.Binary(chunk), chunk_hash))
                
                c.execute("""INSERT INTO pdf_storage
                    (paper_id, pdf_data, pdf_hash, original_url, file_size, page_count, compression_method)
                    VALUES (?, ?, ?, ?, ?, ?, 'chunked')""",
                    (paper_id, sqlite3.Binary(b''), pdf_hash, pdf_url, file_size, page_count))
                
                c.executemany("""INSERT INTO pdf_chunks
                    (paper_id, chunk_index, chunk_data, chunk_hash)
                    VALUES (?, ?, ?, ?)""", chunks)
            else:
                # Store as single blob
                c.execute("""INSERT OR REPLACE INTO pdf_storage
                    (paper_id, pdf_data, pdf_hash, original_url, file_size, page_count)
                    VALUES (?, ?, ?, ?, ?, ?)""",
                    (paper_id, sqlite3.Binary(pdf_bytes), pdf_hash, pdf_url, file_size, page_count))
            
            # Update metadata
            c.execute("UPDATE papers SET pdf_stored = 1, pdf_size = ? WHERE id = ?", (file_size, paper_id))
            conn.commit()
            conn.close()
            
            logger.info(f"Stored PDF for {paper_id} ({file_size/1024:.1f} KB, {page_count} pages)")
            return True
        except Exception as e:
            logger.error(f"Failed to store PDF for {paper_id}: {e}")
            return False

    def store_fulltext(self, paper_id: str, title: str, abstract: str,
                       full_text: str, page_count: int = 0) -> bool:
        """Store full text with advanced analysis."""
        try:
            start_time = time.time()
            text_hash = hashlib.md5(full_text.encode()).hexdigest()
            word_count = len(full_text.split())
            
            # Enhanced analysis
            analysis = analyze_dopant_beta_relevance(full_text)
            enhanced_score = analysis["enhanced_relevance_score"]
            
            conn = sqlite3.connect(self.universe_db)
            c = conn.cursor()
            
            # Store full text
            c.execute("""INSERT OR REPLACE INTO papers_fulltext
                (paper_id, title, abstract, full_text, text_hash, word_count, page_count)
                VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (paper_id, title, abstract, full_text, text_hash, word_count, page_count))
            
            # Update FTS index
            c.execute("""INSERT OR REPLACE INTO papers_fts
                (paper_id, title, abstract, full_text)
                VALUES (?, ?, ?, ?)""",
                (paper_id, title, abstract, full_text))
            
            conn.commit()
            conn.close()

            # Update metadata with analysis results
            conn = sqlite3.connect(self.metadata_db)
            c = conn.cursor()
            c.execute("""UPDATE papers SET fulltext_stored = 1,
                enhanced_relevance_score = ?,
                dopant_present = ?,
                beta_phase_present = ?,
                word_count = ?
                WHERE id = ?""",
                (enhanced_score, analysis["dopant_present"],
                 analysis["beta_phase_present"], word_count, paper_id))
            conn.commit()
            conn.close()
            
            extraction_time = time.time() - start_time
            logger.info(f"Stored full text for {paper_id} ({word_count} words, score: {enhanced_score:.1f}, time: {extraction_time:.2f}s)")
            return True
        except Exception as e:
            logger.error(f"Failed to store full text for {paper_id}: {e}")
            return False

    def get_pdf(self, paper_id: str) -> Optional[bytes]:
        """Retrieve PDF data, handling both chunked and single-blob storage."""
        try:
            conn = sqlite3.connect(self.pdf_db)
            c = conn.cursor()
            
            # Check storage method
            c.execute("SELECT compression_method, file_size FROM pdf_storage WHERE paper_id = ?", (paper_id,))
            result = c.fetchone()
            if not result:
                conn.close()
                return None
            
            compression_method, file_size = result
            
            if compression_method == 'chunked':
                # Retrieve chunks
                c.execute("SELECT chunk_data FROM pdf_chunks WHERE paper_id = ? ORDER BY chunk_index", (paper_id,))
                chunks = c.fetchall()
                if not chunks:
                    conn.close()
                    return None
                
                pdf_bytes = b''.join(chunk[0] for chunk in chunks if chunk[0])
                logger.info(f"Retrieved chunked PDF for {paper_id} ({len(pdf_bytes)/1024/1024:.1f}MB)")
            else:
                # Retrieve single blob
                c.execute("SELECT pdf_data FROM pdf_storage WHERE paper_id = ?", (paper_id,))
                result = c.fetchone()
                pdf_bytes = result[0] if result else None
            
            conn.close()
            return pdf_bytes
        except Exception as e:
            logger.error(f"Failed to retrieve PDF for {paper_id}: {e}")
            return None

    def get_paper_info(self, paper_id: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive paper information."""
        try:
            conn = sqlite3.connect(self.metadata_db)
            c = conn.cursor()
            c.execute("""SELECT title, authors, year, abstract, pdf_url,
                relevance_score, enhanced_relevance_score, dopant_present, beta_phase_present,
                download_status, pdf_stored, fulltext_stored,
                pdf_size, word_count
                FROM papers WHERE id = ?""", (paper_id,))
            meta = c.fetchone()
            conn.close()
            if not meta:
                return None
            
            # Get PDF data
            pdf_bytes = self.get_pdf(paper_id)
            
            # Get fulltext info
            conn = sqlite3.connect(self.universe_db)
            c = conn.cursor()
            c.execute("SELECT word_count, page_count FROM papers_fulltext WHERE paper_id = ?", (paper_id,))
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
                'pdf_size': meta[12],
                'word_count': meta[13] or (fulltext_info[0] if fulltext_info else 0),
                'page_count': fulltext_info[1] if fulltext_info else 0,
                'pdf_bytes': pdf_bytes
            }
        except Exception as e:
            logger.error(f"Failed to get paper info for {paper_id}: {e}")
            return None

    def create_zip_from_papers(
        self, 
        papers: List[Dict], 
        progress_callback: Optional[Callable[[float, str], None]] = None,
        max_concurrent: int = 3
    ) -> io.BytesIO:
        """
        Create ZIP archive from paper list with progress tracking and memory management.
        
        Args:
            papers: List of paper dictionaries
            progress_callback: Function to call with progress (0-1) and status message
            max_concurrent: Maximum number of concurrent downloads
        """
        zip_buffer = io.BytesIO()
        total_papers = len(papers)
        
        try:
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED, allowZip64=True) as zip_file:
                # Process papers in batches to manage memory
                batch_size = max(1, min(10, max_concurrent * 2))
                
                for batch_start in range(0, total_papers, batch_size):
                    batch_end = min(batch_start + batch_size, total_papers)
                    batch_papers = papers[batch_start:batch_end]
                    
                    # Process batch with concurrent downloads
                    with concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrent) as executor:
                        future_to_paper = {
                            executor.submit(self._get_paper_pdf_data, paper): paper 
                            for paper in batch_papers
                        }
                        
                        for future in concurrent.futures.as_completed(future_to_paper):
                            paper = future_to_paper[future]
                            try:
                                pdf_data, filename = future.result()
                                if pdf_data:
                                    zip_file.writestr(filename, pdf_data)
                                    logger.info(f"Added {paper['id']} to ZIP ({len(pdf_data)/1024:.1f} KB)")
                            except Exception as e:
                                logger.error(f"Error processing {paper['id']}: {e}")
                    
                    # Update progress
                    if progress_callback:
                        progress = (batch_end) / total_papers
                        status = f"Processed {batch_end}/{total_papers} papers"
                        progress_callback(progress, status)
                    
                    # Clean up memory after each batch
                    cleanup_memory()
            
            zip_buffer.seek(0)
            logger.info(f"Created ZIP with {total_papers} papers, size: {zip_buffer.getbuffer().nbytes/1024/1024:.1f}MB")
            return zip_buffer
            
        except Exception as e:
            logger.error(f"Failed to create ZIP: {e}")
            traceback.print_exc()
            return io.BytesIO()

    def _get_paper_pdf_data(self, paper: Dict) -> Tuple[Optional[bytes], Optional[str]]:
        """Helper method to get PDF data and filename for a single paper."""
        paper_id = paper['id']
        
        # Get PDF bytes - check session state first, then database
        pdf_bytes = None
        if paper_id in st.session_state.downloaded_pdfs:
            pdf_bytes = st.session_state.downloaded_pdfs[paper_id].get('pdf_bytes')
        
        if pdf_bytes is None:
            pdf_bytes = self.get_pdf(paper_id)
        
        if pdf_bytes is None:
            # Try to download if not available
            logger.info(f"PDF not found for {paper_id}, attempting download")
            pdf_bytes = download_pdf_bytes(paper['pdf_url'])
            if pdf_bytes:
                # Store in session state for future use
                st.session_state.downloaded_pdfs[paper_id] = {
                    'pdf_bytes': pdf_bytes,
                    'title': paper['title'],
                    'authors': paper['authors'],
                    'year': paper['year']
                }
                # Store in database
                self.store_pdf_data(paper_id, pdf_bytes, paper['pdf_url'])
        
        if not pdf_bytes:
            return None, None
        
        # Create clean filename
        title = re.sub(r'[^\w\s-]', '', paper['title'])[:100]
        authors = paper['authors'].split(',')[0][:50] if paper['authors'] else 'unknown'
        filename = f"{paper_id}_{authors}_{paper['year']}_{title}.pdf"
        filename = re.sub(r'\s+', '_', filename)
        filename = re.sub(r'_{2,}', '_', filename)  # Remove multiple underscores
        
        return pdf_bytes, filename

    def export_metadata(self, format: str = "csv", papers: Optional[List[Dict]] = None) -> io.BytesIO:
        """Export metadata with flexible paper selection."""
        try:
            conn = sqlite3.connect(self.metadata_db)
            
            if papers:
                # Export specific papers
                paper_ids = [p['id'] for p in papers]
                placeholders = ','.join(['?'] * len(paper_ids))
                query = f"SELECT * FROM papers WHERE id IN ({placeholders})"
                df = pd.read_sql_query(query, conn, params=paper_ids)
            else:
                # Export all papers
                df = pd.read_sql_query("SELECT * FROM papers", conn)
            
            output = io.BytesIO()
            
            if format.lower() == "csv":
                df.to_csv(output, index=False, encoding='utf-8')
            elif format.lower() == "json":
                df.to_json(output, orient="records", indent=2, force_ascii=False)
            elif format.lower() == "excel":
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    df.to_excel(writer, index=False, sheet_name='Papers')
                    # Add summary sheet
                    summary = pd.DataFrame({
                        'Metric': ['Total Papers', 'PDFs Stored', 'Full Text Stored', 'Average Relevance'],
                        'Value': [
                            len(df),
                            df['pdf_stored'].sum(),
                            df['fulltext_stored'].sum(),
                            df['enhanced_relevance_score'].mean()
                        ]
                    })
                    summary.to_excel(writer, sheet_name='Summary', index=False)
            elif format.lower() == "parquet":
                import pyarrow as pa
                import pyarrow.parquet as pq
                table = pa.Table.from_pandas(df)
                pq.write_table(table, output)
            else:
                return io.BytesIO()
            
            output.seek(0)
            conn.close()
            return output
        except Exception as e:
            logger.error(f"Export failed: {e}")
            return io.BytesIO()

    def stream_database_backup(self, db_path: str, chunk_size: int = 8192) -> Generator[bytes, None, None]:
        """Stream database backup in chunks to avoid memory issues."""
        if not os.path.exists(db_path):
            logger.error(f"Database file not found: {db_path}")
            return
        
        try:
            file_size = os.path.getsize(db_path)
            chunks_sent = 0
            
            with open(db_path, 'rb') as f:
                while True:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    
                    chunks_sent += 1
                    if chunks_sent % 100 == 0:  # Log every 100 chunks
                        progress = (chunks_sent * chunk_size) / file_size
                        logger.info(f"Streaming backup: {progress:.1%} complete")
                    
                    yield chunk
            
            logger.info(f"Database backup streamed successfully: {db_path}")
            
        except Exception as e:
            logger.error(f"Error streaming database backup: {e}")
            yield b''

# Initialize DB
db_manager = DatabaseManager()

# -------------------------- DOWNLOAD & QUERY FUNCTIONS WITH BATCH SUPPORT --------------------------
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((requests.exceptions.RequestException, requests.exceptions.Timeout))
)
def download_pdf_bytes(pdf_url: str, timeout: int = 60) -> Optional[bytes]:
    """Download PDF with retries and timeout, optimized for large files."""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'application/pdf,text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate',
        'DNT': '1',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1'
    }
    
    try:
        session = requests.Session()
        retries = Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
        session.mount('https://', HTTPAdapter(max_retries=retries))
        session.mount('http://', HTTPAdapter(max_retries=retries))
        
        logger.info(f"Starting download from: {pdf_url}")
        response = session.get(pdf_url, headers=headers, timeout=timeout, stream=True)
        response.raise_for_status()
        
        # Check content type
        content_type = response.headers.get('Content-Type', '').lower()
        if 'pdf' not in content_type and 'octet-stream' not in content_type:
            logger.warning(f"Unexpected content type: {content_type} for {pdf_url}")
        
        # Stream download to avoid memory issues for large files
        pdf_bytes = b''
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                pdf_bytes += chunk
                # Check for unreasonable file size
                if len(pdf_bytes) > 100 * 1024 * 1024:  # 100MB
                    raise ValueError("PDF file too large")
        
        if len(pdf_bytes) < 1024:
            raise ValueError("PDF file too small")
        
        logger.info(f"Downloaded PDF: {len(pdf_bytes)/1024/1024:.2f}MB")
        return pdf_bytes
    except Exception as e:
        logger.error(f"Download failed for {pdf_url}: {e}")
        return None

def extract_text_from_bytes(pdf_bytes: bytes, max_pages: int = 50, max_chars: int = 2000000) -> str:
    """Extract text from PDF bytes with limits and error handling."""
    try:
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            text = ""
            for page_num in range(min(max_pages, len(doc))):
                try:
                    page_text = doc[page_num].get_text()
                    text += page_text
                    if len(text) > max_chars:
                        break
                except Exception as e:
                    logger.warning(f"Error extracting page {page_num}: {e}")
        
        # Clean text
        text = re.sub(r'\s+', ' ', text).strip()
        return text[:max_chars]
    except Exception as e:
        logger.error(f"PDF text extraction failed: {e}")
        return f"Error extracting text: {str(e)}"

def handle_paper_download(paper: Dict[str, Any], manual_download: bool = False) -> Dict[str, Any]:
    """Handle paper download with comprehensive error handling and memory management."""
    paper_id = paper['id']
    if manual_download and paper_id in st.session_state.downloaded_pdfs:
        logger.info(f"PDF for {paper_id} already in session cache")
        return paper
    
    try:
        logger.info(f"Starting download for {paper_id} from {paper['pdf_url']}...")
        
        # Check memory before download
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024
        if memory_mb > 512:  # 512MB threshold
            cleanup_memory(force=True)
            logger.info(f"Memory cleanup performed before download (was {memory_mb:.1f}MB)")
        
        # Download PDF
        pdf_bytes = download_pdf_bytes(
            paper['pdf_url'], 
            timeout=120  # Longer timeout for large files
        )
        
        if pdf_bytes is None:
            paper['download_status'] = "Failed to download"
            logger.error(f"❌ Download failed for {paper_id}")
            return paper
        
        # Extract text
        full_text = extract_text_from_bytes(pdf_bytes, max_pages=100, max_chars=5000000)
        
        # Store PDF and full text
        pdf_stored = db_manager.store_pdf_data(paper_id, pdf_bytes, paper['pdf_url'])
        
        text_stored = False
        if not full_text.startswith("Error"):
            text_stored = db_manager.store_fulltext(paper_id, paper['title'], paper['abstract'], full_text)
            # Update paper info with analysis results
            info = db_manager.get_paper_info(paper_id)
            if info:
                paper['enhanced_relevance_score'] = info['enhanced_relevance_score']
                paper['dopant_present'] = info['dopant_present']
                paper['beta_phase_present'] = info['beta_phase_present']
                paper['word_count'] = info['word_count']
        else:
            logger.warning(f"Text extraction error for {paper_id}: {full_text}")
        
        # Update paper metadata
        paper['pdf_stored'] = 1 if pdf_stored else 0
        paper['fulltext_stored'] = 1 if text_stored else 0
        paper['pdf_size'] = len(pdf_bytes)
        paper['download_time'] = datetime.now().isoformat()
        paper['download_status'] = "Successfully downloaded and stored"
        
        # Store in session state for quick access
        st.session_state.downloaded_pdfs[paper_id] = {
            'pdf_bytes': pdf_bytes,
            'title': paper['title'],
            'authors': paper['authors'],
            'year': paper['year'],
            'size': len(pdf_bytes)
        }
        
        # Update database
        db_manager.store_paper_metadata(paper)
        
        logger.info(f"✅ Successfully processed {paper_id} ({len(pdf_bytes)/1024/1024:.2f} MB)")
        return paper
    except Exception as e:
        error_msg = f"Failed to process {paper_id}: {str(e)[:500]}"
        paper['download_status'] = error_msg
        logger.error(f"❌ {error_msg}")
        logger.error(traceback.format_exc())
        return paper
    finally:
        # Force garbage collection
        gc.collect()

@st.cache_data(ttl=3600, show_spinner=False)
def query_arxiv(query: str, categories: List[str], max_results: int,
                start_year: int, end_year: int, batch_size: int = 100) -> List[Dict[str, Any]]:
    """Query arXiv with batched results and enhanced filtering."""
    client = arxiv.Client(page_size=batch_size, delay_seconds=3)
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
        sort_order=arxiv.SortOrder.Descending
    )
    
    results = []
    query_terms = {term.strip('"').lower() for term in query.split('OR')}
    processed_ids = set()
    
    try:
        for result in client.results(search):
            # Skip duplicates
            if result.entry_id in processed_ids:
                continue
            
            # Apply year and category filters
            if not (start_year <= result.published.year <= end_year):
                continue
            
            if categories and not any(cat in result.categories for cat in categories):
                continue
            
            # Enhanced relevance scoring
            abstract_lower = result.summary.lower()
            title_lower = result.title.lower()
            
            matched_terms = [term for term in query_terms if term in abstract_lower or term in title_lower]
            if not matched_terms:
                continue
            
            # Calculate relevance score
            relevance_score = min(len(matched_terms) / len(query_terms) * 100, 100)
            
            # Create paper dictionary
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
            processed_ids.add(result.entry_id)
            
            if len(results) >= max_results:
                break
        
        # Sort by relevance score
        results.sort(key=lambda x: x['relevance_score'], reverse=True)
        logger.info(f"Found {len(results)} papers from arXiv")
        return results
    
    except Exception as e:
        logger.error(f"arXiv query failed: {e}")
        logger.error(traceback.format_exc())
        return []

# -------------------------- UI COMPONENTS WITH PROPER KEY MANAGEMENT --------------------------
def show_logs(expanded: bool = False, namespace: str = "main"):
    """Display logs with unique key management to avoid duplicate widget errors."""
    if st.session_state.log_buffer:
        # Generate unique key for this instance
        log_key = key_manager.generate_unique_key("logs_display", namespace=namespace)
        
        with st.expander("📋 Processing Logs", expanded=expanded):
            # Use unique key based on timestamp and namespace
            current_time = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            unique_key = f"{log_key}_{current_time}"
            
            st.text_area(
                "Logs", 
                "\n".join(st.session_state.log_buffer[-100:]), 
                height=250, 
                key=unique_key
            )

def create_dashboard():
    """Create comprehensive dashboard with statistics and visualizations."""
    stats = db_manager.get_db_stats()
    
    st.subheader("📊 Database Statistics")
    
    # Main metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1: st.metric("Total Papers", stats.get('total_papers', 0))
    with col2: st.metric("PDFs Stored", stats.get('pdfs_stored', 0), 
                       f"{stats.get('pdf_coverage', 0)}% coverage")
    with col3: st.metric("Full Text Papers", stats.get('fulltext_stored', 0),
                       f"{stats.get('fulltext_coverage', 0)}% coverage")
    with col4: st.metric("Total Size", f"{stats.get('total_pdf_size_mb', 0):.1f} MB")
    with col5: st.metric("Avg. Relevance", f"{stats.get('avg_relevance_score', 0):.1f}%")
    
    # Progress bars
    if stats.get('total_papers', 0) > 0:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**PDF Coverage**")
            st.progress(stats.get('pdf_coverage', 0) / 100)
        with col2:
            st.markdown("**Full Text Coverage**")
            st.progress(stats.get('fulltext_coverage', 0) / 100)
    
    # Memory usage
    if st.session_state.memory_usage:
        mem_stats = st.session_state.memory_usage
        st.subheader("💡 Memory Usage")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("RSS Memory", f"{mem_stats['rss_mb']:.1f} MB")
        with col2:
            st.metric("Virtual Memory", f"{mem_stats['vms_mb']:.1f} MB")

# -------------------------- MAIN APP --------------------------
st.title("🔬 Piezoelectricity in PVDF Research Hub")
st.markdown("""
**Advanced research platform for piezoelectricity studies in PVDF materials.** This tool combines intelligent search, automated analysis, and comprehensive data management to accelerate your research workflow.
""")

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Search configuration
    st.subheader("🔍 Search Settings")
    default_query = ' OR '.join([
        '"piezoelectricity"', '"PVDF"', '"beta phase"', '"electrospun"', '"dopants"',
        '"SnO2"', '"ZnO"', '"graphene"', '"nanogenerator"', '"energy harvesting"',
        '"d33 coefficient"', '"output voltage"', '"flexible sensor"'
    ])
    query = st.text_area("Search Query", value=default_query, height=120)
    
    default_cats = ["cond-mat.mtrl-sci", "physics.app-ph", "cond-mat.soft", "physics.chem-ph", "physics.ins-det"]
    categories = st.multiselect("Categories", default_cats, default=default_cats[:3])
    
    current_year = datetime.now().year
    col1, col2 = st.columns(2)
    with col1: start_year = st.number_input("Start Year", 1990, current_year, 2010)
    with col2: end_year = st.number_input("End Year", start_year, current_year, current_year)
    
    max_results = st.slider("Maximum Results", 1, 1000, 100)
    relevance_threshold = st.slider("Relevance Threshold (%)", 0, 100, 40)
    
    # Download settings
    st.subheader("💾 Download Settings")
    auto_download = st.checkbox("Auto-download PDFs", value=not IS_CLOUD, disabled=IS_CLOUD)
    
    max_concurrent = st.slider("Max Concurrent Downloads", 1, 10, 3)
    download_timeout = st.slider("Download Timeout (seconds)", 30, 300, 120)
    
    # Export options
    st.subheader("📤 Export Options")
    export_formats = st.multiselect(
        "Select export formats",
        ["ZIP Archive", "CSV", "JSON", "Excel", "Parquet", "Database Backup"],
        default=["ZIP Archive", "CSV", "Excel"]
    )
    
    # Action buttons
    st.subheader("⚡ Actions")
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        search_btn = st.button("🔍 Search arXiv", type="primary", use_container_width=True)
    with col_btn2:
        if st.button("🔄 Reset Session", use_container_width=True):
            for key in list(st.session_state.keys()):
                if key not in ["page_config_set", "log_buffer"]:
                    st.session_state[key] = DEFAULT_STATE[key]
            key_manager.clear_all_keys()
            st.rerun()
    
    # Database search
    st.subheader("🔍 Database Search")
    db_query = st.text_input("Search stored papers", placeholder="e.g., d33 coefficient, beta phase")
    search_scope = st.selectbox("Search Scope", ["Title & Abstract", "Full Text", "Metadata"])
    if st.button("Search Database", use_container_width=True) and db_query:
        with st.spinner("Searching database..."):
            if search_scope == "Full Text":
                conn = sqlite3.connect(db_manager.universe_db)
                c = conn.cursor()
                c.execute("""SELECT paper_id, title, snippet(papers_fts, 2, '<b>', '</b>', '...', 50)
                         FROM papers_fts WHERE papers_fts MATCH ? LIMIT 20""", (db_query,))
                results = c.fetchall()
                conn.close()
                
                if results:
                    st.success(f"Found {len(results)} papers with full text matches")
                    for paper_id, title, snippet in results:
                        with st.expander(f"{title[:100]}..."):
                            st.write(f"**ID:** {paper_id}")
                            st.markdown(f"**Snippet:** {snippet}")
                else:
                    st.warning("No results found in full text")
            else:
                # Search in metadata
                conn = sqlite3.connect(db_manager.metadata_db)
                c = conn.cursor()
                c.execute("""SELECT id, title, abstract FROM papers 
                         WHERE title LIKE ? OR abstract LIKE ? OR authors LIKE ?
                         LIMIT 20""", (f"%{db_query}%", f"%{db_query}%", f"%{db_query}%"))
                results = c.fetchall()
                conn.close()
                
                if results:
                    st.success(f"Found {len(results)} papers in metadata")
                    for paper_id, title, abstract in results:
                        with st.expander(f"{title[:100]}..."):
                            st.write(f"**ID:** {paper_id}")
                            st.markdown(f"**Abstract:** {abstract[:200]}...")
                else:
                    st.warning("No results found in metadata")

# Show logs once at the top with proper key management
show_logs(expanded=False, namespace="header")

create_dashboard()

# Main content area with tabs
tab1, tab2, tab3 = st.tabs(["🔍 Search & Batch Download", "🗄️ Database Management", "📊 Analytics"])

with tab1:
    st.header("Academic Paper Search & Batch Operations")
    
    if search_btn:
        if not query.strip():
            st.error("Please enter a search query")
            st.stop()
        if not categories:
            st.error("Please select at least one category")
            st.stop()
        
        st.session_state.processing = True
        start_time = time.time()
        
        with st.spinner("🔍 Searching arXiv database..."):
            papers = query_arxiv(query, categories, max_results, start_year, end_year)
        
        if not papers:
            st.warning("No papers found matching your criteria")
            st.session_state.processing = False
            st.stop()
        
        # Apply relevance threshold
        relevant_papers = [p for p in papers if p['relevance_score'] >= relevance_threshold]
        
        if not relevant_papers:
            st.warning(f"No papers above {relevance_threshold}% relevance threshold")
            st.session_state.processing = False
            st.stop()
        
        st.success(f"Found **{len(relevant_papers)}** relevant papers")
        
        # Auto-download if enabled
        if auto_download and not IS_CLOUD:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                with concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrent) as executor:
                    futures = {executor.submit(handle_paper_download, paper): i for i, paper in enumerate(relevant_papers)}
                    completed = 0
                    
                    for future in concurrent.futures.as_completed(futures):
                        idx = futures[future]
                        paper = future.result()
                        relevant_papers[idx] = paper
                        
                        completed += 1
                        progress = completed / len(relevant_papers)
                        progress_bar.progress(progress)
                        status_text.text(f"Processed {completed}/{len(relevant_papers)} papers ({progress:.1%})")
                        
                        # Periodic memory cleanup
                        if completed % 5 == 0:
                            cleanup_memory()
            
            except Exception as e:
                logger.error(f"Bulk download error: {e}")
                st.error(f"Error during bulk download: {e}")
            finally:
                progress_bar.empty()
                status_text.empty()
        
        st.session_state.relevant_papers = relevant_papers
        st.session_state.processing_time = time.time() - start_time
        
        update_log(f"Search completed in {st.session_state.processing_time:.1f} seconds")
        st.rerun()
    
    # Display results if available
    if st.session_state.get('relevant_papers'):
        papers = st.session_state.relevant_papers
        
        # Batch download section
        st.subheader("⚡ Bulk Operations")
        
        col_bulk1, col_bulk2, col_bulk3 = st.columns(3)
        
        with col_bulk1:
            if st.button("📦 Download All PDFs", type="primary", use_container_width=True,
                         key=key_manager.generate_unique_key("bulk_download_all", namespace="search")):
                if not papers:
                    st.warning("No papers to download")
                else:
                    st.session_state.bulk_download_progress = 0
                    st.session_state.bulk_download_status = "Starting download..."
                    st.session_state.bulk_download_complete = False
                    
                    # Create progress containers
                    progress_container = st.empty()
                    status_container = st.empty()
                    time_container = st.empty()
                    
                    start_time = time.time()
                    
                    try:
                        # Download papers that aren't already downloaded
                        papers_to_download = []
                        for paper in papers:
                            paper_id = paper['id']
                            if (paper_id not in st.session_state.downloaded_pdfs and 
                                not paper.get('pdf_stored')):
                                papers_to_download.append(paper)
                        
                        if papers_to_download:
                            st.info(f"Downloading {len(papers_to_download)} new PDFs...")
                            
                            # Reset namespace for this operation
                            key_manager.reset_namespace("bulk_download")
                            
                            with concurrent.futures.ThreadPoolExecutor(
                                max_workers=max_concurrent
                            ) as executor:
                                futures = {executor.submit(handle_paper_download, paper): i 
                                         for i, paper in enumerate(papers_to_download)}
                                
                                completed = 0
                                for future in concurrent.futures.as_completed(futures):
                                    idx = futures[future]
                                    paper = future.result()
                                    papers_to_download[idx] = paper
                                    
                                    completed += 1
                                    progress = completed / len(papers_to_download)
                                    elapsed = time.time() - start_time
                                    eta = (elapsed / completed) * (len(papers_to_download) - completed) if completed > 0 else 0
                                    
                                    progress_container.progress(progress)
                                    status_container.text(f"Downloading {completed}/{len(papers_to_download)} papers")
                                    time_container.text(f"Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s")
                                    
                                    # Update session state
                                    st.session_state.bulk_download_progress = progress
                                    st.session_state.bulk_download_status = f"Downloaded {completed}/{len(papers_to_download)}"
                                    
                                    # Periodic memory cleanup
                                    if completed % 3 == 0:
                                        cleanup_memory()
                        
                        # Create ZIP of all papers
                        st.info("Creating ZIP archive...")
                        def progress_callback(progress, message):
                            progress_container.progress(progress)
                            status_container.text(message)
                            elapsed = time.time() - start_time
                            time_container.text(f"Elapsed: {elapsed:.1f}s | {message}")
                        
                        # Use improved ZIP creation method
                        zip_buffer = db_manager.create_zip_from_papers(
                            papers, 
                            progress_callback=progress_callback,
                            max_concurrent=max_concurrent
                        )
                        
                        if zip_buffer.getbuffer().nbytes > 0:
                            st.session_state.zip_buffer = zip_buffer
                            st.session_state.bulk_download_complete = True
                            total_time = time.time() - start_time
                            st.success(f"✅ Successfully created ZIP with {len(papers)} papers in {total_time:.1f} seconds!")
                        else:
                            st.error("❌ Failed to create ZIP archive")
                    
                    except Exception as e:
                        st.error(f"❌ Bulk download failed: {e}")
                        logger.error(f"Bulk download error: {e}")
                        logger.error(traceback.format_exc())
                    finally:
                        # Clean up progress containers
                        progress_container.empty()
                        status_container.empty()
                        time_container.empty()
                        cleanup_memory()
        
        with col_bulk2:
            # Single-click download button for all papers
            if st.button("🚀 One-Click Download All", type="secondary", use_container_width=True,
                         key=key_manager.generate_unique_key("one_click_download", namespace="search")):
                if papers:
                    st.session_state.download_queue = papers.copy()
                    st.info(f"Queueing {len(papers)} papers for download...")
                    
                    # Create ZIP directly without storing in session state
                    progress_container = st.empty()
                    status_container = st.empty()
                    
                    def progress_callback(progress, message):
                        progress_container.progress(progress)
                        status_container.text(message)
                    
                    zip_buffer = db_manager.create_zip_from_papers(
                        papers,
                        progress_callback=progress_callback,
                        max_concurrent=max_concurrent
                    )
                    
                    if zip_buffer.getbuffer().nbytes > 0:
                        st.session_state.zip_buffer = zip_buffer
                        st.success("✅ One-click download completed!")
                    else:
                        st.error("❌ One-click download failed")
                    
                    progress_container.empty()
                    status_container.empty()
        
        with col_bulk3:
            # Universe database download
            if st.button("🗄️ Download Universe DB", use_container_width=True,
                         key=key_manager.generate_unique_key("download_universe_db", namespace="search")):
                with st.spinner("Preparing universe database for download..."):
                    try:
                        # Create backup filename
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        backup_filename = f"pvdf_universe_db_{timestamp}.db"
                        
                        # Stream the database file
                        db_stream = db_manager.stream_database_backup(UNIVERSE_DB)
                        
                        # Create a BytesIO buffer to hold the streamed content
                        db_buffer = io.BytesIO()
                        for chunk in db_stream:
                            db_buffer.write(chunk)
                        
                        db_buffer.seek(0)
                        
                        if db_buffer.getbuffer().nbytes > 0:
                            st.session_state.universe_db_buffer = db_buffer
                            st.success(f"✅ Universe database prepared for download ({db_buffer.getbuffer().nbytes/1024/1024:.1f}MB)")
                        else:
                            st.error("❌ Failed to prepare universe database")
                    
                    except Exception as e:
                        st.error(f"❌ Error preparing universe database: {e}")
                        logger.error(f"Universe DB download error: {e}")
        
        # Show download buttons if available
        if st.session_state.zip_buffer and st.session_state.bulk_download_complete:
            st.subheader("📥 Download Results")
            col1, col2 = st.columns(2)
            
            with col1:
                zip_size = st.session_state.zip_buffer.getbuffer().nbytes / 1024 / 1024
                st.download_button(
                    label=f"⬇️ Download All Papers ZIP ({zip_size:.1f}MB)",
                    data=st.session_state.zip_buffer.getvalue(),
                    file_name=f"pvdf_piezoelectricity_papers_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                    mime="application/zip",
                    use_container_width=True,
                    key=key_manager.generate_unique_key("download_zip", namespace="search")
                )
            
            with col2:
                if st.session_state.universe_db_buffer:
                    db_size = st.session_state.universe_db_buffer.getbuffer().nbytes / 1024 / 1024
                    st.download_button(
                        label=f"⬇️ Download Universe DB ({db_size:.1f}MB)",
                        data=st.session_state.universe_db_buffer.getvalue(),
                        file_name=f"pvdf_universe_db_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db",
                        mime="application/octet-stream",
                        use_container_width=True,
                        key=key_manager.generate_unique_key("download_universe_db_final", namespace="search")
                    )
                else:
                    st.info("Click 'Download Universe DB' above to prepare the database for download")
        
        # Export metadata
        if st.button("📊 Export Metadata", use_container_width=True,
                     key=key_manager.generate_unique_key("export_metadata", namespace="search")):
            with st.spinner("Exporting metadata..."):
                try:
                    export_buffers = {}
                    for format in export_formats:
                        if format in ["CSV", "JSON", "Excel", "Parquet"]:
                            format_name = format.lower()
                            export_buffers[format_name] = db_manager.export_metadata(
                                format_name, 
                                papers
                            )
                    
                    st.session_state.export_buffers = export_buffers
                    st.success("Metadata exported successfully!")
                except Exception as e:
                    st.error(f"Export failed: {e}")
                    logger.error(f"Metadata export error: {e}")
        
        # Show export buttons if available
        if hasattr(st.session_state, 'export_buffers') and st.session_state.export_buffers:
            st.subheader("📥 Metadata Downloads")
            cols = st.columns(len(st.session_state.export_buffers))
            for i, (format_name, buffer) in enumerate(st.session_state.export_buffers.items()):
                with cols[i]:
                    if format_name == "csv":
                        st.download_button(
                            label="⬇️ Download CSV",
                            data=buffer.getvalue(),
                            file_name=f"pvdf_metadata_{datetime.now().strftime('%Y%m%d')}.csv",
                            mime="text/csv",
                            use_container_width=True,
                            key=key_manager.generate_unique_key(f"download_csv_{i}", namespace="search")
                        )
                    elif format_name == "json":
                        st.download_button(
                            label="⬇️ Download JSON",
                            data=buffer.getvalue(),
                            file_name=f"pvdf_metadata_{datetime.now().strftime('%Y%m%d')}.json",
                            mime="application/json",
                            use_container_width=True,
                            key=key_manager.generate_unique_key(f"download_json_{i}", namespace="search")
                        )
                    elif format_name == "excel":
                        st.download_button(
                            label="⬇️ Download Excel",
                            data=buffer.getvalue(),
                            file_name=f"pvdf_metadata_{datetime.now().strftime('%Y%m%d')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True,
                            key=key_manager.generate_unique_key(f"download_excel_{i}", namespace="search")
                        )
                    elif format_name == "parquet":
                        st.download_button(
                            label="⬇️ Download Parquet",
                            data=buffer.getvalue(),
                            file_name=f"pvdf_metadata_{datetime.now().strftime('%Y%m%d')}.parquet",
                            mime="application/octet-stream",
                            use_container_width=True,
                            key=key_manager.generate_unique_key(f"download_parquet_{i}", namespace="search")
                        )
        
        # Results display
        st.subheader(f"📄 Search Results ({len(papers)} papers)")
        
        # Filter and sort options
        col_filter1, col_filter2, col_filter3 = st.columns(3)
        
        with col_filter1:
            year_range = st.slider(
                "Year Range",
                min_value=min(p['year'] for p in papers),
                max_value=max(p['year'] for p in papers),
                value=(min(p['year'] for p in papers), max(p['year'] for p in papers)),
                key=key_manager.generate_unique_key("year_slider", namespace="search")
            )
        
        with col_filter2:
            min_relevance = st.slider(
                "Min Relevance Score",
                0, 100, 0,
                key=key_manager.generate_unique_key("relevance_slider", namespace="search")
            )
        
        with col_filter3:
            sort_by = st.selectbox(
                "Sort By",
                ["Enhanced Relevance", "Year (Newest)", "Year (Oldest)", "Title"],
                index=0,
                key=key_manager.generate_unique_key("sort_select", namespace="search")
            )
        
        # Apply filters and sorting
        filtered_papers = papers.copy()
        
        filtered_papers = [p for p in filtered_papers if year_range[0] <= p['year'] <= year_range[1]]
        filtered_papers = [p for p in filtered_papers if p['enhanced_relevance_score'] >= min_relevance]
        
        if sort_by == "Enhanced Relevance":
            filtered_papers.sort(key=lambda x: x['enhanced_relevance_score'], reverse=True)
        elif sort_by == "Year (Newest)":
            filtered_papers.sort(key=lambda x: x['year'], reverse=True)
        elif sort_by == "Year (Oldest)":
            filtered_papers.sort(key=lambda x: x['year'])
        elif sort_by == "Title":
            filtered_papers.sort(key=lambda x: x['title'].lower())
        
        st.info(f"Showing {len(filtered_papers)} papers after filtering")
        
        # Display papers with proper widget keys
        key_manager.reset_namespace("papers_display")
        
        for i, paper in enumerate(filtered_papers):
            enhanced = paper.get('enhanced_relevance_score', 0)
            dopant = "🟢" if paper.get('dopant_present') else "⚪"
            beta = "🔵" if paper.get('beta_phase_present') else "⚪"
            
            namespace = f"paper_{i}_{paper['id']}"
            key_manager.reset_namespace(namespace)
            
            with st.expander(f"**{paper['title']}** ({paper['year']}) - Basic: {paper['relevance_score']}% | Enhanced: {enhanced:.1f}% {dopant}{beta}", expanded=i < 3):
                col_info, col_actions = st.columns([3, 1])
                
                with col_info:
                    st.write(f"**Authors:** {paper['authors']}")
                    st.write(f"**Categories:** {paper['categories']}")
                    st.write(f"**Matched Terms:** {paper['matched_terms']}")
                    st.write(f"**Status:** {paper['download_status']}")
                    
                    # Use unique key for toggle
                    show_abstract = st.toggle("Show Abstract", 
                                             key=key_manager.generate_unique_key("toggle_abstract", namespace=namespace))
                    if show_abstract:
                        st.markdown(f"> {paper['abstract']}")
                
                with col_actions:
                    paper_id = paper['id']
                    
                    # Show download status and buttons with unique keys
                    if paper_id in st.session_state.downloaded_pdfs or paper.get('pdf_stored'):
                        pdf_bytes = None
                        if paper_id in st.session_state.downloaded_pdfs:
                            pdf_bytes = st.session_state.downloaded_pdfs[paper_id]['pdf_bytes']
                        else:
                            pdf_bytes = db_manager.get_pdf(paper_id)
                        
                        if pdf_bytes:
                            safe_title = re.sub(r'[^\w\s-]', '', paper['title'])[:50]
                            filename = f"{paper_id}_{safe_title}.pdf".replace(' ', '_')
                            st.download_button(
                                label="📥 Download PDF",
                                data=pdf_bytes,
                                file_name=filename,
                                mime="application/pdf",
                                key=key_manager.generate_unique_key("dl_single_pdf", namespace=namespace),
                                use_container_width=True
                            )
                    else:
                        if st.button("⬇️ Download Now", 
                                   key=key_manager.generate_unique_key("manual_download", namespace=namespace), 
                                   use_container_width=True):
                            with st.spinner("Downloading..."):
                                updated_paper = handle_paper_download(paper, manual_download=True)
                                # Update the specific paper in the list
                                for j, p in enumerate(filtered_papers):
                                    if p['id'] == paper_id:
                                        filtered_papers[j] = updated_paper
                                        break
                                st.rerun()
                    
                    st.markdown(f"[🌐 arXiv Page]({paper['pdf_url'].replace('/pdf/', '/abs/')})")
                    st.markdown(f"[📄 Direct PDF]({paper['pdf_url']})")

with tab2:
    st.header("🗄️ Database Management")
    
    col_stats, col_clean, col_backup = st.columns(3)
    
    with col_stats:
        if st.button("🔄 Refresh Statistics", use_container_width=True,
                     key=key_manager.generate_unique_key("refresh_stats", namespace="database")):
            st.session_state.db_stats = db_manager.get_db_stats()
            st.rerun()
    
    with col_clean:
        if st.button("🧹 Clean Temporary Files", use_container_width=True,
                     key=key_manager.generate_unique_key("clean_temp", namespace="database")):
            temp_dir = os.path.join(DB_DIR, "temp")
            cleaned_count = 0
            if os.path.exists(temp_dir):
                for file in os.listdir(temp_dir):
                    try:
                        os.remove(os.path.join(temp_dir, file))
                        cleaned_count += 1
                    except:
                        pass
            st.session_state.temp_files = []
            st.success(f"🧹 Cleaned {cleaned_count} temporary files")
    
    with col_backup:
        if st.button("💾 Create Full Backup", use_container_width=True,
                     key=key_manager.generate_unique_key("create_backup", namespace="database")):
            with st.spinner("Creating database backup..."):
                backup_dir = os.path.join(DB_DIR, "backups")
                os.makedirs(backup_dir, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                backup_files = []
                for db_name, db_path in [
                    ("Metadata", METADATA_DB),
                    ("Fulltext", UNIVERSE_DB), 
                    ("PDF Storage", PDF_STORAGE_DB),
                    ("Analytics", ANALYTICS_DB)
                ]:
                    if os.path.exists(db_path):
                        backup_name = f"{db_name.lower()}_db_{timestamp}.db"
                        backup_path = os.path.join(backup_dir, backup_name)
                        try:
                            import shutil
                            shutil.copy2(db_path, backup_path)
                            backup_files.append(backup_name)
                        except Exception as e:
                            logger.error(f"Backup failed for {db_path}: {e}")
                
                if backup_files:
                    st.success(f"✅ Created backups: {', '.join(backup_files)}")
                else:
                    st.warning("⚠️ No databases were backed up")
    
    # Show statistics
    if st.session_state.db_stats:
        with st.expander("📊 Detailed Database Statistics"):
            stats = st.session_state.db_stats
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.subheader("Storage Metrics")
                st.metric("Total Papers", stats['total_papers'])
                st.metric("PDFs Stored", stats['pdfs_stored'])
                st.metric("Full Text Papers", stats['fulltext_stored'])
                st.metric("Total PDF Size", f"{stats['total_pdf_size_mb']:.1f} MB")
            
            with col2:
                st.subheader("Content Analysis")
                st.metric("Dopant Papers", stats['dopant_papers'])
                st.metric("Beta Phase Papers", stats['beta_phase_papers'])
                st.metric("Average Relevance", f"{stats['avg_relevance_score']}%")
                st.metric("Average Words/Paper", stats['avg_words_per_paper'])
            
            with col3:
                st.subheader("Performance Metrics")
                st.metric("Average PDF Size", f"{stats['avg_pdf_size_kb']:.1f} KB")
                st.metric("Average Pages/PDF", stats['avg_pages_per_pdf'])
                st.metric("Years Covered", stats['years_covered'])
                st.metric("Average Papers/Year", stats['avg_papers_per_year'])
    
    # Database browser
    st.subheader("Database Browser")
    
    db_to_browse = st.selectbox(
        "Select Database to Browse",
        ["Metadata DB", "Full Text DB", "PDF Storage DB", "Analytics DB"],
        key=key_manager.generate_unique_key("db_select", namespace="database")
    )
    
    rows_to_show = st.slider("Number of rows to display", 5, 100, 20,
                           key=key_manager.generate_unique_key("rows_slider", namespace="database"))
    
    if st.button("🔍 Browse Database", use_container_width=True,
                 key=key_manager.generate_unique_key("browse_db", namespace="database")):
        try:
            if db_to_browse == "Metadata DB":
                conn = sqlite3.connect(db_manager.metadata_db)
                df = pd.read_sql_query(f"SELECT id, title, authors, year, enhanced_relevance_score, dopant_present, beta_phase_present FROM papers LIMIT {rows_to_show}", conn)
            elif db_to_browse == "Full Text DB":
                conn = sqlite3.connect(db_manager.universe_db)
                df = pd.read_sql_query(f"SELECT paper_id, title, word_count, page_count FROM papers_fulltext LIMIT {rows_to_show}", conn)
            elif db_to_browse == "PDF Storage DB":
                conn = sqlite3.connect(db_manager.pdf_db)
                df = pd.read_sql_query(f"SELECT paper_id, file_size, page_count, stored_at FROM pdf_storage LIMIT {rows_to_show}", conn)
            elif db_to_browse == "Analytics DB":
                conn = sqlite3.connect(db_manager.analytics_db)
                df = pd.read_sql_query(f"SELECT * FROM research_trends LIMIT {rows_to_show}", conn)
            
            st.dataframe(df, use_container_width=True)
            conn.close()
        except Exception as e:
            st.error(f"Error browsing database: {e}")

with tab3:
    st.header("📊 Research Analytics")
    
    if not st.session_state.get('relevant_papers'):
        st.info("Perform a search first to see analytics")
        st.stop()
    
    papers = st.session_state.relevant_papers
    
    # Relevance distribution
    st.subheader("📈 Relevance Score Distribution")
    
    scores = [paper['enhanced_relevance_score'] for paper in papers]
    years = [paper['year'] for paper in papers]
    
    fig = px.histogram(
        x=scores,
        nbins=20,
        title='Distribution of Enhanced Relevance Scores',
        labels={'x': 'Enhanced Relevance Score (%)', 'y': 'Number of Papers'},
        color_discrete_sequence=['#636EFA']
    )
    
    fig.update_layout(
        xaxis_title='Enhanced Relevance Score (%)',
        yaxis_title='Number of Papers',
        showlegend=False,
        hovermode='x unified'
    )
    
    # Add vertical line for mean
    mean_score = np.mean(scores)
    fig.add_vline(x=mean_score, line_dash="dash", line_color="red",
                 annotation_text=f'Mean: {mean_score:.1f}%', annotation_position="top right")
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Yearly trends
    st.subheader("🗓️ Yearly Research Trends")
    
    # Group by year
    yearly_data = defaultdict(lambda: {'count': 0, 'total_score': 0.0, 'dopant': 0, 'beta': 0})
    for paper in papers:
        year = paper['year']
        yearly_data[year]['count'] += 1
        yearly_data[year]['total_score'] += paper['enhanced_relevance_score']
        if paper['dopant_present']:
            yearly_data[year]['dopant'] += 1
        if paper['beta_phase_present']:
            yearly_data[year]['beta'] += 1
    
    years = sorted(yearly_data.keys())
    counts = [yearly_data[year]['count'] for year in years]
    avg_scores = [yearly_data[year]['total_score'] / yearly_data[year]['count'] for year in years]
    dopant_counts = [yearly_data[year]['dopant'] for year in years]
    beta_counts = [yearly_data[year]['beta'] for year in years]
    
    # Create subplot with two y-axes
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add traces
    fig.add_trace(
        go.Bar(x=years, y=counts, name="Paper Count", marker_color='#636EFA'),
        secondary_y=False,
    )
    
    fig.add_trace(
        go.Scatter(x=years, y=avg_scores, name="Avg Relevance Score", 
                  mode='lines+markers', line=dict(color='#EF553B', width=3)),
        secondary_y=True,
    )
    
    fig.add_trace(
        go.Scatter(x=years, y=dopant_counts, name="Dopant Papers", 
                  mode='lines+markers', line=dict(color='#00CC96', width=2, dash='dot')),
        secondary_y=False,
    )
    
    fig.add_trace(
        go.Scatter(x=years, y=beta_counts, name="Beta Phase Papers", 
                  mode='lines+markers', line=dict(color='#AB63FA', width=2, dash='dash')),
        secondary_y=False,
    )
    
    # Set titles
    fig.update_layout(
        title_text="Yearly Research Trends in PVDF Piezoelectricity",
        hovermode="x unified"
    )
    
    fig.update_xaxes(title_text="Year")
    fig.update_yaxes(title_text="Number of Papers", secondary_y=False)
    fig.update_yaxes(title_text="Average Relevance Score (%)", secondary_y=True)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Topic analysis
    st.subheader("🔍 Topic Analysis")
    
    # Extract keywords from abstracts
    all_abstracts = " ".join([paper['abstract'].lower() for paper in papers])
    
    # Count occurrences of key terms
    term_counts = {}
    for term in list(DOPANT_TERMS) + list(BETA_PHASE_TERMS) + list(PVDF_TERMS):
        term_counts[term] = all_abstracts.count(term)
    
    # Filter and sort
    significant_terms = {term: count for term, count in term_counts.items() if count >= 2}
    sorted_terms = sorted(significant_terms.items(), key=lambda x: x[1], reverse=True)[:20]
    
    if sorted_terms:
        terms_df = pd.DataFrame(sorted_terms, columns=['Term', 'Count'])
        
        fig = px.bar(terms_df, x='Term', y='Count', 
                     title='Most Frequent Research Terms',
                     color='Count', color_continuous_scale='Viridis')
        
        fig.update_layout(xaxis_tickangle=45, height=400)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Not enough data for topic analysis")

# Footer with system information
st.divider()
st.caption(f"""
**Piezoelectricity in PVDF Research Hub** | 
Running on {'☁️ Streamlit Cloud' if IS_CLOUD else '💻 Local Machine'} | 
Memory Usage: {st.session_state.memory_usage.get('rss_mb', 0):.1f} MB | 
Data Directory: `{DB_DIR}` | 
Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 
Version: 3.0.0
""")

# Show logs at the bottom
show_logs(expanded=False, namespace="footer")

# Periodic cleanup
if random.random() < 0.1:  # 10% chance to run cleanup
    cleanup_memory()

# Register cleanup function for app exit
import atexit
def cleanup_on_exit():
    """Clean up resources when app exits."""
    cleanup_memory(force=True)
    logger.info("Application cleanup completed on exit")

atexit.register(cleanup_on_exit)
