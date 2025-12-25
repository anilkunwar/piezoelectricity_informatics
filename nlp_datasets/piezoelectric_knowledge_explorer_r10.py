# --------------------------------------------------------------
#  Piezoelectricity in PVDF – ENHANCED VERSION
#  ✅ Single-click bulk PDF download
#  ✅ Fixed Streamlit duplicate key error
#  ✅ Memory-optimized large file handling
#  ✅ Advanced visualization and analytics
#  ✅ Cloud-optimized storage management
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
from typing import List, Dict, Any, Optional, Tuple, Union, Generator
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from tenacity import retry, stop_after_attempt, wait_fixed, wait_exponential, retry_if_exception_type
from numba import njit, prange
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx
from collections import Counter, defaultdict
import textwrap
import math
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.spatial.distance import cosine
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image
import base64
import json
import csv
import xml.etree.ElementTree as ET
from io import BytesIO
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='streamlit')

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
            'Get Help': 'https://github.com/yourusername/pvdf-research',
            'Report a bug': "https://github.com/yourusername/pvdf-research/issues",
            'About': "Advanced research tool for PVDF piezoelectricity studies"
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
    'quantum dot', 'mxene', 'boron nitride', 'cellulose', 'halloysite', 'r-go', 'reduced graphene oxide'
}

BETA_PHASE_TERMS = {
    'beta phase', 'β-phase', 'beta-phase', '1270 cm⁻¹', '1275 cm-1', '1270cm⁻¹', '1275cm-1',
    '840 cm⁻¹', '840 cm-1', '840cm⁻¹', '840cm-1', 'ftir beta', 'fraction beta', 'beta content',
    'beta-phase content', 'phase fraction', 'beta polymorph', 'pvdf beta', 'β polymorph',
    'stretching ratio', 'poling', 'electrospinning', 'mechanical stretching', 'annealing temperature'
}

PVDF_TERMS = {'pvdf', 'polyvinylidene fluoride', 'poly(vinylidene fluoride)', 'pvdf polymer', 'fluro polymer'}

PIEZOELECTRIC_TERMS = {
    'piezoelectric', 'piezoelectricity', 'd33 coefficient', 'd31 coefficient', 'voltage output',
    'current density', 'power density', 'energy harvesting', 'nanogenerator', 'sensor', 'actuator',
    'electromechanical coupling', 'piezoelectric constant', 'output voltage', 'output current'
}

# -------------------------- CONFIG --------------------------
if IS_CLOUD:
    DB_DIR = "/tmp/pvdf_research"
    st.info("🌐 Running on Streamlit Cloud: Using temporary storage")
else:
    DB_DIR = os.path.join(os.path.expanduser("~"), "Desktop", "piezoelectricity_data")
    os.makedirs(DB_DIR, exist_ok=True)

METADATA_DB = os.path.join(DB_DIR, "piezoelectricity_metadata.db")
UNIVERSE_DB = os.path.join(DB_DIR, "piezoelectricity_universe.db")
PDF_STORAGE_DB = os.path.join(DB_DIR, "piezoelectricity_pdfs.db")
ANALYTICS_DB = os.path.join(DB_DIR, "piezoelectricity_analytics.db")

TEMP_DIR = os.path.join(DB_DIR, "temp")
os.makedirs(TEMP_DIR, exist_ok=True)

DOWNLOAD_DIR = os.path.join(DB_DIR, "downloads")
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

# Configure logging with rotation
log_file = os.path.join(DB_DIR, "piezoelectricity_query.log")
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='a'
)

# Set up matplotlib style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# -------------------------- SESSION STATE INITIALIZATION --------------------------
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
    "bulk_download_progress": 0,
    "bulk_download_status": "",
    "bulk_download_complete": False,
    "current_view": "search",
    "selected_paper_id": None,
    "show_advanced_filters": False,
    "user_preferences": {
        "auto_download": not IS_CLOUD,
        "max_concurrent_downloads": 3,
        "download_timeout": 60,
        "chunk_size": 8192,
        "memory_threshold_mb": 512
    },
    "analytics_data": {},
    "search_history": [],
    "visualization_cache": {},
    "paper_clusters": None,
    "wordcloud_cache": {},
    "export_queue": [],
    "memory_usage": {},
    "last_cleanup": None
}

for k, v in DEFAULT_STATE.items():
    if k not in st.session_state:
        st.session_state[k] = v

def update_log(message: str):
    """Update log with timestamped message and rotate buffer."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = f"[{timestamp}] {message}"
    st.session_state.log_buffer.append(entry)
    if len(st.session_state.log_buffer) > 100:
        st.session_state.log_buffer.pop(0)
    logging.info(message)
    return entry

def cleanup_memory(force: bool = False):
    """Clean up memory by clearing caches and removing temporary files."""
    try:
        if force or (datetime.now() - st.session_state.get('last_cleanup', datetime.now())).total_seconds() > 300:
            # Clear visualization caches
            st.session_state.visualization_cache.clear()
            st.session_state.wordcloud_cache.clear()
            
            # Clear export queue
            st.session_state.export_queue.clear()
            
            # Remove temp files
            for temp_file in st.session_state.temp_files[:]:
                try:
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                        st.session_state.temp_files.remove(temp_file)
                except Exception as e:
                    update_log(f"Error removing temp file {temp_file}: {e}")
            
            # Force garbage collection
            gc.collect()
            
            # Update memory usage stats
            process = psutil.Process(os.getpid())
            st.session_state.memory_usage = {
                'rss_mb': process.memory_info().rss / 1024 / 1024,
                'vms_mb': process.memory_info().vms / 1024 / 1024,
                'last_update': datetime.now()
            }
            
            st.session_state.last_cleanup = datetime.now()
            update_log("Memory cleanup completed successfully")
    except Exception as e:
        update_log(f"Error during memory cleanup: {e}")

# -------------------------- NUMBA OPTIMIZED SCORERS --------------------------
@njit(parallel=True)
def compute_relevance_score_numba(
    dopant_hits: np.ndarray,
    beta_hits: np.ndarray,
    pvdf_hits: np.ndarray,
    piezo_hits: np.ndarray,
    weights: np.ndarray
) -> float:
    """Numba-optimized relevance scoring with parallel execution."""
    score = (
        weights[0] * np.sum(dopant_hits) +
        weights[1] * np.sum(beta_hits) +
        weights[2] * np.sum(pvdf_hits) +
        weights[3] * np.sum(piezo_hits)
    )
    max_possible = weights.sum() * 4.0
    return min(100.0, (score / (max_possible + 1e-8)) * 100.0)

@njit
def fast_text_matching(text_lower: str, terms: set) -> bool:
    """Fast text matching using Numba (placeholder for string operations)."""
    # Numba doesn't support set operations on strings well, so this is a placeholder
    # Actual implementation would be in Python
    return False

def analyze_dopant_beta_relevance(text: str, title: str = "", abstract: str = "") -> Dict[str, Any]:
    """Enhanced analysis with piezoelectric terms and context awareness."""
    text_lower = text.lower()
    title_lower = title.lower() if title else ""
    abstract_lower = abstract.lower() if abstract else ""
    
    # Context-aware term detection
    has_dopant = any(term in text_lower for term in DOPANT_TERMS)
    has_beta = any(term in text_lower for term in BETA_PHASE_TERMS)
    has_pvdf = any(term in text_lower for term in PVDF_TERMS)
    has_piezo = any(term in text_lower for term in PIEZOELECTRIC_TERMS)
    
    # Boost score if terms appear in title or abstract
    title_boost = 0.2 if (has_dopant and any(term in title_lower for term in DOPANT_TERMS)) else 0
    title_boost += 0.2 if (has_beta and any(term in title_lower for term in BETA_PHASE_TERMS)) else 0
    title_boost += 0.1 if (has_pvdf and any(term in title_lower for term in PVDF_TERMS)) else 0
    
    abstract_boost = 0.1 if (has_dopant and any(term in abstract_lower for term in DOPANT_TERMS)) else 0
    abstract_boost += 0.1 if (has_beta and any(term in abstract_lower for term in BETA_PHASE_TERMS)) else 0
    
    # Convert to arrays for Numba
    dopant_arr = np.array([1 if has_dopant else 0], dtype=np.int8)
    beta_arr = np.array([1 if has_beta else 0], dtype=np.int8)
    pvdf_arr = np.array([1 if has_pvdf else 0], dtype=np.int8)
    piezo_arr = np.array([1 if has_piezo else 0], dtype=np.int8)
    
    weights = np.array([0.4, 0.35, 0.15, 0.1], dtype=np.float32)  # Updated weights
    
    base_score = compute_relevance_score_numba(
        dopant_arr, beta_arr, pvdf_arr, piezo_arr, weights
    )
    
    # Apply boosts
    enhanced_score = min(100.0, base_score * (1 + title_boost + abstract_boost))
    
    return {
        "dopant_present": bool(has_dopant),
        "beta_phase_present": bool(has_beta),
        "pvdf_present": bool(has_pvdf),
        "piezoelectric_present": bool(has_piezo),
        "enhanced_relevance_score": float(enhanced_score),
        "title_boost": float(title_boost),
        "abstract_boost": float(abstract_boost),
        "base_score": float(base_score)
    }

# -------------------------- DATABASE MANAGER WITH ANALYTICS --------------------------
class DatabaseManager:
    def __init__(self):
        self.metadata_db = METADATA_DB
        self.universe_db = UNIVERSE_DB
        self.pdf_db = PDF_STORAGE_DB
        self.analytics_db = ANALYTICS_DB
        self.init_databases()
        self.init_analytics_database()
        update_log("Database manager initialized with analytics support")

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
            piezoelectric_present BOOLEAN,
            cluster_id INTEGER DEFAULT -1,
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
        c.execute("CREATE INDEX IF NOT EXISTS idx_cluster ON papers(cluster_id)")
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
            extraction_time REAL,
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
            dopant_type TEXT,
            beta_phase_mentions INTEGER,
            piezoelectric_mentions INTEGER,
            avg_relevance_score REAL,
            paper_count INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )""")
        
        # Store author collaboration networks
        c.execute("""CREATE TABLE IF NOT EXISTS author_networks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            author_name TEXT,
            collaborator_name TEXT,
            paper_count INTEGER,
            first_collaboration_year INTEGER,
            last_collaboration_year INTEGER,
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
        
        # Store paper clusters
        c.execute("""CREATE TABLE IF NOT EXISTS paper_clusters (
            cluster_id INTEGER PRIMARY KEY,
            cluster_label TEXT,
            paper_count INTEGER,
            dominant_keywords TEXT,
            avg_year REAL,
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
            c.execute("SELECT COUNT(*) FROM papers WHERE piezoelectric_present = 1")
            stats['piezoelectric_papers'] = c.fetchone()[0]
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

            # Analytics stats
            conn = sqlite3.connect(self.analytics_db)
            c = conn.cursor()
            c.execute("SELECT COUNT(DISTINCT cluster_id) FROM paper_clusters")
            stats['clusters_count'] = c.fetchone()[0] or 0
            c.execute("SELECT COUNT(*) FROM research_trends")
            stats['trends_count'] = c.fetchone()[0] or 0
            c.execute("SELECT COUNT(*) FROM author_networks")
            stats['collaborations_count'] = c.fetchone()[0] or 0
            conn.close()

            # Calculate storage efficiency
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
            update_log(f"Error getting DB stats: {e}")
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
                dopant_present, beta_phase_present, piezoelectric_present,
                cluster_id, word_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
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
                    paper.get('piezoelectric_present', False),
                    paper.get('cluster_id', -1),
                    paper.get('word_count', 0)
                ))
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            update_log(f"Failed to store metadata for {paper.get('id')}: {e}")
            return False

    def store_pdf_data(self, paper_id: str, pdf_bytes: bytes, pdf_url: str) -> bool:
        """Store PDF data with memory optimization and chunking."""
        try:
            pdf_hash = hashlib.sha256(pdf_bytes).hexdigest()
            file_size = len(pdf_bytes)
            
            # Extract page count safely
            page_count = 0
            try:
                with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
                    page_count = len(doc)
            except Exception as e:
                update_log(f"Error extracting page count for {paper_id}: {e}")

            conn = sqlite3.connect(self.pdf_db)
            c = conn.cursor()
            
            # Check if PDF already exists
            c.execute("SELECT paper_id FROM pdf_storage WHERE pdf_hash = ?", (pdf_hash,))
            existing = c.fetchone()
            if existing:
                update_log(f"PDF already exists for {paper_id}, updating reference")
                c.execute("UPDATE papers SET pdf_stored = 1, pdf_size = ? WHERE id = ?", (file_size, paper_id))
                conn.commit()
                conn.close()
                return True

            # Store PDF in chunks if large
            if file_size > 50 * 1024 * 1024:  # 50MB
                update_log(f"Large PDF detected ({file_size/1024/1024:.1f}MB), storing in chunks")
                chunk_size = 10 * 1024 * 1024  # 10MB chunks
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
            
            update_log(f"Stored PDF for {paper_id} ({file_size/1024:.1f} KB, {page_count} pages)")
            return True
        except Exception as e:
            update_log(f"Failed to store PDF for {paper_id}: {e}")
            return False

    def store_fulltext(self, paper_id: str, title: str, abstract: str,
                       full_text: str, page_count: int = 0) -> bool:
        """Store full text with advanced analysis and entity extraction."""
        try:
            start_time = time.time()
            text_hash = hashlib.md5(full_text.encode()).hexdigest()
            word_count = len(full_text.split())
            
            # Enhanced analysis
            analysis = analyze_dopant_beta_relevance(full_text, title, abstract)
            enhanced_score = analysis["enhanced_relevance_score"]
            
            # Entity extraction (simplified)
            entities = self.extract_entities(full_text, paper_id)
            
            conn = sqlite3.connect(self.universe_db)
            c = conn.cursor()
            
            # Store full text
            c.execute("""INSERT OR REPLACE INTO papers_fulltext
                (paper_id, title, abstract, full_text, text_hash, word_count, page_count, extraction_time)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (paper_id, title, abstract, full_text, text_hash, word_count, page_count, time.time() - start_time))
            
            # Update FTS index
            c.execute("""INSERT OR REPLACE INTO papers_fts
                (paper_id, title, abstract, full_text)
                VALUES (?, ?, ?, ?)""",
                (paper_id, title, abstract, full_text))
            
            # Store entities
            if entities:
                entity_records = []
                for entity in entities:
                    entity_records.append((
                        paper_id,
                        entity['type'],
                        entity['text'],
                        entity['context'],
                        entity['page'],
                        entity['confidence']
                    ))
                
                c.executemany("""INSERT OR IGNORE INTO extracted_entities
                    (paper_id, entity_type, entity_text, context, page_number, confidence)
                    VALUES (?, ?, ?, ?, ?, ?)""", entity_records)
            
            conn.commit()
            conn.close()

            # Update metadata with analysis results
            conn = sqlite3.connect(self.metadata_db)
            c = conn.cursor()
            c.execute("""UPDATE papers SET fulltext_stored = 1,
                enhanced_relevance_score = ?,
                dopant_present = ?,
                beta_phase_present = ?,
                piezoelectric_present = ?,
                word_count = ?
                WHERE id = ?""",
                (enhanced_score, analysis["dopant_present"],
                 analysis["beta_phase_present"], analysis["piezoelectric_present"],
                 word_count, paper_id))
            conn.commit()
            conn.close()
            
            extraction_time = time.time() - start_time
            update_log(f"Stored full text for {paper_id} ({word_count} words, score: {enhanced_score:.1f}, time: {extraction_time:.2f}s)")
            return True
        except Exception as e:
            update_log(f"Failed to store full text for {paper_id}: {e}")
            return False

    def extract_entities(self, text: str, paper_id: str) -> List[Dict]:
        """Extract scientific entities from text (simplified version)."""
        entities = []
        
        # Simple pattern matching for demonstration
        dopant_patterns = list(DOPANT_TERMS)
        beta_patterns = list(BETA_PHASE_TERMS)
        piezo_patterns = list(PIEZOELECTRIC_TERMS)
        
        # Find entity occurrences
        for pattern in dopant_patterns:
            if pattern in text.lower():
                entities.append({
                    'type': 'dopant',
                    'text': pattern,
                    'context': text.lower().split(pattern)[0][-50:] + pattern + text.lower().split(pattern)[1][:50],
                    'page': 1,
                    'confidence': 0.8
                })
        
        for pattern in beta_patterns:
            if pattern in text.lower():
                entities.append({
                    'type': 'beta_phase',
                    'text': pattern,
                    'context': text.lower().split(pattern)[0][-50:] + pattern + text.lower().split(pattern)[1][:50],
                    'page': 1,
                    'confidence': 0.85
                })
        
        for pattern in piezo_patterns:
            if pattern in text.lower():
                entities.append({
                    'type': 'piezoelectric',
                    'text': pattern,
                    'context': text.lower().split(pattern)[0][-50:] + pattern + text.lower().split(pattern)[1][:50],
                    'page': 1,
                    'confidence': 0.9
                })
        
        return entities[:10]  # Limit to top 10 entities

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
                pdf_bytes = b''.join(chunk[0] for chunk in chunks if chunk[0])
                update_log(f"Retrieved chunked PDF for {paper_id} ({len(pdf_bytes)/1024/1024:.1f}MB)")
            else:
                # Retrieve single blob
                c.execute("SELECT pdf_data FROM pdf_storage WHERE paper_id = ?", (paper_id,))
                result = c.fetchone()
                pdf_bytes = result[0] if result else None
            
            conn.close()
            return pdf_bytes
        except Exception as e:
            update_log(f"Failed to retrieve PDF for {paper_id}: {e}")
            return None

    def get_paper_info(self, paper_id: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive paper information."""
        try:
            conn = sqlite3.connect(self.metadata_db)
            c = conn.cursor()
            c.execute("""SELECT title, authors, year, abstract, pdf_url,
                relevance_score, enhanced_relevance_score, dopant_present, beta_phase_present,
                piezoelectric_present, download_status, pdf_stored, fulltext_stored,
                pdf_size, word_count, cluster_id
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
                'piezoelectric_present': meta[9],
                'download_status': meta[10],
                'has_pdf': meta[11],
                'has_fulltext': meta[12],
                'pdf_size': meta[13],
                'word_count': meta[14] or (fulltext_info[0] if fulltext_info else 0),
                'page_count': fulltext_info[1] if fulltext_info else 0,
                'cluster_id': meta[15],
                'pdf_bytes': pdf_bytes
            }
        except Exception as e:
            update_log(f"Failed to get paper info for {paper_id}: {e}")
            return None

    def create_zip_from_papers(self, papers: List[Dict], progress_callback=None) -> io.BytesIO:
        """Create ZIP archive from paper list with progress tracking."""
        zip_buffer = io.BytesIO()
        total_papers = len(papers)
        
        try:
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                for i, paper in enumerate(papers):
                    paper_id = paper['id']
                    
                    # Get PDF bytes - check session state first, then database
                    pdf_bytes = None
                    if paper_id in st.session_state.downloaded_pdfs:
                        pdf_bytes = st.session_state.downloaded_pdfs[paper_id].get('pdf_bytes')
                    
                    if pdf_bytes is None:
                        pdf_bytes = self.get_pdf(paper_id)
                    
                    if pdf_bytes is None:
                        # Try to download if not available
                        update_log(f"PDF not found for {paper_id}, attempting download")
                        pdf_bytes = download_pdf_bytes(paper['pdf_url'])
                        if pdf_bytes:
                            # Store in session state for future use
                            st.session_state.downloaded_pdfs[paper_id] = {
                                'pdf_bytes': pdf_bytes,
                                'title': paper['title'],
                                'authors': paper['authors'],
                                'year': paper['year']
                            }
                    
                    if pdf_bytes:
                        # Create clean filename
                        title = re.sub(r'[^\w\s-]', '', paper['title'])[:100]
                        authors = paper['authors'].split(',')[0][:50] if paper['authors'] else 'unknown'
                        filename = f"{paper_id}_{authors}_{paper['year']}_{title}.pdf"
                        filename = re.sub(r'\s+', '_', filename)
                        filename = re.sub(r'_{2,}', '_', filename)  # Remove multiple underscores
                        
                        zip_file.writestr(filename, pdf_bytes)
                        update_log(f"Added {paper_id} to ZIP ({len(pdf_bytes)/1024:.1f} KB)")
                    else:
                        update_log(f"Skipping {paper_id} - no PDF available")
                    
                    # Update progress
                    if progress_callback and total_papers > 1:
                        progress = (i + 1) / total_papers
                        progress_callback(progress, f"Processing {i+1}/{total_papers} papers")
            
            zip_buffer.seek(0)
            update_log(f"Created ZIP with {total_papers} papers")
            return zip_buffer
        except Exception as e:
            update_log(f"Failed to create ZIP: {e}")
            return io.BytesIO()

    def export_metadata(self, format: str = "csv", papers: List[Dict] = None) -> io.BytesIO:
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
                df.to_csv(output, index=False)
            elif format.lower() == "json":
                df.to_json(output, orient="records", indent=2)
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
                table = pa.Table.from_pandas(df)
                pq.write_table(table, output)
            else:
                return io.BytesIO()
            
            output.seek(0)
            conn.close()
            return output
        except Exception as e:
            update_log(f"Export failed: {e}")
            return io.BytesIO()

    def update_paper_cluster(self, paper_id: str, cluster_id: int):
        """Update paper cluster assignment."""
        try:
            conn = sqlite3.connect(self.metadata_db)
            c = conn.cursor()
            c.execute("UPDATE papers SET cluster_id = ? WHERE id = ?", (cluster_id, paper_id))
            conn.commit()
            conn.close()
        except Exception as e:
            update_log(f"Failed to update cluster for {paper_id}: {e}")

    def store_cluster_info(self, cluster_id: int, cluster_label: str, paper_count: int,
                          dominant_keywords: List[str], avg_year: float):
        """Store cluster information in analytics database."""
        try:
            conn = sqlite3.connect(self.analytics_db)
            c = conn.cursor()
            c.execute("""INSERT OR REPLACE INTO paper_clusters
                (cluster_id, cluster_label, paper_count, dominant_keywords, avg_year)
                VALUES (?, ?, ?, ?, ?)""",
                (cluster_id, cluster_label, paper_count, ','.join(dominant_keywords), avg_year))
            conn.commit()
            conn.close()
        except Exception as e:
            update_log(f"Failed to store cluster info: {e}")

# Initialize DB
db_manager = DatabaseManager()

# -------------------------- DOWNLOAD & QUERY FUNCTIONS WITH MEMORY MANAGEMENT --------------------------
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((requests.exceptions.RequestException, requests.exceptions.Timeout))
)
def download_pdf_bytes(pdf_url: str, timeout: int = 30) -> Optional[bytes]:
    """Download PDF with retries and timeout."""
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
        
        response = session.get(pdf_url, headers=headers, timeout=timeout, stream=True)
        response.raise_for_status()
        
        # Stream download to avoid memory issues for large files
        pdf_bytes = b''
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                pdf_bytes += chunk
        
        if len(pdf_bytes) < 1024:
            raise ValueError("PDF file too small")
        
        return pdf_bytes
    except Exception as e:
        update_log(f"Download failed for {pdf_url}: {e}")
        return None

def extract_text_from_bytes(pdf_bytes: bytes, max_pages: int = 50, max_chars: int = 1000000) -> str:
    """Extract text from PDF bytes with limits."""
    try:
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            text = ""
            for page_num in range(min(max_pages, len(doc))):
                text += doc[page_num].get_text()
                if len(text) > max_chars:
                    break
        
        # Clean text
        text = re.sub(r'\s+', ' ', text).strip()
        return text[:max_chars]
    except Exception as e:
        return f"Error extracting text: {str(e)}"

def handle_paper_download(paper: Dict[str, Any], manual_download: bool = False) -> Dict[str, Any]:
    """Handle paper download with comprehensive error handling."""
    paper_id = paper['id']
    if manual_download and paper_id in st.session_state.downloaded_pdfs:
        update_log(f"PDF for {paper_id} already in session")
        return paper
    
    try:
        update_log(f"Downloading PDF for {paper_id} from {paper['pdf_url']}...")
        
        # Check memory before download
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024
        if memory_mb > st.session_state.user_preferences['memory_threshold_mb']:
            cleanup_memory(force=True)
            update_log(f"Memory cleanup performed before download (was {memory_mb:.1f}MB)")
        
        # Download PDF
        pdf_bytes = download_pdf_bytes(
            paper['pdf_url'], 
            timeout=st.session_state.user_preferences['download_timeout']
        )
        
        if pdf_bytes is None:
            paper['download_status'] = "Failed to download"
            update_log(f"❌ Download failed for {paper_id}")
            return paper
        
        # Extract text
        full_text = extract_text_from_bytes(pdf_bytes)
        
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
                paper['piezoelectric_present'] = info['piezoelectric_present']
                paper['word_count'] = info['word_count']
        else:
            update_log(f"Text extraction error for {paper_id}: {full_text}")
        
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
        
        update_log(f"✅ Successfully processed {paper_id} ({len(pdf_bytes)/1024:.1f} KB)")
        return paper
    except Exception as e:
        error_msg = f"Failed to process {paper_id}: {str(e)[:200]}"
        paper['download_status'] = error_msg
        update_log(f"❌ {error_msg}")
        return paper
    finally:
        # Clean up memory
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
                'beta_phase_present': False,
                'piezoelectric_present': False,
                'cluster_id': -1
            }
            
            results.append(paper)
            processed_ids.add(result.entry_id)
            
            if len(results) >= max_results:
                break
        
        # Sort by relevance score
        results.sort(key=lambda x: x['relevance_score'], reverse=True)
        return results
    
    except Exception as e:
        update_log(f"arXiv query failed: {e}")
        return []

# -------------------------- ADVANCED ANALYTICS AND VISUALIZATION --------------------------
def perform_paper_clustering(papers: List[Dict]) -> Tuple[np.ndarray, Dict[int, str]]:
    """Perform K-means clustering on paper abstracts."""
    try:
        if len(papers) < 3:
            return np.array([0] * len(papers)), {0: "All Papers"}
        
        # Extract abstracts and titles
        texts = []
        for paper in papers:
            text = f"{paper['title']} {paper['abstract']}"
            texts.append(text.lower())
        
        # TF-IDF vectorization
        vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        tfidf_matrix = vectorizer.fit_transform(texts)
        
        # Determine optimal number of clusters
        n_clusters = min(5, max(2, len(papers) // 10))
        
        # K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(tfidf_matrix)
        
        # Generate cluster labels based on dominant keywords
        cluster_labels = {}
        feature_names = vectorizer.get_feature_names_out()
        
        for cluster_id in range(n_clusters):
            # Get indices of papers in this cluster
            cluster_indices = np.where(clusters == cluster_id)[0]
            
            # Get top keywords for this cluster
            cluster_tfidf = tfidf_matrix[cluster_indices].mean(axis=0)
            top_indices = np.argsort(cluster_tfidf)[0, -5:].flatten().tolist()[::-1]
            top_keywords = [feature_names[i] for i in top_indices]
            
            # Create label
            label = " ".join(top_keywords[:2]).title()
            cluster_labels[cluster_id] = label
            
            # Update database with cluster assignments
            for idx in cluster_indices:
                paper_id = papers[idx]['id']
                db_manager.update_paper_cluster(paper_id, int(cluster_id))
                db_manager.store_cluster_info(
                    cluster_id=cluster_id,
                    cluster_label=label,
                    paper_count=len(cluster_indices),
                    dominant_keywords=top_keywords,
                    avg_year=sum(papers[i]['year'] for i in cluster_indices) / len(cluster_indices)
                )
        
        return clusters, cluster_labels
    
    except Exception as e:
        update_log(f"Clustering failed: {e}")
        return np.array([0] * len(papers)), {0: "All Papers"}

def generate_wordcloud(text: str, max_words: int = 100) -> Optional[plt.Figure]:
    """Generate word cloud from text."""
    try:
        if not text or len(text.split()) < 10:
            return None
        
        # Clean text
        text = re.sub(r'[^\w\s]', '', text.lower())
        
        # Generate word cloud
        wordcloud = WordCloud(
            width=800, height=400,
            background_color='white',
            max_words=max_words,
            colormap='viridis',
            contour_width=3,
            contour_color='steelblue'
        ).generate(text)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        plt.tight_layout()
        
        return fig
    
    except Exception as e:
        update_log(f"Word cloud generation failed: {e}")
        return None

def create_relevance_distribution_chart(papers: List[Dict]) -> go.Figure:
    """Create interactive relevance distribution chart."""
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
    
    return fig

def create_yearly_trends_chart(papers: List[Dict]) -> go.Figure:
    """Create yearly trends chart for paper counts and average scores."""
    if not papers:
        return go.Figure()
    
    # Group by year
    yearly_data = defaultdict(lambda: {'count': 0, 'total_score': 0.0})
    for paper in papers:
        yearly_data[paper['year']]['count'] += 1
        yearly_data[paper['year']]['total_score'] += paper['enhanced_relevance_score']
    
    years = sorted(yearly_data.keys())
    counts = [yearly_data[year]['count'] for year in years]
    avg_scores = [yearly_data[year]['total_score'] / yearly_data[year]['count'] for year in years]
    
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
    
    # Set titles
    fig.update_layout(
        title_text="Yearly Research Trends",
        hovermode="x unified"
    )
    
    fig.update_xaxes(title_text="Year")
    fig.update_yaxes(title_text="Number of Papers", secondary_y=False)
    fig.update_yaxes(title_text="Average Relevance Score (%)", secondary_y=True)
    
    return fig

# -------------------------- UI COMPONENTS --------------------------
def show_logs(expanded: bool = False):
    """Display logs with unique key to avoid duplicate widget error."""
    if st.session_state.log_buffer:
        with st.expander("📋 Processing Logs", expanded=expanded):
            # Use a unique key based on timestamp to avoid duplicate widget error
            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            st.text_area(
                "Logs", 
                "\n".join(st.session_state.log_buffer[-50:]), 
                height=200, 
                key=f"log_display_{current_time}"
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
    
    # Detailed statistics in expander
    with st.expander("🔍 Detailed Statistics"):
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Content Analysis")
            st.metric("Dopant Papers", stats.get('dopant_papers', 0))
            st.metric("Beta Phase Papers", stats.get('beta_phase_papers', 0))
            st.metric("Piezoelectric Papers", stats.get('piezoelectric_papers', 0))
            st.metric("Avg. Words/Paper", stats.get('avg_words_per_paper', 0))
        
        with col2:
            st.subheader("Performance Metrics")
            st.metric("Avg. PDF Size", f"{stats.get('avg_pdf_size_kb', 0):.1f} KB")
            st.metric("Avg. Pages/PDF", stats.get('avg_pages_per_pdf', 0))
            st.metric("Avg. Papers/Year", stats.get('avg_papers_per_year', 0))
            st.metric("Total Clusters", stats.get('clusters_count', 0))
    
    # Memory usage
    if st.session_state.memory_usage:
        mem_stats = st.session_state.memory_usage
        st.subheader("💡 Memory Usage")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("RSS Memory", f"{mem_stats['rss_mb']:.1f} MB")
        with col2:
            st.metric("Virtual Memory", f"{mem_stats['vms_mb']:.1f} MB")

def show_paper_details(paper: Dict):
    """Show detailed paper information with tabs."""
    paper_id = paper['id']
    
    # Get paper info from database
    info = db_manager.get_paper_info(paper_id)
    if not info:
        st.error("Paper information not available")
        return
    
    # Create tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["📚 Basic Info", "📄 Full Text", "📊 Analysis", "🔍 Entities"])
    
    with tab1:
        st.subheader(f"{paper['title']} ({paper['year']})")
        st.markdown(f"**Authors:** {paper['authors']}")
        st.markdown(f"**Categories:** {paper['categories']}")
        st.markdown(f"**Matched Terms:** {paper['matched_terms']}")
        st.markdown(f"**Status:** {paper['download_status']}")
        st.markdown(f"**Relevance:** Basic: {paper['relevance_score']}% | Enhanced: {paper['enhanced_relevance_score']:.1f}%")
        
        if info['pdf_size'] > 0:
            st.markdown(f"**PDF Size:** {info['pdf_size']/1024:.1f} KB")
            st.markdown(f"**Pages:** {info.get('page_count', 'Unknown')}")
            st.markdown(f"**Word Count:** {info.get('word_count', 'Unknown')}")
        
        # Download buttons
        col1, col2 = st.columns(2)
        with col1:
            if paper_id in st.session_state.downloaded_pdfs or info['has_pdf']:
                pdf_bytes = st.session_state.downloaded_pdfs[paper_id]['pdf_bytes'] if paper_id in st.session_state.downloaded_pdfs else info['pdf_bytes']
                if pdf_bytes:
                    safe_title = re.sub(r'[^\w\s-]', '', paper['title'])[:50]
                    filename = f"{paper_id}_{safe_title}.pdf".replace(' ', '_')
                    st.download_button(
                        label="📥 Download PDF",
                        data=pdf_bytes,
                        file_name=filename,
                        mime="application/pdf",
                        key=f"dl_single_{paper_id}",
                        use_container_width=True
                    )
        
        with col2:
            st.markdown(f"[🌐 arXiv Page]({paper['pdf_url'].replace('/pdf/', '/abs/')})")
            st.markdown(f"[📄 Direct PDF]({paper['pdf_url']})")
        
        # Abstract with toggle
        show_abstract = st.toggle("Show Abstract", key=f"toggle_abstract_{paper_id}")
        if show_abstract:
            st.markdown("**Abstract:**")
            st.markdown(f"> {paper['abstract']}")
    
    with tab2:
        if info['has_fulltext']:
            st.subheader("Full Text Content")
            
            # Show word cloud if cached
            if paper_id in st.session_state.wordcloud_cache:
                st.pyplot(st.session_state.wordcloud_cache[paper_id])
            else:
                with st.spinner("Generating word cloud..."):
                    wordcloud_fig = generate_wordcloud(info['abstract'] + " " + (info.get('full_text', '')[:1000] if info.get('full_text') else ''))
                    if wordcloud_fig:
                        st.session_state.wordcloud_cache[paper_id] = wordcloud_fig
                        st.pyplot(wordcloud_fig)
            
            # Show text content
            show_full_text = st.toggle("Show Full Text", key=f"toggle_fulltext_{paper_id}")
            if show_full_text and info.get('full_text'):
                st.text_area("Full Text", info['full_text'][:5000] + "...", height=400, key=f"fulltext_{paper_id}")
        else:
            st.info("Full text not available. Download the PDF to extract text content.")
    
    with tab3:
        st.subheader("Paper Analysis")
        
        # Relevance breakdown
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Dopant Present", "✅ Yes" if info['dopant_present'] else "❌ No")
        with col2:
            st.metric("Beta Phase Present", "✅ Yes" if info['beta_phase_present'] else "❌ No")
        with col3:
            st.metric("Piezoelectric Present", "✅ Yes" if info['piezoelectric_present'] else "❌ No")
        with col4:
            st.metric("Cluster ID", info['cluster_id'])
        
        # Enhanced relevance details
        st.subheader("Enhanced Relevance Breakdown")
        st.progress(info['enhanced_relevance_score'] / 100)
        st.markdown(f"""
        **Scoring Details:**
        - Base Score: {paper.get('base_score', 0):.1f}%
        - Title Boost: {paper.get('title_boost', 0):.2f}%
        - Abstract Boost: {paper.get('abstract_boost', 0):.2f}%
        """)
    
    with tab4:
        st.subheader("Extracted Entities")
        st.info("Entity extraction feature coming soon in future updates!")

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
        '"SnO2"', '"ZnO"', '"graphene"', '"nanogenerator"', '"energy harvesting"'
    ])
    query = st.text_area("Search Query", value=default_query, height=100)
    
    default_cats = ["cond-mat.mtrl-sci", "physics.app-ph", "cond-mat.soft", "physics.chem-ph"]
    categories = st.multiselect("Categories", default_cats, default=default_cats[:3])
    
    current_year = datetime.now().year
    col1, col2 = st.columns(2)
    with col1: start_year = st.number_input("Start Year", 1990, current_year, 2010)
    with col2: end_year = st.number_input("End Year", start_year, current_year, current_year)
    
    max_results = st.slider("Maximum Results", 1, 1000, 100)
    relevance_threshold = st.slider("Relevance Threshold (%)", 0, 100, 40)
    
    # Download settings
    st.subheader("💾 Download Settings")
    auto_download = st.checkbox("Auto-download PDFs", 
                              value=st.session_state.user_preferences['auto_download'],
                              disabled=IS_CLOUD)
    
    max_concurrent = st.slider("Max Concurrent Downloads", 1, 10, 
                             st.session_state.user_preferences['max_concurrent_downloads'])
    
    # Memory settings
    st.subheader("🧠 Memory Management")
    memory_threshold = st.slider("Memory Threshold (MB)", 128, 2048, 
                               st.session_state.user_preferences['memory_threshold_mb'])
    
    # Save preferences
    if st.button("💾 Save Preferences", use_container_width=True):
        st.session_state.user_preferences.update({
            'auto_download': auto_download,
            'max_concurrent_downloads': max_concurrent,
            'memory_threshold_mb': memory_threshold
        })
        st.success("Preferences saved!")
    
    # Export options
    st.subheader("📤 Export Options")
    export_formats = st.multiselect(
        "Select export formats",
        ["ZIP Archive", "CSV", "JSON", "Excel", "Parquet"],
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
                if key not in ["page_config_set", "log_buffer", "user_preferences"]:
                    del st.session_state[key]
            st.rerun()
    
    # Advanced search
    with st.expander("🔍 Advanced Database Search"):
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

# Show logs once at the top (fixes duplicate key error)
show_logs(expanded=False)

create_dashboard()

# Main content area tabs
main_tab1, main_tab2, main_tab3, main_tab4 = st.tabs(["🔍 Search", "📊 Analytics", "🗄️ Database", "⚙️ Management"])

with main_tab1:
    st.header("Academic Paper Search")
    
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
        
        # Perform clustering
        with st.spinner("Performing paper clustering analysis..."):
            clusters, cluster_labels = perform_paper_clustering(relevant_papers)
            for i, paper in enumerate(relevant_papers):
                paper['cluster_id'] = int(clusters[i])
                paper['cluster_label'] = cluster_labels.get(clusters[i], f"Cluster {clusters[i]}")
        
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
                update_log(f"Bulk download error: {e}")
                st.error(f"Error during bulk download: {e}")
            finally:
                progress_bar.empty()
                status_text.empty()
        
        st.session_state.relevant_papers = relevant_papers
        st.session_state.processing_time = time.time() - start_time
        st.session_state.search_results = papers
        
        update_log(f"Search completed in {st.session_state.processing_time:.1f} seconds")
        st.rerun()
    
    # Display results if available
    if st.session_state.get('relevant_papers'):
        papers = st.session_state.relevant_papers
        
        # Bulk download section with progress tracking
        st.subheader("⚡ Bulk Operations")
        
        col_bulk1, col_bulk2, col_bulk3 = st.columns(3)
        
        with col_bulk1:
            if st.button("📦 Download All PDFs", type="primary", use_container_width=True):
                if not papers:
                    st.warning("No papers to download")
                else:
                    st.session_state.bulk_download_progress = 0
                    st.session_state.bulk_download_status = "Starting download..."
                    st.session_state.bulk_download_complete = False
                    
                    # Create progress bar and status
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
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
                            
                            with concurrent.futures.ThreadPoolExecutor(
                                max_workers=st.session_state.user_preferences['max_concurrent_downloads']
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
                                    progress_bar.progress(progress)
                                    status_text.text(f"Downloading {completed}/{len(papers_to_download)} papers")
                                    
                                    # Update session state
                                    st.session_state.bulk_download_progress = progress
                                    st.session_state.bulk_download_status = f"Downloaded {completed}/{len(papers_to_download)}"
                                    
                                    # Periodic memory cleanup
                                    if completed % 3 == 0:
                                        cleanup_memory()
                        
                        # Create ZIP of all papers
                        st.info("Creating ZIP archive...")
                        def progress_callback(progress, message):
                            progress_bar.progress(progress)
                            status_text.text(message)
                            st.session_state.bulk_download_progress = progress
                            st.session_state.bulk_download_status = message
                        
                        zip_buffer = db_manager.create_zip_from_papers(papers, progress_callback)
                        
                        if zip_buffer.getbuffer().nbytes > 0:
                            st.session_state.zip_buffer = zip_buffer
                            st.session_state.bulk_download_complete = True
                            st.success(f"✅ Successfully created ZIP with {len(papers)} papers!")
                        else:
                            st.error("❌ Failed to create ZIP archive")
                    
                    except Exception as e:
                        st.error(f"❌ Bulk download failed: {e}")
                        update_log(f"Bulk download error: {e}")
                    finally:
                        progress_bar.empty()
                        status_text.empty()
                        cleanup_memory()
        
        with col_bulk2:
            if st.session_state.zip_buffer and st.session_state.bulk_download_complete:
                st.download_button(
                    label="⬇️ Download All Papers ZIP",
                    data=st.session_state.zip_buffer.getvalue(),
                    file_name=f"pvdf_piezoelectricity_papers_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                    mime="application/zip",
                    use_container_width=True,
                    type="primary"
                )
        
        with col_bulk3:
            if st.button("📊 Export Metadata", use_container_width=True):
                with st.spinner("Exporting metadata..."):
                    export_buffers = {}
                    for format in export_formats:
                        if format in ["CSV", "JSON", "Excel", "Parquet"]:
                            format_name = format.lower()
                            export_buffers[format_name] = db_manager.export_metadata(
                                format_name, 
                                papers
                            )
                    
                    st.session_state.export_queue = export_buffers
                    st.success("Metadata exported successfully!")
        
        # Show export buttons if available
        if st.session_state.export_queue:
            st.subheader("📥 Export Downloads")
            cols = st.columns(len(st.session_state.export_queue))
            for i, (format_name, buffer) in enumerate(st.session_state.export_queue.items()):
                with cols[i]:
                    if format_name == "csv":
                        st.download_button(
                            label="⬇️ Download CSV",
                            data=buffer.getvalue(),
                            file_name=f"pvdf_metadata_{datetime.now().strftime('%Y%m%d')}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                    elif format_name == "json":
                        st.download_button(
                            label="⬇️ Download JSON",
                            data=buffer.getvalue(),
                            file_name=f"pvdf_metadata_{datetime.now().strftime('%Y%m%d')}.json",
                            mime="application/json",
                            use_container_width=True
                        )
                    elif format_name == "excel":
                        st.download_button(
                            label="⬇️ Download Excel",
                            data=buffer.getvalue(),
                            file_name=f"pvdf_metadata_{datetime.now().strftime('%Y%m%d')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True
                        )
                    elif format_name == "parquet":
                        st.download_button(
                            label="⬇️ Download Parquet",
                            data=buffer.getvalue(),
                            file_name=f"pvdf_metadata_{datetime.now().strftime('%Y%m%d')}.parquet",
                            mime="application/octet-stream",
                            use_container_width=True
                        )
        
        # Results display
        st.subheader(f"📄 Search Results ({len(papers)} papers)")
        
        # Filter and sort options
        col_filter1, col_filter2, col_filter3 = st.columns(3)
        
        with col_filter1:
            cluster_filter = st.multiselect(
                "Filter by Cluster",
                options=sorted(set(p['cluster_label'] for p in papers)),
                default=None
            )
        
        with col_filter2:
            year_range = st.slider(
                "Year Range",
                min_value=min(p['year'] for p in papers),
                max_value=max(p['year'] for p in papers),
                value=(min(p['year'] for p in papers), max(p['year'] for p in papers))
            )
        
        with col_filter3:
            sort_by = st.selectbox(
                "Sort By",
                ["Enhanced Relevance", "Year (Newest)", "Year (Oldest)", "Title"],
                index=0
            )
        
        # Apply filters and sorting
        filtered_papers = papers.copy()
        
        if cluster_filter:
            filtered_papers = [p for p in filtered_papers if p['cluster_label'] in cluster_filter]
        
        filtered_papers = [p for p in filtered_papers if year_range[0] <= p['year'] <= year_range[1]]
        
        if sort_by == "Enhanced Relevance":
            filtered_papers.sort(key=lambda x: x['enhanced_relevance_score'], reverse=True)
        elif sort_by == "Year (Newest)":
            filtered_papers.sort(key=lambda x: x['year'], reverse=True)
        elif sort_by == "Year (Oldest)":
            filtered_papers.sort(key=lambda x: x['year'])
        elif sort_by == "Title":
            filtered_papers.sort(key=lambda x: x['title'].lower())
        
        st.info(f"Showing {len(filtered_papers)} papers after filtering")
        
        # Display papers
        for i, paper in enumerate(filtered_papers):
            enhanced = paper.get('enhanced_relevance_score', 0)
            dopant = "🟢" if paper.get('dopant_present') else "⚪"
            beta = "🔵" if paper.get('beta_phase_present') else "⚪"
            piezo = "⚡" if paper.get('piezoelectric_present') else "⚪"
            cluster_label = paper.get('cluster_label', f"Cluster {paper.get('cluster_id', 0)}")
            
            with st.expander(f"**{paper['title']}** ({paper['year']}) - {cluster_label} | Basic: {paper['relevance_score']}% | Enhanced: {enhanced:.1f}% {dopant}{beta}{piezo}", expanded=i < 3):
                col_info, col_actions = st.columns([3, 1])
                
                with col_info:
                    st.write(f"**Authors:** {paper['authors']}") [[23]]
                    st.write(f"**Categories:** {paper['categories']}")
                    st.write(f"**Matched Terms:** {paper['matched_terms']}")
                    st.write(f"**Status:** {paper['download_status']}")
                    
                    show_abstract = st.toggle("Show Abstract", key=f"toggle_abstract_{paper['id']}_{i}")
                    if show_abstract:
                        st.markdown(f"> {paper['abstract']}")
                
                with col_actions:
                    paper_id = paper['id']
                    
                    # Show download status and buttons
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
                                key=f"dl_{paper_id}_{i}",
                                use_container_width=True
                            )
                    else:
                        if st.button("⬇️ Download Now", key=f"manual_{paper_id}_{i}", use_container_width=True):
                            with st.spinner("Downloading..."):
                                updated_paper = handle_paper_download(paper, manual_download=True)
                                filtered_papers[i] = updated_paper
                                st.rerun()
                    
                    st.markdown(f"[🌐 arXiv Page]({paper['pdf_url'].replace('/pdf/', '/abs/')})")
                    st.markdown(f"[📄 Direct PDF]({paper['pdf_url']})")
                
                # Show detailed analysis button
                if st.button("🔍 Show Detailed Analysis", key=f"detail_{paper_id}_{i}", use_container_width=True):
                    st.session_state.selected_paper_id = paper_id
                    st.rerun()
        
        # Visualization section
        if filtered_papers:
            st.subheader("📈 Research Analytics")
            
            col_viz1, col_viz2 = st.columns(2)
            
            with col_viz1:
                st.subheader("Relevance Distribution")
                fig_relevance = create_relevance_distribution_chart(filtered_papers)
                st.plotly_chart(fig_relevance, use_container_width=True)
            
            with col_viz2:
                st.subheader("Yearly Trends")
                fig_trends = create_yearly_trends_chart(filtered_papers)
                st.plotly_chart(fig_trends, use_container_width=True)
            
            # Cluster visualization
            st.subheader("Research Clusters")
            cluster_counts = Counter(p['cluster_label'] for p in filtered_papers)
            
            fig_clusters = px.pie(
                values=list(cluster_counts.values()),
                names=list(cluster_counts.keys()),
                title='Paper Distribution by Research Cluster'
            )
            st.plotly_chart(fig_clusters, use_container_width=True)

with main_tab2:
    st.header("📊 Advanced Analytics")
    
    if not st.session_state.get('relevant_papers'):
        st.info("Perform a search first to see analytics")
        st.stop()
    
    papers = st.session_state.relevant_papers
    
    # Cluster analysis
    st.subheader("Research Clusters Analysis")
    
    # Get cluster information from database
    cluster_info = {}
    conn = sqlite3.connect(db_manager.analytics_db)
    c = conn.cursor()
    c.execute("SELECT cluster_id, cluster_label, paper_count, dominant_keywords, avg_year FROM paper_clusters")
    rows = c.fetchall()
    conn.close()
    
    for row in rows:
        cluster_info[row[0]] = {
            'label': row[1],
            'count': row[2],
            'keywords': row[3].split(','),
            'avg_year': row[4]
        }
    
    # Display clusters
    for cluster_id, info in cluster_info.items():
        with st.expander(f"Cluster {cluster_id}: {info['label']} ({info['count']} papers)"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Key Characteristics")
                st.markdown(f"**Dominant Keywords:** {', '.join(info['keywords'][:5])}")
                st.markdown(f"**Average Publication Year:** {info['avg_year']:.1f}")
                st.markdown(f"**Paper Count:** {info['count']}")
            
            with col2:
                st.subheader("Sample Papers")
                cluster_papers = [p for p in papers if p.get('cluster_id') == cluster_id][:5]
                for paper in cluster_papers:
                    st.markdown(f"- **{paper['title'][:80]}...** ({paper['year']})")
            
            # Show papers in this cluster
            if st.button(f"Show All Papers in {info['label']} Cluster", key=f"show_cluster_{cluster_id}"):
                st.session_state.current_view = f"cluster_{cluster_id}"
                st.rerun()
    
    # Temporal analysis
    st.subheader("Temporal Research Trends")
    
    # Extract trend data
    trend_data = defaultdict(lambda: defaultdict(int))
    for paper in papers:
        year = paper['year']
        if paper['dopant_present']:
            trend_data[year]['dopant_papers'] += 1
        if paper['beta_phase_present']:
            trend_data[year]['beta_papers'] += 1
        if paper['piezoelectric_present']:
            trend_data[year]['piezo_papers'] += 1
        trend_data[year]['total'] += 1
    
    # Create trend chart
    years = sorted(trend_data.keys())
    dopant_counts = [trend_data[year]['dopant_papers'] for year in years]
    beta_counts = [trend_data[year]['beta_papers'] for year in years]
    piezo_counts = [trend_data[year]['piezo_papers'] for year in years]
    total_counts = [trend_data[year]['total'] for year in years]
    
    fig_trends = go.Figure()
    fig_trends.add_trace(go.Scatter(x=years, y=dopant_counts, name='Dopant Papers', mode='lines+markers'))
    fig_trends.add_trace(go.Scatter(x=years, y=beta_counts, name='Beta Phase Papers', mode='lines+markers'))
    fig_trends.add_trace(go.Scatter(x=years, y=piezo_counts, name='Piezoelectric Papers', mode='lines+markers'))
    fig_trends.add_trace(go.Scatter(x=years, y=total_counts, name='Total Papers', mode='lines+markers', line=dict(dash='dash')))
    
    fig_trends.update_layout(
        title='Research Trends Over Time',
        xaxis_title='Year',
        yaxis_title='Number of Papers',
        hovermode='x unified'
    )
    
    st.plotly_chart(fig_trends, use_container_width=True)
    
    # Word cloud analysis
    st.subheader("Text Analytics")
    
    if st.button("Generate Word Cloud from All Abstracts", use_container_width=True):
        with st.spinner("Generating comprehensive word cloud..."):
            all_text = " ".join([paper['abstract'] for paper in papers])
            wordcloud_fig = generate_wordcloud(all_text, max_words=200)
            if wordcloud_fig:
                st.pyplot(wordcloud_fig)
                st.success("Word cloud generated successfully!")
            else:
                st.warning("Could not generate word cloud - insufficient text data")

with main_tab3:
    st.header("🗄️ Database Management")
    
    col_stats, col_clean, col_backup = st.columns(3)
    
    with col_stats:
        if st.button("🔄 Refresh Statistics", use_container_width=True):
            st.session_state.db_stats = db_manager.get_db_stats()
            st.rerun()
    
    with col_clean:
        if st.button("🧹 Clean Temporary Files", use_container_width=True):
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
        if st.button("💾 Create Database Backup", use_container_width=True):
            with st.spinner("Creating database backup..."):
                backup_dir = os.path.join(DB_DIR, "backups")
                os.makedirs(backup_dir, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                backup_files = []
                for db_path in [METADATA_DB, UNIVERSE_DB, PDF_STORAGE_DB, ANALYTICS_DB]:
                    if os.path.exists(db_path):
                        backup_name = f"{os.path.basename(db_path)}_{timestamp}"
                        backup_path = os.path.join(backup_dir, backup_name)
                        try:
                            import shutil
                            shutil.copy2(db_path, backup_path)
                            backup_files.append(backup_name)
                        except Exception as e:
                            update_log(f"Backup failed for {db_path}: {e}")
                
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
                st.metric("Piezoelectric Papers", stats['piezoelectric_papers'])
                st.metric("Average Relevance", f"{stats['avg_relevance_score']}%")
            
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
        ["Metadata DB", "Full Text DB", "PDF Storage DB", "Analytics DB"]
    )
    
    rows_to_show = st.slider("Number of rows to display", 5, 100, 20)
    
    if st.button("🔍 Browse Database", use_container_width=True):
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
                df = pd.read_sql_query(f"SELECT * FROM paper_clusters LIMIT {rows_to_show}", conn)
            
            st.dataframe(df, use_container_width=True)
            conn.close()
        except Exception as e:
            st.error(f"Error browsing database: {e}")

with main_tab4:
    st.header("⚙️ System Management")
    
    # Memory management section
    st.subheader("🧠 Memory Management")
    
    col_mem1, col_mem2 = st.columns(2)
    
    with col_mem1:
        if st.button("🧹 Force Memory Cleanup", use_container_width=True):
            cleanup_memory(force=True)
            st.success("Memory cleanup completed!")
    
    with col_mem2:
        if st.button("📊 Show Memory Usage", use_container_width=True):
            process = psutil.Process(os.getpid())
            mem_info = process.memory_info()
            st.metric("RSS Memory", f"{mem_info.rss/1024/1024:.1f} MB")
            st.metric("Virtual Memory", f"{mem_info.vms/1024/1024:.1f} MB")
            st.metric("CPU Percent", f"{psutil.cpu_percent()}%")
    
    # Configuration management
    st.subheader("🔧 Configuration Management")
    
    config_col1, config_col2 = st.columns(2)
    
    with config_col1:
        st.subheader("Current Preferences")
        st.json(st.session_state.user_preferences)
    
    with config_col2:
        st.subheader("System Information")
        st.markdown(f"""
        - **Running on:** {'☁️ Streamlit Cloud' if IS_CLOUD else '💻 Local Machine'}
        - **Data Directory:** `{DB_DIR}`
        - **Python Version:** {sys.version.split()[0]}
        - **Streamlit Version:** {st.__version__}
        - **Database Files:**
          - Metadata: `{os.path.basename(METADATA_DB)}`
          - Full Text: `{os.path.basename(UNIVERSE_DB)}`
          - PDFs: `{os.path.basename(PDF_STORAGE_DB)}`
          - Analytics: `{os.path.basename(ANALYTICS_DB)}`
        """)
    
    # Log management
    st.subheader("📋 Log Management")
    
    if st.button("📥 Download Complete Log File", use_container_width=True):
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                log_content = f.read()
            st.download_button(
                label="⬇️ Download Log",
                data=log_content,
                file_name=f"pvdf_research_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                use_container_width=True
            )
        else:
            st.warning("Log file not found")
    
    # Show logs at the bottom
    show_logs(expanded=True)

# Show detailed paper view if selected
if st.session_state.selected_paper_id:
    st.header("📄 Paper Details")
    
    paper = next((p for p in st.session_state.relevant_papers if p['id'] == st.session_state.selected_paper_id), None)
    
    if paper:
        show_paper_details(paper)
        
        if st.button("🔙 Back to Search Results", use_container_width=True):
            st.session_state.selected_paper_id = None
            st.rerun()
    else:
        st.error("Paper not found")
        st.session_state.selected_paper_id = None

# Footer with system information
st.divider()
st.caption(f"""
**Piezoelectricity in PVDF Research Hub** | 
Running on {'☁️ Streamlit Cloud' if IS_CLOUD else '💻 Local Machine'} | 
Memory Usage: {st.session_state.memory_usage.get('rss_mb', 0):.1f} MB | 
Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 
Version: 2.1.0
""")

# Periodic cleanup
if random.random() < 0.1:  # 10% chance to run cleanup
    cleanup_memory()

# Register cleanup function for app exit
import atexit
def cleanup_on_exit():
    """Clean up resources when app exits."""
    cleanup_memory(force=True)
    # Close any open database connections
    for db_path in [METADATA_DB, UNIVERSE_DB, PDF_STORAGE_DB, ANALYTICS_DB]:
        try:
            conn = sqlite3.connect(db_path)
            conn.close()
        except:
            pass

atexit.register(cleanup_on_exit)
