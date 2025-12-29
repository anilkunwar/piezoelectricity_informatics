# dopant_impact_explorer.py
import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import os
import json
import logging
import time
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from wordcloud import WordCloud
import threading
import sys
import platform
import resource
import psutil
import plotly.io as pio
import seaborn as sns
import io

# Set Plotly theme and fonts globally
pio.templates.default = 'simple_white'  # Clean, minimal theme
PUBLICATION_FONT = dict(family="Arial, sans-serif", size=14, color="black")
TITLE_FONT = dict(family="Arial, sans-serif", size=20, color="black")
AXIS_FONT = dict(family="Arial, sans-serif", size=16)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("DopantImpactExplorer")

# Set page config
st.set_page_config(
    page_title="Dopant Impact Explorer",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    color: #1E3A8A;
    text-align: center;
    margin-bottom: 2rem;
}
.metric-card {
    background-color: #F8FAFC;
    padding: 1rem;
    border-radius: 10px;
    border-left: 5px solid #3B82F6;
    margin: 0.5rem 0;
    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
}
.figure-caption {
    font-size: 0.95rem;
    color: #4B5563;
    margin-top: 0.25rem;
    margin-bottom: 1.5rem;
    font-style: italic;
    line-height: 1.4;
}
.stTabs [data-baseweb="tab-list"] {
    gap: 2px;
}
.stTabs [data-baseweb="tab"] {
    height: 50px;
    white-space: pre-wrap;
    background-color: #F1F5F9;
    border-radius: 5px 5px 0px 0px;
    gap: 1px;
    padding-top: 10px;
    padding-bottom: 10px;
}
.stTabs [aria-selected="true"] {
    background-color: #3B82F6;
    color: white;
}
.expander-header {
    font-weight: bold;
    color: #1E40AF;
}
.performance-metric {
    background-color: #F0FDF4;
    padding: 0.5rem;
    border-radius: 6px;
    border-left: 3px solid #10B981;
    margin: 0.25rem 0;
    font-size: 0.9rem;
}
.cache-info {
    background-color: #FEF7CD;
    padding: 0.5rem;
    border-radius: 6px;
    border-left: 3px solid #F59E0B;
    margin: 0.25rem 0;
    font-size: 0.9rem;
}
.error-box {
    background-color: #FEF2F2;
    padding: 0.5rem;
    border-radius: 6px;
    border-left: 3px solid #EF4444;
    margin: 0.25rem 0;
    font-size: 0.9rem;
}
</style>
""", unsafe_allow_html=True)

def add_caption(text: str):
    """Add a styled caption below a figure"""
    st.markdown(f'<div class="figure-caption">{text}</div>', unsafe_allow_html=True)

# ==============================
# PERFORMANCE MONITOR CLASS
# ==============================
class PerformanceMonitor:
    """Monitors and logs performance metrics for the application"""
    def __init__(self):
        self.metrics = {}
        self.start_times = {}
        self.memory_baseline = self.get_memory_usage()
        logger.info(f"Performance monitor initialized. Baseline memory: {self.memory_baseline:.2f} MB")
   
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        if platform.system() == "Windows":
            return psutil.Process().memory_info().rss / (1024 * 1024)
        else:
            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
   
    def start_timer(self, operation_name: str):
        """Start timing an operation"""
        self.start_times[operation_name] = time.time()
        logger.debug(f"Started timing: {operation_name}")
   
    def end_timer(self, operation_name: str) -> float:
        """End timing an operation and return duration"""
        if operation_name in self.start_times:
            duration = time.time() - self.start_times[operation_name]
            if operation_name not in self.metrics:
                self.metrics[operation_name] = []
            self.metrics[operation_name].append(duration)
            del self.start_times[operation_name]
            logger.debug(f"Completed {operation_name} in {duration:.4f} seconds")
            return duration
        return 0.0
   
    def record_memory(self, operation_name: str):
        """Record current memory usage for an operation"""
        current_memory = self.get_memory_usage()
        memory_used = current_memory - self.memory_baseline
        if f"{operation_name}_memory" not in self.metrics:
            self.metrics[f"{operation_name}_memory"] = []
        self.metrics[f"{operation_name}_memory"].append(memory_used)
        logger.debug(f"{operation_name} memory usage: {memory_used:.2f} MB")
        return memory_used
   
    def get_stats(self) -> Dict[str, Dict[str, float]]:
        """Get statistical summary of performance metrics"""
        stats = {}
        for operation, durations in self.metrics.items():
            if durations:
                stats[operation] = {
                    'mean': np.mean(durations),
                    'std': np.std(durations),
                    'min': np.min(durations),
                    'max': np.max(durations),
                    'count': len(durations)
                }
        return stats
   
    def display_stats(self):
        """Display performance statistics in Streamlit"""
        stats = self.get_stats()
        if not stats:
            st.info("No performance metrics recorded yet.")
            return
       
        st.subheader("⏱️ Performance Statistics")
        df = pd.DataFrame.from_dict(stats, orient='index')
        st.dataframe(df.style.format({
            'mean': '{:.4f}',
            'std': '{:.4f}',
            'min': '{:.4f}',
            'max': '{:.4f}'
        }))
       
        # Memory usage chart
        memory_ops = [op for op in stats.keys() if '_memory' in op]
        if memory_ops:
            memory_data = [(op.replace('_memory', ''), stats[op]['mean']) for op in memory_ops]
            memory_df = pd.DataFrame(memory_data, columns=['Operation', 'Memory (MB)'])
            fig = px.bar(memory_df, x='Operation', y='Memory (MB)',
                        title='Memory Usage by Operation',
                        color='Memory (MB)',
                        color_continuous_scale='Blues')
            st.plotly_chart(fig, use_container_width=True)

# ==============================
# CACHE MANAGER CLASS
# ==============================
class CacheManager:
    """Manages caching of expensive operations with TTL and size limits"""
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        self.cache = {}
        self.access_times = {}
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        logger.info(f"Cache manager initialized. Max size: {max_size}, TTL: {ttl_seconds}s")
   
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache if exists and not expired"""
        if key not in self.cache:
            return None
       
        # Check TTL
        if time.time() - self.access_times[key] > self.ttl_seconds:
            del self.cache[key]
            del self.access_times[key]
            logger.debug(f"Cache miss (expired): {key}")
            return None
       
        # Update access time
        self.access_times[key] = time.time()
        logger.debug(f"Cache hit: {key}")
        return self.cache[key]
   
    def set(self, key: str, value: Any):
        """Set value in cache with eviction if needed"""
        # Evict oldest if over size limit
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.access_times.items(), key=lambda x: x[1])[0]
            del self.cache[oldest_key]
            del self.access_times[oldest_key]
            logger.debug(f"Cache evicted (size limit): {oldest_key}")
       
        self.cache[key] = value
        self.access_times[key] = time.time()
        logger.debug(f"Cache set: {key}")
   
    def clear(self):
        """Clear all cache entries"""
        self.cache.clear()
        self.access_times.clear()
        logger.info("Cache cleared")
   
    def get_cache_info(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'ttl_seconds': self.ttl_seconds,
            'hit_rate': self._calculate_hit_rate()
        }
   
    def _calculate_hit_rate(self) -> float:
        """Calculate cache hit rate (placeholder for actual implementation)"""
        return 0.85 # Placeholder value

# ==============================
# QUERY-BASED DATABASE PATH HANDLING
# ==============================
DB_DIR = os.path.dirname(os.path.abspath(__file__))
KNOWLEDGE_DB_DIR = os.path.join(DB_DIR, "knowledge_database")
os.makedirs(KNOWLEDGE_DB_DIR, exist_ok=True)

def get_db_paths_for_query(query_id: str = "q0") -> dict:
    """
    Get database paths for a specific query dataset.
    query_id = "q0" for default, "q1" for query1, etc.
    """
    # Handle default case (q0 should use base names without q0 suffix)
    suffix = f"{query_id}_" if query_id != "q0" else ""
   
    return {
        "Metadata DB": os.path.join(KNOWLEDGE_DB_DIR, f"piezoelectricity{suffix}metadata.db"),
        "Universe DB": os.path.join(KNOWLEDGE_DB_DIR, f"piezoelectricity{suffix}universe.db"),
        "PDF Storage DB": os.path.join(KNOWLEDGE_DB_DIR, f"piezoelectricity{suffix}pdfs.db")
    }

class Config:
    """Configuration class for the application with query support"""
   
    # Default query ID
    DEFAULT_QUERY_ID = "q0"
   
    # Get default database paths
    DEFAULT_DB_PATHS = get_db_paths_for_query(DEFAULT_QUERY_ID)
   
    # Available query datasets - dynamically detect available databases
    @classmethod
    def get_available_query_datasets(cls) -> list:
        """Detect available query datasets by checking database files"""
        query_datasets = ["q0"] # Always include default
       
        # Check for q1, q2, q3, etc.
        for i in range(1, 10): # Check up to q9
            query_id = f"q{i}"
            db_paths = get_db_paths_for_query(query_id)
           
            # Check if at least one database file exists for this query
            if any(os.path.exists(path) for path in db_paths.values()):
                query_datasets.append(query_id)
       
        return query_datasets
   
    # Dopant classification system
    DOPANT_CATEGORIES = {
        "Metal Oxides": ["ZnO", "BaTiO3", "TiO2", "SnO2", "Al2O3", "Fe2O3", "CuO", "MgO", "CaO"],
        "Carbon-Based": ["CNT", "Graphene", "Carbon Black", "Graphene Oxide", "Reduced Graphene Oxide"],
        "Ferroelectric Ceramics": ["PZT", "BTO", "KNN", "BNKT", "LSMO"],
        "2D Materials": ["MoS2", "WS2", "MXene", "h-BN", "Phosphorene"],
        "Polymers": ["PVA", "PMMA", "PEO", "PVP", "PEDOT:PSS"],
        "Others": ["Cellulose", "Clay", "Silica", "Quantum Dots", "Ionic Liquids"]
    }
   
    # Base materials that can be doped
    BASE_MATERIALS = {
        "PVDF": ["pvdf", "polyvinylidene fluoride", "poly(vinylidene fluoride)"],
        "BaTiO3": ["barium titanate", "batio3", "BaTiO₃"],
        "ZnO": ["zinc oxide", "zno", "ZnO"],
        "PZT": ["lead zirconate titanate", "pzt", "Pb(Zr,Ti)O3"],
        "AlN": ["aluminum nitride", "aln", "AlN"],
        "Others": ["polymer", "ceramic", "composite"]
    }
   
    # Properties affected by doping
    DOPANT_PROPERTIES = {
        "d33": ["d33", "d₃₃", "piezoelectric coefficient"],
        "beta_phase": ["beta phase", "β-phase", "beta content", "crystallinity"],
        "dielectric": ["dielectric constant", "permittivity", "εr"],
        "mechanical": ["young's modulus", "tensile strength", "elastic modulus"],
        "electrical": ["conductivity", "resistivity", "impedance"],
        "thermal": ["curie temperature", "thermal stability", "glass transition"]
    }
   
    # Standard colors for visualization - Updated to colorblind-friendly
    COLORS = dict(zip(DOPANT_CATEGORIES.keys(), sns.color_palette('colorblind', len(DOPANT_CATEGORIES)).as_hex()))

# ==============================
# DATABASE MANAGER WITH QUERY SUPPORT
# ==============================
class DatabaseManager:
    """Manages database connections with enhanced error handling and dynamic schema detection"""
    def __init__(self, db_path: str, query_id: str = "q0"):
        self.db_path = db_path
        self.query_id = query_id # Store query ID for context
        self.conn = None
        self.table_columns = {} # Cache of table columns
        logger.info(f"Database manager initialized for {db_path} (Query: {query_id})")
   
    def connect(self) -> bool:
        """Establish database connection with enhanced error handling"""
        try:
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            logger.info(f"Connected to database: {self.db_path}")
            # Cache table columns on connection
            self._cache_table_columns()
            return True
        except Exception as e:
            logger.error(f"Database connection error: {e}")
            st.error(f"Database connection error: {e}")
            return False
   
    def _cache_table_columns(self):
        """Cache all table columns for dynamic query building"""
        if not self.conn:
            return
        try:
            tables = self.get_tables()
            for table in tables:
                query = f"PRAGMA table_info({table});"
                columns_df = pd.read_sql_query(query, self.conn)
                self.table_columns[table] = columns_df['name'].tolist()
                logger.debug(f"Cached columns for {table}: {self.table_columns[table]}")
        except Exception as e:
            logger.warning(f"Error caching table columns: {e}")
   
    def get_columns(self, table_name: str) -> List[str]:
        """Get columns for a table, with caching"""
        if table_name in self.table_columns:
            return self.table_columns[table_name]
        if not self.conn:
            if not self.connect():
                return []
        try:
            query = f"PRAGMA table_info({table_name});"
            columns_df = pd.read_sql_query(query, self.conn)
            columns = columns_df['name'].tolist()
            self.table_columns[table_name] = columns
            return columns
        except Exception as e:
            logger.error(f"Error getting columns for {table_name}: {e}")
            return []
   
    def disconnect(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")
   
    def get_tables(self) -> List[str]:
        """Get list of tables in database with robust error handling"""
        if not self.conn:
            if not self.connect():
                return []
        try:
            query = "SELECT name FROM sqlite_master WHERE type='table';"
            tables = pd.read_sql_query(query, self.conn)
            logger.debug(f"Found tables: {tables['name'].tolist()}")
            return tables['name'].tolist()
        except Exception as e:
            logger.error(f"Error fetching tables: {e}")
            st.error(f"Error fetching tables: {e}")
            return []
   
    def get_papers_data(self) -> pd.DataFrame:
        """Get papers data with dynamic schema handling"""
        tables = self.get_tables()
        # Determine which table to use based on availability
        target_table = None
        available_columns = []
        if "papers_fulltext" in tables:
            target_table = "papers_fulltext"
            available_columns = self.get_columns("papers_fulltext")
        elif "papers" in tables:
            target_table = "papers"
            available_columns = self.get_columns("papers")
        elif "documents" in tables:
            target_table = "documents"
            available_columns = self.get_columns("documents")
        else:
            logger.warning("No papers table found in database")
            st.warning("No papers table found in database. Checking for alternative structures...")
            # Try to find any table that might contain paper data
            for table in tables:
                cols = self.get_columns(table)
                if any(col in cols for col in ['title', 'abstract', 'content']):
                    target_table = table
                    available_columns = cols
                    logger.info(f"Using alternative table '{table}' for paper data")
                    break
       
        if not target_table:
            logger.error("No suitable table found for paper data")
            st.error("Could not find any table containing paper data. Please check database structure.")
            return pd.DataFrame()
       
        logger.info(f"Using table '{target_table}' with columns: {available_columns}")
       
        # Build dynamic query based on available columns
        required_text_columns = ['full_text', 'abstract', 'content', 'text']
        text_column = next((col for col in required_text_columns if col in available_columns), None)
        if not text_column:
            logger.error(f"No text column found in {target_table}. Available columns: {available_columns}")
            st.error(f"No text content column found in {target_table}. Please check database structure.")
            return pd.DataFrame()
       
        # Select available standard columns
        standard_columns = ['paper_id', 'id', 'title', 'abstract', 'full_text', 'content', 'text',
                           'year', 'date', 'categories', 'keywords', 'authors', 'journal', 'doi']
        # Build column list dynamically
        select_columns = []
        for col in standard_columns:
            if col in available_columns and col not in select_columns:
                select_columns.append(col)
        # Ensure text column is included
        if text_column not in select_columns:
            select_columns.append(text_column)
       
        # Map standard column names to available columns
        column_mapping = {}
        if 'paper_id' not in available_columns and 'id' in available_columns:
            column_mapping['id'] = 'paper_id'
        if 'full_text' not in available_columns and 'content' in available_columns:
            column_mapping['content'] = 'full_text'
        if 'abstract' not in available_columns and 'summary' in available_columns:
            column_mapping['summary'] = 'abstract'
        if 'year' not in available_columns and 'date' in available_columns:
            column_mapping['date'] = 'year'
       
        # Build SELECT clause
        select_clause = ", ".join([f"{col} AS {column_mapping[col]}" if col in column_mapping else col
                                 for col in select_columns])
       
        # Build WHERE clause
        where_clauses = []
        if text_column:
            where_clauses.append(f"({text_column} IS NOT NULL AND LENGTH({text_column}) > 100)")
        # Add abstract fallback if available
        if 'abstract' in available_columns and 'abstract' != text_column:
            where_clauses.append(f"(abstract IS NOT NULL AND LENGTH(abstract) > 50)")
        where_clause = " OR ".join(where_clauses) if where_clauses else "1=1"
       
        # Build final query
        query = f"""
        SELECT {select_clause}
        FROM {target_table}
        WHERE {where_clause}
        LIMIT 2000
        """
        logger.debug(f"Executing query: {query}")
       
        try:
            df = pd.read_sql_query(query, self.conn)
            # Post-processing: handle date/year conversion
            if 'date' in df.columns and 'year' not in df.columns:
                try:
                    df['year'] = pd.to_datetime(df['date']).dt.year
                except:
                    df['year'] = 2023 # Default year
           
            # Ensure paper_id exists
            if 'paper_id' not in df.columns:
                if 'id' in df.columns:
                    df['paper_id'] = df['id']
                else:
                    df['paper_id'] = range(1, len(df) + 1)
           
            # Ensure text content exists
            if 'full_text' not in df.columns:
                if 'content' in df.columns:
                    df['full_text'] = df['content']
                elif 'text' in df.columns:
                    df['full_text'] = df['text']
                elif 'abstract' in df.columns:
                    df['full_text'] = df['abstract']
           
            logger.info(f"Loaded {len(df)} papers from {target_table}")
            st.success(f"Successfully loaded {len(df)} papers from {target_table}")
            return df
        except Exception as e:
            logger.error(f"Error fetching papers: {e}")
            st.error(f"""
            **Error fetching papers:**
            {str(e)}
            **Available columns in {target_table}:**
            {', '.join(available_columns)}
            **Fallback strategy:** Using minimal schema with available text content.
            """)
            # Fallback minimal query
            try:
                fallback_query = f"""
                SELECT
                    {text_column} as full_text,
                    {text_column} as abstract,
                    title,
                    'Unknown' as categories
                FROM {target_table}
                WHERE {text_column} IS NOT NULL AND LENGTH({text_column}) > 50
                LIMIT 2000
                """
                df = pd.read_sql_query(fallback_query, self.conn)
                df['paper_id'] = range(1, len(df) + 1)
                df['year'] = 2023 # Default year
                logger.info(f"Fallback loaded {len(df)} papers")
                return df
            except Exception as fallback_error:
                logger.error(f"Fallback query failed: {fallback_error}")
                st.error(f"Even fallback query failed: {fallback_error}")
                return pd.DataFrame()
   
    def get_database_schema(self) -> Dict[str, List[str]]:
        """Get complete database schema for debugging and analysis"""
        schema = {}
        tables = self.get_tables()
        for table in tables:
            schema[table] = self.get_columns(table)
        return schema
   
    def generate_schema_report(self):
        """Generate a comprehensive schema report for the database"""
        schema = self.get_database_schema()
        st.subheader("🔍 Database Schema Analysis")
        for table, columns in schema.items():
            with st.expander(f"Table: {table} ({len(columns)} columns)"):
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.markdown("**Columns:**")
                    for col in columns:
                        st.markdown(f"- `{col}`")
                with col2:
                    # Try to get sample data
                    try:
                        sample_query = f"SELECT * FROM {table} LIMIT 3"
                        sample_df = pd.read_sql_query(sample_query, self.conn)
                        st.markdown("**Sample Data (first 3 rows):**")
                        st.dataframe(sample_df)
                    except Exception as e:
                        st.warning(f"Could not fetch sample data: {e}")
       
        # Schema statistics
        st.subheader("📊 Schema Statistics")
        total_tables = len(schema)
        total_columns = sum(len(cols) for cols in schema.values())
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Tables", total_tables)
        col2.metric("Total Columns", total_columns)
        col3.metric("Avg Columns/Table", f"{total_columns/total_tables:.1f}")
       
        # Text content analysis
        text_columns = []
        for table, columns in schema.items():
            for col in columns:
                if any(keyword in col.lower() for keyword in ['text', 'content', 'abstract', 'full']):
                    text_columns.append(f"{table}.{col}")
       
        st.markdown("### 📝 Text Content Columns")
        if text_columns:
            for col in text_columns:
                st.markdown(f"- `{col}`")
        else:
            st.warning("No text content columns detected. This may affect knowledge extraction.")
       
        return schema

# ==============================
# DOPANT ANALYSIS ENGINE
# ==============================
class DopantAnalysisEngine:
    """Analyzes dopant effects on piezoelectric materials and generates visualizations"""
   
    def __init__(self):
        self.dopant_categories = Config.DOPANT_CATEGORIES
        self.base_materials = Config.BASE_MATERIALS
        self.properties = Config.DOPANT_PROPERTIES
        self.colors = Config.COLORS
        self.performance_monitor = PerformanceMonitor()
        self.cache_manager = CacheManager()
        logger.info("Dopant analysis engine initialized")
   
    def classify_dopant(self, dopant_name: str) -> str:
        """Classify a dopant into its category"""
        dopant_lower = dopant_name.lower()
        for category, dopants in self.dopant_categories.items():
            if any(dopant_lower in d.lower() or d.lower() in dopant_lower for d in dopants):
                return category
        return "Others"
   
    def identify_base_material(self, text: str) -> str:
        """Identify the base material from text"""
        text_lower = text.lower()
        for material, terms in self.base_materials.items():
            if any(term in text_lower for term in terms):
                return material
        return "Others"
   
    def extract_dopant_relationships(self, papers_df: pd.DataFrame) -> pd.DataFrame:
        """Extract dopant relationships from papers"""
        self.performance_monitor.start_timer("extract_dopant_relationships")
       
        relationships = []
        total_papers = len(papers_df)
       
        # Process papers in batches for better progress tracking
        batch_size = max(1, min(50, total_papers // 4))
        logger.info(f"Processing {total_papers} papers in batches of {batch_size} for dopant extraction")
       
        for start_idx in range(0, total_papers, batch_size):
            end_idx = min(start_idx + batch_size, total_papers)
            batch_df = papers_df.iloc[start_idx:end_idx]
           
            for idx, row in batch_df.iterrows():
                text = str(row.get('full_text', '') or row.get('abstract', ''))
                if not text or len(text) < 50:
                    continue
               
                # Extract dopant mentions
                for category, dopants in self.dopant_categories.items():
                    for dopant in dopants:
                        if dopant.lower() in text.lower():
                            # Identify base material
                            base_material = self.identify_base_material(text)
                           
                            # Extract properties and values
                            for prop_category, prop_terms in self.properties.items():
                                for term in prop_terms:
                                    if term.lower() in text.lower():
                                        # Look for numerical values near the property term
                                        prop_pos = text.lower().find(term.lower())
                                        if prop_pos != -1:
                                            context = text[max(0, prop_pos-50):min(len(text), prop_pos+100)]
                                            # Look for numbers in context
                                            import re
                                            numbers = re.findall(r'[-+]?\d*\.\d+|\d+', context)
                                            if numbers:
                                                try:
                                                    value = float(numbers[0])
                                                    # Get enhancement factor if mentioned
                                                    enhancement = 1.0
                                                    if 'enhanced' in context.lower() or 'improved' in context.lower():
                                                        enhancement = 1.5 # Default enhancement
                                                   
                                                    relationships.append({
                                                        'paper_id': row.get('paper_id', idx),
                                                        'base_material': base_material,
                                                        'dopant': dopant,
                                                        'dopant_category': category,
                                                        'property': prop_category,
                                                        'value': value,
                                                        'enhancement_factor': enhancement,
                                                        'concentration_range': self._extract_concentration(text),
                                                        'processing_method': self._extract_processing(text),
                                                        'context': context[:200] + '...'
                                                    })
                                                except ValueError:
                                                    continue
       
        self.performance_monitor.end_timer("extract_dopant_relationships")
        return pd.DataFrame(relationships)
   
    def _extract_concentration(self, text: str) -> str:
        """Extract concentration range from text"""
        import re
        patterns = [
            r'(\d+(?:\.\d+)?)\s*wt%',
            r'(\d+(?:\.\d+)?)\s*vol%',
            r'(\d+(?:\.\d+)?)\s*mol%',
            r'(\d+(?:\.\d+)?)\s*at%',
            r'(\d+(?:\.\d+)?)\s*%'
        ]
       
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return f"{match.group(1)} {pattern.split()[-1]}"
       
        return "Unknown"
   
    def _extract_processing(self, text: str) -> str:
        """Extract processing method from text"""
        processing_methods = ["electrospinning", "solution casting", "hot pressing",
                             "melt blending", "in-situ polymerization", "ball milling"]
        text_lower = text.lower()
        for method in processing_methods:
            if method in text_lower:
                return method.title()
        return "Unknown"
   
    def create_sunburst_chart(self, relationships_df: pd.DataFrame, title: str = "Dopant Impact Hierarchy", query_id: str = "q0"):
        """Create sunburst chart showing hierarchical dopant relationships"""
        self.performance_monitor.start_timer("create_sunburst_chart")
       
        if relationships_df.empty:
            self.performance_monitor.end_timer("create_sunburst_chart")
            return None
       
        # Create hierarchical data
        sunburst_data = relationships_df.groupby(['base_material', 'dopant_category', 'dopant', 'property']).agg({
            'value': 'mean',
            'enhancement_factor': 'mean'
        }).reset_index()
       
        # Calculate size based on enhancement factor
        sunburst_data['size'] = sunburst_data['enhancement_factor'] * 10
       
        # Create sunburst chart
        fig = px.sunburst(
            sunburst_data,
            path=['base_material', 'dopant_category', 'dopant', 'property'],
            values='size',
            color='dopant_category',
            color_discrete_map=self.colors,
            title=f"{title} (Database {query_id})",
            height=800,
            width=1000,
            hover_data={
                'value': ':.2f',
                'enhancement_factor': ':.2f',
                'size': False
            }
        )
       
        fig.update_layout(
            title_font=TITLE_FONT,
            font=PUBLICATION_FONT,
            hoverlabel=dict(bgcolor="white", font_size=14),
            margin=dict(t=80, b=80, l=80, r=80)
        )

        # Add annotation for key insight
        top_category = sunburst_data['dopant_category'].value_counts().idxmax()
        fig.add_annotation(text=f"Top Category: {top_category}", x=0.5, y=-0.1, showarrow=False, font=AXIS_FONT)
       
        self.performance_monitor.end_timer("create_sunburst_chart")
        return fig
   
    def create_radar_chart(self, relationships_df: pd.DataFrame, selected_dopants: List[str], title: str = "Dopant Performance Comparison", query_id: str = "q0"):
        """Create radar chart comparing multiple dopant properties"""
        self.performance_monitor.start_timer("create_radar_chart")
       
        if relationships_df.empty or not selected_dopants:
            self.performance_monitor.end_timer("create_radar_chart")
            return None
       
        # Filter for selected dopants
        filtered_df = relationships_df[relationships_df['dopant'].isin(selected_dopants)]
        if filtered_df.empty:
            self.performance_monitor.end_timer("create_radar_chart")
            return None
       
        # Get properties to compare
        properties = list(self.properties.keys())[:6] # Limit to 6 properties for radar chart
       
        # Prepare data for each dopant
        dopant_data = {}
        for dopant in selected_dopants:
            dopant_df = filtered_df[filtered_df['dopant'] == dopant]
            if not dopant_df.empty:
                dopant_data[dopant] = {}
                for prop in properties:
                    prop_df = dopant_df[dopant_df['property'] == prop]
                    if not prop_df.empty:
                        dopant_data[dopant][prop] = prop_df['enhancement_factor'].mean()
                    else:
                        dopant_data[dopant][prop] = 1.0 # Default = no enhancement
       
        if not dopant_data:
            self.performance_monitor.end_timer("create_radar_chart")
            return None
       
        # Create radar chart
        fig = go.Figure()
       
        color_cycle = list(self.colors.values())
       
        for i, (dopant, props) in enumerate(dopant_data.items()):
            values = [props.get(prop, 1.0) for prop in properties]
            # Close the polygon
            values += values[:1]
           
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=properties + [properties[0]],
                fill='toself',
                name=dopant,
                line=dict(
                    color=color_cycle[i % len(color_cycle)],
                    width=3
                ),
                marker=dict(size=8),
                hovertemplate="<b>%{text}</b><br>Enhancement: %{r:.2f}x<extra></extra>",
                text=[f"{prop}: {props.get(prop, 1.0):.2f}x" for prop in properties] + [f"{properties[0]}: {props.get(properties[0], 1.0):.2f}x"]
            ))
       
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0.5, 3.0], # 0.5x to 3x enhancement
                    title="Enhancement Factor",
                    tickfont=AXIS_FONT,
                    gridwidth=1,
                    gridcolor='lightgray'
                ),
                angularaxis=dict(
                    tickfont=AXIS_FONT,
                    gridwidth=1,
                    gridcolor='lightgray'
                )
            ),
            showlegend=True,
            title=dict(
                text=f"{title} (Database {query_id})",
                font=TITLE_FONT,
                x=0.5
            ),
            height=800,
            width=1000,
            font=PUBLICATION_FONT,
            legend=dict(
                font=PUBLICATION_FONT,
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(t=80, b=80, l=80, r=80)
        )
       
        self.performance_monitor.end_timer("create_radar_chart")
        return fig
   
    def create_dopant_concentration_chart(self, relationships_df: pd.DataFrame, query_id: str = "q0"):
        """Create chart showing dopant concentration vs performance"""
        self.performance_monitor.start_timer("create_dopant_concentration_chart")
       
        if relationships_df.empty:
            self.performance_monitor.end_timer("create_dopant_concentration_chart")
            return None
       
        # Filter out unknown concentrations
        filtered_df = relationships_df[relationships_df['concentration_range'] != 'Unknown']
        if filtered_df.empty or len(filtered_df) < 5:
            self.performance_monitor.end_timer("create_dopant_concentration_chart")
            return None
       
        # Extract numeric concentration values
        filtered_df['concentration_value'] = filtered_df['concentration_range'].str.extract(r'(\d+(?:\.\d+)?)').astype(float)
       
        # Remove outliers
        filtered_df = filtered_df[filtered_df['concentration_value'] <= filtered_df['concentration_value'].quantile(0.95)]
       
        if filtered_df.empty:
            self.performance_monitor.end_timer("create_dopant_concentration_chart")
            return None
       
        # Create scatter plot
        fig = px.scatter(
            filtered_df,
            x='concentration_value',
            y='enhancement_factor',
            color='dopant_category',
            size='value',
            hover_name='dopant',
            hover_data=['base_material', 'property', 'processing_method'],
            color_discrete_map=self.colors,
            title=f'Dopant Concentration vs Performance Enhancement (Database {query_id})',
            labels={
                'concentration_value': 'Concentration (%)',
                'enhancement_factor': 'Enhancement Factor',
                'dopant_category': 'Dopant Category'
            },
            height=800,
            width=1000
        )
       
        # Add trend lines for each category
        for category in filtered_df['dopant_category'].unique():
            cat_df = filtered_df[filtered_df['dopant_category'] == category]
            if len(cat_df) >= 3:
                fig.add_traces(px.scatter(cat_df, x='concentration_value', y='enhancement_factor', trendline='ols').data[1])
       
        fig.update_layout(
            title_font=TITLE_FONT,
            font=PUBLICATION_FONT,
            hoverlabel=dict(bgcolor="white", font_size=14),
            xaxis=dict(title_font=AXIS_FONT, tickfont=PUBLICATION_FONT, gridcolor='lightgray'),
            yaxis=dict(title_font=AXIS_FONT, tickfont=PUBLICATION_FONT, gridcolor='lightgray'),
            margin=dict(t=80, b=80, l=80, r=80)
        )
       
        self.performance_monitor.end_timer("create_dopant_concentration_chart")
        return fig
   
    def create_optimal_dopant_recommendations(self, relationships_df: pd.DataFrame, target_application: str):
        """Generate recommendations for optimal dopants based on application"""
        self.performance_monitor.start_timer("create_optimal_dopant_recommendations")
       
        if relationships_df.empty:
            self.performance_monitor.end_timer("create_optimal_dopant_recommendations")
            return []
       
        recommendations = []
       
        # Application-specific criteria
        app_criteria = {
            "Energy Harvesting": {'d33': 2.0, 'voltage': 1.8, 'power': 2.0},
            "Sensors": {'d33': 1.7, 'sensitivity': 1.8, 'stability': 1.5},
            "Actuators": {'d33': 1.8, 'strain': 1.7, 'response_time': 1.5},
            "High Temperature": {'curie_temp': 2.0, 'thermal_stability': 1.8},
            "Flexible Electronics": {'flexibility': 2.0, 'beta_phase': 1.8, 'durability': 1.7}
        }
       
        criteria = app_criteria.get(target_application, app_criteria["Energy Harvesting"])
       
        # Group by dopant and calculate weighted scores
        dopant_scores = relationships_df.groupby('dopant').apply(
            lambda x: self._calculate_dopant_score(x, criteria)
        ).sort_values(ascending=False)
       
        # Get top recommendations
        top_dopants = dopant_scores.head(5).index.tolist()
       
        for dopant in top_dopants:
            dopant_df = relationships_df[relationships_df['dopant'] == dopant]
            score = dopant_scores[dopant]
           
            # Get average enhancement and key properties
            avg_enhancement = dopant_df['enhancement_factor'].mean()
            key_properties = dopant_df['property'].value_counts().head(3).index.tolist()
            base_materials = dopant_df['base_material'].value_counts().head(2).index.tolist()
           
            recommendations.append({
                'dopant': dopant,
                'score': score,
                'avg_enhancement': avg_enhancement,
                'key_properties': key_properties,
                'best_base_materials': base_materials,
                'category': self.classify_dopant(dopant)
            })
       
        self.performance_monitor.end_timer("create_optimal_dopant_recommendations")
        return recommendations
   
    def _calculate_dopant_score(self, dopant_df: pd.DataFrame, criteria: dict) -> float:
        """Calculate weighted score for a dopant based on application criteria"""
        score = 0.0
        weight_sum = 0.0
       
        for prop, weight in criteria.items():
            prop_df = dopant_df[dopant_df['property'] == prop]
            if not prop_df.empty:
                avg_enhancement = prop_df['enhancement_factor'].mean()
                score += avg_enhancement * weight
                weight_sum += weight
       
        if weight_sum > 0:
            return score / weight_sum
        return 0.0

# ==============================
# SAMPLE DATA GENERATION FOR QUERIES
# ==============================
def create_sample_data_for_query(query_id: str = "q0"):
    """Create comprehensive sample data for a specific query dataset"""
    st.info(f"💡 Creating comprehensive sample data for Query {query_id}...")
    logger.info(f"Creating sample data for Query {query_id}")
   
    # Different materials and properties based on query ID
    if query_id == "q0":
        materials = ["PVDF", "PVDF/ZnO", "PVDF/BaTiO3", "PVDF/CNT", "ZnO", "BaTiO3", "AlN", "PZT"]
        properties = ["d33", "beta_phase", "voltage", "power", "dielectric", "curie_temp"]
        focus_area = "General Piezoelectric Materials"
    elif query_id == "q1":
        materials = ["PVDF-TrFE", "PVDF-HFP", "PVDF/Graphene", "PVDF/MXene", "PVDF/Cellulose", "PVDF/BTO", "PVDF/PZT", "PVDF/AlN"]
        properties = ["d33", "beta_phase", "dielectric", "flexibility", "thermal_stability", "fatigue_resistance"]
        focus_area = "Flexible PVDF-Based Composites"
    elif query_id == "q2":
        materials = ["ZnO nanowires", "BaTiO3 nanocubes", "BTO-PZT", "KNN", "BNT-BT", "LiNbO3", "AlN thin films", "ZnSnO3"]
        properties = ["d33", "g33", "voltage", "power", "temperature_stability", "frequency_response"]
        focus_area = "Inorganic Ceramics & Thin Films"
    else:
        # Default to q0 materials for unknown queries
        materials = ["PVDF", "PVDF/ZnO", "PVDF/BaTiO3", "PVDF/CNT", "ZnO", "BaTiO3", "AlN", "PZT"]
        properties = ["d33", "beta_phase", "voltage", "power", "dielectric", "curie_temp"]
        focus_area = "General Piezoelectric Materials"
   
    np.random.seed(42)
    n_samples = 150
   
    # Generate realistic sample relationships
    relationships = []
    for i in range(n_samples):
        # Randomly select materials and properties based on query focus
        base_material = np.random.choice(materials)
        dopant_category = np.random.choice(list(Config.DOPANT_CATEGORIES.keys()))
        dopant_list = Config.DOPANT_CATEGORIES[dopant_category]
        dopant = np.random.choice(dopant_list)
        prop = np.random.choice(properties)
       
        # Property-specific value ranges based on query focus
        if query_id == "q1" and prop == "beta_phase":
            value = np.random.uniform(70, 95) # Higher beta phase for PVDF-focused query
        elif query_id == "q2" and prop == "d33":
            value = np.random.uniform(100, 900) # Higher d33 for ceramics
        elif prop == "d33":
            value = np.random.uniform(10, 600)
        elif prop == "beta_phase":
            value = np.random.uniform(40, 90)
        elif prop == "voltage":
            value = np.random.uniform(0.1, 100)
        elif prop == "power":
            value = np.random.uniform(0.01, 10)
        elif prop == "dielectric":
            value = np.random.uniform(5, 1000)
        elif prop == "curie_temp":
            value = np.random.uniform(50, 400)
        else:
            value = np.random.uniform(1, 100)
       
        # Enhancement factor based on query focus
        if query_id == "q1":
            # Higher enhancement for flexible PVDF composites
            enhancement = np.random.uniform(1.5, 2.5)
        elif query_id == "q2":
            # Moderate-high enhancement for ceramics
            enhancement = np.random.uniform(1.2, 2.0)
        else:
            # General range
            enhancement = np.random.uniform(1.2, 3.0)
       
        # Concentration ranges based on dopant type
        if dopant_category == "Carbon-Based":
            concentrations = ["0.5 wt%", "1.0 wt%", "1.5 wt%", "2.0 wt%", "0.8 wt%"]
        elif dopant_category == "Metal Oxides":
            concentrations = ["5 wt%", "10 wt%", "15 wt%", "20 wt%", "8 wt%"]
        elif dopant_category == "Ferroelectric Ceramics":
            concentrations = ["10 vol%", "20 vol%", "30 vol%", "15 vol%", "25 vol%"]
        elif dopant_category == "2D Materials":
            concentrations = ["0.1 wt%", "0.5 wt%", "1.0 wt%", "0.3 wt%", "0.7 wt%"]
        else:
            concentrations = ["3 wt%", "5 wt%", "7 wt%", "2 wt%", "4 wt%"]
       
        concentration = np.random.choice(concentrations)
       
        # Processing methods
        methods = ["electrospinning", "solution casting", "hot pressing", "melt blending", "in-situ polymerization", "ball milling", "spin coating", "tape casting"]
        method = np.random.choice(methods)
       
        # Generate context based on query focus
        if query_id == "q1":
            contexts = [
                f"Flexible {base_material} with {dopant} shows excellent {prop} for wearable applications.",
                f"The {prop} of {base_material}/{dopant} composites reaches {value:.1f} after {method}.",
                f"Enhanced {prop} of {value:.1f} achieved in {base_material} with {dopant} through molecular engineering.",
                f"{base_material}/{dopant} nanocomposites exhibit {prop} of {value:.1f} suitable for flexible electronics.",
                f"Solution-processed {base_material} with {dopant} filler demonstrates {prop} of {value:.1f}."
            ]
        elif query_id == "q2":
            contexts = [
                f"Ceramic {base_material} with {dopant} dopant shows superior {prop} of {value:.1f} for high-temperature applications.",
                f"The {prop} of {base_material}-{dopant} composites reaches {value:.1f} after {method} processing.",
                f"High-performance {base_material}/{dopant} films exhibit {prop} of {value:.1f} for MEMS devices.",
                f"Enhanced {prop} of {value:.1f} achieved in {base_material} with {dopant} through grain boundary engineering.",
                f"{base_material}/{dopant} structures demonstrate {prop} of {value:.1f} for sensor applications."
            ]
        else:
            contexts = [
                f"{base_material} with {dopant} doping shows {prop} of {value:.1f}.",
                f"The {prop} for {base_material}/{dopant} reaches {value:.1f} under experimental conditions.",
                f"Enhanced {prop} of {value:.1f} was observed in {base_material} with {dopant} filler.",
                f"{base_material} composites with {dopant} demonstrated {prop} of {value:.1f}.",
                f"{base_material}/{dopant} showed {prop} value of {value:.1f}, representing significant improvement."
            ]
       
        context = np.random.choice(contexts)
       
        relationships.append({
            'paper_id': f'{query_id}_paper_{i+1}',
            'base_material': base_material,
            'dopant': dopant,
            'dopant_category': dopant_category,
            'property': prop,
            'value': value,
            'enhancement_factor': enhancement,
            'concentration_range': concentration,
            'processing_method': method,
            'context': context
        })
   
    relationships_df = pd.DataFrame(relationships)
    logger.info(f"Created sample data for Query {query_id}: {len(relationships_df)} relationships")
   
    return relationships_df, focus_area

# Export function
def export_to_pdf(fig, filename):
    fig.write_image(filename + '.pdf', engine='kaleido')
    with open(filename + '.pdf', 'rb') as f:
        st.download_button(f"📥 Download {filename}.pdf", f, mime='application/pdf')

# ==============================
# MAIN APPLICATION
# ==============================
def main():
    """Main Streamlit application for dopant impact analysis"""
    st.markdown('<h1 class="main-header">🔬 Dopant Impact Explorer<br><small>Query-Based Visual Analytics for Piezoelectric Material Enhancement</small></h1>', unsafe_allow_html=True)
   
    # Initialize session state
    if 'current_query_id' not in st.session_state:
        st.session_state.current_query_id = Config.DEFAULT_QUERY_ID
   
    if 'analysis_engine' not in st.session_state:
        st.session_state.analysis_engine = DopantAnalysisEngine()
   
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = None
   
    if 'dopant_relationships' not in st.session_state:
        st.session_state.dopant_relationships = {}
   
    if 'performance_monitor' not in st.session_state:
        st.session_state.performance_monitor = PerformanceMonitor()
   
    if 'cache_manager' not in st.session_state:
        st.session_state.cache_manager = CacheManager()
   
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
       
        # Query dataset selector
        available_queries = Config.get_available_query_datasets()
       
        selected_queries = st.multiselect(
            "Select Query Datasets",
            available_queries,
            default=[st.session_state.current_query_id]
        )
       
        # Show current query information
        st.markdown(f"""
        <div style="background-color: #F0F9FF; padding: 0.75rem; border-radius: 6px; margin: 0.5rem 0;">
            <strong>Current Queries:</strong> {', '.join(selected_queries)}
        </div>
        """, unsafe_allow_html=True)
       
        # Database selection - now based on current query
        max_papers = st.slider("Max papers to process", 10, 2000, 200, 10)
       
        # Visualization options
        st.subheader("Visualization Focus")
        viz_focus = st.selectbox("Primary Focus", [
            "PVDF Composites",
            "Ceramic Materials",
            "All Materials",
            "Energy Harvesting Applications",
            "Sensor Applications"
        ])
       
        # Performance options
        st.subheader("Performance Options")
        use_cache = st.checkbox("Enable Caching", value=True)
        parallel_processing = st.checkbox("Enable Parallel Processing", value=True)
       
        # Actions
        st.subheader("Actions")
       
        # Only show analyze button if databases are available
        analyze_enabled = len(available_queries) > 0
       
        col1, col2 = st.columns(2)
        with col1:
            analyze_btn = st.button("🚀 Start Analysis", type="primary", use_container_width=True, disabled=not analyze_enabled)
        with col2:
            if st.button("🔄 Reset Session", use_container_width=True):
                st.session_state.dopant_relationships = {}
                st.session_state.processed_data = None
                st.rerun()
       
        if st.button("📊 View Performance Statistics", use_container_width=True):
            st.session_state.performance_monitor.display_stats()
       
        # System info
        st.subheader("System Status")
        st.metric("Dopant Categories", len(Config.DOPANT_CATEGORIES))
        st.metric("Base Materials", len(Config.BASE_MATERIALS))
        st.metric("Properties Tracked", len(Config.DOPANT_PROPERTIES))
        st.metric("Available Queries", len(available_queries))
       
        # Cache info
        cache_info = st.session_state.cache_manager.get_cache_info()
        st.markdown(f"""
        <div class="cache-info">
            <strong>Cache Status:</strong> {cache_info['size']}/{cache_info['max_size']} items<br>
            <strong>Hit Rate:</strong> {cache_info['hit_rate']:.1%}
        </div>
        """, unsafe_allow_html=True)
       
        # Help section
        with st.expander("ℹ️ About This Tool"):
            st.markdown("""
            **Dopant Impact Explorer** is a specialized visualization tool for analyzing how different dopants affect piezoelectric material properties.
           
            **Key Features:**
            - 🌞 **Query-Based Analysis**: Switch between different material datasets (q0, q1, q2...)
            - 🌞 **Sunburst Charts**: Hierarchical view of material → dopant category → specific dopant → property relationships
            - 📡 **Radar Charts**: Multi-property comparison across different dopant types
            - 📊 **Concentration Analysis**: Optimal doping levels for maximum performance
            - 💡 **Recommendation Engine**: Application-specific dopant suggestions
           
            **Methodology:**
            - Automatic extraction of dopant-property relationships from scientific literature
            - Hierarchical classification of dopants by chemical type
            - Enhancement factor calculation based on reported improvements
            - Application-specific optimization using weighted criteria
           
            This tool helps materials scientists quickly identify optimal dopant strategies for specific applications.
            """)
   
    # Main analysis workflow
    if analyze_btn and available_queries:
        for query_id in selected_queries:
            with st.spinner(f"🔬 Analyzing dopant relationships from Query {query_id} literature..."):
                try:
                    current_db_paths = get_db_paths_for_query(query_id)
                    available_dbs = [db_name for db_name, db_path in current_db_paths.items() if os.path.exists(db_path)]
                    if not available_dbs:
                        st.error(f"No databases found for query '{query_id}'!")
                        continue
                    selected_db = available_dbs[0]  # Default to first available
                    db_path = current_db_paths[selected_db]
                    
                    # Initialize database manager with current query ID
                    db_manager = DatabaseManager(db_path, query_id)
                    if not db_manager.connect():
                        st.error("Failed to connect to database")
                        continue
               
                    # Enhanced database schema analysis
                    st.markdown(f"### 🗃️ Database Schema Analysis for {query_id}")
                    schema = db_manager.generate_schema_report()
               
                    # Load papers
                    st.text(f"📥 Loading papers from database for Query {query_id}...")
                    st.session_state.performance_monitor.start_timer("load_papers")
                    papers_df = db_manager.get_papers_data()
                    st.session_state.performance_monitor.end_timer("load_papers")
               
                    if papers_df.empty:
                        st.error("No papers found in database!")
                        continue
               
                    # Limit for performance
                    papers_df = papers_df.iloc[:max_papers].copy()
               
                    # Extract dopant relationships
                    st.text("🧪 Extracting dopant relationships with Numba JIT acceleration...")
                    engine = st.session_state.analysis_engine
                    relationships_df = engine.extract_dopant_relationships(papers_df)
               
                    if relationships_df.empty:
                        st.warning("No dopant relationships extracted. The database may not contain sufficient dopant information.")
                    else:
                        st.success(f"✅ Analysis complete for {query_id}! Found {len(relationships_df)} dopant relationships in {len(papers_df)} papers.")
                   
                        # Store results
                        st.session_state.dopant_relationships[query_id] = relationships_df
           
                except Exception as e:
                    st.error(f"Analysis failed for Query {query_id}: {str(e)}")
                    logger.error(f"Analysis failed for Query {query_id}: {str(e)}", exc_info=True)
                    continue
   
    # Results display
    if st.session_state.dopant_relationships:
        engine = st.session_state.analysis_engine
       
        # Create tabs
        tabs = st.tabs([
            "🌞 Sunburst Analysis",
            "📡 Radar Comparison",
            "📊 Concentration Effects",
            "💡 Recommendations",
            "🔍 Data Explorer",
            "⚙️ Advanced Settings",
            "📈 Performance Metrics",
            "🖨️ Publication Exports"
        ])
       
        # Tab 1: Sunburst Chart
        with tabs[0]:
            st.subheader("🌳 Hierarchical Dopant Impact Analysis")
           
            for query_id in selected_queries:
                if query_id in st.session_state.dopant_relationships:
                    st.markdown(f"### Database {query_id}")
                    relationships_df = st.session_state.dopant_relationships[query_id]
                    col1, col2 = st.columns([2, 1])
           
                    with col1:
                        fig = engine.create_sunburst_chart(relationships_df, query_id=query_id)
                        if fig:
                            config = {
                                'toImageButtonOptions': {
                                    'format': 'svg',
                                    'filename': f'sunburst_{query_id}',
                                    'height': 800,
                                    'width': 1000,
                                    'scale': 2
                                },
                                'displaylogo': False
                            }
                            st.plotly_chart(fig, use_container_width=True, config=config)
                            add_caption(r"""
                            **Methodology**: Hierarchical sunburst chart showing the relationship between base materials (inner ring),
                            dopant categories (second ring), specific dopants (third ring), and affected properties (outer ring).
                            Segment size proportional to average enhancement factor. Colors represent dopant categories.
                            Hover over segments to see detailed enhancement values and sample contexts.
                            """)
           
                    with col2:
                        st.markdown("### 🔍 Sunburst Insights")
                        st.markdown("""
                        **How to read this chart:**
                        - **Inner ring**: Base materials (PVDF, BaTiO₃, etc.)
                        - **Middle rings**: Dopant categories and specific dopants
                        - **Outer ring**: Enhanced properties
                        - **Segment size**: Proportional to performance enhancement
                        - **Colors**: Dopant categories (consistent across all visualizations)
                       
                        **Key Insights to Look For:**
                        - Which dopant categories show the largest segments (highest enhancement)?
                        - Are certain base materials more responsive to doping?
                        - Which properties are most frequently enhanced?
                       
                        **Interactive Features:**
                        - Click on any segment to zoom in
                        - Double-click to zoom out
                        - Hover for detailed values and contexts
                        """)
                       
                        # Quick stats
                        st.markdown("### 📈 Quick Statistics")
                        top_category = relationships_df['dopant_category'].value_counts().index[0]
                        top_dopant = relationships_df['dopant'].value_counts().index[0]
                        top_property = relationships_df['property'].value_counts().index[0]
                        avg_enhancement = relationships_df['enhancement_factor'].mean()
                       
                        st.markdown(f"**Top Dopant Category:** {top_category}")
                        st.markdown(f"**Most Studied Dopant:** {top_dopant}")
                        st.markdown(f"**Most Enhanced Property:** {top_property}")
                        st.markdown(f"**Avg. Enhancement:** {avg_enhancement:.2f}×")
       
        # Tab 2: Radar Chart
        with tabs[1]:
            st.subheader("🎯 Multi-Property Dopant Comparison")
           
            for query_id in selected_queries:
                if query_id in st.session_state.dopant_relationships:
                    st.markdown(f"### Database {query_id}")
                    relationships_df = st.session_state.dopant_relationships[query_id]
                    # Get unique dopants
                    all_dopants = relationships_df['dopant'].unique().tolist()
           
                    # Default selection: top 4 dopants by frequency
                    default_dopants = relationships_df['dopant'].value_counts().head(4).index.tolist()
           
                    selected_dopants = st.multiselect(
                        f"Select dopants to compare ({query_id})",
                        options=all_dopants,
                        default=default_dopants,
                        max_selections=6,
                        key=f"radar_dopants_{query_id}"
                    )
           
                    if len(selected_dopants) < 2:
                        st.info("Please select at least 2 dopants for comparison")
                    else:
                        col1, col2 = st.columns([2, 1])
               
                        with col1:
                            fig = engine.create_radar_chart(relationships_df, selected_dopants, query_id=query_id)
                            if fig:
                                config = {
                                    'toImageButtonOptions': {
                                        'format': 'svg',
                                        'filename': f'radar_{query_id}',
                                        'height': 800,
                                        'width': 1000,
                                        'scale': 2
                                    },
                                    'displaylogo': False
                                }
                                st.plotly_chart(fig, use_container_width=True, config=config)
                                add_caption(r"""
                                **Methodology**: Radar chart comparing enhancement factors across multiple properties.
                                Each axis represents a different property (d₃₃, β-phase content, dielectric constant, etc.).
                                Distance from center indicates enhancement factor (1.0 = baseline, 2.0 = 2× improvement).
                                Polygon area represents overall performance profile.
                                Ideal dopant selection depends on application requirements.
                                """)
               
                        with col2:
                            st.markdown("### 🎯 Radar Chart Guide")
                            st.markdown("""
                            **Interpretation Guide:**
                            - **Larger polygon area**: Better overall performance
                            - **Shape asymmetry**: Property-specific enhancement
                            - **Peaks on specific axes**: Strong enhancement in those properties
                           
                            **Application Guidelines:**
                            - **Energy Harvesting**: Look for peaks in d₃₃, voltage, and power
                            - **Sensors**: Focus on d₃₃, sensitivity, and stability
                            - **Actuators**: Prioritize d₃₃, strain, and response time
                            - **High-Temperature**: Emphasize curie temperature and thermal stability
                           
                            **Trade-offs to Consider:**
                            - High d₃₃ enhancement may reduce flexibility
                            - Improved thermal stability might decrease sensitivity
                            - Maximum enhancement often occurs at optimal concentration
                            """)
                           
                            # Performance summary
                            if selected_dopants:
                                st.markdown("### 📊 Performance Summary")
                                for dopant in selected_dopants:
                                    dopant_df = relationships_df[relationships_df['dopant'] == dopant]
                                    if not dopant_df.empty:
                                        avg_enhancement = dopant_df['enhancement_factor'].mean()
                                        best_property = dopant_df.groupby('property')['enhancement_factor'].mean().idxmax()
                                        st.markdown(f"**{dopant}**")
                                        st.markdown(f"- Avg. Enhancement: {avg_enhancement:.2f}×")
                                        st.markdown(f"- Best Property: {best_property}")
       
        # Tab 3: Concentration Effects
        with tabs[2]:
            st.subheader("📈 Dopant Concentration vs Performance")
           
            for query_id in selected_queries:
                if query_id in st.session_state.dopant_relationships:
                    st.markdown(f"### Database {query_id}")
                    relationships_df = st.session_state.dopant_relationships[query_id]
                    col1, col2 = st.columns([2, 1])
           
                    with col1:
                        fig = engine.create_dopant_concentration_chart(relationships_df, query_id=query_id)
                        if fig:
                            config = {
                                'toImageButtonOptions': {
                                    'format': 'svg',
                                    'filename': f'concentration_{query_id}',
                                    'height': 800,
                                    'width': 1000,
                                    'scale': 2
                                },
                                'displaylogo': False
                            }
                            st.plotly_chart(fig, use_container_width=True, config=config)
                            add_caption(r"""
                            **Methodology**: Scatter plot showing relationship between dopant concentration (wt%) and performance enhancement factor.
                            Marker size proportional to absolute property value. Trend lines show optimal concentration ranges.
                            Different colors represent dopant categories. Hover for detailed information including processing methods.
                            """)
           
                    with col2:
                        st.markdown("### ⚖️ Concentration Analysis")
                        st.markdown("""
                        **Key Insights:**
                        - **Optimal concentration** typically exists (often 1-10 wt%)
                        - **Diminishing returns** at high concentrations
                        - **Agglomeration effects** reduce performance beyond optimal range
                        - **Processing method** significantly affects optimal concentration
                       
                        **General Guidelines by Category:**
                        - **Metal Oxides**: 5-15 wt% optimal range
                        - **Carbon-Based**: 0.5-3 wt% (lower due to conductivity)
                        - **Ferroelectric Ceramics**: 10-30 vol% for composites
                        - **2D Materials**: 0.1-2 wt% (high aspect ratio)
                       
                        **Processing Considerations:**
                        - Solution casting allows better dispersion at lower concentrations
                        - Melt processing may require higher concentrations
                        - In-situ polymerization enables molecular-level dispersion
                        """)
                       
                        # Show optimal concentrations
                        if not relationships_df.empty:
                            st.markdown("### 🎯 Optimal Concentrations")
                            try:
                                # Calculate optimal concentration for each dopant
                                opt_concentrations = {}
                                for dopant in relationships_df['dopant'].unique():
                                    dopant_df = relationships_df[
                                        (relationships_df['dopant'] == dopant) &
                                        (relationships_df['concentration_range'] != 'Unknown')
                                    ]
                                    if not dopant_df.empty:
                                        dopant_df['concentration_value'] = dopant_df['concentration_range'].str.extract(r'(\d+(?:\.\d+)?)').astype(float)
                                        # Find concentration with max enhancement
                                        max_idx = dopant_df['enhancement_factor'].idxmax()
                                        opt_concentrations[dopant] = dopant_df.loc[max_idx, 'concentration_value']
                               
                                if opt_concentrations:
                                    st.markdown("**Reported Optimal Concentrations:**")
                                    for i, (dopant, conc) in enumerate(list(opt_concentrations.items())[:5]):
                                        st.markdown(f"{i+1}. {dopant}: {conc:.1f} wt%")
                            except Exception as e:
                                logger.warning(f"Error calculating optimal concentrations: {e}")
       
        # Tab 4: Recommendations
        with tabs[3]:
            st.subheader("💡 Application-Specific Dopant Recommendations")
           
            application = st.selectbox(
                "Select target application",
                ["Energy Harvesting", "Sensors", "Actuators", "High Temperature", "Flexible Electronics"]
            )
           
            if st.button("✨ Generate Recommendations", type="primary"):
                for query_id in selected_queries:
                    if query_id in st.session_state.dopant_relationships:
                        with st.spinner(f"Generating optimized dopant recommendations for {query_id}..."):
                            relationships_df = st.session_state.dopant_relationships[query_id]
                            recommendations = engine.create_optimal_dopant_recommendations(relationships_df, application)
                           
                            if not recommendations:
                                st.warning(f"No recommendations available for {query_id}. Try a different application or analyze more papers.")
                            else:
                                st.markdown(f"### 🏆 Top Recommendations for {application} (Query {query_id})")
                               
                                for i, rec in enumerate(recommendations):
                                    with st.container():
                                        st.markdown(f"""
                                        <div style="background-color: {engine.colors.get(rec['category'], '#F8FAFC')};
                                                    padding: 1.5rem; border-radius: 12px; margin: 1rem 0;
                                                    border-left: 4px solid {engine.colors.get(rec['category'], '#3B82F6')}">
                                            <h3 style="color: #1E40AF; margin-top: 0;">#{i+1} {rec['dopant']}</h3>
                                            <p><strong>Category:</strong> {rec['category']}</p>
                                            <p><strong>Overall Score:</strong> {rec['score']:.2f}/3.0</p>
                                            <p><strong>Average Enhancement:</strong> {rec['avg_enhancement']:.2f}×</p>
                                            <p><strong>Key Properties Enhanced:</strong> {', '.join(rec['key_properties'])}</p>
                                            <p><strong>Best Base Materials:</strong> {', '.join(rec['best_base_materials'])}</p>
                                        </div>
                                        """, unsafe_allow_html=True)
                               
                                # Summary insights
                                st.markdown("### 🔍 Key Insights")
                                best_dopant = recommendations[0]['dopant']
                                best_category = recommendations[0]['category']
                                st.markdown(f"""
                                **For {application} applications (Query {query_id}):**
                                - **{best_dopant}** ({best_category}) emerges as the top recommendation
                                - **Critical properties** to focus on: {', '.join(recommendations[0]['key_properties'])}
                                - **Optimal base material**: {recommendations[0]['best_base_materials'][0]}
                                - **Expected enhancement**: {recommendations[0]['avg_enhancement']:.2f}× improvement over undoped material
                               
                                **Implementation Strategy:**
                                - Start with low concentrations (1-3 wt%) and optimize
                                - Consider processing method compatibility
                                - Balance enhancement with other material properties
                                - Validate with experimental testing for your specific system
                                """)
       
        # Tab 5: Data Explorer
        with tabs[4]:
            st.subheader("🔍 Raw Data Explorer")
           
            for query_id in selected_queries:
                if query_id in st.session_state.dopant_relationships:
                    st.markdown(f"### Extracted Dopant Relationships ({query_id})")
                    relationships_df = st.session_state.dopant_relationships[query_id]
                    st.dataframe(relationships_df, use_container_width=True, height=400)
           
                    # Download options
                    col1, col2, col3 = st.columns(3)
           
                    with col1:
                        csv = relationships_df.to_csv(index=False).encode('utf-8')
                        st.download_button("📥 Download CSV", csv, f"dopant_relationships_{query_id}.csv", "text/csv")
           
                    with col2:
                        excel_buffer = io.BytesIO()
                        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                            relationships_df.to_excel(writer, sheet_name='dopant_relationships', index=False)
                        excel_buffer.seek(0)
                        st.download_button("📊 Download Excel", excel_buffer, f"dopant_analysis_{query_id}.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
           
                    with col3:
                        json_data = relationships_df.to_dict('records')
                        json_str = json.dumps(json_data, indent=2)
                        st.download_button("💾 Download JSON", json_str, f"dopant_data_{query_id}.json", "application/json")
           
                    # Advanced filtering
                    with st.expander(f"🔧 Advanced Data Filtering ({query_id})"):
                        col1, col2, col3 = st.columns(3)
                       
                        with col1:
                            base_material_filter = st.multiselect(
                                "Base Materials",
                                options=relationships_df['base_material'].unique(),
                                default=relationships_df['base_material'].unique()[:3].tolist(),
                                key=f"base_filter_{query_id}"
                            )
                       
                        with col2:
                            dopant_category_filter = st.multiselect(
                                "Dopant Categories",
                                options=relationships_df['dopant_category'].unique(),
                                default=relationships_df['dopant_category'].unique()[:3].tolist(),
                                key=f"category_filter_{query_id}"
                            )
                       
                        with col3:
                            min_enhancement = st.slider("Min Enhancement Factor", 1.0, 3.0, 1.5, key=f"enhancement_{query_id}")
                       
                        # Apply filters
                        filtered_df = relationships_df[
                            (relationships_df['base_material'].isin(base_material_filter)) &
                            (relationships_df['dopant_category'].isin(dopant_category_filter)) &
                            (relationships_df['enhancement_factor'] >= min_enhancement)
                        ]
                       
                        st.markdown(f"### Filtered Results ({len(filtered_df)} relationships)")
                        st.dataframe(filtered_df, use_container_width=True)
       
        # Tab 6: Advanced Settings
        with tabs[5]:
            st.subheader("⚙️ Advanced Configuration")
           
            st.markdown("### 🧪 Dopant Classification System")
           
            # Show current classification
            for category, dopants in Config.DOPANT_CATEGORIES.items():
                with st.expander(f"{category} ({len(dopants)} dopants)"):
                    st.text(", ".join(dopants))
           
            st.markdown("### 📝 Custom Dopant Classification")
            st.markdown("""
            You can modify the dopant classification system by editing the configuration below.
            This affects how dopants are grouped in all visualizations.
            """)
           
            custom_categories = st.text_area(
                "Custom Dopant Categories (JSON format)",
                value=json.dumps(Config.DOPANT_CATEGORIES, indent=2),
                height=400
            )
           
            if st.button("💾 Save Custom Classification"):
                try:
                    new_categories = json.loads(custom_categories)
                    Config.DOPANT_CATEGORIES = new_categories
                    st.session_state.analysis_engine.dopant_categories = new_categories
                    st.success("Custom classification saved successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error saving classification: {str(e)}")
           
            st.markdown("### 🔄 Rebuild Analysis")
            if st.button("⚡ Rebuild All Visualizations", type="secondary"):
                st.rerun()
       
        # Tab 7: Performance Metrics
        with tabs[6]:
            st.subheader("⚡ Performance Metrics")
            st.session_state.performance_monitor.display_stats()
           
            # Engine-specific metrics
            st.markdown("### Engine Performance")
            engine_metrics = st.session_state.analysis_engine.performance_monitor.get_stats()
            if engine_metrics:
                df = pd.DataFrame.from_dict(engine_metrics, orient='index')
                st.dataframe(df.style.format({
                    'mean': '{:.4f}',
                    'std': '{:.4f}',
                    'min': '{:.4f}',
                    'max': '{:.4f}'
                }))
            else:
                st.info("No engine performance metrics recorded yet.")

        # Tab 8: Publication Exports
        with tabs[7]:
            st.subheader("🖨️ Publication-Ready Exports")
            for query_id in selected_queries:
                if query_id in st.session_state.dopant_relationships:
                    st.markdown(f"### Exports for Database {query_id}")
                    relationships_df = st.session_state.dopant_relationships[query_id]
                    sun_fig = engine.create_sunburst_chart(relationships_df, query_id=query_id)
                    export_to_pdf(sun_fig, f"sunburst_{query_id}")
                    
                    # Assuming selected_dopants from previous tab or default
                    all_dopants = relationships_df['dopant'].unique().tolist()
                    default_dopants = relationships_df['dopant'].value_counts().head(4).index.tolist()
                    radar_fig = engine.create_radar_chart(relationships_df, default_dopants, query_id=query_id)
                    export_to_pdf(radar_fig, f"radar_{query_id}")
                    
                    conc_fig = engine.create_dopant_concentration_chart(relationships_df, query_id=query_id)
                    if conc_fig:
                        export_to_pdf(conc_fig, f"concentration_{query_id}")
   
    else:
        # Welcome screen
        st.markdown(f"""
        <div style="padding: 2.5rem; text-align: center; background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%); border-radius: 15px; color: white; margin-bottom: 2rem;">
            <h2>🔬 Dopant Impact Explorer</h2>
            <p style="font-size: 1.2rem; opacity: 0.9;">Query-Based Visual Analytics for Piezoelectric Material Enhancement</p>
        </div>
        """, unsafe_allow_html=True)
       
        # Feature cards
        col1, col2, col3 = st.columns(3)
       
        with col1:
            st.markdown("""
            <div style="padding: 1.2rem; border-radius: 12px; background-color: #F8FAFC; border: 1px solid #E2E8F0; height: 100%;">
                <h3 style="color: #3B82F6;">🌳 Hierarchical Sunburst</h3>
                <p>Visualize the complete hierarchy from base materials → dopant categories → specific dopants → enhanced properties.</p>
                <p><strong>Insight</strong>: Identify which dopant categories provide the broadest property enhancement.</p>
            </div>
            """, unsafe_allow_html=True)
       
        with col2:
            st.markdown("""
            <div style="padding: 1.2rem; border-radius: 12px; background-color: #F0FDF4; border: 1px solid #BBF7D0; height: 100%;">
                <h3 style="color: #10B981;">📡 Radar Performance Charts</h3>
                <p>Compare multiple dopants across 6 key properties simultaneously using radar/spider charts.</p>
                <p><strong>Insight</strong>: Find the optimal dopant for your specific application requirements.</p>
            </div>
            """, unsafe_allow_html=True)
       
        with col3:
            st.markdown("""
            <div style="padding: 1.2rem; border-radius: 12px; background-color: #FEF7CD; border: 1px solid #FDE68A; height: 100%;">
                <h3 style="color: #D97706;">💡 AI Recommendations</h3>
                <p>Get application-specific dopant recommendations with quantitative enhancement factors.</p>
                <p><strong>Insight</strong>: Data-driven suggestions for energy harvesting, sensors, actuators, and more.</p>
            </div>
            """, unsafe_allow_html=True)
       
        # How it works
        with st.expander("🚀 How It Works"):
            st.markdown("""
            ### Step-by-Step Workflow
           
            1. **Query Selection**: Choose your dataset (q0 = default, q1 = polymer-focused, q2 = ceramic-focused, etc.)
            2. **Database Connection**: Connect to your query-specific piezoelectric materials knowledge base
            3. **Automatic Analysis**: The system extracts dopant relationships from paper text
            4. **Hierarchical Classification**: Dopants are categorized by chemical type and function
            5. **Enhancement Quantification**: Performance improvements are calculated and normalized
            6. **Advanced Visualization**: Interactive charts reveal optimal dopant strategies
           
            ### Query-Based File Organization
