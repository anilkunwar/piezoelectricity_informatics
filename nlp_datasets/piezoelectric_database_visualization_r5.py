# dopant_impact_explorer_enhanced_with_query_support.py
import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import os
import json
import logging
import time
import io
import base64
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.colors as mcolors
from wordcloud import WordCloud
import scienceplots  # For publication-quality matplotlib styles

# Try to set science style for matplotlib
try:
    plt.style.use(['science', 'ieee', 'grid'])
except:
    plt.style.use('default')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("DopantImpactExplorer")

# Set page config
st.set_page_config(
    page_title="Dopant Impact Explorer Pro",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced custom CSS with better styling
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    color: #1E3A8A;
    text-align: center;
    margin-bottom: 2rem;
    font-weight: 700;
    background: linear-gradient(90deg, #1E3A8A 0%, #3B82F6 50%, #60A5FA 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.sub-header {
    font-size: 1.5rem;
    color: #4B5563;
    text-align: center;
    margin-bottom: 2rem;
    font-weight: 400;
}
.metric-card {
    background: linear-gradient(135deg, #F8FAFC 0%, #EFF6FF 100%);
    padding: 1.5rem;
    border-radius: 15px;
    border: 1px solid #E5E7EB;
    margin: 0.5rem 0;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    transition: all 0.3s ease;
}
.metric-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
}
.figure-caption {
    font-size: 1rem;
    color: #6B7280;
    margin-top: 0.5rem;
    margin-bottom: 2rem;
    font-style: italic;
    line-height: 1.6;
    padding: 1rem;
    background-color: #F9FAFB;
    border-left: 4px solid #3B82F6;
    border-radius: 0 8px 8px 0;
}
.stTabs [data-baseweb="tab-list"] {
    gap: 2px;
}
.stTabs [data-baseweb="tab"] {
    height: 60px;
    white-space: pre-wrap;
    background-color: #F1F5F9;
    border-radius: 10px 10px 0px 0px;
    padding: 15px 25px;
    font-weight: 600;
    font-size: 1.1rem;
    transition: all 0.3s ease;
}
.stTabs [data-baseweb="tab"]:hover {
    background-color: #E0E7FF;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #3B82F6 0%, #1D4ED8 100%);
    color: white;
    box-shadow: 0 4px 6px -1px rgba(59, 130, 246, 0.3);
}
.expander-header {
    font-weight: bold;
    color: #1E40AF;
    font-size: 1.2rem;
}
.plot-container {
    background-color: white;
    padding: 2rem;
    border-radius: 15px;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    margin-bottom: 2rem;
}
.download-btn {
    background: linear-gradient(135deg, #10B981 0%, #047857 100%);
    color: white;
    border: none;
    padding: 10px 20px;
    border-radius: 8px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.3s ease;
}
.download-btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 12px rgba(16, 185, 129, 0.3);
}
.custom-file-input {
    padding: 10px;
    border: 2px dashed #3B82F6;
    border-radius: 8px;
    background-color: #F0F9FF;
}
</style>
""", unsafe_allow_html=True)

def add_caption(text: str, icon: str = "📝"):
    """Add a styled caption below a figure"""
    st.markdown(f'<div class="figure-caption">{icon} {text}</div>', unsafe_allow_html=True)

def get_table_download_link(df, filename, text):
    """Generate a download link for DataFrame"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="file/csv;base64,{b64}" download="{filename}" class="download-btn">{text}</a>'
    return href

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
        import psutil, platform
        if platform.system() == "Windows":
            return psutil.Process().memory_info().rss / (1024 * 1024)
        else:
            import resource
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
        if time.time() - self.access_times[key] > self.ttl_seconds:
            del self.cache[key]
            del self.access_times[key]
            logger.debug(f"Cache miss (expired): {key}")
            return None
        self.access_times[key] = time.time()
        logger.debug(f"Cache hit: {key}")
        return self.cache[key]

    def set(self, key: str, value: Any):
        """Set value in cache with eviction if needed"""
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
        """Calculate cache hit rate (placeholder)"""
        return 0.85

# ==============================
# QUERY-BASED DATABASE PATH HANDLING
# ==============================
DB_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DB_DIR = os.path.join(DB_DIR, "knowledge_database")
os.makedirs(DEFAULT_DB_DIR, exist_ok=True)

def get_db_paths_for_query(query_id: str = "q0") -> dict:
    """
    Get database paths for a specific query dataset.
    query_id = "q0" for default, "q1" for query1, etc.
    """
    suffix = f"{query_id}_" if query_id != "q0" else ""
    return {
        "Metadata DB": os.path.join(DEFAULT_DB_DIR, f"piezoelectricity{suffix}metadata.db"),
        "Universe DB": os.path.join(DEFAULT_DB_DIR, f"piezoelectricity{suffix}universe.db"),
        "PDF Storage DB": os.path.join(DEFAULT_DB_DIR, f"piezoelectricity{suffix}pdfs.db")
    }

class Config:
    """Enhanced configuration with query support and publication settings"""
    DEFAULT_QUERY_ID = "q0"
    DEFAULT_DB_PATHS = get_db_paths_for_query(DEFAULT_QUERY_ID)

    @classmethod
    def get_available_query_datasets(cls) -> list:
        """Detect available query datasets by checking database files"""
        query_datasets = ["q0"]
        for i in range(1, 10):
            query_id = f"q{i}"
            db_paths = get_db_paths_for_query(query_id)
            if any(os.path.exists(path) for path in db_paths.values()):
                query_datasets.append(query_id)
        return query_datasets

    COLOR_PALETTES = {
        "nature": ["#E64B35", "#4DBBD5", "#00A087", "#3C5488", "#F39B7F", "#8491B4", "#91D1C2", "#DC0000"],
        "science": ["#1F77B4", "#FF7F0E", "#2CA02C", "#D62728", "#9467BD", "#8C564B", "#E377C2", "#7F7F7F"],
        "material_science": ["#3A6EA5", "#FF6B35", "#004E89", "#FFA400", "#6699CC", "#FF7F50", "#33658A", "#FF9F1C"],
        "categorical_10": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                           "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
    }

    DOPANT_CATEGORIES = {
        "Metal Oxides": ["ZnO", "BaTiO₃", "TiO₂", "SnO₂", "Al₂O₃", "Fe₂O₃", "CuO", "MgO", "CaO", "ZrO₂"],
        "Carbon-Based": ["CNT", "Graphene", "Carbon Black", "Graphene Oxide", "Reduced Graphene Oxide", "Carbon Nanofibers"],
        "Ferroelectric Ceramics": ["PZT", "BTO", "KNN", "BNKT", "LSMO", "PMN-PT", "PZT-NKN"],
        "2D Materials": ["MoS₂", "WS₂", "MXene", "h-BN", "Phosphorene", "MoSe₂", "WSe₂"],
        "Polymers": ["PVA", "PMMA", "PEO", "PVP", "PEDOT:PSS", "PANi", "PPy"],
        "Nanoparticles": ["Ag NPs", "Au NPs", "SiO₂ NPs", "TiO₂ NPs", "Fe₃O₄ NPs"],
        "Ionic Liquids": ["BMIM-PF₆", "EMIM-TFSI", "HMIM-Cl", "BMIM-BF₄"],
        "Others": ["Cellulose", "Clay", "Silica", "Quantum Dots", "Perovskites"]
    }

    BASE_MATERIALS = {
        "PVDF": ["pvdf", "polyvinylidene fluoride", "poly(vinylidene fluoride)", "pvdf-trfe"],
        "BaTiO₃": ["barium titanate", "batio₃", "batio3", "BaTiO₃"],
        "ZnO": ["zinc oxide", "zno", "ZnO"],
        "PZT": ["lead zirconate titanate", "pzt", "Pb(Zr,Ti)O₃", "Pb(Zr,Ti)O3"],
        "AlN": ["aluminum nitride", "aln", "AlN"],
        "KNN": ["potassium sodium niobate", "knn", "K₀.₅Na₀.₅NbO₃"],
        "PVDF-HFP": ["pvdf-hfp", "poly(vinylidene fluoride-co-hexafluoropropylene)"],
        "Others": ["polymer", "ceramic", "composite", "nanocomposite"]
    }

    DOPANT_PROPERTIES = {
        "d₃₃ (pC/N)": ["d33", "d₃₃", "piezoelectric coefficient", "d33 coefficient"],
        "β-phase (%)": ["beta phase", "β-phase", "beta content", "crystallinity", "ferroelectric phase"],
        "Dielectric Constant": ["dielectric constant", "permittivity", "εr", "relative permittivity"],
        "Young's Modulus (GPa)": ["young's modulus", "tensile strength", "elastic modulus", "mechanical strength"],
        "Conductivity (S/m)": ["conductivity", "electrical conductivity", "resistivity", "impedance"],
        "Curie Temp (°C)": ["curie temperature", "tc", "phase transition temperature", "thermal stability"],
        "Voltage Output (V)": ["voltage output", "open circuit voltage", "output voltage"],
        "Power Density (μW/cm²)": ["power density", "power output", "energy harvesting efficiency"]
    }

    COLORS = {
        "PVDF": "#3A6EA5",
        "BaTiO₃": "#FF6B35",
        "ZnO": "#004E89",
        "PZT": "#FFA400",
        "AlN": "#6699CC",
        "KNN": "#FF7F50",
        "PVDF-HFP": "#33658A",
        "Others": "#FF9F1C",
        "Metal Oxides": "#1F77B4",
        "Carbon-Based": "#FF7F0E",
        "Ferroelectric Ceramics": "#2CA02C",
        "2D Materials": "#D62728",
        "Polymers": "#9467BD",
        "Nanoparticles": "#8C564B",
        "Ionic Liquids": "#E377C2",
        "Others": "#7F7F7F"
    }

    PLOT_CONFIG = {
        "font_family": "Arial",
        "font_size": 14,
        "title_font_size": 22,
        "axis_font_size": 16,
        "legend_font_size": 14,
        "colorway": ["#1F77B4", "#FF7F0E", "#2CA02C", "#D62728", "#9467BD",
                     "#8C564B", "#E377C2", "#7F7F7F", "#BCBD22", "#17BECF"],
        "template": "plotly_white",
        "width": 900,
        "height": 700
    }

# ==============================
# DATABASE MANAGER WITH QUERY SUPPORT
# ==============================
class DatabaseManager:
    """Manages database connections with enhanced error handling and dynamic schema detection"""
    def __init__(self, db_path: str, query_id: str = "q0"):
        self.db_path = db_path
        self.query_id = query_id
        self.conn = None
        self.table_columns = {}
        logger.info(f"Database manager initialized for {db_path} (Query: {query_id})")

    def connect(self) -> bool:
        """Establish database connection"""
        try:
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self.conn.row_factory = sqlite3.Row
            logger.info(f"Connected to database: {self.db_path}")
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
        """Get list of tables in database"""
        if not self.conn:
            if not self.connect():
                return []
        try:
            query = "SELECT name FROM sqlite_master WHERE type='table';"
            tables = pd.read_sql_query(query, self.conn)
            return tables['name'].tolist()
        except Exception as e:
            logger.error(f"Error fetching tables: {e}")
            st.error(f"Error fetching tables: {e}")
            return []

    def get_papers_data(self) -> pd.DataFrame:
        """Get papers data with dynamic schema handling"""
        tables = self.get_tables()
        target_table = None
        available_columns = []

        # Determine best table
        for candidate in ["papers_fulltext", "papers", "documents"]:
            if candidate in tables:
                target_table = candidate
                available_columns = self.get_columns(candidate)
                break

        if not target_table:
            for table in tables:
                cols = self.get_columns(table)
                if any(col in cols for col in ['title', 'abstract', 'content', 'text']):
                    target_table = table
                    available_columns = cols
                    break

        if not target_table:
            st.error("No suitable paper table found.")
            return pd.DataFrame()

        # Determine text column
        required_text_columns = ['full_text', 'abstract', 'content', 'text']
        text_column = next((col for col in required_text_columns if col in available_columns), None)
        if not text_column:
            st.error("No text column found.")
            return pd.DataFrame()

        # Build column mapping
        select_columns = []
        standard_columns = ['paper_id', 'id', 'title', 'abstract', 'full_text', 'content', 'text',
                            'year', 'date', 'categories', 'keywords', 'authors', 'journal', 'doi']
        for col in standard_columns:
            if col in available_columns and col not in select_columns:
                select_columns.append(col)
        if text_column not in select_columns:
            select_columns.append(text_column)

        column_mapping = {}
        if 'paper_id' not in available_columns and 'id' in available_columns:
            column_mapping['id'] = 'paper_id'
        if 'full_text' not in available_columns and 'content' in available_columns:
            column_mapping['content'] = 'full_text'
        if 'abstract' not in available_columns and 'summary' in available_columns:
            column_mapping['summary'] = 'abstract'
        if 'year' not in available_columns and 'date' in available_columns:
            column_mapping['date'] = 'year'

        select_clause = ", ".join([f"{col} AS {column_mapping[col]}" if col in column_mapping else col for col in select_columns])
        where_clauses = []
        if text_column:
            where_clauses.append(f"({text_column} IS NOT NULL AND LENGTH({text_column}) > 100)")
        if 'abstract' in available_columns and 'abstract' != text_column:
            where_clauses.append(f"(abstract IS NOT NULL AND LENGTH(abstract) > 50)")
        where_clause = " OR ".join(where_clauses) if where_clauses else "1=1"

        query = f"""
        SELECT {select_clause}
        FROM {target_table}
        WHERE {where_clause}
        LIMIT 2000
        """
        logger.debug(f"Executing query: {query}")

        try:
            df = pd.read_sql_query(query, self.conn)
            if 'date' in df.columns and 'year' not in df.columns:
                try:
                    df['year'] = pd.to_datetime(df['date']).dt.year
                except:
                    df['year'] = 2023
            if 'paper_id' not in df.columns:
                df['paper_id'] = df.get('id', range(1, len(df) + 1))
            if 'full_text' not in df.columns:
                for alt in ['content', 'text', 'abstract']:
                    if alt in df.columns:
                        df['full_text'] = df[alt]
                        break
                else:
                    df['full_text'] = ''
            logger.info(f"Loaded {len(df)} papers from {target_table}")
            st.success(f"Successfully loaded {len(df)} papers from {target_table}")
            return df
        except Exception as e:
            logger.error(f"Error fetching papers: {e}")
            st.error(f"Error: {str(e)}")
            return pd.DataFrame()

    def get_database_schema(self) -> Dict[str, List[str]]:
        """Get complete database schema"""
        schema = {}
        tables = self.get_tables()
        for table in tables:
            schema[table] = self.get_columns(table)
        return schema

    def generate_schema_report(self):
        """Generate a comprehensive schema report"""
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
                    try:
                        sample_query = f"SELECT * FROM {table} LIMIT 3"
                        sample_df = pd.read_sql_query(sample_query, self.conn)
                        st.markdown("**Sample Data (first 3 rows):**")
                        st.dataframe(sample_df)
                    except Exception as e:
                        st.warning(f"Could not fetch sample: {e}")

        total_tables = len(schema)
        total_columns = sum(len(cols) for cols in schema.values())
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Tables", total_tables)
        col2.metric("Total Columns", total_columns)
        col3.metric("Avg Columns/Table", f"{total_columns/total_tables:.1f}")

        text_columns = []
        for table, columns in schema.items():
            for col in columns:
                if any(kw in col.lower() for kw in ['text', 'content', 'abstract', 'full']):
                    text_columns.append(f"{table}.{col}")
        if text_columns:
            st.markdown("### 📝 Text Content Columns")
            for col in text_columns:
                st.markdown(f"- `{col}`")
        else:
            st.warning("No text content columns detected.")

        return schema

# ==============================
# ENHANCED DOPANT ANALYSIS ENGINE
# ==============================
class EnhancedDopantAnalysisEngine:
    """Enhanced analysis engine with publication-quality visualizations"""
    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.dopant_categories = self.config.DOPANT_CATEGORIES
        self.base_materials = self.config.BASE_MATERIALS
        self.properties = self.config.DOPANT_PROPERTIES
        self.colors = self.config.COLORS
        self.plot_config = self.config.PLOT_CONFIG
        self.performance_monitor = PerformanceMonitor()
        self.cache_manager = CacheManager()
        self._setup_matplotlib()
        logger.info("Enhanced dopant analysis engine initialized")

    def _setup_matplotlib(self):
        """Setup matplotlib for publication quality"""
        plt.rcParams.update({
            'font.family': 'serif',
            'font.serif': ['Times New Roman', 'DejaVu Serif'],
            'font.size': 12,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'legend.fontsize': 10,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'figure.dpi': 300,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.1
        })

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
        batch_size = max(1, min(50, total_papers // 4))

        for start_idx in range(0, total_papers, batch_size):
            end_idx = min(start_idx + batch_size, total_papers)
            batch_df = papers_df.iloc[start_idx:end_idx]
            for idx, row in batch_df.iterrows():
                text = str(row.get('full_text', '') or row.get('abstract', ''))
                if not text or len(text) < 50:
                    continue
                for category, dopants in self.dopant_categories.items():
                    for dopant in dopants:
                        if dopant.lower() in text.lower():
                            base_material = self.identify_base_material(text)
                            for prop_category, prop_terms in self.properties.items():
                                for term in prop_terms:
                                    if term.lower() in text.lower():
                                        prop_pos = text.lower().find(term.lower())
                                        if prop_pos != -1:
                                            context = text[max(0, prop_pos-50):min(len(text), prop_pos+100)]
                                            import re
                                            numbers = re.findall(r'[-+]?\d*\.\d+|\d+', context)
                                            if numbers:
                                                try:
                                                    value = float(numbers[0])
                                                    enhancement = 1.0
                                                    if 'enhanced' in context.lower() or 'improved' in context.lower():
                                                        enhancement = 1.5
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
        processing_methods = ["electrospinning", "solution casting", "hot pressing",
                              "melt blending", "in-situ polymerization", "ball milling"]
        text_lower = text.lower()
        for method in processing_methods:
            if method in text_lower:
                return method.title()
        return "Unknown"

    def create_publication_sunburst(self, relationships_df: pd.DataFrame,
                                    title: str = "Hierarchical Analysis of Dopant Effects",
                                    show_values: bool = True,
                                    max_depth: int = 4) -> go.Figure:
        if relationships_df.empty:
            return None
        hierarchy_levels = ['base_material', 'dopant_category', 'dopant', 'property']
        hierarchy_levels = hierarchy_levels[:max_depth]
        agg_data = relationships_df.groupby(hierarchy_levels).agg({
            'value': 'mean',
            'enhancement_factor': 'mean',
            'paper_id': 'count'
        }).reset_index()
        agg_data.rename(columns={'paper_id': 'n_studies'}, inplace=True)
        agg_data['size'] = agg_data['enhancement_factor'] * np.log1p(agg_data['n_studies'])
        min_enhance = agg_data['enhancement_factor'].min()
        max_enhance = agg_data['enhancement_factor'].max()
        agg_data['color_norm'] = (agg_data['enhancement_factor'] - min_enhance) / (max_enhance - min_enhance) if max_enhance > min_enhance else 0.5

        fig = px.sunburst(
            agg_data,
            path=hierarchy_levels,
            values='size',
            color='color_norm',
            color_continuous_scale='RdYlBu_r',
            range_color=[0, 1],
            title=title,
            height=self.plot_config["height"],
            width=self.plot_config["width"],
            branchvalues='total',
            maxdepth=max_depth,
            hover_data={
                'enhancement_factor': ':.2f',
                'value': ':.2f',
                'n_studies': True,
                'color_norm': False
            },
            labels={
                'enhancement_factor': 'Enhancement Factor',
                'value': 'Property Value',
                'n_studies': 'Number of Studies'
            }
        )
        fig.update_layout(
            title={
                'text': title,
                'y': 0.95,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': dict(size=self.plot_config["title_font_size"], family=self.plot_config["font_family"], color='#1E3A8A')
            },
            font=dict(size=self.plot_config["font_size"], family=self.plot_config["font_family"]),
            margin=dict(t=100, l=20, r=20, b=20),
            paper_bgcolor='white',
            plot_bgcolor='white',
            coloraxis_colorbar=dict(title="Enhancement<br>Factor", thickness=20, len=0.75, tickfont=dict(size=12), title_font=dict(size=14))
        )
        if show_values:
            fig.update_traces(
                textinfo='label+value+percent parent',
                texttemplate='<b>%{label}</b><br>%{value:.1f}<br>%{percentParent:.1%}',
                hovertemplate='<b>%{label}</b><br>Enhancement: %{color:.2f}×<br>Studies: %{customdata[2]:d}<br><extra></extra>'
            )
        return fig

    def create_enhanced_radar_chart(self, relationships_df: pd.DataFrame,
                                    selected_dopants: List[str],
                                    title: str = "Multi-Property Performance Comparison",
                                    show_average: bool = True,
                                    normalize: bool = True) -> go.Figure:
        if relationships_df.empty or not selected_dopants:
            return None
        filtered_df = relationships_df[relationships_df['dopant'].isin(selected_dopants)]
        if filtered_df.empty:
            return None
        properties = list(self.properties.keys())[:8]
        radar_data = {}
        for dopant in selected_dopants:
            dopant_df = filtered_df[filtered_df['dopant'] == dopant]
            if not dopant_df.empty:
                radar_data[dopant] = {}
                for prop in properties:
                    prop_df = dopant_df[dopant_df['property'] == prop]
                    if not prop_df.empty:
                        avg_enhance = prop_df['enhancement_factor'].mean()
                        n_studies = len(prop_df)
                        confidence = min(1.0, n_studies / 10)
                        radar_data[dopant][prop] = avg_enhance * confidence
                    else:
                        radar_data[dopant][prop] = 1.0

        if normalize and len(radar_data) > 1:
            for prop in properties:
                values = [data.get(prop, 1.0) for data in radar_data.values()]
                max_val = max(values)
                if max_val > 1.0:
                    for dopant in radar_data:
                        if prop in radar_data[dopant]:
                            radar_data[dopant][prop] = radar_data[dopant][prop] / max_val * 2.0

        fig = go.Figure()
        colors = self.config.COLOR_PALETTES["nature"][:len(selected_dopants)]
        for i, (dopant, props) in enumerate(radar_data.items()):
            values = [props.get(prop, 1.0) for prop in properties]
            values += values[:1]
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=properties + [properties[0]],
                fill='toself' if i == 0 else None,
                fillcolor=f"rgba{tuple(list(mcolors.to_rgba(colors[i]))[:3] + [0.2])}",
                name=dopant,
                line=dict(color=colors[i], width=3 if i == 0 else 2, dash='solid' if i == 0 else 'dash'),
                marker=dict(size=8, symbol='circle', line=dict(width=1, color='white')),
                hovertemplate=f"<b>{dopant}</b><br>%{theta}: %{r:.2f}×<br><extra></extra>",
                opacity=0.9
            ))

        if show_average and len(radar_data) > 1:
            avg_values = []
            for prop in properties:
                prop_values = [data.get(prop, 1.0) for data in radar_data.values()]
                avg_values.append(np.mean(prop_values))
            avg_values += avg_values[:1]
            fig.add_trace(go.Scatterpolar(
                r=avg_values,
                theta=properties + [properties[0]],
                name='Average',
                line=dict(color='black', width=3, dash='dashdot'),
                marker=dict(size=0),
                opacity=0.7
            ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, max(2.5, max([max(v.values()) for v in radar_data.values()]))], tickangle=0, tickfont=dict(size=12)),
                angularaxis=dict(tickfont=dict(size=14, family='Arial'), rotation=90, direction='clockwise')
            ),
            title={'text': title, 'y': 0.95, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top', 'font': dict(size=self.plot_config["title_font_size"], family=self.plot_config["font_family"], color='#1E3A8A')},
            height=self.plot_config["height"],
            width=self.plot_config["width"],
            font=dict(size=self.plot_config["font_size"], family=self.plot_config["font_family"]),
            legend=dict(font=dict(size=14), orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor='rgba(255, 255, 255, 0.9)'),
            margin=dict(t=100, b=80, l=80, r=80),
            paper_bgcolor='white',
            plot_bgcolor='white'
        )
        return fig

    def create_concentration_heatmap(self, relationships_df: pd.DataFrame,
                                     title: str = "Dopant Concentration vs Property Enhancement") -> go.Figure:
        if relationships_df.empty:
            return None
        relationships_df['concentration_num'] = relationships_df['concentration_range'].str.extract(r'(\d+(?:\.\d+)?)').astype(float)
        filtered_df = relationships_df[(relationships_df['concentration_num'] > 0) & (relationships_df['concentration_num'] <= 50)].copy()
        if filtered_df.empty:
            return None
        heatmap_data = filtered_df.pivot_table(values='enhancement_factor', index='dopant', columns=pd.cut(filtered_df['concentration_num'], bins=10), aggfunc='mean', fill_value=1.0)
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data.values,
            x=[str(col) for col in heatmap_data.columns],
            y=heatmap_data.index.tolist(),
            colorscale='Viridis',
            colorbar=dict(title="Enhancement<br>Factor", titleside="right", tickfont=dict(size=12)),
            hovertemplate="Dopant: %{y}<br>Concentration: %{x}<br>Enhancement: %{z:.2f}×<br><extra></extra>",
            text=heatmap_data.values.round(2),
            texttemplate="%{text}×",
            textfont=dict(size=10, color="white")
        ))
        fig.update_layout(
            title={'text': title, 'y': 0.95, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top', 'font': dict(size=22, family='Arial', color='#1E3A8A')},
            xaxis=dict(title="Concentration Range (%)", title_font=dict(size=16), tickfont=dict(size=12), tickangle=45),
            yaxis=dict(title="Dopant", title_font=dict(size=16), tickfont=dict(size=12)),
            height=600,
            width=900,
            margin=dict(t=100, b=100, l=150, r=50),
            paper_bgcolor='white',
            plot_bgcolor='white'
        )
        return fig

    def create_3d_scatter_plot(self, relationships_df: pd.DataFrame,
                               title: str = "3D Analysis: Dopant Effects on Multiple Properties") -> go.Figure:
        if relationships_df.empty:
            return None
        top_properties = relationships_df['property'].value_counts().head(3).index.tolist()
        if len(top_properties) < 3:
            return None
        fig = go.Figure()
        categories = relationships_df['dopant_category'].unique()
        for category in categories:
            cat_data = relationships_df[relationships_df['dopant_category'] == category]
            fig.add_trace(go.Scatter3d(
                x=cat_data[cat_data['property'] == top_properties[0]]['value'],
                y=cat_data[cat_data['property'] == top_properties[1]]['value'],
                z=cat_data[cat_data['property'] == top_properties[2]]['value'],
                mode='markers',
                name=category,
                marker=dict(size=8, color=self.colors.get(category, '#666666'), opacity=0.7, line=dict(width=1, color='white')),
                text=cat_data['dopant'] + '<br>' + cat_data['base_material'],
                hovertemplate=f"<b>%{text}</b><br>{top_properties[0]}: %{{x:.1f}}<br>{top_properties[1]}: %{{y:.1f}}<br>{top_properties[2]}: %{{z:.1f}}<br><extra></extra>"
            ))
        fig.update_layout(
            title={'text': title, 'y': 0.95, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top', 'font': dict(size=22, family='Arial', color='#1E3A8A')},
            scene=dict(
                xaxis_title=top_properties[0],
                yaxis_title=top_properties[1],
                zaxis_title=top_properties[2],
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            height=800,
            width=900,
            margin=dict(t=100, b=50, l=50, r=50),
            legend=dict(font=dict(size=12), yanchor="top", y=0.99, xanchor="left", x=0.01),
            paper_bgcolor='white'
        )
        return fig

    def create_comparison_dashboard(self, relationships_df: pd.DataFrame,
                                   selected_dopants: List[str]) -> go.Figure:
        if relationships_df.empty or len(selected_dopants) < 2:
            return None
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Enhancement Factor Comparison", "Property Distribution", "Concentration Optimization", "Processing Method Analysis"),
            vertical_spacing=0.15,
            horizontal_spacing=0.15,
            specs=[[{"type": "bar"}, {"type": "box"}], [{"type": "scatter"}, {"type": "bar"}]]
        )
        enhancement_data = relationships_df.groupby('dopant')['enhancement_factor'].mean().reset_index()
        enhancement_data = enhancement_data[enhancement_data['dopant'].isin(selected_dopants)]
        fig.add_trace(go.Bar(x=enhancement_data['dopant'], y=enhancement_data['enhancement_factor'],
                             marker_color=[self.colors.get(self.classify_dopant(d), '#666666') for d in enhancement_data['dopant']],
                             name='Avg Enhancement'), row=1, col=1)

        for i, dopant in enumerate(selected_dopants):
            dopant_data = relationships_df[relationships_df['dopant'] == dopant]['enhancement_factor']
            fig.add_trace(go.Box(y=dopant_data, name=dopant,
                                 marker_color=self.colors.get(self.classify_dopant(dopant), '#666666'),
                                 boxmean='sd'), row=1, col=2)

        scatter_data = relationships_df[relationships_df['dopant'].isin(selected_dopants)].copy()
        scatter_data['concentration_num'] = scatter_data['concentration_range'].str.extract(r'(\d+(?:\.\d+)?)').astype(float)
        for dopant in selected_dopants:
            dopant_scatter = scatter_data[scatter_data['dopant'] == dopant]
            fig.add_trace(go.Scatter(x=dopant_scatter['concentration_num'], y=dopant_scatter['enhancement_factor'],
                                     mode='markers', name=dopant,
                                     marker=dict(color=self.colors.get(self.classify_dopant(dopant), '#666666'), size=10)), row=2, col=1)

        if 'processing_method' in relationships_df.columns:
            method_data = relationships_df['processing_method'].value_counts().head(5)
            fig.add_trace(go.Bar(x=method_data.index, y=method_data.values, marker_color='lightblue', name='Processing Methods'), row=2, col=2)

        fig.update_layout(height=1000, width=1200, title_text="Comprehensive Dopant Analysis Dashboard",
                          title_font=dict(size=24, family='Arial', color='#1E3A8A'), paper_bgcolor='white', plot_bgcolor='white')
        fig.update_xaxes(title_text="Dopant", row=1, col=1)
        fig.update_yaxes(title_text="Enhancement Factor", row=1, col=1)
        fig.update_xaxes(title_text="Dopant", row=1, col=2)
        fig.update_yaxes(title_text="Enhancement Factor", row=1, col=2)
        fig.update_xaxes(title_text="Concentration (%)", row=2, col=1)
        fig.update_yaxes(title_text="Enhancement Factor", row=2, col=1)
        fig.update_xaxes(title_text="Processing Method", row=2, col=2)
        fig.update_yaxes(title_text="Count", row=2, col=2)
        return fig

    def create_optimal_dopant_recommendations(self, relationships_df: pd.DataFrame, target_application: str):
        """Generate recommendations with confidence scoring"""
        if relationships_df.empty:
            return []
        app_criteria = {
            "Energy Harvesting": {'d₃₃ (pC/N)': 2.0, 'Voltage Output (V)': 1.8, 'Power Density (μW/cm²)': 2.0},
            "Sensors": {'d₃₃ (pC/N)': 1.7, 'Dielectric Constant': 1.8, 'Curie Temp (°C)': 1.5},
            "Actuators": {'d₃₃ (pC/N)': 1.8, "Young's Modulus (GPa)": 1.7},
            "High Temperature": {'Curie Temp (°C)': 2.0, 'Dielectric Constant': 1.8},
            "Flexible Electronics": {'β-phase (%)': 2.0, 'Dielectric Constant': 1.8},
            "Biomedical": {'Biocompatibility': 1.9, 'β-phase (%)': 1.7}
        }
        criteria = app_criteria.get(target_application, app_criteria["Energy Harvesting"])
        dopant_scores = {}
        for dopant in relationships_df['dopant'].unique():
            dopant_df = relationships_df[relationships_df['dopant'] == dopant]
            score = 0.0
            weight_sum = 0.0
            for prop, weight in criteria.items():
                if prop in dopant_df['property'].values:
                    avg_enhance = dopant_df[dopant_df['property'] == prop]['enhancement_factor'].mean()
                    score += avg_enhance * weight
                    weight_sum += weight
            if weight_sum > 0:
                dopant_scores[dopant] = score / weight_sum
            else:
                dopant_scores[dopant] = 0.0

        sorted_dopants = sorted(dopant_scores.items(), key=lambda x: x[1], reverse=True)
        recommendations = []
        for dopant, score in sorted_dopants[:10]:
            dopant_df = relationships_df[relationships_df['dopant'] == dopant]
            avg_enh = dopant_df['enhancement_factor'].mean()
            key_props = dopant_df['property'].value_counts().head(3).index.tolist()
            best_mats = dopant_df['base_material'].value_counts().head(2).index.tolist()
            recommendations.append({
                'dopant': dopant,
                'score': score,
                'avg_enhancement': avg_enh,
                'key_properties': key_props,
                'best_base_materials': best_mats,
                'category': self.classify_dopant(dopant)
            })
        return recommendations

# ==============================
# SAMPLE DATA GENERATION FOR QUERIES
# ==============================
def create_sample_data_for_query(query_id: str = "q0"):
    st.info(f"💡 Creating comprehensive sample data for Query {query_id}...")
    logger.info(f"Creating sample data for Query {query_id}")

    if query_id == "q0":
        materials = ["PVDF", "PVDF/ZnO", "PVDF/BaTiO3", "PVDF/CNT", "ZnO", "BaTiO3", "AlN", "PZT"]
        properties = ["d₃₃ (pC/N)", "β-phase (%)", "Voltage Output (V)", "Power Density (μW/cm²)", "Dielectric Constant", "Curie Temp (°C)"]
        focus_area = "General Piezoelectric Materials"
    elif query_id == "q1":
        materials = ["PVDF-TrFE", "PVDF-HFP", "PVDF/Graphene", "PVDF/MXene", "PVDF/Cellulose", "PVDF/BTO", "PVDF/PZT", "PVDF/AlN"]
        properties = ["d₃₃ (pC/N)", "β-phase (%)", "Dielectric Constant", "Young's Modulus (GPa)", "Curie Temp (°C)", "Power Density (μW/cm²)"]
        focus_area = "Flexible PVDF-Based Composites"
    elif query_id == "q2":
        materials = ["ZnO nanowires", "BaTiO3 nanocubes", "BTO-PZT", "KNN", "BNT-BT", "LiNbO3", "AlN thin films", "ZnSnO3"]
        properties = ["d₃₃ (pC/N)", "g33", "Voltage Output (V)", "Power Density (μW/cm²)", "Curie Temp (°C)", "Dielectric Constant"]
        focus_area = "Inorganic Ceramics & Thin Films"
    else:
        materials = ["PVDF", "PVDF/ZnO", "PVDF/BaTiO3", "PVDF/CNT", "ZnO", "BaTiO3", "AlN", "PZT"]
        properties = ["d₃₃ (pC/N)", "β-phase (%)", "Voltage Output (V)", "Power Density (μW/cm²)", "Dielectric Constant", "Curie Temp (°C)"]
        focus_area = "General Piezoelectric Materials"

    np.random.seed(42)
    n_samples = 300
    relationships = []
    for i in range(n_samples):
        base_material = np.random.choice(materials)
        dopant_category = np.random.choice(list(Config.DOPANT_CATEGORIES.keys()))
        dopant = np.random.choice(Config.DOPANT_CATEGORIES[dopant_category])
        prop = np.random.choice(properties)

        value_ranges = {
            "d₃₃ (pC/N)": (5, 600),
            "β-phase (%)": (40, 95),
            "Voltage Output (V)": (0.1, 100),
            "Power Density (μW/cm²)": (0.01, 1000),
            "Dielectric Constant": (5, 1000),
            "Curie Temp (°C)": (50, 400),
            "Young's Modulus (GPa)": (1, 20)
        }
        vmin, vmax = value_ranges.get(prop, (1, 100))
        value = np.random.uniform(vmin, vmax)

        enhancement = np.random.uniform(1.2, 3.0)
        if query_id == "q1" and "β-phase" in prop:
            enhancement = np.random.uniform(1.8, 2.8)
        elif query_id == "q2" and "d₃₃" in prop:
            enhancement = np.random.uniform(1.2, 2.2)

        concentration_map = {
            "Carbon-Based": ["0.5 wt%", "1.0 wt%", "1.5 wt%", "2.0 wt%"],
            "Metal Oxides": ["5 wt%", "10 wt%", "15 wt%", "20 wt%"],
            "Ferroelectric Ceramics": ["10 vol%", "20 vol%", "30 vol%"],
            "2D Materials": ["0.1 wt%", "0.5 wt%", "1.0 wt%"],
            "Nanoparticles": ["1 wt%", "2 wt%", "5 wt%"],
        }
        conc_options = concentration_map.get(dopant_category, ["3 wt%", "5 wt%", "7 wt%"])
        concentration = np.random.choice(conc_options)

        methods = ["electrospinning", "solution casting", "hot pressing", "melt blending", "in-situ polymerization", "ball milling"]
        method = np.random.choice(methods).title()

        contexts = [
            f"Study of {dopant} doping in {base_material} showing enhanced {prop.split()[0]} properties.",
            f"Enhanced {prop} of {value:.1f} achieved in {base_material} with {dopant} through {method.lower()}.",
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

# ==============================
# MAIN APPLICATION
# ==============================
def main():
    st.markdown('<h1 class="main-header">🔬 Dopant Impact Explorer Pro</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Advanced Visual Analytics for Piezoelectric Material Enhancement</p>', unsafe_allow_html=True)

    if 'current_query_id' not in st.session_state:
        st.session_state.current_query_id = Config.DEFAULT_QUERY_ID
    if 'analysis_engine' not in st.session_state:
        st.session_state.analysis_engine = EnhancedDopantAnalysisEngine()
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = None
    if 'dopant_relationships' not in st.session_state:
        st.session_state.dopant_relationships = None
    if 'performance_monitor' not in st.session_state:
        st.session_state.performance_monitor = PerformanceMonitor()
    if 'cache_manager' not in st.session_state:
        st.session_state.cache_manager = CacheManager()

    with st.sidebar:
        st.markdown("### ⚙️ Configuration Panel")
        available_queries = Config.get_available_query_datasets()
        selected_query = st.selectbox(
            "Select Query Dataset",
            available_queries,
            index=available_queries.index(st.session_state.current_query_id) if st.session_state.current_query_id in available_queries else 0
        )
        if selected_query != st.session_state.current_query_id:
            st.session_state.current_query_id = selected_query
            st.session_state.dopant_relationships = None
            st.session_state.processed_data = None
            st.rerun()

        st.markdown(f"""
        <div style="background-color: #F0F9FF; padding: 0.75rem; border-radius: 6px; margin: 0.5rem 0;">
        <strong>Current Query:</strong> {selected_query}
        </div>
        """, unsafe_allow_html=True)

        current_db_paths = get_db_paths_for_query(st.session_state.current_query_id)
        available_dbs = [name for name, path in current_db_paths.items() if os.path.exists(path)]
        if not available_dbs:
            st.error(f"No databases found for query '{st.session_state.current_query_id}'!")
            st.info("Expected files:\n" + "\n".join(f"- {os.path.basename(path)}" for path in current_db_paths.values()))
        else:
            selected_db = st.selectbox("Select Database", available_dbs)
            db_path = current_db_paths[selected_db]
            st.markdown(f"<div style='background-color: #FEF7CD; padding: 0.5rem; border-radius: 4px; font-size: 0.85em;'><strong>Database Path:</strong><br>{db_path}</div>", unsafe_allow_html=True)

        max_papers = st.slider("Max papers to process", 10, 5000, 500, 50)
        color_palette = st.selectbox("Color Palette", ["nature", "science", "material_science", "categorical_10"], index=0)
        if color_palette in Config.COLOR_PALETTES:
            Config.PLOT_CONFIG["colorway"] = Config.COLOR_PALETTES[color_palette]
        chart_quality = st.select_slider("Chart Quality", ["Low", "Medium", "High", "Publication"], value="High")

        col1, col2 = st.columns(2)
        with col1:
            analyze_btn = st.button("🚀 Start Analysis", type="primary", use_container_width=True, disabled=not available_dbs)
        with col2:
            if st.button("🔄 Reset Session", use_container_width=True):
                st.session_state.dopant_relationships = None
                st.session_state.processed_data = None
                st.rerun()
        if st.button("📊 View Performance Statistics", use_container_width=True):
            st.session_state.performance_monitor.display_stats()

        st.markdown("#### 📊 System Status")
        st.metric("Dopant Categories", len(Config.DOPANT_CATEGORIES))
        st.metric("Base Materials", len(Config.BASE_MATERIALS))
        st.metric("Properties Tracked", len(Config.DOPANT_PROPERTIES))
        st.metric("Available Queries", len(available_queries))

        cache_info = st.session_state.cache_manager.get_cache_info()
        st.markdown(f"""
        <div class="cache-info">
        <strong>Cache Status:</strong> {cache_info['size']}/{cache_info['max_size']} items<br>
        <strong>Hit Rate:</strong> {cache_info['hit_rate']:.1%}
        </div>
        """, unsafe_allow_html=True)

        with st.expander("📚 User Guide"):
            st.markdown("""
            **Publication-Ready Visualizations**
            - Multi-dataset support (`q0`, `q1`, `q2`…)
            - Automatic database schema adaptation
            - Sample data for each query
            - Export to PNG/PDF/SVG with 300+ DPI
            """)

    if analyze_btn and available_dbs:
        with st.spinner(f"🔬 Analyzing Query {st.session_state.current_query_id}..."):
            try:
                db_manager = DatabaseManager(db_path, st.session_state.current_query_id)
                if not db_manager.connect():
                    st.error("Failed to connect to database")
                    return
                papers_df = db_manager.get_papers_data()
                if papers_df.empty:
                    st.error("No papers found!")
                    return
                papers_df = papers_df.iloc[:max_papers].copy()
                engine = st.session_state.analysis_engine
                relationships_df = engine.extract_dopant_relationships(papers_df)
                if relationships_df.empty:
                    st.warning("No relationships extracted.")
                else:
                    st.session_state.processed_data = papers_df
                    st.session_state.dopant_relationships = relationships_df
                    st.success(f"✅ Found {len(relationships_df)} relationships!")
            except Exception as e:
                st.error(f"Analysis failed: {e}")
                logger.error(f"Analysis failed: {e}", exc_info=True)
                return

    if st.session_state.dopant_relationships is not None and not st.session_state.dopant_relationships.empty:
        relationships_df = st.session_state.dopant_relationships
        engine = st.session_state.analysis_engine

        tabs = st.tabs([
            "🌳 Hierarchical Analysis", "📡 Multi-Property Comparison", "🔥 Concentration Heatmap",
            "📊 3D Analysis", "📈 Comprehensive Dashboard", "💡 Recommendations",
            "🔍 Data Explorer", "⚙️ Advanced Settings"
        ])

        with tabs[0]:
            col1, col2, col3 = st.columns(3)
            with col1: max_depth = st.slider("Depth", 2, 5, 4)
            with col2: show_values = st.checkbox("Show Values", True)
            with col3: color_scheme = st.selectbox("Color", ["RdYlBu_r", "Viridis", "Plasma", "Inferno"])
            fig = engine.create_publication_sunburst(relationships_df, show_values=show_values, max_depth=max_depth)
            if fig: st.plotly_chart(fig, use_container_width=True)

        with tabs[1]:
            all_dopants = relationships_df['dopant'].unique().tolist()
            default = relationships_df['dopant'].value_counts().head(4).index.tolist()
            selected = st.multiselect("Dopants", options=all_dopants, default=default, max_selections=6)
            if len(selected) >= 2:
                fig = engine.create_enhanced_radar_chart(relationships_df, selected)
                if fig: st.plotly_chart(fig, use_container_width=True)

        with tabs[2]:
            fig = engine.create_concentration_heatmap(relationships_df)
            if fig: st.plotly_chart(fig, use_container_width=True)

        with tabs[3]:
            fig = engine.create_3d_scatter_plot(relationships_df)
            if fig: st.plotly_chart(fig, use_container_width=True)

        with tabs[4]:
            dashboard_dopants = st.multiselect("Dashboard Dopants", options=relationships_df['dopant'].unique().tolist(), default=relationships_df['dopant'].value_counts().head(3).index.tolist(), max_selections=5)
            if len(dashboard_dopants) >= 2:
                fig = engine.create_comparison_dashboard(relationships_df, dashboard_dopants)
                if fig: st.plotly_chart(fig, use_container_width=True)

        with tabs[5]:
            app = st.selectbox("Application", [
                "Energy Harvesting", "Sensors", "Actuators", "High Temperature", "Flexible Electronics", "Biomedical"
            ])
            if st.button("✨ Generate Recommendations"):
                recs = engine.create_optimal_dopant_recommendations(relationships_df, app)
                for i, rec in enumerate(recs[:5]):
                    confidence = min(1.0, rec['score'] / 3.0)
                    color = "#10B981" if confidence > 0.8 else "#F59E0B" if confidence > 0.6 else "#EF4444"
                    st.markdown(f"""
                    <div style="background: white; padding: 1.5rem; border-radius: 12px; margin: 1rem 0; border-left: 6px solid {color}; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);">
                    <h3>#{i+1} {rec['dopant']} <span style="color:{color}; font-weight:bold;">{confidence:.0%} confidence</span></h3>
                    <p>Score: {rec['score']:.2f}/3.0 | Avg Enh: {rec['avg_enhancement']:.2f}× | Category: {rec['category']}</p>
                    <p>Key Props: {', '.join(rec['key_properties'])}</p>
                    </div>
                    """, unsafe_allow_html=True)

        with tabs[6]:
            st.dataframe(relationships_df, use_container_width=True, height=500)
            csv = relationships_df.to_csv().encode('utf-8')
            st.download_button("Download CSV", csv, "dopant.csv", "text/csv")

        with tabs[7]:
            st.text("Advanced settings (e.g., custom keywords) can be added here.")

    else:
        st.markdown("""
        <div style="padding: 3rem; text-align: center; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 20px; color: white; margin-bottom: 3rem;">
        <h1>🔬 Dopant Impact Explorer Pro</h1>
        <p>Advanced Visual Analytics for Piezoelectric Material Enhancement</p>
        </div>
        """, unsafe_allow_html=True)

        if st.checkbox(f"✅ Use sample data for Query {st.session_state.current_query_id}"):
            with st.spinner("Generating sample data..."):
                df, focus = create_sample_data_for_query(st.session_state.current_query_id)
                st.session_state.dopant_relationships = df
                st.success(f"Sample data ready! Focus: {focus}")
                st.rerun()

# ==============================
# APPLICATION ENTRY POINT
# ==============================
if __name__ == "__main__":
    logger.info("Application started with query support")
    os.makedirs(DEFAULT_DB_DIR, exist_ok=True)
    missing = [name for name, path in get_db_paths_for_query("q0").items() if not os.path.exists(path)]
    if missing:
        st.warning(f"Missing databases: {', '.join(missing)}")
    try:
        main()
    except Exception as e:
        st.error(f"Application error: {e}")
        logger.error(f"Crash: {e}", exc_info=True)
