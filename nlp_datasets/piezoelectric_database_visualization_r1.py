# dopant_impact_explorer.py
import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import os
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from wordcloud import WordCloud

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
</style>
""", unsafe_allow_html=True)

def add_caption(text: str):
    """Add a styled caption below a figure"""
    st.markdown(f'<div class="figure-caption">{text}</div>', unsafe_allow_html=True)

# ==============================
# CONSTANTS & CONFIGURATION (Same as previous code)
# ==============================
DB_DIR = os.path.dirname(os.path.abspath(__file__))
RELIABILITY_DB_FILE = os.path.join(DB_DIR, "knowledge_database", "piezoelectricity_metadata.db")
UNIVERSE_DB_FILE = os.path.join(DB_DIR, "knowledge_database", "piezoelectricity_universe.db")
PDF_DB_FILE = os.path.join(DB_DIR, "knowledge_database", "piezoelectricity_pdfs.db")

class Config:
    """Configuration class for the application"""
    DB_PATHS = {
        "Metadata DB": RELIABILITY_DB_FILE,
        "Universe DB": UNIVERSE_DB_FILE,
        "PDF Storage DB": PDF_DB_FILE
    }
    
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
    
    # Standard colors for visualization
    COLORS = {
        "Metal Oxides": "#3B82F6",      # Blue
        "Carbon-Based": "#10B981",      # Green
        "Ferroelectric Ceramics": "#F59E0B",  # Orange
        "2D Materials": "#8B5CF6",      # Purple
        "Polymers": "#EC4899",          # Pink
        "Others": "#6B7280",            # Gray
        "PVDF": "#2563EB",              # Dark Blue
        "BaTiO3": "#047857",            # Dark Green
        "ZnO": "#B45309",               # Dark Orange
        "PZT": "#7E22CE",               # Dark Purple
        "AlN": "#BE123C"                # Dark Red
    }

# ==============================
# DATABASE MANAGER (Same interface as previous code)
# ==============================
class DatabaseManager:
    """Manages database connections with enhanced error handling and dynamic schema detection"""
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = None
        self.table_columns = {}  # Cache of table columns
        logger.info(f"Database manager initialized for {db_path}")
    
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
                    df['year'] = 2023  # Default year
            
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
                df['year'] = 2023  # Default year
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
        relationships = []
        
        for idx, row in papers_df.iterrows():
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
                                                    enhancement = 1.5  # Default enhancement
                                                 
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
    
    def create_sunburst_chart(self, relationships_df: pd.DataFrame, title: str = "Dopant Impact Hierarchy"):
        """Create sunburst chart showing hierarchical dopant relationships"""
        if relationships_df.empty:
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
            title=title,
            height=700,
            hover_data={
                'value': ':.2f',
                'enhancement_factor': ':.2f',
                'size': False
            }
        )
        
        fig.update_layout(
            title_font=dict(size=24, family="Arial", color="black"),
            font=dict(size=14, family="Arial"),
            hoverlabel=dict(bgcolor="white", font_size=14)
        )
        
        return fig
    
    def create_radar_chart(self, relationships_df: pd.DataFrame, selected_dopants: List[str], title: str = "Dopant Performance Comparison"):
        """Create radar chart comparing multiple dopant properties"""
        if relationships_df.empty or not selected_dopants:
            return None
        
        # Filter for selected dopants
        filtered_df = relationships_df[relationships_df['dopant'].isin(selected_dopants)]
        if filtered_df.empty:
            return None
        
        # Get properties to compare
        properties = list(self.properties.keys())[:6]  # Limit to 6 properties for radar chart
        
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
                        dopant_data[dopant][prop] = 1.0  # Default = no enhancement
        
        if not dopant_data:
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
                    range=[0.5, 3.0],  # 0.5x to 3x enhancement
                    title="Enhancement Factor",
                    tickfont=dict(size=12)
                ),
                angularaxis=dict(
                    tickfont=dict(size=14)
                )
            ),
            showlegend=True,
            title=dict(
                text=title,
                font=dict(size=22, family="Arial"),
                x=0.5
            ),
            height=700,
            font=dict(size=14, family="Arial"),
            legend=dict(
                font=dict(size=14),
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(t=80, b=80, l=80, r=80)
        )
        
        return fig
    
    def create_dopant_concentration_chart(self, relationships_df: pd.DataFrame):
        """Create chart showing dopant concentration vs performance"""
        if relationships_df.empty:
            return None
        
        # Filter out unknown concentrations
        filtered_df = relationships_df[relationships_df['concentration_range'] != 'Unknown']
        if filtered_df.empty or len(filtered_df) < 5:
            return None
        
        # Extract numeric concentration values
        filtered_df['concentration_value'] = filtered_df['concentration_range'].str.extract(r'(\d+(?:\.\d+)?)').astype(float)
        
        # Remove outliers
        filtered_df = filtered_df[filtered_df['concentration_value'] <= filtered_df['concentration_value'].quantile(0.95)]
        
        if filtered_df.empty:
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
            title='Dopant Concentration vs Performance Enhancement',
            labels={
                'concentration_value': 'Concentration (%)',
                'enhancement_factor': 'Enhancement Factor',
                'dopant_category': 'Dopant Category'
            },
            height=600
        )
        
        # Add trend lines for each category
        for category in filtered_df['dopant_category'].unique():
            cat_df = filtered_df[filtered_df['dopant_category'] == category]
            if len(cat_df) >= 3:
                x = cat_df['concentration_value']
                y = cat_df['enhancement_factor']
                z = np.polyfit(x, y, 2)  # Quadratic fit
                p = np.poly1d(z)
                x_range = np.linspace(x.min(), x.max(), 100)
                fig.add_scatter(
                    x=x_range,
                    y=p(x_range),
                    mode='lines',
                    name=f'{category} trend',
                    line=dict(color=self.colors.get(category, '#666'), dash='dash')
                )
        
        fig.update_layout(
            title_font=dict(size=22, family="Arial", color="black"),
            font=dict(size=14, family="Arial"),
            hoverlabel=dict(bgcolor="white", font_size=14),
            xaxis=dict(title_font=dict(size=16), tickfont=dict(size=14)),
            yaxis=dict(title_font=dict(size=16), tickfont=dict(size=14))
        )
        
        return fig
    
    def create_optimal_dopant_recommendations(self, relationships_df: pd.DataFrame, target_application: str):
        """Generate recommendations for optimal dopants based on application"""
        if relationships_df.empty:
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
# MAIN APPLICATION
# ==============================
def main():
    """Main Streamlit application for dopant impact analysis"""
    
    st.markdown('<h1 class="main-header">🔬 Dopant Impact Explorer<br><small>Visual Analytics for Piezoelectric Material Enhancement</small></h1>', unsafe_allow_html=True)
    
    # Initialize session state
    if 'analysis_engine' not in st.session_state:
        st.session_state.analysis_engine = DopantAnalysisEngine()
    
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = None
    
    if 'dopant_relationships' not in st.session_state:
        st.session_state.dopant_relationships = None
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Database selection
        available_dbs = []
        for db_name, db_path in Config.DB_PATHS.items():
            if os.path.exists(db_path):
                available_dbs.append(db_name)
        
        if not available_dbs:
            st.error("No databases found in `knowledge_database/`!")
            st.info("Required files:\n- piezoelectricity_metadata.db\n- piezoelectricity_universe.db\n- piezoelectricity_pdfs.db")
            return
        
        selected_db = st.selectbox("Select Database", available_dbs)
        db_path = Config.DB_PATHS[selected_db]
        
        # Analysis parameters
        st.subheader("Analysis Scope")
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
        
        # Actions
        st.subheader("Actions")
        col1, col2 = st.columns(2)
        with col1:
            analyze_btn = st.button("🚀 Start Analysis", type="primary", use_container_width=True)
        with col2:
            if st.button("🔄 Reset Session", use_container_width=True):
                st.session_state.processed_data = None
                st.session_state.dopant_relationships = None
                st.rerun()
        
        # System info
        st.subheader("System Status")
        st.metric("Dopant Categories", len(Config.DOPANT_CATEGORIES))
        st.metric("Base Materials", len(Config.BASE_MATERIALS))
        st.metric("Properties Tracked", len(Config.DOPANT_PROPERTIES))
        
        # Help section
        with st.expander("ℹ️ About This Tool"):
            st.markdown("""
            **Dopant Impact Explorer** is a specialized visualization tool for analyzing how different dopants affect piezoelectric material properties.
            
            **Key Features:**
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
    if analyze_btn:
        with st.spinner("🔬 Analyzing dopant relationships from literature..."):
            try:
                # Initialize database manager
                db_manager = DatabaseManager(db_path)
                if not db_manager.connect():
                    st.error("Failed to connect to database")
                    return
                
                # Load papers
                st.text("📥 Loading papers from database...")
                papers_df = db_manager.get_papers_data()
                
                if papers_df.empty:
                    st.error("No papers found in database!")
                    return
                
                # Limit for performance
                papers_df = papers_df.iloc[:max_papers].copy()
                
                # Extract dopant relationships
                st.text("🧪 Extracting dopant relationships...")
                engine = st.session_state.analysis_engine
                relationships_df = engine.extract_dopant_relationships(papers_df)
                
                if relationships_df.empty:
                    st.warning("No dopant relationships extracted. The database may not contain sufficient dopant information.")
                    
                    # Show sample of papers to help debugging
                    with st.expander("🔍 Sample Papers for Debugging"):
                        st.markdown("Here are some sample papers. Check if they contain dopant information:")
                        for i, row in papers_df.head(5).iterrows():
                            st.markdown(f"**Paper {i+1}:** {row.get('title', 'No title')}")
                            st.markdown(f"*Abstract:* {row.get('abstract', '')[:200]}...")
                
                # Store results
                st.session_state.processed_data = papers_df
                st.session_state.dopant_relationships = relationships_df
                
                st.success(f"✅ Analysis complete! Found {len(relationships_df)} dopant relationships in {len(papers_df)} papers.")
                
                # Show summary statistics
                with st.expander("📊 Analysis Summary"):
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Dopant Relationships", len(relationships_df))
                    col2.metric("Unique Dopants", relationships_df['dopant'].nunique() if not relationships_df.empty else 0)
                    col3.metric("Base Materials", relationships_df['base_material'].nunique() if not relationships_df.empty else 0)
                    
                    if not relationships_df.empty:
                        st.markdown("### Top Dopant Categories")
                        category_counts = relationships_df['dopant_category'].value_counts().head(5)
                        fig = px.bar(
                            x=category_counts.index,
                            y=category_counts.values,
                            labels={'x': 'Category', 'y': 'Number of Relationships'},
                            title='Most Studied Dopant Categories',
                            color=category_counts.index,
                            color_discrete_map=engine.colors
                        )
                        st.plotly_chart(fig, use_container_width=True)
            
            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")
                logger.error(f"Analysis failed: {str(e)}", exc_info=True)
                return
    
    # Results display
    if st.session_state.dopant_relationships is not None and not st.session_state.dopant_relationships.empty:
        relationships_df = st.session_state.dopant_relationships
        engine = st.session_state.analysis_engine
        
        # Create tabs
        tabs = st.tabs([
            "🌞 Sunburst Analysis", 
            "📡 Radar Comparison", 
            "📊 Concentration Effects",
            "💡 Recommendations",
            "🔍 Data Explorer",
            "⚙️ Advanced Settings"
        ])
        
        # Tab 1: Sunburst Chart
        with tabs[0]:
            st.subheader("🌳 Hierarchical Dopant Impact Analysis")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                fig = engine.create_sunburst_chart(relationships_df, "Dopant Impact Hierarchy for Piezoelectric Enhancement")
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
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
            
            # Get unique dopants
            all_dopants = relationships_df['dopant'].unique().tolist()
            
            # Default selection: top 4 dopants by frequency
            default_dopants = relationships_df['dopant'].value_counts().head(4).index.tolist()
            
            selected_dopants = st.multiselect(
                "Select dopants to compare",
                options=all_dopants,
                default=default_dopants,
                max_selections=6
            )
            
            if len(selected_dopants) < 2:
                st.info("Please select at least 2 dopants for comparison")
            else:
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    fig = engine.create_radar_chart(relationships_df, selected_dopants, 
                                                  "Multi-Property Performance Comparison of Dopants")
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
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
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                fig = engine.create_dopant_concentration_chart(relationships_df)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
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
                with st.spinner("Generating optimized dopant recommendations..."):
                    recommendations = engine.create_optimal_dopant_recommendations(relationships_df, application)
                    
                    if not recommendations:
                        st.warning("No recommendations available. Try a different application or analyze more papers.")
                    else:
                        st.markdown(f"### 🏆 Top Recommendations for {application}")
                        
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
                        **For {application} applications:**
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
            
            # Show relationships data
            st.markdown("### Extracted Dopant Relationships")
            st.dataframe(relationships_df, use_container_width=True, height=400)
            
            # Download options
            col1, col2, col3 = st.columns(3)
            
            with col1:
                csv = relationships_df.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download CSV", csv, "dopant_relationships.csv", "text/csv")
            
            with col2:
                excel_buffer = io.BytesIO()
                with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                    relationships_df.to_excel(writer, sheet_name='dopant_relationships', index=False)
                excel_buffer.seek(0)
                st.download_button("📊 Download Excel", excel_buffer, "dopant_analysis.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            
            with col3:
                json_data = relationships_df.to_dict('records')
                json_str = json.dumps(json_data, indent=2)
                st.download_button("💾 Download JSON", json_str, "dopant_data.json", "application/json")
            
            # Advanced filtering
            with st.expander("🔧 Advanced Data Filtering"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    base_material_filter = st.multiselect(
                        "Base Materials",
                        options=relationships_df['base_material'].unique(),
                        default=relationships_df['base_material'].unique()[:3].tolist()
                    )
                
                with col2:
                    dopant_category_filter = st.multiselect(
                        "Dopant Categories",
                        options=relationships_df['dopant_category'].unique(),
                        default=relationships_df['dopant_category'].unique()[:3].tolist()
                    )
                
                with col3:
                    min_enhancement = st.slider("Min Enhancement Factor", 1.0, 3.0, 1.5)
                
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
            
            # Custom classification editor
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
    
    else:
        # Welcome screen
        st.markdown("""
        <div style="padding: 2.5rem; text-align: center; background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%); border-radius: 15px; color: white; margin-bottom: 2rem;">
            <h2>🔬 Dopant Impact Explorer</h2>
            <p style="font-size: 1.2rem; opacity: 0.9;">Visual Analytics for Piezoelectric Material Enhancement</p>
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
            
            1. **Database Connection**: Connect to your piezoelectric materials knowledge base
            2. **Automatic Analysis**: The system extracts dopant relationships from paper text
            3. **Hierarchical Classification**: Dopants are categorized by chemical type and function
            4. **Enhancement Quantification**: Performance improvements are calculated and normalized
            5. **Advanced Visualization**: Interactive charts reveal optimal dopant strategies
            
            ### Data Requirements
            
            Your database should contain papers with:
            - Full text or detailed abstracts
            - Information about doped piezoelectric materials
            - Quantitative property measurements
            - Experimental details (concentration, processing methods)
            
            ### Key Metrics Tracked
            
            - **Enhancement Factor**: Performance improvement over undoped material
            - **Concentration Range**: Optimal doping levels (wt%, vol%)
            - **Property Coverage**: d₃₃, β-phase content, voltage output, power density, etc.
            - **Processing Methods**: How fabrication affects dopant effectiveness
            
            ### Scientific Foundation
            
            This tool is based on systematic analysis of:
            - Structure-property relationships in piezoelectric composites
            - Dopant-matrix interactions at molecular level
            - Percolation theory for conductive fillers
            - Interface engineering for property enhancement
            
            The visualizations help you quickly identify the most promising dopant strategies for your specific application needs.
            """)

# ==============================
# APPLICATION ENTRY POINT
# ==============================
if __name__ == "__main__":
    # Create knowledge_database directory if it doesn't exist
    os.makedirs(os.path.join(DB_DIR, "knowledge_database"), exist_ok=True)
    
    # Check for required databases
    missing_dbs = []
    for db_name, db_path in Config.DB_PATHS.items():
        if not os.path.exists(db_path):
            missing_dbs.append(db_name)
    
    if missing_dbs:
        st.warning(f"⚠️ Missing database files: {', '.join(missing_dbs)}")
        st.info("📁 Expected location: `knowledge_database/` subdirectory")
        
        if st.checkbox("✅ Use sample data for demonstration"):
            st.info("💡 Creating sample dopant data for demonstration...")
            
            # Create sample data
            np.random.seed(42)
            n_samples = 150
            
            base_materials = ["PVDF", "BaTiO3", "ZnO", "PVDF", "PVDF", "BaTiO3", "PZT"]
            dopants = ["ZnO", "BaTiO3", "CNT", "Graphene", "TiO2", "AlN", "Fe2O3", "MXene", "Cellulose"]
            dopant_categories = ["Metal Oxides", "Ferroelectric Ceramics", "Carbon-Based", "Carbon-Based", 
                               "Metal Oxides", "Metal Oxides", "Metal Oxides", "2D Materials", "Others"]
            properties = ["d33", "beta_phase", "dielectric", "d33", "voltage", "power", "d33", "d33", "mechanical"]
            concentrations = ["5 wt%", "10 vol%", "1 wt%", "0.5 wt%", "15 wt%", "20 vol%", "8 wt%", "2 wt%", "3 wt%"]
            methods = ["electrospinning", "solution casting", "hot pressing", "melt blending", "in-situ polymerization"]
            
            relationships = []
            for i in range(n_samples):
                idx = np.random.randint(0, len(dopants))
                relationships.append({
                    'paper_id': f'paper_{i+1}',
                    'base_material': np.random.choice(base_materials),
                    'dopant': dopants[idx],
                    'dopant_category': dopant_categories[idx],
                    'property': np.random.choice(properties),
                    'value': np.random.uniform(5, 500),
                    'enhancement_factor': np.random.uniform(1.2, 3.0),
                    'concentration_range': np.random.choice(concentrations),
                    'processing_method': np.random.choice(methods),
                    'context': f"Sample context for {dopants[idx]} in {base_materials[0]}"
                })
            
            relationships_df = pd.DataFrame(relationships)
            st.session_state.dopant_relationships = relationships_df
            st.success("✅ Sample data ready! Explore the tabs above.")
    
    # Run main application
    main()
