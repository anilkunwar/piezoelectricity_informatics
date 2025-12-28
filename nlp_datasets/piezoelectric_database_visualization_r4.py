# quantitative_ner_analyzer.py
import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import os
import re
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import matplotlib
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib backend and style
matplotlib.use('Agg')
plt.style.use('default')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("QuantitativeNERAnalyzer")

# Set page config
st.set_page_config(
    page_title="Quantitative NER Analyzer for PVDF Materials",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
.main-header {
    font-size: 2.8rem;
    color: #1E3A8A;
    text-align: center;
    margin-bottom: 1rem;
    font-weight: 700;
    background: linear-gradient(90deg, #1E3A8A 0%, #3B82F6 50%, #60A5FA 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.sub-header {
    font-size: 1.3rem;
    color: #4B5563;
    text-align: center;
    margin-bottom: 2rem;
    font-weight: 400;
}
.ner-card {
    background: linear-gradient(135deg, #F8FAFC 0%, #EFF6FF 100%);
    padding: 1.2rem;
    border-radius: 12px;
    border: 1px solid #E5E7EB;
    margin: 0.5rem 0;
    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
}
.metric-highlight {
    font-size: 1.5rem;
    font-weight: bold;
    color: #1E40AF;
    text-align: center;
}
.entity-tag {
    display: inline-block;
    padding: 3px 8px;
    margin: 2px;
    border-radius: 4px;
    font-size: 0.85rem;
    font-weight: 600;
}
.entity-PROPERTY { background-color: #DBEAFE; color: #1E40AF; }
.entity-CONCENTRATION { background-color: #D1FAE5; color: #065F46; }
.entity-VALUE { background-color: #FEF3C7; color: #92400E; }
.entity-MATERIAL { background-color: #E0E7FF; color: #3730A3; }
.entity-PROCESS { background-color: #FCE7F3; color: #9D174D; }
.stTabs [data-baseweb="tab"] {
    height: 50px;
    font-size: 1rem;
    font-weight: 600;
}
.parameter-box {
    background: white;
    padding: 1rem;
    border-radius: 8px;
    border-left: 4px solid #3B82F6;
    margin: 0.5rem 0;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
.stat-badge {
    display: inline-block;
    padding: 4px 10px;
    margin: 2px;
    border-radius: 15px;
    font-size: 0.8rem;
    font-weight: 600;
    background-color: #E5E7EB;
    color: #4B5563;
}
</style>
""", unsafe_allow_html=True)

# ==============================
# DATABASE LOADER
# ==============================
class DatabaseManager:
    """Manages database connections for loading papers data"""
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = None
        logger.info(f"Database manager initialized for {db_path}")
    
    def connect(self) -> bool:
        """Establish database connection"""
        try:
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            logger.info(f"Connected to database: {self.db_path}")
            return True
        except Exception as e:
            logger.error(f"Database connection error: {e}")
            st.error(f"Database connection error: {e}")
            return False
    
    def get_papers_data(self) -> pd.DataFrame:
        """Get papers data from database"""
        if not self.connect():
            return pd.DataFrame()
        
        try:
            # Try different table names
            tables = ["papers_fulltext", "papers", "documents"]
            for table in tables:
                try:
                    query = f"SELECT * FROM {table} LIMIT 2000"
                    df = pd.read_sql_query(query, self.conn)
                    if not df.empty:
                        logger.info(f"Loaded {len(df)} papers from {table}")
                        return df
                except:
                    continue
            
            # If no standard table found, try to find any table
            query = "SELECT name FROM sqlite_master WHERE type='table';"
            all_tables = pd.read_sql_query(query, self.conn)['name'].tolist()
            for table in all_tables:
                try:
                    query = f"SELECT * FROM {table} LIMIT 1000"
                    df = pd.read_sql_query(query, self.conn)
                    if not df.empty:
                        logger.info(f"Loaded {len(df)} papers from {table}")
                        return df
                except:
                    continue
            
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error loading papers: {e}")
            return pd.DataFrame()

# ==============================
# QUANTITATIVE NER ENGINE
# ==============================
class QuantitativeNERAnalyzer:
    """Advanced NER engine for extracting quantitative parameters from materials science text"""
    
    def __init__(self):
        # PVDF-specific keywords and patterns
        self.pvdf_keywords = [
            "pvdf", "polyvinylidene fluoride", "poly(vinylidene fluoride)",
            "pvdf-trfe", "pvdf-hfp", "pvdf based", "pvdf composite"
        ]
        
        # Dopant categories for PVDF
        self.dopant_categories = {
            "Metal Oxides": ["zno", "tio2", "batio3", "al2o3", "fe2o3", "cuo", "mgo", "zro2"],
            "Carbon-Based": ["cnt", "graphene", "carbon black", "graphene oxide", "mwcnt", "carbon nanotube"],
            "Ceramics": ["pzt", "knn", "bnkt", "pzn-pt", "pmn-pt"],
            "2D Materials": ["mos2", "ws2", "mxene", "h-bn", "phosphorene"],
            "Polymers": ["pva", "pmma", "peo", "pvp", "pani", "ppy"],
            "Nanoparticles": ["ag nanoparticles", "au nanoparticles", "sio2 nanoparticles", "tio2 nanoparticles"],
            "Others": ["clay", "cellulose", "silica", "quantum dots"]
        }
        
        # Regular expressions for quantitative parameter extraction
        self.patterns = {
            'd33': [
                r'd33[:\s]*([\d\.]+)\s*(?:pC/N|pC N⁻¹|pm/V|pC/N⁻¹)',
                r'piezoelectric coefficient[:\s]*([\d\.]+)\s*(?:pC/N|pC N⁻¹|pm/V)',
                r'd₃₃[:\s]*([\d\.]+)\s*(?:pC/N|pC N⁻¹|pm/V)',
                r'([\d\.]+)\s*(?:pC/N|pC N⁻¹|pm/V).*d33',
                r'd33.*?([\d\.]+)\s*(?:pC/N|pC N⁻¹|pm/V)',
                r'd33\s*=\s*([\d\.]+)\s*(?:pC/N|pC N⁻¹|pm/V)',
                r'd33.*?of\s*([\d\.]+)\s*(?:pC/N|pC N⁻¹|pm/V)',
                r'd33.*?value.*?([\d\.]+)\s*(?:pC/N|pC N⁻¹|pm/V)',
                r'([\d\.]+)\s*pC/N.*?d33',
                r'd33.*?approximately\s*([\d\.]+)\s*(?:pC/N|pC N⁻¹|pm/V)',
                r'd33.*?reached\s*([\d\.]+)\s*(?:pC/N|pC N⁻¹|pm/V)',
                r'd33.*?was\s*([\d\.]+)\s*(?:pC/N|pC N⁻¹|pm/V)',
                r'd33.*?measured.*?([\d\.]+)\s*(?:pC/N|pC N⁻¹|pm/V)'
            ],
            'beta_phase': [
                r'beta.*?phase.*?([\d\.]+)\s*%',
                r'β-phase.*?([\d\.]+)\s*%',
                r'([\d\.]+)\s*%.*?beta.*?phase',
                r'ferroelectric phase.*?([\d\.]+)\s*%',
                r'β.*?content.*?([\d\.]+)\s*%',
                r'beta.*?content.*?([\d\.]+)\s*%',
                r'([\d\.]+)\s*%.*?β.*?phase',
                r'phase.*?content.*?([\d\.]+)\s*%'
            ],
            'concentration': [
                r'([\d\.]+)\s*(?:wt|wt\.|weight)\s*%',
                r'([\d\.]+)\s*(?:vol|vol\.|volume)\s*%',
                r'([\d\.]+)\s*(?:mol|mol\.|molar)\s*%',
                r'([\d\.]+)\s*%.*?(?:doping|addition|loading|content)',
                r'concentration.*?([\d\.]+)\s*%',
                r'([\d\.]+)\s*%.*?filler',
                r'([\d\.]+)\s*%.*?dopant',
                r'doping.*?([\d\.]+)\s*%',
                r'([\d\.]+)\s*weight percent',
                r'([\d\.]+)\s*wt\.?\s*percent'
            ],
            'voltage': [
                r'([\d\.]+)\s*(?:V|volt).*?output',
                r'voltage.*?([\d\.]+)\s*(?:V|volt)',
                r'output.*?([\d\.]+)\s*(?:V|volt)',
                r'([\d\.]+)\s*V.*?generated',
                r'([\d\.]+)\s*V.*?produced',
                r'open.*?circuit.*?voltage.*?([\d\.]+)\s*(?:V|volt)',
                r'ocv.*?([\d\.]+)\s*(?:V|volt)'
            ],
            'dielectric': [
                r'dielectric.*?constant.*?([\d\.]+)',
                r'permittivity.*?([\d\.]+)',
                r'εr.*?([\d\.]+)',
                r'([\d\.]+).*?dielectric.*?constant',
                r'dielectric.*?([\d\.]+)\s*(?:at|@)',
                r'relative.*?permittivity.*?([\d\.]+)'
            ],
            'youngs_modulus': [
                r'young.*?modulus.*?([\d\.]+)\s*(?:GPa|MPa)',
                r'([\d\.]+)\s*(?:GPa|MPa).*?young',
                r'elastic modulus.*?([\d\.]+)\s*(?:GPa|MPa)',
                r'tensile.*?modulus.*?([\d\.]+)\s*(?:GPa|MPa)',
                r'mechanical.*?modulus.*?([\d\.]+)\s*(?:GPa|MPa)'
            ]
        }
        
        # Units mapping
        self.units = {
            'd33': 'pC/N',
            'beta_phase': '%',
            'concentration': 'wt%',
            'voltage': 'V',
            'dielectric': 'εr',
            'youngs_modulus': 'GPa'
        }
        
        # Property display names
        self.property_names = {
            'd33': 'd₃₃ Coefficient (pC/N)',
            'beta_phase': 'β-Phase Content (%)',
            'concentration': 'Dopant Concentration (wt%)',
            'voltage': 'Output Voltage (V)',
            'dielectric': 'Dielectric Constant (εr)',
            'youngs_modulus': "Young's Modulus (GPa)"
        }
        
        # Color schemes for visualization
        self.colors = {
            'd33': '#3B82F6',      # Blue
            'beta_phase': '#10B981', # Green
            'concentration': '#F59E0B', # Orange
            'voltage': '#EF4444',   # Red
            'dielectric': '#8B5CF6', # Purple
            'youngs_modulus': '#EC4899' # Pink
        }
        
        # Dopant category colors
        self.dopant_colors = {
            "Metal Oxides": "#1F77B4",
            "Carbon-Based": "#FF7F0E",
            "Ceramics": "#2CA02C",
            "2D Materials": "#D62728",
            "Polymers": "#9467BD",
            "Nanoparticles": "#8C564B",
            "Others": "#7F7F7F"
        }
    
    def is_pvdf_related(self, text: str) -> bool:
        """Check if text is related to PVDF"""
        if not text or not isinstance(text, str):
            return False
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in self.pvdf_keywords)
    
    def extract_quantitative_parameters(self, text: str, paper_id: str = None) -> Dict[str, List[Dict]]:
        """Extract quantitative parameters from text using regex patterns"""
        if not text or not isinstance(text, str):
            return {}
        
        text_lower = text.lower()
        results = {}
        
        # Extract each parameter type
        for param, patterns in self.patterns.items():
            param_results = []
            for pattern in patterns:
                matches = re.finditer(pattern, text_lower, re.IGNORECASE)
                for match in matches:
                    try:
                        value = float(match.group(1))
                        # Context extraction
                        start = max(0, match.start() - 100)
                        end = min(len(text_lower), match.end() + 100)
                        context = text_lower[start:end]
                        
                        # Extract dopant from context if concentration
                        dopant = None
                        dopant_category = None
                        if param == 'concentration':
                            dopant = self._extract_dopant_from_context(context)
                            if dopant:
                                dopant_category = self._categorize_dopant(dopant)
                        
                        param_results.append({
                            'value': value,
                            'unit': self.units.get(param, ''),
                            'context': context,
                            'sentence': self._extract_sentence(text, match.start(), match.end()),
                            'dopant': dopant,
                            'dopant_category': dopant_category,
                            'paper_id': paper_id
                        })
                    except (ValueError, IndexError):
                        continue
            
            if param_results:
                results[param] = param_results
        
        # Extract dopants and their concentrations
        dopant_results = self._extract_dopant_concentrations(text_lower, paper_id)
        if dopant_results:
            results['dopant_details'] = dopant_results
        
        return results
    
    def _extract_dopant_from_context(self, context: str) -> Optional[str]:
        """Extract dopant name from context"""
        # Look for dopant names near concentration
        for category, dopants in self.dopant_categories.items():
            for dopant in dopants:
                if dopant in context:
                    # Try to get the full dopant name
                    start = max(0, context.find(dopant) - 20)
                    end = min(len(context), context.find(dopant) + len(dopant) + 20)
                    dopant_text = context[start:end]
                    
                    # Clean and return
                    clean_dopant = self._clean_dopant_name(dopant_text)
                    if clean_dopant:
                        return clean_dopant
        return None
    
    def _categorize_dopant(self, dopant: str) -> Optional[str]:
        """Categorize dopant into predefined categories"""
        if not dopant:
            return None
        
        dopant_lower = dopant.lower()
        for category, dopants in self.dopant_categories.items():
            if any(d in dopant_lower for d in dopants):
                return category
        return "Others"
    
    def _extract_sentence(self, text: str, start: int, end: int) -> str:
        """Extract full sentence containing the match"""
        # Find sentence boundaries
        sentence_start = text.rfind('.', 0, start)
        if sentence_start == -1:
            sentence_start = 0
        else:
            sentence_start += 1
        
        sentence_end = text.find('.', end)
        if sentence_end == -1:
            sentence_end = len(text)
        
        return text[sentence_start:sentence_end].strip()
    
    def _extract_dopant_concentrations(self, text: str, paper_id: str = None) -> List[Dict]:
        """Extract specific dopant-concentration pairs"""
        results = []
        
        # Enhanced patterns for dopant-concentration extraction
        patterns = [
            # Pattern: Dopant (X wt%)
            r'([a-zA-Z\s\d\(\)\-/]+)\s*\(([\d\.]+)\s*(?:wt|wt\.|weight)\s*%\)',
            # Pattern: X wt% Dopant
            r'([\d\.]+)\s*(?:wt|wt\.|weight)\s*%\s*([a-zA-Z\s\d\(\)\-/]+)',
            # Pattern: Dopant with X wt%
            r'([a-zA-Z\s\d\(\)\-/]+)\s*with\s*([\d\.]+)\s*(?:wt|wt\.|weight)\s*%',
            # Pattern: Dopant content of X wt%
            r'([a-zA-Z\s\d\(\)\-/]+)\s*content.*?([\d\.]+)\s*(?:wt|wt\.|weight)\s*%',
            # Pattern: X wt% of Dopant
            r'([\d\.]+)\s*(?:wt|wt\.|weight)\s*%\s*of\s*([a-zA-Z\s\d\(\)\-/]+)',
            # Pattern: Dopant-doped at X wt%
            r'([a-zA-Z\s\d\(\)\-/]+).*?doped.*?([\d\.]+)\s*(?:wt|wt\.|weight)\s*%',
            # Pattern: Dopant concentration X wt%
            r'([a-zA-Z\s\d\(\)\-/]+)\s*concentration.*?([\d\.]+)\s*(?:wt|wt\.|weight)\s*%'
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    # Try both group orders
                    try:
                        dopant = match.group(1).strip()
                        concentration = float(match.group(2))
                    except:
                        # Try reverse order
                        dopant = match.group(2).strip()
                        concentration = float(match.group(1))
                    
                    # Clean dopant name
                    dopant = self._clean_dopant_name(dopant)
                    
                    # Skip if dopant is obviously not a material
                    invalid_indicators = ['content', 'concentration', 'doped', 'weight', 'wt', 'percent', '%']
                    if any(indicator in dopant.lower() for indicator in invalid_indicators):
                        continue
                    
                    if dopant and 0.001 <= concentration <= 100:  # Reasonable concentration range
                        category = self._categorize_dopant(dopant)
                        
                        results.append({
                            'dopant': dopant,
                            'concentration': concentration,
                            'unit': 'wt%',
                            'category': category,
                            'paper_id': paper_id
                        })
                except (ValueError, IndexError, AttributeError):
                    continue
        
        return results
    
    def _clean_dopant_name(self, dopant: str) -> str:
        """Clean and standardize dopant names"""
        if not dopant:
            return ""
        
        dopant = dopant.strip()
        
        # Remove common prefixes/suffixes and unwanted characters
        removals = [
            'nanoparticles', 'nanoparticle', 'nps', 'np', 'nanocomposite', 
            'composite', 'doped', 'doping', 'added', 'addition', 'content',
            'concentration', 'with', 'of', 'and', 'the', 'a', 'an', 'at',
            'weight', 'wt', 'percent', '%', '(', ')', '[', ']', '{', '}',
            '  ', '   '
        ]
        
        for removal in removals:
            dopant = re.sub(r'\b' + re.escape(removal) + r'\b', '', dopant, flags=re.IGNORECASE)
            dopant = dopant.replace(removal, '')
        
        # Remove extra spaces and special characters
        dopant = re.sub(r'\s+', ' ', dopant).strip()
        dopant = dopant.strip(' ,.-_/')
        
        # Standardize common dopants
        replacements = {
            'cnts': 'CNT',
            'carbon nanotubes': 'CNT',
            'multi-walled carbon nanotubes': 'MWCNT',
            'multi walled carbon nanotubes': 'MWCNT',
            'graphene oxide': 'GO',
            'reduced graphene oxide': 'RGO',
            'titanium dioxide': 'TiO₂',
            'zinc oxide': 'ZnO',
            'barium titanate': 'BaTiO₃',
            'lead zirconate titanate': 'PZT',
            'molybdenum disulfide': 'MoS₂',
            'silver nanoparticles': 'Ag NPs',
            'silver nps': 'Ag NPs'
        }
        
        dopant_lower = dopant.lower()
        for old, new in replacements.items():
            if old in dopant_lower:
                dopant = new
                break
        
        # Capitalize properly
        if dopant and not any(c.isupper() for c in dopant):
            # For chemical formulas, keep as is
            if re.match(r'^[a-z0-9]+$', dopant, re.IGNORECASE):
                dopant = dopant.upper()
            else:
                # For names, title case
                dopant = dopant.title()
        
        return dopant
    
    def analyze_papers_batch(self, papers_df: pd.DataFrame, text_column: str = 'abstract') -> Dict[str, Any]:
        """Analyze batch of papers and extract quantitative parameters"""
        all_results = {
            'parameters': {},
            'dopants': [],
            'papers_analyzed': 0,
            'papers_with_pvdf': 0,
            'papers_with_quantitative_data': 0,
            'total_extractions': 0
        }
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, row in papers_df.iterrows():
            paper_id = row.get('paper_id', f"paper_{idx}")
            text = str(row.get(text_column, '') or row.get('full_text', '') or 
                      row.get('content', '') or row.get('abstract', ''))
            
            if not text or len(text) < 50:
                continue
            
            all_results['papers_analyzed'] += 1
            
            # Update progress
            progress = min(1.0, all_results['papers_analyzed'] / len(papers_df))
            progress_bar.progress(progress)
            status_text.text(f"Analyzing paper {all_results['papers_analyzed']} of {len(papers_df)}...")
            
            # Check if PVDF-related
            if self.is_pvdf_related(text):
                all_results['papers_with_pvdf'] += 1
                
                # Extract parameters
                paper_results = self.extract_quantitative_parameters(text, paper_id)
                
                if paper_results:
                    all_results['papers_with_quantitative_data'] += 1
                    
                    # Aggregate parameter results
                    for param, values in paper_results.items():
                        if param != 'dopant_details':
                            if param not in all_results['parameters']:
                                all_results['parameters'][param] = []
                            all_results['parameters'][param].extend(values)
                            all_results['total_extractions'] += len(values)
                    
                    # Aggregate dopant details
                    if 'dopant_details' in paper_results:
                        all_results['dopants'].extend(paper_results['dopant_details'])
                        all_results['total_extractions'] += len(paper_results['dopant_details'])
        
        progress_bar.empty()
        status_text.empty()
        
        # Convert to DataFrames for easier analysis
        param_dfs = {}
        for param, values in all_results['parameters'].items():
            if values:
                param_dfs[param] = pd.DataFrame(values)
        
        all_results['parameter_dfs'] = param_dfs
        
        if all_results['dopants']:
            all_results['dopant_df'] = pd.DataFrame(all_results['dopants'])
        
        return all_results
    
    def create_parameter_distribution_chart(self, param_data: pd.DataFrame, param_name: str) -> go.Figure:
        """Create distribution chart for a specific parameter"""
        if param_data.empty:
            return None
        
        fig = go.Figure()
        
        # Add histogram
        fig.add_trace(go.Histogram(
            x=param_data['value'],
            nbinsx=30,
            name='Distribution',
            marker_color=self.colors.get(param_name, '#3B82F6'),
            opacity=0.7,
            hovertemplate='Value: %{x}<br>Count: %{y}<extra></extra>'
        ))
        
        # Calculate statistics
        mean_val = param_data['value'].mean()
        median_val = param_data['value'].median()
        std_val = param_data['value'].std()
        
        fig.update_layout(
            title={
                'text': f"Distribution of {self.property_names.get(param_name, param_name)}",
                'font': {'size': 20, 'color': '#1E3A8A'}
            },
            xaxis_title=self.property_names.get(param_name, param_name),
            yaxis_title="Count",
            height=500,
            template="plotly_white",
            showlegend=True,
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
            hoverlabel=dict(bgcolor="white", font_size=12),
            annotations=[
                dict(
                    x=0.02, y=0.98,
                    xref="paper", yref="paper",
                    text=f"Mean: {mean_val:.2f}<br>Median: {median_val:.2f}<br>Std: {std_val:.2f}",
                    showarrow=False,
                    bgcolor="white",
                    bordercolor="gray",
                    borderwidth=1,
                    borderpad=4,
                    font=dict(size=12)
                )
            ]
        )
        
        return fig
    
    def create_dopant_concentration_heatmap(self, dopant_df: pd.DataFrame) -> go.Figure:
        """Create heatmap of dopant concentrations by category"""
        if dopant_df.empty or 'category' not in dopant_df.columns:
            return None
        
        # Group by dopant and category
        grouped = dopant_df.groupby(['dopant', 'category']).agg({
            'concentration': ['mean', 'count', 'std']
        }).round(2)
        
        # Flatten column names
        grouped.columns = ['_'.join(col).strip() for col in grouped.columns]
        grouped = grouped.reset_index()
        
        # Create pivot table for heatmap
        pivot = grouped.pivot_table(
            values='concentration_mean',
            index='dopant',
            columns='category',
            aggfunc='mean'
        ).fillna(0)
        
        # Sort by average concentration
        pivot['avg'] = pivot.mean(axis=1)
        pivot = pivot.sort_values('avg', ascending=False).drop('avg', axis=1)
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=pivot.values,
            x=pivot.columns.tolist(),
            y=pivot.index.tolist(),
            colorscale='Viridis',
            colorbar=dict(title="Avg Concentration (wt%)"),
            hovertemplate="Dopant: %{y}<br>Category: %{x}<br>Concentration: %{z:.2f} wt%<br><extra></extra>",
            text=pivot.values.round(2),
            texttemplate="%{text} wt%",
            textfont=dict(size=10, color="white")
        ))
        
        fig.update_layout(
            title={
                'text': "Dopant Concentration by Category",
                'font': {'size': 20, 'color': '#1E3A8A'}
            },
            xaxis_title="Dopant Category",
            yaxis_title="Dopant",
            height=600,
            width=800,
            template="plotly_white"
        )
        
        return fig
    
    def create_dopant_concentration_boxplot(self, dopant_df: pd.DataFrame) -> go.Figure:
        """Create box plot of dopant concentrations by category"""
        if dopant_df.empty or 'category' not in dopant_df.columns:
            return None
        
        fig = go.Figure()
        
        # Get unique categories
        categories = dopant_df['category'].unique()
        
        for category in categories:
            category_data = dopant_df[dopant_df['category'] == category]['concentration']
            if len(category_data) > 0:
                fig.add_trace(go.Box(
                    y=category_data,
                    name=category,
                    marker_color=self.dopant_colors.get(category, '#7F7F7F'),
                    boxmean='sd'
                ))
        
        fig.update_layout(
            title={
                'text': "Dopant Concentration Distribution by Category",
                'font': {'size': 20, 'color': '#1E3A8A'}
            },
            yaxis_title="Concentration (wt%)",
            xaxis_title="Dopant Category",
            height=500,
            template="plotly_white",
            showlegend=False
        )
        
        return fig
    
    def create_radar_chart_comparison(self, param_dfs: Dict[str, pd.DataFrame], 
                                    selected_params: List[str]) -> go.Figure:
        """Create radar chart comparing multiple parameters"""
        if not param_dfs or not selected_params:
            return None
        
        # Calculate average values for each parameter
        param_means = {}
        param_counts = {}
        
        for param in selected_params:
            if param in param_dfs and not param_dfs[param].empty:
                param_means[param] = param_dfs[param]['value'].mean()
                param_counts[param] = len(param_dfs[param])
            else:
                param_means[param] = 0
                param_counts[param] = 0
        
        if not param_means:
            return None
        
        # Normalize values for radar chart (0-1 scale)
        max_vals = {
            'd33': 100,  # Typical max d33 for PVDF composites
            'beta_phase': 100,  # Percentage
            'concentration': 20,  # wt%
            'voltage': 50,  # Volts
            'dielectric': 100,  # εr
            'youngs_modulus': 10  # GPa
        }
        
        normalized_means = {}
        for param, mean_val in param_means.items():
            if mean_val > 0:
                max_val = max_vals.get(param, mean_val * 2)
                normalized_means[param] = min(mean_val / max_val, 1.0)
            else:
                normalized_means[param] = 0
        
        # Create radar chart
        categories = [self.property_names.get(p, p) for p in selected_params]
        values = [normalized_means[p] for p in selected_params]
        values += values[:1]  # Close the polygon
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories + [categories[0]],
            fill='toself',
            fillcolor='rgba(59, 130, 246, 0.3)',
            line=dict(color='#3B82F6', width=2),
            name='Average Values',
            hovertemplate="%{theta}: %{r:.2f} (normalized)<br>Actual: %{customdata}<extra></extra>",
            customdata=[f"{param_means[p]:.2f} {self.units.get(p, '')}" for p in selected_params] + 
                      [f"{param_means[selected_params[0]]:.2f} {self.units.get(selected_params[0], '')}"]
        ))
        
        # Add reference lines
        for i in [0.25, 0.5, 0.75, 1.0]:
            fig.add_trace(go.Scatterpolar(
                r=[i] * (len(categories) + 1),
                theta=categories + [categories[0]],
                mode='lines',
                line=dict(color='gray', width=0.5, dash='dash'),
                showlegend=False,
                hoverinfo='skip'
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1.1],
                    tickvals=[0, 0.25, 0.5, 0.75, 1.0],
                    ticktext=['0', '0.25', '0.5', '0.75', '1.0'],
                    tickangle=0,
                    tickfont=dict(size=12)
                ),
                angularaxis=dict(
                    tickfont=dict(size=12),
                    rotation=90
                )
            ),
            showlegend=True,
            title={
                'text': "Normalized Parameter Comparison",
                'font': {'size': 20, 'color': '#1E3A8A'}
            },
            height=600,
            width=700,
            template="plotly_white",
            legend=dict(
                font=dict(size=12)
            )
        )
        
        return fig
    
    def create_sunburst_dopant_hierarchy(self, dopant_df: pd.DataFrame) -> go.Figure:
        """Create sunburst chart for dopant hierarchy"""
        if dopant_df.empty or 'category' not in dopant_df.columns:
            return None
        
        # Prepare hierarchical data
        hierarchy_data = dopant_df.groupby(['category', 'dopant']).agg({
            'concentration': ['mean', 'count']
        }).reset_index()
        
        hierarchy_data.columns = ['category', 'dopant', 'avg_concentration', 'count']
        
        # Filter out categories with too few data points
        hierarchy_data = hierarchy_data[hierarchy_data['count'] >= 1]
        
        if hierarchy_data.empty:
            return None
        
        # Create sunburst
        fig = px.sunburst(
            hierarchy_data,
            path=['category', 'dopant'],
            values='count',
            color='avg_concentration',
            color_continuous_scale='RdYlBu_r',
            range_color=[hierarchy_data['avg_concentration'].min(), 
                        hierarchy_data['avg_concentration'].max()],
            title="Dopant Hierarchy and Concentration Analysis",
            height=700,
            hover_data=['avg_concentration', 'count'],
            labels={
                'avg_concentration': 'Avg Concentration (wt%)',
                'count': 'Number of Studies'
            }
        )
        
        fig.update_layout(
            template="plotly_white",
            title_font=dict(size=20, color='#1E3A8A'),
            coloraxis_colorbar=dict(
                title="Concentration (wt%)",
                thickness=20
            )
        )
        
        return fig
    
    def create_parameter_correlation_matrix(self, param_dfs: Dict[str, pd.DataFrame]) -> go.Figure:
        """Create correlation matrix between different parameters"""
        # Create combined dataframe with all parameters
        combined_data = []
        
        for param, df in param_dfs.items():
            if not df.empty:
                # Take average value per paper for each parameter
                paper_values = df.groupby('paper_id')['value'].mean().reset_index()
                paper_values['parameter'] = param
                combined_data.append(paper_values[['paper_id', 'parameter', 'value']])
        
        if len(combined_data) < 2:
            return None
        
        combined_df = pd.concat(combined_data, ignore_index=True)
        
        # Pivot to get parameter matrix
        pivot_df = combined_df.pivot_table(
            index='paper_id',
            columns='parameter',
            values='value'
        )
        
        # Calculate correlation matrix
        corr_matrix = pivot_df.corr()
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns.tolist(),
            y=corr_matrix.index.tolist(),
            colorscale='RdBu_r',
            zmid=0,
            colorbar=dict(title="Correlation"),
            text=corr_matrix.values.round(2),
            texttemplate="%{text}",
            textfont=dict(size=12, color='black'),
            hovertemplate="Parameter X: %{x}<br>Parameter Y: %{y}<br>Correlation: %{z:.2f}<extra></extra>"
        ))
        
        fig.update_layout(
            title={
                'text': "Parameter Correlation Matrix",
                'font': {'size': 20, 'color': '#1E3A8A'}
            },
            height=600,
            width=700,
            template="plotly_white",
            xaxis_title="Parameters",
            yaxis_title="Parameters"
        )
        
        return fig
    
    def create_d33_vs_concentration_scatter(self, param_dfs: Dict[str, pd.DataFrame], 
                                          dopant_df: pd.DataFrame) -> go.Figure:
        """Create scatter plot of d33 vs dopant concentration"""
        if 'd33' not in param_dfs or param_dfs['d33'].empty or dopant_df.empty:
            return None
        
        # Merge d33 data with dopant data by paper_id
        d33_data = param_dfs['d33'][['paper_id', 'value']].rename(columns={'value': 'd33'})
        
        # For papers with multiple d33 values, take average
        d33_avg = d33_data.groupby('paper_id')['d33'].mean().reset_index()
        
        # For dopants, take average concentration per paper
        dopant_avg = dopant_df.groupby('paper_id')['concentration'].mean().reset_index()
        
        # Merge
        merged = pd.merge(d33_avg, dopant_avg, on='paper_id', how='inner')
        
        if merged.empty:
            return None
        
        # Create scatter plot
        fig = px.scatter(
            merged,
            x='concentration',
            y='d33',
            trendline='ols',
            trendline_color_override='red',
            title='d33 vs Dopant Concentration',
            labels={
                'concentration': 'Dopant Concentration (wt%)',
                'd33': 'd33 Coefficient (pC/N)'
            },
            height=500
        )
        
        # Add correlation coefficient
        correlation = merged['concentration'].corr(merged['d33'])
        
        fig.update_layout(
            annotations=[
                dict(
                    x=0.05, y=0.95,
                    xref="paper", yref="paper",
                    text=f"Correlation: {correlation:.3f}",
                    showarrow=False,
                    bgcolor="white",
                    bordercolor="gray",
                    borderwidth=1,
                    borderpad=4,
                    font=dict(size=12)
                )
            ],
            template="plotly_white"
        )
        
        return fig
    
    def generate_summary_statistics(self, analysis_results: Dict[str, Any]) -> pd.DataFrame:
        """Generate summary statistics table"""
        summary_data = []
        
        for param, df in analysis_results.get('parameter_dfs', {}).items():
            if not df.empty:
                summary_data.append({
                    'Parameter': self.property_names.get(param, param),
                    'Count': len(df),
                    'Mean': df['value'].mean(),
                    'Median': df['value'].median(),
                    'Std Dev': df['value'].std(),
                    'Min': df['value'].min(),
                    'Max': df['value'].max(),
                    'Unit': self.units.get(param, '')
                })
        
        return pd.DataFrame(summary_data)

# ==============================
# STREAMLIT APPLICATION
# ==============================
def main():
    """Main Streamlit application for quantitative NER analysis"""
    
    st.markdown('<h1 class="main-header">🔬 Quantitative NER Analyzer for PVDF Materials</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Extract and Analyze d33 Coefficients, Dopant Concentrations, and Material Properties from Research Papers</p>', unsafe_allow_html=True)
    
    # Initialize session state
    if 'ner_analyzer' not in st.session_state:
        st.session_state.ner_analyzer = QuantitativeNERAnalyzer()
    
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    
    if 'papers_data' not in st.session_state:
        st.session_state.papers_data = None
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        
        # Database selection
        db_dir = "knowledge_database"
        available_dbs = []
        
        if os.path.exists(db_dir):
            db_files = [f for f in os.listdir(db_dir) if f.endswith('.db')]
            available_dbs = db_files
        
        use_sample = False
        
        if not available_dbs:
            st.warning("No database files found in 'knowledge_database/' directory!")
            st.info("You can:")
            st.markdown("1. Place SQLite database files in `knowledge_database/` folder")
            st.markdown("2. Use sample data for demonstration")
            use_sample = st.checkbox("Use sample data for demonstration", value=True)
        else:
            selected_db = st.selectbox("Select Database", available_dbs)
            db_path = os.path.join(db_dir, selected_db)
            use_sample = st.checkbox("Use sample data instead", value=False)
        
        # Analysis parameters
        st.markdown("### 🔬 Analysis Parameters")
        max_papers = st.slider("Maximum papers to analyze", 10, 5000, 500, 10)
        
        text_source = st.selectbox(
            "Text Source for Analysis",
            ["abstract", "full_text", "content", "title"],
            index=0,
            help="Select which text field to analyze for parameter extraction"
        )
        
        # NER settings
        st.markdown("### 🎯 NER Settings")
        extract_dopants = st.checkbox("Extract dopant concentrations", value=True)
        
        # Actions
        st.markdown("### ⚡ Actions")
        analyze_btn = st.button("🚀 Start NER Analysis", type="primary", use_container_width=True)
        
        if st.button("🔄 Clear Results", use_container_width=True):
            for key in ['analysis_results', 'papers_data']:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()
        
        # Statistics
        if st.session_state.analysis_results:
            st.markdown("### 📊 Analysis Summary")
            results = st.session_state.analysis_results
            st.metric("Papers Analyzed", results.get('papers_analyzed', 0))
            st.metric("PVDF Papers", results.get('papers_with_pvdf', 0))
            st.metric("Quantitative Data", results.get('papers_with_quantitative_data', 0))
            st.metric("Total Extractions", results.get('total_extractions', 0))
        
        # Information
        with st.expander("ℹ️ About This Tool"):
            st.markdown("""
            **Quantitative NER Analyzer** extracts numerical parameters from materials science literature.
            
            **Extracts:**
            - **d33 coefficients** (pC/N)
            - **β-phase content** (%)  
            - **Dopant concentrations** (wt%, vol%)
            - **Dielectric constants** (εr)
            - **Output voltages** (V)
            - **Young's modulus** (GPa)
            
            **Features:**
            - Regex-based NER for quantitative parameters
            - Dopant categorization and analysis
            - Statistical distributions and correlations
            - Publication-quality visualizations
            - Data export capabilities
            
            **Methodology:**
            1. Identify PVDF-related papers
            2. Extract quantitative parameters using pattern matching
            3. Categorize dopants and concentrations
            4. Generate statistical analysis and visualizations
            """)
    
    # Main analysis workflow
    if analyze_btn or use_sample:
        with st.spinner("🔍 Performing quantitative NER analysis..."):
            try:
                if use_sample:
                    # Create sample data
                    st.info("Using sample data for demonstration...")
                    sample_data = create_sample_data()
                    papers_df = sample_data['papers_df']
                    st.session_state.papers_data = papers_df
                    
                    # Perform analysis
                    analyzer = st.session_state.ner_analyzer
                    analysis_results = analyzer.analyze_papers_batch(papers_df, 'abstract')
                    st.session_state.analysis_results = analysis_results
                    
                else:
                    # Load from database
                    db_manager = DatabaseManager(db_path)
                    papers_df = db_manager.get_papers_data()
                    
                    if papers_df.empty:
                        st.error("No papers found in database!")
                        return
                    
                    # Limit papers for performance
                    papers_df = papers_df.iloc[:max_papers].copy()
                    st.session_state.papers_data = papers_df
                    
                    # Perform NER analysis
                    analyzer = st.session_state.ner_analyzer
                    analysis_results = analyzer.analyze_papers_batch(papers_df, text_source)
                    st.session_state.analysis_results = analysis_results
                
                # Display results summary
                if analysis_results:
                    st.success(f"""
                    ✅ Analysis Complete!
                    
                    **Summary:**
                    - 📄 **Total papers analyzed**: {analysis_results['papers_analyzed']}
                    - 🧪 **PVDF-related papers**: {analysis_results['papers_with_pvdf']}
                    - 📊 **Papers with quantitative data**: {analysis_results['papers_with_quantitative_data']}
                    - 🔢 **Total extractions**: {analysis_results['total_extractions']}
                    - 📈 **Parameters extracted**: {len(analysis_results.get('parameter_dfs', {}))}
                    """)
                
            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")
                logger.error(f"Analysis failed: {str(e)}", exc_info=True)
    
    # Display results
    if st.session_state.analysis_results:
        analysis_results = st.session_state.analysis_results
        analyzer = st.session_state.ner_analyzer
        
        # Create tabs for different visualizations
        tabs = st.tabs([
            "📊 Parameter Distributions",
            "🔥 Dopant Analysis", 
            "📈 Radar Comparison",
            "🌳 Hierarchical View",
            "🔗 Correlations",
            "📋 Summary & Export"
        ])
        
        # Tab 1: Parameter Distributions
        with tabs[0]:
            st.markdown("### 📊 Quantitative Parameter Distributions")
            
            param_dfs = analysis_results.get('parameter_dfs', {})
            
            if not param_dfs:
                st.warning("No quantitative parameters extracted.")
            else:
                # Select parameters to display
                available_params = list(param_dfs.keys())
                selected_params = st.multiselect(
                    "Select parameters to visualize",
                    options=available_params,
                    default=available_params[:min(3, len(available_params))],
                    key="param_viz"
                )
                
                # Create charts for each selected parameter
                for param in selected_params:
                    if param in param_dfs and not param_dfs[param].empty:
                        st.markdown(f"#### {analyzer.property_names.get(param, param)}")
                        
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            fig = analyzer.create_parameter_distribution_chart(param_dfs[param], param)
                            if fig:
                                st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            # Display sample extracted values
                            st.markdown("**Sample Extracted Values:**")
                            sample_df = param_dfs[param].head(5)[['value', 'unit', 'dopant', 'context']].copy()
                            for idx, row in sample_df.iterrows():
                                with st.container():
                                    st.markdown(f"""
                                    <div class="parameter-box">
                                    <div class="metric-highlight">{row['value']} {row['unit']}</div>
                                    <small>Dopant: {row.get('dopant', 'N/A')}</small><br>
                                    <small><i>{row['context'][:100]}...</i></small>
                                    </div>
                                    """, unsafe_allow_html=True)
        
        # Tab 2: Dopant Analysis
        with tabs[1]:
            st.markdown("### 🔥 Dopant Concentration Analysis")
            
            if 'dopant_df' in analysis_results and not analysis_results['dopant_df'].empty:
                dopant_df = analysis_results['dopant_df']
                
                # Overall statistics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Unique Dopants", dopant_df['dopant'].nunique())
                with col2:
                    st.metric("Avg Concentration", f"{dopant_df['concentration'].mean():.2f} wt%")
                with col3:
                    st.metric("Max Concentration", f"{dopant_df['concentration'].max():.2f} wt%")
                with col4:
                    st.metric("Min Concentration", f"{dopant_df['concentration'].min():.2f} wt%")
                
                # Heatmap
                st.markdown("#### Concentration Heatmap by Dopant Category")
                fig = analyzer.create_dopant_concentration_heatmap(dopant_df)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                
                # Box plot
                st.markdown("#### Concentration Distribution by Category")
                fig = analyzer.create_dopant_concentration_boxplot(dopant_df)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                
                # Dopant frequency
                st.markdown("#### Most Frequently Studied Dopants")
                dopant_counts = dopant_df['dopant'].value_counts().head(10)
                
                fig = px.bar(
                    x=dopant_counts.values,
                    y=dopant_counts.index,
                    orientation='h',
                    title="Top 10 Most Studied Dopants in PVDF",
                    labels={'x': 'Number of Studies', 'y': 'Dopant'},
                    color=dopant_counts.values,
                    color_continuous_scale='Viridis'
                )
                fig.update_layout(height=400, template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)
                
                # Raw data
                with st.expander("📋 View Raw Dopant Data"):
                    st.dataframe(dopant_df, use_container_width=True)
            else:
                st.info("No dopant concentration data extracted. Try adjusting analysis parameters.")
        
        # Tab 3: Radar Comparison
        with tabs[2]:
            st.markdown("### 📈 Multi-Parameter Radar Comparison")
            
            param_dfs = analysis_results.get('parameter_dfs', {})
            
            if len(param_dfs) >= 2:
                available_params = list(param_dfs.keys())
                selected_params = st.multiselect(
                    "Select parameters for radar chart",
                    options=available_params,
                    default=available_params[:min(6, len(available_params))],
                    key="radar_params"
                )
                
                if len(selected_params) >= 2:
                    fig = analyzer.create_radar_chart_comparison(param_dfs, selected_params)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Add interpretation
                        st.markdown("""
                        **How to interpret the radar chart:**
                        - **Larger area**: Higher overall parameter values
                        - **Shape symmetry**: Balanced property enhancement
                        - **Peaks**: Parameters with highest relative values
                        - **Values are normalized** (0-1 scale) for comparison
                        """)
                else:
                    st.warning("Select at least 2 parameters for radar chart.")
            else:
                st.info("Need at least 2 different parameter types for radar comparison.")
        
        # Tab 4: Hierarchical View
        with tabs[3]:
            st.markdown("### 🌳 Hierarchical Dopant Analysis")
            
            if 'dopant_df' in analysis_results and not analysis_results['dopant_df'].empty:
                # Sunburst chart
                st.markdown("#### Dopant Hierarchy Sunburst")
                fig = analyzer.create_sunburst_dopant_hierarchy(analysis_results['dopant_df'])
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                
                # Category analysis
                st.markdown("#### Dopant Category Analysis")
                if 'category' in analysis_results['dopant_df'].columns:
                    category_stats = analysis_results['dopant_df'].groupby('category').agg({
                        'concentration': ['mean', 'count', 'std'],
                        'dopant': 'nunique'
                    }).round(2)
                    
                    # Flatten column names
                    category_stats.columns = ['Avg Conc (wt%)', 'Studies', 'Std Dev', 'Unique Dopants']
                    st.dataframe(category_stats, use_container_width=True)
            else:
                st.info("No hierarchical data available. Enable dopant extraction in analysis settings.")
        
        # Tab 5: Correlations
        with tabs[4]:
            st.markdown("### 🔗 Parameter Correlations")
            
            param_dfs = analysis_results.get('parameter_dfs', {})
            
            # Correlation matrix
            st.markdown("#### Parameter Correlation Matrix")
            if len(param_dfs) >= 2:
                fig = analyzer.create_parameter_correlation_matrix(param_dfs)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Correlation insights
                    st.markdown("""
                    **Interpreting Correlations:**
                    - **Positive values** (blue): Parameters increase together
                    - **Negative values** (red): Parameters have inverse relationship
                    - **Near zero**: Little to no relationship
                    """)
            else:
                st.info("Need at least 2 parameter types for correlation analysis.")
            
            # d33 vs Concentration scatter
            st.markdown("#### d33 vs Dopant Concentration")
            if 'dopant_df' in analysis_results:
                fig = analyzer.create_d33_vs_concentration_scatter(param_dfs, analysis_results['dopant_df'])
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
        
        # Tab 6: Summary & Export
        with tabs[5]:
            st.markdown("### 📋 Summary Statistics & Data Export")
            
            # Summary statistics table
            st.markdown("#### 📊 Parameter Summary Statistics")
            summary_df = analyzer.generate_summary_statistics(analysis_results)
            if not summary_df.empty:
                st.dataframe(summary_df, use_container_width=True)
            
            # Export options
            st.markdown("#### 📥 Export Extracted Data")
            
            col1, col2, col3 = st.columns(3)
            
            # Export parameter data
            if analysis_results.get('parameter_dfs'):
                with col1:
                    # Combine all parameter data
                    all_param_data = []
                    for param, df in analysis_results['parameter_dfs'].items():
                        df['parameter'] = param
                        all_param_data.append(df)
                    
                    if all_param_data:
                        combined_df = pd.concat(all_param_data, ignore_index=True)
                        csv = combined_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            "📊 Download Parameter Data (CSV)",
                            csv,
                            "parameter_extractions.csv",
                            "text/csv",
                            use_container_width=True
                        )
            
            # Export dopant data
            if 'dopant_df' in analysis_results:
                with col2:
                    csv = analysis_results['dopant_df'].to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "🧪 Download Dopant Data (CSV)",
                        csv,
                        "dopant_extractions.csv",
                        "text/csv",
                        use_container_width=True
                    )
            
            # Export summary report
            with col3:
                # Create comprehensive report
                report = generate_analysis_report(analysis_results, analyzer)
                st.download_button(
                    "📄 Download Analysis Report (TXT)",
                    report,
                    "ner_analysis_report.txt",
                    "text/plain",
                    use_container_width=True
                )
            
            # Sample extractions
            st.markdown("#### 🔍 Sample Extractions")
            if analysis_results.get('parameter_dfs'):
                with st.expander("View sample extracted sentences"):
                    for param, df in list(analysis_results['parameter_dfs'].items())[:3]:
                        if not df.empty:
                            st.markdown(f"**{analyzer.property_names.get(param, param)}:**")
                            for i, row in df.head(2).iterrows():
                                st.markdown(f"- `{row['sentence'][:200]}...`")
                                st.markdown(f"  *Extracted value: {row['value']} {row['unit']}*")
                                st.markdown("---")
    
    else:
        # Welcome screen
        st.markdown("""
        <div style="padding: 2rem; text-align: center; background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%); 
                    border-radius: 15px; color: white; margin-bottom: 2rem;">
            <h2>🔬 Quantitative NER Analyzer</h2>
            <p style="font-size: 1.2rem;">Extract and Analyze Material Parameters from PVDF Research</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Feature cards
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="ner-card">
                <h3>📊 Parameter Extraction</h3>
                <p>Automatically extract quantitative parameters from research papers:</p>
                <ul>
                    <li>d33 coefficients (pC/N)</li>
                    <li>Dopant concentrations (wt%)</li>
                    <li>β-phase content (%)</li>
                    <li>Dielectric constants</li>
                    <li>Mechanical properties</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="ner-card">
                <h3>📈 Advanced Analysis</h3>
                <p>Comprehensive analysis and visualization:</p>
                <ul>
                    <li>Statistical distributions</li>
                    <li>Dopant categorization</li>
                    <li>Parameter correlations</li>
                    <li>Radar chart comparisons</li>
                    <li>Hierarchical analysis</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # Quick start guide
        with st.expander("🚀 Quick Start Guide"):
            st.markdown("""
            1. **Prepare your database:**
               - Place SQLite database files in `knowledge_database/` folder
               - Database should contain papers with abstracts/full text
            
            2. **Configure analysis:**
               - Select database file from sidebar
               - Set maximum papers to analyze
               - Choose text source (abstract, full_text, etc.)
            
            3. **Run analysis:**
               - Click "Start NER Analysis"
               - View results in different tabs
               - Export data for further analysis
            
            4. **Interpret results:**
               - Check parameter distributions
               - Analyze dopant concentrations
               - Examine parameter correlations
               - Export for publication
            
            **Sample Data:** If you don't have a database, check "Use sample data" in the sidebar.
            """)

def create_sample_data() -> Dict[str, Any]:
    """Create sample data for demonstration"""
    np.random.seed(42)
    
    # Create sample papers
    n_papers = 50
    papers = []
    
    # Sample dopants and their typical concentrations
    dopant_info = {
        "ZnO": {"category": "Metal Oxides", "conc_range": (1.0, 10.0)},
        "TiO₂": {"category": "Metal Oxides", "conc_range": (0.5, 5.0)},
        "BaTiO₃": {"category": "Metal Oxides", "conc_range": (5.0, 20.0)},
        "CNT": {"category": "Carbon-Based", "conc_range": (0.1, 3.0)},
        "Graphene": {"category": "Carbon-Based", "conc_range": (0.05, 2.0)},
        "PZT": {"category": "Ceramics", "conc_range": (10.0, 30.0)},
        "MoS₂": {"category": "2D Materials", "conc_range": (0.5, 5.0)},
        "Ag NPs": {"category": "Nanoparticles", "conc_range": (0.5, 5.0)},
        "Fe₂O₃": {"category": "Metal Oxides", "conc_range": (2.0, 15.0)},
        "Al₂O₃": {"category": "Metal Oxides", "conc_range": (3.0, 12.0)}
    }
    
    # Sample abstracts with quantitative data
    for i in range(n_papers):
        paper_id = f'paper_{i+1:04d}'
        
        # Randomly select 1-2 dopants
        n_dopants = np.random.randint(1, 3)
        selected_dopants = np.random.choice(list(dopant_info.keys()), n_dopants, replace=False)
        
        # Generate abstract with quantitative data
        abstract_parts = []
        abstract_parts.append(f"Study of PVDF composites ")
        
        if selected_dopants:
            dopant_str = ", ".join(selected_dopants)
            abstract_parts.append(f"doped with {dopant_str}. ")
        
        # Add concentrations
        for dopant in selected_dopants:
            conc_range = dopant_info[dopant]["conc_range"]
            concentration = np.random.uniform(conc_range[0], conc_range[1])
            abstract_parts.append(f"The {dopant} concentration was {concentration:.1f} wt%. ")
        
        # Add d33 values (correlated with dopant type)
        d33_base = 30  # Base d33 for pure PVDF
        
        for dopant in selected_dopants:
            # Enhancement factor based on dopant type
            if dopant in ["CNT", "Graphene"]:
                enhancement = np.random.uniform(1.5, 3.0)
            elif dopant in ["BaTiO₃", "PZT"]:
                enhancement = np.random.uniform(2.0, 4.0)
            else:
                enhancement = np.random.uniform(1.2, 2.5)
            
            d33_value = d33_base * enhancement
            abstract_parts.append(f"The d33 coefficient reached {d33_value:.1f} pC/N for {dopant} composite. ")
        
        # Add beta-phase content (correlated with d33)
        beta_phase = np.random.uniform(60, 95)
        abstract_parts.append(f"The β-phase content was {beta_phase:.1f}%. ")
        
        # Add dielectric constant
        dielectric = np.random.uniform(15, 120)
        abstract_parts.append(f"The dielectric constant was {dielectric:.1f}. ")
        
        # Add voltage output
        voltage = np.random.uniform(5, 45)
        abstract_parts.append(f"The output voltage reached {voltage:.1f} V. ")
        
        # Final part
        abstract_parts.append("The composite was prepared by solution casting method and characterized using XRD, FTIR, and piezoelectric measurements.")
        
        # Combine all parts
        abstract = " ".join(abstract_parts)
        
        papers.append({
            'paper_id': paper_id,
            'title': f'PVDF Composite Study {i+1}: {", ".join(selected_dopants)} Doping',
            'abstract': abstract,
            'full_text': abstract * 3,  # Simulate full text
            'year': np.random.randint(2018, 2024)
        })
    
    papers_df = pd.DataFrame(papers)
    
    return {
        'papers_df': papers_df,
        'dopant_info': dopant_info
    }

def generate_analysis_report(analysis_results: Dict[str, Any], analyzer: QuantitativeNERAnalyzer) -> str:
    """Generate comprehensive analysis report"""
    report_lines = []
    
    report_lines.append("=" * 80)
    report_lines.append("QUANTITATIVE NER ANALYSIS REPORT")
    report_lines.append("=" * 80)
    report_lines.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    
    # Summary statistics
    report_lines.append("SUMMARY STATISTICS")
    report_lines.append("-" * 40)
    report_lines.append(f"Total papers analyzed: {analysis_results.get('papers_analyzed', 0)}")
    report_lines.append(f"PVDF-related papers: {analysis_results.get('papers_with_pvdf', 0)}")
    report_lines.append(f"Papers with quantitative data: {analysis_results.get('papers_with_quantitative_data', 0)}")
    report_lines.append(f"Total extractions: {analysis_results.get('total_extractions', 0)}")
    report_lines.append("")
    
    # Parameter statistics
    param_dfs = analysis_results.get('parameter_dfs', {})
    if param_dfs:
        report_lines.append("PARAMETER EXTRACTION STATISTICS")
        report_lines.append("-" * 40)
        for param, df in param_dfs.items():
            if not df.empty:
                report_lines.append(f"\n{analyzer.property_names.get(param, param)}:")
                report_lines.append(f"  Count: {len(df)}")
                report_lines.append(f"  Mean: {df['value'].mean():.2f} {analyzer.units.get(param, '')}")
                report_lines.append(f"  Median: {df['value'].median():.2f}")
                report_lines.append(f"  Std Dev: {df['value'].std():.2f}")
                report_lines.append(f"  Range: {df['value'].min():.2f} - {df['value'].max():.2f}")
    
    # Dopant statistics
    if 'dopant_df' in analysis_results and not analysis_results['dopant_df'].empty:
        dopant_df = analysis_results['dopant_df']
        report_lines.append("\n" + "=" * 80)
        report_lines.append("DOPANT ANALYSIS")
        report_lines.append("-" * 40)
        report_lines.append(f"Total dopant extractions: {len(dopant_df)}")
        report_lines.append(f"Unique dopants: {dopant_df['dopant'].nunique()}")
        report_lines.append(f"Average concentration: {dopant_df['concentration'].mean():.2f} wt%")
        report_lines.append(f"Concentration range: {dopant_df['concentration'].min():.2f} - {dopant_df['concentration'].max():.2f} wt%")
        
        # By category
        if 'category' in dopant_df.columns:
            report_lines.append("\nBy Category:")
            for category, group in dopant_df.groupby('category'):
                report_lines.append(f"  {category}: {len(group)} studies, avg {group['concentration'].mean():.2f} wt%")
    
    report_lines.append("\n" + "=" * 80)
    report_lines.append("END OF REPORT")
    report_lines.append("=" * 80)
    
    return "\n".join(report_lines)

# ==============================
# APPLICATION ENTRY POINT
# ==============================
if __name__ == "__main__":
    # Create knowledge_database directory if it doesn't exist
    os.makedirs("knowledge_database", exist_ok=True)
    
    # Run the application
    main()
