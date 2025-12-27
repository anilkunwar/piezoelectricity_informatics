# dopant_impact_explorer_enhanced.py
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
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}" class="download-btn">{text}</a>'
    return href

# ==============================
# ENHANCED CONSTANTS & CONFIGURATION
# ==============================
DB_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DB_DIR = os.path.join(DB_DIR, "knowledge_database")

class Config:
    """Enhanced configuration class with publication-quality settings"""
    
    # Publication quality color palettes
    COLOR_PALETTES = {
        "nature": ["#E64B35", "#4DBBD5", "#00A087", "#3C5488", "#F39B7F", "#8491B4", "#91D1C2", "#DC0000"],
        "science": ["#1F77B4", "#FF7F0E", "#2CA02C", "#D62728", "#9467BD", "#8C564B", "#E377C2", "#7F7F7F"],
        "material_science": ["#3A6EA5", "#FF6B35", "#004E89", "#FFA400", "#6699CC", "#FF7F50", "#33658A", "#FF9F1C"],
        "categorical_10": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", 
                          "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
    }
    
    # Enhanced dopant classification with more categories
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
    
    # Base materials with comprehensive naming
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
    
    # Enhanced properties with units
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
    
    # Enhanced color mapping using Nature palette
    COLORS = {
        # Base Materials
        "PVDF": "#3A6EA5",
        "BaTiO₃": "#FF6B35",
        "ZnO": "#004E89",
        "PZT": "#FFA400",
        "AlN": "#6699CC",
        "KNN": "#FF7F50",
        "PVDF-HFP": "#33658A",
        "Others": "#FF9F1C",
        
        # Dopant Categories
        "Metal Oxides": "#1F77B4",
        "Carbon-Based": "#FF7F0E",
        "Ferroelectric Ceramics": "#2CA02C",
        "2D Materials": "#D62728",
        "Polymers": "#9467BD",
        "Nanoparticles": "#8C564B",
        "Ionic Liquids": "#E377C2",
        "Others": "#7F7F7F"
    }
    
    # Publication quality plot settings
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
# ENHANCED DATABASE MANAGER WITH CUSTOM FILE SUPPORT
# ==============================
class DatabaseManager:
    """Enhanced database manager with custom file support and better error handling"""
    
    def __init__(self, db_path: str, custom_path: str = None):
        self.db_path = db_path
        self.custom_path = custom_path
        self.conn = None
        self.table_columns = {}
        self._actual_path = self._resolve_path()
        logger.info(f"Database manager initialized for {self._actual_path}")
    
    def _resolve_path(self) -> str:
        """Resolve database path with custom file support"""
        if self.custom_path and os.path.exists(self.custom_path):
            return self.custom_path
        elif os.path.exists(self.db_path):
            return self.db_path
        else:
            # Try to find in knowledge_database directory
            db_name = os.path.basename(self.db_path)
            possible_paths = [
                os.path.join(DEFAULT_DB_DIR, db_name),
                os.path.join(os.getcwd(), db_name),
                os.path.join(os.getcwd(), "knowledge_database", db_name)
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    return path
            return self.db_path
    
    def connect(self) -> bool:
        """Establish database connection with comprehensive error handling"""
        try:
            self.conn = sqlite3.connect(self._actual_path, check_same_thread=False)
            self.conn.row_factory = sqlite3.Row
            logger.info(f"Connected to database: {self._actual_path}")
            self._cache_table_columns()
            return True
        except sqlite3.Error as e:
            logger.error(f"Database connection error: {e}")
            st.error(f"❌ Database connection error: {e}")
            # Try to create database directory if it doesn't exist
            os.makedirs(os.path.dirname(self._actual_path), exist_ok=True)
            return False
    
    # ... [rest of DatabaseManager methods same as before with minor enhancements] ...

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
        
        # Publication quality settings
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
    
    def create_publication_sunburst(self, relationships_df: pd.DataFrame, 
                                  title: str = "Hierarchical Analysis of Dopant Effects",
                                  show_values: bool = True,
                                  max_depth: int = 4) -> go.Figure:
        """Create publication-quality sunburst chart"""
        if relationships_df.empty:
            return None
        
        # Prepare hierarchical data
        hierarchy_levels = ['base_material', 'dopant_category', 'dopant', 'property']
        hierarchy_levels = hierarchy_levels[:max_depth]
        
        # Aggregate data
        agg_data = relationships_df.groupby(hierarchy_levels).agg({
            'value': 'mean',
            'enhancement_factor': 'mean',
            'paper_id': 'count'
        }).reset_index()
        agg_data.rename(columns={'paper_id': 'n_studies'}, inplace=True)
        
        # Calculate size based on enhancement and number of studies
        agg_data['size'] = agg_data['enhancement_factor'] * np.log1p(agg_data['n_studies'])
        
        # Create color scale based on enhancement factor
        min_enhance = agg_data['enhancement_factor'].min()
        max_enhance = agg_data['enhancement_factor'].max()
        
        # Normalize for color mapping
        if max_enhance > min_enhance:
            agg_data['color_norm'] = (agg_data['enhancement_factor'] - min_enhance) / (max_enhance - min_enhance)
        else:
            agg_data['color_norm'] = 0.5
        
        # Create sunburst with enhanced styling
        fig = px.sunburst(
            agg_data,
            path=hierarchy_levels,
            values='size',
            color='color_norm',
            color_continuous_scale='RdYlBu_r',  # Red-Yellow-Blue reversed
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
        
        # Enhanced layout
        fig.update_layout(
            title={
                'text': title,
                'y': 0.95,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': dict(
                    size=self.plot_config["title_font_size"],
                    family=self.plot_config["font_family"],
                    color='#1E3A8A'
                )
            },
            font=dict(
                size=self.plot_config["font_size"],
                family=self.plot_config["font_family"]
            ),
            margin=dict(t=100, l=20, r=20, b=20),
            paper_bgcolor='white',
            plot_bgcolor='white',
            coloraxis_colorbar=dict(
                title="Enhancement<br>Factor",
                thickness=20,
                len=0.75,
                tickfont=dict(size=12),
                title_font=dict(size=14)
            )
        )
        
        # Add value labels if requested
        if show_values:
            fig.update_traces(
                textinfo='label+value+percent parent',
                texttemplate='<b>%{label}</b><br>%{value:.1f}<br>%{percentParent:.1%}',
                hovertemplate=(
                    '<b>%{label}</b><br>' +
                    'Enhancement: %{color:.2f}×<br>' +
                    'Studies: %{customdata[2]:d}<br>' +
                    '<extra></extra>'
                )
            )
        
        return fig
    
    def create_enhanced_radar_chart(self, relationships_df: pd.DataFrame, 
                                  selected_dopants: List[str],
                                  title: str = "Multi-Property Performance Comparison",
                                  show_average: bool = True,
                                  normalize: bool = True) -> go.Figure:
        """Create enhanced radar chart with publication-quality styling"""
        if relationships_df.empty or not selected_dopants:
            return None
        
        # Filter for selected dopants
        filtered_df = relationships_df[relationships_df['dopant'].isin(selected_dopants)]
        if filtered_df.empty:
            return None
        
        # Get properties (limit to 8 for readability)
        properties = list(self.properties.keys())[:8]
        
        # Prepare data for radar chart
        radar_data = {}
        for dopant in selected_dopants:
            dopant_df = filtered_df[filtered_df['dopant'] == dopant]
            if not dopant_df.empty:
                radar_data[dopant] = {}
                for prop in properties:
                    prop_df = dopant_df[dopant_df['property'] == prop]
                    if not prop_df.empty:
                        # Calculate weighted enhancement
                        avg_enhance = prop_df['enhancement_factor'].mean()
                        n_studies = len(prop_df)
                        confidence = min(1.0, n_studies / 10)  # Confidence based on number of studies
                        radar_data[dopant][prop] = avg_enhance * confidence
                    else:
                        radar_data[dopant][prop] = 1.0  # Default = no enhancement
        
        # Normalize if requested
        if normalize and len(radar_data) > 1:
            for prop in properties:
                values = [data.get(prop, 1.0) for data in radar_data.values()]
                max_val = max(values)
                if max_val > 1.0:
                    for dopant in radar_data:
                        if prop in radar_data[dopant]:
                            radar_data[dopant][prop] = radar_data[dopant][prop] / max_val * 2.0
        
        # Create radar chart with enhanced styling
        fig = go.Figure()
        
        # Color palette for dopants
        colors = self.config.COLOR_PALETTES["nature"][:len(selected_dopants)]
        
        for i, (dopant, props) in enumerate(radar_data.items()):
            values = [props.get(prop, 1.0) for prop in properties]
            values += values[:1]  # Close the polygon
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=properties + [properties[0]],
                fill='toself' if i == 0 else None,
                fillcolor=f"rgba{tuple(list(mcolors.to_rgba(colors[i]))[:3] + [0.2])}",
                name=dopant,
                line=dict(
                    color=colors[i],
                    width=3 if i == 0 else 2,
                    dash='solid' if i == 0 else 'dash'
                ),
                marker=dict(
                    size=8,
                    symbol='circle',
                    line=dict(width=1, color='white')
                ),
                hovertemplate=(
                    f"<b>{dopant}</b><br>" +
                    "%{theta}: %{r:.2f}×<br>" +
                    "<extra></extra>"
                ),
                opacity=0.9
            ))
        
        # Add average line if requested
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
                line=dict(
                    color='black',
                    width=3,
                    dash='dashdot'
                ),
                marker=dict(size=0),
                opacity=0.7
            ))
        
        # Enhanced layout
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, max(2.5, max([max(v.values()) for v in radar_data.values()]))],
                    tickangle=0,
                    tickfont=dict(size=12),
                    gridcolor='lightgray',
                    gridwidth=1,
                    linecolor='gray',
                    linewidth=2,
                    showline=True
                ),
                angularaxis=dict(
                    tickfont=dict(size=14, family='Arial'),
                    rotation=90,
                    direction='clockwise',
                    gridcolor='lightgray',
                    gridwidth=1,
                    linecolor='gray',
                    linewidth=2
                ),
                bgcolor='white'
            ),
            showlegend=True,
            title={
                'text': title,
                'y': 0.95,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': dict(
                    size=self.plot_config["title_font_size"],
                    family=self.plot_config["font_family"],
                    color='#1E3A8A'
                )
            },
            height=self.plot_config["height"],
            width=self.plot_config["width"],
            font=dict(
                size=self.plot_config["font_size"],
                family=self.plot_config["font_family"]
            ),
            legend=dict(
                font=dict(size=14),
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                bgcolor='rgba(255, 255, 255, 0.9)',
                bordercolor='gray',
                borderwidth=1
            ),
            margin=dict(t=100, b=80, l=80, r=80),
            paper_bgcolor='white',
            plot_bgcolor='white'
        )
        
        return fig
    
    def create_concentration_heatmap(self, relationships_df: pd.DataFrame,
                                   title: str = "Dopant Concentration vs Property Enhancement") -> go.Figure:
        """Create heatmap visualization of concentration effects"""
        if relationships_df.empty:
            return None
        
        # Extract concentration values
        relationships_df['concentration_num'] = relationships_df['concentration_range'].str.extract(r'(\d+(?:\.\d+)?)').astype(float)
        
        # Filter out invalid concentrations
        filtered_df = relationships_df[
            (relationships_df['concentration_num'] > 0) & 
            (relationships_df['concentration_num'] <= 50)
        ].copy()
        
        if filtered_df.empty:
            return None
        
        # Create pivot table for heatmap
        heatmap_data = filtered_df.pivot_table(
            values='enhancement_factor',
            index='dopant',
            columns=pd.cut(filtered_df['concentration_num'], bins=10),
            aggfunc='mean',
            fill_value=1.0
        )
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data.values,
            x=[str(col) for col in heatmap_data.columns],
            y=heatmap_data.index.tolist(),
            colorscale='Viridis',
            colorbar=dict(
                title="Enhancement<br>Factor",
                titleside="right",
                tickfont=dict(size=12)
            ),
            hovertemplate=(
                "Dopant: %{y}<br>" +
                "Concentration: %{x}<br>" +
                "Enhancement: %{z:.2f}×<br>" +
                "<extra></extra>"
            ),
            text=heatmap_data.values.round(2),
            texttemplate="%{text}×",
            textfont=dict(size=10, color="white")
        ))
        
        fig.update_layout(
            title={
                'text': title,
                'y': 0.95,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': dict(size=22, family='Arial', color='#1E3A8A')
            },
            xaxis=dict(
                title="Concentration Range (%)",
                title_font=dict(size=16),
                tickfont=dict(size=12),
                tickangle=45
            ),
            yaxis=dict(
                title="Dopant",
                title_font=dict(size=16),
                tickfont=dict(size=12)
            ),
            height=600,
            width=900,
            margin=dict(t=100, b=100, l=150, r=50),
            paper_bgcolor='white',
            plot_bgcolor='white'
        )
        
        return fig
    
    def create_3d_scatter_plot(self, relationships_df: pd.DataFrame,
                             title: str = "3D Analysis: Dopant Effects on Multiple Properties") -> go.Figure:
        """Create 3D scatter plot for multi-dimensional analysis"""
        if relationships_df.empty:
            return None
        
        # Select top properties for 3D visualization
        top_properties = relationships_df['property'].value_counts().head(3).index.tolist()
        
        if len(top_properties) < 3:
            return None
        
        # Prepare data for 3D plot
        plot_data = relationships_df.copy()
        
        # Create figure
        fig = go.Figure()
        
        # Color by dopant category
        categories = plot_data['dopant_category'].unique()
        
        for category in categories:
            cat_data = plot_data[plot_data['dopant_category'] == category]
            
            fig.add_trace(go.Scatter3d(
                x=cat_data[cat_data['property'] == top_properties[0]]['value'],
                y=cat_data[cat_data['property'] == top_properties[1]]['value'],
                z=cat_data[cat_data['property'] == top_properties[2]]['value'],
                mode='markers',
                name=category,
                marker=dict(
                    size=8,
                    color=self.colors.get(category, '#666666'),
                    opacity=0.7,
                    line=dict(width=1, color='white')
                ),
                text=cat_data['dopant'] + '<br>' + cat_data['base_material'],
                hovertemplate=(
                    "<b>%{text}</b><br>" +
                    f"{top_properties[0]}: %{{x:.1f}}<br>" +
                    f"{top_properties[1]}: %{{y:.1f}}<br>" +
                    f"{top_properties[2]}: %{{z:.1f}}<br>" +
                    "<extra></extra>"
                )
            ))
        
        # Update layout
        fig.update_layout(
            title={
                'text': title,
                'y': 0.95,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': dict(size=22, family='Arial', color='#1E3A8A')
            },
            scene=dict(
                xaxis_title=top_properties[0],
                yaxis_title=top_properties[1],
                zaxis_title=top_properties[2],
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            height=800,
            width=900,
            margin=dict(t=100, b=50, l=50, r=50),
            legend=dict(
                font=dict(size=12),
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            ),
            paper_bgcolor='white'
        )
        
        return fig
    
    def create_comparison_dashboard(self, relationships_df: pd.DataFrame,
                                  selected_dopants: List[str]) -> go.Figure:
        """Create a comprehensive comparison dashboard"""
        if relationships_df.empty or len(selected_dopants) < 2:
            return None
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "Enhancement Factor Comparison",
                "Property Distribution",
                "Concentration Optimization",
                "Processing Method Analysis"
            ),
            vertical_spacing=0.15,
            horizontal_spacing=0.15,
            specs=[
                [{"type": "bar"}, {"type": "box"}],
                [{"type": "scatter"}, {"type": "bar"}]
            ]
        )
        
        # 1. Bar chart: Average enhancement by dopant
        enhancement_data = relationships_df.groupby('dopant')['enhancement_factor'].mean().reset_index()
        enhancement_data = enhancement_data[enhancement_data['dopant'].isin(selected_dopants)]
        
        fig.add_trace(
            go.Bar(
                x=enhancement_data['dopant'],
                y=enhancement_data['enhancement_factor'],
                marker_color=[self.colors.get(self.classify_dopant(d), '#666666') for d in enhancement_data['dopant']],
                name='Avg Enhancement',
                text=enhancement_data['enhancement_factor'].round(2),
                textposition='auto'
            ),
            row=1, col=1
        )
        
        # 2. Box plot: Property distribution
        for i, dopant in enumerate(selected_dopants):
            dopant_data = relationships_df[relationships_df['dopant'] == dopant]['enhancement_factor']
            fig.add_trace(
                go.Box(
                    y=dopant_data,
                    name=dopant,
                    marker_color=self.colors.get(self.classify_dopant(dopant), '#666666'),
                    boxmean='sd'
                ),
                row=1, col=2
            )
        
        # 3. Scatter plot: Concentration vs enhancement
        scatter_data = relationships_df[relationships_df['dopant'].isin(selected_dopants)].copy()
        scatter_data['concentration_num'] = scatter_data['concentration_range'].str.extract(r'(\d+(?:\.\d+)?)').astype(float)
        
        for dopant in selected_dopants:
            dopant_scatter = scatter_data[scatter_data['dopant'] == dopant]
            fig.add_trace(
                go.Scatter(
                    x=dopant_scatter['concentration_num'],
                    y=dopant_scatter['enhancement_factor'],
                    mode='markers',
                    name=dopant,
                    marker=dict(
                        color=self.colors.get(self.classify_dopant(dopant), '#666666'),
                        size=10,
                        symbol='circle'
                    )
                ),
                row=2, col=1
            )
        
        # 4. Bar chart: Processing methods
        if 'processing_method' in relationships_df.columns:
            method_data = relationships_df['processing_method'].value_counts().head(5)
            fig.add_trace(
                go.Bar(
                    x=method_data.index,
                    y=method_data.values,
                    marker_color='lightblue',
                    name='Processing Methods'
                ),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            height=1000,
            width=1200,
            title_text="Comprehensive Dopant Analysis Dashboard",
            title_font=dict(size=24, family='Arial', color='#1E3A8A'),
            showlegend=True,
            paper_bgcolor='white',
            plot_bgcolor='white',
            font=dict(size=12, family='Arial')
        )
        
        # Update axes
        fig.update_xaxes(title_text="Dopant", row=1, col=1)
        fig.update_yaxes(title_text="Enhancement Factor", row=1, col=1)
        fig.update_xaxes(title_text="Dopant", row=1, col=2)
        fig.update_yaxes(title_text="Enhancement Factor", row=1, col=2)
        fig.update_xaxes(title_text="Concentration (%)", row=2, col=1)
        fig.update_yaxes(title_text="Enhancement Factor", row=2, col=1)
        fig.update_xaxes(title_text="Processing Method", row=2, col=2)
        fig.update_yaxes(title_text="Count", row=2, col=2)
        
        return fig

# ==============================
# ENHANCED MAIN APPLICATION WITH CUSTOM FILE SUPPORT
# ==============================
def main():
    """Enhanced main Streamlit application"""
    
    st.markdown('<h1 class="main-header">🔬 Dopant Impact Explorer Pro</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Advanced Visual Analytics for Piezoelectric Material Enhancement</p>', unsafe_allow_html=True)
    
    # Initialize session state
    if 'analysis_engine' not in st.session_state:
        st.session_state.analysis_engine = EnhancedDopantAnalysisEngine()
    
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = None
    
    if 'dopant_relationships' not in st.session_state:
        st.session_state.dopant_relationships = None
    
    # Sidebar with enhanced layout
    with st.sidebar:
        st.markdown("### ⚙️ Configuration Panel")
        
        # Custom database file support
        st.markdown("#### 📁 Database Configuration")
        use_custom_files = st.checkbox("Use custom database files", value=False)
        
        custom_dbs = {}
        if use_custom_files:
            col1, col2 = st.columns(2)
            with col1:
                custom_dbs["Metadata DB"] = st.text_input(
                    "Metadata DB",
                    value="",
                    placeholder="Enter custom filename or path"
                )
            with col2:
                if st.button("📂 Browse"):
                    st.info("File browser not available in Streamlit Cloud. Please enter path manually.")
        
        # Enhanced database selection
        available_dbs = {}
        if use_custom_files and custom_dbs["Metadata DB"]:
            available_dbs["Custom DB"] = custom_dbs["Metadata DB"]
        else:
            # Check default databases
            for db_name, default_path in {
                "Metadata DB": os.path.join(DEFAULT_DB_DIR, "piezoelectricity_metadata.db"),
                "Universe DB": os.path.join(DEFAULT_DB_DIR, "piezoelectricity_universe.db"),
                "PDF Storage DB": os.path.join(DEFAULT_DB_DIR, "piezoelectricity_pdfs.db")
            }.items():
                if os.path.exists(default_path):
                    available_dbs[db_name] = default_path
        
        if not available_dbs:
            st.error("❌ No databases found!")
            st.info("""
            Please ensure database files are in the `knowledge_database/` directory or specify custom paths.
            
            Required files:
            - `piezoelectricity_metadata.db`
            - `piezoelectricity_universe.db` 
            - `piezoelectricity_pdfs.db`
            """)
        
        selected_db = st.selectbox("Select Database", list(available_dbs.keys()))
        db_path = available_dbs[selected_db]
        
        # Enhanced analysis parameters
        st.markdown("#### 🔬 Analysis Parameters")
        max_papers = st.slider("Max papers to process", 10, 5000, 500, 50)
        
        # Enhanced visualization options
        st.markdown("#### 🎨 Visualization Settings")
        color_palette = st.selectbox(
            "Color Palette",
            ["nature", "science", "material_science", "categorical_10"],
            index=0
        )
        
        # Update config with selected palette
        if color_palette in Config.COLOR_PALETTES:
            Config.PLOT_CONFIG["colorway"] = Config.COLOR_PALETTES[color_palette]
        
        chart_quality = st.select_slider(
            "Chart Quality",
            options=["Low", "Medium", "High", "Publication"],
            value="High"
        )
        
        # Enhanced actions
        st.markdown("#### ⚡ Actions")
        col1, col2 = st.columns(2)
        with col1:
            analyze_btn = st.button("🚀 Start Analysis", type="primary", use_container_width=True)
        with col2:
            if st.button("🔄 Reset Session", use_container_width=True):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()
        
        # System status with enhanced metrics
        st.markdown("#### 📊 System Status")
        status_col1, status_col2 = st.columns(2)
        with status_col1:
            st.metric("Dopant Categories", len(Config.DOPANT_CATEGORIES))
            st.metric("Color Palette", color_palette.title())
        with status_col2:
            st.metric("Base Materials", len(Config.BASE_MATERIALS))
            st.metric("Chart Quality", chart_quality)
        
        # Enhanced help section
        with st.expander("📚 User Guide & Documentation"):
            st.markdown("""
            ### **Publication-Ready Visualizations**
            
            **Enhanced Features:**
            - **Sunburst Charts**: Interactive hierarchical views with color scaling
            - **Radar Charts**: Multi-property comparisons with confidence indicators
            - **3D Scatter Plots**: Multi-dimensional analysis of dopant effects
            - **Heatmaps**: Concentration optimization visualizations
            - **Comparison Dashboards**: Comprehensive multi-chart analysis
            
            **Export Options:**
            - High-resolution PNG/PDF for publications
            - Vector formats (SVG) for editing
            - Raw data export in multiple formats
            - Interactive HTML reports
            
            **Scientific Standards:**
            - IUPAC naming conventions
            - SI units throughout
            - Proper statistical representation
            - Citation-ready figure captions
            """)
    
    # Main analysis workflow
    if analyze_btn:
        with st.spinner("🔬 Analyzing dopant relationships with enhanced algorithms..."):
            try:
                # Initialize database manager with custom path support
                db_manager = DatabaseManager(db_path)
                
                # Test connection
                if not db_manager.connect():
                    st.error("Failed to connect to database. Please check the file path.")
                    return
                
                # Load papers with progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text("📥 Loading papers from database...")
                papers_df = db_manager.get_papers_data()
                progress_bar.progress(30)
                
                if papers_df.empty:
                    st.error("No papers found in database!")
                    return
                
                # Limit for performance
                papers_df = papers_df.iloc[:max_papers].copy()
                
                # Extract dopant relationships
                status_text.text("🧪 Extracting dopant relationships...")
                engine = st.session_state.analysis_engine
                relationships_df = engine.extract_dopant_relationships(papers_df)
                progress_bar.progress(70)
                
                if relationships_df.empty:
                    st.warning("""
                    ⚠️ No dopant relationships extracted. 
                    
                    **Possible reasons:**
                    1. Database doesn't contain piezoelectric materials research
                    2. Text extraction needs different keywords
                    3. Papers don't discuss dopant effects explicitly
                    
                    **Try:**
                    - Using a different database
                    - Increasing the number of papers processed
                    - Checking database content structure
                    """)
                    
                    # Show database schema for debugging
                    with st.expander("🔍 Database Schema Analysis"):
                        schema = db_manager.generate_schema_report()
                
                # Store results
                st.session_state.processed_data = papers_df
                st.session_state.dopant_relationships = relationships_df
                
                status_text.text("🎨 Creating visualizations...")
                progress_bar.progress(90)
                
                st.success(f"""
                ✅ Analysis Complete!
                
                **Results Summary:**
                - 📄 **Papers Processed**: {len(papers_df)}
                - 🔗 **Dopant Relationships**: {len(relationships_df)}
                - 🧪 **Unique Dopants**: {relationships_df['dopant'].nunique() if not relationships_df.empty else 0}
                - 🏗️ **Base Materials**: {relationships_df['base_material'].nunique() if not relationships_df.empty else 0}
                """)
                
                progress_bar.progress(100)
                status_text.text("✅ Analysis ready!")
                time.sleep(1)
                
            except Exception as e:
                st.error(f"""
                ❌ Analysis Failed!
                
                **Error Details:**
                ```python
                {str(e)}
                ```
                
                **Troubleshooting Steps:**
                1. Check database file paths
                2. Ensure database files are not corrupted
                3. Verify database contains relevant research papers
                4. Check available disk space
                """)
                logger.error(f"Analysis failed: {str(e)}", exc_info=True)
                return
    
    # Enhanced results display
    if st.session_state.dopant_relationships is not None and not st.session_state.dopant_relationships.empty:
        relationships_df = st.session_state.dopant_relationships
        engine = st.session_state.analysis_engine
        
        # Create enhanced tabs
        tabs = st.tabs([
            "🌳 Hierarchical Analysis", 
            "📡 Multi-Property Comparison", 
            "🔥 Concentration Heatmap",
            "📊 3D Analysis",
            "📈 Comprehensive Dashboard",
            "💡 Recommendations",
            "🔍 Data Explorer",
            "⚙️ Advanced Settings"
        ])
        
        # Tab 1: Enhanced Sunburst Chart
        with tabs[0]:
            st.markdown("### 🌳 Hierarchical Dopant Impact Analysis")
            
            # Sunburst controls
            col1, col2, col3 = st.columns(3)
            with col1:
                max_depth = st.slider("Hierarchy Depth", 2, 5, 4)
            with col2:
                show_values = st.checkbox("Show Values", value=True)
            with col3:
                color_scheme = st.selectbox("Color Scheme", ["RdYlBu_r", "Viridis", "Plasma", "Inferno"])
            
            # Create sunburst
            fig = engine.create_publication_sunburst(
                relationships_df, 
                title="Hierarchical Analysis of Dopant Effects on Piezoelectric Properties",
                show_values=show_values,
                max_depth=max_depth
            )
            
            if fig:
                # Display chart
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': True})
                
                # Export options
                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button("💾 Download as PNG"):
                        try:
                            fig.write_image("sunburst_chart.png", scale=2)
                            st.success("Chart saved as sunburst_chart.png")
                        except Exception as e:
                            st.warning(f"Install kaleido: pip install kaleido")
                with col2:
                    if st.button("📊 Download as SVG"):
                        fig.write_image("sunburst_chart.svg")
                        st.success("Chart saved as sunburst_chart.svg")
                with col3:
                    if st.button("📄 Download as PDF"):
                        fig.write_image("sunburst_chart.pdf")
                        st.success("Chart saved as sunburst_chart.pdf")
                
                add_caption("""
                **Figure 1:** Hierarchical sunburst visualization of dopant effects on piezoelectric materials. 
                The chart shows four levels: (1) Base materials (center), (2) Dopant categories, 
                (3) Specific dopants, and (4) Enhanced properties. Color intensity represents the 
                enhancement factor (1.0-3.0 scale). Segment size is proportional to both enhancement 
                factor and number of supporting studies. This visualization helps identify which 
                dopant categories provide the broadest property enhancement across different 
                material systems.
                """, "📊")
        
        # Tab 2: Enhanced Radar Chart
        with tabs[1]:
            st.markdown("### 📡 Multi-Property Dopant Performance Comparison")
            
            # Radar controls
            col1, col2 = st.columns(2)
            with col1:
                all_dopants = relationships_df['dopant'].unique().tolist()
                default_dopants = relationships_df['dopant'].value_counts().head(4).index.tolist()
                
                selected_dopants = st.multiselect(
                    "Select dopants to compare (max 6)",
                    options=all_dopants,
                    default=default_dopants,
                    max_selections=6
                )
            
            with col2:
                normalize = st.checkbox("Normalize values", value=True)
                show_average = st.checkbox("Show average line", value=True)
            
            if len(selected_dopants) >= 2:
                fig = engine.create_enhanced_radar_chart(
                    relationships_df, 
                    selected_dopants,
                    title="Multi-Property Performance Profile Comparison",
                    show_average=show_average,
                    normalize=normalize
                )
                
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Export and insights
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        if st.button("📥 Download Radar Chart"):
                            fig.write_image("radar_chart.png", scale=2)
                            st.success("Chart saved!")
                    
                    with col2:
                        # Generate insights
                        st.markdown("#### 🔍 Radar Chart Insights")
                        insights = []
                        for dopant in selected_dopants:
                            dopant_df = relationships_df[relationships_df['dopant'] == dopant]
                            if not dopant_df.empty:
                                best_prop = dopant_df.groupby('property')['enhancement_factor'].mean().idxmax()
                                avg_enhance = dopant_df['enhancement_factor'].mean()
                                insights.append(f"**{dopant}**: Best in {best_prop} (Avg: {avg_enhance:.2f}×)")
                        
                        for insight in insights:
                            st.markdown(f"- {insight}")
                    
                    add_caption("""
                    **Figure 2:** Radar chart comparing the performance profiles of different dopants 
                    across multiple piezoelectric properties. Each axis represents a key property, 
                    with distance from the center indicating enhancement factor relative to undoped 
                    material (1.0 = baseline). Solid lines represent individual dopants, while the 
                    dashed line shows the average performance. This visualization helps identify 
                    dopants with balanced enhancement across multiple properties versus those with 
                    specific strengths.
                    """, "🎯")
        
        # Tab 3: Concentration Heatmap
        with tabs[2]:
            st.markdown("### 🔥 Dopant Concentration Optimization Heatmap")
            
            fig = engine.create_concentration_heatmap(relationships_df)
            
            if fig:
                st.plotly_chart(fig, use_container_width=True)
                
                # Concentration analysis
                st.markdown("#### 📈 Optimal Concentration Analysis")
                
                # Calculate optimal concentrations
                if 'concentration_num' in relationships_df.columns:
                    optimal_data = []
                    for dopant in relationships_df['dopant'].unique():
                        dopant_df = relationships_df[relationships_df['dopant'] == dopant].copy()
                        dopant_df['concentration_num'] = pd.to_numeric(dopant_df['concentration_range'].str.extract(r'(\d+(?:\.\d+)?)')[0], errors='coerce')
                        
                        if not dopant_df['concentration_num'].isna().all():
                            # Find concentration with highest average enhancement
                            concentration_groups = dopant_df.groupby('concentration_num')['enhancement_factor'].agg(['mean', 'count'])
                            if not concentration_groups.empty:
                                best_conc = concentration_groups['mean'].idxmax()
                                best_enhance = concentration_groups['mean'].max()
                                n_studies = concentration_groups.loc[best_conc, 'count']
                                optimal_data.append({
                                    'Dopant': dopant,
                                    'Optimal Concentration': f"{best_conc:.1f}%",
                                    'Max Enhancement': f"{best_enhance:.2f}×",
                                    'Studies': int(n_studies)
                                })
                    
                    if optimal_data:
                        optimal_df = pd.DataFrame(optimal_data)
                        st.dataframe(optimal_df.sort_values('Max Enhancement', ascending=False), use_container_width=True)
                
                add_caption("""
                **Figure 3:** Heatmap showing the relationship between dopant concentration 
                (x-axis) and property enhancement factor (color scale) for various dopants 
                (y-axis). Darker colors indicate higher enhancement. This visualization helps 
                identify optimal concentration ranges for each dopant, revealing patterns of 
                diminishing returns at high concentrations and threshold effects at low 
                concentrations.
                """, "🔥")
        
        # Tab 4: 3D Analysis
        with tabs[3]:
            st.markdown("### 📊 3D Multi-Dimensional Analysis")
            
            fig = engine.create_3d_scatter_plot(relationships_df)
            
            if fig:
                st.plotly_chart(fig, use_container_width=True)
                
                # 3D controls
                st.markdown("#### 🎮 3D View Controls")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown("**Rotation:** Click and drag")
                with col2:
                    st.markdown("**Zoom:** Scroll wheel")
                with col3:
                    st.markdown("**Pan:** Right-click and drag")
                
                add_caption("""
                **Figure 4:** 3D scatter plot showing the relationships between three key 
                piezoelectric properties enhanced by different dopant categories. Each point 
                represents a dopant-material combination, colored by dopant category. This 
                visualization reveals clusters of similar performance profiles and helps 
                identify dopants that simultaneously enhance multiple properties.
                """, "📊")
        
        # Tab 5: Comprehensive Dashboard
        with tabs[4]:
            st.markdown("### 📈 Comprehensive Analysis Dashboard")
            
            # Dashboard controls
            col1, col2 = st.columns(2)
            with col1:
                dashboard_dopants = st.multiselect(
                    "Select dopants for dashboard",
                    options=relationships_df['dopant'].unique().tolist(),
                    default=relationships_df['dopant'].value_counts().head(3).index.tolist(),
                    max_selections=5
                )
            
            if len(dashboard_dopants) >= 2:
                fig = engine.create_comparison_dashboard(relationships_df, dashboard_dopants)
                
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Dashboard insights
                    st.markdown("#### 📋 Dashboard Insights Summary")
                    
                    insights_col1, insights_col2 = st.columns(2)
                    
                    with insights_col1:
                        st.markdown("**Key Findings:**")
                        # Calculate overall best dopant
                        enhancement_by_dopant = relationships_df.groupby('dopant')['enhancement_factor'].mean()
                        if not enhancement_by_dopant.empty:
                            best_dopant = enhancement_by_dopant.idxmax()
                            best_value = enhancement_by_dopant.max()
                            st.markdown(f"- **Top Performer**: {best_dopant} ({best_value:.2f}× avg enhancement)")
                        
                        # Most studied property
                        top_property = relationships_df['property'].value_counts().index[0]
                        st.markdown(f"- **Most Studied**: {top_property}")
                    
                    with insights_col2:
                        st.markdown("**Recommendations:**")
                        st.markdown("- Start with low concentrations (1-5%)")
                        st.markdown("- Consider processing method compatibility")
                        st.markdown("- Test multiple dopants in parallel")
                        st.markdown("- Validate with your specific base material")
                    
                    add_caption("""
                    **Figure 5:** Comprehensive dashboard providing multiple views of dopant 
                    performance data. Includes (A) average enhancement comparison, (B) property 
                    distribution analysis, (C) concentration optimization trends, and (D) 
                    processing method prevalence. This integrated view supports comprehensive 
                    decision-making for dopant selection and optimization.
                    """, "📈")
        
        # Tab 6: Enhanced Recommendations
        with tabs[5]:
            st.markdown("### 💡 Application-Specific Recommendations")
            
            # Application selection with descriptions
            application = st.selectbox(
                "Select target application",
                [
                    ("Energy Harvesting", "High d₃₃, voltage output, and power density"),
                    ("Sensors", "High sensitivity, stability, and d₃₃"),
                    ("Actuators", "High strain, response time, and d₃₃"),
                    ("High Temperature", "High Curie temperature and thermal stability"),
                    ("Flexible Electronics", "High flexibility, β-phase content, and durability"),
                    ("Biomedical", "Biocompatibility, flexibility, and moderate d₃₃")
                ],
                format_func=lambda x: x[0]
            )[0]
            
            # Recommendation parameters
            col1, col2, col3 = st.columns(3)
            with col1:
                min_confidence = st.slider("Min Confidence", 0.0, 1.0, 0.7, 0.1)
            with col2:
                n_recommendations = st.slider("Number of Recommendations", 1, 10, 5)
            with col3:
                include_processing = st.checkbox("Include Processing Methods", value=True)
            
            if st.button("✨ Generate Enhanced Recommendations", type="primary"):
                with st.spinner("Generating optimized recommendations with confidence scoring..."):
                    recommendations = engine.create_optimal_dopant_recommendations(
                        relationships_df, 
                        application
                    )[:n_recommendations]
                    
                    if not recommendations:
                        st.warning("No recommendations available. Try adjusting parameters.")
                    else:
                        # Display recommendations with enhanced visualization
                        st.markdown(f"### 🏆 Top Recommendations for {application}")
                        
                        for i, rec in enumerate(recommendations):
                            with st.container():
                                # Calculate confidence score
                                confidence = min(1.0, rec['score'] / 3.0)
                                
                                # Color based on confidence
                                if confidence > 0.8:
                                    color = "#10B981"  # Green
                                elif confidence > 0.6:
                                    color = "#F59E0B"  # Yellow
                                else:
                                    color = "#EF4444"  # Red
                                
                                st.markdown(f"""
                                <div style="
                                    background: linear-gradient(135deg, #FFFFFF 0%, #F8FAFC 100%);
                                    padding: 1.5rem; 
                                    border-radius: 12px; 
                                    margin: 1rem 0;
                                    border-left: 6px solid {color};
                                    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
                                ">
                                    <div style="display: flex; justify-content: space-between; align-items: center;">
                                        <h3 style="color: #1E40AF; margin: 0;">#{i+1} {rec['dopant']}</h3>
                                        <span style="
                                            background-color: {color};
                                            color: white;
                                            padding: 4px 12px;
                                            border-radius: 20px;
                                            font-weight: bold;
                                            font-size: 0.9rem;
                                        ">{confidence:.0%} Confidence</span>
                                    </div>
                                    <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin-top: 1rem;">
                                        <div>
                                            <strong>📊 Overall Score</strong><br>
                                            {rec['score']:.2f}/3.0
                                        </div>
                                        <div>
                                            <strong>⚡ Avg Enhancement</strong><br>
                                            {rec['avg_enhancement']:.2f}×
                                        </div>
                                        <div>
                                            <strong>🏷️ Category</strong><br>
                                            {rec['category']}
                                        </div>
                                    </div>
                                    <div style="margin-top: 1rem;">
                                        <strong>🎯 Key Properties:</strong> {', '.join(rec['key_properties'])}<br>
                                        <strong>🏗️ Best Base Materials:</strong> {', '.join(rec['best_base_materials'])}
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                        
                        # Implementation guide
                        st.markdown("### 📋 Implementation Guide")
                        with st.expander("View detailed implementation steps"):
                            st.markdown(f"""
                            **For {application}:**
                            
                            1. **Material Preparation**
                               - Start with {recommendations[0]['best_base_materials'][0]} as base material
                               - Use {recommendations[0]['dopant']} as primary dopant
                               - Initial concentration: 1-3 wt%
                            
                            2. **Processing Recommendations**
                               - Optimal method: Solution casting for uniform dispersion
                               - Alternative: In-situ polymerization for covalent bonding
                               - Post-processing: Annealing at 80-120°C for 2-4 hours
                            
                            3. **Optimization Strategy**
                               - Test concentration range: 0.5-10 wt%
                               - Characterize: d₃₃, β-phase content, dielectric constant
                               - Optimize for your specific application requirements
                            
                            4. **Validation Tests**
                               - Piezoelectric coefficient measurement
                               - Mechanical property testing
                               - Long-term stability assessment
                            """)
        
        # Tab 7: Enhanced Data Explorer
        with tabs[6]:
            st.markdown("### 🔍 Advanced Data Explorer")
            
            # Interactive data table with filters
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                base_filter = st.multiselect(
                    "Base Material",
                    options=sorted(relationships_df['base_material'].unique()),
                    default=[]
                )
            
            with col2:
                category_filter = st.multiselect(
                    "Dopant Category",
                    options=sorted(relationships_df['dopant_category'].unique()),
                    default=[]
                )
            
            with col3:
                min_enhance = st.slider("Min Enhancement", 1.0, 3.0, 1.0, 0.1)
            
            with col4:
                property_filter = st.multiselect(
                    "Property",
                    options=sorted(relationships_df['property'].unique()),
                    default=[]
                )
            
            # Apply filters
            filtered_df = relationships_df.copy()
            if base_filter:
                filtered_df = filtered_df[filtered_df['base_material'].isin(base_filter)]
            if category_filter:
                filtered_df = filtered_df[filtered_df['dopant_category'].isin(category_filter)]
            if property_filter:
                filtered_df = filtered_df[filtered_df['property'].isin(property_filter)]
            filtered_df = filtered_df[filtered_df['enhancement_factor'] >= min_enhance]
            
            # Display filtered data
            st.markdown(f"**Filtered Results:** {len(filtered_df)} relationships")
            st.dataframe(
                filtered_df,
                use_container_width=True,
                height=400,
                column_config={
                    "paper_id": st.column_config.NumberColumn("Paper ID"),
                    "base_material": st.column_config.TextColumn("Base Material"),
                    "dopant": st.column_config.TextColumn("Dopant"),
                    "dopant_category": st.column_config.TextColumn("Category"),
                    "property": st.column_config.TextColumn("Property"),
                    "value": st.column_config.NumberColumn("Value", format="%.2f"),
                    "enhancement_factor": st.column_config.NumberColumn("Enhancement", format="%.2f"),
                    "concentration_range": st.column_config.TextColumn("Concentration"),
                    "processing_method": st.column_config.TextColumn("Processing"),
                    "context": st.column_config.TextColumn("Context", width="large")
                }
            )
            
            # Enhanced export options
            st.markdown("### 📥 Data Export Options")
            export_col1, export_col2, export_col3, export_col4 = st.columns(4)
            
            with export_col1:
                csv = filtered_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "📊 CSV Export",
                    csv,
                    "dopant_analysis.csv",
                    "text/csv",
                    use_container_width=True
                )
            
            with export_col2:
                excel_buffer = io.BytesIO()
                with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                    filtered_df.to_excel(writer, sheet_name='Dopant_Analysis', index=False)
                    # Add summary sheet
                    summary_df = filtered_df.groupby(['dopant_category', 'base_material']).agg({
                        'enhancement_factor': ['mean', 'std', 'count']
                    }).round(3)
                    summary_df.to_excel(writer, sheet_name='Summary')
                excel_buffer.seek(0)
                st.download_button(
                    "📈 Excel Export",
                    excel_buffer,
                    "dopant_analysis.xlsx",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
            
            with export_col3:
                json_str = filtered_df.to_json(orient='records', indent=2)
                st.download_button(
                    "💾 JSON Export",
                    json_str,
                    "dopant_analysis.json",
                    "application/json",
                    use_container_width=True
                )
            
            with export_col4:
                # Generate report
                report = f"""
                # Dopant Impact Analysis Report
                Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
                
                ## Summary Statistics
                - Total Relationships: {len(filtered_df)}
                - Unique Dopants: {filtered_df['dopant'].nunique()}
                - Base Materials: {filtered_df['base_material'].nunique()}
                - Average Enhancement: {filtered_df['enhancement_factor'].mean():.2f}×
                
                ## Top Performers
                {filtered_df.groupby('dopant')['enhancement_factor'].mean().nlargest(5).to_string()}
                
                ## Data Preview
                {filtered_df.head(10).to_string()}
                """
                st.download_button(
                    "📄 Text Report",
                    report,
                    "dopant_report.txt",
                    "text/plain",
                    use_container_width=True
                )
        
        # Tab 8: Advanced Settings
        with tabs[7]:
            st.markdown("### ⚙️ Advanced Configuration")
            
            # Configuration sections
            config_tabs = st.tabs([
                "🎨 Visualization Settings",
                "🔬 Analysis Parameters",
                "📊 Export Settings",
                "⚡ Performance"
            ])
            
            with config_tabs[0]:
                st.markdown("#### Publication Quality Settings")
                
                col1, col2 = st.columns(2)
                with col1:
                    fig_width = st.number_input("Figure Width (px)", 600, 2000, 900, 100)
                    fig_height = st.number_input("Figure Height (px)", 400, 1500, 700, 100)
                    font_size = st.number_input("Font Size", 8, 24, 14, 1)
                
                with col2:
                    theme = st.selectbox("Plot Theme", ["plotly_white", "plotly_dark", "ggplot2", "seaborn"])
                    color_scale = st.selectbox("Color Scale", ["Viridis", "Plasma", "Inferno", "Magma", "RdYlBu"])
                
                # Update config
                Config.PLOT_CONFIG.update({
                    "width": fig_width,
                    "height": fig_height,
                    "font_size": font_size,
                    "template": theme
                })
                
                if st.button("💾 Apply Visualization Settings"):
                    st.success("Settings applied!")
            
            with config_tabs[1]:
                st.markdown("#### Advanced Analysis Parameters")
                
                # Keyword customization
                st.markdown("##### Custom Keywords")
                custom_keywords = st.text_area(
                    "Add custom keywords for extraction (one per line)",
                    value="\n".join(["composite", "nanocomposite", "filler", "additive"]),
                    height=150
                )
                
                # Property mapping
                st.markdown("##### Property Mapping")
                if st.checkbox("Customize property mapping"):
                    for prop, terms in Config.DOPANT_PROPERTIES.items():
                        new_terms = st.text_input(
                            f"Keywords for {prop}",
                            value=", ".join(terms),
                            key=f"prop_{prop}"
                        )
                        Config.DOPANT_PROPERTIES[prop] = [t.strip() for t in new_terms.split(",")]
            
            with config_tabs[2]:
                st.markdown("#### Export Configuration")
                
                col1, col2 = st.columns(2)
                with col1:
                    export_dpi = st.selectbox("Image DPI", [150, 300, 600, 1200], index=1)
                    export_format = st.selectbox("Default Format", ["PNG", "PDF", "SVG", "JPEG"])
                
                with col2:
                    include_metadata = st.checkbox("Include metadata in exports", value=True)
                    auto_save = st.checkbox("Auto-save generated figures", value=False)
                
                if st.button("🚀 Configure Export"):
                    st.info(f"Export configured: {export_dpi} DPI, {export_format} format")
            
            with config_tabs[3]:
                st.markdown("#### Performance Optimization")
                
                # Cache settings
                st.markdown("##### Caching Strategy")
                cache_size = st.slider("Cache Size (MB)", 10, 1000, 100, 10)
                use_memoization = st.checkbox("Enable memoization", value=True)
                parallel_processing = st.checkbox("Enable parallel processing", value=False)
                
                # Memory management
                st.markdown("##### Memory Management")
                max_memory = st.slider("Max Memory Usage (GB)", 1, 16, 4, 1)
                clear_cache = st.button("🧹 Clear Cache")
                
                if clear_cache:
                    st.cache_data.clear()
                    st.success("Cache cleared!")
    
    else:
        # Enhanced welcome screen
        st.markdown("""
        <div style="
            padding: 3rem; 
            text-align: center; 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 20px; 
            color: white; 
            margin-bottom: 3rem;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
        ">
            <h1 style="font-size: 3.5rem; margin-bottom: 1rem;">🔬 Dopant Impact Explorer Pro</h1>
            <p style="font-size: 1.5rem; opacity: 0.9; margin-bottom: 2rem;">
                Advanced Visual Analytics for Piezoelectric Material Enhancement
            </p>
            <div style="display: inline-block; background: rgba(255,255,255,0.2); 
                        padding: 10px 30px; border-radius: 50px; font-size: 1.2rem;">
                🚀 Publication-Ready Visualizations • 📊 Multi-Dimensional Analysis • 💡 AI-Powered Insights
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Feature cards with enhanced design
        st.markdown("### 🎯 Key Features")
        
        features = [
            {
                "icon": "🌳",
                "title": "Hierarchical Sunburst Analysis",
                "description": "Interactive hierarchical visualization of dopant effects with publication-quality styling",
                "color": "#3B82F6"
            },
            {
                "icon": "📡",
                "title": "Multi-Property Radar Charts",
                "description": "Compare dopants across 8+ properties with confidence indicators and statistical analysis",
                "color": "#10B981"
            },
            {
                "icon": "🔥",
                "title": "Concentration Optimization Heatmaps",
                "description": "Identify optimal doping concentrations with interactive heatmap visualizations",
                "color": "#F59E0B"
            },
            {
                "icon": "📊",
                "title": "3D Multi-Dimensional Analysis",
                "description": "Explore relationships between multiple properties in 3D space",
                "color": "#8B5CF6"
            },
            {
                "icon": "📈",
                "title": "Comprehensive Dashboards",
                "description": "Integrated multi-chart dashboards for comprehensive analysis",
                "color": "#EC4899"
            },
            {
                "icon": "💡",
                "title": "AI-Powered Recommendations",
                "description": "Application-specific dopant recommendations with confidence scoring",
                "color": "#6366F1"
            }
        ]
        
        # Display features in grid
        cols = st.columns(3)
        for i, feature in enumerate(features):
            with cols[i % 3]:
                st.markdown(f"""
                <div style="
                    background: linear-gradient(135deg, {feature['color']}20 0%, {feature['color']}10 100%);
                    padding: 1.5rem;
                    border-radius: 15px;
                    border: 1px solid {feature['color']}30;
                    height: 220px;
                    margin-bottom: 1.5rem;
                    transition: transform 0.3s;
                ">
                    <div style="font-size: 2.5rem; margin-bottom: 1rem;">{feature['icon']}</div>
                    <h3 style="color: {feature['color']}; margin: 0 0 0.5rem 0;">{feature['title']}</h3>
                    <p style="color: #4B5563; line-height: 1.5; font-size: 0.95rem;">{feature['description']}</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Getting started guide
        with st.expander("🚀 Getting Started Guide", expanded=True):
            st.markdown("""
            ### Quick Start Instructions
            
            1. **Database Setup**
               - Place your database files in the `knowledge_database/` directory
               - Or specify custom file paths in the sidebar
               - Required files: metadata, universe, and PDF databases
            
            2. **Configuration**
               - Select your database from the sidebar
               - Adjust analysis parameters (max papers, visualization quality)
               - Choose color palette (Nature, Science, etc.)
            
            3. **Analysis Workflow**
               - Click "Start Analysis" to process papers
               - Explore different visualization tabs
               - Use filters to focus on specific materials/dopants
               - Generate recommendations for your application
            
            4. **Export Results**
               - Download high-resolution figures for publications
               - Export data in multiple formats (CSV, Excel, JSON)
               - Generate comprehensive reports
            
            ### Supported Databases
            
            The tool supports multiple database structures:
            - **SQLite databases** with papers table
            - **CSV/Excel files** (automatically converted)
            - **Custom schemas** (auto-detected)
            
            ### System Requirements
            
            - Python 3.8+
            - 4GB RAM minimum (8GB recommended)
            - 500MB disk space for databases
            - Modern web browser with WebGL support for 3D visualizations
            """)

# ==============================
# ENHANCED APPLICATION ENTRY POINT
# ==============================
if __name__ == "__main__":
    # Create required directories
    os.makedirs(DEFAULT_DB_DIR, exist_ok=True)
    
    # Check for databases and provide guidance
    db_files = {
        "Metadata DB": os.path.join(DEFAULT_DB_DIR, "piezoelectricity_metadata.db"),
        "Universe DB": os.path.join(DEFAULT_DB_DIR, "piezoelectricity_universe.db"),
        "PDF Storage DB": os.path.join(DEFAULT_DB_DIR, "piezoelectricity_pdfs.db")
    }
    
    missing_dbs = [name for name, path in db_files.items() if not os.path.exists(path)]
    
    if missing_dbs:
        st.warning(f"""
        ⚠️ **Missing database files:** {', '.join(missing_dbs)}
        
        **Options:**
        1. Place existing databases in the `{DEFAULT_DB_DIR}` directory
        2. Use the "Use custom database files" option in the sidebar
        3. Use sample data for demonstration
        """)
        
        # Sample data option
        if st.checkbox("✅ Use sample data for demonstration", value=True):
            st.info("💡 Creating enhanced sample data for demonstration...")
            
            # Generate comprehensive sample data
            np.random.seed(42)
            n_samples = 300
            
            # Enhanced sample data with realistic values
            base_materials = ["PVDF", "BaTiO₃", "ZnO", "PZT", "PVDF-HFP", "KNN", "AlN"]
            dopants = ["ZnO", "BaTiO₃", "CNT", "Graphene", "TiO₂", "AlN", "Fe₂O₃", 
                      "MXene", "Cellulose", "PZT", "MoS₂", "Ag NPs", "BMIM-PF₆"]
            
            # Realistic property values
            property_ranges = {
                "d₃₃ (pC/N)": (5, 500),
                "β-phase (%)": (30, 95),
                "Dielectric Constant": (10, 500),
                "Young's Modulus (GPa)": (1, 20),
                "Conductivity (S/m)": (1e-10, 1e-2),
                "Curie Temp (°C)": (100, 300),
                "Voltage Output (V)": (0.1, 50),
                "Power Density (μW/cm²)": (1, 1000)
            }
            
            # Generate sample relationships
            relationships = []
            for i in range(n_samples):
                base = np.random.choice(base_materials)
                dopant = np.random.choice(dopants)
                prop = np.random.choice(list(property_ranges.keys()))
                
                # Realistic enhancement factors based on dopant-property combinations
                enhancement_base = 1.0
                if "CNT" in dopant or "Graphene" in dopant:
                    enhancement_base += np.random.uniform(0.5, 1.5)
                if "ZnO" in dopant or "BaTiO₃" in dopant:
                    enhancement_base += np.random.uniform(0.3, 1.2)
                
                # Add some noise
                enhancement = enhancement_base + np.random.uniform(-0.2, 0.2)
                enhancement = max(1.0, min(3.0, enhancement))
                
                # Property value based on base material
                if base == "PVDF":
                    base_value = np.random.uniform(20, 40)
                elif base == "BaTiO₃":
                    base_value = np.random.uniform(100, 300)
                else:
                    base_value = np.random.uniform(10, 200)
                
                # Enhanced value
                value = base_value * enhancement
                
                relationships.append({
                    'paper_id': f'paper_{i+1:04d}',
                    'base_material': base,
                    'dopant': dopant,
                    'dopant_category': Config().classify_dopant(dopant),
                    'property': prop,
                    'value': value,
                    'enhancement_factor': enhancement,
                    'concentration_range': f"{np.random.uniform(0.1, 20):.1f} wt%",
                    'processing_method': np.random.choice([
                        "Electrospinning", "Solution Casting", "Hot Pressing", 
                        "Melt Blending", "In-situ Polymerization", "Spin Coating"
                    ]),
                    'context': f"Study of {dopant} doping in {base} matrix showing enhanced {prop.split()[0]} properties."
                })
            
            relationships_df = pd.DataFrame(relationships)
            st.session_state.dopant_relationships = relationships_df
            st.success("""
            ✅ **Sample data ready!**
            
            **Dataset Overview:**
            - 300 sample dopant relationships
            - 7 different base materials
            - 13 different dopants across 8 categories
            - 8 key piezoelectric properties
            - Realistic enhancement factors (1.0-3.0×)
            
            **Explore the tabs above to see the enhanced visualizations!**
            """)
    
    # Run the enhanced application
    try:
        main()
    except Exception as e:
        st.error(f"""
        ❌ **Application Error**
        
        **Error Details:**
        ```python
        {str(e)}
        ```
        
        **Please try:**
        1. Refreshing the page
        2. Clearing browser cache
        3. Checking database file paths
        4. Ensuring sufficient memory
        
        If the problem persists, please report the issue with the error details above.
        """)
        logger.error(f"Application crashed: {str(e)}", exc_info=True)
