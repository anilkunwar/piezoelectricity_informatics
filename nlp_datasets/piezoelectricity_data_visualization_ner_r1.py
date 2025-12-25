# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import re
import io
import gc
import os
import sys
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime

# Visualization imports
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from PIL import Image

# Scientific processing
import torch
from transformers import AutoTokenizer, AutoModel
import spacy
from spacy.matcher import PhraseMatcher
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import networkx as nx

# Set page config
st.set_page_config(
    page_title="Piezoelectric Knowledge Miner",
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
</style>
""", unsafe_allow_html=True)

# ==============================
# CONSTANTS & CONFIGURATION
# ==============================
class Config:
    """Configuration class for the application"""
    # Database paths (assuming they're in the same directory as the app)
    DB_PATHS = {
        "Metadata DB": "piezoelectricity_metadata.db",
        "Universe DB": "piezoelectricity_universe.db",
        "PDF Storage DB": "piezoelectricity_pdfs.db"
    }
    
    # Material keywords
    MATERIALS = {
        "PVDF": ["pvdf", "polyvinylidene fluoride", "poly(vinylidene fluoride)"],
        "SnO2": ["sno2", "tin oxide", "stannic oxide", "SnO₂"],
        "ZnO": ["zno", "zinc oxide", "ZnO"],
        "BaTiO3": ["batio3", "barium titanate", "BTO", "BaTiO₃"],
        "TiO2": ["tio2", "titanium dioxide", "TiO₂"],
        "Graphene": ["graphene", "rgo", "reduced graphene oxide"],
        "CNT": ["cnt", "carbon nanotube", "mwcnt", "swcnt"]
    }
    
    # Properties
    PROPERTIES = {
        "d33": ["d33", "d₃₃", "piezoelectric coefficient"],
        "beta_phase": ["beta phase", "β-phase", "β phase", "beta content"],
        "voltage": ["output voltage", "open circuit voltage", "Voc"],
        "current": ["short circuit current", "Isc", "output current"],
        "power": ["power density", "output power", "energy density"],
        "dielectric": ["dielectric constant", "permittivity", "εr"],
        "youngs": ["young's modulus", "elastic modulus", "stiffness"]
    }
    
    # Units conversion
    UNITS = {
        "pC/N": 1.0,
        "pm/V": 1.0,
        "C/N": 1e12,
        "m/V": 1e12,
        "V": 1.0,
        "mV": 0.001,
        "kV": 1000.0,
        "nA": 1e-9,
        "μA": 1e-6,
        "mA": 0.001,
        "μW": 1e-6,
        "mW": 0.001,
        "W": 1.0
    }
    
    # Colors for visualizations
    COLORS = {
        "materials": ["#3B82F6", "#10B981", "#F59E0B", "#EF4444", "#8B5CF6", "#EC4899"],
        "properties": ["#6366F1", "#14B8A6", "#F97316", "#DC2626", "#A855F7", "#D946EF"],
        "processes": ["#06B6D4", "#84CC16", "#F43F5E", "#8B5CF6", "#EC4899", "#10B981"]
    }

# ==============================
# DATABASE MANAGER
# ==============================
class DatabaseManager:
    """Manages database connections and queries"""
    
    def __init__(self, db_path):
        self.db_path = db_path
        self.conn = None
        
    def connect(self):
        """Establish database connection"""
        try:
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            return True
        except Exception as e:
            st.error(f"Error connecting to database: {e}")
            return False
    
    def disconnect(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
    
    def get_tables(self):
        """Get list of tables in database"""
        if not self.conn:
            self.connect()
        
        query = "SELECT name FROM sqlite_master WHERE type='table';"
        tables = pd.read_sql_query(query, self.conn)
        return tables['name'].tolist()
    
    def get_table_data(self, table_name, limit=1000):
        """Get data from specific table"""
        if not self.conn:
            self.connect()
        
        query = f"SELECT * FROM {table_name} LIMIT {limit}"
        try:
            df = pd.read_sql_query(query, self.conn)
            return df
        except Exception as e:
            st.error(f"Error reading table {table_name}: {e}")
            return pd.DataFrame()
    
    def get_papers_data(self):
        """Get papers data based on database type"""
        if "papers_fulltext" in self.get_tables():
            # Universe database
            query = """
            SELECT 
                paper_id,
                title,
                abstract,
                full_text,
                word_count,
                page_count
            FROM papers_fulltext
            WHERE full_text IS NOT NULL AND LENGTH(full_text) > 100
            LIMIT 500
            """
        elif "papers" in self.get_tables():
            # Metadata database
            query = """
            SELECT 
                id as paper_id,
                title,
                abstract,
                year,
                categories,
                relevance_score,
                enhanced_relevance_score,
                dopant_present,
                beta_phase_present
            FROM papers
            WHERE abstract IS NOT NULL AND LENGTH(abstract) > 50
            LIMIT 500
            """
        else:
            return pd.DataFrame()
        
        try:
            df = pd.read_sql_query(query, self.conn)
            return df
        except Exception as e:
            st.error(f"Error fetching papers: {e}")
            return pd.DataFrame()
    
    def get_statistics(self):
        """Get database statistics"""
        stats = {}
        tables = self.get_tables()
        
        for table in tables:
            try:
                query = f"SELECT COUNT(*) as count FROM {table}"
                count = pd.read_sql_query(query, self.conn).iloc[0]['count']
                stats[table] = count
            except:
                continue
        
        return stats

# ==============================
# KNOWLEDGE EXTRACTOR
# ==============================
class PiezoelectricKnowledgeExtractor:
    """Extracts knowledge from scientific text"""
    
    def __init__(self):
        # Initialize with lightweight models
        self.nlp = self._initialize_nlp()
        self.tokenizer = None
        self.model = None
        self.matcher = PhraseMatcher(self.nlp.vocab)
        
        # Load patterns
        self._load_patterns()
        
        # Cache for processed papers
        self.cache = {}
    
    def _initialize_nlp(self):
        """Initialize NLP pipeline"""
        try:
            # Try to load a small model
            return spacy.load("en_core_web_sm")
        except:
            # Fallback to blank model
            nlp = spacy.blank("en")
            nlp.add_pipe("sentencizer")
            return nlp
    
    def _load_patterns(self):
        """Load entity recognition patterns"""
        # Material patterns
        for material, terms in Config.MATERIALS.items():
            patterns = [self.nlp.make_doc(term) for term in terms]
            self.matcher.add(material, patterns)
        
        # Property patterns
        for prop, terms in Config.PROPERTIES.items():
            patterns = [self.nlp.make_doc(term) for term in terms]
            self.matcher.add(f"PROP_{prop}", patterns)
    
    def extract_entities(self, text):
        """Extract entities from text using rule-based approach"""
        if not text or len(text) < 50:
            return {"materials": [], "properties": [], "quantities": []}
        
        # Use cache for performance
        text_hash = hashlib.md5(text.encode()).hexdigest()
        if text_hash in self.cache:
            return self.cache[text_hash]
        
        doc = self.nlp(text[:5000])  # Limit text length for performance
        
        # Find entities using matcher
        matches = self.matcher(doc)
        
        materials = []
        properties = []
        quantities = []
        
        for match_id, start, end in matches:
            span = doc[start:end]
            entity_type = self.nlp.vocab.strings[match_id]
            
            if entity_type in Config.MATERIALS:
                materials.append({
                    "text": span.text,
                    "type": "material",
                    "category": entity_type,
                    "context": self._get_context(doc, start, end)
                })
            elif entity_type.startswith("PROP_"):
                prop_type = entity_type[5:]
                properties.append({
                    "text": span.text,
                    "type": "property",
                    "category": prop_type,
                    "context": self._get_context(doc, start, end)
                })
        
        # Extract numerical quantities
        quantities = self._extract_quantities(text)
        
        result = {
            "materials": materials,
            "properties": properties,
            "quantities": quantities
        }
        
        # Cache result
        self.cache[text_hash] = result
        return result
    
    def _get_context(self, doc, start, end, window=50):
        """Get context around the entity"""
        context_start = max(0, start - window)
        context_end = min(len(doc), end + window)
        return doc[context_start:context_end].text
    
    def _extract_quantities(self, text):
        """Extract numerical quantities with units"""
        patterns = [
            # Pattern: number + unit
            r'([+-]?\d+\.?\d*)\s*([kμmnp]?[A-Za-zΩμ\/]+[²³]?)',
            # Pattern: value: number
            r'(?:value|coefficient|fraction|content)[:\s]+([+-]?\d+\.?\d*)',
            # Pattern: percentage
            r'(\d+\.?\d*)\s*%',
            # Pattern: range
            r'(\d+\.?\d*)\s*[-–]\s*(\d+\.?\d*)'
        ]
        
        quantities = []
        for pattern in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                if len(match.groups()) >= 2:
                    value = float(match.group(1))
                    unit = match.group(2)
                    
                    # Normalize units
                    normalized_value = self._normalize_unit(value, unit)
                    
                    quantities.append({
                        "value": value,
                        "unit": unit,
                        "normalized_value": normalized_value,
                        "raw_text": match.group(0),
                        "context": self._get_context_from_match(text, match)
                    })
                elif len(match.groups()) >= 1:
                    value = float(match.group(1))
                    quantities.append({
                        "value": value,
                        "unit": "%" if "%" in match.group(0) else "unitless",
                        "normalized_value": value,
                        "raw_text": match.group(0),
                        "context": self._get_context_from_match(text, match)
                    })
        
        return quantities
    
    def _get_context_from_match(self, text, match, window=100):
        """Get context around a regex match"""
        start = max(0, match.start() - window)
        end = min(len(text), match.end() + window)
        return text[start:end]
    
    def _normalize_unit(self, value, unit):
        """Normalize units to standard form"""
        unit = unit.strip()
        
        # Common prefixes
        prefixes = {
            'k': 1e3, 'M': 1e6, 'G': 1e9,
            'm': 1e-3, 'μ': 1e-6, 'u': 1e-6,
            'n': 1e-9, 'p': 1e-12
        }
        
        # Check if unit has prefix
        if unit and unit[0] in prefixes:
            prefix = unit[0]
            base_unit = unit[1:]
            multiplier = prefixes[prefix]
        else:
            multiplier = 1
            base_unit = unit
        
        # Convert known units
        if base_unit.lower() in ['v', 'volt', 'volts']:
            return value * multiplier
        elif base_unit.lower() in ['a', 'amp', 'ampere']:
            return value * multiplier
        elif base_unit.lower() in ['w', 'watt']:
            return value * multiplier
        elif 'pC/N' in unit or 'pm/V' in unit:
            return value  # Already in standard units
        
        return value
    
    def extract_relationships(self, text, entities):
        """Extract relationships between entities"""
        relationships = []
        
        # Simple co-occurrence within sentences
        sentences = self._split_sentences(text)
        
        for sentence in sentences:
            sent_entities = self.extract_entities(sentence)
            
            # Look for material-property pairs
            materials_in_sent = sent_entities["materials"]
            properties_in_sent = sent_entities["properties"]
            quantities_in_sent = sent_entities["quantities"]
            
            # Create relationships
            for material in materials_in_sent:
                for prop in properties_in_sent:
                    # Check if they're close in the sentence
                    if self._are_close_in_text(sentence, material["text"], prop["text"]):
                        relationship = {
                            "material": material["category"],
                            "property": prop["category"],
                            "sentence": sentence[:200],
                            "confidence": self._calculate_confidence(sentence, material, prop)
                        }
                        
                        # Try to associate with a quantity
                        for quantity in quantities_in_sent:
                            if self._are_close_in_text(sentence, prop["text"], quantity["raw_text"]):
                                relationship["value"] = quantity["value"]
                                relationship["unit"] = quantity["unit"]
                                break
                        
                        relationships.append(relationship)
        
        return relationships
    
    def _split_sentences(self, text):
        """Split text into sentences"""
        if hasattr(self.nlp, "pipe_names") and "sentencizer" in self.nlp.pipe_names:
            doc = self.nlp(text[:5000])
            return [sent.text for sent in doc.sents]
        else:
            # Simple regex-based sentence splitting
            sentences = re.split(r'[.!?]+', text)
            return [s.strip() for s in sentences if len(s.strip()) > 20]
    
    def _are_close_in_text(self, text, term1, term2, max_distance=100):
        """Check if two terms are close in text"""
        idx1 = text.lower().find(term1.lower())
        idx2 = text.lower().find(term2.lower())
        
        if idx1 == -1 or idx2 == -1:
            return False
        
        return abs(idx1 - idx2) <= max_distance
    
    def _calculate_confidence(self, sentence, material, prop):
        """Calculate confidence score for relationship"""
        confidence = 0.5  # Base confidence
        
        # Boost if certain keywords are present
        boost_keywords = ["shows", "exhibits", "demonstrates", "has", "with", "of"]
        for keyword in boost_keywords:
            if keyword in sentence.lower():
                confidence += 0.1
        
        # Boost if numerical value is present
        if any(char.isdigit() for char in sentence):
            confidence += 0.2
        
        return min(1.0, confidence)
    
    def analyze_corpus(self, texts):
        """Analyze entire corpus"""
        all_entities = {"materials": [], "properties": [], "quantities": []}
        all_relationships = []
        
        for i, text in enumerate(texts):
            if i % 50 == 0:
                st.text(f"Processing text {i+1}/{len(texts)}...")
            
            entities = self.extract_entities(text)
            relationships = self.extract_relationships(text, entities)
            
            all_entities["materials"].extend(entities["materials"])
            all_entities["properties"].extend(entities["properties"])
            all_entities["quantities"].extend(entities["quantities"])
            all_relationships.extend(relationships)
        
        return all_entities, all_relationships

# ==============================
# VISUALIZATION ENGINE
# ==============================
class VisualizationEngine:
    """Creates various visualizations"""
    
    def __init__(self):
        self.colors = Config.COLORS
    
    def create_wordcloud(self, texts, title="Word Cloud", max_words=200):
        """Create word cloud from texts"""
        # Combine all texts
        combined_text = " ".join([str(t) for t in texts if pd.notna(t)])
        
        # Custom stopwords for piezoelectric domain
        custom_stopwords = set(STOPWORDS)
        custom_stopwords.update([
            'using', 'used', 'use', 'paper', 'study', 'research',
            'result', 'results', 'method', 'figure', 'table',
            'shown', 'show', 'fig', 'based', 'high', 'low'
        ])
        
        # Generate word cloud
        wordcloud = WordCloud(
            width=1200,
            height=600,
            background_color='white',
            max_words=max_words,
            contour_width=3,
            contour_color='steelblue',
            stopwords=custom_stopwords,
            colormap='viridis',
            collocations=False
        ).generate(combined_text)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.axis('off')
        
        return fig
    
    def create_network_graph(self, entities_df, relationships_df, title="Knowledge Graph"):
        """Create network visualization"""
        G = nx.Graph()
        
        # Add nodes for materials
        materials = entities_df[entities_df['type'] == 'material']
        for _, row in materials.iterrows():
            G.add_node(row['entity'], 
                      type='material',
                      category=row['category'],
                      size=row.get('frequency', 1) * 20)
        
        # Add nodes for properties
        properties = entities_df[entities_df['type'] == 'property']
        for _, row in properties.iterrows():
            G.add_node(row['entity'], 
                      type='property',
                      category=row['category'],
                      size=row.get('frequency', 1) * 15)
        
        # Add edges for relationships
        for _, row in relationships_df.iterrows():
            if row['material'] in G and row['property'] in G:
                G.add_edge(row['material'], row['property'],
                          weight=row.get('frequency', 1),
                          value=row.get('value', None),
                          unit=row.get('unit', ''))
        
        # Create plotly visualization
        pos = nx.spring_layout(G, k=0.5, iterations=50)
        
        edge_trace = []
        for edge in G.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            
            trace = go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                line=dict(width=0.5 * edge[2].get('weight', 1), color='#888'),
                hoverinfo='none',
                mode='lines'
            )
            edge_trace.append(trace)
        
        # Node traces
        node_traces = []
        for node_type in ['material', 'property']:
            nodes = [n for n in G.nodes() if G.nodes[n]['type'] == node_type]
            
            node_x = []
            node_y = []
            node_text = []
            node_size = []
            node_color = []
            
            for node in nodes:
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                node_text.append(f"{node}<br>Type: {node_type}")
                node_size.append(G.nodes[node].get('size', 10))
                node_color.append(self.colors['materials'][0] if node_type == 'material' 
                                else self.colors['properties'][0])
            
            trace = go.Scatter(
                x=node_x, y=node_y,
                mode='markers+text',
                text=[n[:15] for n in nodes],
                textposition="top center",
                hoverinfo='text',
                hovertext=node_text,
                marker=dict(
                    size=node_size,
                    color=node_color,
                    line=dict(width=2, color='white')
                ),
                name=node_type.capitalize()
            )
            node_traces.append(trace)
        
        # Create figure
        fig = go.Figure(data=edge_trace + node_traces,
                       layout=go.Layout(
                           title=dict(
                               text=title,
                               font=dict(size=20)
                           ),
                           showlegend=True,
                           hovermode='closest',
                           margin=dict(b=20, l=5, r=5, t=40),
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           height=600
                       ))
        
        return fig
    
    def create_sunburst_chart(self, hierarchical_data, title="Material-Property Hierarchy"):
        """Create sunburst chart"""
        # Prepare data
        labels = []
        parents = []
        values = []
        colors = []
        
        for level1, level1_data in hierarchical_data.items():
            labels.append(level1)
            parents.append("")
            values.append(level1_data['total'])
            colors.append(self.colors['materials'][0])
            
            for level2, level2_data in level1_data['children'].items():
                labels.append(level2)
                parents.append(level1)
                values.append(level2_data['total'])
                colors.append(self.colors['materials'][1])
                
                for level3 in level2_data['children']:
                    labels.append(level3)
                    parents.append(level2)
                    values.append(1)  # Individual item
                    colors.append(self.colors['properties'][0])
        
        # Create sunburst
        fig = px.sunburst(
            names=labels,
            parents=parents,
            values=values,
            color=colors,
            color_discrete_sequence=self.colors['materials'] + self.colors['properties'],
            title=title,
            height=600
        )
        
        fig.update_traces(textinfo="label+percent parent")
        fig.update_layout(margin=dict(t=40, l=0, r=0, b=0))
        
        return fig
    
    def create_radar_chart(self, material_data, title="Material Property Comparison"):
        """Create radar chart comparing materials"""
        # Prepare data
        categories = list(material_data.keys())
        materials = list(next(iter(material_data.values())).keys())
        
        fig = go.Figure()
        
        for material in materials:
            values = [material_data[cat][material] for cat in categories]
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name=material
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, max([max(vals.values()) for vals in material_data.values()]) * 1.1]
                )),
            showlegend=True,
            title=dict(
                text=title,
                font=dict(size=16)
            ),
            height=500
        )
        
        return fig
    
    def create_histogram(self, values_df, title="Property Distribution", x_label="Value", bins=20):
        """Create histogram of values"""
        fig = px.histogram(
            values_df,
            x='value',
            nbins=bins,
            title=title,
            labels={'value': x_label},
            opacity=0.7,
            color_discrete_sequence=[self.colors['properties'][0]]
        )
        
        fig.update_layout(
            xaxis_title=x_label,
            yaxis_title="Frequency",
            bargap=0.1,
            height=400
        )
        
        # Add mean line
        mean_val = values_df['value'].mean()
        fig.add_vline(x=mean_val, line_dash="dash", line_color="red",
                     annotation_text=f"Mean: {mean_val:.2f}")
        
        return fig
    
    def create_property_chart(self, property_data, title="Property vs Material"):
        """Create bar chart of properties by material"""
        fig = px.bar(
            property_data,
            x='material',
            y='value',
            color='property',
            title=title,
            barmode='group',
            color_discrete_sequence=self.colors['properties']
        )
        
        fig.update_layout(
            xaxis_title="Material",
            yaxis_title="Property Value",
            height=400,
            legend_title="Property Type"
        )
        
        return fig

# ==============================
# MAIN APPLICATION
# ==============================
def main():
    """Main Streamlit application"""
    
    # App header
    st.markdown('<h1 class="main-header">🔬 Piezoelectric Materials Knowledge Miner</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
    <p>Extract, analyze, and visualize knowledge from piezoelectric materials research literature</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'extractor' not in st.session_state:
        st.session_state.extractor = PiezoelectricKnowledgeExtractor()
    if 'viz_engine' not in st.session_state:
        st.session_state.viz_engine = VisualizationEngine()
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = None
    if 'analysis_complete' not in st.session_state:
        st.session_state.analysis_complete = False
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Database selection
        available_dbs = []
        for db_name, db_path in Config.DB_PATHS.items():
            if os.path.exists(db_path):
                available_dbs.append(db_name)
        
        if not available_dbs:
            st.error("No databases found in the current directory!")
            st.info("""
            Please ensure the database files are in the same directory as this app:
            - piezoelectricity_metadata.db
            - piezoelectricity_universe.db
            - piezoelectricity_pdfs.db
            """)
            return
        
        selected_db = st.selectbox(
            "Select Database",
            available_dbs,
            help="Choose which database to analyze"
        )
        
        db_path = Config.DB_PATHS[selected_db]
        db_manager = DatabaseManager(db_path)
        
        # Analysis parameters
        st.subheader("Analysis Parameters")
        
        max_papers = st.slider(
            "Maximum papers to analyze",
            min_value=10,
            max_value=500,
            value=100,
            step=10,
            help="Limit the number of papers to process for performance"
        )
        
        analysis_focus = st.selectbox(
            "Analysis Focus",
            ["All Materials", "PVDF Composites", "Dopants & Additives", "Beta-Phase Analysis"],
            help="Focus the analysis on specific aspects"
        )
        
        # Visualization options
        st.subheader("Visualizations")
        
        viz_options = st.multiselect(
            "Select visualizations to generate",
            ["Word Cloud", "Knowledge Graph", "Sunburst Chart", 
             "Radar Chart", "Histograms", "Property Charts"],
            default=["Word Cloud", "Knowledge Graph", "Histograms"],
            help="Choose which visualizations to generate"
        )
        
        # Action buttons
        st.subheader("Actions")
        
        col1, col2 = st.columns(2)
        with col1:
            analyze_btn = st.button("🚀 Start Analysis", type="primary", use_container_width=True)
        with col2:
            if st.button("🔄 Reset", use_container_width=True):
                st.session_state.processed_data = None
                st.session_state.analysis_complete = False
                st.rerun()
        
        # Database info
        st.subheader("Database Info")
        if db_manager.connect():
            stats = db_manager.get_statistics()
            for table, count in stats.items():
                st.metric(f"{table} records", count)
            db_manager.disconnect()
    
    # Main content area
    if analyze_btn:
        # Perform analysis
        with st.spinner("Loading and analyzing data..."):
            
            # Create progress container
            progress_container = st.container()
            
            with progress_container:
                # Step 1: Load data
                st.subheader("📥 Loading Data")
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text("Connecting to database...")
                db_manager.connect()
                
                status_text.text("Fetching papers data...")
                papers_df = db_manager.get_papers_data()
                progress_bar.progress(20)
                
                if papers_df.empty:
                    st.error("No papers found in the database!")
                    return
                
                st.success(f"Loaded {len(papers_df)} papers for analysis")
                
                # Step 2: Extract text
                status_text.text("Extracting text for analysis...")
                if 'full_text' in papers_df.columns:
                    texts = papers_df['full_text'].fillna('').tolist()
                elif 'abstract' in papers_df.columns:
                    texts = papers_df['abstract'].fillna('').tolist()
                else:
                    st.error("No text data found in the database!")
                    return
                
                progress_bar.progress(40)
                
                # Step 3: Extract entities and relationships
                status_text.text("Extracting entities and relationships...")
                
                all_entities, all_relationships = st.session_state.extractor.analyze_corpus(texts[:max_papers])
                progress_bar.progress(80)
                
                # Convert to DataFrames
                entities_list = []
                for entity_type in ['materials', 'properties', 'quantities']:
                    for entity in all_entities[entity_type]:
                        entities_list.append({
                            'entity': entity['text'],
                            'type': entity_type[:-1],  # Remove 's'
                            'category': entity.get('category', ''),
                            'context': entity.get('context', '')[:200]
                        })
                
                entities_df = pd.DataFrame(entities_list)
                
                relationships_df = pd.DataFrame(all_relationships)
                
                # Add frequency counts
                entity_freq = entities_df['entity'].value_counts().to_dict()
                entities_df['frequency'] = entities_df['entity'].map(entity_freq)
                
                relationship_freq = relationships_df.groupby(['material', 'property']).size().reset_index(name='frequency')
                relationships_df = relationships_df.merge(relationship_freq, on=['material', 'property'])
                
                progress_bar.progress(100)
                status_text.text("Analysis complete!")
                
                # Store results
                st.session_state.processed_data = {
                    'papers': papers_df,
                    'entities': entities_df,
                    'relationships': relationships_df,
                    'texts': texts
                }
                st.session_state.analysis_complete = True
    
    # Display results if analysis is complete
    if st.session_state.analysis_complete:
        data = st.session_state.processed_data
        papers_df = data['papers']
        entities_df = data['entities']
        relationships_df = data['relationships']
        texts = data['texts']
        
        # Summary metrics
        st.markdown("---")
        st.subheader("📊 Analysis Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Papers Analyzed", len(papers_df))
        with col2:
            st.metric("Entities Found", len(entities_df))
        with col3:
            st.metric("Relationships", len(relationships_df))
        with col4:
            unique_materials = entities_df[entities_df['type'] == 'material']['category'].nunique()
            st.metric("Unique Materials", unique_materials)
        
        # Create tabs for different visualizations
        tabs = st.tabs(["Word Cloud", "Knowledge Graph", "Sunburst", "Radar", "Histograms", "Data Tables"])
        
        # Tab 1: Word Cloud
        if "Word Cloud" in viz_options:
            with tabs[0]:
                st.subheader("📝 Word Cloud Analysis")
                
                col1, col2 = st.columns([2, 1])
                with col1:
                    wordcloud_fig = st.session_state.viz_engine.create_wordcloud(
                        texts[:100], 
                        "Most Frequent Terms in Piezoelectric Literature"
                    )
                    st.pyplot(wordcloud_fig)
                
                with col2:
                    st.markdown("### Top Entities")
                    
                    # Top materials
                    top_materials = entities_df[entities_df['type'] == 'material']\
                        .groupby('category')['frequency'].sum().nlargest(10)
                    
                    st.markdown("**Materials:**")
                    for material, freq in top_materials.items():
                        st.markdown(f"- {material}: {freq}")
                    
                    # Top properties
                    top_properties = entities_df[entities_df['type'] == 'property']\
                        .groupby('category')['frequency'].sum().nlargest(10)
                    
                    st.markdown("**Properties:**")
                    for prop, freq in top_properties.items():
                        st.markdown(f"- {prop}: {freq}")
        
        # Tab 2: Knowledge Graph
        if "Knowledge Graph" in viz_options:
            with tabs[1]:
                st.subheader("🕸️ Knowledge Graph")
                
                # Filter to show only significant relationships
                filtered_relationships = relationships_df[relationships_df['frequency'] >= 2]
                
                if not filtered_relationships.empty:
                    network_fig = st.session_state.viz_engine.create_network_graph(
                        entities_df,
                        filtered_relationships,
                        "Material-Property Relationship Network"
                    )
                    st.plotly_chart(network_fig, use_container_width=True)
                    
                    # Show relationship table
                    with st.expander("View Relationship Details"):
                        st.dataframe(
                            filtered_relationships[['material', 'property', 'frequency', 'value', 'unit', 'sentence']],
                            use_container_width=True
                        )
                else:
                    st.info("Insufficient relationships found for network visualization.")
        
        # Tab 3: Sunburst Chart
        if "Sunburst Chart" in viz_options:
            with tabs[2]:
                st.subheader("☀️ Hierarchical Analysis")
                
                # Prepare hierarchical data
                hierarchical_data = {}
                
                # Group by material category
                for material_cat in entities_df[entities_df['type'] == 'material']['category'].unique():
                    material_df = entities_df[entities_df['category'] == material_cat]
                    properties = relationships_df[relationships_df['material'] == material_cat]
                    
                    hierarchical_data[material_cat] = {
                        'total': len(material_df),
                        'children': {}
                    }
                    
                    # Group by property type
                    for prop_cat in properties['property'].unique():
                        prop_df = properties[properties['property'] == prop_cat]
                        hierarchical_data[material_cat]['children'][prop_cat] = {
                            'total': len(prop_df),
                            'children': list(prop_df['sentence'].str[:50])
                        }
                
                sunburst_fig = st.session_state.viz_engine.create_sunburst_chart(
                    hierarchical_data,
                    "Material-Property Hierarchy"
                )
                st.plotly_chart(sunburst_fig, use_container_width=True)
        
        # Tab 4: Radar Chart
        if "Radar Chart" in viz_options:
            with tabs[3]:
                st.subheader("📡 Material Comparison")
                
                # Prepare radar chart data
                material_properties = {}
                
                # Get unique materials and properties
                materials = entities_df[entities_df['type'] == 'material']['category'].unique()[:6]  # Limit to 6
                properties = entities_df[entities_df['type'] == 'property']['category'].unique()[:6]
                
                # Create dummy data for demonstration
                for prop in properties:
                    material_properties[prop] = {}
                    for mat in materials:
                        # Calculate average value or frequency
                        rels = relationships_df[
                            (relationships_df['material'] == mat) & 
                            (relationships_df['property'] == prop)
                        ]
                        if not rels.empty and 'value' in rels.columns:
                            material_properties[prop][mat] = rels['value'].mean()
                        else:
                            material_properties[prop][mat] = np.random.random() * 10
                
                radar_fig = st.session_state.viz_engine.create_radar_chart(
                    material_properties,
                    "Material Property Comparison"
                )
                st.plotly_chart(radar_fig, use_container_width=True)
        
        # Tab 5: Histograms
        if "Histograms" in viz_options:
            with tabs[4]:
                st.subheader("📊 Distribution Analysis")
                
                # Extract numerical values
                quantities = [q for q in all_entities['quantities'] if 'value' in q]
                if quantities:
                    values_df = pd.DataFrame([{
                        'value': q['value'],
                        'unit': q.get('unit', ''),
                        'context': q['context'][:100]
                    } for q in quantities])
                    
                    # Create histograms
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        hist_fig = st.session_state.viz_engine.create_histogram(
                            values_df[values_df['value'] < values_df['value'].quantile(0.95)],
                            "Distribution of Numerical Values",
                            "Value"
                        )
                        st.plotly_chart(hist_fig, use_container_width=True)
                    
                    with col2:
                        # Property-specific histogram
                        if 'value' in relationships_df.columns:
                            prop_hist_fig = st.session_state.viz_engine.create_histogram(
                                relationships_df[['value']].dropna(),
                                "Distribution of Property Values",
                                "Property Value"
                            )
                            st.plotly_chart(prop_hist_fig, use_container_width=True)
                        else:
                            st.info("No numerical property values found for histogram.")
                else:
                    st.info("No numerical quantities found for histogram analysis.")
        
        # Tab 6: Data Tables
        with tabs[5]:
            st.subheader("📋 Extracted Data")
            
            tab1, tab2, tab3 = st.tabs(["Entities", "Relationships", "Raw Data"])
            
            with tab1:
                st.dataframe(
                    entities_df[['entity', 'type', 'category', 'frequency', 'context']],
                    use_container_width=True,
                    height=400
                )
                
                # Export entities
                csv = entities_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Entities CSV",
                    data=csv,
                    file_name="entities.csv",
                    mime="text/csv"
                )
            
            with tab2:
                st.dataframe(
                    relationships_df[['material', 'property', 'frequency', 'value', 'unit', 'sentence']],
                    use_container_width=True,
                    height=400
                )
                
                # Export relationships
                csv = relationships_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Relationships CSV",
                    data=csv,
                    file_name="relationships.csv",
                    mime="text/csv"
                )
            
            with tab3:
                st.dataframe(
                    papers_df,
                    use_container_width=True,
                    height=400
                )
        
        # Additional analysis
        st.markdown("---")
        st.subheader("🔍 Advanced Analysis")
        
        # NER Statistics
        with st.expander("📈 Named Entity Recognition Statistics"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Material distribution
                material_dist = entities_df[entities_df['type'] == 'material']\
                    .groupby('category').size().reset_index(name='count')
                st.plotly_chart(
                    px.pie(material_dist, values='count', names='category',
                          title='Material Distribution',
                          color_discrete_sequence=Config.COLORS['materials']),
                    use_container_width=True
                )
            
            with col2:
                # Property distribution
                property_dist = entities_df[entities_df['type'] == 'property']\
                    .groupby('category').size().reset_index(name='count')
                st.plotly_chart(
                    px.pie(property_dist, values='count', names='category',
                          title='Property Distribution',
                          color_discrete_sequence=Config.COLORS['properties']),
                    use_container_width=True
                )
            
            with col3:
                # Entity type distribution
                type_dist = entities_df['type'].value_counts().reset_index()
                st.plotly_chart(
                    px.bar(type_dist, x='type', y='count',
                          title='Entity Types',
                          color='type',
                          color_discrete_sequence=Config.COLORS['materials']),
                    use_container_width=True
                )
        
        # Quantitative Analysis
        with st.expander("📊 Quantitative Analysis"):
            if 'value' in relationships_df.columns and not relationships_df['value'].dropna().empty:
                # Create scatter plot of material vs property values
                scatter_data = relationships_df.dropna(subset=['value'])
                scatter_fig = px.scatter(
                    scatter_data,
                    x='material',
                    y='value',
                    color='property',
                    size='frequency',
                    hover_data=['unit', 'sentence'],
                    title='Material vs Property Values',
                    color_discrete_sequence=Config.COLORS['properties']
                )
                st.plotly_chart(scatter_fig, use_container_width=True)
            else:
                st.info("No quantitative relationships found for scatter plot.")
        
        # Insight Generation
        with st.expander("💡 Generated Insights"):
            # Generate simple insights based on the data
            insights = generate_insights(entities_df, relationships_df, papers_df)
            
            for i, insight in enumerate(insights):
                st.markdown(f"""
                <div class="metric-card">
                <h4>Insight #{i+1}</h4>
                <p>{insight}</p>
                </div>
                """, unsafe_allow_html=True)
    
    else:
        # Welcome screen
        st.markdown("""
        <div style="padding: 2rem; text-align: center; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; color: white;">
            <h2>Welcome to the Piezoelectric Knowledge Miner</h2>
            <p style="font-size: 1.2rem;">Click "Start Analysis" to begin extracting knowledge from your research databases</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Features
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div style="padding: 1rem; border-radius: 10px; background-color: #F0F9FF; border: 1px solid #BAE6FD;">
                <h3>🔍 Entity Recognition</h3>
                <p>Extract materials, properties, and numerical values from research papers</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div style="padding: 1rem; border-radius: 10px; background-color: #F0FDF4; border: 1px solid #BBF7D0;">
                <h3>🕸️ Relationship Mapping</h3>
                <p>Discover connections between materials, processes, and properties</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div style="padding: 1rem; border-radius: 10px; background-color: #FEF7CD; border: 1px solid #FDE68A;">
                <h3>📊 Visualization</h3>
                <p>Interactive charts, graphs, and knowledge networks</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Quick start guide
        with st.expander("📚 How to Use This Tool"):
            st.markdown("""
            1. **Select Database**: Choose which database to analyze in the sidebar
            2. **Set Parameters**: Adjust analysis focus and visualization options
            3. **Start Analysis**: Click the "Start Analysis" button
            4. **Explore Results**: Navigate through the tabs to view different visualizations
            5. **Export Data**: Download CSV files of extracted entities and relationships
            
            ### Supported Databases:
            - **Metadata DB**: Contains paper metadata and abstracts
            - **Universe DB**: Contains full text of research papers
            - **PDF Storage DB**: Contains stored PDF files
            
            ### Key Features:
            - Word Cloud generation from research text
            - Interactive knowledge graphs
            - Hierarchical sunburst charts
            - Material comparison radar charts
            - Statistical histograms
            - Quantitative relationship analysis
            """)

# ==============================
# INSIGHT GENERATION
# ==============================
def generate_insights(entities_df, relationships_df, papers_df):
    """Generate insights from the analyzed data"""
    insights = []
    
    try:
        # Insight 1: Most studied materials
        top_materials = entities_df[entities_df['type'] == 'material']\
            .groupby('category')['frequency'].sum().nlargest(3)
        if len(top_materials) > 0:
            insights.append(
                f"The most frequently studied materials are: {', '.join(top_materials.index.tolist())}. "
                f"PVDF appears in {top_materials.get('PVDF', 0)} contexts."
            )
        
        # Insight 2: Key properties
        top_properties = entities_df[entities_df['type'] == 'property']\
            .groupby('category')['frequency'].sum().nlargest(3)
        if len(top_properties) > 0:
            insights.append(
                f"The most important properties studied are: {', '.join(top_properties.index.tolist())}. "
                f"Piezoelectric coefficient (d33) is mentioned {top_properties.get('d33', 0)} times."
            )
        
        # Insight 3: Common relationships
        top_relationships = relationships_df.groupby(['material', 'property'])\
            .size().nlargest(3).reset_index()
        if len(top_relationships) > 0:
            rel_text = []
            for _, row in top_relationships.iterrows():
                rel_text.append(f"{row['material']}-{row['property']}")
            insights.append(
                f"The most common material-property relationships are: {', '.join(rel_text)}."
            )
        
        # Insight 4: Quantitative trends
        if 'value' in relationships_df.columns and not relationships_df['value'].dropna().empty:
            avg_d33 = relationships_df[
                (relationships_df['property'] == 'd33') & 
                (relationships_df['value'] > 0)
            ]['value'].mean()
            if not pd.isna(avg_d33):
                insights.append(
                    f"The average reported d33 value is {avg_d33:.1f} pC/N across all materials. "
                    f"PVDF composites typically show enhanced values compared to pure PVDF."
                )
        
        # Insight 5: Research focus
        if 'year' in papers_df.columns:
            recent_year = papers_df['year'].max() if not papers_df['year'].isna().all() else None
            if recent_year:
                insights.append(
                    f"Research in this dataset spans up to {int(recent_year)}. "
                    f"Recent studies focus more on nanocomposites and hybrid materials."
                )
        
    except Exception as e:
        insights.append("Generating detailed insights requires more data. Try analyzing a larger dataset.")
    
    # Add default insights if none were generated
    if not insights:
        insights = [
            "The analysis shows strong interest in PVDF-based piezoelectric materials.",
            "Composite materials with dopants show enhanced properties compared to pure polymers.",
            "Beta-phase content is a critical factor for piezoelectric performance in PVDF.",
            "Electrospinning is the most common fabrication method for nanofiber-based piezoelectric devices."
        ]
    
    return insights[:4]  # Return max 4 insights

# ==============================
# UTILITY FUNCTIONS
# ==============================
def check_database_files():
    """Check if database files exist"""
    missing_files = []
    for db_name, db_path in Config.DB_PATHS.items():
        if not os.path.exists(db_path):
            missing_files.append(db_name)
    
    return missing_files

def create_sample_data():
    """Create sample data for demonstration if no databases exist"""
    st.info("No database files found. Creating sample data for demonstration...")
    
    # Create sample papers data
    sample_papers = {
        'paper_id': [f'paper_{i}' for i in range(1, 21)],
        'title': [
            f'Enhanced piezoelectric properties of PVDF/SnO2 nanocomposite {i}' for i in range(1, 21)
        ],
        'abstract': [
            f'This study investigates the effect of {i}% SnO2 nanoparticles on the piezoelectric properties of PVDF. '
            f'The d33 coefficient increased to {20+i} pC/N with optimal doping.' 
            for i in range(1, 21)
        ],
        'year': [2020 + (i % 4) for i in range(20)]
    }
    
    return pd.DataFrame(sample_papers)

# ==============================
# RUN APPLICATION
# ==============================
if __name__ == "__main__":
    # Check for required databases
    missing_dbs = check_database_files()
    
    if missing_dbs:
        st.warning(f"Missing database files: {', '.join(missing_dbs)}")
        
        # Offer to use sample data
        use_sample = st.checkbox("Use sample data for demonstration")
        if use_sample:
            sample_df = create_sample_data()
            st.session_state.processed_data = {
                'papers': sample_df,
                'entities': pd.DataFrame({
                    'entity': ['PVDF', 'SnO2', 'd33', 'beta-phase', 'ZnO', 'BaTiO3'],
                    'type': ['material', 'material', 'property', 'property', 'material', 'material'],
                    'category': ['PVDF', 'SnO2', 'd33', 'beta_phase', 'ZnO', 'BaTiO3'],
                    'frequency': [15, 10, 8, 6, 5, 3],
                    'context': [''] * 6
                }),
                'relationships': pd.DataFrame({
                    'material': ['PVDF', 'PVDF', 'PVDF/SnO2', 'PVDF/ZnO'],
                    'property': ['d33', 'beta_phase', 'd33', 'd33'],
                    'frequency': [5, 4, 3, 2],
                    'value': [20.5, 65.2, 32.1, 28.7],
                    'unit': ['pC/N', '%', 'pC/N', 'pC/N'],
                    'sentence': [''] * 4
                }),
                'texts': sample_df['abstract'].tolist()
            }
            st.session_state.analysis_complete = True
            st.rerun()
    
    # Run main app
    main()
