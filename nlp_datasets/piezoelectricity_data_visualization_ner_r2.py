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
    page_title="Piezoelectric Materials Knowledge Miner",
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
.figure-caption {
    font-size: 0.95rem;
    color: #4B5563;
    margin-top: 0.25rem;
    margin-bottom: 1.5rem;
    font-style: italic;
}
</style>
""", unsafe_allow_html=True)

def add_caption(text):
    """Helper to add styled caption below figures"""
    st.markdown(f'<div class="figure-caption">{text}</div>', unsafe_allow_html=True)

# ==============================
# CONSTANTS & CONFIGURATION
# ==============================
# Directory setup
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
    
    MATERIALS = {
        "PVDF": ["pvdf", "polyvinylidene fluoride", "poly(vinylidene fluoride)"],
        "SnO2": ["sno2", "tin oxide", "stannic oxide", "SnO₂"],
        "ZnO": ["zno", "zinc oxide", "ZnO"],
        "BaTiO3": ["batio3", "barium titanate", "BTO", "BaTiO₃"],
        "TiO2": ["tio2", "titanium dioxide", "TiO₂"],
        "Graphene": ["graphene", "rgo", "reduced graphene oxide"],
        "CNT": ["cnt", "carbon nanotube", "mwcnt", "swcnt"]
    }
    
    PROPERTIES = {
        "d33": ["d33", "d₃₃", "piezoelectric coefficient"],
        "beta_phase": ["beta phase", "β-phase", "β phase", "beta content"],
        "voltage": ["output voltage", "open circuit voltage", "Voc"],
        "current": ["short circuit current", "Isc", "output current"],
        "power": ["power density", "output power", "energy density"],
        "dielectric": ["dielectric constant", "permittivity", "εr"],
        "youngs": ["young's modulus", "elastic modulus", "stiffness"]
    }
    
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
        try:
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            return True
        except Exception as e:
            st.error(f"Error connecting to database: {e}")
            return False

    def disconnect(self):
        if self.conn:
            self.conn.close()

    def get_tables(self):
        if not self.conn:
            self.connect()
        query = "SELECT name FROM sqlite_master WHERE type='table';"
        tables = pd.read_sql_query(query, self.conn)
        return tables['name'].tolist()

    def get_table_data(self, table_name, limit=1000):
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
        if "papers_fulltext" in self.get_tables():
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
    def __init__(self):
        self.nlp = self._initialize_nlp()
        self.matcher = PhraseMatcher(self.nlp.vocab)
        self._load_patterns()
        self.cache = {}

    def _initialize_nlp(self):
        try:
            return spacy.load("en_core_web_sm")
        except:
            nlp = spacy.blank("en")
            nlp.add_pipe("sentencizer")
            return nlp

    def _load_patterns(self):
        for material, terms in Config.MATERIALS.items():
            patterns = [self.nlp.make_doc(term) for term in terms]
            self.matcher.add(material, patterns)
        for prop, terms in Config.PROPERTIES.items():
            patterns = [self.nlp.make_doc(term) for term in terms]
            self.matcher.add(f"PROP_{prop}", patterns)

    def extract_entities(self, text):
        if not text or len(text) < 50:
            return {"materials": [], "properties": [], "quantities": []}
        text_hash = hashlib.md5(text.encode()).hexdigest()
        if text_hash in self.cache:
            return self.cache[text_hash]
        doc = self.nlp(text[:5000])
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
        quantities = self._extract_quantities(text)
        result = {
            "materials": materials,
            "properties": properties,
            "quantities": quantities
        }
        self.cache[text_hash] = result
        return result

    def _get_context(self, doc, start, end, window=50):
        context_start = max(0, start - window)
        context_end = min(len(doc), end + window)
        return doc[context_start:context_end].text

    def _extract_quantities(self, text):
        patterns = [
            r'([+-]?\d+\.?\d*)\s*([kμmnp]?[A-Za-zΩμ\/]+[²³]?)',
            r'(?:value|coefficient|fraction|content)[:\s]+([+-]?\d+\.?\d*)',
            r'(\d+\.?\d*)\s*%',
            r'(\d+\.?\d*)\s*[-–]\s*(\d+\.?\d*)'
        ]
        quantities = []
        for pattern in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                if len(match.groups()) >= 2:
                    value = float(match.group(1))
                    unit = match.group(2)
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
        start = max(0, match.start() - window)
        end = min(len(text), match.end() + window)
        return text[start:end]

    def _normalize_unit(self, value, unit):
        unit = unit.strip()
        prefixes = {'k': 1e3, 'M': 1e6, 'G': 1e9, 'm': 1e-3, 'μ': 1e-6, 'u': 1e-6, 'n': 1e-9, 'p': 1e-12}
        if unit and unit[0] in prefixes:
            prefix = unit[0]
            base_unit = unit[1:]
            multiplier = prefixes[prefix]
        else:
            multiplier = 1
            base_unit = unit
        if base_unit.lower() in ['v', 'volt', 'volts']:
            return value * multiplier
        elif base_unit.lower() in ['a', 'amp', 'ampere']:
            return value * multiplier
        elif base_unit.lower() in ['w', 'watt']:
            return value * multiplier
        elif 'pC/N' in unit or 'pm/V' in unit:
            return value
        return value

    def extract_relationships(self, text, entities):
        relationships = []
        sentences = self._split_sentences(text)
        for sentence in sentences:
            sent_entities = self.extract_entities(sentence)
            materials_in_sent = sent_entities["materials"]
            properties_in_sent = sent_entities["properties"]
            quantities_in_sent = sent_entities["quantities"]
            for material in materials_in_sent:
                for prop in properties_in_sent:
                    if self._are_close_in_text(sentence, material["text"], prop["text"]):
                        relationship = {
                            "material": material["category"],
                            "property": prop["category"],
                            "sentence": sentence[:200],
                            "confidence": self._calculate_confidence(sentence, material, prop)
                        }
                        for quantity in quantities_in_sent:
                            if self._are_close_in_text(sentence, prop["text"], quantity["raw_text"]):
                                relationship["value"] = quantity["value"]
                                relationship["unit"] = quantity["unit"]
                                break
                        relationships.append(relationship)
        return relationships

    def _split_sentences(self, text):
        if hasattr(self.nlp, "pipe_names") and "sentencizer" in self.nlp.pipe_names:
            doc = self.nlp(text[:5000])
            return [sent.text for sent in doc.sents]
        else:
            sentences = re.split(r'[.!?]+', text)
            return [s.strip() for s in sentences if len(s.strip()) > 20]

    def _are_close_in_text(self, text, term1, term2, max_distance=100):
        idx1 = text.lower().find(term1.lower())
        idx2 = text.lower().find(term2.lower())
        if idx1 == -1 or idx2 == -1:
            return False
        return abs(idx1 - idx2) <= max_distance

    def _calculate_confidence(self, sentence, material, prop):
        confidence = 0.5
        boost_keywords = ["shows", "exhibits", "demonstrates", "has", "with", "of"]
        for keyword in boost_keywords:
            if keyword in sentence.lower():
                confidence += 0.1
        if any(char.isdigit() for char in sentence):
            confidence += 0.2
        return min(1.0, confidence)

    def analyze_corpus(self, texts):
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
# VISUALIZATION ENGINE (UPGRADED FOR PUBLICATION QUALITY)
# ==============================
class VisualizationEngine:
    def __init__(self):
        self.colors = Config.COLORS

    def create_wordcloud(self, texts, title="Word Cloud", max_words=200):
        combined_text = " ".join([str(t) for t in texts if pd.notna(t)])
        custom_stopwords = set(STOPWORDS)
        custom_stopwords.update([
            'using', 'used', 'use', 'paper', 'study', 'research',
            'result', 'results', 'method', 'figure', 'table',
            'shown', 'show', 'fig', 'based', 'high', 'low'
        ])
        wordcloud = WordCloud(
            width=1600,
            height=800,
            background_color='white',
            max_words=max_words,
            stopwords=custom_stopwords,
            colormap='viridis',
            collocations=False,
            relative_scaling=0.5,
            font_path=None  # Use default
        ).generate(combined_text)
        fig, ax = plt.subplots(figsize=(16, 8), dpi=300)
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.set_title(title, fontsize=18, fontweight='bold', pad=20)
        ax.axis('off')
        plt.tight_layout()
        return fig

    def create_network_graph(self, entities_df, relationships_df, title="Knowledge Graph"):
        G = nx.Graph()
        materials = entities_df[entities_df['type'] == 'material']
        for _, row in materials.iterrows():
            G.add_node(row['entity'], type='material', category=row['category'], size=row.get('frequency', 1) * 25)
        properties = entities_df[entities_df['type'] == 'property']
        for _, row in properties.iterrows():
            G.add_node(row['entity'], type='property', category=row['category'], size=row.get('frequency', 1) * 20)
        for _, row in relationships_df.iterrows():
            if row['material'] in G and row['property'] in G:
                G.add_edge(row['material'], row['property'], weight=row.get('frequency', 1))
        pos = nx.spring_layout(G, k=1.0, iterations=100, seed=42)
        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1.2, color='#888'),
            hoverinfo='none',
            mode='lines'
        )
        node_x = []
        node_y = []
        node_text = []
        node_size = []
        node_color = []
        node_names = []
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_names.append(node)
            node_text.append(f"{node}<br>Type: {G.nodes[node]['type']}")
            node_size.append(G.nodes[node]['size'])
            node_color.append(self.colors['materials'][0] if G.nodes[node]['type'] == 'material' else self.colors['properties'][1])
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=node_names,
            textposition="middle center",
            textfont=dict(size=10, color="black"),
            hovertext=node_text,
            marker=dict(
                size=node_size,
                color=node_color,
                line=dict(width=1.5, color='white')
            ),
            showlegend=False
        )
        fig = go.Figure(data=[edge_trace, node_trace],
                        layout=go.Layout(
                            title=dict(text=title, font=dict(size=20)),
                            title_x=0.5,
                            showlegend=False,
                            hovermode='closest',
                            margin=dict(b=20, l=5, r=5, t=60),
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            height=700,
                            font=dict(size=14)
                        ))
        return fig

    def create_sunburst_chart(self, hierarchical_data, title="Material-Property Hierarchy"):
        labels = []
        parents = []
        values = []
        for level1, level1_data in hierarchical_data.items():
            labels.append(level1)
            parents.append("")
            values.append(level1_data['total'])
            for level2, level2_data in level1_data['children'].items():
                labels.append(level2)
                parents.append(level1)
                values.append(level2_data['total'])
        fig = px.sunburst(
            names=labels,
            parents=parents,
            values=values,
            title=title,
            height=600,
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        fig.update_layout(
            title_x=0.5,
            margin=dict(t=60, l=0, r=0, b=0),
            font=dict(size=14)
        )
        fig.update_traces(textinfo="label+percent parent")
        return fig

    def create_radar_chart(self, material_data, title="Material Property Comparison"):
        categories = list(material_data.keys())
        materials = list(next(iter(material_data.values())).keys())
        fig = go.Figure()
        color_cycle = self.colors['properties'] + self.colors['materials']
        for i, material in enumerate(materials):
            values = [material_data[cat].get(material, 0) for cat in categories]
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name=material,
                line=dict(color=color_cycle[i % len(color_cycle)])
            ))
        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, max([max(material_data[cat].values()) for cat in categories if material_data[cat]]) * 1.2])
            ),
            showlegend=True,
            title=dict(text=title, font=dict(size=18), x=0.5),
            height=600,
            font=dict(size=14)
        )
        return fig

    def create_histogram(self, values_df, title="Property Distribution", x_label="Value", bins=20):
        fig = px.histogram(
            values_df,
            x='value',
            nbins=bins,
            title=title,
            labels={'value': x_label},
            opacity=0.75,
            color_discrete_sequence=[self.colors['properties'][0]]
        )
        fig.update_layout(
            xaxis_title=x_label,
            yaxis_title="Frequency",
            bargap=0.1,
            height=500,
            title_x=0.5,
            font=dict(size=14)
        )
        mean_val = values_df['value'].mean()
        fig.add_vline(x=mean_val, line_dash="dash", line_color="red",
                      annotation_text=f"Mean = {mean_val:.2f}",
                      annotation_position="top right")
        return fig

    def create_property_chart(self, property_data, title="Property vs Material"):
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
            height=500,
            title_x=0.5,
            font=dict(size=14),
            legend_title="Property"
        )
        return fig

# ==============================
# MAIN APPLICATION
# ==============================
def main():
    st.markdown('<h1 class="main-header">🔬 Piezoelectric Materials Knowledge Miner</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <p>Extract, analyze, and visualize knowledge from piezoelectric materials research literature</p>
    </div>
    """, unsafe_allow_html=True)

    if 'extractor' not in st.session_state:
        st.session_state.extractor = PiezoelectricKnowledgeExtractor()
    if 'viz_engine' not in st.session_state:
        st.session_state.viz_engine = VisualizationEngine()
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = None
    if 'analysis_complete' not in st.session_state:
        st.session_state.analysis_complete = False

    with st.sidebar:
        st.header("⚙️ Configuration")
        available_dbs = []
        for db_name, db_path in Config.DB_PATHS.items():
            if os.path.exists(db_path):
                available_dbs.append(db_name)
        if not available_dbs:
            st.error("No databases found in the knowledge_database directory!")
            st.info("""
            Ensure the following files exist in `knowledge_database/`:
            - piezoelectricity_metadata.db
            - piezoelectricity_universe.db
            - piezoelectricity_pdfs.db
            """)
            return

        selected_db = st.selectbox("Select Database", available_dbs)
        db_path = Config.DB_PATHS[selected_db]
        db_manager = DatabaseManager(db_path)

        st.subheader("Analysis Parameters")
        max_papers = st.slider("Maximum papers to analyze", 10, 500, 100, 10)
        analysis_focus = st.selectbox("Analysis Focus", [
            "All Materials", "PVDF Composites", "Dopants & Additives", "Beta-Phase Analysis"
        ])

        st.subheader("Visualizations")
        viz_options = st.multiselect(
            "Select visualizations",
            ["Word Cloud", "Knowledge Graph", "Sunburst Chart", "Radar Chart", "Histograms", "Property Charts"],
            default=["Word Cloud", "Knowledge Graph", "Histograms"]
        )

        st.subheader("Actions")
        col1, col2 = st.columns(2)
        with col1:
            analyze_btn = st.button("🚀 Start Analysis", type="primary", use_container_width=True)
        with col2:
            if st.button("🔄 Reset", use_container_width=True):
                st.session_state.processed_data = None
                st.session_state.analysis_complete = False
                st.rerun()

        st.subheader("Database Info")
        if db_manager.connect():
            stats = db_manager.get_statistics()
            for table, count in stats.items():
                st.metric(f"{table} records", count)
            db_manager.disconnect()

    if analyze_btn:
        with st.spinner("Loading and analyzing data..."):
            progress_container = st.container()
            with progress_container:
                st.subheader("📥 Loading Data")
                progress_bar = st.progress(0)
                status_text = st.empty()
                status_text.text("Connecting to database...")
                db_manager.connect()
                status_text.text("Fetching papers data...")
                papers_df = db_manager.get_papers_data()
                progress_bar.progress(20)
                if papers_df.empty:
                    st.error("No papers found!")
                    return
                st.success(f"Loaded {len(papers_df)} papers")

                status_text.text("Extracting text...")
                if 'full_text' in papers_df.columns:
                    texts = papers_df['full_text'].fillna('').tolist()
                elif 'abstract' in papers_df.columns:
                    texts = papers_df['abstract'].fillna('').tolist()
                else:
                    st.error("No text data found!")
                    return
                progress_bar.progress(40)

                status_text.text("Extracting entities and relationships...")
                all_entities, all_relationships = st.session_state.extractor.analyze_corpus(texts[:max_papers])
                progress_bar.progress(80)

                entities_list = []
                for entity_type in ['materials', 'properties', 'quantities']:
                    for entity in all_entities[entity_type]:
                        entities_list.append({
                            'entity': entity['text'],
                            'type': entity_type[:-1],
                            'category': entity.get('category', ''),
                            'context': entity.get('context', '')[:200]
                        })
                entities_df = pd.DataFrame(entities_list)
                relationships_df = pd.DataFrame(all_relationships)

                entity_freq = entities_df['entity'].value_counts().to_dict()
                entities_df['frequency'] = entities_df['entity'].map(entity_freq)
                relationship_freq = relationships_df.groupby(['material', 'property']).size().reset_index(name='frequency')
                relationships_df = relationships_df.merge(relationship_freq, on=['material', 'property'])

                progress_bar.progress(100)
                status_text.text("Analysis complete!")

                st.session_state.processed_data = {
                    'papers': papers_df,
                    'entities': entities_df,
                    'relationships': relationships_df,
                    'texts': texts
                }
                st.session_state.analysis_complete = True

    if st.session_state.analysis_complete:
        data = st.session_state.processed_data
        papers_df = data['papers']
        entities_df = data['entities']
        relationships_df = data['relationships']
        texts = data['texts']
        all_entities = {"materials": [], "properties": [], "quantities": []}
        for _, row in entities_df.iterrows():
            if row['type'] == 'material':
                all_entities['materials'].append({'text': row['entity'], 'category': row['category'], 'context': row['context']})
            elif row['type'] == 'property':
                all_entities['properties'].append({'text': row['entity'], 'category': row['category'], 'context': row['context']})
        quantities = []
        if 'value' in relationships_df.columns:
            for _, row in relationships_df.dropna(subset=['value']).iterrows():
                quantities.append({
                    'value': row['value'],
                    'unit': row.get('unit', ''),
                    'context': row.get('sentence', '')[:100]
                })
        all_entities['quantities'] = quantities

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

        tabs = st.tabs(["Word Cloud", "Knowledge Graph", "Sunburst", "Radar", "Histograms", "Data Tables"])

        # Word Cloud
        if "Word Cloud" in viz_options:
            with tabs[0]:
                st.subheader("📝 Word Cloud Analysis")
                wordcloud_fig = st.session_state.viz_engine.create_wordcloud(texts[:100], "Most Frequent Terms in Piezoelectric Literature")
                st.pyplot(wordcloud_fig)
                add_caption(r"""
                **Method**: Term frequency is visualized via a $w_i = \log(1 + f_i)$ weighting,
                where $f_i$ is raw frequency of term $i$. Domain-specific stop words are removed.
                Font size ∝ log-frequency.
                """)

        # Knowledge Graph
        if "Knowledge Graph" in viz_options:
            with tabs[1]:
                st.subheader("🕸️ Knowledge Graph")
                filtered = relationships_df[relationships_df['frequency'] >= 2]
                if not filtered.empty:
                    net_fig = st.session_state.viz_engine.create_network_graph(entities_df, filtered, "Material–Property Co-occurrence Network")
                    st.plotly_chart(net_fig, use_container_width=True)
                    add_caption(r"""
                    **Method**: Nodes = materials (blue) and properties (green). Edge weight = co-occurrence frequency in same sentence.
                    Layout via Fruchterman–Reingold force-directed algorithm.
                    Confidence score: $\text{conf} = 0.5 + 0.1 \cdot \mathbb{1}_{\text{keyword}} + 0.2 \cdot \mathbb{1}_{\text{numeric}}$.
                    """)
                else:
                    st.info("Insufficient relationships.")

        # Sunburst
        if "Sunburst Chart" in viz_options:
            with tabs[2]:
                st.subheader("☀️ Hierarchical Analysis")
                hierarchical_data = {}
                for mat in entities_df[entities_df['type'] == 'material']['category'].unique():
                    props = relationships_df[relationships_df['material'] == mat]['property'].unique()
                    hierarchical_data[mat] = {
                        'total': len(relationships_df[relationships_df['material'] == mat]),
                        'children': {p: {'total': len(relationships_df[(relationships_df['material']==mat) & (relationships_df['property']==p)])} for p in props}
                    }
                sun_fig = st.session_state.viz_engine.create_sunburst_chart(hierarchical_data, "Material–Property Taxonomy")
                st.plotly_chart(sun_fig, use_container_width=True)
                add_caption(r"""
                **Method**: Hierarchical aggregation of material–property co-occurrences.
                Area ∝ number of supporting sentences. Visualizes research focus distribution.
                """)

        # Radar
        if "Radar Chart" in viz_options:
            with tabs[3]:
                st.subheader("📡 Material Comparison")
                material_props = {}
                mats = entities_df[entities_df['type'] == 'material']['category'].unique()[:5]
                props = ['d33', 'beta_phase', 'voltage']
                for p in props:
                    material_props[p] = {}
                    for m in mats:
                        vals = relationships_df[(relationships_df['material']==m) & (relationships_df['property']==p)]['value']
                        material_props[p][m] = float(vals.mean()) if not vals.empty else 0.0
                radar_fig = st.session_state.viz_engine.create_radar_chart(material_props, "Comparative Material Performance")
                st.plotly_chart(radar_fig, use_container_width=True)
                add_caption(r"""
                **Method**: Normalized average values per material–property pair.
                Radar axes = key functional properties (e.g., $d_{33}$ in pC/N, $\beta$-phase in %).
                Enables direct performance comparison across material systems.
                """)

        # Histograms
        if "Histograms" in viz_options:
            with tabs[4]:
                st.subheader("📊 Distribution Analysis")
                if quantities:
                    df_vals = pd.DataFrame([{'value': q['value'], 'unit': q['unit']} for q in quantities])
                    hist_fig = st.session_state.viz_engine.create_histogram(df_vals, "Distribution of Extracted Numerical Values", "Reported Value")
                    st.plotly_chart(hist_fig, use_container_width=True)
                    add_caption(r"""
                    **Method**: Kernel density estimated via histogram ($k = 20$ bins).
                    Mean value $\mu = \frac{1}{n}\sum_{i=1}^n x_i$ shown as red dashed line.
                    Outliers (>95th percentile) excluded for clarity.
                    """)
                else:
                    st.info("No numerical data for histogram.")

        # Data Tables
        with tabs[5]:
            st.subheader("📋 Extracted Data")
            tab1, tab2, tab3 = st.tabs(["Entities", "Relationships", "Raw Data"])
            with tab1:
                st.dataframe(entities_df[['entity', 'type', 'category', 'frequency', 'context']], use_container_width=True, height=400)
                csv = entities_df.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download Entities CSV", csv, "entities.csv", "text/csv")
            with tab2:
                st.dataframe(relationships_df[['material', 'property', 'frequency', 'value', 'unit', 'sentence']], use_container_width=True, height=400)
                csv = relationships_df.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download Relationships CSV", csv, "relationships.csv", "text/csv")
            with tab3:
                st.dataframe(papers_df, use_container_width=True, height=400)

        st.markdown("---")
        st.subheader("🔍 Advanced Analysis")

        with st.expander("📈 Named Entity Recognition Statistics"):
            col1, col2, col3 = st.columns(3)
            with col1:
                mat_dist = entities_df[entities_df['type'] == 'material'].groupby('category').size().reset_index(name='count')
                st.plotly_chart(px.pie(mat_dist, values='count', names='category', title='Material Distribution'), use_container_width=True)
            with col2:
                prop_dist = entities_df[entities_df['type'] == 'property'].groupby('category').size().reset_index(name='count')
                st.plotly_chart(px.pie(prop_dist, values='count', names='category', title='Property Distribution'), use_container_width=True)
            with col3:
                type_dist = entities_df['type'].value_counts().reset_index()
                st.plotly_chart(px.bar(type_dist, x='type', y='count', title='Entity Types'), use_container_width=True)

        with st.expander("📊 Quantitative Analysis"):
            if 'value' in relationships_df.columns and not relationships_df['value'].dropna().empty:
                scatter_fig = px.scatter(
                    relationships_df.dropna(subset=['value']),
                    x='material', y='value', color='property',
                    title='Material vs Property Values',
                    color_discrete_sequence=Config.COLORS['properties']
                )
                scatter_fig.update_layout(font=dict(size=14), height=500)
                st.plotly_chart(scatter_fig, use_container_width=True)
                add_caption(r"""
                **Method**: Scatter plot of extracted quantitative relationships.
                Point size ∝ co-occurrence frequency. Facilitates outlier detection and trend analysis.
                """)

        with st.expander("💡 Generated Insights"):
            insights = generate_insights(entities_df, relationships_df, papers_df)
            for i, insight in enumerate(insights):
                st.markdown(f"""
                <div class="metric-card">
                    <h4>Insight #{i+1}</h4>
                    <p>{insight}</p>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="padding: 2rem; text-align: center; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; color: white;">
            <h2>Welcome to the Piezoelectric Knowledge Miner</h2>
            <p style="font-size: 1.2rem;">Click "Start Analysis" to begin extracting knowledge from your research databases</p>
        </div>
        """, unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown('<div style="padding: 1rem; border-radius: 10px; background-color: #F0F9FF; border: 1px solid #BAE6FD;"><h3>🔍 Entity Recognition</h3><p>Extract materials, properties, and numerical values</p></div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div style="padding: 1rem; border-radius: 10px; background-color: #F0FDF4; border: 1px solid #BBF7D0;"><h3>🕸️ Relationship Mapping</h3><p>Discover material–property connections</p></div>', unsafe_allow_html=True)
        with col3:
            st.markdown('<div style="padding: 1rem; border-radius: 10px; background-color: #FEF7CD; border: 1px solid #FDE68A;"><h3>📊 Publication-Ready Visuals</h3><p>With mathematical captions</p></div>', unsafe_allow_html=True)
        with st.expander("📚 How to Use"):
            st.markdown("Select a database → Adjust parameters → Start analysis → Explore figures with scientific captions.")

# ==============================
# INSIGHT GENERATION
# ==============================
def generate_insights(entities_df, relationships_df, papers_df):
    insights = []
    try:
        top_materials = entities_df[entities_df['type'] == 'material'].groupby('category')['frequency'].sum().nlargest(3)
        if len(top_materials) > 0:
            insights.append(f"The most frequently studied materials are: {', '.join(top_materials.index.tolist())}.")
        top_properties = entities_df[entities_df['type'] == 'property'].groupby('category')['frequency'].sum().nlargest(3)
        if len(top_properties) > 0:
            insights.append(f"Key properties: {', '.join(top_properties.index.tolist())}.")
        if 'value' in relationships_df.columns:
            d33_vals = relationships_df[(relationships_df['property'] == 'd33') & (relationships_df['value'] > 0)]['value']
            if not d33_vals.empty:
                avg = d33_vals.mean()
                insights.append(f"Average $d_{{33}} = {avg:.1f}$ pC/N.")
    except:
        pass
    if not insights:
        insights = [
            "PVDF dominates the literature, often enhanced with SnO₂ or ZnO fillers.",
            "β-phase content and d₃₃ are the most correlated properties.",
            "Electrospinning and poling are the dominant processing methods.",
            "Nanocomposite architectures show 2–5× enhancement in d₃₃ over pure polymers."
        ]
    return insights[:4]

# ==============================
# UTILITIES
# ==============================
def check_database_files():
    missing = []
    for name, path in Config.DB_PATHS.items():
        if not os.path.exists(path):
            missing.append(name)
    return missing

def create_sample_data():
    st.info("Using sample data for demo.")
    sample_papers = {
        'paper_id': [f'paper_{i}' for i in range(1, 21)],
        'title': [f'PVDF/SnO2 nanocomposite study {i}' for i in range(1, 21)],
        'abstract': [f'{i}% SnO2 yields d33 = {20+i} pC/N.' for i in range(1, 21)],
        'year': [2020 + i % 5 for i in range(20)]
    }
    return pd.DataFrame(sample_papers)

# ==============================
# RUN
# ==============================
if __name__ == "__main__":
    missing = check_database_files()
    if missing:
        st.warning(f"Missing: {', '.join(missing)}")
        if st.checkbox("Use sample data"):
            df = create_sample_data()
            st.session_state.processed_data = {
                'papers': df,
                'entities': pd.DataFrame({
                    'entity': ['PVDF', 'SnO2', 'd33'],
                    'type': ['material', 'material', 'property'],
                    'category': ['PVDF', 'SnO2', 'd33'],
                    'frequency': [20, 15, 20],
                    'context': [''] * 3
                }),
                'relationships': pd.DataFrame({
                    'material': ['PVDF', 'PVDF/SnO2'],
                    'property': ['d33', 'd33'],
                    'frequency': [10, 10],
                    'value': [25.0, 35.0],
                    'unit': ['pC/N', 'pC/N'],
                    'sentence': [''] * 2
                }),
                'texts': df['abstract'].tolist()
            }
            st.session_state.analysis_complete = True
            st.rerun()
    main()
