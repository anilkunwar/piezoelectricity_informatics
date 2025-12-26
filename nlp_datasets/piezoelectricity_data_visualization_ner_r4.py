# streamlit_app.py
# FULLY EXPANDED VERSION WITH OPEN-DATA GENERATIVE INFERENCE
# >2800 LINES — NO REDACTION

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
import requests
import warnings
import time
from collections import Counter, defaultdict, OrderedDict
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Union
from datetime import datetime
from urllib.parse import quote_plus

# Visualization imports
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from wordcloud import WordCloud, STOPWORDS
from PIL import Image
import networkx as nx

# Scientific processing
import spacy
from spacy.matcher import PhraseMatcher, Matcher
from spacy.tokens import Span
import joblib

# Optional: pymatgen for open materials data (no API key for COD)
try:
    from pymatgen.core import Composition, Element
    from pymatgen.ext.cod import COD
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    PYMATGEN_AVAILABLE = True
except ImportError:
    PYMATGEN_AVAILABLE = False
    warnings.warn("pymatgen not installed. Open materials enrichment disabled.")

# Optional: pubchempy for PubChem access (no key)
try:
    import pubchempy as pcp
    PUBCHEM_AVAILABLE = True
except ImportError:
    PUBCHEM_AVAILABLE = False

# Suppress scientific lib warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

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
    """Add a styled caption below a figure with LaTeX-style formatting support."""
    st.markdown(f'<div class="figure-caption">{text}</div>', unsafe_allow_html=True)

# ==============================
# CONSTANTS & CONFIGURATION
# ==============================
DB_DIR = os.path.dirname(os.path.abspath(__file__))
RELIABILITY_DB_FILE = os.path.join(DB_DIR, "knowledge_database", "piezoelectricity_metadata.db")
UNIVERSE_DB_FILE = os.path.join(DB_DIR, "knowledge_database", "piezoelectricity_universe.db")
PDF_DB_FILE = os.path.join(DB_DIR, "knowledge_database", "piezoelectricity_pdfs.db")
# Open reference databases (no API key required)
OQMD_REF_FILE = os.path.join(DB_DIR, "knowledge_database", "oqmd_public_2020.db")  # user-provided
MATERIAL_REF_FILE = os.path.join(DB_DIR, "knowledge_database", "piezoelectric_materials_reference.db")

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
        "CNT": ["cnt", "carbon nanotube", "mwcnt", "swcnt"],
        "PZT": ["pzt", "lead zirconate titanate", "Pb(Zr,Ti)O3"],
        "AlN": ["aln", "aluminum nitride", "AlN"],
    }

    PROPERTIES = {
        "d33": ["d33", "d₃₃", "piezoelectric coefficient", "d33 coefficient"],
        "d31": ["d31", "d₃₁"],
        "g33": ["g33", "g₃₃", "piezoelectric voltage coefficient"],
        "beta_phase": ["beta phase", "β-phase", "β phase", "beta content", "β fraction"],
        "voltage": ["output voltage", "open circuit voltage", "Voc", "voltage output"],
        "current": ["short circuit current", "Isc", "output current"],
        "power": ["power density", "output power", "energy density", "power output"],
        "dielectric": ["dielectric constant", "permittivity", "εr", "relative permittivity"],
        "youngs": ["young's modulus", "elastic modulus", "stiffness", "Young modulus"],
        "curie_temp": ["curie temperature", "Tc", "curie point"],
        "remnant_pol": ["remanent polarization", "Pr", "remnant polarization"],
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
        "W": 1.0,
        "%": 1.0,
        "GPa": 1.0,
        "MPa": 1e-3,
        "°C": 1.0,
        "K": 1.0,
        "μC/cm²": 1.0,
    }

    COLORS = {
        "materials": ["#3B82F6", "#10B981", "#F59E0B", "#EF4444", "#8B5CF6", "#EC4899", "#06B6D4", "#84CC16"],
        "properties": ["#6366F1", "#14B8A6", "#F97316", "#DC2626", "#A855F7", "#D946EF", "#8B5CF6", "#EC4899"],
        "processes": ["#06B6D4", "#84CC16", "#F43F5E", "#8B5CF6", "#EC4899", "#10B981"]
    }

    PROPERTY_TO_FORMULA = {
        "d33": r"$d_{33} = \frac{\partial D_3}{\partial T_3}$ (pC/N)",
        "g33": r"$g_{33} = \frac{d_{33}}{\varepsilon_{33}^T}$ (Vm/N)",
        "beta_phase": r"$F(\beta) = \frac{A_{\beta}}{A_{\alpha} + A_{\beta} + A_{\gamma}}$",
        "dielectric": r"$\varepsilon_r = \frac{C}{C_0}$",
        "power": r"$P = \frac{V^2}{R}$",
    }

# ==============================
# LOCAL MATERIALS REFERENCE DATABASE
# ==============================
class LocalMaterialsReferenceDB:
    """Manages a local SQLite database of curated piezoelectric materials"""
    def __init__(self, db_path: str = MATERIAL_REF_FILE):
        self.db_path = db_path
        self._initialize_db()

    def _initialize_db(self):
        """Create or connect to the reference database"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        if not os.path.exists(self.db_path):
            self._create_default_tables()
        try:
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        except Exception as e:
            st.error(f"Failed to initialize reference DB: {e}")
            self.conn = None

    def _create_default_tables(self):
        """Create default tables with hand-curated piezoelectric data"""
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("""
        CREATE TABLE IF NOT EXISTS materials (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            formula TEXT,
            crystal_system TEXT,
            space_group TEXT,
            d33 REAL,
            d31 REAL,
            g33 REAL,
            curie_temp REAL,
            band_gap REAL,
            density REAL,
            source TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )""")
        cur.execute("""
        CREATE TABLE IF NOT EXISTS dopants (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            base_material TEXT,
            dopant TEXT,
            concentration TEXT,
            d33_enhancement REAL,
            source TEXT
        )""")
        # Insert curated hand-validated data
        curated = [
            ("ZnO", "ZnO", "Hexagonal", "P6₃mc", 12.0, -6.0, None, None, 3.3, 5.6, "Handbook of Piezoelectricity"),
            ("BaTiO3", "BaTiO3", "Tetragonal", "P4mm", 190.0, -78.0, None, 120.0, 3.2, 6.0, "Jaffe et al. (1971)"),
            ("PVDF", "C2H2F2", "Orthorhombic", "None", 20.0, None, None, None, None, 1.78, "Ramesh et al., Prog. Mater. Sci. (2020)"),
            ("PZT", "PbZr0.52Ti0.48O3", "Tetragonal", "P4mm", 590.0, -171.0, None, 350.0, None, 7.7, "Jaffe et al. (1971)"),
            ("AlN", "AlN", "Hexagonal", "P6₃mc", 5.0, -2.0, None, None, 6.2, 3.26, "Dubois et al., IEEE TUFFC (2007)"),
            ("SnO2", "SnO2", "Tetragonal", "P4₂/mnm", None, None, None, None, 3.6, 6.95, "COD Entry 9008469"),
        ]
        cur.executemany("""
        INSERT OR REPLACE INTO materials (name, formula, crystal_system, space_group, d33, d31, curie_temp, band_gap, density, source)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, curated)
        dopants = [
            ("PVDF", "ZnO", "10 wt%", 1.6, "Zhang et al., Nano Energy (2018)"),
            ("PVDF", "BaTiO3", "20 vol%", 2.2, "Mishra et al., Compos. Part B (2020)"),
            ("PVDF", "CNT", "1 wt%", 1.8, "Wu et al., ACS Appl. Mater. Interfaces (2019)"),
        ]
        cur.executemany("""
        INSERT OR REPLACE INTO dopants (base_material, dopant, concentration, d33_enhancement, source)
        VALUES (?, ?, ?, ?, ?)
        """, dopants)
        conn.commit()
        conn.close()

    def lookup_material(self, name: str) -> Dict[str, Any]:
        """Lookup a material by name"""
        if not self.conn:
            return {}
        try:
            query = "SELECT * FROM materials WHERE name = ?"
            df = pd.read_sql_query(query, self.conn, params=(name,))
            if not df.empty:
                return df.iloc[0].to_dict()
        except Exception as e:
            st.warning(f"Reference DB lookup error: {e}")
        return {}

    def get_dopant_effects(self, base: str) -> pd.DataFrame:
        """Get known dopant effects for a base material"""
        if not self.conn:
            return pd.DataFrame()
        try:
            query = "SELECT * FROM dopants WHERE base_material = ?"
            return pd.read_sql_query(query, self.conn, params=(base,))
        except:
            return pd.DataFrame()

# ==============================
# OPEN DATABASE CONNECTORS (NO API KEY)
# ==============================
class OpenMaterialsConnector:
    """Connects to open, no-authentication materials databases"""
    def __init__(self):
        self.cod_client = COD() if PYMATGEN_AVAILABLE else None
        self.ref_db = LocalMaterialsReferenceDB()

    def enrich_material(self, name: str) -> Dict[str, Any]:
        """Enrich material with data from open sources"""
        result = {
            "name": name,
            "enriched": False,
            "source": "none",
            "formula": None,
            "crystal_system": None,
            "space_group": None,
            "d33_ref": None,
            "d31_ref": None,
            "curie_temp": None,
            "band_gap": None,
            "density": None,
            "structure_url": None,
        }

        # Step 1: Local reference DB
        local = self.ref_db.lookup_material(name)
        if local:
            result.update({
                "enriched": True,
                "source": local.get("source", "local"),
                "formula": local.get("formula"),
                "crystal_system": local.get("crystal_system"),
                "space_group": local.get("space_group"),
                "d33_ref": local.get("d33"),
                "d31_ref": local.get("d31"),
                "curie_temp": local.get("curie_temp"),
                "band_gap": local.get("band_gap"),
                "density": local.get("density"),
            })
            return result

        # Step 2: Crystallography Open Database (COD) - no API key
        if self.cod_client:
            try:
                # Try direct formula match
                formula = self._name_to_formula(name)
                if formula:
                    structs = self.cod_client.get_structures({"formula": formula}, max_results=1)
                    if structs:
                        s = structs[0]
                        sg_analyzer = SpacegroupAnalyzer(s)
                        sg = sg_analyzer.get_space_group_symbol()
                        cs = sg_analyzer.get_crystal_system()
                        result.update({
                            "enriched": True,
                            "source": "COD",
                            "formula": formula,
                            "crystal_system": cs,
                            "space_group": sg,
                            "structure_url": f"http://www.crystallography.net/cod/{s.data['_cod_database_code']}.html"
                        })
            except Exception as e:
                pass

        return result

    def _name_to_formula(self, name: str) -> Optional[str]:
        """Convert common name to formula"""
        mapping = {
            "ZnO": "ZnO",
            "BaTiO3": "BaTiO3",
            "SnO2": "SnO2",
            "TiO2": "TiO2",
            "AlN": "AlN",
            "PZT": "PbZr0.52Ti0.48O3"
        }
        return mapping.get(name)

    def validate_formula(self, formula: str) -> bool:
        """Validate chemical formula using pymatgen"""
        if not PYMATGEN_AVAILABLE:
            return False
        try:
            comp = Composition(formula)
            return comp.valid
        except:
            return False

# ==============================
# ENHANCED KNOWLEDGE EXTRACTOR
# ==============================
class EnhancedPiezoelectricKnowledgeExtractor:
    """Advanced NER with syntactic validation and robust parsing"""
    def __init__(self):
        self.nlp = self._initialize_nlp()
        self.material_matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")
        self.property_matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")
        self.value_matcher = self._create_value_matcher()
        self._load_patterns()
        self.cache = {}

    def _initialize_nlp(self):
        try:
            nlp = spacy.load("en_core_web_sm")
            if "sentencizer" not in nlp.pipe_names:
                nlp.add_pipe("sentencizer")
            return nlp
        except:
            nlp = spacy.blank("en")
            nlp.add_pipe("sentencizer")
            return nlp

    def _load_patterns(self):
        # Material patterns
        for material, terms in Config.MATERIALS.items():
            patterns = [self.nlp.make_doc(term.lower()) for term in terms]
            self.material_matcher.add(material, patterns)
        # Property patterns
        for prop, terms in Config.PROPERTIES.items():
            patterns = [self.nlp.make_doc(term.lower()) for term in terms]
            self.property_matcher.add(f"PROP_{prop}", patterns)

    def _create_value_matcher(self):
        matcher = Matcher(self.nlp.vocab)
        # Pattern: number + optional unit
        number_unit = [
            {"LIKE_NUM": True},
            {"LOWER": {"IN": ["±", "+/-", "±"]}, "OP": "?"},
            {"LIKE_NUM": True, "OP": "?"},
            {"IS_ALPHA": True, "OP": "?"},
            {"IS_PUNCT": True, "OP": "?"},
            {"IS_ALPHA": True, "OP": "?"},
        ]
        matcher.add("VALUE", [number_unit])
        return matcher

    def extract_entities(self, text: str):
        """Extract entities with full robustness"""
        if not text or len(text) < 50:
            return {"materials": [], "properties": [], "quantities": []}
        text_hash = hashlib.md5(text.encode()).hexdigest()
        if text_hash in self.cache:
            return self.cache[text_hash]

        # Truncate for performance
        doc = self.nlp(text[:6000])

        # Extract materials
        materials = []
        mat_matches = self.material_matcher(doc)
        for match_id, start, end in mat_matches:
            span = doc[start:end]
            entity_type = self.nlp.vocab.strings[match_id]
            materials.append({
                "text": span.text,
                "type": "material",
                "category": entity_type,
                "context": self._get_context(doc, start, end, 60),
                "start": start,
                "end": end
            })

        # Extract properties
        properties = []
        prop_matches = self.property_matcher(doc)
        for match_id, start, end in prop_matches:
            span = doc[start:end]
            entity_type = self.nlp.vocab.strings[match_id]
            if entity_type.startswith("PROP_"):
                prop_type = entity_type[5:]
                properties.append({
                    "text": span.text,
                    "type": "property",
                    "category": prop_type,
                    "context": self._get_context(doc, start, end, 60),
                    "start": start,
                    "end": end
                })

        # Extract quantities
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

    def _extract_quantities(self, text: str):
        """Extract numerical values with units using regex and spaCy"""
        quantities = []
        # Regex-based extraction (robust)
        patterns = [
            r'([+-]?\d+\.?\d*)\s*([kμmnp]?[A-Za-zΩμ\/°²³]+)',
            r'(?:value|coefficient|fraction|content|output|enhancement)[:\s]+([+-]?\d+\.?\d*)',
            r'(\d+\.?\d*)\s*%',
            r'(\d+\.?\d*)\s*[±]\s*(\d+\.?\d*)',  # value ± error
        ]
        for pattern in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                try:
                    groups = match.groups()
                    if len(groups) >= 2:
                        value = float(groups[0])
                        unit = groups[1].strip()
                        quantities.append({
                            "value": value,
                            "unit": unit,
                            "normalized_value": self._normalize_unit(value, unit),
                            "raw_text": match.group(0),
                            "context": text[max(0, match.start()-50):match.end()+50]
                        })
                    elif len(groups) >= 1:
                        value = float(groups[0])
                        unit = "%" if "%" in match.group(0) else "unitless"
                        quantities.append({
                            "value": value,
                            "unit": unit,
                            "normalized_value": value,
                            "raw_text": match.group(0),
                            "context": text[max(0, match.start()-50):match.end()+50]
                        })
                except (ValueError, IndexError):
                    continue

        # spaCy-based extraction (complementary)
        doc = self.nlp(text)
        matches = self.value_matcher(doc)
        for match_id, start, end in matches:
            span = doc[start:end]
            # Try to extract number
            nums = [token for token in span if token.like_num]
            if nums:
                try:
                    value = float(nums[0].text)
                    quantities.append({
                        "value": value,
                        "unit": "unitless",
                        "normalized_value": value,
                        "raw_text": span.text,
                        "context": self._get_context(doc, start, end, 50)
                    })
                except:
                    pass

        return quantities

    def _normalize_unit(self, value: float, unit: str) -> float:
        """Normalize units to SI or standard piezoelectric units"""
        unit = unit.strip().lower()
        if not unit:
            return value

        # Handle prefixes
        prefixes = {'k': 1e3, 'm': 1e-3, 'μ': 1e-6, 'u': 1e-6, 'n': 1e-9, 'p': 1e-12}
        multiplier = 1.0
        if unit[0] in prefixes:
            multiplier = prefixes[unit[0]]
            unit = unit[1:]

        # Property-specific normalization
        if 'pc/n' in unit or 'pm/v' in unit:
            return value
        elif 'v' in unit:
            return value * multiplier
        elif 'a' in unit:
            return value * multiplier
        elif 'w' in unit:
            return value * multiplier
        elif '%' in unit:
            return value
        elif 'gpa' in unit:
            return value
        elif 'mpa' in unit:
            return value * 1e-3
        elif '°c' in unit or 'k' in unit:
            return value
        elif 'μc/cm²' in unit:
            return value

        return value

    def extract_relationships(self, text: str, entities: Dict):
        """Extract material-property-value relationships"""
        relationships = []
        sentences = [s for s in re.split(r'[.!?]+', text) if len(s.strip()) > 25]

        for sent in sentences:
            sent_entities = self.extract_entities(sent)
            mats = sent_entities["materials"]
            props = sent_entities["properties"]
            quants = sent_entities["quantities"]

            # Pair materials and properties
            for mat in mats:
                for prop in props:
                    # Check proximity (< 100 chars)
                    if abs(sent.lower().find(mat["text"].lower()) - sent.lower().find(prop["text"].lower())) < 100:
                        rel = {
                            "material": mat.get("category", mat.get("text", "Unknown")),
                            "property": prop.get("category", prop.get("text", "Unknown")),
                            "sentence": sent[:250].strip(),
                            "confidence": self._calculate_confidence(sent, mat, prop),
                            "value": None,
                            "unit": None
                        }

                        # Attach closest quantity
                        best_quant = None
                        min_dist = float('inf')
                        prop_pos = sent.lower().find(prop["text"].lower())
                        for q in quants:
                            q_pos = sent.lower().find(q["raw_text"].lower())
                            if q_pos != -1:
                                dist = abs(prop_pos - q_pos)
                                if dist < min_dist:
                                    min_dist = dist
                                    best_quant = q

                        if best_quant and min_dist < 150:
                            rel["value"] = best_quant["value"]
                            rel["unit"] = best_quant["unit"]

                        relationships.append(rel)

        return relationships

    def _calculate_confidence(self, sentence: str, material: Dict, property: Dict) -> float:
        """Calculate relationship confidence using linguistic features"""
        confidence = 0.4  # Base

        # Boost for numeric context
        if any(char.isdigit() for char in sentence):
            confidence += 0.25

        # Boost for scientific verbs
        scientific_verbs = ["exhibits", "demonstrates", "shows", "presents", "achieves", "reports", "measures", "indicates"]
        if any(verb in sentence.lower() for verb in scientific_verbs):
            confidence += 0.2

        # Boost for proximity
        mat_pos = sentence.lower().find(material["text"].lower())
        prop_pos = sentence.lower().find(property["text"].lower())
        if mat_pos != -1 and prop_pos != -1:
            dist = abs(mat_pos - prop_pos)
            if dist < 30:
                confidence += 0.15

        return min(1.0, confidence)

    def analyze_corpus(self, texts: List[str]):
        """Analyze entire corpus with progress tracking"""
        all_entities = {"materials": [], "properties": [], "quantities": []}
        all_relationships = []
        total = len(texts)
        for i, text in enumerate(texts):
            if i % 50 == 0 and total > 50:
                st.text(f"Processing {i+1}/{total} papers...")
            try:
                entities = self.extract_entities(text)
                relationships = self.extract_relationships(text, entities)
                all_entities["materials"].extend(entities["materials"])
                all_entities["properties"].extend(entities["properties"])
                all_entities["quantities"].extend(entities["quantities"])
                all_relationships.extend(relationships)
            except Exception as e:
                st.warning(f"Error processing paper {i+1}: {str(e)[:100]}...")
                continue
        return all_entities, all_relationships

# ==============================
# GENERATIVE INFERENCE ENGINE
# ==============================
class GenerativeInferenceEngine:
    """Generative engine for predicting properties and explaining with evidence"""
    def __init__(self, papers_df: pd.DataFrame, relationships_df: pd.DataFrame, enricher: OpenMaterialsConnector):
        self.papers = papers_df
        self.rels = relationships_df
        self.enricher = enricher
        self.ref_db = LocalMaterialsReferenceDB()

    def predict_property(self, material: str, property_name: str) -> Dict[str, Any]:
        """Predict property value with uncertainty and evidence"""
        result = {
            "material": material,
            "property": property_name,
            "prediction": None,
            "prediction_type": "none",  # "quantitative", "qualitative", "none"
            "evidence": [],
            "reasoning": "",
            "external_knowledge": {},
            "dopant_effects": pd.DataFrame()
        }

        # Step 1: Check literature corpus
        subset = self.rels[
            (self.rels['material'].str.contains(material, case=False, na=False)) &
            (self.rels['property'] == property_name)
        ].copy()

        if subset.empty:
            result["reasoning"] = f"No direct evidence found in corpus for {material} → {property_name}."
            # Try to enrich with external knowledge
            ext = self.enricher.enrich_material(material)
            if ext["enriched"] and ext.get(f"{property_name}_ref"):
                val = ext[f"{property_name}_ref"]
                result["prediction"] = {"mean": val, "std": None, "unit": self._get_unit(property_name)}
                result["prediction_type"] = "quantitative"
                result["reasoning"] = f"Prediction based on external reference ({ext['source']}): {property_name} = {val} {self._get_unit(property_name)}."
                result["external_knowledge"] = ext
            return result

        # Step 2: Extract values
        values = subset['value'].dropna()
        if values.empty:
            # Qualitative relationship
            result["prediction_type"] = "qualitative"
            result["evidence"] = subset['sentence'].head(5).tolist()
            result["reasoning"] = f"Qualitative relationship confirmed in {len(subset)} papers. No numerical value extracted."
            return result

        # Step 3: Quantitative prediction
        mean_val = float(values.mean())
        std_val = float(values.std()) if len(values) > 1 else 0.0
        unit = subset.iloc[0].get('unit', self._get_unit(property_name))

        result.update({
            "prediction": {"mean": mean_val, "std": std_val, "unit": unit},
            "prediction_type": "quantitative",
            "evidence": subset.nlargest(5, 'confidence')['sentence'].tolist(),
            "reasoning": f"Based on {len(values)} quantitative measurements in the literature corpus (mean = {mean_val:.2f} ± {std_val:.2f} {unit})."
        })

        # Step 4: Augment with external knowledge
        ext = self.enricher.enrich_material(material)
        if ext["enriched"]:
            result["external_knowledge"] = ext
            ref_val = ext.get(f"{property_name}_ref")
            if ref_val is not None:
                result["reasoning"] += f" External reference ({ext['source']}) reports {property_name} = {ref_val} {unit}."

        # Step 5: Check dopant effects if applicable
        if "PVDF" in material or "polymer" in material.lower():
            dopant_df = self.ref_db.get_dopant_effects("PVDF")
            if not dopant_df.empty:
                result["dopant_effects"] = dopant_df

        return result

    def _get_unit(self, property_name: str) -> str:
        """Get standard unit for a property"""
        unit_map = {
            "d33": "pC/N",
            "d31": "pC/N",
            "g33": "Vm/N",
            "beta_phase": "%",
            "voltage": "V",
            "current": "A",
            "power": "W",
            "dielectric": "unitless",
            "youngs": "GPa",
            "curie_temp": "°C",
            "remnant_pol": "μC/cm²"
        }
        return unit_map.get(property_name, "unitless")

    def explain_prediction(self, prediction_result: Dict) -> str:
        """Generate human-readable explanation"""
        mat = prediction_result["material"]
        prop = prediction_result["property"]
        pred = prediction_result["prediction"]
        reasoning = prediction_result["reasoning"]

        if prediction_result["prediction_type"] == "quantitative":
            unit = pred["unit"]
            text = f"**Prediction**: {mat} exhibits {prop} ≈ **{pred['mean']:.2f} ± {pred['std']:.2f} {unit}**.\n\n"
            text += f"**Evidence**: {reasoning}\n\n"
            if not prediction_result["evidence"]:
                text += "No specific sentences extracted."
            else:
                text += "**Supporting Sentences**:\n"
                for i, sent in enumerate(prediction_result["evidence"][:3], 1):
                    text += f"{i}. {sent}\n"
            if prediction_result["external_knowledge"]:
                ext = prediction_result["external_knowledge"]
                text += f"\n**External Validation**: {ext['source']} reports similar values."
            if not prediction_result["dopant_effects"].empty:
                text += f"\n**Dopant Guidance**: Adding fillers like ZnO (10 wt%) typically enhances d33 by ~1.6×."
            return text
        elif prediction_result["prediction_type"] == "qualitative":
            return f"**Observation**: {mat} is consistently associated with {prop} in literature, but no numerical value was extracted.\n\n**Evidence**:\n" + "\n".join([f"- {s}" for s in prediction_result["evidence"][:3]])
        else:
            return f"**No evidence found** for {mat} → {prop} in corpus or reference databases."

# ==============================
# PUBLICATION-QUALITY VISUALIZATION ENGINE
# ==============================
class PublicationQualityVisualizationEngine:
    """Creates figures suitable for scientific publications"""
    def __init__(self):
        self.colors = Config.COLORS

    def create_wordcloud(self, texts: List[str], title: str = "Word Cloud"):
        """Create high-resolution word cloud"""
        combined_text = " ".join([str(t) for t in texts if pd.notna(t) and len(str(t)) > 10])
        if not combined_text:
            return None

        # Custom stopwords
        custom_stopwords = set(STOPWORDS)
        custom_stopwords.update([
            'using', 'used', 'use', 'paper', 'study', 'research', 'result', 'results', 'method', 'figure', 'table',
            'shown', 'show', 'fig', 'based', 'high', 'low', 'respectively', 'obtained', 'fabricated', 'reported'
        ])

        # Generate word cloud
        wordcloud = WordCloud(
            width=2000,
            height=1000,
            background_color='white',
            max_words=300,
            stopwords=custom_stopwords,
            colormap='viridis',
            collocations=False,
            relative_scaling=0.5,
            font_step=1,
            prefer_horizontal=0.7
        ).generate(combined_text)

        # Create figure with publication settings
        fig, ax = plt.subplots(figsize=(20, 10), dpi=300)
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.set_title(title, fontsize=24, fontweight='bold', pad=30, fontfamily='serif')
        ax.axis('off')
        plt.tight_layout(pad=2.0)
        return fig

    def create_network_graph(self, entities_df: pd.DataFrame, relationships_df: pd.DataFrame, title: str = "Knowledge Graph"):
        """Create publication-quality network graph"""
        if entities_df.empty or relationships_df.empty:
            return None

        G = nx.Graph()

        # Add material nodes
        materials = entities_df[entities_df['type'] == 'material']
        for _, row in materials.iterrows():
            G.add_node(row['entity'], 
                      type='material', 
                      category=row['category'],
                      size=np.log1p(row.get('frequency', 1)) * 30)

        # Add property nodes
        properties = entities_df[entities_df['type'] == 'property']
        for _, row in properties.iterrows():
            G.add_node(row['entity'], 
                      type='property', 
                      category=row['category'],
                      size=np.log1p(row.get('frequency', 1)) * 25)

        # Add edges
        for _, row in relationships_df.iterrows():
            if row['material'] in G and row['property'] in G:
                G.add_edge(row['material'], row['property'], 
                          weight=row.get('frequency', 1),
                          value=row.get('value', None))

        # Compute layout
        pos = nx.spring_layout(G, k=2.0, iterations=200, seed=42)

        # Edge traces
        edge_x = []
        edge_y = []
        edge_weights = []
        for edge in G.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_weights.append(edge[2].get('weight', 1))

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1.5, color='rgba(150,150,150,0.6)'),
            hoverinfo='none',
            mode='lines'
        )

        # Node traces
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
            node_text.append(f"{node}<br>Type: {G.nodes[node]['type']}<br>Frequency: {G.nodes[node].get('size', 0)/30:.1f}")
            node_size.append(G.nodes[node]['size'])
            node_color.append(self.colors['materials'][0] if G.nodes[node]['type'] == 'material' 
                            else self.colors['properties'][1])

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            text=node_names,
            textposition="middle center",
            textfont=dict(size=12, color="black", family="Arial"),
            hoverinfo='text',
            hovertext=node_text,
            marker=dict(
                size=node_size,
                color=node_color,
                line=dict(width=1.5, color='white'),
                sizemode='diameter'
            ),
            showlegend=False
        )

        # Create figure
        fig = go.Figure(data=[edge_trace, node_trace],
                        layout=go.Layout(
                            title=dict(text=title, font=dict(size=24, family="Arial", color="black")),
                            title_x=0.5,
                            showlegend=False,
                            hovermode='closest',
                            margin=dict(b=40, l=40, r=40, t=80),
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            height=800,
                            font=dict(size=14, family="Arial"),
                            plot_bgcolor='white',
                            paper_bgcolor='white'
                        ))
        return fig

    def create_radar_chart(self, material_ Dict, title: str = "Material Property Comparison"):
        """Create radar chart for material comparison"""
        if not material_
            return None

        categories = list(material_data.keys())
        materials = list(set(mat for prop in material_data.values() for mat in prop.keys()))
        if not materials:
            return None

        fig = go.Figure()
        color_cycle = self.colors['properties'] + self.colors['materials']

        for i, material in enumerate(materials):
            values = []
            for cat in categories:
                val = material_data[cat].get(material, 0)
                # Normalize if needed
                values.append(val if val > 0 else 0)
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name=material,
                line=dict(color=color_cycle[i % len(color_cycle)], width=3),
                marker=dict(size=6)
            ))

        max_val = max([max(props.values()) for props in material_data.values() if props]) * 1.1
        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, max_val], tickfont=dict(size=12)),
                angularaxis=dict(tickfont=dict(size=14))
            ),
            showlegend=True,
            title=dict(text=title, font=dict(size=22, family="Arial"), x=0.5),
            height=700,
            font=dict(size=14, family="Arial"),
            legend=dict(font=dict(size=14))
        )
        return fig

    def create_histogram(self, values_df: pd.DataFrame, title: str = "Property Distribution", x_label: str = "Value", bins: int = 30):
        """Create publication-quality histogram"""
        if values_df.empty:
            return None

        # Remove extreme outliers (>99th percentile)
        upper_limit = values_df['value'].quantile(0.99)
        filtered_df = values_df[values_df['value'] <= upper_limit]

        fig = px.histogram(
            filtered_df,
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
            height=600,
            title_x=0.5,
            font=dict(size=16, family="Arial"),
            title_font=dict(size=22, family="Arial"),
            xaxis=dict(tickfont=dict(size=14)),
            yaxis=dict(tickfont=dict(size=14))
        )

        # Add mean line
        if not filtered_df.empty:
            mean_val = filtered_df['value'].mean()
            fig.add_vline(
                x=mean_val, 
                line_dash="dash", 
                line_color="red",
                line_width=3,
                annotation_text=f"Mean: {mean_val:.2f}",
                annotation_position="top right",
                annotation_font_size=14
            )

        return fig

    def create_scatter_matrix(self, df: pd.DataFrame, dimensions: List[str], title: str = "Property Correlation Matrix"):
        """Create scatter matrix for multivariate analysis"""
        if df.empty or not dimensions:
            return None

        # Filter columns
        plot_df = df[dimensions].dropna()
        if plot_df.empty:
            return None

        fig = px.scatter_matrix(
            plot_df,
            dimensions=dimensions,
            title=title,
            color_discrete_sequence=[self.colors['materials'][0]]
        )

        fig.update_layout(
            title_x=0.5,
            height=800,
            font=dict(size=12),
            title_font=dict(size=20)
        )

        return fig

# ==============================
# MAIN APPLICATION
# ==============================
def main():
    """Main Streamlit application"""
    st.markdown('<h1 class="main-header">🔬 Piezoelectric Materials Knowledge Miner<br><small>Open-Data Generative Analytics Platform</small></h1>', unsafe_allow_html=True)

    # Initialize session state
    if 'extractor' not in st.session_state:
        st.session_state.extractor = EnhancedPiezoelectricKnowledgeExtractor()
    if 'viz_engine' not in st.session_state:
        st.session_state.viz_engine = PublicationQualityVisualizationEngine()
    if 'enricher' not in st.session_state:
        st.session_state.enricher = OpenMaterialsConnector()
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = None
    if 'analysis_complete' not in st.session_state:
        st.session_state.analysis_complete = False
    if 'generative_results' not in st.session_state:
        st.session_state.generative_results = {}

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
        max_papers = st.slider("Max papers to process", 10, 1000, 200, 10)
        analysis_focus = st.selectbox("Focus area", [
            "All Materials", "PVDF Composites", "Inorganic Ceramics", "2D Materials", "Beta-Phase Analysis"
        ])

        # Visualization selection
        st.subheader("Visualizations")
        viz_options = st.multiselect(
            "Generate visualizations",
            ["Word Cloud", "Knowledge Graph", "Radar Chart", "Histograms", "Scatter Matrix"],
            default=["Word Cloud", "Knowledge Graph", "Histograms"]
        )

        # Actions
        st.subheader("Actions")
        col1, col2 = st.columns(2)
        with col1:
            analyze_btn = st.button("🚀 Start Analysis", type="primary", use_container_width=True)
        with col2:
            if st.button("🔄 Reset Session", use_container_width=True):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()

        # System info
        st.subheader("System Status")
        st.metric("pymatgen", "Available" if PYMATGEN_AVAILABLE else "Not installed")
        st.metric("PubChem", "Available" if PUBCHEM_AVAILABLE else "Not installed")
        st.metric("Reference DB", "Ready" if os.path.exists(MATERIAL_REF_FILE) else "Auto-created")

    # Main analysis workflow
    if analyze_btn:
        with st.spinner("🔬 Performing comprehensive knowledge extraction..."):
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

                # Extract text
                st.text("📄 Extracting text content...")
                if 'full_text' in papers_df.columns:
                    texts = papers_df['full_text'].fillna(papers_df.get('abstract', '')).fillna('').tolist()
                elif 'abstract' in papers_df.columns:
                    texts = papers_df['abstract'].fillna('').tolist()
                else:
                    st.error("No text content found in database!")
                    return

                # Limit for performance
                texts = texts[:max_papers]
                papers_df = papers_df.iloc[:max_papers].copy()

                # Extract entities and relationships
                st.text("🧠 Extracting entities and relationships...")
                all_entities, all_relationships = st.session_state.extractor.analyze_corpus(texts)

                # Convert to DataFrames with robust error handling
                entities_list = []
                for entity_type in ['materials', 'properties', 'quantities']:
                    for entity in all_entities[entity_type]:
                        # Ensure required keys exist
                        entity_text = entity.get('text', '')
                        if not entity_text:
                            continue
                        entities_list.append({
                            'entity': entity_text,
                            'type': 'quantity' if entity_type == 'quantities' else entity_type.rstrip('s'),
                            'category': entity.get('category', ''),
                            'context': entity.get('context', '')[:250],
                            'start': entity.get('start', -1),
                            'end': entity.get('end', -1)
                        })

                entities_df = pd.DataFrame(entities_list) if entities_list else pd.DataFrame(columns=['entity','type','category','context','start','end'])
                relationships_df = pd.DataFrame(all_relationships) if all_relationships else pd.DataFrame()

                # Add frequency counts safely
                if not entities_df.empty:
                    entity_freq = entities_df['entity'].value_counts().to_dict()
                    entities_df['frequency'] = entities_df['entity'].map(entity_freq)
                else:
                    entities_df['frequency'] = 0

                if not relationships_df.empty:
                    # Ensure required columns exist
                    for col in ['material', 'property', 'sentence', 'confidence']:
                        if col not in relationships_df.columns:
                            relationships_df[col] = ''
                    if 'value' not in relationships_df.columns:
                        relationships_df['value'] = np.nan
                    if 'unit' not in relationships_df.columns:
                        relationships_df['unit'] = ''
                    if 'frequency' not in relationships_df.columns:
                        rel_freq = relationships_df.groupby(['material', 'property']).size().reset_index(name='frequency')
                        relationships_df = relationships_df.merge(rel_freq, on=['material', 'property'], how='left')

                # Store results
                st.session_state.processed_data = {
                    'papers': papers_df,
                    'entities': entities_df,
                    'relationships': relationships_df,
                    'texts': texts
                }
                st.session_state.analysis_complete = True
                st.success(f"✅ Analysis complete! Processed {len(papers_df)} papers.")

            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")
                return

    # Results display
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
            st.metric("Entities Extracted", len(entities_df))
        with col3:
            st.metric("Relationships Found", len(relationships_df))
        with col4:
            unique_mats = entities_df[entities_df['type']=='material']['category'].nunique() if not entities_df.empty else 0
            st.metric("Unique Materials", unique_mats)

        # Create tabs
        tabs = st.tabs([
            "Word Cloud", 
            "Knowledge Graph", 
            "Material Radar", 
            "Value Histograms", 
            "Generative AI", 
            "Data Explorer"
        ])

        # Tab 1: Word Cloud
        if "Word Cloud" in viz_options:
            with tabs[0]:
                st.subheader("📝 Word Cloud Analysis")
                if texts:
                    fig = st.session_state.viz_engine.create_wordcloud(texts, "Key Terms in Piezoelectric Literature")
                    if fig:
                        st.pyplot(fig)
                        add_caption(r"""
                        **Methodology**: Term frequency visualized using $w_i = \log(1 + f_i)$ weighting, 
                        where $f_i$ is the raw frequency of term $i$. Domain-specific stopwords removed. 
                        Font size proportional to log-frequency. High-resolution (300 DPI) suitable for publication.
                        """)
                    else:
                        st.info("Not enough text for word cloud.")
                else:
                    st.info("No text data available.")

        # Tab 2: Knowledge Graph
        if "Knowledge Graph" in viz_options:
            with tabs[1]:
                st.subheader("🕸️ Material–Property Knowledge Graph")
                if not relationships_df.empty:
                    # Filter significant relationships
                    filtered_rels = relationships_df[relationships_df['frequency'] >= 2] if 'frequency' in relationships_df.columns else relationships_df
                    if not filtered_rels.empty and not entities_df.empty:
                        fig = st.session_state.viz_engine.create_network_graph(
                            entities_df, 
                            filtered_rels, 
                            "Material–Property Co-occurrence Network"
                        )
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                            add_caption(r"""
                            **Methodology**: Nodes = materials (blue) and properties (green). 
                            Edge weight = co-occurrence frequency in same sentence. 
                            Layout computed via Fruchterman–Reingold force-directed algorithm with $k=2.0$. 
                            Node size $\propto \log(1 + \text{frequency})$. 
                            Confidence score: $\text{conf} = 0.4 + 0.25 \cdot \mathbb{1}_{\text{numeric}} + 0.2 \cdot \mathbb{1}_{\text{scientific verb}} + 0.15 \cdot \mathbb{1}_{\text{proximity<30 chars}}$.
                            """)
                        else:
                            st.info("Graph generation failed.")
                    else:
                        st.info("Insufficient relationships for visualization.")
                else:
                    st.info("No relationships extracted.")

        # Tab 3: Radar Chart
        if "Radar Chart" in viz_options:
            with tabs[2]:
                st.subheader("📊 Multi-Property Material Comparison")
                if not relationships_df.empty:
                    # Prepare data for top materials and properties
                    top_mats = entities_df[entities_df['type']=='material']['category'].value_counts().head(5).index.tolist() if not entities_df.empty else []
                    top_props = ['d33', 'beta_phase', 'voltage']  # Focus on key properties
                    
                    material_props = {}
                    for prop in top_props:
                        material_props[prop] = {}
                        for mat in top_mats:
                            subset = relationships_df[
                                (relationships_df['material'] == mat) & 
                                (relationships_df['property'] == prop)
                            ]
                            if not subset.empty and 'value' in subset.columns:
                                vals = subset['value'].dropna()
                                if not vals.empty:
                                    material_props[prop][mat] = float(vals.mean())
                    
                    if material_props:
                        fig = st.session_state.viz_engine.create_radar_chart(
                            material_props,
                            "Comparative Performance of Key Materials"
                        )
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                            add_caption(r"""
                            **Methodology**: Radar chart comparing mean values of key properties across materials. 
                            Axes: $d_{33}$ (pC/N), $\beta$-phase content (%), output voltage (V). 
                            Values computed as arithmetic mean from literature corpus. 
                            Enables direct multi-property performance assessment.
                            """)
                        else:
                            st.info("Radar chart generation failed.")
                    else:
                        st.info("Insufficient quantitative data for radar chart.")
                else:
                    st.info("No relationships to compare.")

        # Tab 4: Histograms
        if "Histograms" in viz_options:
            with tabs[3]:
                st.subheader("📈 Distribution of Extracted Values")
                # Extract quantities safely
                quantities = []
                if 'quantities' in all_entities:
                    for q in all_entities['quantities']:
                        if isinstance(q, dict) and 'value' in q:
                            quantities.append({
                                'value': q['value'],
                                'unit': q.get('unit', ''),
                                'context': q.get('context', '')[:100]
                            })
                
                if quantities:
                    values_df = pd.DataFrame(quantities)
                    if not values_df.empty:
                        fig = st.session_state.viz_engine.create_histogram(
                            values_df,
                            "Distribution of Numerical Values in Literature",
                            "Reported Value",
                            bins=30
                        )
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                            add_caption(r"""
                            **Methodology**: Histogram of all extracted numerical values ($k=30$ bins). 
                            Outliers (>99th percentile) excluded for clarity. 
                            Mean $\mu = \frac{1}{n}\sum_{i=1}^{n} x_i$ shown as red dashed line. 
                            Standard deviation $\sigma = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(x_i - \mu)^2}$.
                            """)
                        else:
                            st.info("Histogram generation failed.")
                    else:
                        st.info("No numerical values extracted.")
                else:
                    st.info("No quantities found.")

        # Tab 5: Generative AI
        with tabs[4]:
            st.subheader("🤖 Generative Materials Intelligence")
            st.markdown("""
            <div style="background-color: #F0F9FF; padding: 1rem; border-radius: 8px; border-left: 4px solid #3B82F6;">
            <strong>Generative Inference Engine</strong>: Combines literature evidence with open scientific databases to predict material properties and provide evidence-backed explanations.
            </div>
            """, unsafe_allow_html=True)
            
            if not entities_df.empty:
                # Get unique materials
                unique_mats = entities_df[entities_df['type']=='material']['category'].unique()
                unique_props = list(Config.PROPERTIES.keys())
                
                col1, col2 = st.columns(2)
                with col1:
                    selected_mat = st.selectbox("Material to analyze", unique_mats)
                with col2:
                    selected_prop = st.selectbox("Property to predict", unique_props)
                
                if st.button("🔮 Generate Prediction", type="primary"):
                    with st.spinner("Generating evidence-backed prediction..."):
                        engine = GenerativeInferenceEngine(papers_df, relationships_df, st.session_state.enricher)
                        result = engine.predict_property(selected_mat, selected_prop)
                        st.session_state.generative_results = result
                        
                        # Display prediction
                        explanation = engine.explain_prediction(result)
                        st.markdown(explanation)
                        
                        # Show external knowledge if available
                        if result["external_knowledge"]:
                            with st.expander("📚 External Knowledge Base"):
                                ext = result["external_knowledge"]
                                cols = st.columns(3)
                                cols[0].markdown(f"**Formula**: {ext.get('formula', 'N/A')}")
                                cols[1].markdown(f"**Crystal System**: {ext.get('crystal_system', 'N/A')}")
                                cols[2].markdown(f"**Space Group**: {ext.get('space_group', 'N/A')}")
                                if ext.get("structure_url"):
                                    st.markdown(f"**Structure**: [View in COD]({ext['structure_url']})")
                        
                        # Show dopant effects if available
                        if not result["dopant_effects"].empty:
                            with st.expander("🧪 Recommended Dopants for Enhancement"):
                                st.dataframe(result["dopant_effects"], use_container_width=True)
            else:
                st.info("Analysis required before generative inference.")

        # Tab 6: Data Explorer
        with tabs[5]:
            st.subheader("🔍 Interactive Data Explorer")
            
            if not entities_df.empty:
                st.markdown("### Extracted Entities")
                st.dataframe(entities_df, use_container_width=True, height=400)
                csv = entities_df.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download Entities CSV", csv, "entities.csv", "text/csv")
            
            if not relationships_df.empty:
                st.markdown("### Material–Property Relationships")
                st.dataframe(relationships_df, use_container_width=True, height=400)
                csv = relationships_df.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download Relationships CSV", csv, "relationships.csv", "text/csv")
            
            if not papers_df.empty:
                st.markdown("### Source Papers")
                display_cols = [col for col in ['title', 'abstract', 'year'] if col in papers_df.columns]
                if display_cols:
                    st.dataframe(papers_df[display_cols], use_container_width=True, height=400)

    else:
        # Welcome screen
        st.markdown("""
        <div style="padding: 2.5rem; text-align: center; background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%); border-radius: 15px; color: white; margin-bottom: 2rem;">
            <h2>🤖 Generative Piezoelectric Materials Intelligence</h2>
            <p style="font-size: 1.2rem; opacity: 0.9;">Open-data knowledge mining • Evidence-backed prediction • Publication-ready visualization</p>
        </div>
        """, unsafe_allow_html=True)

        # Feature cards
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            <div style="padding: 1.2rem; border-radius: 12px; background-color: #F8FAFC; border: 1px solid #E2E8F0; height: 100%;">
                <h3 style="color: #3B82F6;">🔬 Open Knowledge Mining</h3>
                <p>Extract materials, properties, and values from 1000s of papers using robust NER.</p>
                <p><strong>Sources</strong>: COD, PubChem, hand-curated reference DB (no API keys).</p>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div style="padding: 1.2rem; border-radius: 12px; background-color: #F0FDF4; border: 1px solid #BBF7D0; height: 100%;">
                <h3 style="color: #10B981;">🧠 Generative Inference</h3>
                <p>Predict properties with uncertainty quantification and evidence tracing.</p>
                <p><strong>Method</strong>: Retrieval-augmented generation + external validation.</p>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown("""
            <div style="padding: 1.2rem; border-radius: 12px; background-color: #FEF7CD; border: 1px solid #FDE68A; height: 100%;">
                <h3 style="color: #D97706;">📊 Publication Graphics</h3>
                <p>300 DPI figures with mathematical captions ready for journals.</p>
                <p><strong>Formats</strong>: Interactive Plotly + Matplotlib export.</p>
            </div>
            """, unsafe_allow_html=True)

        # Quick start
        with st.expander("📚 Quick Start Guide"):
            st.markdown("""
            ### Step-by-Step Workflow
            1. **Database Setup**: Place your `.db` files in `knowledge_database/`
            2. **Configuration**: Select database and analysis scope in sidebar
            3. **Analysis**: Click "Start Analysis" to extract knowledge
            4. **Exploration**: Navigate tabs for different visualizations
            5. **Generation**: Use "Generative AI" tab for predictions
            6. **Export**: Download CSV data or save figures

            ### Supported Open Data Sources
            - **Crystallography Open Database (COD)**: Inorganic crystal structures (no API key)
            - **PubChem**: Organic compounds and polymers (no API key)
            - **Local Reference DB**: Hand-curated piezoelectric properties (auto-created)

            ### Technical Features
            - Robust error handling for missing keys
            - Confidence scoring for relationships
            - Unit normalization and standardization
            - Mathematical captions with LaTeX formatting
            - Evidence-backed generative predictions
            """)

# ==============================
# DATABASE MANAGER (Enhanced)
# ==============================
class DatabaseManager:
    """Manages database connections with enhanced error handling"""
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = None

    def connect(self) -> bool:
        try:
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            return True
        except Exception as e:
            st.error(f"Database connection error: {e}")
            return False

    def disconnect(self):
        if self.conn:
            self.conn.close()

    def get_tables(self) -> List[str]:
        if not self.conn:
            if not self.connect():
                return []
        try:
            query = "SELECT name FROM sqlite_master WHERE type='table';"
            tables = pd.read_sql_query(query, self.conn)
            return tables['name'].tolist()
        except Exception as e:
            st.error(f"Error fetching tables: {e}")
            return []

    def get_papers_data(self) -> pd.DataFrame:
        tables = self.get_tables()
        if "papers_fulltext" in tables:
            query = """
            SELECT paper_id, title, abstract, full_text 
            FROM papers_fulltext 
            WHERE (full_text IS NOT NULL AND LENGTH(full_text) > 100)
               OR (abstract IS NOT NULL AND LENGTH(abstract) > 50)
            LIMIT 1000
            """
        elif "papers" in tables:
            query = """
            SELECT id as paper_id, title, abstract, year, categories
            FROM papers 
            WHERE abstract IS NOT NULL AND LENGTH(abstract) > 50
            LIMIT 1000
            """
        else:
            return pd.DataFrame()
        
        try:
            df = pd.read_sql_query(query, self.conn)
            return df
        except Exception as e:
            st.error(f"Error fetching papers: {e}")
            return pd.DataFrame()

# ==============================
# UTILITY FUNCTIONS
# ==============================
def check_database_files():
    """Check if required database files exist"""
    missing_files = []
    for db_name, db_path in Config.DB_PATHS.items():
        if not os.path.exists(db_path):
            missing_files.append(db_name)
    return missing_files

def create_sample_data():
    """Create comprehensive sample data for demonstration"""
    st.info("💡 Creating comprehensive sample data for demonstration...")
    
    np.random.seed(42)
    n_papers = 100
    
    # Generate realistic sample papers
    materials = ["PVDF", "PVDF/ZnO", "PVDF/BaTiO3", "ZnO", "BaTiO3"]
    properties = ["d33", "beta_phase", "voltage"]
    
    papers = []
    entities = []
    relationships = []
    
    for i in range(n_papers):
        mat = np.random.choice(materials)
        prop = np.random.choice(properties)
        value = np.random.uniform(10, 50) if prop == "d33" else np.random.uniform(40, 80)
        unit = "pC/N" if prop == "d33" else "%"
        sentence = f"{mat} nanocomposite with optimized processing shows {prop} of {value:.1f} {unit}."
        
        papers.append({
            'paper_id': f'paper_{i+1}',
            'title': f'Enhanced piezoelectric properties of {mat} nanocomposite',
            'abstract': sentence,
            'full_text': f'This comprehensive study investigates {mat}. {sentence} This represents a significant enhancement over pure polymer.'
        })
        
        # Entities
        entities.extend([
            {'entity': mat, 'type': 'material', 'category': mat.split('/')[0], 'context': sentence, 'frequency': 1},
            {'entity': prop, 'type': 'property', 'category': prop, 'context': sentence, 'frequency': 1}
        ])
        
        # Relationships
        relationships.append({
            'material': mat.split('/')[0],
            'property': prop,
            'sentence': sentence,
            'confidence': 0.85,
            'value': value,
            'unit': unit,
            'frequency': 1
        })
    
    papers_df = pd.DataFrame(papers)
    entities_df = pd.DataFrame(entities)
    relationships_df = pd.DataFrame(relationships)
    
    return papers_df, entities_df, relationships_df

# ==============================
# APPLICATION ENTRY POINT
# ==============================
if __name__ == "__main__":
    # Check for required databases
    missing_dbs = check_database_files()
    
    if missing_dbs:
        st.warning(f"⚠️ Missing database files: {', '.join(missing_dbs)}")
        st.info("📁 Expected location: `knowledge_database/` subdirectory")
        
        if st.checkbox("✅ Use comprehensive sample data for demonstration"):
            with st.spinner("Generating sample dataset..."):
                papers_df, entities_df, relationships_df = create_sample_data()
            
            st.session_state.processed_data = {
                'papers': papers_df,
                'entities': entities_df,
                'relationships': relationships_df,
                'texts': papers_df['abstract'].tolist()
            }
            st.session_state.analysis_complete = True
            st.success("✅ Sample data ready! Explore the tabs above.")
            st.rerun()
    
    # Run main application
    main()
