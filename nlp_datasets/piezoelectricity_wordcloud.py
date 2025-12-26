# streamlit_app.py — CATEGORICAL WORD CLOUD (PVDF / Dopants / Properties)
import streamlit as st
import pandas as pd
import sqlite3
import os
import re
import io
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import logging

logging.basicConfig(level=logging.INFO)
st.set_page_config(page_title="Categorical Piezoelectric Word Cloud", layout="wide")

st.markdown("""
<style>
.main-header { font-size: 2.2rem; color: #1E3A8A; text-align: center; margin-bottom: 1.5rem; }
.figure-caption { font-size: 0.95rem; color: #4B5563; margin-top: 0.25rem; margin-bottom: 1.5rem; font-style: italic; line-height: 1.4; }
</style>
""", unsafe_allow_html=True)

def add_caption(text: str):
    st.markdown(f'<div class="figure-caption">{text}</div>', unsafe_allow_html=True)

# === CATEGORICAL LEXICONS ===
PVDF_TERMS = {
    'pvdf', 'polyvinylidene', 'poly(vinylidene', 'polyvinylidenefluoride', 'polyvinylidene fluoride',
    'polymer', 'copolymer', 'homopolymer', 'flexible', 'film', 'electrospun', 'nanofiber',
    'β-phase', 'beta phase', 'beta-phase', 'alpha phase', 'gamma phase', 'crystalline',
    'amorphous', 'poling', 'stretching', 'annealing', 'quenching', 'solution casting',
    'melt processing', 'phase transformation', 'dipole', 'ferroelectric', 'piezopolymer'
}

DOPANT_TERMS = {
    # Common fillers/dopants in PVDF composites
    'zno', 'tio2', 'sno2', 'batio3', 'cnt', 'carbon nanotube', 'graphene', 'rgo',
    'mxene', 'bnt', 'bt', 'pzt', 'aln', 'mgo', 'sro', 'caco3', 'sic', 'bn', 'mofs',
    'cellulose', 'clay', 'talc', 'nanoclay', 'tio', 'zro2', 'fe3o4', 'nio', 'cofe2o4',
    'ba0.85ca0.15zr0.1ti0.9o3', 'bczt', 'pmn-pt', 'knn', 'linbo3', 'nbt', 'bt-bzt'
}

PROPERTY_TERMS = {
    'd33', 'd31', 'g33', 'voltage', 'current', 'power', 'energy', 'density', 'output',
    'dielectric', 'permittivity', 'capacitance', 'impedance', 'resistance', 'conductivity',
    'young', 'modulus', 'stiffness', 'elastic', 'tensile', 'strength', 'strain',
    'curie', 'temperature', 'tc', 'coercive', 'remanent', 'polarization', 'hysteresis',
    'electromechanical', 'coupling', 'quality factor', 'mechanical', 'loss', 'tan delta',
    'bandgap', 'band gap', 'crystallinity', 'crystalline', 'beta content', 'phase fraction'
}

STOPWORDS = {
    'using', 'used', 'study', 'result', 'show', 'figure', 'table', 'high', 'low',
    'obtained', 'reported', 'demonstrated', 'exhibited', 'fabricated', 'prepared',
    'investigated', 'characterized', 'measured', 'synthesized', 'paper', 'method',
    'based', 'respectively', 'within', 'between', 'under', 'via', 'through', 'during',
    'after', 'before', 'from', 'into', 'over', 'with', 'without', 'than', 'that',
    'which', 'this', 'these', 'those', 'also', 'further', 'more', 'most', 'such',
    'well', 'very', 'much', 'many', 'each', 'every', 'both', 'either', 'neither'
}

# Normalize all lexicons to lowercase
PVDF_TERMS = {t.lower() for t in PVDF_TERMS}
DOPANT_TERMS = {t.lower() for t in DOPANT_TERMS}
PROPERTY_TERMS = {t.lower() for t in PROPERTY_TERMS}
STOPWORDS = {t.lower() for t in STOPWORDS}

# Color mapping
CATEGORY_COLORS = {
    'pvdf': '#3B82F6',      # Blue
    'dopant': '#EF4444',    # Red
    'property': '#10B981',  # Green
    'other': '#6B7280'      # Gray
}

def get_db_paths_for_query(query_type: str) -> dict:
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "knowledge_database")
    return {
        "Metadata DB": os.path.join(base_dir, f"piezoelectricity{query_type}_metadata.db"),
        "Universe DB": os.path.join(base_dir, f"piezoelectricity{query_type}_universe.db"),
        "PDF Storage DB": os.path.join(base_dir, f"piezoelectricity{query_type}_pdfs.db")
    }

def extract_terms_for_wordcloud(texts: list) -> Counter:
    combined = " ".join(str(t).lower() for t in texts if pd.notna(t))
    combined = re.sub(r'[^a-z0-9\s\-]', ' ', combined)
    words = [w for w in combined.split() if len(w) > 2 and not w.isdigit() and w not in STOPWORDS]
    terms = []

    # Unigrams
    terms.extend(words)
    # Bigrams
    bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
    terms.extend(bigrams)

    return Counter(terms)

def categorize_term(term: str) -> str:
    t = term.lower()
    if t in PVDF_TERMS:
        return 'pvdf'
    if t in DOPANT_TERMS:
        return 'dopant'
    if t in PROPERTY_TERMS:
        return 'property'
    return 'other'

def main():
    st.markdown('<h1 class="main-header">🌈 Categorical Word Cloud: PVDF Composites</h1>', unsafe_allow_html=True)

    if 'wordcloud_terms' not in st.session_state:
        st.session_state.wordcloud_terms = None
    if 'query_type' not in st.session_state:
        st.session_state.query_type = "q1"

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        query_type = st.selectbox("Select Query Dataset", ["q1", "q2", "q3"], key="query_selector")
        st.session_state.query_type = query_type

        db_paths = get_db_paths_for_query(query_type)
        available_dbs = [name for name, path in db_paths.items() if os.path.exists(path)]
        if not available_dbs:
            st.error(f"No databases found for `{query_type}`!")
            st.info("Expected files:\n"
                    f"- piezoelectricity{query_type}_metadata.db\n"
                    f"- piezoelectricity{query_type}_universe.db\n"
                    f"- piezoelectricity{query_type}_pdfs.db")
            return

        selected_db_name = st.radio("Available Databases", available_dbs)
        db_path = db_paths[selected_db_name]

        if st.button("🚀 Load & Extract Terms", type="primary"):
            try:
                conn = sqlite3.connect(db_path)
                tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table';", conn)['name'].tolist()
                target_table = text_col = None
                for table in tables:
                    cols = [row[1] for row in conn.execute(f"PRAGMA table_info({table})")]
                    for col in ['full_text', 'abstract', 'content', 'text']:
                        if col in cols:
                            target_table, text_col = table, col
                            break
                    if target_table: break
                if not target_table:
                    st.error("No text column found!")
                    return
                df = pd.read_sql(f"SELECT {text_col} FROM {target_table} WHERE {text_col} IS NOT NULL AND LENGTH({text_col}) > 100 LIMIT 1000", conn)
                conn.close()
                texts = df[text_col].fillna('').tolist()
                if not texts:
                    st.error("No valid text!")
                    return
                st.session_state.wordcloud_terms = extract_terms_for_wordcloud(texts)
                st.success(f"✅ Extracted {len(st.session_state.wordcloud_terms)} unique terms.")
            except Exception as e:
                st.error(f"DB error: {e}")

    # Main content
    st.subheader("🌈 Categorical Word Cloud (PVDF / Dopants / Properties)")

    if not st.session_state.wordcloud_terms:
        st.info("👈 Select a query dataset and click **Load & Extract Terms**.")
        return

    term_counts = st.session_state.wordcloud_terms

    # Category filters
    col1, col2 = st.columns([1, 2])
    with col1:
        top_n = st.slider("Top N Terms", 10, 500, 100, 10)
        show_pvdf = st.checkbox("Show PVDF Terms", True)
        show_dopants = st.checkbox("Show Dopants", True)
        show_properties = st.checkbox("Show Properties", True)
        show_other = st.checkbox("Show Other Terms", False)
        custom_stop = st.text_area("Exclude Terms (comma-separated)", 
                                   value="using,used,study,result,show,figure,table,high,low,obtained")
        custom_stop_set = set(w.strip().lower() for w in custom_stop.split(",") if w.strip())

    # Build category-to-color map for included terms
    filtered_terms = {}
    for term, count in term_counts.items():
        if term in custom_stop_set:
            continue
        cat = categorize_term(term)
        if (cat == 'pvdf' and show_pvdf) or \
           (cat == 'dopant' and show_dopants) or \
           (cat == 'property' and show_properties) or \
           (cat == 'other' and show_other):
            filtered_terms[term] = count

    top_terms = dict(Counter(filtered_terms).most_common(top_n))
    if not top_terms:
        st.warning("No terms remain after filtering.")
        return

    # Build color function
    def color_func(word, **kwargs):
        cat = categorize_term(word)
        if cat == 'pvdf' and show_pvdf:
            return CATEGORY_COLORS['pvdf']
        elif cat == 'dopant' and show_dopants:
            return CATEGORY_COLORS['dopant']
        elif cat == 'property' and show_properties:
            return CATEGORY_COLORS['property']
        elif cat == 'other' and show_other:
            return CATEGORY_COLORS['other']
        return CATEGORY_COLORS['other']  # fallback

    # Generate word cloud
    wordcloud = WordCloud(
        width=2000,
        height=1000,
        background_color='white',
        max_words=top_n,
        color_func=color_func,
        collocations=False,
        relative_scaling=0.5,
        regexp=r"\w[\w\ ]+"
    ).generate_from_frequencies(top_terms)

    # High-res figure
    fig, ax = plt.subplots(figsize=(20, 10), dpi=300)
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.set_title(
        f"Categorical Word Cloud ({st.session_state.query_type.upper()})",
        fontsize=28, fontweight='bold', pad=30, fontfamily='serif'
    )
    ax.axis('off')
    plt.tight_layout(pad=2.0)
    st.pyplot(fig, use_container_width=False)

    # Legend (simulated via markdown)
    legend_items = []
    if show_pvdf:
        legend_items.append(f'<span style="color:{CATEGORY_COLORS["pvdf"]}">■</span> PVDF/Polymer Terms')
    if show_dopants:
        legend_items.append(f'<span style="color:{CATEGORY_COLORS["dopant"]}">■</span> Dopants/Fillers')
    if show_properties:
        legend_items.append(f'<span style="color:{CATEGORY_COLORS["property"]}">■</span> Physicochemical Properties')
    if show_other:
        legend_items.append(f'<span style="color:{CATEGORY_COLORS["other"]}">■</span> Other Terms')
    st.markdown(" &nbsp; ".join(legend_items), unsafe_allow_html=True)

    add_caption(r"""
    **Methodology**: Terms categorized using domain-specific lexicons.
    - **Blue**: PVDF polymer, processing, phases
    - **Red**: Dopants (ZnO, BaTiO₃, CNT, etc.)
    - **Green**: Properties (d₃₃, voltage, β-phase, etc.)
    Font: Serif. Resolution: 300 DPI. LaTeX-style caption.
    """)

    # Download
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    st.download_button("📥 Download High-Res Word Cloud", buf.getvalue(), "categorical_wordcloud.png", "image/png")

if __name__ == "__main__":
    main()
