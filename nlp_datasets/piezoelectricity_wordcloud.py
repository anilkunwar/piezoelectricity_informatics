# streamlit_app.py — FOCUSED WORDCLOUD MODE
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

# Setup
logging.basicConfig(level=logging.INFO)
st.set_page_config(page_title="Piezoelectric Word Cloud Miner", layout="wide")

st.markdown("""
<style>
.main-header { font-size: 2.2rem; color: #1E3A8A; text-align: center; margin-bottom: 1.5rem; }
.figure-caption { font-size: 0.95rem; color: #4B5563; margin-top: 0.25rem; margin-bottom: 1.5rem; font-style: italic; line-height: 1.4; }
</style>
""", unsafe_allow_html=True)

def add_caption(text: str):
    st.markdown(f'<div class="figure-caption">{text}</div>', unsafe_allow_html=True)

# Helper: get DB paths by query type
def get_db_paths_for_query(query_type: str) -> dict:
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "knowledge_database")
    return {
        "Metadata DB": os.path.join(base_dir, f"piezoelectricity{query_type}_metadata.db"),
        "Universe DB": os.path.join(base_dir, f"piezoelectricity{query_type}_universe.db"),
        "PDF Storage DB": os.path.join(base_dir, f"piezoelectricity{query_type}_pdfs.db")
    }

# Helper: extract terms for word cloud (unigrams + bigrams)
def extract_terms_for_wordcloud(texts: list, ngram_range=(1, 2)) -> dict:
    combined = " ".join(str(t).lower() for t in texts if pd.notna(t))
    combined = re.sub(r'[^a-z0-9\s\-]', ' ', combined)
    words = [w for w in combined.split() if len(w) > 2 and not w.isdigit()]
    terms = []

    if ngram_range[0] <= 1:
        terms.extend(words)
    if ngram_range[1] >= 2:
        bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
        terms.extend(bigrams)

    return Counter(terms)

# Main app
def main():
    st.markdown('<h1 class="main-header">☁️ Piezoelectric Literature Word Cloud Miner</h1>', unsafe_allow_html=True)

    # Initialize session state
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
            # Load database
            try:
                conn = sqlite3.connect(db_path)
                # Auto-detect text column
                tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table';", conn)['name'].tolist()
                target_table = None
                text_col = None
                for table in tables:
                    cols = [row[1] for row in conn.execute(f"PRAGMA table_info({table})")]
                    for col in ['full_text', 'abstract', 'content', 'text']:
                        if col in cols:
                            target_table = table
                            text_col = col
                            break
                    if target_table:
                        break
                if not target_table:
                    st.error("No text column found in any table!")
                    return
                df = pd.read_sql(f"SELECT {text_col} FROM {target_table} WHERE {text_col} IS NOT NULL AND LENGTH({text_col}) > 100 LIMIT 1000", conn)
                conn.close()
                texts = df[text_col].fillna('').tolist()
                if not texts:
                    st.error("No valid text extracted!")
                    return
                st.session_state.wordcloud_terms = extract_terms_for_wordcloud(texts)
                st.success(f"✅ Extracted {len(st.session_state.wordcloud_terms)} unique terms from {len(texts)} documents.")
            except Exception as e:
                st.error(f"Failed to load DB: {e}")

    # Main content: Word Cloud only
    st.subheader("☁️ Publication-Quality Word Cloud")

    if not st.session_state.wordcloud_terms:
        st.info("👈 Select a query dataset and click **Load & Extract Terms** to begin.")
        return

    term_counts = st.session_state.wordcloud_terms

    # Controls
    col1, col2 = st.columns([1, 2])
    with col1:
        top_n = st.slider("Top N Terms", 10, 500, 100, 10)
        ngram_choice = st.selectbox("N-gram Range", ["Unigrams Only", "Bigrams Only", "Unigrams + Bigrams"])
        custom_stop = st.text_area(
            "Exclude Terms (comma-separated)",
            value="using,used,study,result,show,figure,table,high,low,obtained,reported,demonstrated,exhibited"
        )
        stopwords_set = set(w.strip().lower() for w in custom_stop.split(",") if w.strip())

    # Filter terms
    ngram_filter = {
        "Unigrams Only": lambda t: " " not in t,
        "Bigrams Only": lambda t: " " in t,
        "Unigrams + Bigrams": lambda t: True
    }[ngram_choice]

    filtered = {
        term: count for term, count in term_counts.items()
        if term not in stopwords_set and ngram_filter(term)
    }
    top_terms = dict(Counter(filtered).most_common(top_n))

    if not top_terms:
        st.warning("No terms remain after filtering.")
        return

    # Generate word cloud
    wordcloud = WordCloud(
        width=2000,
        height=1000,
        background_color='white',
        max_words=top_n,
        colormap='viridis',
        collocations=False,
        relative_scaling=0.5,
        regexp=r"\w[\w\ ]+"
    ).generate_from_frequencies(top_terms)

    # High-res figure
    fig, ax = plt.subplots(figsize=(20, 10), dpi=300)
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.set_title(
        f"Top {len(top_terms)} Key Terms in Piezoelectric Literature ({st.session_state.query_type.upper()})",
        fontsize=28, fontweight='bold', pad=30, fontfamily='serif'
    )
    ax.axis('off')
    plt.tight_layout(pad=2.0)
    st.pyplot(fig, use_container_width=False)

    add_caption(r"""
    **Methodology**: Term frequency $f_i$ visualized with $\text{size} \propto \log(1 + f_i)$.
    Unigrams and/or bigrams extracted via regex. Domain noise terms excluded.
    Font: Serif. Resolution: 300 DPI. Suitable for journal submission.
    """)

    # Download
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    st.download_button("📥 Download High-Res Word Cloud", buf.getvalue(), "wordcloud.png", "image/png")

if __name__ == "__main__":
    main()
