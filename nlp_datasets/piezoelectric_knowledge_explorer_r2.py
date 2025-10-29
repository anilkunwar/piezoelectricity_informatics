# --------------------------------------------------------------
#  Piezoelectricity in PVDF – PRODUCTION VERSION
#  • No rerun on download • Individual PDFs • Persistent results
# --------------------------------------------------------------
import arxiv
import fitz
import pandas as pd
import streamlit as st
import os
import re
import sqlite3
from datetime import datetime
import logging
import time
import random
from pathlib import Path
import zipfile
import gc
import psutil
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import concurrent.futures
import json

from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from tenacity import retry, stop_after_attempt, wait_fixed

# -------------------------- CONFIG --------------------------
if os.path.exists("/tmp"):
    DB_DIR = "/tmp"
else:
    DB_DIR = os.path.join(os.path.expanduser("~"), "Desktop")

os.makedirs(DB_DIR, exist_ok=True)
pdf_dir = os.path.join(DB_DIR, "pdfs")
os.makedirs(pdf_dir, exist_ok=True)

METADATA_DB = os.path.join(DB_DIR, "piezoelectricity_metadata.db")
UNIVERSE_DB = os.path.join(DB_DIR, "piezoelectricity_universe.db")

logging.basicConfig(
    filename=os.path.join(DB_DIR, "piezoelectricity_query.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# -------------------------- INITIALIZE SESSION STATE --------------------------
if "log_buffer" not in st.session_state:
    st.session_state.log_buffer = []
if "search_results" not in st.session_state:
    st.session_state.search_results = None
if "relevant_papers" not in st.session_state:
    st.session_state.relevant_papers = None
if "download_paths" not in st.session_state:
    st.session_state.download_paths = []
if "zip_path" not in st.session_state:
    st.session_state.zip_path = None
if "metadata_saved" not in st.session_state:
    st.session_state.metadata_saved = False

def update_log(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = f"[{ts}] {msg}"
    st.session_state.log_buffer.append(entry)
    if len(st.session_state.log_buffer) > 50:
        st.session_state.log_buffer.pop(0)
    logging.info(msg)

# -------------------------- SYSTEM HEALTH --------------------------
def mem_usage():
    try:
        return psutil.Process().memory_info().rss / 1024 / 1024
    except:
        return 0

def health_check():
    mem = mem_usage()
    free_gb = psutil.disk_usage(DB_DIR).free / (1024**3)
    update_log(f"RAM {mem:.1f} MB | Disk free {free_gb:.1f} GB")
    if mem > 1500:
        st.warning(f"High RAM ({mem:.1f} MB)")
    if free_gb < 0.5:
        st.error("Low disk space")
        return False
    return True

# -------------------------- HTTP RETRY --------------------------
def retry_session():
    s = requests.Session()
    r = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    s.mount("http://", HTTPAdapter(max_retries=r))
    s.mount("https://", HTTPAdapter(max_retries=r))
    return s

# -------------------------- SciBERT --------------------------
@st.cache_resource
def load_scibert():
    tok = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
    mdl = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased")
    mdl.eval()
    update_log("SciBERT loaded")
    return tok, mdl

scibert_tok, scibert_mdl = load_scibert()

# -------------------------- NORMALISATION & PATTERNS --------------------------
def norm(txt: str) -> str:
    greek = {"α": "alpha", "β": "beta", "γ": "gamma", "δ": "delta", "ε": "epsilon",
             "Α": "alpha", "Β": "beta", "Γ": "gamma", "Δ": "delta", "Ε": "epsilon"}
    subs = {"₀":"0","₁":"1","₂":"2","₃":"3","₄":"4","₅":"5","₆":"6","₇":"7","₈":"8","₉":"9"}
    supers = {"⁰":"0","¹":"1","²":"2","³":"3","⁴":"4","⁵":"5","⁶":"6","⁷":"7","⁸":"8","⁹":"9"}
    for d, r in {**greek, **subs, **supers}.items():
        txt = txt.replace(d, r)
    return txt.lower()

KEY_PATTERNS = [
    r"\bpiezoelectric(?:ity| effect| performance| properties| coefficient| constant| polymer| materials)?\b",
    r"\belectrospun (?:nano)?fibers?|nanofiber mats|nanofibrous membranes?\b",
    r"\bpvdf|polyvinylidene fluoride|poly\s*\(?\s*vinylidene fluoride\s*\)?|pvd?f\b",
    r"\b(alpha|beta|gamma|delta|epsilon)\s*(?:phase|polymorph|crystal|crystals?|crystalline phase)\b",
    r"\befficiency|piezoelectric efficiency\b",
    r"\belectricity generation|electrical power generation|power output|voltage output\b",
    r"\bmechanical (?:force|stress|deformation|energy)\b",
    r"\bsno2|tin oxide|tin dioxide|stannic oxide\b",
    r"\bdopants?|doped|doping effects?\b",
    r"\bdoped pvdf\b",
    r"\b(?:beta|alpha|gamma|delta|epsilon|phase) fraction|phase content|fraction of phase\b",
    r"\benergy harvesting|nanogenerators?|scavenging mechanical energy\b",
    r"\bpolarization|ferroelectric polarization|pyroelectric\b",
    r"\bferroelectric(?:ity| properties)?\b",
    r"\bcurrent density\b",
    r"\bpower density\b",
    r"\bcrystallinity|semicrystalline\b",
    r"\bd33|d31|g33\b",
]

COMPILED = [re.compile(p, re.IGNORECASE) for p in KEY_PATTERNS]

def score_abstract(abstract: str) -> float:
    try:
        enc = scibert_tok(abstract, return_tensors="pt", truncation=True, max_length=512, padding=True)
        with torch.no_grad():
            out = scibert_mdl(**enc, output_attentions=True)
        n = sum(bool(p.search(norm(abstract))) for p in COMPILED)
        prob = np.sqrt(n) / np.sqrt(len(KEY_PATTERNS))

        toks = scibert_tok.convert_ids_to_tokens(enc["input_ids"][0])
        kw_idx = [i for i, t in enumerate(toks) if any(k in t.lower() for k in ["pvdf","piezo","phase","beta","alpha"])]
        if kw_idx:
            att = out.attentions[-1][0,0].cpu().numpy()
            boost = np.mean(att[kw_idx, :])
            if boost > 0.1:
                prob = min(prob + 0.2 * len(kw_idx)/len(toks), 1.0)
        return prob
    except Exception as e:
        update_log(f"SciBERT fail: {e}")
        n = sum(bool(p.search(norm(abstract))) for p in COMPILED)
        return np.sqrt(n) / np.sqrt(len(KEY_PATTERNS))

# -------------------------- PDF FUNCTIONS --------------------------
@st.cache_data
def pdf_text(path: str) -> str:
    try:
        doc = fitz.open(path)
        txt = "".join(p.get_text() for p in doc)
        doc.close()
        return txt
    except Exception as e:
        return f"Error: {e}"

@retry(stop=stop_after_attempt(4), wait=wait_fixed(2))
def _download(url: str, dest: Path) -> float:
    req = requests.Request("GET", url, headers={"User-Agent": "PiezoelectricityTool/1.0"})
    prep = req.prepare()
    s = retry_session()
    resp = s.send(prep, timeout=30)
    resp.raise_for_status()
    with open(dest, "wb") as f:
        f.write(resp.content)
    kb = len(resp.content) / 1024
    if kb < 1:
        raise RuntimeError("Empty file")
    return kb

def download_one(paper: dict) -> dict:
    pid = paper["id"]
    dest = Path(pdf_dir) / f"{pid}.pdf"
    try:
        kb = _download(paper["pdf_url"], dest)
        time.sleep(random.uniform(0.4, 0.9))
        txt = pdf_text(str(dest))
        if txt.startswith("Error"):
            raise RuntimeError(txt)
        
        # Update universe DB
        conn = sqlite3.connect(UNIVERSE_DB)
        c = conn.cursor()
        c.execute("""INSERT OR REPLACE INTO papers
                     (id, title, authors, year, content)
                     VALUES (?,?,?,?,?)""",
                  (pid, paper.get("title",""), paper.get("authors","Unknown"),
                   paper.get("year",0), txt))
        conn.commit()
        conn.close()
        
        paper.update({
            "download_status": f"Downloaded ({kb:.1f} KB)",
            "pdf_path": str(dest),
            "content": txt,
        })
        update_log(f"✓ {pid}: {kb:.1f} KB")
    except Exception as e:
        paper.update({
            "download_status": f"Failed: {str(e)[:50]}",
            "pdf_path": None,
            "content": f"Error: {e}",
        })
        if dest.exists() and dest.stat().st_size == 0:
            dest.unlink(missing_ok=True)
        update_log(f"✗ {pid}: {e}")
    return paper

# -------------------------- ARXIV QUERY --------------------------
@st.cache_data(ttl=3600)
def query_arxiv(_query: str, cats: list, max_res: int, sy: int, ey: int):
    client = arxiv.Client()
    search = arxiv.Search(query=_query, max_results=max_res,
                          sort_by=arxiv.SortCriterion.Relevance,
                          sort_order=arxiv.SortOrder.Descending)
    out = []
    qwords = {w.strip('"').lower() for w in _query.split("OR")}
    for r in client.results(search):
        if not (any(c in r.categories for c in cats) and sy <= r.published.year <= ey):
            continue
        matched = [w for w in qwords if w in r.summary.lower() or w in r.title.lower()]
        if not matched:
            continue
        rel = score_abstract(r.summary)
        out.append({
            "id": r.entry_id.split("/")[-1],
            "title": r.title,
            "authors": ", ".join(a.name for a in r.authors),
            "year": r.published.year,
            "categories": ", ".join(r.categories),
            "abstract": r.summary,
            "pdf_url": r.pdf_url,
            "download_status": "Pending",
            "matched_terms": ", ".join(matched),
            "relevance_prob": round(rel * 100, 2),
            "pdf_path": None,
            "content": None,
        })
        if len(out) >= max_res:
            break
    return sorted(out, key=lambda x: x["relevance_prob"], reverse=True)

# -------------------------- ZIP & METADATA --------------------------
def make_zip(paths: list) -> str:
    if not paths:
        return None
    zip_path = os.path.join(DB_DIR, "piezoelectricity_pdfs.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for p in paths:
            if p and os.path.exists(p):
                z.write(p, os.path.basename(p))
    return zip_path

def save_metadata(df: pd.DataFrame):
    conn = sqlite3.connect(METADATA_DB)
    # Ensure table exists
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS papers
                 (id TEXT PRIMARY KEY, title TEXT, authors TEXT, year INTEGER,
                  categories TEXT, abstract TEXT, pdf_url TEXT,
                  download_status TEXT, matched_terms TEXT,
                  relevance_prob REAL, pdf_path TEXT, content TEXT)""")
    df.to_sql("papers", conn, if_exists="replace", index=False)
    conn.commit()
    conn.close()
    st.session_state.metadata_saved = True

# -------------------------- UI --------------------------
st.set_page_config(page_title="Piezoelectricity in PVDF", layout="wide")
st.title("🔬 Piezoelectricity in PVDF Explorer")
st.markdown("**SciBERT-powered search** for PVDF piezoelectricity (α/β phases, dopants, electrospinning, efficiency…)")

# Fixed log area (single key)
log_placeholder = st.empty()

def show_logs():
    if st.session_state.log_buffer:
        log_placeholder.text_area(
            "📋 Processing Logs",
            "\n".join(st.session_state.log_buffer[-25:]),
            height=200,
            key="log_area_fixed"
        )
    else:
        log_placeholder.info("Ready to search…")

show_logs()

# -------------------------- SIDEBAR --------------------------
with st.sidebar:
    st.header("🔍 Search Parameters")
    default_query = ' OR '.join(f'"{t}"' for t in [
        "piezoelectricity", "PVDF", "beta phase", "electrospun nanofibers",
        "SnO2", "dopants", "efficiency", "nanogenerators"
    ])
    q = st.text_input("Query", value=default_query, key="query_input")
    
    cats = st.multiselect("Categories", 
                         ["cond-mat.mtrl-sci", "physics.app-ph", "physics.chem-ph"],
                         default=["cond-mat.mtrl-sci", "physics.app-ph"], 
                         key="cat_select")
    
    col1, col2 = st.columns(2)
    max_res = col1.slider("Max results", 5, 100, 30, key="max_results")
    sy = col2.number_input("Start year", 2000, 2025, 2015, key="start_year")
    
    ey = st.number_input("End year", sy, 2025, 2025, key="end_year")
    rel_thr = st.slider("🎯 Relevance threshold (%)", 0, 100, 30, key="rel_threshold")
    
    st.header("📤 Output")
    out_fmt = st.multiselect("Formats", ["SQLite", "CSV", "JSON"], default=["SQLite", "CSV"], key="output_formats")
    
    col3, col4 = st.columns(2)
    with col3:
        if st.button("🔍 Search arXiv", key="search_button", use_container_width=True):
            st.session_state.search_results = None  # Clear cache
            st.rerun()
    with col4:
        if st.button("🗑️ Clear Results", key="clear_button", use_container_width=True):
            for k in ["search_results", "relevant_papers", "download_paths", "zip_path", "metadata_saved"]:
                if k in st.session_state:
                    del st.session_state[k]
            st.success("Results cleared")
            st.rerun()

# -------------------------- MAIN LOGIC --------------------------
if st.session_state.search_results is None and st.button("🔍 Search arXiv", key="main_search_trigger"):
    if not q.strip() or not cats or sy > ey:
        st.error("❌ Please fix inputs")
    elif not health_check():
        st.stop()
    else:
        with st.spinner("🔍 Querying arXiv + SciBERT scoring…"):
            st.session_state.search_results = query_arxiv(q, cats, max_res, sy, ey)
        st.rerun()

# Show results if available
if st.session_state.search_results is not None:
    all_papers = st.session_state.search_results
    relevant = [p for p in all_papers if p["relevance_prob"] >= rel_thr]
    
    col1, col2, col3 = st.columns([1, 3, 1])
    col1.metric("📄 Total papers", len(all_papers))
    col2.metric("⭐ Relevant papers", len(relevant), f"≥ {rel_thr}%")
    col3.metric("💾 PDFs ready", len([p for p in relevant if p.get("pdf_path")]))
    
    if relevant:
        # Download PDFs if not done
        if not any(p.get("pdf_path") for p in relevant):
            with st.spinner(f"📥 Downloading {len(relevant)} PDFs…"):
                with concurrent.futures.ThreadPoolExecutor(max_workers=4) as exe:
                    futures = {exe.submit(download_one, p): i for i, p in enumerate(relevant)}
                    for fut in concurrent.futures.as_completed(futures):
                        idx = futures[fut]
                        relevant[idx] = fut.result()
                
                # Update session state
                paths = [p.get("pdf_path") for p in relevant if p.get("pdf_path")]
                st.session_state.relevant_papers = relevant
                st.session_state.download_paths = paths
                
                # Create ZIP
                zip_p = make_zip(paths)
                st.session_state.zip_path = zip_p
                
                # Save metadata
                if "SQLite" in out_fmt:
                    save_metadata(pd.DataFrame(relevant))
        
        # Display results table
        df = pd.DataFrame(relevant)
        st.subheader("📋 Results")
        
        # Add PDF download column
        pdf_col = []
        for _, row in df.iterrows():
            if row.get("pdf_path") and os.path.exists(row["pdf_path"]):
                with open(row["pdf_path"], "rb") as f:
                    pdf_col.append(st.download_button(
                        label="📄 PDF",
                        data=f.read(),
                        file_name=f"{row['id']}.pdf",
                        mime="application/pdf",
                        key=f"pdf_{row['id']}",
                        use_container_width=True
                    ))
            else:
                pdf_col.append("❌")
        
        df_display = df[["id", "title", "authors", "year", "relevance_prob", "download_status"]].copy()
        df_display["PDF"] = pdf_col
        
        st.dataframe(df_display, use_container_width=True, hide_index=True)
        
        # Bulk downloads
        st.subheader("💾 Bulk Downloads")
        col_bulk1, col_bulk2, col_bulk3 = st.columns(3)
        
        # ZIP
        if st.session_state.zip_path and os.path.exists(st.session_state.zip_path):
            with open(st.session_state.zip_path, "rb") as f:
                col_bulk1.download_button(
                    label="📦 All PDFs (ZIP)",
                    data=f.read(),
                    file_name="piezoelectricity_pdfs.zip",
                    mime="application/zip",
                    use_container_width=True
                )
        
        # CSV
        if "CSV" in out_fmt:
            csv_data = df.to_csv(index=False)
            col_bulk2.download_button(
                label="📊 CSV",
                data=csv_data,
                file_name="piezoelectricity_papers.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        # JSON
        if "JSON" in out_fmt:
            json_data = df.to_json(orient="records", indent=2)
            col_bulk3.download_button(
                label="🔍 JSON",
                data=json_data,
                file_name="piezoelectricity_papers.json",
                mime="application/json",
                use_container_width=True
            )
        
        # Database downloads
        if st.session_state.metadata_saved:
            st.subheader("🗄️ Database Files")
            col_db1, col_db2 = st.columns(2)
            with col_db1:
                if os.path.exists(METADATA_DB):
                    with open(METADATA_DB, "rb") as f:
                        st.download_button(
                            label="📋 Metadata DB",
                            data=f.read(),
                            file_name="piezoelectricity_metadata.db",
                            mime="application/octet-stream",
                            use_container_width=True
                        )
            with col_db2:
                if os.path.exists(UNIVERSE_DB):
                    with open(UNIVERSE_DB, "rb") as f:
                        st.download_button(
                            label="🌌 Full-text DB",
                            data=f.read(),
                            file_name="piezoelectricity_universe.db",
                            mime="application/octet-stream",
                            use_container_width=True
                        )
    
    else:
        st.warning(f"⚠️ No papers above {rel_thr}% relevance threshold")

show_logs()
