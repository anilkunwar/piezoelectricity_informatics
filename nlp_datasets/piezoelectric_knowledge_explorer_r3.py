# --------------------------------------------------------------
#  Piezoelectricity in PVDF – FINAL, PRODUCTION-READY
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

from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from tenacity import retry, stop_after_attempt, wait_fixed

# ========================= MUST BE FIRST =========================
# Guard against multiple calls (important for Streamlit Cloud)
if "page_config_set" not in st.session_state:
    st.set_page_config(page_title="Piezoelectricity in PVDF", layout="wide")
    st.session_state.page_config_set = True
# =================================================================

# -------------------------- CONFIG --------------------------
DB_DIR = "/tmp" if os.path.exists("/tmp") else os.path.join(os.path.expanduser("~"), "Desktop")
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

# -------------------------- SESSION STATE --------------------------
DEFAULT_STATE = {
    "log_buffer": [],
    "processing": False,
    "search_results": None,
    "relevant_papers": None,
    "pdf_paths": [],
    "zip_path": None,
}
for k, v in DEFAULT_STATE.items():
    if k not in st.session_state:
        st.session_state[k] = v

def update_log(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = f"[{ts}] {msg}"
    st.session_state.log_buffer.append(entry)
    if len(st.session_state.log_buffer) > 50:
        st.session_state.log_buffer.pop(0)
    logging.info(msg)

# -------------------------- HEALTH --------------------------
def health_check() -> bool:
    mem = psutil.Process().memory_info().rss / 1024 / 1024
    free_gb = psutil.disk_usage(DB_DIR).free / (1024**3)
    update_log(f"RAM {mem:.1f} MB | Disk free {free_gb:.1f} GB")
    if mem > 1500:
        st.warning(f"High RAM ({mem:.1f} MB)")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
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

# -------------------------- NORMALISATION --------------------------
def norm(txt: str) -> str:
    greek = {"α":"alpha","β":"beta","γ":"gamma","δ":"delta","ε":"epsilon",
             "Α":"alpha","Β":"beta","Γ":"gamma","Δ":"delta","Ε":"epsilon"}
    subs = {f"_{i}":str(i) for i in range(10)}
    supers = {f"^{i}":str(i) for i in range(10)}
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
        update_log(f"SciBERT error: {e}")
        n = sum(bool(p.search(norm(abstract))) for p in COMPILED)
        return np.sqrt(n) / np.sqrt(len(KEY_PATTERNS))

# -------------------------- PDF TEXT --------------------------
@st.cache_data
def pdf_text(path: str) -> str:
    try:
        doc = fitz.open(path)
        txt = "".join(p.get_text() for p in doc)
        doc.close()
        return txt
    except Exception as e:
        return f"Error: {e}"

# -------------------------- DATABASE --------------------------
def init_db(path: str):
    conn = sqlite3.connect(path)
    c = conn.cursor()
    if "universe" in path:
        c.execute("""CREATE TABLE IF NOT EXISTS papers
                     (id TEXT PRIMARY KEY, title TEXT, authors TEXT, year INTEGER, content TEXT)""")
    else:
        c.execute("""CREATE TABLE IF NOT EXISTS papers
                     (id TEXT PRIMARY KEY, title TEXT, authors TEXT, year INTEGER,
                      categories TEXT, abstract TEXT, pdf_url TEXT,
                      download_status TEXT, matched_terms TEXT,
                      relevance_prob REAL, pdf_path TEXT, content TEXT)""")
        c.execute("""CREATE TABLE IF NOT EXISTS parameters
                     (paper_id TEXT, entity_text TEXT, entity_label TEXT,
                      value REAL, unit TEXT, context TEXT, phase TEXT,
                      score REAL, co_occurrence BOOLEAN,
                      FOREIGN KEY(paper_id) REFERENCES papers(id))""")
        c.execute("CREATE INDEX IF NOT EXISTS idx_pid ON parameters(paper_id)")
    conn.commit()
    conn.close()

init_db(METADATA_DB)
init_db(UNIVERSE_DB)

@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def universe_insert(paper: dict):
    conn = sqlite3.connect(UNIVERSE_DB)
    c = conn.cursor()
    c.execute("""INSERT OR REPLACE INTO papers
                 (id, title, authors, year, content)
                 VALUES (?,?,?,?,?)""",
              (paper["id"], paper.get("title",""), paper.get("authors","Unknown"),
               paper.get("year",0), paper.get("content","")))
    conn.commit()
    conn.close()

def save_metadata(df: pd.DataFrame):
    conn = sqlite3.connect(METADATA_DB)
    df.to_sql("papers", conn, if_exists="replace", index=False)
    conn.commit()
    conn.close()
    st.success(f"Metadata saved → `{METADATA_DB}`")

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

# -------------------------- PDF DOWNLOAD --------------------------
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

        universe_insert({"id": pid, "title": paper["title"],
                         "authors": paper["authors"], "year": paper["year"],
                         "content": txt})

        paper.update({
            "download_status": f"Downloaded ({kb:.1f} KB)",
            "pdf_path": str(dest),
            "content": txt,
        })
        update_log(f"Success {pid} – {kb:.1f} KB")
    except Exception as e:
        paper.update({
            "download_status": f"Failed: {e}",
            "pdf_path": None,
            "content": f"Error: {e}",
        })
        if dest.exists() and dest.stat().st_size == 0:
            dest.unlink(missing_ok=True)
        update_log(f"Failed {pid} – {e}")
    return paper

# -------------------------- ZIP --------------------------
def make_zip(paths: list) -> str | None:
    if not paths:
        return None
    zip_path = os.path.join(DB_DIR, "piezoelectricity_pdfs.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for p in paths:
            if p and os.path.exists(p):
                z.write(p, os.path.basename(p))
    return zip_path

# -------------------------- UI --------------------------
st.title("Piezoelectricity in PVDF – SciBERT + Full-text Search")
st.markdown(
    """
Search arXiv for **piezoelectricity in PVDF** (dopants, α/β phases, electro-spinning, efficiency …).  
SciBERT scores abstracts; a **slider** lets you set the relevance cut-off.  
All PDFs, metadata and full-text are stored and downloadable.
"""
)

# ---------- LOG AREA (static key) ----------
log_placeholder = st.empty()
def show_logs():
    if st.session_state.log_buffer:
        log_placeholder.text_area(
            "Processing Logs",
            "\n".join(st.session_state.log_buffer[-30:]),
            height=180,
            key="log_area"
        )
    else:
        log_placeholder.empty()

show_logs()

# ---------- SIDEBAR ----------
with st.sidebar:
    st.header("Search")
    q = st.text_input(
        "Query",
        value=' OR '.join(f'"{t}"' for t in [
            "piezoelectricity", "PVDF", "beta phase", "electrospun nanofibers",
            "SnO2", "dopants", "efficiency", "nanogenerators"
        ]),
        key="q_input"
    )
    cats = st.multiselect(
        "Categories", ["cond-mat.mtrl-sci", "physics.app-ph"],
        default=["cond-mat.mtrl-sci"], key="cat_sel"
    )
    max_res = st.slider("Max results", 1, 200, 30, key="max_res")
    col1, col2 = st.columns(2)
    sy = col1.number_input("Start year", 1990, datetime.now().year, 2010, key="sy")
    ey = col2.number_input("End year", sy, datetime.now().year, datetime.now().year, key="ey")
    rel_thr = st.slider("Relevance threshold (%)", 0, 100, 30, key="rel_thr")
    out_fmt = st.multiselect("Output", ["SQLite", "CSV", "JSON"], default=["SQLite"], key="out_fmt")

    col_btn1, col_btn2 = st.columns(2)
    search_btn = col_btn1.button("Search arXiv", key="search_btn")
    reset_btn = col_btn2.button("Reset All", key="reset_btn")

    if st.session_state.relevant_papers:
        if st.button("Search Again", key="search_again"):
            for k in ["search_results", "relevant_papers", "pdf_paths", "zip_path"]:
                if k in st.session_state:
                    del st.session_state[k]
            st.rerun()

if reset_btn:
    for k in list(st.session_state.keys()):
        if k != "page_config_set":
            del st.session_state[k]
    st.success("Session cleared – reload the page")
    st.rerun()

# -------------------------- MAIN LOGIC --------------------------
if search_btn:
    if not q.strip() or not cats or sy > ey:
        st.error("Check inputs")
    elif not health_check():
        st.stop()
    else:
        st.session_state.processing = True
        with st.spinner("Querying arXiv + SciBERT scoring…"):
            all_papers = query_arxiv(q, cats, max_res, sy, ey)

        if not all_papers:
            st.warning("No papers found")
        else:
            relevant = [p for p in all_papers if p["relevance_prob"] >= rel_thr]
            st.success(f"**{len(relevant)}** papers ≥ {rel_thr}% relevance")

            if relevant:
                prog = st.progress(0)
                stat = st.empty()
                paths = []

                with concurrent.futures.ThreadPoolExecutor(max_workers=4) as exe:
                    futures = {exe.submit(download_one, p): i for i, p in enumerate(relevant)}
                    for fut in concurrent.futures.as_completed(futures):
                        idx = futures[fut]
                        paper = fut.result()
                        relevant[idx] = paper
                        if paper["pdf_path"]:
                            paths.append(paper["pdf_path"])
                        prog.progress((idx + 1) / len(relevant))
                        stat.text(f"{idx+1}/{len(relevant)} – {paper['title'][:60]}…")

                prog.empty()
                stat.empty()

                # Persist
                st.session_state.relevant_papers = relevant
                st.session_state.pdf_paths = paths
                zip_path = make_zip(paths)
                st.session_state.zip_path = zip_path

                # Save DB if requested
                if "SQLite" in out_fmt:
                    save_metadata(pd.DataFrame(relevant).drop(columns=["content"], errors="ignore"))
        st.session_state.processing = False

# -------------------------- DISPLAY RESULTS --------------------------
if st.session_state.relevant_papers:
    df = pd.DataFrame(st.session_state.relevant_papers)

    # ---- Individual PDF buttons ----
    pdf_buttons = []
    for _, row in df.iterrows():
        if row.get("pdf_path") and os.path.exists(row["pdf_path"]):
            with open(row["pdf_path"], "rb") as f:
                pdf_buttons.append(
                    st.download_button(
                        label="PDF",
                        data=f.read(),
                        file_name=f"{row['id']}.pdf",
                        mime="application/pdf",
                        key=f"pdf_{row['id']}"
                    )
                )
        else:
            pdf_buttons.append("Failed")
    df_display = df[["id", "title", "year", "relevance_prob", "download_status"]].copy()
    df_display.insert(0, "PDF", pdf_buttons)

    st.subheader("Results")
    st.dataframe(df_display, use_container_width=True, hide_index=True)

    # ---- Bulk downloads ----
    st.subheader("Bulk Downloads")
    col1, col2, col3 = st.columns(3)

    # ZIP
    if st.session_state.zip_path and os.path.exists(st.session_state.zip_path):
        with open(st.session_state.zip_path, "rb") as f:
            col1.download_button(
                "All PDFs (ZIP)",
                f.read(),
                "piezoelectricity_pdfs.zip",
                "application/zip",
                key="zip_dl"
            )

    # CSV
    if "CSV" in out_fmt:
        col2.download_button(
            "CSV",
            df.to_csv(index=False),
            "piezoelectricity_papers.csv",
            "text/csv",
            key="csv_dl"
        )

    # JSON
    if "JSON" in out_fmt:
        col3.download_button(
            "JSON",
            df.to_json(orient="records"),
            "piezoelectricity_papers.json",
            "application/json",
            key="json_dl"
        )

    # DB files
    st.subheader("Database Files")
    col_db1, col_db2 = st.columns(2)
    if os.path.exists(METADATA_DB):
        with open(METADATA_DB, "rb") as f:
            col_db1.download_button(
                "Metadata DB",
                f.read(),
                os.path.basename(METADATA_DB),
                "application/octet-stream",
                key="db_meta"
            )
    if os.path.exists(UNIVERSE_DB):
        with open(UNIVERSE_DB, "rb") as f:
            col_db2.download_button(
                "Full-text DB",
                f.read(),
                os.path.basename(UNIVERSE_DB),
                "application/octet-stream",
                key="db_uni"
            )

# Always show logs
show_logs()
