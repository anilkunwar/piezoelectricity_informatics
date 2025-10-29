# --------------------------------------------------------------
#  Piezoelectricity in PVDF – Full-featured, Cloud-ready
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
from scipy.special import softmax
from tenacity import retry, stop_after_attempt, wait_fixed

# -------------------------- CONFIG --------------------------
if os.path.exists("/tmp"):                     # Streamlit Cloud / most clouds
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

# -------------------------- STREAMLIT UI --------------------------
st.set_page_config(page_title="Piezoelectricity in PVDF", layout="wide")
st.title("Piezoelectricity in PVDF – SciBERT + Full-text Search")
st.markdown(
    """
Search arXiv for **piezoelectricity in PVDF** (dopants, α/β phases, electro-spinning, efficiency …).  
SciBERT scores abstracts; a **slider** lets you set the relevance cut-off.  
All PDFs, metadata and full-text are stored and downloadable.
"""
)

# -------------------------- SESSION STATE --------------------------
for key in [
    "log_buffer",
    "processing",
    "download_files",
    "search_results",
    "relevant_papers",
]:
    if key not in st.session_state:
        st.session_state[key] = (
            [] if key == "log_buffer" else False if key == "processing" else {"pdf_paths": [], "zip_path": None}
        )

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

def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def health_check():
    mem = mem_usage()
    free_gb = psutil.disk_usage(DB_DIR).free / (1024**3)
    update_log(f"RAM {mem:.1f} MB | Disk free {free_gb:.1f} GB")
    if mem > 1500:
        st.warning(f"High RAM ({mem:.1f} MB)")
        cleanup()
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
        # ---- pattern count ----
        n = sum(bool(p.search(norm(abstract))) for p in COMPILED)
        prob = np.sqrt(n) / np.sqrt(len(KEY_PATTERNS))

        # ---- attention boost ----
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
    conn.close()

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

# -------------------------- PDF DOWNLOAD (THREAD-SAFE) --------------------------
@retry(stop=stop_after_attempt(4), wait=wait_fixed(2))
def _download(url: str, dest: Path) -> float:
    req = requests.Request("GET", url,
                           headers={"User-Agent": "PiezoelectricityTool/1.0"})
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
    except Exception as e:
        paper.update({
            "download_status": f"Failed: {e}",
            "pdf_path": None,
            "content": f"Error: {e}",
        })
        if dest.exists() and dest.stat().st_size == 0:
            dest.unlink(missing_ok=True)
    return paper

# -------------------------- ZIP --------------------------
def make_zip(paths: list) -> str:
    zip_path = os.path.join(DB_DIR, "piezoelectricity_pdfs.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for p in paths:
            if p and os.path.exists(p):
                z.write(p, os.path.basename(p))
    return zip_path

# -------------------------- UI --------------------------
with st.sidebar:
    st.header("Search")
    q = st.text_input("Query", value=' OR '.join(f'"{t}"' for t in [
        "piezoelectricity", "PVDF", "beta phase", "electrospun nanofibers",
        "SnO2", "dopants", "efficiency", "nanogenerators"
    ]), key="q_input")
    cats = st.multiselect("Categories", ["cond-mat.mtrl-sci", "physics.app-ph"],
                          default=["cond-mat.mtrl-sci"], key="cat_sel")
    max_res = st.slider("Max results", 1, 200, 30, key="max_res")
    col1, col2 = st.columns(2)
    sy = col1.number_input("Start year", 1990, datetime.now().year, 2010, key="sy")
    ey = col2.number_input("End year", sy, datetime.now().year, datetime.now().year, key="ey")
    rel_thr = st.slider("Relevance threshold (%)", 0, 100, 30, key="rel_thr")
    out_fmt = st.multiselect("Output", ["SQLite", "CSV", "JSON"], default=["SQLite"], key="out_fmt")
    search_btn = st.button("Search arXiv", key="search_btn")
    reset_btn = st.button("Reset", key="reset_btn")

if reset_btn:
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    st.success("Reset – reload the page")
    st.experimental_rerun()

log_area = st.empty()
def show_logs():
    if st.session_state.log_buffer:
        log_area.text_area(
            "Logs",
            "\n".join(st.session_state.log_buffer[-30:]),
            height=180,
            key=f"log_{int(time.time())}",   # <-- unique key every render
        )

# -------------------------- MAIN LOGIC --------------------------
if search_btn:
    if not q.strip() or not cats or sy > ey:
        st.error("Check inputs")
    else:
        st.session_state.processing = True
        if not health_check():
            st.stop()

        with st.spinner("Querying arXiv…"):
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

                # Store for later download buttons
                st.session_state.relevant_papers = relevant
                st.session_state.download_files["pdf_paths"] = paths
                zip_path = make_zip(paths)
                st.session_state.download_files["zip_path"] = zip_path

                df = pd.DataFrame(relevant)
                st.subheader("Results")
                st.dataframe(
                    df[["id", "title", "year", "relevance_prob", "download_status"]],
                    use_container_width=True,
                )

                # ---- OUTPUT ----
                if "SQLite" in out_fmt:
                    save_metadata(df.drop(columns=["content"], errors="ignore"))
                    st.info("Metadata saved to DB")
                if "CSV" in out_fmt:
                    st.download_button(
                        "CSV",
                        df.to_csv(index=False),
                        "piezoelectricity_papers.csv",
                        "text/csv",
                        key=f"csv_{int(time.time())}",
                    )
                if "JSON" in out_fmt:
                    st.download_button(
                        "JSON",
                        df.to_json(orient="records"),
                        "piezoelectricity_papers.json",
                        "application/json",
                        key=f"json_{int(time.time())}",
                    )

                # ---- ZIP ----
                if zip_path and os.path.exists(zip_path):
                    with open(zip_path, "rb") as f:
                        st.download_button(
                            "All PDFs (ZIP)",
                            f.read(),
                            "piezoelectricity_pdfs.zip",
                            "application/zip",
                            key=f"zip_{int(time.time())}",
                        )

                # ---- DB files ----
                for name, path in [("Metadata DB", METADATA_DB), ("Universe DB", UNIVERSE_DB)]:
                    if os.path.exists(path):
                        with open(path, "rb") as f:
                            st.download_button(
                                name,
                                f.read(),
                                os.path.basename(path),
                                "application/octet-stream",
                                key=f"db_{name}_{int(time.time())}",
                            )
        st.session_state.processing = False
        show_logs()

# Always show logs (outside any conditional block)
show_logs()
