# app.py
# Evidence Synthesis (v4.3 AI) — clean PDF-only upload, no samples, fresh-run caching, single status bar.

import io, os, re, json, math, hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Any
import streamlit as st
import pandas as pd
from pandas import DataFrame
import os, sys
import pytesseract

# When running as a PyInstaller exe, files are unpacked under _MEIPASS
BASE_DIR = getattr(sys, "_MEIPASS", os.path.abspath(os.path.dirname(__file__)))

# These paths will be injected by the workflow when we package on Windows:
#   vendor\tesseract\tesseract.exe
#   vendor\poppler\bin\*.exe
TESS_PATH = os.path.join(BASE_DIR, "vendor", "tesseract", "tesseract.exe")
POPPLER_BIN = os.path.join(BASE_DIR, "vendor", "poppler", "bin")

# Tell pytesseract where tesseract.exe lives
if os.path.exists(TESS_PATH):
    pytesseract.pytesseract.tesseract_cmd = TESS_PATH

# Ensure Poppler bin is on PATH (for pdf2image)
if os.path.isdir(POPPLER_BIN):
    os.environ["PATH"] = POPPLER_BIN + os.pathsep + os.environ.get("PATH", "")



# ---------- Optional deps (graceful fallbacks) ----------
pdfplumber = None
try:
    import pdfplumber as _pdfplumber
    pdfplumber = _pdfplumber
except Exception:
    pass

PyPDF2 = None
try:
    import PyPDF2 as _PyPDF2
    PyPDF2 = _PyPDF2
except Exception:
    pass

fitz = None
try:
    import fitz as _fitz  # PyMuPDF
    fitz = _fitz
except Exception:
    pass

pytesseract = None
pdf2image = None
from PIL import Image

try:
    import pytesseract as _pytesseract
    pytesseract = _pytesseract
except Exception:
    pass

try:
    from pdf2image import convert_from_bytes as _convert_from_bytes
    pdf2image = _convert_from_bytes
except Exception:
    pass

# matplotlib check for pandas Styler background_gradient
HAS_MPL = True
try:
    import matplotlib  # noqa: F401
except Exception:
    HAS_MPL = False

# ---------- AI (optional) ----------
HAVE_SBERT = True
HAVE_ZSHOT = True
try:
    from sentence_transformers import SentenceTransformer, util as sbert_util
except Exception:
    HAVE_SBERT = False

try:
    from transformers import pipeline
except Exception:
    HAVE_ZSHOT = False

st.set_page_config(page_title="Evidence Synthesis • Auditable (v4.3 AI)", page_icon="📊", layout="wide")

# ---------- Upload limits ----------
MAX_FILES = 600
MAX_TOTAL_BYTES = 2 * 1024 * 1024 * 1024  # 2 GB

# ---------- Dimensions & Keywords (base) ----------
DIMENSIONS = [
    "Child rights deprivations",
    "Underlying causes",
    "Bottlenecks/barriers",
    "Effectiveness of interventions",
    "UNICEF comparative advantage",
]

BASE_KEYMAP: Dict[str, List[str]] = {
    "Child rights deprivations": [
        "poverty","indigence","learning loss","dropout","out-of-school","mental health","anxiety","depression",
        "violence","abuse","food insecurity","malnutrition","disaster","hurricane","volcano","suicide","bullying",
        "exclusion","inequality","overweight","obese"
    ],
    "Underlying causes": [
        "underlying cause","root cause","driver","determinant","led to","due to","because","as a result",
        "climate change","covid","covid-19","pandemic","migration","irregular status","debt","fiscal","norms",
        "gender","stigma","economy","inflation","policy environment","structural","systemic"
    ],
    "Bottlenecks/barriers": [
        "barrier","bottleneck","constraint","limitation","challenge","hindrance","access barrier","affordability",
        "availability","stigma","taboo","underreporting","data gap","coordination","coverage","restrictive",
        "legal status","bureaucracy","capacity","sustainability","fragmented","inequity","quality gap"
    ],
    "Effectiveness of interventions": [
        "intervention","programme","program","initiative","effectiveness","effective","impact","outcome","results",
        "child friendly spaces","cfs","mhpss","psychosocial","u-report","online learning","referral","tvet",
        "school feeding","cash transfer","coverage increased","improved","evidence shows","evaluation finds"
    ],
    "UNICEF comparative advantage": [
        "comparative advantage","unicef","mandate","crc","oecs","leadership","coordination","convening",
        "r4v","technical support","advocacy","evidence generation","youth engagement","u-report","innovation",
        "partnership","core commitments for children","ccc"
    ],
}

DEFAULT_HEADMAP = {
    r"(driver|underlying cause|determinant)s?": "Underlying causes",
    r"(barrier|bottleneck|constraint|challenge|limitation)s?": "Bottlenecks/barriers",
    r"(intervention|result|effectiveness|impact|outcome)s?": "Effectiveness of interventions",
}

SECTION_CUES: Dict[str, List[str]] = {
    "Underlying causes": ["underlying causes","drivers","determinants","root causes","context analysis"],
    "Bottlenecks/barriers": ["barriers","bottlenecks","constraints","challenges","limitations"],
    "Effectiveness of interventions": ["interventions","programme effectiveness","results","impact","outcomes","what worked","effectiveness"],
}

# ---------- v3-style seed ----------
DEFAULT_V3_BLOCK = """
• learning loss, mental health, anxiety, violence, abuse, bullying	• climate change, covid, covid-19, pandemic, migration, gender	• stigma, taboo, coverage, affordability, access	• mhpss, psychosocial, u-report, online learning, school feeding, evaluation	• oecs, leadership, technical, evidence, u-report
• poverty, dropout, mental health, anxiety, violence, abuse	• climate change, covid, covid-19, pandemic, migration, irregular status	• stigma, data gaps, coordination, coverage, access, legal status	• cash transfer, child friendly spaces, cfs, psychosocial, online learning, tvet	• coordination, mandate, crc, oecs, technical, advocacy
• violence, abuse, bullying	• covid, covid-19, norms, gender, stigma	• stigma, access, capacity	• effectiveness, impact	• technical, advocacy, evidence
• poverty, mental health, violence, abuse, indigence, out-of-school	• climate change, covid, covid-19, pandemic, migration, debt	• stigma, data gaps, coordination, coverage, access, capacity	• cash transfer, child friendly spaces, cfs, psychosocial, u-report, school feeding	• coordination, mandate, crc, oecs, leadership, technical
• dropout, mental health, violence, abuse, exclusion, out-of-school	• covid, covid-19, pandemic, migration, irregular status, norms	• stigma, coordination, coverage, access, legal status, bureaucracy	• cash transfer, child friendly spaces, cfs, mhpss, psychosocial, u-report	• coordination, mandate, oecs, leadership, r4v, technical
""".strip()

def parse_custom_block(block: str) -> Dict[str, list]:
    agg = {d: set() for d in DIMENSIONS}
    if not block or not block.strip():
        return {d: [] for d in DIMENSIONS}
    lines = [ln for ln in block.splitlines() if ln.strip()]
    for ln in lines:
        parts = [p.strip(" •\t") for p in re.split(r"[•]", ln) if p.strip(" •\t")]
        if len(parts) < 5:
            parts = [p.strip() for p in re.split(r"\t+", ln) if p.strip()]
        if len(parts) != 5:
            continue
        for idx, grp in enumerate(parts[:5]):
            for tok in [t.strip().lower() for t in grp.split(",") if t.strip()]:
                agg[DIMENSIONS[idx]].add(tok)
    return {d: sorted(v) for d, v in agg.items()}

# Initialize working KEYMAP in session (so edits persist)
if "keymap" not in st.session_state:
    st.session_state.keymap = {d: sorted(set(map(str.lower, BASE_KEYMAP[d]))) for d in DIMENSIONS}

# Seed once with user-provided v3 block (append-only)
if "custom_v3_seed_applied" not in st.session_state:
    st.session_state.custom_v3_seed_applied = False
if not st.session_state.custom_v3_seed_applied:
    seed = parse_custom_block(DEFAULT_V3_BLOCK)
    for d in DIMENSIONS:
        merged = set(st.session_state.keymap[d])
        merged.update(seed.get(d, []))
        st.session_state.keymap[d] = sorted(merged)
    st.session_state.custom_v3_seed_applied = True

# Use session keymap for classification
KEYMAP = st.session_state.keymap

# ---------- Run freshness + file hashing ----------
def _sha1_bytes(b: bytes) -> str:
    h = hashlib.sha1(); h.update(b); return h.hexdigest()

def _files_signature(file_tuples: List[Tuple[str, bytes]]) -> str:
    # names + sizes only (order-insensitive) to detect a truly new upload set
    parts = [f"{n}:{len(b)}" for (n, b) in file_tuples]
    parts.sort()
    return hashlib.sha1("|".join(parts).encode()).hexdigest()

if "run_id" not in st.session_state:
    st.session_state.run_id = ""
if "last_sig" not in st.session_state:
    st.session_state.last_sig = ""

# ---------- PDF extraction ----------
def extract_with_pdfplumber(blob: bytes) -> List[str]:
    if pdfplumber is None: return []
    pages: List[str] = []
    try:
        with pdfplumber.open(io.BytesIO(blob)) as pdf:
            for p in pdf.pages:
                t = p.extract_text() or ""
                pages.append(t)
    except Exception:
        return []
    return pages

def extract_with_pypdf(blob: bytes) -> List[str]:
    if PyPDF2 is None: return []
    try:
        reader = PyPDF2.PdfReader(io.BytesIO(blob))
        out: List[str] = []
        for page in reader.pages:
            t = page.extract_text() or ""
            out.append(t)
        return out
    except Exception:
        return []

def extract_with_pymupdf(blob: bytes) -> List[str]:
    if fitz is None: return []
    pages: List[str] = []
    try:
        doc = fitz.open(stream=blob, filetype="pdf")
        for page in doc:
            t = page.get_text("text") or ""
            pages.append(t)
    except Exception:
        return []
    return pages

def extract_with_ocr(blob: bytes, dpi: int = 300, max_pages: int = 10, lang: str = "eng") -> List[str]:
    if pdf2image is None or pytesseract is None: return []
    texts: List[str] = []
    try:
        images = pdf2image(blob, dpi=dpi)
        for i, img in enumerate(images[:max_pages]):
            if not isinstance(img, Image.Image):
                from PIL import Image as _Image
                try:
                    img = _Image.fromarray(img)
                except Exception:
                    texts.append("")
                    continue
            t = pytesseract.image_to_string(img, lang=lang, config="--psm 3 --oem 3")
            texts.append(t or "")
        if len(images) > max_pages:
            texts.extend([""] * (len(images) - max_pages))
    except Exception:
        return []
    return texts

@st.cache_data(show_spinner=False)
def cached_extract_pages(file_hash: str,
                         run_id: str,
                         blob: bytes,
                         dpi: int,
                         max_pages: int,
                         lang: str) -> Tuple[List[str], str]:
    """Cache per (file_hash, run_id, dpi, max_pages, lang)."""
    for fn, name in [
        (extract_with_pymupdf, "PyMuPDF") if fitz is not None else (lambda b: [], "PyMuPDF"),
        (extract_with_pdfplumber, "pdfplumber"),
        (extract_with_pypdf, "PyPDF2"),
    ]:
        pages = fn(blob)
        if sum(len(p.strip()) for p in pages) >= 50:
            return pages, name
    pages = extract_with_ocr(blob, dpi=dpi, max_pages=max_pages, lang=lang)
    if sum(len(p.strip()) for p in pages) >= 30:
        return pages, "OCR"
    return [], "none"

SENT_SPLIT = re.compile(r'(?<=[\.\?\!])\s+(?=[A-Z0-9“"])')

def split_sentences(text: str) -> List[str]:
    text = re.sub(r'\s+', ' ', text).strip()
    parts = re.split(SENT_SPLIT, text)
    out: List[str] = []
    for p in parts:
        out.extend([s.strip() for s in re.split(r'\s*;\s*', p) if s.strip()])
    return out

def detect_heading_lines(page_text: str) -> List[str]:
    lines = [l.strip() for l in page_text.splitlines() if l.strip()]
    heads: List[str] = []
    for l in lines:
        if len(l) <= 120 and (l.isupper() or re.match(r'^[A-Z][A-Za-z0-9 \-:&\/]{2,}$', l)):
            heads.append(l.lower())
    uniq = []
    for h in heads:
        if h not in uniq:
            uniq.append(h)
    return uniq

def headmap_dim_for_text(text: str, headmap: Dict[str, str]) -> str:
    low = (text or "").lower()
    for pattern, dim in headmap.items():
        try:
            if re.search(pattern, low):
                return dim
        except re.error:
            continue
    return ""

# ---------- AI loaders (cached) ----------
@st.cache_resource(show_spinner=False)
def load_sbert(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    if not HAVE_SBERT: return None
    try:
        return SentenceTransformer(model_name)
    except Exception:
        return None

@st.cache_resource(show_spinner=False)
def load_zeroshot(model_name: str = "MoritzLaurer/deberta-v3-base-zeroshot-v2"):
    if not HAVE_ZSHOT: return None
    try:
        return pipeline("zero-shot-classification", model=model_name)
    except Exception:
        return None

# ---------- Scoring ----------
def score_classic(hits: int) -> float:
    return float(min(10, hits))

def score_length_normalized(hits: int, chars: int, pivot_chars: int = 10000) -> float:
    if chars <= 0: return 0.0
    norm = hits / (chars / pivot_chars + 1.0)
    return float(min(10.0, round(10.0 * norm, 1)))

def score_curved(hits: int, chars: int, per_k_chars: int = 1000, tau: float = 2.0) -> float:
    if chars <= 0: return 0.0
    hits_per_k = hits / (max(1.0, chars / per_k_chars))
    s = 10.0 * (1.0 - math.exp(- hits_per_k / max(0.1, tau)))
    return float(round(min(10.0, s), 1))

def interpret_heatmap(matrix_df: DataFrame) -> str:
    dims = [d for d in matrix_df.columns if d in DIMENSIONS]
    if not dims: return "No scores available to interpret."
    means = matrix_df[dims].mean().sort_values(ascending=False)
    strongest = means.index[0]; weakest = means.index[-1]
    bullets = []
    bullets.append(f"• Strongest dimension across documents: **{strongest}** (avg {means[strongest]:.1f}/10).")
    bullets.append(f"• Weakest dimension across documents: **{weakest}** (avg {means[weakest]:.1f}/10).")
    for _, r in matrix_df.iterrows():
        doc = r['Document']
        top_dim = r[dims].idxmax()
        bullets.append(f"• **{doc}** emphasizes **{top_dim}** ({r[top_dim]:.1f}/10).")
    return "\n".join(bullets)

def _pick_excel_engine() -> str:
    try:
        import xlsxwriter  # noqa: F401
        return "xlsxwriter"
    except Exception:
        try:
            import openpyxl  # noqa: F401
            return "openpyxl"
        except Exception:
            return ""

def export_to_excel(dfs: Dict[str, DataFrame]) -> bytes:
    engine = _pick_excel_engine()
    if not engine:
        raise RuntimeError("No Excel writer engine available. Install either 'xlsxwriter' or 'openpyxl'.")
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine=engine) as writer:
        for name, df in dfs.items():
            sname = name[:31] if len(name) > 31 else name
            df.to_excel(writer, index=False, sheet_name=sname)
    bio.seek(0)
    return bio.read()

# ---------- Base (rules) classifier ----------
def classify_sentence_rules(sentence: str, current_section: str = "", mapped_dim: str = "", boost_strength: int = 5, force_mapped: bool = False) -> Tuple[str, Dict[str,int], float]:
    low = sentence.lower()
    counts: Dict[str, int] = {dim: 0 for dim in DIMENSIONS}
    for dim, keys in KEYMAP.items():
        hits = 0
        for k in keys:
            hits += len(re.findall(r'\b' + re.escape(k) + r'\b', low))
        counts[dim] = hits

    if current_section:
        for dim, cues in SECTION_CUES.items():
            if any(c in current_section for c in cues):
                counts[dim] += 1

    if mapped_dim:
        if force_mapped:
            distinct_dims = sum(1 for v in counts.values() if v > 0)
            confidence = min(1.0, (1 + distinct_dims) / 3.0)
            return mapped_dim, counts, confidence
        else:
            counts[mapped_dim] += max(0, int(boost_strength))

    best_dim = max(counts, key=lambda d: counts[d])
    best_hits = counts[best_dim]
    distinct_dims = sum(1 for v in counts.values() if v > 0)
    confidence = 0.0 if best_hits <= 0 else min(1.0, (distinct_dims + (1 if mapped_dim else 0)) / 4.0)
    return (best_dim if best_hits > 0 else ""), counts, confidence

# ---------- AI-assisted classify ----------
def build_dim_prompts_from_keymap(keymap: Dict[str, List[str]], top_n: int = 12) -> Dict[str, str]:
    prompts = {}
    for dim in DIMENSIONS:
        kws = ", ".join(keymap.get(dim, [])[:top_n])
        prompts[dim] = f"{dim.lower()} — {kws}"
    return prompts

def classify_sentence_ai(sentence: str,
                         counts: Dict[str,int],
                         ai_mode: str,
                         sbert_model,
                         label_embeds,
                         zshot_pipe,
                         zshot_thresh: float,
                         sem_thresh: float,
                         z_boost: int,
                         sem_boost: int) -> Tuple[Dict[str,int], List[str], float]:
    method_tags: List[str] = []
    ai_conf = 0.0

    if ai_mode in {"Semantic (SBERT)", "Ensemble"} and sbert_model is not None and label_embeds is not None:
        try:
            s_vec = sbert_model.encode([sentence], convert_to_tensor=True, normalize_embeddings=True)
            sims = sbert_util.cos_sim(s_vec, label_embeds)[0]
            best_idx = int(sims.argmax().item())
            best_dim = DIMENSIONS[best_idx]
            best_sim = float(sims[best_idx])
            if best_sim >= sem_thresh:
                counts[best_dim] += max(1, int(round(sem_boost * best_sim)))
                method_tags.append("semantic")
                ai_conf = max(ai_conf, best_sim)
        except Exception:
            pass

    if ai_mode in {"Zero-shot (NLI)", "Ensemble"} and zshot_pipe is not None:
        try:
            res = zshot_pipe(sentence, candidate_labels=DIMENSIONS, multi_label=False)
            labels = res.get("labels", [])
            scores = res.get("scores", [])
            if labels and scores:
                best_dim = labels[0]
                best_score = float(scores[0])
                if best_score >= zshot_thresh:
                    counts[best_dim] += max(1, int(round(z_boost * best_score)))
                    method_tags.append("zeroshot")
                    ai_conf = max(ai_conf, best_score)
        except Exception:
            pass

    return counts, method_tags, ai_conf

def gather_evidence(pages: List[str],
                    headmap: Dict[str, str],
                    boost_strength: int,
                    force_mapped: bool,
                    ai_mode: str,
                    sbert_model,
                    label_embeds,
                    zshot_pipe,
                    zshot_thresh: float,
                    sem_thresh: float,
                    z_boost: int,
                    sem_boost: int,
                    top_k_per_dim: int = 3) -> Tuple[Dict[str, List[Tuple[str,int,float,str]]], Dict[str,int], int]:
    evidence: Dict[str, List[Tuple[str,int,float,str]]] = {dim: [] for dim in DIMENSIONS}
    hit_counts: Dict[str, int] = {dim: 0 for dim in DIMENSIONS}
    total_chars = sum(len(p) for p in pages)

    for pi, ptext in enumerate(pages, start=1):
        heads = detect_heading_lines(ptext)
        section_hint = " ".join(heads[:3]).lower() if heads else ""
        mapped_dim = headmap_dim_for_text(section_hint, headmap)
        sentences = split_sentences(ptext)

        for s in sentences:
            dim, counts, conf = classify_sentence_rules(
                s, current_section=section_hint, mapped_dim=mapped_dim,
                boost_strength=boost_strength, force_mapped=force_mapped
            )
            method = "rules"

            if ai_mode != "Off":
                counts, tags, ai_conf = classify_sentence_ai(
                    s, counts, ai_mode, sbert_model, label_embeds, zshot_pipe,
                    zshot_thresh=zshot_thresh, sem_thresh=sem_thresh,
                    z_boost=z_boost, sem_boost=sem_boost
                )
                if tags:
                    method = "rules+" + ("ensemble" if len(tags) == 2 else tags[0])
                    conf = min(1.0, max(conf, ai_conf * 0.9))
                dim = max(counts, key=lambda d: counts[d]) if any(v > 0 for v in counts.values()) else ""

            for d, c in counts.items():
                hit_counts[d] += c

            if dim:
                evidence[dim].append((s, pi, conf, method))

    for dim in DIMENSIONS:
        ev = evidence[dim]
        ev = sorted(ev, key=lambda x: (len(x[0]), x[2]), reverse=True)[:top_k_per_dim]
        evidence[dim] = ev
    return evidence, hit_counts, total_chars

# ---------- UI ----------
st.title("📊 Evidence Synthesis: Auditable (v4.3 AI)")

with st.sidebar:
    st.header("Inputs")
    up_files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
    st.caption("Tip: For scanned PDFs, configure OCR below.")
    st.divider()

    st.header("OCR Settings")
    tesseract_path = st.text_input("Tesseract path (Windows)", value="")
    poppler_path = st.text_input("Poppler bin path (Windows)", value="")
    dpi = st.slider("OCR DPI", 150, 400, 300, 25)
    max_pages = st.slider("OCR max pages", 1, 20, 10, 1)
    lang = st.text_input("OCR language(s)", value="eng")
    preview = st.checkbox("Show page text previews", value=False)

    st.divider()
    st.header("Scoring")
    scoring_mode = st.selectbox("Scoring mode", ["Curved (anti-saturation)","Length-normalized","Classic (cap at 10)"], index=0)
    tau = st.slider("Curved: τ (higher = flatter)", 0.5, 5.0, 2.0, 0.1)
    pivot_chars = st.slider("Length-normalized: pivot chars", 5000, 50000, 20000, 1000)

    st.divider()
    st.header("Heading → Dimension mapping")
    if "headmap" not in st.session_state:
        st.session_state.headmap = DEFAULT_HEADMAP.copy()

    st.divider()
    st.header("AI assist (optional)")
    ai_mode = st.selectbox("Mode", ["Off","Semantic (SBERT)","Zero-shot (NLI)","Ensemble"], index=0)
    sem_model_name = st.text_input("SBERT model", "sentence-transformers/all-MiniLM-L6-v2")
    z_model_name   = st.text_input("Zero-shot model", "MoritzLaurer/deberta-v3-base-zeroshot-v2")
    sem_thresh = st.slider("Semantic: similarity threshold", 0.30, 0.90, 0.45, 0.01)
    zshot_thresh = st.slider("Zero-shot: confidence threshold", 0.40, 0.95, 0.65, 0.01)
    sem_boost = st.slider("Semantic: boost", 1, 10, 4, 1)
    z_boost   = st.slider("Zero-shot: boost", 1, 12, 5, 1)
    if ai_mode != "Off":
        if ai_mode in {"Semantic (SBERT)","Ensemble"} and not HAVE_SBERT:
            st.warning("`sentence-transformers` not installed. Run: pip install sentence-transformers")
        if ai_mode in {"Zero-shot (NLI)","Ensemble"} and not HAVE_ZSHOT:
            st.warning("`transformers` not installed. Run: pip install transformers torch --upgrade")

# Configure OCR paths
if tesseract_path:
    try:
        import pytesseract as _pt
        _pt.pytesseract.tesseract_cmd = tesseract_path
        pytesseract = _pt
        st.success("Tesseract path set.")
    except Exception as e:
        st.error(f"Failed to set Tesseract path: {e}")

if poppler_path:
    os.environ["PATH"] = poppler_path + os.pathsep + os.environ.get("PATH","")
    st.success("Poppler path added to PATH.")

# -------- Load docs (upload only, PDF) --------
docs: List[Tuple[str, bytes]] = []

if not up_files:
    st.info("Upload one or more PDF files to start analysis.")
    st.stop()

if len(up_files) > MAX_FILES:
    st.error(f"Too many files uploaded ({len(up_files)}). Limit: {MAX_FILES}.")
    st.stop()

total_bytes = sum(getattr(f, "size", 0) for f in up_files)
if total_bytes > MAX_TOTAL_BYTES:
    mb = total_bytes / (1024 * 1024)
    st.error(f"Uploads too large ({mb:.1f} MB total). Limit ≈ {MAX_TOTAL_BYTES/(1024*1024):.0f} MB.")
    st.stop()

for f in up_files:
    docs.append((f.name, f.read()))

if not docs:
    st.info("Upload PDFs to begin analysis.")
    st.stop()

# ---------- Fresh run per new upload set ----------
current_sig = _files_signature(docs)  # docs = [(name, bytes)]
if current_sig != st.session_state.last_sig:
    st.session_state.last_sig = current_sig
    st.session_state.run_id = hashlib.sha1(current_sig.encode()).hexdigest()[:12]
    st.cache_data.clear()

# -------- Build heading mapping UI (quick 2-page probe) --------
candidate_heads = set()
for fname, blob in docs:
    fh = _sha1_bytes(blob)
    pages_tmp, _ = cached_extract_pages(fh, st.session_state.run_id, blob, dpi=dpi, max_pages=2, lang=lang)
    for ptxt in pages_tmp:
        for h in detect_heading_lines(ptxt):
            candidate_heads.add(h)

st.sidebar.markdown("### Review detected headings")
with st.sidebar.expander("Map headings to dimensions", expanded=False):
    exact_map = {}
    heads_list = sorted(candidate_heads, key=len, reverse=True)[:60]
    for h in heads_list:
        choice = st.selectbox(f"Map: {h}", ["(auto)"] + DIMENSIONS, index=0, key=f"map_{h}")
        if choice != "(auto)":
            exact_map[re.escape(h)] = choice

    force_mapped = st.checkbox("Force mapped dimension (instead of boost)", value=False)
    boost_strength = st.slider("Boost strength (if not forcing)", 0, 10, 5, 1)

    colA, colB, _ = st.columns(3)
    with colA:
        if st.button("Apply mappings"):
            st.session_state.headmap.update(exact_map)
            st.success("Mappings applied.")
    with colB:
        if st.button("Reset to defaults"):
            st.session_state.headmap = DEFAULT_HEADMAP.copy()
            st.success("Mappings reset.")

with st.sidebar.expander("Save / Load mappings (JSON)", expanded=False):
    col1, col2 = st.columns(2)
    with col1:
        headmap_bytes = json.dumps(st.session_state.headmap, indent=2).encode("utf-8")

    with col2:
        up_map = st.file_uploader("Load headmap.json", type=["json"], accept_multiple_files=False)
        if up_map is not None:
            try:
                new_map = json.load(up_map)
                if isinstance(new_map, dict):
                    st.session_state.headmap = new_map
                    st.success("Mappings loaded.")
                else:
                    st.error("Invalid JSON structure. Expecting an object/dict.")
            except Exception as e:
                st.error(f"Failed to load JSON: {e}")

# -------- Prepare AI handles (once) --------
sbert_model = None
label_embeds = None
zshot_pipe = None
if ai_mode in {"Semantic (SBERT)", "Ensemble"} and HAVE_SBERT:
    sbert_model = load_sbert(sem_model_name)
    if sbert_model is not None:
        prompts = build_dim_prompts_from_keymap(KEYMAP, top_n=12)
        label_texts = [prompts[d] for d in DIMENSIONS]
        try:
            label_embeds = sbert_model.encode(label_texts, convert_to_tensor=True, normalize_embeddings=True)
        except Exception:
            label_embeds = None

if ai_mode in {"Zero-shot (NLI)", "Ensemble"} and HAVE_ZSHOT:
    zshot_pipe = load_zeroshot(z_model_name)

# -------- Process documents --------
matrix_rows: List[Dict[str, Any]] = []
evidence_tables: List[DataFrame] = []
diagnostics_rows: List[Dict[str, Any]] = []

progress = st.progress(0.0)
status = st.empty()

with st.spinner("Extracting text, classifying sentences, and building audit trail…"):
    total_docs = len(docs)
    for idx, (fname, blob) in enumerate(docs, start=1):
        status.write(f"Processing {idx}/{total_docs}… {fname}")
        fh = _sha1_bytes(blob)
        pages, method = cached_extract_pages(fh, st.session_state.run_id, blob, dpi=dpi, max_pages=max_pages, lang=lang)
        if preview:
            for i, ptxt in enumerate(pages[:3], start=1):
                snippet = (ptxt[:1500] + "…") if len(ptxt) > 1500 else ptxt
                st.expander(f"Preview: {fname} • page {i}").write(snippet)

        if not pages:
            mrow = {"Document": Path(fname).stem}
            for d in DIMENSIONS: mrow[d] = 0.0
            matrix_rows.append(mrow)
            ev_df = pd.DataFrame([{
                "Document": Path(fname).stem, "Dimension": d, "Evidence (sentence)": "(no text extracted)",
                "Page": None, "Confidence": 0.0, "Method": "rules"
            } for d in DIMENSIONS])
            evidence_tables.append(ev_df)
            diagnostics_rows.append({"Document": Path(fname).stem, "chars": 0, "method": method})
            progress.progress(idx / total_docs)
            continue

        evidence, hit_counts, total_chars = gather_evidence(
            pages,
            headmap=st.session_state.headmap,
            boost_strength=boost_strength,
            force_mapped=force_mapped,
            ai_mode=ai_mode,
            sbert_model=sbert_model,
            label_embeds=label_embeds,
            zshot_pipe=zshot_pipe,
            zshot_thresh=zshot_thresh,
            sem_thresh=sem_thresh,
            z_boost=z_boost,
            sem_boost=sem_boost,
            top_k_per_dim=3
        )

        scores: Dict[str, float] = {}
        for dim in DIMENSIONS:
            hits = hit_counts[dim]
            if scoring_mode.startswith("Curved"):
                scores[dim] = score_curved(hits, total_chars, tau=tau)
            elif scoring_mode.startswith("Length"):
                scores[dim] = score_length_normalized(hits, total_chars, pivot_chars=pivot_chars)
            else:
                scores[dim] = score_classic(hits)

        matrix_row: Dict[str, Any] = {"Document": Path(fname).stem}
        matrix_row.update(scores)
        matrix_rows.append(matrix_row)

        rows = []
        for dim in DIMENSIONS:
            ev = evidence[dim]
            if ev:
                for s, pg, conf, method_tag in ev:
                    rows.append({
                        "Document": Path(fname).stem,
                        "Dimension": dim,
                        "Evidence (sentence)": s,
                        "Page": pg,
                        "Confidence": round(conf, 2),
                        "Method": method_tag
                    })
            else:
                rows.append({
                    "Document": Path(fname).stem,
                    "Dimension": dim,
                    "Evidence (sentence)": "(no strong sentence detected)",
                    "Page": None,
                    "Confidence": 0.0,
                    "Method": "rules" if ai_mode == "Off" else ("rules+" + ("ensemble" if ai_mode=="Ensemble" else ("semantic" if ai_mode.startswith("Semantic") else "zeroshot")))
                })
        ev_df = pd.DataFrame(rows, columns=["Document","Dimension","Evidence (sentence)","Page","Confidence","Method"])
        evidence_tables.append(ev_df)

        diag = {
            "Document": Path(fname).stem,
            "method": method,
            "chars": total_chars,
            "AI mode": ai_mode,
            "SBERT model": (sem_model_name if sbert_model else None),
            "Z-shot model": (z_model_name if zshot_pipe else None),
        }
        for dim in DIMENSIONS:
            diag[f"{dim} hits"] = hit_counts[dim]
        diagnostics_rows.append(diag)

        progress.progress(idx / total_docs)

status.write("Processing complete.")
st.success(f"Done. Processed {len(docs)} document(s).")

matrix_df = pd.DataFrame(matrix_rows, columns=["Document"] + DIMENSIONS)
evidence_df = pd.concat(evidence_tables, ignore_index=True)
diagnostics_df = pd.DataFrame(diagnostics_rows)

# -------- Outputs --------
st.subheader("Synthesis Matrix (0–10)")
st.caption("Table view of coverage per document × dimension.")
st.dataframe(matrix_df, use_container_width=True)

st.subheader("Evidence with Page Numbers (for audit)")
st.dataframe(evidence_df, use_container_width=True, height=420)

st.subheader("Heatmap")
heat = matrix_df.set_index("Document")[DIMENSIONS]
if HAS_MPL:
    st.dataframe(heat.style.background_gradient(axis=None), use_container_width=True)
else:
    st.info("matplotlib is not installed. Showing unstyled table instead. To enable colored heatmap: pip install matplotlib")
    st.dataframe(heat, use_container_width=True)

st.subheader("Heatmap Interpretation")
interp = interpret_heatmap(matrix_df)
st.markdown(interp)

# ---------- Criteria table + hidden editor ----------
st.subheader("Criteria (keywords)")
crit_df = pd.DataFrame({
    "Dimension": DIMENSIONS,
    "Keywords": [", ".join(KEYMAP[d]) for d in DIMENSIONS]
})
st.dataframe(crit_df, use_container_width=True, height=220)
st.download_button("Download criteria (CSV)", crit_df.to_csv(index=False).encode(), file_name="criteria_keywords.csv", use_container_width=True, key="criteria_csv")

with st.expander("Edit / override criteria (optional)", expanded=False):
    st.markdown("Update the keyword lists used for classification. Changes persist during this session.")
    dsel = st.selectbox("Select dimension", DIMENSIONS)
    mode = st.radio("Apply as", ["Append", "Replace"], index=0, horizontal=True)
    current = ", ".join(st.session_state.keymap[dsel])
    text = st.text_area("Comma-separated keywords/phrases", value=current, height=120)
    colE1, colE2, colE3 = st.columns(3)
    with colE1:
        if st.button("Apply to selected"):
            tokens = [t.strip().lower() for t in text.split(",") if t.strip()]
            if mode == "Replace":
                st.session_state.keymap[dsel] = sorted(set(tokens))
            else:
                merged = set(st.session_state.keymap[dsel])
                merged.update(tokens)
                st.session_state.keymap[dsel] = sorted(merged)
            st.success(f"Updated keywords for: {dsel}")
    with colE2:
        if st.button("Reset ALL to defaults"):
            st.session_state.keymap = {d: sorted(set(map(str.lower, BASE_KEYMAP[d]))) for d in DIMENSIONS}
            st.session_state.custom_v3_seed_applied = False
            st.success("All dimensions reset to base defaults.")
    with colE3:
        st.write("")

    st.markdown("---")
    st.caption("Paste 5 groups per line (Deprivations | Causes | Barriers | Interventions | UNICEF advantage).")
    block = st.text_area("Bulk paste (v3-style)", placeholder="• learning loss, mental health, ... \t • climate change, ... \t • ... \t • ... \t • ...", height=120)
    mode2 = st.radio("Apply bulk as", ["Append", "Replace (all dims)"], index=0, horizontal=True, key="bulkmode")
    if st.button("Apply bulk"):
        parsed = parse_custom_block(block)
        if mode2.startswith("Replace"):
            st.session_state.keymap = parsed
        else:
            for d in DIMENSIONS:
                merged = set(st.session_state.keymap[d])
                merged.update(parsed.get(d, []))
                st.session_state.keymap[d] = sorted(merged)
        st.success("Bulk keywords applied.")

    st.markdown("---")
    colJ1, colJ2 = st.columns(2)
    with colJ1:
        keymap_bytes = json.dumps(st.session_state.keymap, indent=2).encode("utf-8")
        st.download_button(
            "💾 Download criteria keymap.json",
            data=keymap_bytes,
            file_name="keymap.json",
            mime="application/json",
            use_container_width=True,
            key="download_keymap_json",
        )
    with colJ2:
        upk = st.file_uploader("Load criteria JSON", type=["json"], accept_multiple_files=False)
        if upk is not None:
            try:
                newk = json.load(upk)
                if isinstance(newk, dict):
                    st.session_state.keymap = {d: sorted(set(map(str.lower, newk.get(d, [])))) for d in DIMENSIONS}
                    st.success("Criteria JSON loaded.")
                else:
                    st.error("Invalid JSON structure.")
            except Exception as e:
                st.error(f"Failed to load JSON: {e}")

st.subheader("Diagnostics (extraction + raw hits)")
st.dataframe(diagnostics_df, use_container_width=True)

# -------- Export --------
st.subheader("Export")
col1, col2 = st.columns(2)
with col1:
    excel_bytes = export_to_excel({
        "Matrix": matrix_df,
        "Evidence": evidence_df,
        "Diagnostics": diagnostics_df,
        "Mappings": pd.DataFrame([{"pattern": k, "dimension": v} for k, v in st.session_state.headmap.items()]),
        "Criteria": pd.DataFrame([{"dimension": d, "keywords": ", ".join(st.session_state.keymap[d])} for d in DIMENSIONS]),
        "AI_Settings": pd.DataFrame([{
            "AI mode": ai_mode,
            "SBERT model": sem_model_name if sbert_model else None,
            "Zero-shot model": z_model_name if zshot_pipe else None,
            "Semantic threshold": sem_thresh,
            "Zero-shot threshold": zshot_thresh,
            "Semantic boost": sem_boost,
            "Zero-shot boost": z_boost,
        }]),
    })
    st.download_button(
        "⬇️ Download Excel (Matrix + Evidence + Mappings + Diagnostics + Criteria + AI Settings)",
        data=excel_bytes,
        file_name="Evidence_Synthesis_Audit_v43_AI.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
        key="download_excel_pack",
    )
with col2:
    st.caption("Need a Word/PPT export? I can add that: gmashaka@unicef.org")

# --- Programmatic launcher so the PyInstaller-built EXE starts Streamlit ---
if __name__ == "__main__":
    try:
        # Streamlit 1.30+ way
        from streamlit.web.bootstrap import run as st_run
        import os, sys, pathlib
        app_path = pathlib.Path(__file__).resolve()
        # Equivalent to: streamlit run app1.py
        st_run(str(app_path), "", [], flag_options={})
    except Exception:
        # Fallback for older versions
        import sys as _sys
        import streamlit.web.cli as stcli
        _sys.argv = ["streamlit", "run", __file__]
        _sys.exit(stcli.main())
