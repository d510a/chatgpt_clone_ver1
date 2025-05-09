# app3.py – SAFE uploader version (2025‑05‑09)
"""Streamlit ChatGPT‑like app with PDF/Word extraction.
• Displays greeting "質問してみましょう"
• Safe file_uploader that auto‑clears stale SessionState to avoid
  StreamlitAPIException on deserialize.
"""

import os
import json
import logging
import sys
from io import BytesIO
from pathlib import Path
from typing import Final, List

import importlib.metadata as imd
import streamlit as st
from dotenv import load_dotenv
from streamlit.errors import StreamlitAPIException

# extraction libs
from docx import Document
import mammoth
import docx2txt

# ----------------------------- 0) settings ----------------------------
load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s ▶ %(message)s")

username = password = api_key = proxy_host = ""

if "api_key" in st.secrets:  # cloud
    api_key = st.secrets["api_key"]
    username = st.secrets.get("proxy_username", "")
    password = st.secrets.get("proxy_password", "")
    proxy_host = st.secrets.get("proxy_host", "proxy01.hm.jp.honda.com:8080")
else:  # local proxy_config.json
    cfg_file = Path.home() / "Documents" / "proxy_config.json"
    if cfg_file.exists():
        try:
            with cfg_file.open(encoding="utf-8") as f:
                cfg = json.load(f)
            api_key = cfg.get("apikey", "")
            username = cfg.get("username", "")
            password = cfg.get("password", "")
            proxy_host = cfg.get("proxyhost", "proxy01.hm.jp.honda.com:8080")
        except Exception as e:
            logging.warning("proxy_config.json read failed: %s", e)

proxy_url = f"http://{username}:{password}@{proxy_host}" if username and password and proxy_host else None

st.set_page_config(page_title="ChatGPT_clone")
if not api_key:
    st.error("APIキーが設定されていません。Secrets または proxy_config.json を確認してください。")
    st.stop()

# -------------------- 1) OpenAI wrapper (v0/v1) ----------------------

def openai_v1() -> bool:
    try:
        return int(imd.version("openai").split(".")[0]) >= 1
    except Exception:
        return False

_V1 = openai_v1()
if _V1:
    from openai import OpenAI
else:
    import openai as _openai_legacy  # noqa: F401


class OpenAIWrapper:
    def __init__(self, key: str, proxy: str | None):
        if proxy:
            os.environ["HTTP_PROXY"] = os.environ["HTTPS_PROXY"] = proxy
        else:
            os.environ.pop("HTTP_PROXY", None)
            os.environ.pop("HTTPS_PROXY", None)
        if _V1:
            self.client = OpenAI(api_key=key)
            self.v1 = True
        else:
            import openai
            openai.api_key = key
            self.client = openai
            self.v1 = False

    def list_models(self):
        return self.client.models.list() if self.v1 else self.client.Model.list()

    def stream_chat(self, messages, model="o3-2025-04-16"):
        if self.v1:
            return self.client.chat.completions.create(model=model, messages=messages, stream=True)
        return self.client.ChatCompletion.create(model=model, messages=messages, stream=True)


def get_openai(key: str) -> OpenAIWrapper:
    w = OpenAIWrapper(key, proxy_url)
    try:
        w.list_models()
        logging.info("OpenAI connectivity OK (%s)", "proxy" if proxy_url else "direct")
        return w
    except Exception:
        return OpenAIWrapper(key, None)


client = get_openai(api_key)

# -------------------- 2) resources (OCR binaries) --------------------
BASE_DIR = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
POPPLER_DIR = BASE_DIR / "poppler" / "bin"
TESSERACT_EXE = BASE_DIR / "tesseract" / "tesseract.exe"

# -------------------- 3) session init --------------------------------
GREETING = "質問してみましょう"
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "assistant", "content": GREETING},
    ]

st.session_state.setdefault("uploaded_files", {})

# -------------------- 4) safe file_uploader --------------------------
ALLOWED_EXTS: List[str] = ["txt", "md", "pdf", "docx", "doc"]


def safe_file_uploader(*, key: str, **kwargs):
    """Wrap st.file_uploader; clear stale SessionState if deserialize fails."""
    if key in st.session_state:
        try:
            _ = st.session_state[key]
        except StreamlitAPIException:
            st.session_state.pop(key, None)
    return st.file_uploader(key=key, **kwargs)


st.sidebar.header("ファイルを添付")
uploaded = safe_file_uploader(
    label="テキスト / Markdown / PDF / Word",
    type=ALLOWED_EXTS,
    accept_multiple_files=False,
    key="file_up",
)

# -------------------- 5) extraction helpers --------------------------

def read_text(file) -> str:
    raw = file.read()
    for enc in ("utf-8", "cp932"):
        try:
            return raw.decode(enc, errors="ignore")[:180_000]
        except UnicodeDecodeError:
            continue
    return raw.decode(errors="ignore")[:180_000]


def pdf_text(f) -> str:
    data = f.read(); bio = BytesIO(data)
    try:
        from pdfminer.high_level import extract_text
        t = extract_text(bio)
        if t.strip():
            return t[:180_000]
    except Exception:
        pass
    try:
        import PyPDF2
        reader = PyPDF2.PdfReader(BytesIO(data))
        t = "\n".join(p.extract_text() or "" for p in reader.pages)
        if t.strip():
            return t[:180_000]
    except Exception:
        pass
    try:
        import fitz
        doc = fitz.open(stream=data, filetype="pdf")
        t = "\n".join(p.get_text() for p in doc)
        if t.strip():
            return t[:180_000]
    except Exception:
        pass
    try:
        from pdf2image import convert_from_bytes
        import pytesseract
        pages = convert_from_bytes(data, dpi=300, fmt="png", poppler_path=str(POPPLER_DIR))
        pytesseract.pytesseract.tesseract_cmd = str(TESSERACT_EXE)
        t = "\n".join(pytesseract.image_to_string(p, lang="jpn") for p in pages)
        if t.strip():
            return t[:180_000]
    except Exception:
        pass
    return "(PDF からテキストを抽出できませんでした)"


def docx_text(f) -> str:
    data = f.read(); bio = BytesIO(data)
    try:
        d = Document(bio)
        return "\n".join(p.text for p in d.paragraphs)[:180_000]
    except Exception:
        pass
    try:
        return docx2txt.process(bio)[:180_000]
    except Exception:
        pass
    return "(Word(.docx) から抽出できませんでした)"


def doc_text(f) -> str:
    data = f.read(); bio = BytesIO(data)
    try:
        res = mammoth.extract_raw_text(bio)
        return res.value[:180_000]
    except Exception:
        pass
    return "(.doc から抽出できませんでした)"

# handle upload
if uploaded:
    st.sidebar.write(f"**{uploaded.name}** を読み込みました")
    if uploaded.name not in st.session_state.uploaded_files:
        ext = Path(uploaded.name).suffix.lower()
        if ext == ".pdf":
            extracted = pdf_text(uploaded)
        elif ext == ".docx":
            extracted = docx_text(uploaded)
        elif ext == ".doc":
            extracted = doc_text(uploaded)
        else:
            extracted = read_text(uploaded)
        st.session_state.uploaded_files[uploaded.name] = extracted

    if st.sidebar.button("ファイル内容を送信"):
        txt = st.session_state.uploaded_files[uploaded.name]
        st.session_state.messages.append({"role": "system", "content": txt})
        st.session_state.messages.append({"role": "user", "content": f"ファイル **{uploaded.name}** を送信しました。"})
        st.sidebar.success("ファイルをチャ
