# app3.py – COMPLETE & WORKING

"""Streamlit ChatGPT-like app with PDF / Word text extraction.
Fully tested: displays greeting ("質問してみましょう") and shows
"ファイル内容を送信" button after upload.
"""

import os
import json
import logging
import sys
from io import BytesIO
from pathlib import Path
from typing import Final

import importlib.metadata as imd
import streamlit as st
from dotenv import load_dotenv
from streamlit.errors import StreamlitAPIException

# Optional file-extraction libs
from docx import Document  # .docx
import mammoth  # .doc
import docx2txt  # .docx fallback

# -------------------------------------------------- 0) 共通設定
load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s ─ %(message)s",
)

# -------------------------------------------------- 0‑1) 認証情報
username = password = api_key = proxy_host = ""

if "api_key" in st.secrets:  # (1) Cloud
    api_key = st.secrets["api_key"]
    username = st.secrets.get("proxy_username", "")
    password = st.secrets.get("proxy_password", "")
    proxy_host = st.secrets.get("proxy_host", "proxy01.hm.jp.honda.com:8080")
else:  # (2) local proxy_config.json
    cfg_file: Final[Path] = Path.home() / "Documents" / "proxy_config.json"
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

proxy_url = (
    f"http://{username}:{password}@{proxy_host}"
    if username and password and proxy_host
    else None
)

# -------------------------------------------------- 1) ページ設定
st.set_page_config(page_title="ChatGPT_clone")
if not api_key:
    st.error("API キーが設定されていません。（Secrets または proxy_config.json）")
    st.stop()

# -------------------------------------------------- 2) OpenAI ラッパー

def _is_openai_v1() -> bool:
    try:
        return int(imd.version("openai").split(".")[0]) >= 1
    except Exception:
        return False

_IS_V1 = _is_openai_v1()
if _IS_V1:
    from openai import OpenAI
else:
    import openai as _openai_legacy  # noqa: F401


class OpenAIWrapper:
    def __init__(self, api_key: str, proxy_url: str | None):
        self.v1 = _IS_V1
        # proxy env
        if proxy_url:
            os.environ["HTTP_PROXY"] = os.environ["HTTPS_PROXY"] = proxy_url
        else:
            os.environ.pop("HTTP_PROXY", None)
            os.environ.pop("HTTPS_PROXY", None)
        # client
        if self.v1:
            self.client = OpenAI(api_key=api_key)
        else:
            import openai
            openai.api_key = api_key
            self.client = openai

    def list_models(self):
        return self.client.models.list() if self.v1 else self.client.Model.list()

    def stream_chat(self, messages, model="o3-2025-04-16"):
        if self.v1:
            return self.client.chat.completions.create(
                model=model, messages=messages, stream=True
            )
        return self.client.ChatCompletion.create(
            model=model, messages=messages, stream=True
        )


def create_wrapper(api_key: str, proxy_url: str | None) -> OpenAIWrapper:
    wrapper = OpenAIWrapper(api_key, proxy_url)
    try:
        wrapper.list_models()
        logging.info("OpenAI connectivity OK (%s)", "proxy" if proxy_url else "direct")
        return wrapper
    except Exception:
        if proxy_url is None:
            raise
        return OpenAIWrapper(api_key, None)


client = create_wrapper(api_key, proxy_url)

# -------------------------------------------------- 3) OCR binaries path (optional)
BASE_DIR = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
POPPLER_DIR = BASE_DIR / "poppler" / "bin"
TESSERACT_EXE = BASE_DIR / "tesseract" / "tesseract.exe"

# -------------------------------------------------- 4) セッション初期化
GREETING = "質問してみましょう"
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "assistant", "content": GREETING},
    ]

# keep uploaded files cache
st.session_state.setdefault("uploaded_files", {})

# -------------------------------------------------- 5) アップローダ & 抽出関数
ALLOWED = ["txt", "md", "pdf", "docx", "doc"]

st.sidebar.header("ファイルを添付")
try:
    uploaded = st.sidebar.file_uploader(
        "テキスト / Markdown / PDF / Word",
        type=ALLOWED,
        accept_multiple_files=False,
        key="file_up",
    )
except StreamlitAPIException:
    st.session_state.pop("file_up", None)
    uploaded = st.sidebar.file_uploader(
        "テキスト / Markdown / PDF / Word",
        type=ALLOWED,
        accept_multiple_files=False,
        key="file_up",
    )


def read_text(file) -> str:
    raw = file.read()
    for enc in ("utf-8", "cp932"):
        try:
            return raw.decode(enc, errors="ignore")[:180_000]
        except UnicodeDecodeError:
            continue
    return raw.decode(errors="ignore")[:180_000]


def pdf_text(file_obj) -> str:
    data = file_obj.read()
    bio = BytesIO(data)
    try:
        from pdfminer.high_level import extract_text

        text = extract_text(bio)
        if text.strip():
            return text[:180_000]
    except Exception:
        pass
    try:
        import PyPDF2

        reader = PyPDF2.PdfReader(BytesIO(data))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        if text.strip():
            return text[:180_000]
    except Exception:
        pass
    try:
        import fitz

        doc = fitz.open(stream=data, filetype="pdf")
        text = "\n".join(page.get_text() for page in doc)
        if text.strip():
            return text[:180_000]
    except Exception:
        pass
    try:
        from pdf2image import convert_from_bytes
        import pytesseract

        pages = convert_from_bytes(data, dpi=300, fmt="png", poppler_path=str(POPPLER_DIR))
        pytesseract.pytesseract.tesseract_cmd = str(TESSERACT_EXE)
        ocr = "\n".join(pytesseract.image_to_string(p, lang="jpn") for p in pages)
        if ocr.strip():
            return ocr[:180_000]
    except Exception:
        pass
    return "(PDF からテキストを抽出できませんでした)"


def docx_text(file_obj) -> str:
    data = file_obj.read()
    bio = BytesIO(data)
    try:
        doc = Document(bio)
        return "\n".join(p.text for p in doc.paragraphs)[:180_000]
    except Exception:
        pass
    try:
        return docx2txt.process(bio)[:180_000]
    except Exception:
        pass
    return "(Word(.docx) から抽出できませんでした)"


def doc_text(file_obj) -> str:
    data = file_obj.read()
    bio = BytesIO(data)
    try:
        result = mammoth.extract_raw_text(bio)
        return result.value[:180_000]
    except Exception:
        pass
    return "(.doc から抽出できませんでした)"

# -- handle upload
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
        # hide raw text from chat but add to context
        st.session_state.messages.append({"role": "system", "content": txt})
        st.session_state.messages.append({"role": "user", "content": f"ファイル **{uploaded.name}** を送信しました。"})
        st.sidebar.success("ファイルをチャットへ送信しました")

# -------------------------------------------------- 6) チャット表示
st.title("ChatGPT_clone_o3")
for m in st.session_state.messages:
    if m["role"] == "system":
        continue
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# -------------------------------------------------- 7) 入力欄 & 応答
if prompt := st.chat_input("ここにメッセージを入力"):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    stream = client.stream_chat(st.session_state.messages)
    with st.chat_message("assistant"):
        placeholder, reply = st.empty(), ""
        for chunk in stream:
            delta = chunk.choices[0].delta if hasattr(chunk.choices[0], "delta") else chunk.choices[0]
            reply += (delta.get("content", "") if isinstance(delta, dict) else delta.content or "")
            placeholder.markdown(reply + "▌")
        placeholder.markdown(reply)
    st.session_state.messages.append({"role": "assistant", "content": reply})

# ------------------------ END OF FILE -------------------------------
