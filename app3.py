
# app3.py — FULL WORKING VERSION (2025‑05‑09)
"""
Streamlit ChatGPT‑like app that can read plain text, Markdown, PDF, DOCX, and
legacy DOC files, extract up to 180 kB of text, and feed it to OpenAI chat
completions.

Key points
-----------
1. **safe_file_uploader** — purges stale widget state to prevent
   `StreamlitAPIException` after code updates.
2. Initial assistant greeting **「質問してみましょう」** always shows.
3. Side‑bar button **「ファイル内容を送信」** appears after upload.
4. Works with both *openai* v0 (legacy) and v1 libraries.
5. Multiple PDF fallbacks + OCR (Poppler/Tesseract) included.
"""

from __future__ import annotations

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

# extraction libraries
from docx import Document  # DOCX
import mammoth             # DOC → text
import docx2txt            # DOCX fallback

# ────────────────────────── 0) 基本設定 ───────────────────────────
load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s ▸ %(message)s")

username = password = api_key = proxy_host = ""

# ❶ Cloud: Streamlit Secrets
if "api_key" in st.secrets:
    api_key = st.secrets["api_key"]
    username = st.secrets.get("proxy_username", "")
    password = st.secrets.get("proxy_password", "")
    proxy_host = st.secrets.get("proxy_host", "proxy01.hm.jp.honda.com:8080")
# ❷ Local: proxy_config.json
else:
    cfg_path = Path.home() / "Documents" / "proxy_config.json"
    if cfg_path.exists():
        try:
            with cfg_path.open(encoding="utf-8") as f:
                cfg = json.load(f)
            api_key = cfg.get("apikey", "")
            username = cfg.get("username", "")
            password = cfg.get("password", "")
            proxy_host = cfg.get("proxyhost", "proxy01.hm.jp.honda.com:8080")
        except Exception as e:
            logging.warning("proxy_config.json 読み込み失敗: %s", e)

proxy_url = f"http://{username}:{password}@{proxy_host}" if username and password and proxy_host else None

st.set_page_config(page_title="ChatGPT_clone")
if not api_key:
    st.error("APIキーが設定されていません。Secrets または proxy_config.json を確認してください。")
    st.stop()

# ───────────────── 1) OpenAI ラッパー ────────────────────────

def _is_openai_v1() -> bool:
    try:
        return int(imd.version("openai").split(".")[0]) >= 1
    except Exception:
        return False

_OPENAI_V1 = _is_openai_v1()
if _OPENAI_V1:
    from openai import OpenAI
else:
    import openai as _openai_legacy  # noqa: F401


class OpenAIWrapper:
    """Unifies v0 / v1 APIs with optional proxy fail‑over."""

    def __init__(self, key: str, proxy: str | None):
        # proxy env
        if proxy:
            os.environ["HTTP_PROXY"] = os.environ["HTTPS_PROXY"] = proxy
        else:
            os.environ.pop("HTTP_PROXY", None)
            os.environ.pop("HTTPS_PROXY", None)
        # client
        if _OPENAI_V1:
            self.client = OpenAI(api_key=key)
            self.v1 = True
        else:
            import openai
            openai.api_key = key
            self.client = openai
            self.v1 = False

    def list_models(self):
        return self.client.models.list() if self.v1 else self.client.Model.list()

    def stream_chat(self, messages, model: str = "o3-2025-04-16"):
        if self.v1:
            return self.client.chat.completions.create(model=model, messages=messages, stream=True)
        return self.client.ChatCompletion.create(model=model, messages=messages, stream=True)


def get_openai_wrapper(key: str) -> OpenAIWrapper:
    w = OpenAIWrapper(key, proxy_url)
    try:
        w.list_models()
        logging.info("OpenAI 接続確認 OK (%s)", "proxy" if proxy_url else "direct")
        return w
    except Exception:
        logging.info("proxy 経由失敗。直接接続を再試行…")
        return OpenAIWrapper(key, None)


client = get_openai_wrapper(api_key)

# ───────────────── 2) OCR 実行ファイルパス ───────────────────
BASE_DIR = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
POPPLER_DIR = BASE_DIR / "poppler" / "bin"
TESSERACT_EXE = BASE_DIR / "tesseract" / "tesseract.exe"

# ───────────────── 3) セッション初期化 ──────────────────────
GREETING = "質問してみましょう"
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "assistant", "content": GREETING},
    ]

st.session_state.setdefault("uploaded_files", {})

# ───────────────── 4) 安全な file_uploader ────────────────────
ALLOWED_EXTS: List[str] = ["txt", "md", "pdf", "docx", "doc"]


def safe_file_uploader(label: str, *, key: str, **kwargs):
    """Wrap st.file_uploader; clear stale state to avoid deserialize errors."""
    if key in st.session_state:
        try:
            _ = st.session_state[key]
        except StreamlitAPIException:
            st.session_state.pop(key, None)
    return st.file_uploader(label, key=key, **kwargs)


st.sidebar.header("ファイルを添付")
uploaded_file = safe_file_uploader(
    "テキスト / Markdown / PDF / Word",
    type=ALLOWED_EXTS,
    accept_multiple_files=False,
    key="file_up",
)

# ───────────────── 5) 抽出ヘルパ関数 ────────────────────────

def read_text(file) -> str:
    raw = file.read()
    for enc in ("utf-8", "cp932"):
        try:
            return raw.decode(enc, errors="ignore")[:180_000]
        except UnicodeDecodeError:
            continue
    return raw.decode(errors="ignore")[:180_000]


def pdf_text(file_obj) -> str:
    data = file_obj.read(); bio = BytesIO(data)
    try:
        from pdfminer.high_level import extract_text
        txt = extract_text(bio)
        if txt.strip():
            return txt[:180_000]
    except Exception:
        pass
    try:
        import PyPDF2
        reader = PyPDF2.PdfReader(BytesIO(data))
        txt = "\n".join(p.extract_text() or "" for p in reader.pages)
        if txt.strip():
            return txt[:180_000]
    except Exception:
        pass
    try:
        import fitz
        doc = fitz.open(stream=data, filetype="pdf")
        txt = "\n".join(p.get_text() for p in doc)
        if txt.strip():
            return txt[:180_000]
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
    data = file_obj.read(); bio = BytesIO(data)
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
    data = file_obj.read(); bio = BytesIO(data)
    try:
        res = mammoth.extract_raw_text(bio)
        return res.value[:180_000]
    except Exception:
        pass
    return "(.doc から


# -------------------------- 6) ファイルアップロード処理 --------------------------
ALLOWED_EXTS = ["txt", "md", "pdf", "docx", "doc"]

def safe_file_uploader(label: str, **kwargs):
    """
    SessionState に残った旧ウィジェット情報が原因で
    StreamlitAPIException が出る場合に自動クリアして再生成するラッパー。
    """
    from streamlit.errors import StreamlitAPIException

    key = kwargs.get("key")
    if key and key in st.session_state:
        try:
            _ = st.session_state[key]          # 復元テスト
        except StreamlitAPIException:
            st.session_state.pop(key, None)    # 壊れた状態をクリア

    return st.file_uploader(label, **kwargs)

uploaded_file = safe_file_uploader(
    "テキスト / Markdown / PDF / Word",
    type=ALLOWED_EXTS,
    accept_multiple_files=False,
    key="file_uploader",
)

if uploaded_file:
    st.sidebar.write(
        f" **{uploaded_file.name}** "
        f"({uploaded_file.size // 1024:,} KB) を読み込みました"
    )
    if uploaded_file.name not in st.session_state.uploaded_files:
        ext = Path(uploaded_file.name).suffix.lower().lstrip(".")
        if ext == "pdf":
            content = extract_text_from_pdf(uploaded_file)
        elif ext == "docx":
            content = extract_text_from_docx(uploaded_file)
        elif ext == "doc":
            content = extract_text_from_doc(uploaded_file)
        else:                                  # txt / md
            content = read_text_file(uploaded_file)
        st.session_state.uploaded_files[uploaded_file.name] = content

    # ------- 送信ボタン（1 つだけ） -------
    if st.sidebar.button("ファイル内容を送信", key="send_file"):
        txt = st.session_state.uploaded_files[uploaded_file.name]
        # バックグラウンドの system メッセージとして追加
        st.session_state.messages.append({"role": "system", "content": txt})
        # チャット欄には通知だけ表示
        notice = f"ファイル **{uploaded_file.name}** を送信しました。"
        st.session_state.messages.append({"role": "user", "content": notice})
        st.sidebar.success("ファイルをチャットへ送信しました")

# -------------------------- 7) チャット表示 --------------------------
st.title("ChatGPT_clone_o3")
st.caption("Streamlit + OpenAI")

for m in st.session_state.messages:
    if m["role"] == "system":      # system は非表示
        continue
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# -------------------------- 8) ユーザー入力 --------------------------
if prompt := st.chat_input("ここにメッセージを入力"):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # OpenAI ストリーミング応答
    stream = client.stream_chat_completion(
        messages=st.session_state.messages,
        model="o3-2025-04-16",
    )

    with st.chat_message("assistant"):
        placeholder, reply = st.empty(), ""
        for chunk in stream:
            delta = (
                chunk.choices[0].delta
                if hasattr(chunk.choices[0], "delta")
                else chunk.choices[0]
            )
            reply += (
                delta.get("content", "")
                if isinstance(delta, dict)
                else (delta.content or "")
            )
            placeholder.markdown(reply + "▌")
        placeholder.markdown(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})

# -------------------------- 6) ファイルアップロード処理 --------------------------
ALLOWED_EXTS = ["txt", "md", "pdf", "docx", "doc"]

def safe_file_uploader(label: str, **kwargs):
    """
    SessionState に残った旧ウィジェット情報が原因で
    StreamlitAPIException が出る場合に自動クリアして再生成するラッパー。
    """
    from streamlit.errors import StreamlitAPIException

    key = kwargs.get("key")
    if key and key in st.session_state:
        try:
            _ = st.session_state[key]          # 復元テスト
        except StreamlitAPIException:
            st.session_state.pop(key, None)    # 壊れた状態をクリア

    return st.file_uploader(label, **kwargs)

uploaded_file = safe_file_uploader(
    "テキスト / Markdown / PDF / Word",
    type=ALLOWED_EXTS,
    accept_multiple_files=False,
    key="file_uploader",
)

if uploaded_file:
    st.sidebar.write(
        f" **{uploaded_file.name}** "
        f"({uploaded_file.size // 1024:,} KB) を読み込みました"
    )
    if uploaded_file.name not in st.session_state.uploaded_files:
        ext = Path(uploaded_file.name).suffix.lower().lstrip(".")
        if ext == "pdf":
            content = extract_text_from_pdf(uploaded_file)
        elif ext == "docx":
            content = extract_text_from_docx(uploaded_file)
        elif ext == "doc":
            content = extract_text_from_doc(uploaded_file)
        else:                                  # txt / md
            content = read_text_file(uploaded_file)
        st.session_state.uploaded_files[uploaded_file.name] = content

    # ------- 送信ボタン（1 つだけ） -------
    if st.sidebar.button("ファイル内容を送信", key="send_file"):
        txt = st.session_state.uploaded_files[uploaded_file.name]
        # バックグラウンドの system メッセージとして追加
        st.session_state.messages.append({"role": "system", "content": txt})
        # チャット欄には通知だけ表示
        notice = f"ファイル **{uploaded_file.name}** を送信しました。"
        st.session_state.messages.append({"role": "user", "content": notice})
        st.sidebar.success("ファイルをチャットへ送信しました")

# -------------------------- 7) チャット表示 --------------------------
st.title("ChatGPT_clone_o3")
st.caption("Streamlit + OpenAI")

for m in st.session_state.messages:
    if m["role"] == "system":      # system は非表示
        continue
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# -------------------------- 8) ユーザー入力 --------------------------
if prompt := st.chat_input("ここにメッセージを入力"):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # OpenAI ストリーミング応答
    stream = client.stream_chat_completion(
        messages=st.session_state.messages,
        model="o3-2025-04-16",
    )

    with st.chat_message("assistant"):
        placeholder, reply = st.empty(), ""
        for chunk in stream:
            delta = (
                chunk.choices[0].delta
                if hasattr(chunk.choices[0], "delta")
                else chunk.choices[0]
            )
            reply += (
                delta.get("content", "")
                if isinstance(delta, dict)
                else (delta.content or "")
            )
            placeholder.markdown(reply + "▌")
        placeholder.markdown(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})



