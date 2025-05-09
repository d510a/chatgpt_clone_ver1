# app3.py

import os
import json
import logging
import sys
import importlib.metadata as imd
from io import BytesIO
from pathlib import Path
from typing import Final

import streamlit as st
from dotenv import load_dotenv
from docx import Document               # Word(.docx)対応
import mammoth                          # .doc 対応
import docx2txt                        # .docx 抽出フォールバック

# -------------------------- 0) 共通設定 --------------------------
load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s ─ %(message)s")

# --- 0-1) 認証情報読み込み ----------------------------------------
username = password = api_key = proxy_host = ""

# ❶ Streamlit Secrets（Cloud 環境）
if "api_key" in st.secrets:
    api_key    = st.secrets["api_key"]
    username   = st.secrets.get("proxy_username", "")
    password   = st.secrets.get("proxy_password", "")
    proxy_host = st.secrets.get("proxy_host", "proxy01.hm.jp.honda.com:8080")
# ❷ ローカル proxy_config.json
else:
    documents_folder: Final[Path] = Path.home() / "Documents"
    config_file:     Final[Path] = documents_folder / "proxy_config.json"
    if config_file.exists():
        try:
            with config_file.open(encoding="utf-8") as f:
                cfg        = json.load(f)
                api_key    = cfg.get("apikey", "")
                username   = cfg.get("username", "")
                password   = cfg.get("password", "")
                proxy_host = cfg.get("proxyhost", "proxy01.hm.jp.honda.com:8080")
        except Exception as e:
            logging.warning("proxy_config.json の読み込みに失敗: %s", e)
    else:
        logging.info("proxy_config.json が見つかりません: %s", config_file)

# --- 0-2) プロキシ URL 組立 ---------------------------------------
proxy_url = (
    f"http://{username}:{password}@{proxy_host}"
    if username and password and proxy_host
    else None
)

# -------------------------- 1) ページ設定 --------------------------
st.set_page_config(page_title="ChatGPT_clone")
if not api_key:
    st.error("API キーが設定されていません。（Secrets または proxy_config.json）")
    st.stop()

# -------------------------- 2) OpenAI v0/v1 互換ラッパー --------------
def detect_openai_v1() -> bool:
    try:
        return int(imd.version("openai").split(".")[0]) >= 1
    except Exception:
        return False

_IS_V1 = detect_openai_v1()
if _IS_V1:
    from openai import OpenAI
else:
    import openai as _openai_legacy  # noqa: F401

class OpenAIWrapper:
    def __init__(self, api_key: str, proxy_url: str | None):
        self.v1 = _IS_V1
        if proxy_url:
            os.environ["HTTP_PROXY"] = os.environ["HTTPS_PROXY"] = proxy_url
        else:
            os.environ.pop("HTTP_PROXY", None)
            os.environ.pop("HTTPS_PROXY", None)
        if self.v1:
            self.client = OpenAI(api_key=api_key)
        else:
            import openai
            openai.api_key = api_key
            self.client = openai

    def list_models(self):
        return self.client.models.list() if self.v1 else self.client.Model.list()

    def stream_chat_completion(self, messages, model="o3-2025-04-16"):
        if self.v1:
            return self.client.chat.completions.create(model=model, messages=messages, stream=True)
        return self.client.ChatCompletion.create(model=model, messages=messages, stream=True)

def create_openai_wrapper(api_key: str, proxy_url: str | None) -> "OpenAIWrapper":
    wrapper = OpenAIWrapper(api_key, proxy_url)
    try:
        wrapper.list_models()
        logging.info("Connectivity OK via %s", "proxy" if proxy_url else "direct")
        return wrapper
    except Exception:
        if proxy_url is None:
            raise
        return OpenAIWrapper(api_key, None)

client = create_openai_wrapper(api_key, proxy_url)

# -------------------------- 3) ポータブル資産パス ---------------------
BASE_DIR = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
POPPLER_DIR   = BASE_DIR / "poppler" / "bin"
TESSERACT_EXE = BASE_DIR / "tesseract" / "tesseract.exe"

# -------------------------- 4) セッション初期化 -----------------------
GREETING = "質問してみましょう"
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": "You are a helpful assistant."}]
if not any(m["role"] == "assistant" and m["content"] == GREETING for m in st.session_state.messages):
    st.session_state.messages.insert(1, {"role": "assistant", "content": GREETING})
st.session_state.setdefault("uploaded_files", {})

# -------------------------- 5) ファイル添付 & テキスト抽出 -------------
st.sidebar.header("ファイルを添付")
uploaded_file = st.sidebar.file_uploader(
    "テキスト / Markdown / PDF / Word",
    type=["txt", "md", "pdf", "docx", "doc"],
    accept_multiple_files=False
)

def read_text_file(file) -> str:
    raw = file.read()
    for enc in ("utf-8", "cp932"):
        try:
            return raw.decode(enc, errors="ignore")[:180_000]
        except UnicodeDecodeError:
            continue
    return raw.decode(errors="ignore")[:180_000]

def extract_text_from_pdf(file_obj) -> str:
    data = file_obj.read(); bio = BytesIO(data)
    try:
        from pdfminer.high_level import extract_text
        text = extract_text(bio)
        if text.strip(): return text[:180_000]
    except: pass
    try:
        import PyPDF2
        reader = PyPDF2.PdfReader(BytesIO(data))
        text = "\n".join(p.extract_text() or "" for p in reader.pages)
        if text.strip(): return text[:180_000]
    except: pass
    try:
        import fitz
        doc = fitz.open(stream=data, filetype="pdf")
        text = "\n".join(p.get_text() for p in doc)
        if text.strip(): return text[:180_000]
    except: pass
    try:
        from pdf2image import convert_from_bytes
        import pytesseract
        pages = convert_from_bytes(data, dpi=300, fmt="png", poppler_path=str(POPPLER_DIR))
        pytesseract.pytesseract.tesseract_cmd = str(TESSERACT_EXE)
        ocr = "\n".join(pytesseract.image_to_string(p, lang="jpn") for p in pages)
        if ocr.strip(): return ocr[:180_000]
    except: pass
    return "(PDF からテキストを抽出できませんでした)"

def extract_text_from_docx(file_obj) -> str:
    data = file_obj.read(); bio = BytesIO(data)
    try:
        doc = Document(bio)
        return "\n".join(p.text for p in doc.paragraphs)[:180_000]
    except: pass
    try:
        return docx2txt.process(bio)[:180_000]
    except: pass
    return "(Word(.docx) から抽出できませんでした)"

def extract_text_from_doc(file_obj) -> str:
    data = file_obj.read(); bio = BytesIO(data)
    try:
        result = mammoth.extract_raw_text(bio)
        return result.value[:180_000]
    except: pass
    return "(.doc から抽出できませんでした)"

if uploaded_file:
    st.sidebar.write(f" **{uploaded_file.name}** を読み込みました")
    if uploaded_file.name not in st.session_state.uploaded_files:
        ext = Path(uploaded_file.name).suffix.lower()[1:]
        if ext == "pdf":
            content = extract_text_from_pdf(uploaded_file)
        elif ext == "docx":
            content = extract_text_from_docx(uploaded_file)
        elif ext == "doc":
            content = extract_text_from_doc(uploaded_file)
        else:
            content = read_text_file(uploaded_file)
        st.session_state.uploaded_files[uploaded_file.name] = content

    if st.sidebar.button("ファイル内容を送信"):
        txt = st.session_state.uploaded_files[uploaded_file.name]
        st.session_state.messages.append({"role": "system", "content": txt})
        notice = f"ファイル **{uploaded_file.name}** を送信しました。"
        st.session_state.messages.append({"role": "user", "content": notice})
        st.sidebar.success("ファイルをチャットへ送信しました")

# -------------------------- 6) メッセージ描画 --------------------------
st.title("ChatGPT_clone_o3")
for m in st.session_state.messages:
    if m["role"] == "system":
        continue
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# -------------------------- 7) チャット入力 --------------------------
if prompt := st.chat_input("ここにメッセージを入力"):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    stream = client.stream_chat_completion(
        messages=st.session_state.messages, model="o3-2025-04-16"
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
                else delta.content or ""
            )
            placeholder.markdown(reply + "▌")
        placeholder.markdown(reply)
    st.session_state.messages.append({"role": "assistant", "content": reply})
