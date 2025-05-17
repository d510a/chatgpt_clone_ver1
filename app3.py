import os
import sys
import logging
import importlib.metadata as imd
from io import BytesIO
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# -------- 設定 ----------------------------------------------------
load_dotenv()
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s ─ %(message)s")

# --- OpenAI -------------------------------------------------------
api_key = st.secrets.get("api_key", os.getenv("API_KEY", ""))
if not api_key:
    st.error("APIキーが設定されていません (.env または Secrets)")
    st.stop()

def _is_v1() -> bool:
    try:
        return int(imd.version("openai").split(".")[0]) >= 1
    except Exception:
        return False

_IS_V1 = _is_v1()
if _IS_V1:
    from openai import OpenAI
else:
    import openai as _openai_legacy

class OpenAIWrapper:
    def __init__(self, key: str):
        if _IS_V1:
            self.client = OpenAI(api_key=key)
        else:
            _openai_legacy.api_key = key
            self.client = _openai_legacy

    def list_models(self):
        return (self.client.models.list() if _IS_V1
                else self.client.Model.list())

    def stream_chat(self, messages, model="o3-2025-04-16"):
        if _IS_V1:
            return self.client.chat.completions.create(
                model=model, messages=messages, stream=True)
        return self.client.ChatCompletion.create(
            model=model, messages=messages, stream=True)

try:
    client = OpenAIWrapper(api_key)
    client.list_models()
except Exception as e:
    st.error(f"OpenAI 接続に失敗しました: {e}")
    st.stop()

# -------- Streamlit ページ ---------------------------------------
st.set_page_config(page_title="ChatGPT_clone")
st.title("ChatGPT_clone_o3")
st.caption("Streamlit + OpenAI")

# -------- セッション初期化 ---------------------------------------
GREETING = "質問してみましょう"
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "assistant", "content": GREETING},
    ]
st.session_state.setdefault("uploaded", {})

# -------- ユーティリティ -----------------------------------------
import tiktoken
ENC = tiktoken.encoding_for_model("gpt-3.5-turbo")

def clip_text(text: str, char_limit=16_000, tok_limit=3_500) -> str:
    if len(text) > char_limit:
        text = text[:char_limit] + "\n...(truncated)"
    tokens = ENC.encode(text)
    if len(tokens) > tok_limit:
        text = ENC.decode(tokens[:tok_limit]) + "\n...(truncated)"
    return text

def read_text_file(file) -> str:
    file.seek(0)
    raw = file.read()
    enc_guess = None
    try:
        import chardet
        enc_guess = chardet.detect(raw)["encoding"]  # type: ignore
    except Exception:
        pass
    for enc in filter(None, (enc_guess, "utf-8", "cp932", "shift_jis", "euc-jp")):
        try:
            return clip_text(raw.decode(enc, errors="ignore"))
        except Exception:
            continue
    return "(テキストの文字コード判定に失敗しました)"

def extract_pdf(file) -> str:
    file.seek(0)
    data = file.read()
    # 1) pdfminer.six
    try:
        from pdfminer.high_level import extract_text
        text = extract_text(BytesIO(data))
        if text.strip():
            return clip_text(text)
    except Exception as e:
        logging.warning("pdfminer 失敗: %s", e)
    # 2) pypdf / PyPDF2
    try:
        try:
            from pypdf import PdfReader          # pypdf >=3
        except ImportError:
            from PyPDF2 import PdfReader          # fallback
        reader = PdfReader(BytesIO(data))
        text = "\n".join(p.extract_text() or "" for p in reader.pages)
        if text.strip():
            return clip_text(text)
    except Exception as e:
        logging.warning("pypdf 失敗: %s", e)
    # 3) PyMuPDF
    try:
        import fitz
        doc = fitz.open(stream=data, filetype="pdf")
        text = "\n".join(p.get_text() for p in doc)
        if text.strip():
            return clip_text(text)
    except Exception as e:
        logging.warning("PyMuPDF 失敗: %s", e)
    return "(PDF からテキストを抽出できませんでした)"

def extract_word(file) -> str:
    file.seek(0)
    data = file.read()
    try:
        from docx import Document
        doc = Document(BytesIO(data))
        text = "\n".join(p.text for p in doc.paragraphs)
        if text.strip():
            return clip_text(text)
    except Exception as e:
        logging.warning("python-docx 失敗: %s", e)
    try:
        import docx2txt
        text = docx2txt.process(BytesIO(data))
        if text.strip():
            return clip_text(text)
    except Exception as e:
        logging.warning("docx2txt 失敗: %s", e)
    try:
        import zipfile
        import xml.etree.ElementTree as ET
        with zipfile.ZipFile(BytesIO(data)) as z:
            with z.open("word/document.xml") as fxml:
                xml = fxml.read()
        root = ET.fromstring(xml)
        texts = []
        for node in root.iter("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t"):
            if node.text:
                texts.append(node.text)
        text = "\n".join(texts)
        if text.strip():
            return clip_text(text)
    except Exception as e:
        logging.warning("zipxml 失敗: %s", e)
    return "(Word ファイルからテキストを抽出できませんでした)"

# -------- ファイルアップローダー ----------------------------------
st.sidebar.header("ファイルを添付")
uploaded = st.sidebar.file_uploader(
    "TXT / PDF / DOCX", type=["txt", "md", "pdf", "docx", "doc"])

if uploaded:
    st.sidebar.success(f"**{uploaded.name}** を読み込みました "
                       f"({uploaded.size//1024:,} KB)")
    if uploaded.name not in st.session_state.uploaded:
        mime = uploaded.type or ""
        name_l = uploaded.name.lower()
        if mime == "application/pdf" or name_l.endswith(".pdf"):
            content = extract_pdf(uploaded)
        elif (mime in ("application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                       "application/msword") or
              name_l.endswith((".docx", ".doc"))):
            content = extract_word(uploaded)
        else:
            content = read_text_file(uploaded)
        st.session_state.uploaded[uploaded.name] = content

    if st.sidebar.button("ファイル内容を送信"):
        txt = st.session_state.uploaded[uploaded.name]
        st.session_state.messages.append({"role": "system", "content": txt})
        st.session_state.messages.append(
            {"role": "user", "content": f"{uploaded.name} を送信しました"})
        st.sidebar.info("チャットへ送信しました")

# -------- 既存メッセージ描画 --------------------------------------
for m in st.session_state.messages:
    if m["role"] == "system":
        continue
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# -------- 入力ボックス & 応答 ------------------------------------
if prompt := st.chat_input("ここに入力"):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    stream = client.stream_chat(st.session_state.messages)
    with st.chat_message("assistant"):
        placeholder, reply = st.empty(), ""
        for chunk in stream:
            delta = (chunk.choices[0].delta
                     if hasattr(chunk.choices[0], "delta")
                     else chunk.choices[0])
            reply += (delta.get("content", "") if isinstance(delta, dict)
                      else delta.content or "")
            placeholder.markdown(reply + "▌")
        placeholder.markdown(reply)
    st.session_state.messages.append({"role": "assistant", "content": reply})
