# -*- coding: utf-8 -*-
"""
Streamlit アプリ: PDF / Word / テキストをアップロードし、内容を抽出して OpenAI ChatCompletion
に送信します。
- 400 Bad Request を防ぐためにトークン概算でチャンク分割
- PDF は pdfminer → PyPDF2 → PyMuPDF → pdfplumber → OCR の順でテキスト抽出
- Word (.docx) は python-docx で抽出
- TXT はそのまま読み込み
必要環境:
  python -m pip install streamlit openai python-docx pdfminer.six PyPDF2 PyMuPDF pdfplumber pdf2image pytesseract python-dotenv
"""

from __future__ import annotations

import os
import sys
import logging
from io import BytesIO
from pathlib import Path
import re
from typing import Optional, List

import streamlit as st
from dotenv import load_dotenv
import openai
from docx import Document
import pdfplumber  # noqa: pip install pdfplumber

# ==================================================
# 環境変数 & OpenAI 初期化
# --------------------------------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY が設定されていません。 .env もしくは環境変数で設定してください。")
    st.stop()
openai.api_key = OPENAI_API_KEY

# ===== ログ設定 ====================================
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
    stream=sys.stderr,
)

# ===== 定数 ========================================
TOKEN_LIMIT = 7_000           # 1 回の ChatCompletion への最大トークン（概算）
CHUNK_MARGIN = 500            # system/assistant プロンプトに回す余裕トークン
POPPLER_DIR = Path(os.getenv("POPPLER_DIR", "/usr/bin"))
TESSERACT_EXE = Path(os.getenv("TESSERACT_EXE", "/usr/bin/tesseract"))

# ===== ガーベージ判定ユーティリティ =================
CID_RE = re.compile(r"\(cid:\d+\)")
REPLACEMENT_CHAR = "\uFFFD"  # '�'


def looks_garbled(text: str, threshold: float = 0.20) -> bool:
    """PDF 抽出結果が文字化けしているか簡易判定"""
    if not text or not text.strip():
        return True
    garbled_tokens = (
        text.count(REPLACEMENT_CHAR)
        + len(CID_RE.findall(text))
    )
    ratio = garbled_tokens / max(len(text), 1)
    return ratio > threshold


def clean_cid(text: str) -> str:
    """(cid:123) を除去"""
    return CID_RE.sub("", text)

# ===== PDF 抽出ロジック =============================

def extract_with_pdfminer(bio: BytesIO) -> Optional[str]:
    try:
        from pdfminer.high_level import extract_text
        text = extract_text(bio)
        return text
    except Exception as e:
        logging.warning("pdfminer 失敗: %s", e)
        return None


def extract_with_pypdf2(bio: BytesIO) -> Optional[str]:
    try:
        import PyPDF2
        reader = PyPDF2.PdfReader(bio)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception as e:
        logging.warning("PyPDF2 失敗: %s", e)
        return None


def extract_with_pymupdf(data: bytes) -> Optional[str]:
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(stream=data, filetype="pdf")
        return "\n".join(p.get_text() for p in doc)
    except Exception as e:
        logging.warning("PyMuPDF 失敗: %s", e)
        return None


def extract_with_pdfplumber(bio: BytesIO) -> Optional[str]:
    try:
        with pdfplumber.open(bio) as pdf:
            text = "\n".join(p.extract_text() or "" for p in pdf.pages)
        return text
    except Exception as e:
        logging.warning("pdfplumber 失敗: %s", e)
        return None


def ocr_with_tesseract(data: bytes) -> str:
    try:
        from pdf2image import convert_from_bytes
        import pytesseract
        pages = convert_from_bytes(data, dpi=300, fmt="png", poppler_path=str(POPPLER_DIR))
        pytesseract.pytesseract.tesseract_cmd = str(TESSERACT_EXE)
        return "\n".join(pytesseract.image_to_string(p, lang="jpn") for p in pages)
    except Exception as e:
        logging.warning("OCR 失敗: %s", e)
        return ""


def extract_text_from_pdf(file_obj) -> str:
    data = file_obj.read()
    bio = BytesIO(data)

    for extractor in (
        extract_with_pdfminer,
        extract_with_pypdf2,
        extract_with_pymupdf,
        extract_with_pdfplumber,
    ):
        bio.seek(0)
        text = extractor(bio if extractor != extract_with_pymupdf else data)
        if text and not looks_garbled(text):
            logging.info("%s 成功", extractor.__name__)
            return clean_cid(text)[:180_000]

    logging.info("全テキスト抽出手法が失敗 → OCR フォールバック")
    ocr_text = ocr_with_tesseract(data)
    if ocr_text.strip():
        return ocr_text[:180_000]
    return "(PDF からテキストを抽出できませんでした)"

# ===== Word 抽出 ====================================

def extract_text_from_docx(file_obj) -> str:
    try:
        file_obj.seek(0)
        doc = Document(BytesIO(file_obj.read()))
        text = "\n".join(p.text for p in doc.paragraphs)
        return text[:180_000] if text else ""
    except Exception as e:
        logging.warning("DOCX 失敗: %s", e)
        return "(Word からテキストを抽出できませんでした)"

# ===== TXT 読み込み =================================

def extract_text_from_txt(file_obj) -> str:
    try:
        return file_obj.read().decode(errors="ignore")[:180_000]
    except Exception:
        file_obj.seek(0)
        return file_obj.read().decode("shift_jis", errors="ignore")[:180_000]

# ===== トークン概算 & チャンク ======================

def rough_token_len(txt: str) -> int:
    """おおよそ 1 token ≒ 4 文字で概算"""
    return max(len(txt) // 4, 1)


def split_into_chunks(txt: str, limit: int = TOKEN_LIMIT) -> List[str]:
    if rough_token_len(txt) <= limit:
        return [txt]
    chunks, buff = [], []
    for line in txt.splitlines():
        buff.append(line)
        if rough_token_len("\n".join(buff)) > (limit - CHUNK_MARGIN):
            chunks.append("\n".join(buff))
            buff = []
    if buff:
        chunks.append("\n".join(buff))
    return chunks

# ===== OpenAI ChatCompletion ラッパ ==================

def chat_completion(user_text: str, model: str = "gpt-4o-mini") -> str:
    resp = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": "あなたは優秀な日本語アシスタントです。"},
            {"role": "user", "content": user_text},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()

# ==================================================
# Streamlit UI
# --------------------------------------------------

st.set_page_config(page_title="ファイル ChatGPT", layout="wide")
st.title("📄 ファイルを ChatGPT で解析")

with st.sidebar:
    st.header("設定")
    model = st.selectbox("モデル", ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo"])  # お好みで追加
    st.markdown("---")
    st.caption("トークン上限を超える場合は自動でチャンクに分割します。")

uploaded = st.file_uploader(
    "PDF / Word / テキストを選択",
    type=["pdf", "docx", "txt"],
)

if uploaded:
    suffix = Path(uploaded.name).suffix.lower()
    with st.spinner("テキスト抽出中 …"):
        if suffix == ".pdf":
            extracted = extract_text_from_pdf(uploaded)
        elif suffix == ".docx":
            extracted = extract_text_from_docx(uploaded)
        else:
            extracted = extract_text_from_txt(uploaded)

    if not extracted or extracted.startswith("("):
        st.error("テキストを抽出できませんでした。")
        st.stop()

    chunks = split_into_chunks(extracted)
    st.success(f"{len(chunks)} チャンクに分割して送信します …")

    for i, ck in enumerate(chunks, 1):
        with st.spinner(f"ChatGPT ({i}/{len(chunks)}) 処理中 …"):
            response = chat_completion(ck, model=model)
        st.subheader(f"✅ チャンク {i}")
        st.write(response)

    st.download_button(
        "抽出テキストをダウンロード",
        data=extracted,
        file_name=f"{Path(uploaded.name).stem}_extracted.txt",
        mime="text/plain",
    )
