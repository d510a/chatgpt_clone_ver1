#!/usr/bin/env python
# coding: utf-8
# ================================================================
#  Chat + File-upload ユーティリティ
# ================================================================
import os
import sys
import logging
import uuid
import re
from io import BytesIO, StringIO
from pathlib import Path
from typing import Optional, List

import importlib.metadata as imd
import streamlit as st
from dotenv import load_dotenv

# ----- 追加依存 -----
import pdfplumber                 # pip install pdfplumber
import docx2txt                   # pip install docx2txt
from docx import Document         # pip install python-docx
import chardet                    # (任意) 文字コード自動判定
# --------------------

# --------------------------------------------------
# 環境変数 & OpenAI ラッパー（元コードそのまま）
# --------------------------------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# （以降、OpenAI 呼び出し用のヘルパー関数は既存のまま）

# --------------------------------------------------
# 定数
# --------------------------------------------------
MAX_CHARS = 180_000
CID_RE = re.compile(r"\(cid:\d+\)")
REPLACEMENT_CHAR = "\uFFFD"  # '�'

# poppler & tesseract: 既存の環境変数や設定を流用
POPPLER_DIR = Path(os.getenv("POPPLER_DIR", r"C:\poppler"))
TESSERACT_EXE = Path(os.getenv("TESSERACT_EXE", r"C:\Program Files\Tesseract-OCR\tesseract.exe"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ================================================================
# 1) 汎用ユーティリティ
# ================================================================
def looks_garbled(text: str, threshold: float = 0.10) -> bool:
    """� や (cid:n) の割合が一定以上なら読めないと判断"""
    if not text:
        return True
    garbled_tokens = (
        text.count(REPLACEMENT_CHAR) +
        len(CID_RE.findall(text)) +
        text.count("�")
    )
    return garbled_tokens / max(len(text), 1) > threshold

def clean_text(text: str) -> str:
    """(cid:n) 除去＋先頭末尾空白整形"""
    return CID_RE.sub("", text).strip()

# ================================================================
# 2) TXT
# ================================================================
def extract_text_from_txt(file_obj) -> str:
    raw = file_obj.read()
    if isinstance(raw, str):          # StringIO が来るパス
        return raw[:MAX_CHARS]

    # bytes → まず UTF-8
    try:
        return raw.decode("utf-8")[:MAX_CHARS]
    except UnicodeDecodeError:
        enc_guess = chardet.detect(raw)["encoding"] or "cp932"
        return raw.decode(enc_guess, errors="ignore")[:MAX_CHARS]

# ================================================================
# 3) DOCX
# ================================================================
def extract_text_from_docx(file_obj) -> str:
    """python-docx → 空文字なら docx2txt にフォールバック"""
    try:
        doc = Document(file_obj)
        txt = "\n".join(p.text for p in doc.paragraphs)
        if txt.strip():
            return txt[:MAX_CHARS]
    except Exception as e:
        logging.warning("python-docx failed: %s", e)

    # フォールバック：一旦 temp に保存して docx2txt
    try:
        import tempfile, shutil
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir) / "tmp.docx"
            with open(tmp_path, "wb") as f:
                file_obj.seek(0)
                f.write(file_obj.read())
            txt = docx2txt.process(str(tmp_path)) or ""
            return txt[:MAX_CHARS] if txt else ""
    except Exception as e:
        logging.warning("docx2txt failed: %s", e)
        return "(DOCX からテキストを抽出できませんでした)"

# ================================================================
# 4) PDF
# ================================================================
def extract_with_pdfminer(bio: BytesIO) -> Optional[str]:
    try:
        from pdfminer.high_level import extract_text
        bio.seek(0)
        return extract_text(bio)
    except Exception as e:
        logging.debug("pdfminer 失敗: %s", e)
        return None

def extract_with_pypdf2(bio: BytesIO) -> Optional[str]:
    try:
        import PyPDF2
        reader = PyPDF2.PdfReader(bio)
        return "\n".join((p.extract_text() or "") for p in reader.pages)
    except Exception as e:
        logging.debug("PyPDF2 失敗: %s", e)
        return None

def extract_with_pymupdf(data: bytes) -> Optional[str]:
    try:
        import fitz
        doc = fitz.open(stream=data, filetype="pdf")
        return "\n".join(p.get_text() for p in doc)
    except Exception as e:
        logging.debug("PyMuPDF 失敗: %s", e)
        return None

def extract_with_pdfplumber(bio: BytesIO) -> Optional[str]:
    try:
        with pdfplumber.open(bio) as pdf:
            return "\n".join((p.extract_text() or "") for p in pdf.pages)
    except Exception as e:
        logging.debug("pdfplumber 失敗: %s", e)
        return None

def ocr_with_tesseract(data: bytes) -> str:
    try:
        from pdf2image import convert_from_bytes
        import pytesseract
        pytesseract.pytesseract.tesseract_cmd = str(TESSERACT_EXE)
        images = convert_from_bytes(data, dpi=300, fmt="png", poppler_path=str(POPPLER_DIR))
        return "\n".join(pytesseract.image_to_string(img, lang="jpn") for img in images)
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
        candidate = extractor(bio if extractor != extract_with_pymupdf else data)
        if candidate and not looks_garbled(candidate):
            logging.info("%s で成功", extractor.__name__)
            return clean_text(candidate)[:MAX_CHARS]

    # 全滅なら OCR（ページ数が少ない場合のみ）
    try:
        import fitz
        if fitz.open(stream=data, filetype="pdf").page_count > 2:
            return "(画像 PDF と判断・OCR はスキップしました)"
    except Exception:
        pass

    logging.info("全テキスト抽出手法が失敗 → OCR")
    ocr_txt = ocr_with_tesseract(data)
    return ocr_txt[:MAX_CHARS] if ocr_txt.strip() else "(PDF からテキストを抽出できませんでした)"

# ================================================================
# 5) Dispatcher
# ================================================================
def extract_text_any(uploaded_file) -> str:
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix == ".txt":
        return extract_text_from_txt(uploaded_file)
    if suffix in (".docx", ".doc"):
        return extract_text_from_docx(uploaded_file)
    if suffix == ".pdf":
        return extract_text_from_pdf(uploaded_file)
    return f"(未対応の拡張子: {suffix})"

# ================================================================
# 6) Streamlit UI  —  見た目は一切変更なし
# ================================================================
st.set_page_config(page_title="Chat & File Uploader", layout="wide")

st.sidebar.header("ファイルをアップロード")
uploaded_files = st.sidebar.file_uploader(
    "PDF / Word / Text を選択",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True,
    key="file_uploader",
)
file_texts: List[str] = []

if uploaded_files:
    for uf in uploaded_files:
        # 400 Bad Request 回避: ファイル内容を即時読み込み
        unique_name = f"{uuid.uuid4().hex}_{uf.name}"
        with st.expander(f"⬇ {uf.name}"):
            text = extract_text_any(uf)
            st.text_area("抽出結果", text, height=300)
            file_texts.append(text)

# === 以下、元コードのチャット処理（OpenAI 呼び出しなど）はそのまま ===
# （プロンプト生成に file_texts を連結して渡すなど、元ロジックを活かして下さい）
