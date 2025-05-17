#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streamlit file-uploader → text extractor → API sender
"""

# == 基本ライブラリ ==========================================================
import os, sys, logging, re
from io import BytesIO
from pathlib import Path
from typing import Optional

import streamlit as st
from dotenv import load_dotenv

# 依存ライブラリ -------------------------------------------------------------
import importlib.metadata as imd  # ← 元コードのまま
import pdfplumber                 # pdf テキスト抽出
import requests                   # API 送信
from docx import Document         # Word 読み込み

# OCR 系ライブラリ（requirements に追記済み）
from pdf2image import convert_from_bytes
import pytesseract
import fitz                       # PyMuPDF
import PyPDF2
from pdfminer.high_level import extract_text as pdfminer_extract

# == 設定 ===================================================================
load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# ★ 追加: アップロード最大サイズを 1 GB に拡大（MB 単位）
st.set_option("server.maxUploadSize", 1024)

# 環境変数（バックエンド URL, OCR コマンド等）
API_URL      = os.getenv("UPLOAD_API_URL", "https://example.com/upload")
POPPLER_DIR  = Path(os.getenv("POPPLER_DIR", "/usr/bin"))
TESSERACT_EXE = Path(os.getenv("TESSERACT_EXE", "/usr/bin/tesseract"))

# == ユーティリティ =========================================================
CID_RE = re.compile(r"\(cid:\d+\)")
REPLACEMENT_CHAR = "\uFFFD"   # �

def looks_garbled(text: str, threshold: float = 0.10) -> bool:
    if not text:
        return True
    garbled = text.count(REPLACEMENT_CHAR) + len(CID_RE.findall(text)) + text.count("�")
    return garbled / max(len(text), 1) > threshold

def clean_cid(text: str) -> str:
    return CID_RE.sub("", text)

# == PDF 抽出 ===============================================================
def extract_with_pdfminer(bio: BytesIO) -> Optional[str]:
    try:
        return pdfminer_extract(bio)
    except Exception as e:
        logging.warning("pdfminer 失敗: %s", e)
        return None

def extract_with_pypdf2(bio: BytesIO) -> Optional[str]:
    try:
        reader = PyPDF2.PdfReader(bio)
        return "\n".join(p.extract_text() or "" for p in reader.pages)
    except Exception as e:
        logging.warning("PyPDF2 失敗: %s", e)
        return None

def extract_with_pymupdf(data: bytes) -> Optional[str]:
    try:
        doc = fitz.open(stream=data, filetype="pdf")
        return "\n".join(p.get_text() for p in doc)
    except Exception as e:
        logging.warning("PyMuPDF 失敗: %s", e)
        return None

def extract_with_pdfplumber(bio: BytesIO) -> Optional[str]:
    try:
        with pdfplumber.open(bio) as pdf:
            return "\n".join(p.extract_text() or "" for p in pdf.pages)
    except Exception as e:
        logging.warning("pdfplumber 失敗: %s", e)
        return None

def ocr_with_tesseract(data: bytes) -> str:
    try:
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

    logging.info("全テキスト抽出手法が失敗 → OCR にフォールバック")
    ocr_text = ocr_with_tesseract(data)
    return ocr_text[:180_000] if ocr_text.strip() else "(PDF から抽出できませんでした)"

# == Word 抽出（★追加） ======================================================
def extract_text_from_docx(file_obj) -> str:
    file_obj.seek(0)
    doc = Document(file_obj)
    return "\n".join(p.text for p in doc.paragraphs)[:180_000]

# == TXT 読み込み ===========================================================
def extract_text_from_txt(file_obj) -> str:
    file_obj.seek(0)
    return file_obj.read().decode("utf-8", errors="ignore")[:180_000]

# == 汎用ディスパッチ（★追加） =============================================
def extract_text_generic(uploaded_file) -> str:
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix == ".pdf":
        return extract_text_from_pdf(uploaded_file)
    elif suffix == ".docx":
        return extract_text_from_docx(uploaded_file)
    elif suffix in (".txt", ".text"):
        return extract_text_from_txt(uploaded_file)
    else:
        return "(未対応のファイル形式です)"

# == API 送信（★追加） ======================================================
def send_file_to_api(uploaded_file) -> dict:
    """
    Multipart/form-data でファイルを送信し JSON レスポンスを返す
    """
    files = {
        "file": (uploaded_file.name, uploaded_file.getvalue(), 
                 "application/octet-stream"),
    }
    resp = requests.post(API_URL, files=files, timeout=60)
    resp.raise_for_status()
    return resp.json()

# == Streamlit UI ===========================================================
st.title("ファイル添付＆送信デモ")

uploaded_files = st.file_uploader(
    "PDF / Word / TXT を選択（複数可）",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True,
)

if uploaded_files:
    for uf in uploaded_files:
        st.subheader(f"プレビュー: {uf.name}")
        text_preview = extract_text_generic(uf)
        st.text_area(
            label="抽出テキスト（先頭 4 000 字）",
            value=text_preview[:4000],
            height=200,
        )

    if st.button("📤 送信 / Upload"):
        for uf in uploaded_files:
            try:
                result = send_file_to_api(uf)
                st.success(f"{uf.name}: 送信成功 → {result}")
            except Exception as e:
                st.error(f"{uf.name}: 送信失敗 ({e})")

st.sidebar.markdown("### 設定")
st.sidebar.write(f"API_URL = `{API_URL}`")
st.sidebar.write("`.streamlit/config.toml` で `server.maxUploadSize` を恒久設定できます。")
