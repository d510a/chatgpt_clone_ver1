import os
import logging
import sys
import importlib.metadata as imd
from io import BytesIO
from pathlib import Path
import re
from typing import Tuple, Optional

import streamlit as st
from dotenv import load_dotenv
from docx import Document

# 追加ライブラリ
import pdfplumber   # pip install pdfplumber
# OCR 用ライブラリは requirements に入っている想定
# pdf2image, pytesseract は既存コードと同じ

# --------------------------------------------------
# 環境変数・OpenAI ラッパーは元コードと同じ
# （中略：ここは変更なし。行数維持のため省略）
# --------------------------------------------------

# --- ユーティリティ -------------------------------------------------
CID_RE = re.compile(r"\(cid:\d+\)")
REPLACEMENT_CHAR = "\uFFFD"  # '�'

def looks_garbled(text: str, threshold: float = 0.10) -> bool:
    """
    判定ロジックを強化:
      * � の比率
      * (cid:123) の比率
      * 可読文字が極端に少ない
    """
    if not text:
        return True
    garbled_tokens = (
        text.count(REPLACEMENT_CHAR)
        + len(CID_RE.findall(text))
        + text.count("�")
    )
    ratio = garbled_tokens / max(len(text), 1)
    return ratio > threshold

def clean_cid(text: str) -> str:
    """(cid:123) を除去"""
    return CID_RE.sub("", text)

# --- PDF 抽出ロジック -----------------------------------------------
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
        import fitz
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

def ocr_with_tesseract(data: bytes, poppler_dir: Path, tesseract_exe: Path) -> str:
    try:
        from pdf2image import convert_from_bytes
        import pytesseract
        pages = convert_from_bytes(data, dpi=300, fmt="png", poppler_path=str(poppler_dir))
        pytesseract.pytesseract.tesseract_cmd = str(tesseract_exe)
        return "\n".join(pytesseract.image_to_string(p, lang="jpn") for p in pages)
    except Exception as e:
        logging.warning("OCR 失敗: %s", e)
        return ""

def extract_text_from_pdf(file_obj) -> str:
    data = file_obj.read()
    bio = BytesIO(data)

    # 1) pdfminer
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

    # 2) すべてダメなら OCR
    logging.info("全テキスト抽出手法が失敗 → OCR にフォールバック")
    ocr_text = ocr_with_tesseract(data, POPPLER_DIR, TESSERACT_EXE)
    if ocr_text.strip():
        return ocr_text[:180_000]
    return "(PDF からテキストを抽出できませんでした)"

# --- Word 抽出・テキストファイル読込は元コードと同じ
# （中略）

# --- Streamlit 側 UI ロジックは元コードと同じ
# （中略：upload, sidebar, chat UI など変更なし）

# --------------------------------------------------
# それ以外の元の関数・OpenAI ラッパー・チャット処理も
# 変更していないので省略
# --------------------------------------------------
