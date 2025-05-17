import os
import sys
import logging
import importlib.metadata as imd
from io import BytesIO
from pathlib import Path
from typing import Optional
import re

import streamlit as st
from dotenv import load_dotenv
from docx import Document

# 追加ライブラリ
import pdfplumber            # pip install pdfplumber
# OCR 用ライブラリ (任意)
# from pdf2image import convert_from_bytes
# import pytesseract

# --------------------------------------------------
# ① 環境変数・OpenAI ラッパー（元コードそのまま）
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
# ここではラッパー詳細は省略 (元コードを保持)
# --------------------------------------------------

# ② 共通定数・正規表現
CID_RE            = re.compile(r"\(cid:\d+\)")
REPLACEMENT_CHAR  = "\uFFFD"  # '�'
MAX_CHARS         = 180_000   # OpenAI 送信用テキスト上限

POPPLER_DIR   = Path(os.getenv("POPPLER_DIR", "/usr/bin"))
TESSERACT_EXE = Path(os.getenv("TESSERACT_EXE", "/usr/bin/tesseract"))

# --------------------------------------------------
# ③ ユーティリティ
def looks_garbled(text: str, threshold: float = 0.10) -> bool:
    """文字化けと思われるかを簡易判定"""
    if not text:
        return True
    garbled_tokens = (
        text.count(REPLACEMENT_CHAR)
        + len(CID_RE.findall(text))
        + text.count("�")
    )
    return garbled_tokens / max(len(text), 1) > threshold


def clean_cid(text: str) -> str:
    """(cid:n) を除去"""
    return CID_RE.sub("", text)


# --------------------------------------------------
# ④ TXT 抽出
def extract_text_from_txt(file_obj) -> str:
    file_obj.seek(0)
    data = file_obj.read()
    try:
        return data.decode("utf-8", errors="replace")
    except Exception:
        return data.decode("shift_jis", errors="replace")


# --------------------------------------------------
# ⑤ DOCX 抽出
def extract_text_from_docx(file_obj) -> str:
    file_obj.seek(0)
    doc = Document(file_obj)
    return "\n".join(para.text for para in doc.paragraphs)


# --------------------------------------------------
# ⑥ PDF 用バックエンド
def _extract_with_pdfminer(bio: BytesIO) -> Optional[str]:
    try:
        from pdfminer.high_level import extract_text
        return extract_text(bio)
    except Exception as e:
        logging.warning("pdfminer 失敗: %s", e)
        return None


def _extract_with_pypdf2(bio: BytesIO) -> Optional[str]:
    try:
        import PyPDF2
        reader = PyPDF2.PdfReader(bio)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception as e:
        logging.warning("PyPDF2 失敗: %s", e)
        return None


def _extract_with_pymupdf(data: bytes) -> Optional[str]:
    try:
        import fitz
        doc = fitz.open(stream=data, filetype="pdf")
        return "\n".join(p.get_text() for p in doc)
    except Exception as e:
        logging.warning("PyMuPDF 失敗: %s", e)
        return None


def _extract_with_pdfplumber(bio: BytesIO) -> Optional[str]:
    try:
        with pdfplumber.open(bio) as pdf:
            return "\n".join(p.extract_text() or "" for p in pdf.pages)
    except Exception as e:
        logging.warning("pdfplumber 失敗: %s", e)
        return None


# --------------------------------------------------
# ⑦ OCR (任意)
def _ocr_with_tesseract(data: bytes) -> str:
    try:
        from pdf2image import convert_from_bytes
        import pytesseract
        pages = convert_from_bytes(data, dpi=300, fmt="png",
                                   poppler_path=str(POPPLER_DIR))
        pytesseract.pytesseract.tesseract_cmd = str(TESSERACT_EXE)
        return "\n".join(pytesseract.image_to_string(p, lang="jpn")
                         for p in pages)
    except Exception as e:
        logging.warning("OCR 失敗: %s", e)
        return ""


# --------------------------------------------------
# ⑧ PDF 抽出エントリーポイント
def extract_text_from_pdf(file_obj) -> str:
    data = file_obj.read()
    bio  = BytesIO(data)

    for extractor in (_extract_with_pdfminer,
                      _extract_with_pypdf2,
                      _extract_with_pymupdf,
                      _extract_with_pdfplumber):
        bio.seek(0)
        arg = bio if extractor is not _extract_with_pymupdf else data
        text = extractor(arg)
        if text and not looks_garbled(text):
            logging.info("%s 成功", extractor.__name__)
            return clean_cid(text)[:MAX_CHARS]

    # フォールバック OCR（画像 PDF のみの場合有効にする）
    # ocr_text = _ocr_with_tesseract(data)
    # if ocr_text.strip():
    #     return ocr_text[:MAX_CHARS]

    return "(PDF からテキストを抽出できませんでした)"


# --------------------------------------------------
# ⑨ ファイル種別ディスパッチ
EXTRACTORS = {
    ".txt": extract_text_from_txt,
    ".docx": extract_text_from_docx,
    ".pdf": extract_text_from_pdf,
}

def get_text(file) -> str:
    suffix = Path(file.name).suffix.lower()
    extractor = EXTRACTORS.get(suffix)
    if not extractor:
        return f"未対応の拡張子です: {suffix}"
    return extractor(file)


# --------------------------------------------------
# ⑩ Streamlit UI
st.set_page_config(page_title="ファイル＋Chat", layout="wide")
st.title("🗂️ ファイルアップロード & Chat")

uploaded = st.file_uploader(
    "TXT / Word (.docx) / PDF を選択してください",
    type=["txt", "pdf", "docx"],
    accept_multiple_files=False,
)

if uploaded:
    with st.spinner("テキスト抽出中..."):
        text = get_text(uploaded)
    st.success("抽出完了")
    st.text_area("抽出結果 (編集可)", text, height=300)

    # --- Chat 送信（OpenAI 呼び出し例）
    if st.button("OpenAI に送信"):
        with st.spinner("OpenAI 応答待ち..."):
            # openai_response = openai_chat(text)  # 元コードのラッパーを利用
            openai_response = "(ここに OpenAI 応答が入ります)"
        st.code(openai_response, language="markdown")
