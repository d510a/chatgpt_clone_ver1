import os
import sys
import logging
import importlib.metadata as imd
from io import BytesIO
from pathlib import Path
from typing import Optional
import re
import tempfile

import streamlit as st
from dotenv import load_dotenv
from docx import Document        # pip install python-docx
import chardet                   # ←★ NEW: 文字コード推定
import pdfplumber                # pip install pdfplumber
# 既存 : pdfminer.six, PyPDF2, PyMuPDF (fitz)

# --------------------------------------------------
# 環境変数や OpenAI ラッパーは元コードのまま
#  （中略：OpenAI API キー取得、chat 関数など変更なし）
# --------------------------------------------------

# --- ユーティリティ -------------------------------------------------
CID_RE = re.compile(r"\(cid:\d+\)")
REPLACEMENT_CHAR = "\uFFFD"  # '�'

def looks_garbled(text: str, threshold: float = 0.15) -> bool:  # ←★ しきい値緩和
    """文字化けらしさを判定"""
    if not text:
        return True
    garbled_tokens = (
        text.count(REPLACEMENT_CHAR)
        + len(CID_RE.findall(text))
        + text.count("�")
    )
    return garbled_tokens / max(len(text), 1) > threshold

def clean_cid(text: str) -> str:
    return CID_RE.sub("", text)

# --- PDF 抽出ロジック -----------------------------------------------
def extract_with_pdfminer(bio: BytesIO) -> Optional[str]:
    try:
        from pdfminer.high_level import extract_text
        return extract_text(bio)
    except Exception as e:
        logging.debug("pdfminer NG: %s", e)
        return None

def extract_with_pypdf2(bio: BytesIO) -> Optional[str]:
    try:
        import PyPDF2
        reader = PyPDF2.PdfReader(bio)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception as e:
        logging.debug("PyPDF2 NG: %s", e)
        return None

def extract_with_pymupdf(data: bytes) -> Optional[str]:
    try:
        import fitz
        doc = fitz.open(stream=data, filetype="pdf")
        return "\n".join(p.get_text() for p in doc)
    except Exception as e:
        logging.debug("PyMuPDF NG: %s", e)
        return None

def extract_with_pdfplumber(bio: BytesIO) -> Optional[str]:
    try:
        with pdfplumber.open(bio) as pdf:
            return "\n".join(p.extract_text() or "" for p in pdf.pages)
    except Exception as e:
        logging.debug("pdfplumber NG: %s", e)
        return None

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
            logging.info("PDF extractor %s succeeded", extractor.__name__)
            return clean_cid(text)[:180_000]

    # ここまで全て失敗 → 画像オンリーと判断し OCR せず終了  ←★ 仕様変更
    return "(テキストレイヤのない PDF のため抽出できませんでした)"

# --- DOCX 抽出 -------------------------------------------------------
def extract_text_from_docx(file_obj) -> str:
    """python-docx → docx2txt フォールバック"""
    try:
        doc = Document(file_obj)
        return "\n".join(p.text for p in doc.paragraphs)[:180_000]
    except Exception as e:
        logging.debug("python-docx NG: %s  → docx2txt fallback", e)
        try:
            import docx2txt                   # pip install docx2txt
            with tempfile.TemporaryDirectory() as td:
                tmp = Path(td) / "up.docx"
                tmp.write_bytes(file_obj.read())
                txt = docx2txt.process(str(tmp))
                return txt[:180_000]
        except Exception as e2:
            logging.warning("docx2txt も失敗: %s", e2)
            return "(DOCX からテキストを抽出できませんでした)"

# --- TXT 読込 --------------------------------------------------------
def extract_text_from_txt(file_obj) -> str:
    data = file_obj.read()
    enc = chardet.detect(data[:4096]).get("encoding") or "utf-8"
    try:
        return data.decode(enc, errors="replace")[:180_000]
    except Exception:
        # 万一推定エンコーディングで失敗したら shift_jis
        return data.decode("shift_jis", errors="replace")[:180_000]

# --- ディスパッチ ----------------------------------------------------
def extract_text_by_extension(uploaded_file) -> str:
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix == ".pdf":
        return extract_text_from_pdf(uploaded_file)
    if suffix == ".docx":
        return extract_text_from_docx(uploaded_file)
    if suffix in (".txt", ".text"):
        return extract_text_from_txt(uploaded_file)
    return f"(未対応の拡張子です: {suffix})"

# --- Streamlit UI ----------------------------------------------------
st.set_page_config(page_title="ファイル → テキスト抽出チャット", layout="wide")
st.title("添付ファイルからテキストだけ抽出して送信")

uploaded = st.file_uploader(
    "▼ PDF / Word(DOCX) / TXT を選択してください",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=False,
)

if uploaded:
    with st.spinner("テキスト抽出中..."):
        extracted_text = extract_text_by_extension(uploaded)

    st.subheader("抽出結果プレビュー (先頭 180,000 文字)")
    st.text_area(
        label="内容",
        value=extracted_text,
        height=300,
        help="この内容がそのまま OpenAI API に送られます",
    )

    if st.button("ChatGPT に送信"):
        with st.spinner("ChatGPT 応答待ち..."):
            # ↓ 元コードの chat 処理を呼び出す ----------------------------------
            # response_text = chat(messages=[...], user_input=extracted_text)
            # st.write(response_text)
            pass  # （中略：具体的な OpenAI チャット処理は元コードのまま）
else:
    st.info("ファイルをアップロードしてください。")

# --------------------------------------------------
# 他の OpenAI ラッパー・チャット関数・sidebar などは
# 変更していないので元コード通りです
# --------------------------------------------------
