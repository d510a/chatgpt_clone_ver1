import os
import logging
import sys
import importlib.metadata as imd
from io import BytesIO
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from docx import Document

# ---------------- 共通ユーティリティ -------------------------------------------------
def _escape_js(s: str) -> str:
    """シングルクォート・バックスラッシュ・改行を JS 文字列リテラル用にエスケープ"""
    return (
        s.replace("\\", "\\\\")
        .replace("'", "\\'")
        .replace("\n", "\\n")
        .replace("\r", "")
    )

def render_copy_button(text: str, key: str) -> None:
    """チャット内にワンクリックコピー用 HTML ボタンを描画"""
    escaped = _escape_js(text)
    st.markdown(
        f"""
        <button id="copy_{key}"
                onclick="navigator.clipboard.writeText('{escaped}')"
                style="
                  border:1px solid #ccc;
                  border-radius:4px;
                  padding:4px 8px;
                  margin-top:6px;
                  background:#f7f7f7;
                  cursor:pointer;">
            📋 コピー
        </button>
        """,
        unsafe_allow_html=True,
    )

# ---------------- 環境変数・API 初期化 ------------------------------------------------
load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s ─ %(message)s")

api_key = ""
if "api_key" in st.secrets:
    api_key = st.secrets["api_key"]
else:
    api_key = os.getenv("API_KEY", "")

st.set_page_config(page_title="ChatGPT_clone")

if not api_key:
    st.error("API キーが設定されていません。（Secrets または .env）")
    st.stop()

def detect_openai_v1() -> bool:
    try:
        return int(imd.version("openai").split(".")[0]) >= 1
    except Exception:
        return False

_IS_V1 = detect_openai_v1()
if _IS_V1:
    from openai import OpenAI
else:
    import openai as _openai_legacy

class OpenAIWrapper:
    def __init__(self, api_key: str):
        self.v1 = _IS_V1
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

def create_openai_wrapper(api_key: str) -> OpenAIWrapper:
    wrapper = OpenAIWrapper(api_key)
    try:
        wrapper.list_models()
        logging.info("Connectivity OK")
        return wrapper
    except Exception as e:
        logging.error("OpenAI 接続に失敗: %s", e)
        st.error("OpenAI への接続に失敗しました。")
        st.stop()

client = create_openai_wrapper(api_key)

# ---------------- セッション・定数 -----------------------------------------------------
BASE_DIR   = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
POPPLER_DIR = BASE_DIR / "poppler" / "bin"
TESSERACT_EXE = BASE_DIR / "tesseract" / "tesseract.exe"

GREETING = "質問してみましょう"
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": "You are a helpful assistant."}]
if not any(m["role"] == "assistant" and m["content"] == GREETING for m in st.session_state.messages):
    st.session_state.messages.insert(1, {"role": "assistant", "content": GREETING})
st.session_state.setdefault("uploaded_files", {})

# ---------------- ファイル添付 & 抽出 --------------------------------------------------
st.sidebar.header("ファイルを添付")
uploaded_file = st.sidebar.file_uploader(
    "テキスト / PDF / Word", type=["txt", "md", "pdf", "docx", "doc"], accept_multiple_files=False
)

def read_text_file(file):
    raw = file.read()
    for enc in ("utf-8", "cp932"):
        try:
            return raw.decode(enc, errors="ignore")[:180_000]
        except UnicodeDecodeError:
            continue
    return raw.decode(errors="ignore")[:180_000]

def looks_garbled(text: str) -> bool:
    if not text:
        return True
    bad = text.count(" ") + text.count("\ufffd") + text.count("(cid:")
    return (bad / len(text)) > 0.10

def extract_text_from_pdf(file_obj) -> str:
    data = file_obj.read()
    bio = BytesIO(data)

    # 1) pdfminer.six
    try:
        from pdfminer.high_level import extract_text
        text = extract_text(bio)
        if text.strip() and not looks_garbled(text):
            return text[:180_000]
    except Exception as e:
        logging.warning("pdfminer 失敗: %s", e)

    # 2) PyPDF2
    try:
        import PyPDF2
        reader = PyPDF2.PdfReader(BytesIO(data))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        if text.strip() and not looks_garbled(text):
            return text[:180_000]
    except Exception as e:
        logging.warning("PyPDF2 失敗: %s", e)

    # 3) PyMuPDF
    try:
        import fitz
        doc = fitz.open(stream=data, filetype="pdf")
        text = "\n".join(page.get_text() for page in doc)
        if text.strip() and not looks_garbled(text):
            return text[:180_000]
    except Exception as e:
        logging.warning("PyMuPDF 失敗: %s", e)

    # 4) OCR – Poppler + Tesseract
    try:
        from pdf2image import convert_from_bytes
        import pytesseract
        pages = convert_from_bytes(data, dpi=300, fmt="png", poppler_path=str(POPPLER_DIR))
        pytesseract.pytesseract.tesseract_cmd = str(TESSERACT_EXE)
        ocr_text = "\n".join(pytesseract.image_to_string(p, lang="jpn") for p in pages)
        if ocr_text.strip():
            return ocr_text[:180_000]
    except Exception as e:
        logging.warning("OCR 失敗: %s", e)

    return "(PDF からテキストを抽出できませんでした)"

def extract_text_from_word(file_obj) -> str:
    try:
        file_obj.seek(0)
        doc = Document(file_obj)
        text = "\n".join(para.text for para in doc.paragraphs)
        if text.strip():
            return text[:180_000]
    except Exception
