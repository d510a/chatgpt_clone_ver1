import os
import logging
import sys
import importlib.metadata as imd
from io import BytesIO
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from docx import Document

# 環境変数読み込み
load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s ─ %(message)s")

# --- 認証情報読み込み ----------------------------------------------
api_key = ""
if "api_key" in st.secrets:
    # Streamlit Secrets から API キーを取得
    api_key = st.secrets["api_key"]
else:
    # .env から API キーを取得
    api_key = os.getenv("API_KEY", "")

# ページ設定
st.set_page_config(page_title="ChatGPT_clone")

if not api_key:
    st.error("API キーが設定されていません。（Secrets または .env）")
    st.stop()

# --- OpenAI v0/v1 互換ラッパー --------------------------------------
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
        # モデル一覧の取得
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

# --- セッション初期化 ----------------------------------------------
BASE_DIR = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
POPPLER_DIR = BASE_DIR / "poppler" / "bin"
TESSERACT_EXE = BASE_DIR / "tesseract" / "tesseract.exe"

GREETING = "質問してみましょう"
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": "You are a helpful assistant."}]
if not any(m["role"] == "assistant" and m["content"] == GREETING for m in st.session_state.messages):
    st.session_state.messages.insert(1, {"role": "assistant", "content": GREETING})
st.session_state.setdefault("uploaded_files", {})

# --- ファイル添付 & 抽出 -------------------------------------------
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
    # DOCX 解析
    try:
        file_obj.seek(0)
        doc = Document(file_obj)
        text = "\n".join(para.text for para in doc.paragraphs)
        if text.strip():
            return text[:180_000]
    except Exception as e:
        logging.warning(".docx 解析失敗: %s", e)

    # DOC 解析 (textract 必要)
    try:
        import textract
        file_obj.seek(0)
        data = file_obj.read()
        text = textract.process(data, extension="doc").decode(errors="ignore")
        if text.strip():
            return text[:180_000]
    except Exception as e:
        logging.warning(".doc 解析失敗: %s", e)

    return "(Word ファイルからテキストを抽出できませんでした)"

if uploaded_file:
    st.sidebar.write(f" **{uploaded_file.name}** ({uploaded_file.size//1024:,} KB) を読み込みました")
    if uploaded_file.name not in st.session_state.uploaded_files:
        if uploaded_file.type == "application/pdf":
            content = extract_text_from_pdf(uploaded_file)
        elif uploaded_file.name.lower().endswith((".docx", ".doc")):
            content = extract_text_from_word(uploaded_file)
        else:
            content = read_text_file(uploaded_file)
        st.session_state.uploaded_files[uploaded_file.name] = content

    if st.sidebar.button("ファイル内容を送信"):
        txt = st.session_state.uploaded_files[uploaded_file.name]
        st.session_state.messages.append({"role": "system", "content": txt})
        notice = f"ファイル **{uploaded_file.name}** を送信しました。"
        st.session_state.messages.append({"role": "user", "content": notice})
        st.sidebar.success("ファイルをチャットへ送信しました")

# --- メッセージ描画 -----------------------------------------------
st.title("ChatGPT_clone_o3")
st.caption("Streamlit + OpenAI")

for m in st.session_state.messages:
    if m["role"] == "system":
        continue
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# --- チャット入力 -----------------------------------------------
if prompt := st.chat_input("ここにメッセージを入力"):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    stream = client.stream_chat_completion(
        messages=st.session_state.messages, model="o3-2025-04-16"
    )

    with st.chat_message("assistant"):
        placeholder, reply = st.empty(), ""
        for chunk in stream:
            delta = chunk.choices[0].delta if hasattr(chunk.choices[0], "delta") else chunk.choices[0]
            reply += (delta.get("content", "") if isinstance(delta, dict) else delta.content or "")
            placeholder.markdown(reply + "▌")
        placeholder.markdown(reply)
    st.session_state.messages.append({"role": "assistant", "content": reply})
