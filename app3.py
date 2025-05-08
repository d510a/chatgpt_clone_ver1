# ===== app8.py – Secrets 対応 + フォント埋め込みで文字化け解消 =====
import os, json, logging, sys, importlib.metadata as imd
from io import BytesIO
from pathlib import Path
from typing import Final
import streamlit as st
from dotenv import load_dotenv

# ------------------------------------------------ 0) 共通設定
load_dotenv()
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s ─ %(message)s")

# --- 0-1) 認証情報読み込み（Secrets → proxy_config.json） ------------
username = password = api_key = proxy_host = ""
if "api_key" in st.secrets:                                   # Cloud
    api_key    = st.secrets["api_key"]
    username   = st.secrets.get("proxy_username", "")
    password   = st.secrets.get("proxy_password", "")
    proxy_host = st.secrets.get("proxy_host",
                                "proxy01.hm.jp.honda.com:8080")
else:                                                         # Local
    cfg_file = Path.home() / "Documents" / "proxy_config.json"
    if cfg_file.exists():
        with cfg_file.open(encoding="utf-8") as f:
            cfg        = json.load(f)
            api_key    = cfg.get("apikey", "")
            username   = cfg.get("username", "")
            password   = cfg.get("password", "")
            proxy_host = cfg.get("proxyhost",
                                 "proxy01.hm.jp.honda.com:8080")

proxy_url = (f"http://{username}:{password}@{proxy_host}"
             if username and password and proxy_host else None)

# ------------------------------------------------ 1) ページ設定 + フォント注入
st.set_page_config(page_title="ChatGPT_clone")

# --- 日本語フォントを読み込む → グローバル CSS で適用 ---------------
st.markdown(
    """
<link href="https://fonts.googleapis.com/css2?family=Noto+Sans+JP:wght@400;700&display=swap"
      rel="stylesheet">
<style>
html, body, [class*="css"] {
  font-family: 'Noto Sans JP','Hiragino Kaku Gothic ProN',
               Meiryo, sans-serif !important;
}
code, pre, div.stMarkdown code {
  font-family: 'Noto Sans JP', Menlo, Consolas, monospace !important;
  white-space: pre-wrap;            /* 折返し */
  word-break: break-word;
}
</style>
""",
    unsafe_allow_html=True
)

if not api_key:
    st.error("API キーが設定されていません。（Secrets または proxy_config.json）")
    st.stop()

# ------------------------------------------------ 2) OpenAI v0/v1 互換ラッパー
def detect_openai_v1() -> bool:
    try:
        return int(importlib.metadata.version("openai").split(".")[0]) >= 1
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

    def list_models(self):               # connectivity check
        return (self.client.models.list()
                if self.v1 else self.client.Model.list())

    def stream_chat_completion(self, messages, model="o3-2025-04-16"):
        if self.v1:
            return self.client.chat.completions.create(
                model=model, messages=messages, stream=True)
        return self.client.ChatCompletion.create(
            model=model, messages=messages, stream=True)

def create_openai_wrapper(api_key: str, proxy_url: str | None):
    w = OpenAIWrapper(api_key, proxy_url)
    try:
        w.list_models()
        logging.info("Connectivity OK via %s",
                     "proxy" if proxy_url else "direct")
        return w
    except Exception:
        if proxy_url is None:
            raise
        return OpenAIWrapper(api_key, None)

client = create_openai_wrapper(api_key, proxy_url)

# ------------------------------------------------ 3) ポータブル資産パス
BASE_DIR = Path(getattr(sys, "_MEIPASS",
                        Path(__file__).resolve().parent))
POPPLER_DIR   = BASE_DIR / "poppler"   / "bin"
TESSERACT_EXE = BASE_DIR / "tesseract" / "tesseract.exe"

# ------------------------------------------------ 4) セッション初期化
GREETING = "質問してみましょう"
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system",
                                  "content": "You are a helpful assistant."}]
if not any(m["role"] == "assistant" and m["content"] == GREETING
           for m in st.session_state.messages):
    st.session_state.messages.insert(1,
        {"role": "assistant", "content": GREETING})
st.session_state.setdefault("uploaded_files", {})

# ------------------------------------------------ 5) ファイル添付 & PDF 抽出
st.sidebar.header("ファイルを添付")
uploaded = st.sidebar.file_uploader(
    "テキスト / Markdown / PDF", type=["txt", "md", "pdf"])

def read_text_file(file):
    raw = file.read()
    for enc in ("utf-8", "cp932"):
        try:
            return raw.decode(enc, errors="ignore")[:180_000]
        except UnicodeDecodeError:
            continue
    return raw.decode(errors="ignore")[:180_000]

def looks_garbled(t: str) -> bool:
    bad = t.count("�") + t.count("\ufffd") + t.count("(cid:")
    return not t or (bad / len(t)) > 0.10

def extract_text_from_pdf(file_obj) -> str:
    data = file_obj.read()
    bio  = BytesIO(data)

    # 1) pdfminer.six
    try:
        from pdfminer.high_level import extract_text
        txt = extract_text(bio)
        if txt.strip() and not looks_garbled(txt):
            return txt[:180_000]
    except Exception as e:
        logging.warning("pdfminer 失敗: %s", e)

    # 2) PyPDF2
    try:
        import PyPDF2
        reader = PyPDF2.PdfReader(BytesIO(data))
        txt = "\n".join(p.extract_text() or "" for p in reader.pages)
        if txt.strip() and not looks_garbled(txt):
            return txt[:180_000]
    except Exception as e:
        logging.warning("PyPDF2 失敗: %s", e)

    # 3) PyMuPDF
    try:
        import fitz
        doc = fitz.open(stream=data, filetype="pdf")
        txt = "\n".join(p.get_text() for p in doc)
        if txt.strip() and not looks_garbled(txt):
            return txt[:180_000]
    except Exception as e:
        logging.warning("PyMuPDF 失敗: %s", e)

    # 4) OCR
    try:
        from pdf2image import convert_from_bytes
        import pytesseract
        pages = convert_from_bytes(data, dpi=300, fmt="png",
                                   poppler_path=str(POPPLER_DIR))
        pytesseract.pytesseract.tesseract_cmd = str(TESSERACT_EXE)
        txt = "\n".join(pytesseract.image_to_string(p, lang="jpn")
                        for p in pages)
        if txt.strip():
            return txt[:180_000]
    except Exception as e:
        logging.warning("OCR 失敗: %s", e)

    return "(PDF からテキストを抽出できませんでした)"

if uploaded:
    st.sidebar.write(f" **{uploaded.name}** "
                     f"({uploaded.size//1024:,} KB) を読み込みました")
    if uploaded.name not in st.session_state.uploaded_files:
        content = (extract_text_from_pdf(uploaded)
                   if uploaded.type == "application/pdf"
                   else read_text_file(uploaded))
        st.session_state.uploaded_files[uploaded.name] = content

    if st.sidebar.button("チャットに送信"):
        txt = st.session_state.uploaded_files[uploaded.name]
        # Markdown の誤解釈を防ぐため全文を ```text ブロックに包む
        prompt = (f"以下はアップロードされたファイル {uploaded.name} の内容です:\n\n"
                  f"```text\n{txt}\n```")
        st.session_state.messages.append({"role": "user",
                                          "content": prompt})
        st.sidebar.success("ファイル内容をチャットへ送信しました")

# ------------------------------------------------ 6) メッセージ描画
st.title("ChatGPT_clone_o3")
st.caption("Streamlit + OpenAI (v0/v1 互換)")

for m in st.session_state.messages:
    if m["role"] == "system":
        continue
    with st.chat_message(m["role"]):
        # 添付テキストはコードブロック内なのでそのまま表示
        st.markdown(m["content"])

# ------------------------------------------------ 7) チャット入力
if prompt := st.chat_input("ここにメッセージを入力"):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user",
                                      "content": prompt})

    stream = client.stream_chat_completion(
        messages=st.session_state.messages, model="o3-2025-04-16")

    with st.chat_message("assistant"):
        ph, reply = st.empty(), ""
        for chunk in stream:
            delta = (chunk.choices[0].delta
                     if hasattr(chunk.choices[0], "delta")
                     else chunk.choices[0])
            reply += (delta.get("content", "")
                      if isinstance(delta, dict)
                      else delta.content or "")
            ph.markdown(reply + "▌")
        ph.markdown(reply)
    st.session_state.messages.append({"role": "assistant",
                                      "content": reply})

# ------------------------------------------------ 依存パッケージ
# pip install streamlit openai python-dotenv pdfminer.six PyPDF2 PyMuPDF
# pip install pdf2image pillow pytesseract
