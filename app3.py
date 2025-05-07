# ===== app8.py – PDF 抽出強化 & v0/v1 互換ラッパー =====
import os, json, logging, traceback, sys, importlib.metadata as imd
from io import BytesIO
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# ------------------------------------------------ 0) 共通設定
load_dotenv()
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s ─ %(message)s")

documents_folder = Path.home() / "Documents"
config_file = documents_folder / "proxy_config.json"

default_username = default_password = default_api_key = ""
if config_file.exists():
    try:
        with config_file.open(encoding="utf-8") as f:
            cfg = json.load(f)
            default_username = cfg.get("username", "")
            default_password = cfg.get("password", "")
            default_api_key  = cfg.get("apikey",  "")
    except Exception as e:
        logging.warning("proxy_config.json の読み込みに失敗: %s", e)

# ------------------------------------------------ 1) ページ設定
st.set_page_config(page_title="ChatGPT_clone")

# ------------------------------------------------ 2) サイドバー：認証
st.sidebar.header("プロキシ & API 設定")
with st.sidebar.form(key="proxy_form"):
    username = st.text_input("プロキシユーザ名",  value=default_username, key="proxy_user")
    password = st.text_input("プロキシパスワード", value=default_password, type="password", key="proxy_pass")
    api_key  = st.text_input("APIキー",            value=default_api_key,  type="password", key="api_key")
    submitted = st.form_submit_button("保存して接続")

if submitted:
    try:
        documents_folder.mkdir(exist_ok=True)
        with config_file.open("w", encoding="utf-8") as f:
            json.dump({"username": username, "password": password, "apikey": api_key},
                      f, ensure_ascii=False, indent=2)
        st.sidebar.success("認証情報を保存しました。")
        st.rerun()
    except Exception as e:
        st.sidebar.error("認証情報の保存に失敗しました。詳細はログを確認してください。")
        logging.error("保存失敗: %s\n%s", e, traceback.format_exc())

if not (username and password and api_key):
    st.info("サイドバーでプロキシ情報と API キーを入力するとチャットが利用できます。")
    st.stop()

# ------------------------------------------------ 3) OpenAI v0/v1 互換ラッパー
def detect_openai_v1() -> bool:
    try:
        return int(imd.version("openai").split(".")[0]) >= 1
    except Exception:
        return False

_IS_V1 = detect_openai_v1()
if _IS_V1:
    from openai import OpenAI
else:
    import openai as _openai_legacy  # noqa: F401

class OpenAIWrapper:
    """v0 (<1.0) / v1 (>=1.0) を吸収する単純ラッパー"""
    def __init__(self, api_key: str, proxy_url: str | None):
        self.v1 = _IS_V1
        # プロキシは環境変数で統一
        if proxy_url:
            os.environ["HTTP_PROXY"]  = proxy_url
            os.environ["HTTPS_PROXY"] = proxy_url
        else:
            os.environ.pop("HTTP_PROXY",  None)
            os.environ.pop("HTTPS_PROXY", None)

        if self.v1:
            self.client = OpenAI(api_key=api_key)
        else:
            import openai  # local alias
            openai.api_key = api_key
            self.client = openai

    # ---- 共通 API ラップ ----
    def list_models(self):
        return self.client.models.list() if self.v1 else self.client.Model.list()

    def stream_chat_completion(self, messages, model="o3-2025-04-16"):
        if self.v1:
            return self.client.chat.completions.create(
                model=model, messages=messages, stream=True
            )
        return self.client.ChatCompletion.create(
            model=model, messages=messages, stream=True
        )

def create_openai_wrapper(api_key: str, proxy_url: str | None) -> OpenAIWrapper:
    wrapper = OpenAIWrapper(api_key, proxy_url)
    try:
        wrapper.list_models()  # connectivity check
        logging.info("Connectivity OK via %s", "proxy" if proxy_url else "direct")
        return wrapper
    except Exception:
        if proxy_url is None:
            raise
        # プロキシ経由が失敗したら直接再試行
        return OpenAIWrapper(api_key, None)

proxy_url = f"http://{username}:{password}@proxy01.hm.jp.honda.com:8080"
client = create_openai_wrapper(api_key, proxy_url)

# ------------------------------------------------ 4) ポータブル資産パス
BASE_DIR = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
POPPLER_DIR   = BASE_DIR / "poppler"   / "bin"   # --add-data で同梱
TESSERACT_EXE = BASE_DIR / "tesseract" / "tesseract.exe"

# ------------------------------------------------ 5) セッション初期化
GREETING = "質問してみましょう"
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system",    "content": "You are a helpful assistant."},
    ]
if not any(m["role"] == "assistant" and m["content"] == GREETING
           for m in st.session_state.messages):
    st.session_state.messages.insert(1, {"role": "assistant",
                                         "content": GREETING})
st.session_state.setdefault("uploaded_files", {})

# ------------------------------------------------ 6) ファイル添付 & PDF 抽出
st.sidebar.header("ファイルを添付")
uploaded_file = st.sidebar.file_uploader(
    "テキスト / Markdown / PDF",
    type=["txt", "md", "pdf"],
    accept_multiple_files=False,
)

def read_text_file(file):
    raw = file.read()
    try:
        return raw.decode("utf-8")[:180_000]
    except UnicodeDecodeError:
        return raw.decode("cp932", errors="ignore")[:180_000]

def extract_text_from_pdf(file_obj) -> str:
    data = file_obj.read()
    bio = BytesIO(data)

    # 1) pdfminer.six
    try:
        from pdfminer.high_level import extract_text
        text = extract_text(BytesIO(data))
        if text.strip():
            return text[:180_000]
    except Exception as e:
        logging.warning("pdfminer 失敗: %s", e)

    # 2) PyPDF2
    try:
        import PyPDF2
        reader = PyPDF2.PdfReader(BytesIO(data))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        if text.strip():
            return text[:180_000]
    except Exception as e:
        logging.warning("PyPDF2 失敗: %s", e)

    # 3) OCR – Poppler + Tesseract
    try:
        from pdf2image import convert_from_bytes
        import pytesseract
        pages = convert_from_bytes(data, dpi=300, fmt="png",
                                   poppler_path=str(POPPLER_DIR))
        pytesseract.pytesseract.tesseract_cmd = str(TESSERACT_EXE)
        ocr_text = "\n".join(
            pytesseract.image_to_string(p, lang="jpn") for p in pages
        )
        if ocr_text.strip():
            return ocr_text[:180_000]
    except Exception as e:
        logging.warning("OCR 失敗: %s", e)

    return "(PDF からテキストを抽出できませんでした)"

if uploaded_file:
    st.sidebar.write(f" **{uploaded_file.name}** "
                     f"({uploaded_file.size//1024:,} KB) を読み込みました")
    if uploaded_file.name not in st.session_state.uploaded_files:
        content = (extract_text_from_pdf(uploaded_file)
                   if uploaded_file.type == "application/pdf"
                   else read_text_file(uploaded_file))
        st.session_state.uploaded_files[uploaded_file.name] = content

    if st.sidebar.button("チャットに送信"):
        txt = st.session_state.uploaded_files[uploaded_file.name]
        prompt = (f"以下はアップロードされたファイル **{uploaded_file.name}** の内容です:\n\n"
                  f"{txt}")
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.sidebar.success("ファイル内容をチャットへ送信しました")

# ------------------------------------------------ 7) メッセージ描画
st.title("ChatGPT_clone_o3")
st.caption("Streamlit + OpenAI (v0/v1 互換)")

for m in st.session_state.messages:
    if m["role"] == "system":  # system は非表示
        continue
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# ------------------------------------------------ 8) チャット入力
if prompt := st.chat_input("ここにメッセージを入力"):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    stream = client.stream_chat_completion(
        messages=st.session_state.messages,
        model="o3-2025-04-16",
    )

    with st.chat_message("assistant"):
        placeholder, reply = st.empty(), ""
        for chunk in stream:
            # v0 と v1 で field 名が一致する
            delta = (chunk.choices[0].delta
                     if hasattr(chunk.choices[0], "delta") else chunk.choices[0])
            reply += (delta.get("content","")  # v0 dict
                      if isinstance(delta, dict)
                      else delta.content or "")  # v1 obj
            placeholder.markdown(reply + "▌")
        placeholder.markdown(reply)
    st.session_state.messages.append({"role": "assistant",
                                      "content": reply})

# ------------------------------------------------ 依存パッケージ (例)
# pip install streamlit openai python-dotenv pdfminer.six PyPDF2
# pip install pdf2image pillow pytesseract
# poppler / tesseract-ocr 実行ファイルを --add-data で同梱
