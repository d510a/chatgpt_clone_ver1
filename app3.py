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
# 全ファイル受け入れ
uploaded_file = st.sidebar.file_uploader(
    "テキスト / PDF / Word", type=None, accept_multiple_files=False
)

allowed_exts = (".txt", ".md", ".pdf", ".docx", ".doc")

if uploaded_file:
    fname = uploaded_file.name
    # 手動で拡張子チェック（小文字化して判定）
    if not fname.lower().endswith(allowed_exts):
        # 対応外ファイルはエラー表示
        st.sidebar.error("application/vnd.openxmlformats-officedocument.spreadsheetml.sheet files are not allowed.")
    else:
        st.sidebar.write(f" **{fname}** ({uploaded_file.size//1024:,} KB) を読み込みました")
        # ファイル内容の抽出
        if fname.lower().endswith(".pdf"):
            content = extract_text_from_pdf(uploaded_file)
        elif fname.lower().endswith((".docx", ".doc")):
            content = extract_text_from_word(uploaded_file)
        else:
            content = read_text_file(uploaded_file)

        # 一度だけ保持
        if fname not in st.session_state.uploaded_files:
            st.session_state.uploaded_files[fname] = content

        # 送信ボタン表示
        if st.sidebar.button("ファイル内容を送信"):
            txt = st.session_state.uploaded_files[fname]
            st.session_state.messages.append({"role": "system", "content": txt})
            notice = f"ファイル **{fname}** を送信しました。"
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
