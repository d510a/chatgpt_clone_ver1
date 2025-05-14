import os
import sys
import json                    # ★追加
import logging
import importlib.metadata as imd
from io import BytesIO
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components  # ★追加
from dotenv import load_dotenv
from docx import Document

# ────────────── 環境変数 & ロギング ──────────────
load_dotenv()
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s ─ %(message)s")

api_key = st.secrets.get("api_key", os.getenv("API_KEY", ""))
st.set_page_config(page_title="ChatGPT_clone")

if not api_key:
    st.error("API キーが設定されていません。（Secrets または .env）")
    st.stop()

# ────────────── OpenAI v0/v1 互換ラッパー ──────────────
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
            return self.client.chat.completions.create(
                model=model, messages=messages, stream=True
            )
        return self.client.ChatCompletion.create(
            model=model, messages=messages, stream=True
        )

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

# ────────────── セッション初期化 ──────────────
BASE_DIR = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
POPPLER_DIR = BASE_DIR / "poppler" / "bin"
TESSERACT_EXE = BASE_DIR / "tesseract" / "tesseract.exe"

GREETING = "質問してみましょう"
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": "You are a helpful assistant."}]
if not any(m["role"] == "assistant" and m["content"] == GREETING for m in st.session_state.messages):
    st.session_state.messages.insert(1, {"role": "assistant", "content": GREETING})
st.session_state.setdefault("uploaded_files", {})

# ────────────── ファイル添付 & 抽出（省略：従来通り） ──────────────
#   …ここは元のコードそのまま …

# ────────────── メッセージ描画 ──────────────
st.title("ChatGPT_clone_o3")
st.caption("Streamlit + OpenAI")

for m in st.session_state.messages:
    if m["role"] == "system":
        continue
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# ────────────── チャット入力 & 応答 ──────────────
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
            delta = chunk.choices[0].delta if hasattr(chunk.choices[0], "delta") else chunk.choices[0]
            reply += (delta.get("content", "") if isinstance(delta, dict) else delta.content or "")
            placeholder.markdown(reply + "▌")
        placeholder.markdown(reply)

        # ★★★ 「コピー」ボタン (コードブロック不要) ★★★
        btn_html = f'''
            <button
                style="margin-top:6px;padding:4px 12px;border:1px solid #bbb;border-radius:4px;cursor:pointer;background:#eee;"
                onClick="navigator.clipboard.writeText({json.dumps(reply)})">
                📋 Copy
            </button>
        '''
        components.html(btn_html, height=38)

    st.session_state.messages.append({"role": "assistant", "content": reply})
