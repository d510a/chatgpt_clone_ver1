# chatgpt_clone_app.py  ★全文
import os
import sys
import logging
import importlib.metadata as imd
from io import BytesIO
from pathlib import Path
import re
import unicodedata

import streamlit as st
from unidecode import unidecode                 # ← NEW
from dotenv import load_dotenv

# ---------------- 共通設定 ----------------
load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s ─ %(message)s",
)

api_key = st.secrets.get("api_key", os.getenv("API_KEY", ""))
if not api_key:
    st.error("APIキーが設定されていません (.env または Secrets)")
    st.stop()

def _is_v1() -> bool:
    try:
        return int(imd.version("openai").split(".")[0]) >= 1
    except Exception:
        return False

_IS_V1 = _is_v1()
if _IS_V1:
    from openai import OpenAI
else:
    import openai as _openai_legacy

class OpenAIWrapper:
    def __init__(self, key: str):
        if _IS_V1:
            self.client = OpenAI(api_key=key)
        else:
            _openai_legacy.api_key = key
            self.client = _openai_legacy

    def list_models(self):
        return self.client.models.list() if _IS_V1 else self.client.Model.list()

    def stream_chat(self, messages, model="o3-2025-04-16"):
        if _IS_V1:
            return self.client.chat.completions.create(
                model=model, messages=messages, stream=True
            )
        return self.client.ChatCompletion.create(
            model=model, messages=messages, stream=True
        )

try:
    client = OpenAIWrapper(api_key)
    client.list_models()
except Exception as e:
    st.error(f"OpenAI 接続に失敗しました: {e}")
    st.stop()

# ---------------- サニタイズ関数 ----------------
VALID_CHARS = r"[^A-Za-z0-9\.\_\-]"
def sanitize_filename(name: str) -> str:
    """
    日本語・絵文字などを含むファイル名を ASCII のみに変換して返す。
    拡張子は保持する。
    """
    stem, ext = os.path.splitext(name)
    stem_norm = unicodedata.normalize("NFKD", stem)          # ①正規化
    stem_ascii = unidecode(stem_norm)                        # ②ASCII 化
    stem_safe = re.sub(VALID_CHARS, "_", stem_ascii).strip("_.") or "file"
    return f"{stem_safe}{ext or ''}"

# ---------------- 画面レイアウト ----------------
st.set_page_config(page_title="ChatGPT_clone")
st.title("ChatGPT_clone_o3")
st.caption("Streamlit + OpenAI")

# -------- セッション初期化 --------
GREETING = "質問してみましょう"
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "assistant", "content": GREETING},
    ]
st.session_state.setdefault("uploaded", {})     # {safe_name: (orig_name, text)}

# -------- テキスト抽出ユーティリティ --------
import tiktoken
ENC = tiktoken.encoding_for_model("gpt-3.5-turbo")

def clip_text(text: str, char_limit=16_000, tok_limit=3_500) -> str:
    if len(text) > char_limit:
        text = text[:char_limit] + "\n...(truncated)"
    tokens = ENC.encode(text)
    if len(tokens) > tok_limit:
        text = ENC.decode(tokens[:tok_limit]) + "\n...(truncated)"
    return text

# ... read_text_file / extract_pdf / extract_word は変更なし ...

# -------- ファイルアップローダ --------
st.sidebar.header("ファイルを添付")
uploaded = st.sidebar.file_uploader(
    "TXT / PDF / DOCX", type=["txt", "md", "pdf", "docx", "doc"]
)

if uploaded:
    safe_name = sanitize_filename(uploaded.name)        # ★ ここで ASCII 化
    st.sidebar.success(
        f"**{uploaded.name}** を読み込みました "
        f"({uploaded.size//1024:,} KB) → 内部名: `{safe_name}`"
    )

    # 同じファイルを再アップロードしてもパースを繰り返さない
    if safe_name not in st.session_state.uploaded:
        mime = uploaded.type or ""
        name_l = uploaded.name.lower()
        if mime == "application/pdf" or name_l.endswith(".pdf"):
            content = extract_pdf(uploaded)
        elif mime in (
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "application/msword",
        ) or name_l.endswith((".docx", ".doc")):
            content = extract_word(uploaded)
        else:
            content = read_text_file(uploaded)
        # (元名, 本文) を保存
        st.session_state.uploaded[safe_name] = (uploaded.name, content)

    if st.sidebar.button("ファイル内容を送信"):
        orig, txt = st.session_state.uploaded[safe_name]
        st.session_state.messages.append(
            {"role": "system", "content": f"【元ファイル名: {orig}】\n\n{txt}"}
        )
        st.session_state.messages.append(
            {"role": "user", "content": f"{orig} を送信しました"}
        )
        st.sidebar.info("チャットへ送信しました")

# -------- メッセージ描画 --------
for m in st.session_state.messages:
    if m["role"] == "system":
        continue
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# -------- 入力 & 応答 --------
if prompt := st.chat_input("ここに入力"):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    stream = client.stream_chat(st.session_state.messages)
    with st.chat_message("assistant"):
        placeholder, reply = st.empty(), ""
        for chunk in stream:
            delta = chunk.choices[0].delta if hasattr(chunk.choices[0], "delta") else chunk.choices[0]
            reply += delta.get("content", "") if isinstance(delta, dict) else (delta.content or "")
            placeholder.markdown(reply + "▌")
        placeholder.markdown(reply)
    st.session_state.messages.append({"role": "assistant", "content": reply})
