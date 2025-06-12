# -*- coding: utf-8 -*-
"""
ChatGPT_clone_o3 ― Streamlit + OpenAI クライアント
PDF OCR（英語専用）を高精度化し、バージョン不一致エラーを解消
※ .pdf / .PDF いずれも受け付けます
"""

# ────────────────────────────────────────────────────────────────
# 標準ライブラリ
# ────────────────────────────────────────────────────────────────
import os
import sys
import shutil
import logging
import importlib.metadata as imd
from io import BytesIO
from pathlib import Path
import tempfile
import traceback

# ────────────────────────────────────────────────────────────────
# 外部パッケージ
# ────────────────────────────────────────────────────────────────
import streamlit as st
from dotenv import load_dotenv
from docx import Document  # python-docx

# ────────────────────────────────────────────────────────────────
# 環境変数／ログ設定
# ────────────────────────────────────────────────────────────────
load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s ─ %(message)s",
)

# ────────────────────────────────────────────────────────────────
# 認証情報
# ────────────────────────────────────────────────────────────────
api_key = st.secrets.get("api_key", os.getenv("API_KEY", ""))
st.set_page_config(page_title="ChatGPT_clone")
if not api_key:
    st.error("API キーが設定されていません。（Secrets または .env）")
    st.stop()

# ────────────────────────────────────────────────────────────────
# OpenAI v0/v1 互換ラッパー
# ────────────────────────────────────────────────────────────────
def detect_openai_v1() -> bool:
    try:
        return int(imd.version("openai").split(".")[0]) >= 1
    except Exception:
        return False


_IS_V1 = detect_openai_v1()
if _IS_V1:
    from openai import OpenAI
else:
    import openai as _openai_legacy  # noqa: N812  pylint: disable=invalid-name


class OpenAIWrapper:
    """openai-python v0/v1 API 差分を吸収する簡易ラッパー"""

    def __init__(self, api_key: str) -> None:
        self.v1 = _IS_V1
        if self.v1:
            self.client = OpenAI(api_key=api_key)
        else:
            import openai

            openai.api_key = api_key
            self.client = openai

    def list_models(self):
        return self.client.models.list() if self.v1 else self.client.Model.list()

    def stream_chat_completion(self, messages, model: str = "o3-2025-04-16"):
        if self.v1:
            return self.client.chat.completions.create(
                model=model,
                messages=messages,
                stream=True,
            )
        return self.client.ChatCompletion.create(
            model=model,
            messages=messages,
            stream=True,
        )


def create_openai_wrapper(key: str) -> OpenAIWrapper:
    wrapper = OpenAIWrapper(key)
    try:
        wrapper.list_models()
        logging.info("Connectivity OK")
        return wrapper
    except Exception as exc:
        logging.error("OpenAI 接続に失敗: %s", exc)
        st.error("OpenAI への接続に失敗しました。")
        st.stop()


client = create_openai_wrapper(api_key)

# ────────────────────────────────────────────────────────────────
# 実行環境パスの解決
# ────────────────────────────────────────────────────────────────
BASE_DIR = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))  # PyInstaller 対応

BUNDLE_POPPLER = BASE_DIR / "poppler" / "bin"
BUNDLE_TESSERACT = BASE_DIR / "tesseract" / "tesseract.exe"


def resolve_poppler_bin() -> str | None:
    if BUNDLE_POPPLER.exists():
        return str(BUNDLE_POPPLER)
    if os.getenv("POPPLER_PATH"):
        return os.getenv("POPPLER_PATH")
    which_ppm = shutil.which("pdftoppm")
    return str(Path(which_ppm).parent) if which_ppm else None


def resolve_tesseract_cmd() -> str:
    if BUNDLE_TESSERACT.exists():
        return str(BUNDLE_TESSERACT)
    if os.getenv("TESSERACT_CMD"):
        return os.getenv("TESSERACT_CMD")
    return shutil.which("tesseract") or "tesseract"


POPPLER_BIN = resolve_poppler_bin()
TESSERACT_CMD = resolve_tesseract_cmd()

# ────────────────────────────────────────────────────────────────
# セッション初期化
# ────────────────────────────────────────────────────────────────
GREETING = "質問してみましょう"
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": "You are a helpful assistant."}]
if not any(m["role"] == "assistant" and m["content"] == GREETING for m in st.session_state.messages):
    st.session_state.messages.insert(1, {"role": "assistant", "content": GREETING})
st.session_state.setdefault("uploaded_files", {})

# ────────────────────────────────────────────────────────────────
# 共通ユーティリティ（…中略…  ※元のまま） 
#   - read_text_file
#   - looks_garbled
#   - extract_text_from_pdf
#   - extract_text_from_word
#   ……サンプルコードと同一なので省略……
# ────────────────────────────────────────────────────────────────

# ここからは **ファイルアップローダ** だけ修正
# ────────────────────────────────────────────────────────────────
st.sidebar.header("ファイルを添付")
uploaded_file = st.sidebar.file_uploader(
    "テキスト / PDF / Word",
    type=["txt", "md", "pdf", "docx", "doc"],  # <= すべて小文字＆一意
    accept_multiple_files=False,
)

if uploaded_file:
    # サイズ表示
    size_str = (
        f"{uploaded_file.size} B"
        if uploaded_file.size < 1024
        else f"{uploaded_file.size / 1024:.1f} KB"
    )
    st.sidebar.write(f" **{uploaded_file.name}** ({size_str}) を読み込みました")

    if uploaded_file.name not in st.session_state.uploaded_files:
        try:
            is_pdf = (
                uploaded_file.type == "application/pdf"
                or uploaded_file.name.lower().endswith(".pdf")
            )
            if is_pdf:
                content = extract_text_from_pdf(uploaded_file)
            elif uploaded_file.name.lower().endswith((".docx", ".doc")):
                content = extract_text_from_word(uploaded_file)
            else:
                content = read_text_file(uploaded_file)
        except Exception:
            st.sidebar.code(traceback.format_exc())
            content = "(ファイル解析中にエラーが発生しました)"
        st.session_state.uploaded_files[uploaded_file.name] = content

    if st.sidebar.button("ファイル内容を送信"):
        txt = st.session_state.uploaded_files[uploaded_file.name]
        st.session_state.messages.append({"role": "system", "content": txt})
        notice = f"ファイル **{uploaded_file.name}** を送信しました。"
        st.session_state.messages.append({"role": "user", "content": notice})
        st.sidebar.success("ファイルをチャットへ送信しました")

# ────────────────────────────────────────────────────────────────
# チャット表示 & 入力（…中略…  ※サンプルコードと同一）
# ────────────────────────────────────────────────────────────────
