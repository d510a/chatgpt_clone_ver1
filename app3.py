# -*- coding: utf-8 -*-
"""
ChatGPT_clone_o3 ― Streamlit + OpenAI クライアント
"""
import os
import sys
import logging
import traceback
from io import BytesIO
from pathlib import Path
import tempfile

import streamlit as st
from dotenv import load_dotenv
from docx import Document  # python-docx
from pdfminer.high_level import extract_text  # pdfminer.six
import fitz  # PyMuPDF

# ────────────────────────────────────────────────────────────────
# 環境変数／ログ設定
# ────────────────────────────────────────────────────────────────
load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s ─ %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
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
# （openai==1.x／0.x のどちらでも動くように簡易判定）
# ────────────────────────────────────────────────────────────────
def detect_openai_v1():
    try:
        import openai

        if hasattr(openai, "OpenAI"):
            return True
    except Exception:
        pass
    return False


def create_openai_client(api_key: str):
    if detect_openai_v1():
        import openai

        return openai.OpenAI(api_key=api_key)
    else:
        import openai

        openai.api_key = api_key
        return openai


client = create_openai_client(api_key)

# ────────────────────────────────────────────────────────────────
# ヘルパー：PDF / Word / テキスト抽出
# ────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def extract_text_from_pdf(file_obj: BytesIO) -> str:
    """PDF → テキスト抽出（pdfminer または PyMuPDF フォールバック）"""
    try:
        return extract_text(file_obj)
    except Exception:
        logging.warning("pdfminer 失敗、PyMuPDF にフォールバック")
        try:
            with fitz.open(stream=file_obj.read(), filetype="pdf") as doc:
                return "\n".join(page.get_text() for page in doc)
        except Exception as e:
            logging.error("PyMuPDF も失敗: %s", e)
            return "(PDF からテキストを抽出できませんでした)"


@st.cache_data(show_spinner=False)
def extract_text_from_word(file_obj: BytesIO) -> str:
    """Word (.docx) → テキスト抽出"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
            tmp.write(file_obj.read())
            tmp_path = tmp.name
        doc = Document(tmp_path)
        txt = "\n".join(p.text for p in doc.paragraphs)
        os.unlink(tmp_path)
        return txt
    except Exception as e:
        logging.error("Word 抽出失敗: %s", e)
        return "(Word ファイルからテキストを抽出できませんでした)"


@st.cache_data(show_spinner=False)
def read_text_file(file_obj: BytesIO) -> str:
    """テキストファイル読み込み（UTF-8 / SHIFT-JIS / CP932 自動判定）"""
    try:
        data = file_obj.read()
        for enc in ("utf-8", "cp932", "shift_jis", "euc_jp"):
            try:
                return data.decode(enc)
            except Exception:
                continue
        return data.decode("utf-8", errors="replace")
    except Exception as e:
        logging.error("テキスト読み込み失敗: %s", e)
        return "(テキストを読み込めませんでした)"

# ────────────────────────────────────────────────────────────────
# ステート初期化
# ────────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = {}

# ────────────────────────────────────────────────────────────────
# サイドバー：ファイル添付 & セッション保存
# ────────────────────────────────────────────────────────────────
st.sidebar.header("ファイルを添付")
uploaded_file = st.sidebar.file_uploader(
    "テキスト / PDF / Word",
    type=["txt", "md", "pdf", "PDF", "docx", "doc"],  # ← 大文字 PDF を許可
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
            if (
                uploaded_file.type == "application/pdf"
                or uploaded_file.name.lower().endswith(".pdf")
            ):
                content = extract_text_from_pdf(uploaded_file)
            elif uploaded_file.name.lower().endswith((".docx", ".doc")):
                content = extract_text_from_word(uploaded_file)
            else:
                content = read_text_file(uploaded_file)
        except Exception:
            # 解析中に落ちたらログをサイドバーに表示
            st.sidebar.code(traceback.format_exc())
            content = "(ファイル解析中にエラーが発生しました)"
        st.session_state.uploaded_files[uploaded_file.name] = content

    if st.sidebar.button("ファイル内容を送信"):
        txt = st.session_state.uploaded_files[uploaded_file.name]
        st.session_state.messages.append({"role": "system", "content": txt})

# ────────────────────────────────────────────────────────────────
# メインチャット画面
# ────────────────────────────────────────────────────────────────
st.title("ChatGPT Clone (o3)")

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).markdown(msg["content"])

if prompt := st.chat_input("メッセージを入力して送信 (Shift+Enter で改行)"):
    st.session_state.messages.append({"role": "user", "content": prompt})

    # --- OpenAI へ問い合わせ ---
    try:
        response_content = "(API レスポンスなし)"
        if detect_openai_v1():
            chat_completion = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": m["role"], "content": m["content"]} for m in st.session_state.messages],
            )
            response_content = chat_completion.choices[0].message.content
        else:
            chat_completion = client.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": m["role"], "content": m["content"]} for m in st.session_state.messages],
            )
            response_content = chat_completion.choices[0].message["content"]

        st.session_state.messages.append(
            {"role": "assistant", "content": response_content}
        )
        st.chat_message("assistant").markdown(response_content)
    except Exception:
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": "⚠️ OpenAI API への問い合わせでエラーが発生しました。",
            }
        )
        st.chat_message("assistant").markdown(
            "⚠️ OpenAI API への問い合わせでエラーが発生しました。\n\n```\n"
            + traceback.format_exc()
            + "\n```"
        )
