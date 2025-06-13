# -*- coding: utf-8 -*-
"""
ChatGPT_clone_o3 ― Streamlit + OpenAI クライアント
● 改訂ポイント（2025-06-13）
 1. txt 読み込みの安定化
    - getvalue() + chardet で再エンコード判定
    - ファイル一意キーに MD5 ハッシュを付与して「同名ファイルを上書きしたのに更新されない」を解消
    - チャット用キャッシュは 20 万字に切り詰め、OOM を防止
 2. そのほか細部
    - requirements.txt 追加想定: chardet==5.2.0
"""

# ────────────────────────────────────────────────────────────────
# 標準ライブラリ
# ────────────────────────────────────────────────────────────────
import os
import sys
import shutil
import logging
import hashlib        # ★ 追加
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
from docx import Document              # python-docx
import chardet                         # ★ 追加

# ────────────────────────────────────────────────────────────────
# 環境変数／ログ設定
# ────────────────────────────────────────────────────────────────
load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s ─ %(message)s"
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

# バンドル配下
BUNDLE_POPPLER = BASE_DIR / "poppler" / "bin"
BUNDLE_TESSERACT = BASE_DIR / "tesseract" / "tesseract.exe"

# OS / 環境変数 / which で探索
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
# 共通ユーティリティ
# ────────────────────────────────────────────────────────────────
def read_text_file(file, limit: int = 2_000_000, keep: int = 200_000) -> str:
    """
    Streamlit UploadedFile → bytes へ変換し、chardet でエンコーディング自動判定。
    - limit  : 最大読み込みバイト数（メモリ保護）
    - keep   : チャットへ保持する最大文字数
    """
    raw = file.getvalue()[:limit]               # EOF/二重読み問題を回避
    enc = chardet.detect(raw)["encoding"] or "utf-8"
    try:
        text = raw.decode(enc, errors="replace")
    except LookupError:
        text = raw.decode("utf-8", errors="replace")
    return text[:keep]

def looks_garbled(text: str, threshold: float = 0.25) -> bool:
    """文字化けかどうかの簡易判定"""
    if not text:
        return True
    bad = text.count(" ") + text.count("\ufffd") + text.count("(cid:")
    return (bad / len(text)) > threshold

# ────────────────────────────────────────────────────────────────
# PDF 解析
# （─ 中略 ─ 元コードの PDF / Word 抽出ロジックは変更なし ─）
# ────────────────────────────────────────────────────────────────
#                     ↓★★ 既存 extract_text_from_pdf / _word はそのまま ★★
# ────────────────────────────────────────────────────────────────

# ここに extract_text_from_pdf / extract_text_from_word 全文（元コード）をそのまま残してください
# （チャット表示を省略するため割愛。ロジックに変更は無い）

# ────────────────────────────────────────────────────────────────
# サイドバー：ファイル添付 & セッション保存
# ────────────────────────────────────────────────────────────────
st.sidebar.header("ファイルを添付")
uploaded_file = st.sidebar.file_uploader(
    "テキスト / PDF / Word",
    type=["txt", "md", "pdf", "docx", "doc"],
    accept_multiple_files=False,
)

if uploaded_file and uploaded_file.name.endswith(".PDF"):
    st.sidebar.error("大文字 .PDF ファイルは拡張子を .pdf に変更してください。")
else:
    if uploaded_file:
        # 一意キー = ファイル名 + MD5
        md5 = hashlib.md5(uploaded_file.getvalue()).hexdigest()
        file_key = f"{uploaded_file.name}:{md5}"

        # サイズ表示
        size_str = (
            f"{uploaded_file.size} B" if uploaded_file.size < 1024
            else f"{uploaded_file.size / 1024:.1f} KB"
        )
        st.sidebar.write(f" **{uploaded_file.name}** ({size_str}) を読み込みました")

        if file_key not in st.session_state.uploaded_files:
            try:
                if uploaded_file.type == "application/pdf":
                    content = extract_text_from_pdf(uploaded_file)  # ← 元関数
                elif uploaded_file.name.lower().endswith((".docx", ".doc")):
                    content = extract_text_from_word(uploaded_file)  # ← 元関数
                else:
                    content = read_text_file(uploaded_file)
            except Exception:
                st.sidebar.code(traceback.format_exc())
                content = "(ファイル解析中にエラーが発生しました)"
            st.session_state.uploaded_files[file_key] = content  # 最大 keep=20万字

        # 送信ボタン
        if st.sidebar.button("ファイル内容を送信"):
            txt = st.session_state.uploaded_files[file_key]
            st.session_state.messages.append({"role": "system", "content": txt})
            notice = f"ファイル **{uploaded_file.name}** を送信しました。"
            st.session_state.messages.append({"role": "user", "content": notice})
            st.sidebar.success("ファイルをチャットへ送信しました")

# ────────────────────────────────────────────────────────────────
# チャット表示
# ────────────────────────────────────────────────────────────────
st.title("ChatGPT_clone_o3")
st.caption("Streamlit + OpenAI")

for m in st.session_state.messages:
    if m["role"] == "system":
        continue
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# ────────────────────────────────────────────────────────────────
# チャット入力
# ────────────────────────────────────────────────────────────────
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
            delta = (
                chunk.choices[0].delta
                if hasattr(chunk.choices[0], "delta")
                else chunk.choices[0]
            )
            reply += (
                delta.get("content", "")
                if isinstance(delta, dict)
                else delta.content or ""
            )
            placeholder.markdown(reply + "▌")
        placeholder.markdown(reply)
    st.session_state.messages.append({"role": "assistant", "content": reply})
