# -*- coding: utf-8 -*-
"""
ChatGPT_clone_o3 ― Streamlit + OpenAI クライアント
PDF OCR（英語専用）を高精度化し、バージョン不一致エラーを解消したフルコード；
モデル切替 UI・リセットボタン・Web 検索機能を追加
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
from typing import Literal, List, Dict

# ────────────────────────────────────────────────────────────────
# 外部パッケージ
# ────────────────────────────────────────────────────────────────
import streamlit as st
from dotenv import load_dotenv
from docx import Document  # python-docx

# Web 検索（duckduckgo_search が無い場合は警告を出して無効化）
try:
    from duckduckgo_search import DDGS  # pip install duckduckgo_search
    _DUCKDUCKGO_AVAILABLE = True
except Exception:                       # duckduckgo_search 未導入でも動くように
    _DUCKDUCKGO_AVAILABLE = False

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
# ★ UI ラベル → OpenAI モデル ID 変換表
# ────────────────────────────────────────────────────────────────
MODEL_NAME_TO_ID = {
    "o3": "o3-2025-04-16",
    "GPT-4.1": "gpt-4.1-2025-04-14",
}

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

    def stream_chat_completion(
        self,
        messages,
        # ★ デフォルトを GPT-4.1 に変更
        model: Literal[
            "o3-2025-04-16",
            "gpt-4.1-2025-04-14"
        ] = "gpt-4.1-2025-04-14",
    ):
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
# Web 検索ユーティリティ
# ────────────────────────────────────────────────────────────────
def search_web(query: str, num_results: int = 5) -> List[Dict[str, str]]:
    """
    DuckDuckGo で簡易検索して {title, url, snippet} を返す。
    duckduckgo_search が無い場合は空配列。
    """
    if not _DUCKDUCKGO_AVAILABLE:
        logging.warning("duckduckgo_search 未導入: Web 検索をスキップします")
        return []

    results: List[Dict[str, str]] = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, region="wt-wt", safesearch="moderate"):
                if len(results) >= num_results:
                    break
                results.append({
                    "title": r.get("title", ""),
                    "url": r.get("href", ""),
                    "snippet": r.get("body", "")
                })
    except Exception as exc:
        logging.warning("Web 検索失敗: %s", exc)

    return results

def format_search_results(results: List[Dict[str, str]]) -> str:
    """検索結果を LLM へ渡しやすい Markdown テキストへ整形"""
    if not results:
        return ""
    lines = ["### Web 検索結果（上位）"]
    for i, r in enumerate(results, 1):
        title = r['title'] or r['url']
        lines.append(f"{i}. **{title}** — {r['snippet']} ({r['url']})")
    return "\n".join(lines)

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
DEFAULT_GREETING = "質問してみましょう。"
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": "You are a helpful assistant."}]
if not any(m["role"] == "assistant" and m["content"] == DEFAULT_GREETING
           for m in st.session_state.messages):
    st.session_state.messages.insert(1, {"role": "assistant", "content": DEFAULT_GREETING})
st.session_state.setdefault("uploaded_files", {})
# ★ デフォルト値を GPT-4.1（大文字表記）に
st.session_state.setdefault("model_name", "GPT-4.1")
# ★ Web 検索トグル
st.session_state.setdefault("use_web_search", False)

# ────────────────────────────────────────────────────────────────
# 共通ユーティリティ
# ────────────────────────────────────────────────────────────────
def reset_chat() -> None:
    """チャット履歴とアップロード済み内容を完全にリセット"""
    st.session_state.messages = [{"role": "system", "content": "You are a helpful assistant."},
                                 {"role": "assistant", "content": DEFAULT_GREETING}]
    st.session_state.uploaded_files = {}
    st.sidebar.success("チャットをリセットしました")

def read_text_file(file) -> str:
    """プレーンテキスト／Markdownを安全に読み込む"""
    file.seek(0)
    raw = file.read()
    for enc in ("utf-8", "cp932"):
        try:
            return raw.decode(enc, errors="ignore")[:990_000]
        except UnicodeDecodeError:
            continue
    return raw.decode(errors="ignore")[:990_000]


def looks_garbled(text: str, threshold: float = 0.25) -> bool:
    """文字化けかどうかの簡易判定"""
    if not text:
        return True
    bad = text.count(" ") + text.count("\ufffd") + text.count("(cid:")
    return (bad / len(text)) > threshold


# ────────────────────────────────────────────────────────────────
# PDF 解析
# ────────────────────────────────────────────────────────────────
def extract_text_from_pdf(file_obj) -> str:
    """
    1) pdfminer.six
    2) PyPDF2
    3) PyMuPDF
    4) Poppler + Tesseract OCR（英語）
    """
    data = file_obj.read()

    # --- 1) pdfminer.six ------------------------------------------------
    try:
        from pdfminer.high_level import extract_text

        try:
            # 新 API (>=20221105) は BytesIO を渡せる
            text = extract_text(BytesIO(data))
        except TypeError:
            # 旧 API はパスのみ可
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(data)
                tmp.flush()
                text = extract_text(tmp.name)

        if text.strip() and not looks_garbled(text):
            file_obj.seek(0)
            return text[:990_000]
    except Exception as exc:
        logging.warning("pdfminer 失敗: %s", exc)

    # --- 2) PyPDF2 ------------------------------------------------------
    try:
        import PyPDF2

        reader = PyPDF2.PdfReader(BytesIO(data))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        if text.strip() and not looks_garbled(text):
            file_obj.seek(0)
            return text[:990_000]
    except Exception as exc:
        logging.warning("PyPDF2 失敗: %s", exc)

    # --- 3) PyMuPDF -----------------------------------------------------
    try:
        import fitz

        doc = fitz.open(stream=data, filetype="pdf")
        text = "\n".join(page.get_text() for page in doc)
        if text.strip() and not looks_garbled(text):
            file_obj.seek(0)
            return text[:990_000]
    except Exception as exc:
        logging.warning("PyMuPDF 失敗: %s", exc)

    # --- 4) OCR (Poppler + Tesseract) ----------------------------------
    try:
        from pdf2image import convert_from_bytes
        import pytesseract

        if not POPPLER_BIN:
            raise RuntimeError("Poppler bin not found")
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

        pages = convert_from_bytes(
            data,
            dpi=400,          # 英語小フォント対策で高解像度
            fmt="png",
            poppler_path=POPPLER_BIN,
        )

        ocr_config = "--oem 3 --psm 6 -l eng"
        ocr_text = "\n".join(
            pytesseract.image_to_string(img, config=ocr_config) for img in pages
        )

        if ocr_text.strip():
            file_obj.seek(0)
            return ocr_text[:990_000]

    except Exception as exc:
        logging.warning("OCR 失敗: %s", exc)
        st.sidebar.error(f"OCR 失敗: {exc}")

    file_obj.seek(0)
    return "(PDF からテキストを抽出できませんでした)"


# ────────────────────────────────────────────────────────────────
# Word 解析
# ────────────────────────────────────────────────────────────────
def extract_text_from_word(file_obj) -> str:
    """
    .docx → python-docx → mammoth → docx2txt
    .doc  → textract
    """
    suffix = Path(file_obj.name).suffix.lower()

    # --- .docx --------------------------------------------------------
    if suffix == ".docx":
        # 1) python-docx
        try:
            file_obj.seek(0)
            doc = Document(file_obj)
            text = "\n".join(p.text for p in doc.paragraphs)
            if text.strip():
                file_obj.seek(0)
                return text[:990_000]
        except Exception as exc:
            logging.warning(".docx 解析失敗 (python-docx): %s", exc)

        # 2) mammoth
        try:
            import mammoth

            file_obj.seek(0)
            result = mammoth.extract_raw_text(file_obj)
            text = result.value
            if text.strip():
                file_obj.seek(0)
                return text[:990_000]
        except Exception as exc:
            logging.warning(".docx 解析失敗 (mammoth): %s", exc)

        # 3) docx2txt
        try:
            import docx2txt

            file_obj.seek(0)
            with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as tmp:
                tmp.write(file_obj.read())
                tmp.flush()
                text = docx2txt.process(tmp.name)
            if text.strip():
                file_obj.seek(0)
                return text[:990_000]
        except Exception as exc:
            logging.warning(".docx 解析失敗 (docx2txt): %s", exc)

    # --- .doc (バイナリ) ---------------------------------------------
    if suffix == ".doc":  # ← textract は pandas をロードするため pdf ルートでは import しない
        try:
            import textract

            file_obj.seek(0)
            text = textract.process(file_obj, extension="doc").decode(errors="ignore")
            if text.strip():
                file_obj.seek(0)
                return text[:990_000]
        except Exception as exc:
            logging.warning(".doc 解析失敗 (textract): %s", exc)

    file_obj.seek(0)
    return "(Word ファイルからテキストを抽出できませんでした)"

# ────────────────────────────────────────────────────────────────
# サイドバー：モデル選択・ファイル添付・リセット・Web検索
# ────────────────────────────────────────────────────────────────
st.sidebar.header("設定")

# ① モデル選択
st.sidebar.selectbox(
    "モデル選択",
    list(MODEL_NAME_TO_ID.keys()),          # ★ ラベル一覧
    key="model_name",
    help="回答に使用する OpenAI モデルを切り替えます"
)

# ② Web 検索トグル
st.sidebar.checkbox(
    "Web 検索を使用する",
    key="use_web_search",
    help="チャット送信前に最新情報を検索し、結果も参照して回答します",
    disabled=not _DUCKDUCKGO_AVAILABLE
)

# ③ ファイルアップロード
st.sidebar.header("ファイル添付")
uploaded_file = st.sidebar.file_uploader(
    "", type=["txt", "md", "pdf", "docx", "doc"],
    accept_multiple_files=False,
)

if uploaded_file and uploaded_file.name.endswith(".PDF"):
    # Streamlit の PDF MIME 判定が大文字 .PDF でズレる時の保険
    st.sidebar.error("大文字 .PDF ファイルは拡張子を .pdf に変更してください。")
else:
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
                if uploaded_file.type == "application/pdf":
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

        if st.sidebar.button("ファイルを送信"):
            txt = st.session_state.uploaded_files[uploaded_file.name]
            st.session_state.messages.append({"role": "system", "content": txt})
            notice = f"ファイル **{uploaded_file.name}** を送信しました。"
            st.session_state.messages.append({"role": "user", "content": notice})
            st.sidebar.success("ファイルをチャットへ送信しました")

# ④ リセットボタン（必ずファイル送信ボタンの下に表示）
st.sidebar.divider()
st.sidebar.button("リセット", on_click=reset_chat)

# ────────────────────────────────────────────────────────────────
# チャット表示
# ────────────────────────────────────────────────────────────────
st.title("ChatGPT_clone")

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

    # ★ UI ラベル → OpenAI モデル ID へ変換
    model_id = MODEL_NAME_TO_ID.get(st.session_state["model_name"],
                                    st.session_state["model_name"])

    # ---- Web 検索を実行（必要なら） ---------------------------------
    messages_to_send = list(st.session_state.messages)  # shallow copy
    if st.session_state.get("use_web_search", False):
        search_results = search_web(prompt, num_results=5)
        formatted = format_search_results(search_results)
        if formatted:
            # LLM へも提示
            messages_to_send.insert(
                1,
                {
                    "role": "system",
                    "content": formatted + "\n\n必要に応じて参照して回答してください。"
                }
            )
            # ユーザー UI にも検索結果を表示
            with st.chat_message("system"):
                st.markdown(formatted)

    # ---- OpenAI へストリーミング ------------------------------------
    stream = client.stream_chat_completion(
        messages=messages_to_send,
        model=model_id,  # <-- OpenAI に渡すのはフル ID
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
