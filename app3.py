import os
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path

# ───────────────────────────────
# 0) 環境設定
# ───────────────────────────────
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

st.set_page_config(page_title="ChatGPT_clone")

# ───────────────────────────────
# 1) セッション初期化
# ───────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = {}   # {filename: text}

# ───────────────────────────────
# 2) サイドバー：ファイル添付
# ───────────────────────────────
st.sidebar.header("ファイルを添付")
uploaded_file = st.sidebar.file_uploader(
    "テキスト / Markdown / PDF（最大 2MB）",
    type=["txt", "md", "pdf"],
    accept_multiple_files=False,
)

def read_text_file(file):
    "バイナリを UTF‑8 (fallback CP932) で文字列化し、8k 文字で切る"
    raw = file.read()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("cp932", errors="ignore")
    return text[:8000]  # 8k 文字に制限

if uploaded_file:
    file_size = uploaded_file.size / 1024
    st.sidebar.write(f"✅ **{uploaded_file.name}** ({file_size:,.0f} KB) を読み込みました")
    # すでに取り込んでいなければ保存
    if uploaded_file.name not in st.session_state.uploaded_files:
        if uploaded_file.type == "application/pdf":
            try:
                import PyPDF2
                reader = PyPDF2.PdfReader(uploaded_file)
                text = "\n".join(page.extract_text() or "" for page in reader.pages)
            except Exception:
                text = "(PDF をテキスト抽出できませんでした)"
        else:
            text = read_text_file(uploaded_file)
        st.session_state.uploaded_files[uploaded_file.name] = text

    if st.sidebar.button("チャットに送信"):
        text = st.session_state.uploaded_files[uploaded_file.name]
        prompt = f"以下はアップロードされたファイル **{uploaded_file.name}** の内容です:\n\n{text}"
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.sidebar.success("ファイル内容をチャットへ送信しました")

# ───────────────────────────────
# 3) ヘッダー
# ───────────────────────────────
st.title("ChatGPT_clone_o3")
st.caption("Streamlit + OpenAI")

# これまでの会話を描画
for m in st.session_state.messages[1:]:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# ───────────────────────────────
# 4) チャット入力
# ───────────────────────────────
if prompt := st.chat_input("ここにメッセージを入力"):
    # ユーザー側
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # OpenAI へ送信（ストリーミング）
    stream = client.chat.completions.create(
        model="o3-2025-04-16",
        messages=st.session_state.messages,
        stream=True,
    )

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_reply = ""
        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            full_reply += delta
            placeholder.markdown(full_reply + "▌")
        placeholder.markdown(full_reply)

    st.session_state.messages.append({"role": "assistant", "content": full_reply})
