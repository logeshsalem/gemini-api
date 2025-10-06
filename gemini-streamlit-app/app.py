import os
import io
import hashlib
from typing import List, Dict, Any

import streamlit as st
from PIL import Image
import google.generativeai as genai


# ----------------------------
# Page / App Configuration
# ----------------------------
st.set_page_config(
    page_title="Gemini Chat (Streamlit)",
    page_icon="✨",
    layout="centered",
    menu_items={
        "About": "A minimal, production-ready Streamlit chat UI for Gemini (Google Generative AI).",
    },
)

st.title("✨ Gemini Chat — Streamlit")
st.caption("Built with `google-generativeai` + Streamlit. Supports chat history, streaming, and image input.")


# ----------------------------
# Helpers
# ----------------------------

def get_api_key() -> str:
    """Resolve API key from secrets, env var, or sidebar input."""
    key = None
    try:
        if "GOOGLE_API_KEY" in st.secrets:
            key = st.secrets["GOOGLE_API_KEY"]
    except Exception:
        pass
    key = key or os.getenv("GOOGLE_API_KEY")
    if not key:
        with st.sidebar:
            st.error("No API key found. Enter your Gemini API key to continue.")
            key = st.text_input("GOOGLE_API_KEY", type="password", placeholder="Paste your API key here…")
            if key:
                st.info("Using the key you entered for this session.")
    return key


def safe_configure_genai(api_key: str) -> bool:
    try:
        genai.configure(api_key=api_key)
        return True
    except Exception as e:
        st.error(f"Failed to initialize Gemini client: {e}")
        return False


def compute_config_hash(model_name: str, system_instruction: str, gen_config: Dict[str, Any]) -> str:
    import json
    h = hashlib.sha256()
    h.update((model_name or "").encode())
    h.update((system_instruction or "").encode())
    h.update(json.dumps(gen_config, sort_keys=True).encode())
    return h.hexdigest()


def show_message(role: str, text: str = "", images: List[bytes] = None):
    """Render a chat message with optional images."""
    with st.chat_message(role):
        if text:
            st.markdown(text)
        if images:
            cols = st.columns(min(3, len(images)))
            for i, img_bytes in enumerate(images):
                try:
                    img = Image.open(io.BytesIO(img_bytes))
                    cols[i % len(cols)].image(img, use_column_width=True)
                except Exception:
                    st.write(f"Attached file #{i+1}")


def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []  # [{role: "user"|"assistant", "text": str, "images": [bytes,...]}]

    if "chat" not in st.session_state:
        st.session_state.chat = None

    if "config_hash" not in st.session_state:
        st.session_state.config_hash = ""

    if "pending_attachments" not in st.session_state:
        st.session_state.pending_attachments = []  # list of raw bytes


init_session_state()


# ----------------------------
# Sidebar Controls
# ----------------------------
with st.sidebar:
    st.subheader("⚙️ Settings")
    model_name = st.selectbox(
        "Model",
        options=["gemini-1.5-flash", "gemini-1.5-pro"],
        index=0,
        help="Choose a Gemini model. Flash is great for fast, general tasks; Pro is better for complex reasoning."
    )

    temperature = st.slider("Temperature", 0.0, 2.0, 0.7, 0.1, help="Creativity vs. determinism.")
    top_p = st.slider("Top-p (nucleus sampling)", 0.0, 1.0, 0.95, 0.05)
    top_k = st.slider("Top-k", 0, 100, 40, 1)
    max_output_tokens = st.slider("Max output tokens", 64, 8192, 1024, 64)

    system_instruction = st.text_area(
        "🧭 System Prompt (optional)",
        placeholder="E.g., 'You are a helpful assistant specialized in data analysis and Python.'",
        height=120,
    )

    st.markdown("---")
    st.caption("🔒 API Key")
    st.write(
        "Set `GOOGLE_API_KEY` via **Secrets** or **Environment**. "
        "You can also paste it below for this session."
    )

    # Attachments manager
    st.markdown("---")
    st.subheader("📎 Attach image(s) for next message")
    uploads = st.file_uploader(
        "Supported: PNG, JPG, JPEG, WEBP",
        type=["png", "jpg", "jpeg", "webp"],
        accept_multiple_files=True
    )
    if uploads:
        st.session_state.pending_attachments = [u.getvalue() for u in uploads]
        st.info(f"{len(st.session_state.pending_attachments)} image(s) will be attached to your next message.")
    if st.button("Clear attachments"):
        st.session_state.pending_attachments = []
        st.success("Pending attachments cleared.")

    st.markdown("---")
    if st.button("🧹 Clear chat"):
        st.session_state.messages = []
        st.session_state.chat = None
        st.success("Chat history cleared.")


# ----------------------------
# Initialize Gemini Client
# ----------------------------
API_KEY = get_api_key()
if not API_KEY:
    st.stop()

if not safe_configure_genai(API_KEY):
    st.stop()

generation_config = {
    "temperature": temperature,
    "top_p": top_p,
    "top_k": top_k,
    "max_output_tokens": max_output_tokens,
}

current_config_hash = compute_config_hash(model_name, system_instruction or "", generation_config)

# Rebuild the chat session if config changed or not initialized
if (st.session_state.chat is None) or (st.session_state.config_hash != current_config_hash):
    try:
        model = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=system_instruction or None,
        )
        st.session_state.chat = model.start_chat(history=[])
        st.session_state.config_hash = current_config_hash
    except Exception as e:
        st.error(f"Failed to start chat: {e}")
        st.stop()


# ----------------------------
# Render Chat History
# ----------------------------
if len(st.session_state.messages) == 0:
    show_message(
        "assistant",
        "Hi! I’m Gemini running in Streamlit. Ask me anything, or attach an image for multimodal reasoning. 🚀"
    )
else:
    for msg in st.session_state.messages:
        show_message(msg["role"], msg.get("text", ""), msg.get("images", []))


# ----------------------------
# Chat Input + Send
# ----------------------------
user_prompt = st.chat_input("Type your message and press Enter…")

if user_prompt:
    # Prepare parts: text + optional image(s)
    parts: List[Any] = []
    if user_prompt.strip():
        parts.append(user_prompt.strip())

    images_for_ui = []
    if st.session_state.pending_attachments:
        for img_bytes in st.session_state.pending_attachments:
            # Detect image mime
            try:
                pil_img = Image.open(io.BytesIO(img_bytes))
                mime = Image.MIME.get(pil_img.format, "application/octet-stream")
            except Exception:
                mime = "application/octet-stream"
            parts.append({"mime_type": mime, "data": img_bytes})
            images_for_ui.append(img_bytes)

    # Display the user message immediately
    st.session_state.messages.append({"role": "user", "text": user_prompt, "images": images_for_ui})
    show_message("user", user_prompt, images_for_ui)

    # Send to Gemini with streaming
    with st.chat_message("assistant"):
        stream_placeholder = st.empty()
        full_text = ""

        try:
            response = st.session_state.chat.send_message(
                parts,
                generation_config=generation_config,
                stream=True
            )

            for chunk in response:
                if hasattr(chunk, "text") and chunk.text:
                    full_text += chunk.text
                    stream_placeholder.markdown(full_text)

            if not full_text.strip():
                stream_placeholder.info("No content returned (might have been blocked or empty).")

            # Save assistant response
            st.session_state.messages.append({"role": "assistant", "text": full_text, "images": []})

        except Exception as e:
            stream_placeholder.error(f"Error: {e}")

    # Clear pending attachments after use
    st.session_state.pending_attachments = []
