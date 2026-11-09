"""Personal AI Image Studio — run with: streamlit run app.py"""

from __future__ import annotations

import base64
from datetime import datetime, timezone
from io import BytesIO
import json
import os
from pathlib import Path
import zipfile
from uuid import uuid4

from PIL import Image, ImageOps, UnidentifiedImageError
import streamlit as st
from openai import OpenAI, OpenAIError

from prompt_engine import PRESETS, PromptSpec, build_prompt, surprise_fields

MODELS = {
    "GPT Image 2.6 Flare · everyday": "gpt-image-2.6-flare",
    "GPT Image 2.6 Sunburst · detail / precision": "gpt-image-2.6-sunburst",
}
SIZES = {
    "Portrait · 1024 × 1536": "1024x1536",
    "Square · 1024 × 1024": "1024x1024",
    "Landscape · 1536 × 1024": "1536x1024",
}
FORMATS = {"PNG": "png", "WebP": "webp", "JPEG": "jpeg"}
MIMES = {"png": "image/png", "webp": "image/webp", "jpeg": "image/jpeg"}

st.set_page_config(page_title="AI Image Studio", page_icon="▧", layout="wide")
st.markdown(
    """
    <style>
    .block-container {max-width: 1200px; padding-top: 2rem;}
    [data-testid="stImage"] img {border-radius: 11px;}
    [data-testid="stMetric"] {border-radius: 10px;}
    .smallnote {font-size: .85rem; opacity: .73;}
    </style>
    """,
    unsafe_allow_html=True,
)


def optional_secret(name: str) -> str:
    try:
        return str(st.secrets.get(name, ""))
    except (FileNotFoundError, KeyError, AttributeError):
        return ""


def require_access() -> None:
    expected = os.getenv("APP_PASSWORD", "") or optional_secret("APP_PASSWORD")
    if not expected:
        return
    if st.session_state.get("authenticated"):
        return
    st.title("Private Image Studio")
    st.write("Enter your studio password to access this app.")
    entered = st.text_input("Studio password", type="password")
    if st.button("Unlock", type="primary"):
        import hmac
        if hmac.compare_digest(entered, expected):
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    st.stop()


require_access()
if "gallery" not in st.session_state:
    st.session_state.gallery = []
if "chosen_reference" not in st.session_state:
    st.session_state.chosen_reference = "None"
if "surprise" not in st.session_state:
    st.session_state.surprise = {}
if "pending_reference" in st.session_state:
    st.session_state.chosen_reference = st.session_state.pop("pending_reference")


def image_payload(uploaded) -> tuple[str, bytes, str]:
    """Convert user-uploaded files to orientation-corrected PNG, within sensible edit limits."""
    raw = uploaded.getvalue()
    if len(raw) > 30 * 1024 * 1024:
        raise ValueError(f"{uploaded.name}: file is over 30 MB. Please use a smaller image.")
    try:
        with Image.open(BytesIO(raw)) as opened:
            im = ImageOps.exif_transpose(opened)
            if im.mode not in {"RGB", "RGBA"}:
                im = im.convert("RGB")
            im.thumbnail((2400, 2400), Image.Resampling.LANCZOS)
            buf = BytesIO()
            im.save(buf, format="PNG", optimize=True)
            return (Path(uploaded.name).stem + ".png", buf.getvalue(), "image/png")
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError(f"{uploaded.name}: unsupported image. Try PNG, JPG or WebP.") from exc


def gallery_zip(items: list[dict]) -> bytes:
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        manifest = []
        for index, item in enumerate(items, start=1):
            filename = f"image_{index:03d}_{item['id'][:8]}.{item['format']}"
            archive.writestr(filename, item["data"])
            manifest.append({k: v for k, v in item.items() if k not in {"data"}} | {"file": filename})
        archive.writestr("gallery_manifest.json", json.dumps(manifest, indent=2, ensure_ascii=False))
    return buffer.getvalue()


st.title("▧ AI Image Studio")
st.caption("Generate distinctive characters and photographs with your own OpenAI API credits. Your gallery lives in this browser session until you export it.")

with st.sidebar:
    st.header("Connection")
    configured_key = os.getenv("OPENAI_API_KEY", "") or optional_secret("OPENAI_API_KEY")
    if configured_key:
        st.success("API key loaded from server configuration")
        key = configured_key
    else:
        key = st.text_input("OpenAI API key", type="password", placeholder="sk-...", help="Never place your API key in a shared prompt or public source code.")
    st.caption("API usage is billed separately from your ChatGPT subscription.")
    st.divider()
    st.header("Rendering")
    model_name = st.selectbox("Model", list(MODELS), index=0)
    quality = st.select_slider("Quality", options=["low", "medium", "high", "xhigh", "max"], value="medium")
    size_name = st.selectbox("Size", list(SIZES), index=0)
    fmt_name = st.selectbox("Download format", list(FORMATS), index=0)
    batch = st.slider("Images per request", 1, 4, 2)
    st.caption("Higher quality, more images, and references may increase costs. Usage limits depend on your API account.")
    st.divider()
    st.markdown("[API usage and billing](https://platform.openai.com/usage) · [Image model pricing](https://developers.openai.com/api/docs/pricing)")

make_tab, gallery_tab, help_tab = st.tabs(["Create", f"Gallery ({len(st.session_state.gallery)})", "Setup & privacy"])

with make_tab:
    left, right = st.columns([1.5, 1], gap="large")
    with right:
        st.subheader("References")
        uploaded_refs = st.file_uploader(
            "Upload photos to edit or preserve a character",
            type=["png", "jpg", "jpeg", "webp"],
            accept_multiple_files=True,
            help="Up to 4 photos. They are sent to the OpenAI API with your image prompt.",
        )
        if len(uploaded_refs) > 4:
            st.error("Please upload at most 4 reference images.")
        for item in uploaded_refs[:4]:
            st.image(item, caption=item.name, width=230)
        gallery_choices = ["None"] + [f"{item['id'][:8]} · {item['created']}" for item in st.session_state.gallery]
        selected_gallery = st.selectbox(
            "Or reuse a previous result",
            gallery_choices,
            key="chosen_reference",
        )
        st.caption("Uploaded references and a selected gallery image can be combined (maximum 4 total).")
        st.info("To keep a face consistent, upload a clear reference photo and choose **Character continuity**.", icon="ℹ️")

    with left:
        st.subheader("Direction")
        if st.button("Surprise me with a new scene", help="Randomizes wardrobe, setting, and camera angle."):
            st.session_state.surprise = surprise_fields()
        surprise = st.session_state.surprise
        with st.form("image_form"):
            preset = st.selectbox("Creative preset", list(PRESETS), index=0)
            idea = st.text_area(
                "Your main prompt",
                value="Create an original fictional adult with a recognizable personality. Make the result feel like a candid photograph rather than a posed studio image.",
                height=110,
            )
            identity = st.text_input("Identity / character", placeholder="e.g. a 26-year-old eccentric music-store owner")
            c1, c2 = st.columns(2)
            with c1:
                face = st.text_input("Facial features", placeholder="e.g. softly rounded face, expressive eyes")
                build = st.text_input("Body shape", placeholder="e.g. midsize, naturally soft silhouette")
                hair = st.text_input("Hair", placeholder="e.g. messy dark shag cut with bangs")
                outfit = st.text_input("Outfit", value=surprise.get("wardrobe", ""), placeholder="e.g. patterned sundress and old boots")
            with c2:
                scene = st.text_input("Location / scene", value=surprise.get("scene", ""), placeholder="e.g. late-night laundromat")
                pose = st.text_input("Pose / activity", placeholder="e.g. leaning on a chair, half-smiling")
                angle = st.text_input("Camera angle", value=surprise.get("angle", ""), placeholder="e.g. left profile, waist-up")
                vibe = st.text_input("Expression / mood", placeholder="e.g. goofy, relaxed, slightly mischievous")
            invent_details = st.checkbox("Let the model invent incidental details", value=True)
            submitted = st.form_submit_button("Generate images", type="primary", use_container_width=True)

    gallery_source = next((item for item in st.session_state.gallery if selected_gallery.startswith(item["id"][:8])), None) if selected_gallery != "None" else None
    source_count = min(len(uploaded_refs), 4) + int(gallery_source is not None)
    uses_refs = source_count > 0
    prompt = build_prompt(
        PromptSpec(
            idea=idea, preset=preset, identity=identity, face=face, build=build,
            hair=hair, outfit=outfit, scene=scene, pose=pose, angle=angle, vibe=vibe,
            has_references=uses_refs, invent_details=invent_details,
        )
    )
    with st.expander("Preview the exact prompt sent to the API"):
        st.code(prompt, language="text", wrap_lines=True)
        st.download_button("Save prompt as text", prompt.encode("utf-8"), "prompt.txt", "text/plain")

    if submitted:
        if not key or not key.strip():
            st.error("Add an OpenAI API key in the left sidebar first.")
        elif source_count > 4:
            st.error("Please provide no more than 4 reference images in total, including a gallery source.")
        elif not idea.strip() and not identity.strip():
            st.error("Provide a main prompt or an identity.")
        else:
            try:
                references = [image_payload(item) for item in uploaded_refs]
                if gallery_source:
                    # The SDK accepts (filename, bytes, mime-type) tuples for edits.
                    references.append((f"gallery_reference.{gallery_source['format']}", gallery_source["data"], gallery_source["mime"]))
                client = OpenAI(api_key=key.strip(), timeout=180.0, max_retries=1)
                params = dict(
                    model=MODELS[model_name], prompt=prompt, quality=quality,
                    size=SIZES[size_name], n=batch, output_format=FORMATS[fmt_name],
                )
                with st.spinner(f"Generating {batch} image{'s' if batch != 1 else ''} using {model_name}..."):
                    if references:
                        result = client.images.edit(image=references, **params)
                    else:
                        result = client.images.generate(**params)
                records = []
                for output in result.data or []:
                    if not output.b64_json:
                        continue
                    records.append({
                        "id": str(uuid4()),
                        "created": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
                        "data": base64.b64decode(output.b64_json),
                        "format": FORMATS[fmt_name],
                        "mime": MIMES[FORMATS[fmt_name]],
                        "model": MODELS[model_name],
                        "quality": quality,
                        "size": SIZES[size_name],
                        "prompt": prompt,
                        "reference_count": len(references),
                    })
                if not records:
                    st.error("The API returned no image data. Check your API usage dashboard for details.")
                else:
                    st.session_state.gallery = (records + st.session_state.gallery)[:48]
                    st.success(f"Added {len(records)} image(s) to your session gallery below.")
                    cols = st.columns(min(3, len(records)))
                    for idx, item in enumerate(records):
                        with cols[idx % len(cols)]:
                            st.image(item["data"], use_container_width=True)
                            st.download_button(
                                "Download image", item["data"],
                                f"image_{item['id'][:8]}.{item['format']}", item["mime"],
                                key=f"new_dl_{item['id']}", use_container_width=True,
                            )
            except OpenAIError as exc:
                # Never print API key or the raw request data.
                st.error(f"The OpenAI API returned an error: {str(exc)[:800]}")
                st.caption("Common causes: no API credits, missing model access, account rate limit, or request restrictions.")
            except ValueError as exc:
                st.error(str(exc))
            except Exception as exc:
                st.error(f"Unable to process this request ({type(exc).__name__}). Check your image format and connection.")

with gallery_tab:
    st.subheader("Your session gallery")
    if not st.session_state.gallery:
        st.write("Your generated images will appear here after your first request.")
    else:
        st.caption("Images remain on the running app server for this session only. Export the ZIP to keep them.")
        st.download_button(
            "Download all images + prompts (.zip)", gallery_zip(st.session_state.gallery),
            "ai_image_studio_gallery.zip", "application/zip", type="primary",
        )
        cols = st.columns(3)
        for i, item in enumerate(st.session_state.gallery):
            with cols[i % 3]:
                st.image(item["data"], use_container_width=True)
                st.caption(f"{item['created']} · {item['model']} · {item['quality']}")
                st.download_button(
                    "Save photo", item["data"],
                    f"studio_{item['id'][:8]}.{item['format']}", item["mime"],
                    key=f"gallery_dl_{item['id']}", use_container_width=True,
                )
                if st.button("Use as reference", key=f"reuse_{item['id']}", use_container_width=True):
                    st.session_state.pending_reference = f"{item['id'][:8]} · {item['created']}"
                    st.rerun()
        if st.button("Clear session gallery"):
            st.session_state.gallery = []
            st.session_state.pending_reference = "None"
            st.rerun()

with help_tab:
    st.markdown(
        """
        #### Getting started
        1. Create an API key at [OpenAI API keys](https://platform.openai.com/api-keys), and check that your API account has billing enabled.
        2. Enter the API key in the sidebar, or configure `OPENAI_API_KEY` on your own server.
        3. Choose a creative preset and write a prompt. Upload a reference image if you want to maintain a particular identity.
        4. Generate. Download individual images or export your session gallery as a ZIP.

        #### Character continuity
        Provide a front-facing photo plus optional different angles. Use **Character continuity**. Character consistency is not guaranteed, even with references; compare results and refine the prompt.

        #### Privacy and safety
        - The API key stays on the Python backend; this app does not embed it into client-side JavaScript.
        - Reference photos and prompts are sent to the OpenAI API to make the requested image.
        - The gallery is stored in memory while the server session is active. It is **not** a permanent cloud library.
        - If you deploy this app on the internet, set a strong `APP_PASSWORD` in your deployment secrets to prevent unauthorized use and charges. For shared deployments, use proper authentication rather than a single password.
        - Set a spending limit in your [OpenAI project billing settings](https://platform.openai.com/settings/organization/limits).
        - Image requests are subject to API pricing, rate limits, and content restrictions. A ChatGPT subscription is billed separately.

        #### Troubleshooting
        **401** = invalid key. **429** = rate limit or insufficient quota. **403** = model/account access problem. If you upload HEIC photos from an iPhone, export them as JPEG or PNG before uploading.
        """
    )
