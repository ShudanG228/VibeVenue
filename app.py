import hashlib
import gradio as gr
import numpy as np
from pathlib import Path
from PIL import Image

import sys
sys.path.insert(0, "src")

_search_fn = None

def _get_search():
    global _search_fn
    if _search_fn is None:
        from inference import search
        _search_fn = search
    return _search_fn


_cache: dict[str, tuple] = {}

def _image_hash(img: Image.Image) -> str:
    arr = np.array(img.resize((64, 64))).tobytes()
    return hashlib.md5(arr).hexdigest()


def search_with_cache(img: Image.Image, top_k: int = 8, use_llm: bool = True):
    key = _image_hash(img) + f"_{top_k}_{use_llm}"
    if key in _cache:
        return _cache[key]
    search = _get_search()
    result = search(img, top_k=top_k, use_llm=use_llm)
    _cache[key] = result
    return result


def format_results_html(results, vibe_desc: str, inference_ms: float) -> str:
    if not results:
        return "<p style='color:#888;text-align:center;padding:40px'>No results found. Try a different image.</p>"

    cards_html = ""
    for i, r in enumerate(results):
        stars = "⭐" * round(r.rating) if r.rating else ""
        photo_src = ""
        if Path(r.photo_path).exists():
            import base64
            with open(r.photo_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            photo_src = f"data:image/jpeg;base64,{b64}"

        photo_html = (
            f'<img src="{photo_src}" style="width:100%;height:160px;object-fit:cover;border-radius:8px 8px 0 0;">'
            if photo_src else
            '<div style="width:100%;height:160px;background:#2a2a2a;border-radius:8px 8px 0 0;display:flex;align-items:center;justify-content:center;color:#555;font-size:2em;">📷</div>'
        )

        rank_badge = f'<div style="position:absolute;top:8px;left:8px;background:#E84855;color:white;border-radius:50%;width:28px;height:28px;display:flex;align-items:center;justify-content:center;font-weight:bold;font-size:12px;">{i+1}</div>'

        score_pct = int(r.ensemble_score * 1000)

        cards_html += f"""
        <div style="background:#1a1a1a;border-radius:10px;overflow:hidden;border:1px solid #2a2a2a;position:relative;transition:transform 0.2s;">
            <div style="position:relative;">
                {photo_html}
                {rank_badge}
                <div style="position:absolute;bottom:8px;right:8px;background:rgba(0,0,0,0.75);color:#FFD700;padding:2px 8px;border-radius:12px;font-size:11px;">{stars} {f"{r.rating:.1f}" if r.rating else "N/A"}</div>
            </div>
            <div style="padding:12px;">
                <div style="font-weight:600;font-size:14px;color:#f0f0f0;margin-bottom:4px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{r.name}</div>
                <div style="color:#888;font-size:12px;margin-bottom:6px;">📍 {r.city}, {r.country}</div>
                <div style="color:#666;font-size:11px;margin-bottom:8px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{r.address}</div>
                <div style="background:#0d2137;border-radius:6px;padding:4px 8px;display:flex;justify-content:space-between;align-items:center;">
                    <span style="color:#4ECDC4;font-size:11px;font-weight:600;">Match score</span>
                    <span style="color:#4ECDC4;font-size:12px;font-weight:700;">{score_pct}</span>
                </div>
            </div>
        </div>
        """

    vibe_box = f"""
    <div style="background:linear-gradient(135deg,#0d2137,#1a1a2e);border:1px solid #2E86AB;border-radius:10px;padding:16px;margin-bottom:20px;">
        <div style="color:#4ECDC4;font-size:12px;font-weight:600;text-transform:uppercase;letter-spacing:1px;margin-bottom:6px;">✨ Detected Vibe</div>
        <div style="color:#e8e8e8;font-size:14px;line-height:1.5;font-style:italic;">"{vibe_desc}"</div>
        <div style="color:#555;font-size:11px;margin-top:8px;">⏱ Inference: {inference_ms:.0f}ms</div>
    </div>
    """ if vibe_desc else ""

    return f"""
    {vibe_box}
    <div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:16px;">
        {cards_html}
    </div>
    """


def run_search(image, top_k_slider, use_llm_checkbox):
    if image is None:
        return "<p style='color:#888;text-align:center;padding:40px'>Please upload a scenery photo to begin.</p>"

    try:
        pil_img = Image.fromarray(image) if not isinstance(image, Image.Image) else image
        results, vibe_desc, inference_ms = search_with_cache(pil_img, top_k=int(top_k_slider), use_llm=use_llm_checkbox)
        return format_results_html(results, vibe_desc, inference_ms)
    except FileNotFoundError:
        return """
        <div style='text-align:center;padding:40px;color:#888;'>
            <div style='font-size:3em;margin-bottom:12px'>⚠️</div>
            <div>Index not built yet. Please run:<br>
            <code style='background:#1a1a1a;padding:4px 8px;border-radius:4px;'>python src/build_index.py</code></div>
        </div>
        """
    except Exception as e:
        return f"<p style='color:#E84855;padding:20px'>Error: {str(e)}</p>"


CUSTOM_CSS = """
* { box-sizing: border-box; }
body, .gradio-container {
    background: #0d0d0d !important;
    font-family: 'DM Sans', 'Helvetica Neue', sans-serif !important;
}
.gradio-container { max-width: 1200px !important; margin: 0 auto !important; }

#header {
    text-align: center;
    padding: 40px 20px 20px;
    background: linear-gradient(180deg, #0d0d0d 0%, #111 100%);
}
#header h1 {
    font-size: 2.8em;
    font-weight: 800;
    background: linear-gradient(135deg, #4ECDC4, #2E86AB, #E84855);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 8px;
    letter-spacing: -1px;
}
#header p {
    color: #888;
    font-size: 1.05em;
    margin: 0;
}

.upload-box { border: 2px dashed #2E86AB !important; border-radius: 12px !important; background: #111 !important; }
.upload-box:hover { border-color: #4ECDC4 !important; }

button.primary { background: linear-gradient(135deg, #2E86AB, #4ECDC4) !important; border: none !important; color: white !important; font-weight: 600 !important; border-radius: 8px !important; }
button.primary:hover { opacity: 0.9 !important; }

.panel { background: #111 !important; border: 1px solid #1e1e1e !important; border-radius: 12px !important; }

label { color: #aaa !important; font-size: 13px !important; }
input[type=range] { accent-color: #4ECDC4; }
"""

def build_demo():
    with gr.Blocks(css=CUSTOM_CSS, title="VibeVenue") as demo:

        gr.HTML("""
        <div id="header">
            <h1>🌏 VibeVenue</h1>
            <p>Upload a scenery photo — discover Asian restaurants &amp; cafes with the same vibe</p>
        </div>
        """)

        with gr.Row():
            with gr.Column(scale=1, elem_classes="panel"):
                gr.HTML("<div style='padding:8px 0 4px;color:#aaa;font-size:13px;font-weight:600;'>📸 Your Scenery Photo</div>")
                image_input = gr.Image(
                    label="",
                    type="pil",
                    elem_classes="upload-box",
                    height=280,
                )
                with gr.Row():
                    top_k_slider = gr.Slider(
                        minimum=4, maximum=20, value=8, step=1,
                        label="Number of results"
                    )
                with gr.Row():
                    use_llm = gr.Checkbox(
                        value=True,
                        label="Use AI vibe description (requires trained model)"
                    )
                search_btn = gr.Button("🔍 Find My Vibe", variant="primary", size="lg")

                gr.HTML("""
                <div style='margin-top:16px;padding:12px;background:#111;border-radius:8px;border:1px solid #1e1e1e;'>
                    <div style='color:#555;font-size:11px;line-height:1.6;'>
                    <b style='color:#666'>Try uploarding:</b><br>
                    🌿 A bamboo forest path<br>
                    🌊 Ocean / beach scenery<br>
                    🏙 City skyline at night<br>
                    🌸 Cherry blossom park<br>
                    🏔 Mountain mist valley
                    </div>
                </div>
                """)

            with gr.Column(scale=2, elem_classes="panel"):
                gr.HTML("<div style='padding:8px 0 4px;color:#aaa;font-size:13px;font-weight:600;'>🍜 Matching Restaurants &amp; Cafes</div>")
                results_html = gr.HTML(
                    value="<p style='color:#444;text-align:center;padding:60px;'>Upload a photo and click search to discover restaurants</p>"
                )

        search_btn.click(
            fn=run_search,
            inputs=[image_input, top_k_slider, use_llm],
            outputs=results_html,
        )
        image_input.change(
            fn=run_search,
            inputs=[image_input, top_k_slider, use_llm],
            outputs=results_html,
        )

    return demo


if __name__ == "__main__":
    demo = build_demo()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )
