"""
app.py — EuroSAT Land Cover Classifier Demo
=============================================
Gradio demo app — connects to the model saved by eurosat_sol.py (your train file).

HOW TO RUN IN GOOGLE COLAB (add as a new cell AFTER your training cells):
─────────────────────────────────────────────────────────────────────────
    # Step 1 — install gradio (once per session)
    !pip install -q gradio

    # Step 2 — write this file to disk
    %%writefile /content/app.py
    <paste entire contents of this file here>

    # Step 3 — run it
    !python /content/app.py

Then click the public link that appears (e.g. https://xxxx.gradio.live)
"""

# ── Imports ────────────────────────────────────────────────────────────────────
import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models
import gradio as gr

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  CONFIG  — must match your eurosat_sol.py settings exactly              ║
# ╚══════════════════════════════════════════════════════════════════════════╝

MODEL_PATH  = "/content/best_model.pth"   # saved by CHECKPOINT in your train file
NUM_CLASSES = 6                            # you set NUM_CLASSES = 6
IMG_SIZE    = 224
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Order MUST match the label integers (0,1,2,3,4,5) in your CSVs
CLASS_NAMES = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
]

CLASS_EMOJI = {
    "AnnualCrop"           : "🌾",
    "Forest"               : "🌲",
    "HerbaceousVegetation" : "🌿",
    "Highway"              : "🛣️",
    "Industrial"           : "🏭",
    "Pasture"              : "🐄",
}

CLASS_DESCRIPTIONS = {
    "AnnualCrop"           : "Fields of crops harvested and replanted every year (e.g. wheat, corn, rice).",
    "Forest"               : "Dense tree cover — both deciduous and evergreen woodland.",
    "HerbaceousVegetation" : "Low-growing non-woody plants: grasslands, meadows, and scrubland.",
    "Highway"              : "Major road infrastructure clearly visible from satellite altitude.",
    "Industrial"           : "Factories, warehouses, power plants, and large commercial zones.",
    "Pasture"              : "Open grazing land maintained for livestock farming.",
}

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Load Model  (same architecture as build_model() in your train file)   ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def load_model():
    m = models.efficientnet_b0(weights=None)
    in_features = m.classifier[1].in_features
    m.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, NUM_CLASSES),
    )
    if os.path.exists(MODEL_PATH):
        m.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print(f"✅ Loaded weights from {MODEL_PATH}")
    else:
        print(f"⚠️  {MODEL_PATH} not found — run your training cells first!")
    m.eval()
    return m.to(DEVICE)


model = load_model()

infer_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Prediction                                                             ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def predict(image):
    if image is None:
        return "Please upload an image.", {}, ""

    tensor = infer_tf(image.convert("RGB")).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1).squeeze().cpu().numpy()

    top_idx   = int(np.argmax(probs))
    top_class = CLASS_NAMES[top_idx]
    top_conf  = float(probs[top_idx])

    sorted_idx = np.argsort(probs)[::-1]
    conf_dict  = {
        f"{CLASS_EMOJI[CLASS_NAMES[i]]}  {CLASS_NAMES[i]}": float(probs[i])
        for i in sorted_idx
    }

    label_str = f"{CLASS_EMOJI[top_class]}  {top_class}   ({top_conf*100:.1f}% confident)"
    desc_str  = CLASS_DESCRIPTIONS[top_class]

    return label_str, conf_dict, desc_str


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Gradio UI                                                              ║
# ╚══════════════════════════════════════════════════════════════════════════╝

CSS = """
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;700&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

body, .gradio-container {
    background: #f5f0e8 !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    color: #1a1a2e !important;
}
#app-header {
    background: #1a1a2e;
    color: #f5f0e8;
    padding: 2rem 2rem 1.5rem;
    border-radius: 12px;
    margin-bottom: 1.5rem;
}
#app-header h1 {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.7rem;
    margin: 0 0 0.4rem;
    color: #e8c547;
    letter-spacing: -0.3px;
}
#app-header p {
    margin: 0;
    font-size: 0.9rem;
    color: #a0a8c0;
    font-weight: 300;
}
.gr-panel, .gr-box, .gr-form {
    background: #ffffff !important;
    border: 1px solid #ddd8cc !important;
    border-radius: 10px !important;
}
#classify-btn {
    background: #1a1a2e !important;
    color: #e8c547 !important;
    border: none !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.9rem !important;
    border-radius: 8px !important;
    letter-spacing: 0.5px !important;
    transition: opacity 0.2s !important;
}
#classify-btn:hover { opacity: 0.85 !important; }
#clear-btn {
    background: #f5f0e8 !important;
    border: 1px solid #ddd8cc !important;
    color: #666 !important;
    border-radius: 8px !important;
}
#pred-label textarea {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 1.05rem !important;
    font-weight: 700 !important;
    color: #1a1a2e !important;
    background: #f5f0e8 !important;
    border: 1px solid #ddd8cc !important;
}
#desc-box textarea {
    font-size: 0.88rem !important;
    color: #555 !important;
    font-style: italic;
    background: #f5f0e8 !important;
    border: 1px solid #ddd8cc !important;
}
.label-confidence-bar-fill {
    background: linear-gradient(90deg, #1a1a2e, #4a4a8e) !important;
}
#footer {
    text-align: center;
    color: #999;
    font-size: 0.78rem;
    font-family: 'IBM Plex Mono', monospace;
    padding: 1rem 0 0.3rem;
}
"""

with gr.Blocks(css=CSS, title="EuroSAT Classifier") as demo:

    gr.HTML("""
    <div id="app-header">
        <h1>🛰️  EuroSAT Land Cover Classifier</h1>
        <p>EfficientNet-B0 trained on Sentinel-2 satellite imagery &nbsp;·&nbsp; 6 land-cover classes</p>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            img_input = gr.Image(type="pil", label="Upload Satellite Image", height=300)
            with gr.Row():
                btn_classify = gr.Button("🔍  Classify Image", elem_id="classify-btn", variant="primary")
                btn_clear    = gr.ClearButton([img_input], value="✕ Clear", elem_id="clear-btn")
            gr.Markdown(
                "**Classes this model recognises:**\n\n" +
                "  ".join([f"{v} {k}" for k, v in CLASS_EMOJI.items()])
            )

        with gr.Column(scale=1):
            out_label = gr.Textbox(label="Prediction", interactive=False, elem_id="pred-label")
            out_conf  = gr.Label(label="Confidence Scores", num_top_classes=6)
            out_desc  = gr.Textbox(label="About this class", interactive=False,
                                   lines=2, elem_id="desc-box")

    gr.HTML('<div id="footer">EfficientNet-B0 · EuroSAT · Sentinel-2 · Trained in Google Colab</div>')

    btn_classify.click(fn=predict, inputs=img_input,
                       outputs=[out_label, out_conf, out_desc])
    img_input.change(fn=predict, inputs=img_input,
                     outputs=[out_label, out_conf, out_desc])


if __name__ == "__main__":
    demo.launch(
        share=True,       # generates public link — required for Colab
        debug=False,
        show_error=True,
    )
