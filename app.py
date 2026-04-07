import streamlit as st
import numpy as np
import json
from PIL import Image
import os
import zipfile
import shutil

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from losses import FocalLoss  # required for deserialising the saved model
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(BASE_DIR, 'soil_classifier_final.keras')
CLASS_JSON  = os.path.join(BASE_DIR, 'class_indices.json')
TRAIN_PLOT  = os.path.join(BASE_DIR, 'training_plots.png')
DATASET_DIR = os.path.join(BASE_DIR, 'soil_dataset')

# ── Design tokens ─────────────────────────────────────────────────────────────
PRIMARY   = '#7C4A1E'   # rich brown  — buttons, accents
ACCENT    = '#C47C2B'   # warm amber  — highlights
SUCCESS   = '#3A7D44'   # forest green
DANGER    = '#B03A2E'   # terracotta red
BG        = '#F9F4EE'   # warm cream page bg
CARD      = '#FFFFFF'   # white cards
BORDER    = '#DDD0C0'   # warm grey border
TEXT      = '#1C1008'   # near-black body text
MUTED     = '#6B5744'   # secondary text
HEADING   = '#3D1F08'   # dark brown headings

CLASS_COLORS = {
    'alluvial': '#3A7D44',
    'arid':     '#C47C2B',
    'black':    '#4A3728',
    'red':      '#B03A2E',
    'yellow':   '#C9A227',
}

# ── Crop recommendations per soil type ───────────────────────────────────────
CROP_DATA = {
    'alluvial': {
        'description': 'Alluvial soil is highly fertile, well-drained, and rich in minerals deposited by rivers. It has good water retention and is ideal for a wide range of crops.',
        'best_for': ['Rice', 'Wheat', 'Sugarcane', 'Cotton', 'Maize', 'Pulses', 'Oilseeds'],
        'also_suitable': ['Vegetables', 'Fruits (Mango, Banana)', 'Jute', 'Tobacco'],
        'avoid': ['Crops requiring very dry conditions'],
        'tips': 'Maintain organic matter levels. Avoid waterlogging in low-lying areas. Excellent for double-cropping systems.',
        'icon': '🌾',
    },
    'arid': {
        'description': 'Arid soil is sandy, low in organic matter, and has poor water retention. It is found in dry and desert regions with low rainfall.',
        'best_for': ['Millet (Bajra)', 'Sorghum', 'Barley', 'Drought-resistant Pulses', 'Groundnut'],
        'also_suitable': ['Date Palm', 'Cactus crops', 'Drought-tolerant vegetables (Onion, Garlic)'],
        'avoid': ['Water-intensive crops like Rice, Sugarcane'],
        'tips': 'Use drip irrigation to conserve water. Add organic compost to improve fertility. Mulching helps retain soil moisture.',
        'icon': '🌵',
    },
    'black': {
        'description': 'Black soil (Regur) is rich in calcium, magnesium, and iron. It has high water retention capacity and swells when wet, making it ideal for rain-fed crops.',
        'best_for': ['Cotton', 'Soybean', 'Sorghum', 'Wheat', 'Sunflower', 'Chickpea'],
        'also_suitable': ['Citrus fruits', 'Tobacco', 'Linseed', 'Millets'],
        'avoid': ['Crops sensitive to waterlogging during monsoon'],
        'tips': 'Deep ploughing improves aeration. Avoid over-irrigation — the soil retains moisture well. Ideal for cotton cultivation.',
        'icon': '🌻',
    },
    'red': {
        'description': 'Red soil is rich in iron oxide, giving it a reddish color. It is generally low in nitrogen and organic matter but responds well to fertilization.',
        'best_for': ['Groundnut', 'Pulses', 'Millets', 'Tobacco', 'Potato', 'Maize'],
        'also_suitable': ['Fruits (Mango, Citrus)', 'Vegetables', 'Oilseeds', 'Cotton'],
        'avoid': ['Crops needing high nitrogen without supplementation'],
        'tips': 'Apply nitrogen-rich fertilizers or compost. Lime application helps correct acidity. Good drainage makes it suitable for root crops.',
        'icon': '🥜',
    },
    'yellow': {
        'description': 'Yellow soil contains iron in hydrated form, giving it a yellow tint. It is moderately fertile and found in areas with moderate rainfall.',
        'best_for': ['Rice', 'Maize', 'Groundnut', 'Pulses', 'Sweet Potato'],
        'also_suitable': ['Vegetables', 'Fruits', 'Millets', 'Oilseeds'],
        'avoid': ['Crops requiring very high fertility without amendment'],
        'tips': 'Enrich with organic matter and phosphorus fertilizers. Good for mixed farming. Responds well to green manuring.',
        'icon': '🌽',
    },
}

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SoilClass ML",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
  /* page */
  .stApp {{ background: {BG}; }}

  /* all default text */
  body, p, span, div, li, td, th {{ color: {TEXT} !important; }}

  /* headings */
  h1 {{ color: {PRIMARY} !important; font-weight: 800 !important; letter-spacing: -0.5px; }}
  h2, h3 {{ color: {HEADING} !important; font-weight: 700 !important; }}

  /* sidebar */
  [data-testid="stSidebar"] > div:first-child {{
      background: {PRIMARY};
  }}
  [data-testid="stSidebar"] p,
  [data-testid="stSidebar"] span,
  [data-testid="stSidebar"] label,
  [data-testid="stSidebar"] div {{
      color: #F9F0E6 !important;
  }}
  [data-testid="stSidebar"] hr {{ border-color: rgba(255,255,255,0.2) !important; }}
  [data-testid="stSidebar"] code {{
      background: rgba(0,0,0,0.25) !important;
      color: #FFD9A0 !important;
      border-radius: 4px;
      padding: 1px 6px;
  }}

  /* metric cards */
  [data-testid="stMetric"] {{
      background: {CARD};
      border: 1px solid {BORDER};
      border-radius: 12px;
      padding: 18px 22px;
      box-shadow: 0 1px 6px rgba(0,0,0,0.07);
  }}
  [data-testid="stMetricLabel"] p {{
      color: {MUTED} !important;
      font-size: 12px !important;
      font-weight: 600 !important;
      text-transform: uppercase;
      letter-spacing: 0.06em;
  }}
  [data-testid="stMetricValue"] {{
      color: {PRIMARY} !important;
      font-size: 30px !important;
      font-weight: 800 !important;
  }}

  /* primary button */
  .stButton > button[kind="primary"] {{
      background: {PRIMARY} !important;
      color: #FFFFFF !important;
      border: none !important;
      border-radius: 8px !important;
      font-weight: 700 !important;
      padding: 10px 24px !important;
  }}
  .stButton > button[kind="primary"]:hover {{
      background: {ACCENT} !important;
      color: #FFFFFF !important;
  }}

  /* secondary button */
  .stButton > button {{
      border-radius: 8px !important;
      font-weight: 600 !important;
  }}

  /* file uploader */
  [data-testid="stFileUploader"] {{
      background: {CARD};
      border: 2px dashed {BORDER};
      border-radius: 12px;
      padding: 10px;
  }}
  [data-testid="stFileUploader"] * {{ color: {TEXT} !important; }}

  /* dataframe */
  [data-testid="stDataFrame"] {{ border-radius: 10px; overflow: hidden; }}
  [data-testid="stDataFrame"] * {{ color: {TEXT} !important; }}

  /* divider */
  hr {{ border-color: {BORDER} !important; margin: 1.5rem 0; }}

  /* alerts */
  [data-testid="stAlert"] p {{ color: {TEXT} !important; }}

  /* progress bar */
  [data-testid="stProgressBar"] > div {{
      background: {BORDER};
      border-radius: 99px;
  }}
  [data-testid="stProgressBar"] > div > div {{
      background: linear-gradient(90deg, {PRIMARY}, {ACCENT});
      border-radius: 99px;
  }}
  /* progress label */
  [data-testid="stProgressBar"] ~ div p {{ color: {TEXT} !important; }}

  /* code / pre */
  code {{
      background: #EDE3D8 !important;
      color: {PRIMARY} !important;
      border-radius: 4px;
      padding: 1px 5px;
      font-size: 0.88em;
  }}
  pre {{
      background: #EDE3D8 !important;
      border: 1px solid {BORDER};
      border-radius: 10px;
  }}
  pre code {{ background: transparent !important; color: {HEADING} !important; }}

  /* markdown tables */
  table {{ border-collapse: collapse; width: 100%; }}
  thead th {{
      background: {PRIMARY} !important;
      color: #FFFFFF !important;
      padding: 9px 14px;
      font-weight: 700;
      font-size: 13px;
  }}
  tbody td {{ color: {TEXT} !important; padding: 8px 14px; font-size: 13px; }}
  tbody tr:nth-child(even) {{ background: #F3EBE0 !important; }}
  tbody tr:nth-child(odd)  {{ background: {CARD} !important; }}

  /* caption */
  [data-testid="stCaptionContainer"] p {{ color: {MUTED} !important; }}

  /* spinner */
  [data-testid="stSpinner"] p {{ color: {TEXT} !important; }}

  /* radio labels in sidebar */
  [data-testid="stRadio"] label p {{ font-weight: 600; font-size: 14px; }}

  /* info box */
  .stAlert {{ border-radius: 10px !important; }}
</style>
""", unsafe_allow_html=True)

# ── Cached loaders ────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model…")
def load_model():
    path = MODEL_PATH
    if not zipfile.is_zipfile(path):
        h5 = path.replace('.keras', '.h5')
        if not os.path.exists(h5):
            shutil.copy2(path, h5)
        path = h5
    return tf.keras.models.load_model(path)

@st.cache_data
def load_labels() -> dict:
    with open(CLASS_JSON) as f:
        idx = json.load(f)
    return {v: k for k, v in idx.items()}

@st.cache_data
def dataset_counts() -> dict:
    counts = {}
    for split in ['train', 'validation', 'test']:
        d = os.path.join(DATASET_DIR, split)
        if not os.path.exists(d):
            continue
        for cls in sorted(os.listdir(d)):
            cd = os.path.join(d, cls)
            if os.path.isdir(cd):
                counts.setdefault(cls, {})[split] = len(os.listdir(cd))
    return counts

def chart_style(fig, ax):
    fig.patch.set_facecolor(CARD)
    ax.set_facecolor('#FBF6F0')
    ax.tick_params(colors=TEXT, labelsize=9)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    for sp in ax.spines.values():
        sp.set_color(BORDER)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"<h2 style='color:#FFD9A0;margin:0 0 2px 0'>🌱 SoilClass ML</h2>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:#E8C99A;font-size:13px;margin:0'>MobileNetV2 · 5 soil types</p>", unsafe_allow_html=True)
    st.divider()
    page = st.radio("", [
        "🏠  Overview",
        "🔍  Classify Image",
        "📊  Evaluation",
        "📁  Dataset",
        "⚙️  Hyperparameters",
    ])
    st.divider()
    st.markdown("<p style='color:#E8C99A;font-size:11px;text-transform:uppercase;letter-spacing:.08em;margin-bottom:2px'>Model</p>", unsafe_allow_html=True)
    st.code("soil_classifier_final.keras", language=None)
    st.markdown("<p style='color:#E8C99A;font-size:11px;text-transform:uppercase;letter-spacing:.08em;margin:8px 0 4px'>Classes</p>", unsafe_allow_html=True)
    for cls, col in CLASS_COLORS.items():
        st.markdown(f"<span style='display:inline-block;width:10px;height:10px;border-radius:50%;background:{col};margin-right:6px'></span><span style='color:#F9F0E6;font-size:13px'>{cls.capitalize()}</span>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠  Overview":
    st.title("🌱 Soil Classification System")
    st.markdown(f"<p style='color:{MUTED};font-size:15px'>Production-quality image classification using <strong style='color:{PRIMARY}'>MobileNetV2</strong> transfer learning.</p>", unsafe_allow_html=True)
    st.divider()

    counts = dataset_counts()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Soil Classes",      "5")
    c2.metric("Training Images",   f"{sum(v.get('train',0) for v in counts.values()):,}")
    c3.metric("Validation Images", f"{sum(v.get('validation',0) for v in counts.values()):,}")
    c4.metric("Test Images",       f"{sum(v.get('test',0) for v in counts.values()):,}")

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Model Architecture")
        st.markdown("""
| Layer | Details |
|---|---|
| Base | MobileNetV2 (ImageNet) |
| Pooling | GlobalAveragePooling2D |
| Dense | 128 units · ReLU |
| Dropout | 0.5 |
| Output | 5 units · Softmax |
""")
        st.subheader("Training Strategy")
        st.markdown(f"""
- **Phase 1** — Frozen base, train head `Adam lr=0.001`
- **Phase 2** — Unfreeze top 20 layers `Adam lr=0.0001`
- **Callbacks** — EarlyStopping (patience=5) + ModelCheckpoint
""")

    with col2:
        st.subheader("Pipeline")
        st.code("""\
split_data.py          → train / val / test split
preprocess_dataset.py  → blur + histogram equalization
train.py               → two-phase transfer learning
evaluate.py            → metrics + confusion matrix
predict.py             → CLI single-image inference
app.py                 → this Streamlit UI""", language="text")
        st.subheader("Input Preprocessing")
        st.markdown("""
- Resize to **224 × 224 px**
- Normalize pixels to **[0, 1]**
- Augmentation *(train only)*: flip · rotation ±20° · zoom ±20° · translate ±10%
""")

    if os.path.exists(TRAIN_PLOT):
        st.divider()
        st.subheader("Training History")
        st.image(TRAIN_PLOT, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# CLASSIFY IMAGE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍  Classify Image":
    st.title("🔍 Classify a Soil Image")
    st.markdown(f"<p style='color:{MUTED}'>Upload a photo and the model will predict the soil type.</p>", unsafe_allow_html=True)

    uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    # clear result when a new image is uploaded
    if uploaded and st.session_state.get('last_upload') != uploaded.name:
        st.session_state.pop('soil_result', None)
        st.session_state['last_upload'] = uploaded.name

    if uploaded:
        img = Image.open(uploaded)
        col1, col2 = st.columns([1, 1], gap="large")

        with col1:
            st.image(img, caption="Uploaded image", use_container_width=True)

        with col2:
            if st.button("▶  Run Classification", type="primary", use_container_width=True):
                with st.spinner("Running inference…"):
                    try:
                        model  = load_model()
                        labels = load_labels()
                        arr    = np.array(img.convert('RGB').resize((224, 224)), dtype=np.float32) / 255.0
                        probs  = model.predict(np.expand_dims(arr, 0), verbose=0)[0]
                        idx    = int(np.argmax(probs))
                        st.session_state['soil_result'] = {
                            'pred_class': labels[idx],
                            'confidence': float(probs[idx]),
                            'all_probs':  {labels[i]: float(p) for i, p in enumerate(probs)}
                        }
                    except Exception as e:
                        st.error(f"Inference failed: {e}")
                        st.stop()

            result = st.session_state.get('soil_result')
            if result:
                pred_class = result['pred_class']
                confidence = result['confidence']
                all_probs  = result['all_probs']
                conf_pct   = confidence * 100
                cls_color  = CLASS_COLORS.get(pred_class, PRIMARY)
                bar_color  = SUCCESS if conf_pct >= 70 else ACCENT if conf_pct >= 40 else DANGER

                st.markdown(f"""
<div style="background:{CARD};border:1.5px solid {BORDER};border-radius:14px;
            padding:22px 24px;margin-bottom:18px;box-shadow:0 2px 10px rgba(0,0,0,0.08)">
  <p style="color:{MUTED};font-size:11px;margin:0;text-transform:uppercase;
            letter-spacing:.1em;font-weight:700">Detected Soil Type</p>
  <p style="color:{cls_color};font-size:34px;font-weight:800;margin:6px 0 14px">{pred_class.capitalize()}</p>
  <p style="color:{MUTED};font-size:11px;margin:0;text-transform:uppercase;
            letter-spacing:.1em;font-weight:700">Confidence</p>
  <p style="color:{bar_color};font-size:30px;font-weight:700;margin:4px 0 10px">{conf_pct:.1f}%</p>
  <div style="background:{BORDER};border-radius:99px;height:10px">
    <div style="background:linear-gradient(90deg,{bar_color},{ACCENT});
                width:{conf_pct}%;height:10px;border-radius:99px"></div>
  </div>
</div>
""", unsafe_allow_html=True)

                st.subheader("Probability breakdown")
                for cls, prob in sorted(all_probs.items(), key=lambda x: x[1], reverse=True):
                    is_top = cls == pred_class
                    lc, bc = st.columns([3, 7])
                    lc.markdown(
                        f"<span style='color:{PRIMARY if is_top else MUTED};"
                        f"font-weight:{'800' if is_top else '500'};font-size:14px'>"
                        f"{cls.capitalize()}</span>", unsafe_allow_html=True)
                    bc.progress(prob, text=f"{prob*100:.1f}%")

        # ── Crop recommendations (full width, below both columns) ─────────────
        result = st.session_state.get('soil_result')
        if result:
            pred_class = result['pred_class']
            crop       = CROP_DATA.get(pred_class, {})
            cls_color  = CLASS_COLORS.get(pred_class, PRIMARY)

            st.divider()
            st.markdown(
                f"<h3 style='color:{cls_color}'>{crop.get('icon','')} "
                f"Crop Recommendations for {pred_class.capitalize()} Soil</h3>",
                unsafe_allow_html=True)
            st.markdown(
                f"<p style='color:{MUTED};font-size:14px'>{crop.get('description','')}</p>",
                unsafe_allow_html=True)

            ca, cb, cc = st.columns(3, gap="medium")
            with ca:
                st.markdown(
                    f"<div style='background:{CARD};border:1px solid {BORDER};border-radius:12px;padding:16px'>"
                    f"<p style='color:{SUCCESS};font-weight:700;font-size:13px;text-transform:uppercase;"
                    f"letter-spacing:.06em;margin-bottom:8px'>✅ Best Crops</p>"
                    + "".join(f"<p style='color:{TEXT};margin:4px 0;font-size:14px'>• {c}</p>"
                              for c in crop.get('best_for', []))
                    + "</div>", unsafe_allow_html=True)
            with cb:
                st.markdown(
                    f"<div style='background:{CARD};border:1px solid {BORDER};border-radius:12px;padding:16px'>"
                    f"<p style='color:{ACCENT};font-weight:700;font-size:13px;text-transform:uppercase;"
                    f"letter-spacing:.06em;margin-bottom:8px'>🌿 Also Suitable</p>"
                    + "".join(f"<p style='color:{TEXT};margin:4px 0;font-size:14px'>• {c}</p>"
                              for c in crop.get('also_suitable', []))
                    + "</div>", unsafe_allow_html=True)
            with cc:
                st.markdown(
                    f"<div style='background:{CARD};border:1px solid {BORDER};border-radius:12px;padding:16px'>"
                    f"<p style='color:{DANGER};font-weight:700;font-size:13px;text-transform:uppercase;"
                    f"letter-spacing:.06em;margin-bottom:8px'>⚠️ Avoid</p>"
                    + "".join(f"<p style='color:{TEXT};margin:4px 0;font-size:14px'>• {c}</p>"
                              for c in crop.get('avoid', []))
                    + "</div>", unsafe_allow_html=True)

            st.markdown(f"""
<div style='background:#F0F7F1;border-left:4px solid {SUCCESS};border-radius:0 10px 10px 0;
            padding:14px 18px;margin-top:16px'>
  <p style='color:{SUCCESS};font-weight:700;font-size:13px;margin:0 0 4px'>💡 Farming Tips</p>
  <p style='color:{TEXT};font-size:14px;margin:0'>{crop.get('tips', '')}</p>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# EVALUATION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊  Evaluation":
    import seaborn as sns
    from sklearn.metrics import classification_report, confusion_matrix

    st.title("📊 Model Evaluation")
    st.markdown(f"<p style='color:{MUTED}'>Runs the model against the full test set and computes metrics.</p>", unsafe_allow_html=True)

    @st.cache_data(show_spinner="Evaluating on test set… this may take a minute.")
    def run_evaluation():
        model       = load_model()
        labels      = load_labels()
        class_names = [labels[i] for i in sorted(labels)]
        test_ds = tf.keras.utils.image_dataset_from_directory(
            os.path.join(DATASET_DIR, 'test'),
            label_mode='categorical', image_size=(224, 224), batch_size=32, shuffle=False
        )
        AUTOTUNE = tf.data.AUTOTUNE
        test_ds = test_ds.map(
            lambda x, y: (tf.cast(x, tf.float32) / 255.0, y),
            num_parallel_calls=AUTOTUNE
        ).prefetch(AUTOTUNE)
        preds  = model.predict(test_ds, verbose=0)
        y_pred = np.argmax(preds, axis=1)
        y_true = []
        for _, lb in test_ds:
            y_true.extend(np.argmax(lb.numpy(), axis=1))
        y_true = np.array(y_true)
        loss, acc = model.evaluate(test_ds, verbose=0)
        report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
        cm     = confusion_matrix(y_true, y_pred)
        return y_true, y_pred, class_names, acc, loss, report, cm

    if st.button("▶  Run Evaluation", type="primary"):
        y_true, y_pred, class_names, acc, loss, report, cm = run_evaluation()

        st.divider()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Test Accuracy", f"{acc*100:.2f}%")
        c2.metric("Test Loss",     f"{loss:.4f}")
        c3.metric("Macro F1",      f"{report['macro avg']['f1-score']*100:.2f}%")
        c4.metric("Weighted F1",   f"{report['weighted avg']['f1-score']*100:.2f}%")

        st.divider()
        col1, col2 = st.columns(2, gap="large")

        # Confusion matrix
        with col1:
            st.subheader("Confusion Matrix")
            fig, ax = plt.subplots(figsize=(6, 5))
            chart_style(fig, ax)
            cmap = sns.light_palette(PRIMARY, as_cmap=True)
            sns.heatmap(cm, annot=True, fmt='d', cmap=cmap,
                        xticklabels=[c.capitalize() for c in class_names],
                        yticklabels=[c.capitalize() for c in class_names],
                        ax=ax, linewidths=0.5, linecolor=BORDER,
                        annot_kws={"color": TEXT, "fontsize": 11, "fontweight": "bold"})
            ax.set_xlabel('Predicted', labelpad=10)
            ax.set_ylabel('True', labelpad=10)
            plt.xticks(rotation=30, ha='right')
            plt.tight_layout()
            st.pyplot(fig)

        # Per-class table + normalised CM
        with col2:
            st.subheader("Per-class Metrics")
            rows = [{"Class": c.capitalize(),
                     "Precision": f"{report[c]['precision']*100:.1f}%",
                     "Recall":    f"{report[c]['recall']*100:.1f}%",
                     "F1-Score":  f"{report[c]['f1-score']*100:.1f}%",
                     "Support":   int(report[c]['support'])}
                    for c in class_names]
            st.dataframe(rows, use_container_width=True, hide_index=True)

            st.subheader("Normalised Confusion Matrix")
            cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
            fig2, ax2 = plt.subplots(figsize=(6, 5))
            chart_style(fig2, ax2)
            cmap2 = sns.light_palette(DANGER, as_cmap=True)
            sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap=cmap2,
                        xticklabels=[c.capitalize() for c in class_names],
                        yticklabels=[c.capitalize() for c in class_names],
                        ax=ax2, linewidths=0.5, linecolor=BORDER, vmin=0, vmax=1,
                        annot_kws={"color": TEXT, "fontsize": 10, "fontweight": "bold"})
            ax2.set_xlabel('Predicted', labelpad=10)
            ax2.set_ylabel('True', labelpad=10)
            plt.xticks(rotation=30, ha='right')
            plt.tight_layout()
            st.pyplot(fig2)

        # F1 bar chart
        st.divider()
        st.subheader("Per-class F1 Score")
        fig3, ax3 = plt.subplots(figsize=(8, 3.5))
        chart_style(fig3, ax3)
        f1_vals  = [report[c]['f1-score'] for c in class_names]
        bar_cols = [CLASS_COLORS.get(c, PRIMARY) for c in class_names]
        bars = ax3.bar([c.capitalize() for c in class_names], f1_vals,
                       color=bar_cols, edgecolor=BORDER, linewidth=0.8)
        for bar, val in zip(bars, f1_vals):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                     f"{val*100:.1f}%", ha='center', va='bottom', color=TEXT, fontsize=9, fontweight='bold')
        ax3.set_ylim(0, 1.15)
        ax3.set_ylabel('F1 Score')
        ax3.axhline(report['macro avg']['f1-score'], color=ACCENT, linestyle='--',
                    linewidth=1.5, label=f"Macro avg {report['macro avg']['f1-score']*100:.1f}%")
        ax3.legend(facecolor=CARD, edgecolor=BORDER)
        plt.tight_layout()
        st.pyplot(fig3)

        if os.path.exists(TRAIN_PLOT):
            st.divider()
            st.subheader("Training History")
            st.image(TRAIN_PLOT, use_container_width=True)
    else:
        st.info("Click **▶ Run Evaluation** to compute metrics on the test set. Results are cached after the first run.")

# ══════════════════════════════════════════════════════════════════════════════
# DATASET
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📁  Dataset":
    st.title("📁 Dataset Explorer")
    st.markdown(f"<p style='color:{MUTED}'>Distribution of images across classes and splits.</p>", unsafe_allow_html=True)

    counts  = dataset_counts()
    classes = sorted(counts.keys())
    splits  = ['train', 'validation', 'test']

    # Summary table
    st.subheader("Image counts")
    table = []
    for cls in classes:
        row = {"Class": cls.capitalize()}
        total = 0
        for sp in splits:
            n = counts[cls].get(sp, 0)
            row[sp.capitalize()] = n
            total += n
        row["Total"] = total
        table.append(row)
    st.dataframe(table, use_container_width=True, hide_index=True)

    st.divider()
    col1, col2 = st.columns(2, gap="large")

    # Stacked bar
    with col1:
        st.subheader("Distribution by split")
        fig, ax = plt.subplots(figsize=(6, 4))
        chart_style(fig, ax)
        split_colors = [SUCCESS, ACCENT, DANGER]
        bottoms = np.zeros(len(classes))
        for sp, color in zip(splits, split_colors):
            vals = [counts[c].get(sp, 0) for c in classes]
            ax.bar([c.capitalize() for c in classes], vals, bottom=bottoms,
                   label=sp.capitalize(), color=color, alpha=0.9,
                   edgecolor=CARD, linewidth=0.8)
            bottoms += np.array(vals)
        ax.legend(facecolor=CARD, edgecolor=BORDER, labelcolor=TEXT)
        ax.set_ylabel('Images')
        plt.xticks(rotation=20, ha='right')
        plt.tight_layout()
        st.pyplot(fig)

    # Donut chart
    with col2:
        st.subheader("Training set class balance")
        train_vals  = [counts[c].get('train', 0) for c in classes]
        pie_colors  = [CLASS_COLORS.get(c, PRIMARY) for c in classes]
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        fig2.patch.set_facecolor(CARD)
        ax2.set_facecolor(CARD)
        wedges, texts, autotexts = ax2.pie(
            train_vals,
            labels=[c.capitalize() for c in classes],
            autopct='%1.1f%%',
            colors=pie_colors,
            startangle=140,
            textprops={'color': TEXT, 'fontsize': 9, 'fontweight': '600'},
            wedgeprops={'edgecolor': CARD, 'linewidth': 2},
            pctdistance=0.78
        )
        centre = plt.Circle((0, 0), 0.55, color=CARD)
        ax2.add_patch(centre)
        ax2.text(0, 0, f"{sum(train_vals):,}\nimages", ha='center', va='center',
                 color=PRIMARY, fontsize=10, fontweight='bold')
        for at in autotexts:
            at.set_color(CARD)
            at.set_fontsize(8)
            at.set_fontweight('bold')
        plt.tight_layout()
        st.pyplot(fig2)

    # Sample images
    st.divider()
    st.subheader("Sample images per class")
    cols = st.columns(len(classes))
    for col, cls in zip(cols, classes):
        cls_dir = os.path.join(DATASET_DIR, 'train', cls)
        imgs = [f for f in os.listdir(cls_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if imgs:
            col.image(os.path.join(cls_dir, imgs[0]), use_container_width=True)
            col.markdown(f"<p style='text-align:center;color:{CLASS_COLORS.get(cls,PRIMARY)};font-weight:700;font-size:13px'>{cls.capitalize()}</p>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# HYPERPARAMETERS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "⚙️  Hyperparameters":
    st.title("⚙️ Hyperparameters & Metrics")
    st.markdown(f"<p style='color:{MUTED}'>All training configuration used in this project, evaluation metrics tracked, and suggested improvements for future experiments.</p>", unsafe_allow_html=True)

    # ── helper to render a section card ──────────────────────────────────────
    def section(title, color=PRIMARY):
        st.markdown(f"<h3 style='color:{color};margin-top:8px'>{title}</h3>", unsafe_allow_html=True)

    def badge(label, value, note="", color=PRIMARY):
        note_html = f"<span style='color:{MUTED};font-size:12px;margin-left:8px'>{note}</span>" if note else ""
        st.markdown(
            f"<div style='background:{CARD};border:1px solid {BORDER};border-radius:10px;"
            f"padding:12px 16px;margin-bottom:8px;display:flex;align-items:center;gap:12px'>"
            f"<span style='color:{MUTED};font-size:13px;min-width:220px'>{label}</span>"
            f"<code style='background:#EDE3D8;color:{color};font-size:13px;font-weight:700;"
            f"padding:3px 10px;border-radius:6px'>{value}</code>"
            f"{note_html}</div>",
            unsafe_allow_html=True)

    def suggest(label, value, reason):
        st.markdown(
            f"<div style='background:#FFFBF5;border:1px dashed {ACCENT};border-radius:10px;"
            f"padding:12px 16px;margin-bottom:8px'>"
            f"<div style='display:flex;align-items:center;gap:10px;margin-bottom:4px'>"
            f"<span style='color:{MUTED};font-size:13px;min-width:220px'>{label}</span>"
            f"<code style='background:#FFF0D6;color:{ACCENT};font-size:13px;font-weight:700;"
            f"padding:3px 10px;border-radius:6px'>{value}</code>"
            f"<span style='color:#888;font-size:11px;font-weight:700;text-transform:uppercase;"
            f"letter-spacing:.06em'>suggested</span></div>"
            f"<p style='color:{MUTED};font-size:12px;margin:0;padding-left:4px'>💡 {reason}</p>"
            f"</div>",
            unsafe_allow_html=True)

    st.divider()

    # ── 1. Model Architecture ─────────────────────────────────────────────────
    section("🏗️ Model Architecture")
    col1, col2 = st.columns(2, gap="large")
    with col1:
        badge("Base model",          "MobileNetV2",          "pre-trained on ImageNet")
        badge("Input shape",         "224 × 224 × 3")
        badge("Custom head — Dense", "128 units",            "ReLU activation")
        badge("Dropout rate",        "0.5",                  "applied after Dense(128)")
        badge("Output units",        "5",                    "Softmax — one per class")
        badge("Fine-tune layers",    "top 20 of MobileNetV2","rest remain frozen")
    with col2:
        suggest("Dense units",    "256 or 512",   "Larger head may capture more complex soil features")
        suggest("Dropout rate",   "0.3 – 0.4",    "Lower dropout if model underfits on small classes (arid)")
        suggest("Extra Dense",    "Dense(64) after Dense(128)", "Adds depth before output; try with BatchNorm")
        suggest("Fine-tune layers","top 30 – 40", "Unfreeze more layers for domain-specific fine-tuning")
        suggest("Base model",     "EfficientNetB0 / B2", "Higher accuracy at similar or lower parameter count")

    st.divider()

    # ── 2. Training Phase 1 ───────────────────────────────────────────────────
    section("🔵 Phase 1 — Initial Training (Frozen Base)")
    col1, col2 = st.columns(2, gap="large")
    with col1:
        badge("Optimizer",        "Adam")
        badge("Learning rate",    "0.001")
        badge("Loss function",    "Categorical Cross-Entropy")
        badge("Metrics tracked",  "accuracy")
        badge("Max epochs",       "15")
        badge("Batch size",       "32")
        badge("EarlyStopping",    "patience = 5",   "monitors val_loss")
        badge("ModelCheckpoint",  "val_accuracy",   "saves best only → soil_classifier_initial.keras")
    with col2:
        suggest("Learning rate",  "0.0005",         "Slower start can improve convergence stability")
        suggest("Batch size",     "64",             "Larger batch = smoother gradients if GPU memory allows")
        suggest("LR scheduler",   "ReduceLROnPlateau", "Automatically reduce LR when val_loss plateaus")
        suggest("Metrics",        "+ val_top_2_accuracy", "Useful when classes are visually similar")
        suggest("EarlyStopping patience", "7 – 10", "Give model more time before stopping")

    st.divider()

    # ── 3. Training Phase 2 ───────────────────────────────────────────────────
    section("🟠 Phase 2 — Fine-Tuning (Top 20 Layers Unfrozen)")
    col1, col2 = st.columns(2, gap="large")
    with col1:
        badge("Optimizer",        "Adam")
        badge("Learning rate",    "0.0001",         "10× lower than Phase 1")
        badge("Loss function",    "Categorical Cross-Entropy")
        badge("Fine-tune epochs", "10")
        badge("Batch size",       "32")
        badge("ModelCheckpoint",  "val_accuracy",   "saves best only → soil_classifier_final.keras")
    with col2:
        suggest("Optimizer",      "SGD + momentum 0.9", "Often outperforms Adam in fine-tuning stage")
        suggest("Learning rate",  "1e-5 with warmup",   "Cosine decay or linear warmup prevents catastrophic forgetting")
        suggest("Mixed precision","float16 + float32",  "Speeds up training on modern GPUs with minimal accuracy loss")
        suggest("Gradient clipping", "clipnorm=1.0",    "Stabilises fine-tuning when unfreezing many layers")

    st.divider()

    # ── 4. Data Augmentation ──────────────────────────────────────────────────
    section("🖼️ Data Augmentation (Training Only)")
    col1, col2 = st.columns(2, gap="large")
    with col1:
        badge("RandomFlip",        "horizontal")
        badge("RandomRotation",    "± 20°  (factor=0.2)")
        badge("RandomZoom",        "± 20%  (factor=0.2)")
        badge("RandomTranslation", "± 10%  (height & width)")
        badge("Normalisation",     "÷ 255.0",        "pixel values scaled to [0, 1]")
    with col2:
        suggest("RandomBrightness",  "factor=0.2",   "Soil images vary in lighting conditions")
        suggest("RandomContrast",    "factor=0.2",   "Helps generalise across different camera settings")
        suggest("RandomSaturation",  "factor=0.3",   "Colour variation is key for soil type distinction")
        suggest("CutMix / MixUp",    "alpha=0.2",    "Advanced augmentation; reduces overfitting significantly")
        suggest("Gaussian noise",    "stddev=0.01",  "Simulates sensor noise in field photography")

    st.divider()

    # ── 5. Preprocessing ──────────────────────────────────────────────────────
    section("🔬 Preprocessing Pipeline")
    col1, col2 = st.columns(2, gap="large")
    with col1:
        badge("Histogram Equalization", "PIL ImageOps.equalize", "improves contrast")
        badge("Gaussian Blur",          "radius = 1.0",          "reduces noise")
        badge("Image resize",           "224 × 224 px")
        badge("Colour mode",            "RGB")
    with col2:
        suggest("CLAHE",              "Contrast Limited AHE",  "Better than global equalization for uneven lighting")
        suggest("Sharpening filter",  "UnsharpMask",           "Enhances soil texture edges")
        suggest("Standardisation",    "mean=0, std=1 per channel", "ImageNet-style normalisation instead of ÷255")

    st.divider()

    # ── 6. Evaluation Metrics ─────────────────────────────────────────────────
    section("📏 Evaluation Metrics", color=SUCCESS)
    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.markdown(f"<p style='color:{MUTED};font-size:13px;font-weight:700;text-transform:uppercase;letter-spacing:.06em;margin-bottom:8px'>Currently Tracked</p>", unsafe_allow_html=True)
        badge("Accuracy",           "correct / total",        "overall test-set accuracy")
        badge("Loss",               "Categorical Cross-Entropy", "lower = better")
        badge("Precision (per class)", "TP / (TP + FP)",      "how many predicted positives are correct")
        badge("Recall (per class)", "TP / (TP + FN)",         "how many actual positives are found")
        badge("F1-Score (per class)","2 × P × R / (P + R)",  "harmonic mean of precision & recall")
        badge("Macro F1",           "unweighted mean F1",     "treats all classes equally")
        badge("Weighted F1",        "support-weighted mean",  "accounts for class imbalance")
        badge("Confusion Matrix",   "raw counts",             "shows misclassification patterns")
        badge("Normalised CM",      "row-normalised",         "per-class recall at a glance")
    with col2:
        st.markdown(f"<p style='color:{MUTED};font-size:13px;font-weight:700;text-transform:uppercase;letter-spacing:.06em;margin-bottom:8px'>Suggested for Future</p>", unsafe_allow_html=True)
        suggest("Top-2 Accuracy",   "—",  "Useful when two soil types are visually similar (e.g. red vs. yellow)")
        suggest("ROC-AUC (per class)", "One-vs-Rest", "Measures discriminability independent of threshold")
        suggest("Cohen's Kappa",    "κ score",        "Measures agreement beyond chance; good for imbalanced classes")
        suggest("Matthews Correlation Coefficient", "MCC", "More informative than F1 for imbalanced datasets")
        suggest("Calibration curve","reliability diagram", "Checks if confidence scores are well-calibrated")
        suggest("Grad-CAM",         "visual explanation",  "Heatmap showing which image regions drove the prediction")

    st.divider()

    # ── 7. Future experiments summary ────────────────────────────────────────
    st.markdown(f"""
<div style='background:#FFFBF5;border:1.5px solid {ACCENT};border-radius:14px;padding:20px 24px'>
  <p style='color:{ACCENT};font-weight:800;font-size:14px;text-transform:uppercase;
            letter-spacing:.08em;margin:0 0 10px'>🚀 Recommended Next Experiments</p>
  <ol style='color:{TEXT};font-size:14px;line-height:2;margin:0;padding-left:20px'>
    <li>Replace MobileNetV2 with <strong>EfficientNetB2</strong> — better accuracy/param tradeoff</li>
    <li>Add <strong>class weights</strong> to loss to handle imbalance (arid has only ~200 train images)</li>
    <li>Try <strong>CutMix / MixUp</strong> augmentation to reduce overfitting on minority classes</li>
    <li>Use <strong>ReduceLROnPlateau</strong> callback instead of fixed LR in Phase 2</li>
    <li>Add <strong>Grad-CAM</strong> visualisation in the Classify page to explain predictions</li>
    <li>Experiment with <strong>label smoothing</strong> (ε=0.1) in the loss function</li>
    <li>Run <strong>k-fold cross-validation</strong> for more reliable performance estimates</li>
  </ol>
</div>
""", unsafe_allow_html=True)
