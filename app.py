import streamlit as st
import numpy as np
import json
from PIL import Image
import os
import zipfile
import shutil

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
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
                        pred_class = labels[idx]
                        confidence = float(probs[idx])
                        all_probs  = {labels[i]: float(p) for i, p in enumerate(probs)}
                    except Exception as e:
                        st.error(f"Inference failed: {e}")
                        st.stop()

                conf_pct  = confidence * 100
                cls_color = CLASS_COLORS.get(pred_class, PRIMARY)
                bar_color = SUCCESS if conf_pct >= 70 else ACCENT if conf_pct >= 40 else DANGER

                # Result card
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
                    label_color = PRIMARY if is_top else MUTED
                    weight = "800" if is_top else "500"
                    lc.markdown(f"<span style='color:{label_color};font-weight:{weight};font-size:14px'>{cls.capitalize()}</span>", unsafe_allow_html=True)
                    bc.progress(prob, text=f"{prob*100:.1f}%")

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
