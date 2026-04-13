"""
MEDREx — Medical Evidence & Decision Reasoning eXchange
CSC 590 Master's Project | CSUDH Spring 2026
Student: Kingslee Dominic Savio Velu | Mentor: Dr. Bin Tang

Run: streamlit run app.py
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
import warnings
warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MEDREx",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR     = r"c:\Users\kings\OneDrive\Desktop\CSUDH_Kingslee\CSUDH-Spring-2026\CSC-590-MastersProject\MastersProject"
REPORTS_PATH = os.path.join(BASE_DIR, "DataSet", "TCGA_Reports.csv", "TCGA_Reports.csv")
LABELS_PATH  = os.path.join(BASE_DIR, "DataSet", "tcga_patient_to_cancer_type.csv")

# ── Cancer type full names ────────────────────────────────────────────────────
CANCER_NAMES = {
    "ACC":  "Adrenocortical carcinoma",
    "BLCA": "Bladder Urothelial Carcinoma",
    "BRCA": "Breast invasive carcinoma",
    "CESC": "Cervical squamous cell carcinoma",
    "CHOL": "Cholangiocarcinoma",
    "COAD": "Colon adenocarcinoma",
    "DLBC": "Diffuse Large B-cell Lymphoma",
    "ESCA": "Esophageal carcinoma",
    "GBM":  "Glioblastoma multiforme",
    "HNSC": "Head and Neck squamous cell carcinoma",
    "KICH": "Kidney Chromophobe",
    "KIRC": "Kidney renal clear cell carcinoma",
    "KIRP": "Kidney renal papillary cell carcinoma",
    "LAML": "Acute Myeloid Leukemia",
    "LGG":  "Brain Lower Grade Glioma",
    "LIHC": "Liver hepatocellular carcinoma",
    "LUAD": "Lung adenocarcinoma",
    "LUSC": "Lung squamous cell carcinoma",
    "MESO": "Mesothelioma",
    "OV":   "Ovarian serous cystadenocarcinoma",
    "PAAD": "Pancreatic adenocarcinoma",
    "PCPG": "Pheochromocytoma and Paraganglioma",
    "PRAD": "Prostate adenocarcinoma",
    "READ": "Rectum adenocarcinoma",
    "SARC": "Sarcoma",
    "SKCM": "Skin Cutaneous Melanoma",
    "STAD": "Stomach adenocarcinoma",
    "TGCT": "Testicular Germ Cell Tumors",
    "THCA": "Thyroid carcinoma",
    "THYM": "Thymoma",
    "UCEC": "Uterine Corpus Endometrial Carcinoma",
    "UCS":  "Uterine Carcinosarcoma",
    "UVM":  "Uveal Melanoma",
}

# ── Example pathology reports for live demo ───────────────────────────────────
EXAMPLE_REPORTS = {
    "Breast Cancer (BRCA)": (
        "SPECIMEN: Right breast, lumpectomy. CLINICAL NOTES: Palpable mass, right breast, "
        "2 o'clock position. GROSS DESCRIPTION: Received in formalin, labeled right breast "
        "lumpectomy, is a 4.5 x 3.2 x 2.8 cm fibrofatty tissue with a firm white-tan area "
        "measuring 2.1 x 1.8 cm. MICROSCOPIC DESCRIPTION: Sections show an invasive ductal "
        "carcinoma, grade 2, with tubule formation and nuclear pleomorphism. Estrogen receptor "
        "positive (90%), progesterone receptor positive (70%), HER2 negative by IHC. No "
        "lymphovascular invasion identified. Surgical margins are free of tumor."
    ),
    "Lung Adenocarcinoma (LUAD)": (
        "SPECIMEN: Right upper lobe, lobectomy. CLINICAL HISTORY: 67-year-old male, former "
        "smoker, 3.2 cm mass right upper lobe on CT scan. GROSS DESCRIPTION: Right upper lobe "
        "lobectomy specimen. Cut surface reveals a 3.2 x 2.8 cm gray-white firm mass with "
        "irregular borders in the peripheral parenchyma. MICROSCOPIC DESCRIPTION: Sections "
        "reveal an adenocarcinoma with predominantly acinar pattern. Tumor cells form glandular "
        "structures with moderate nuclear atypia. TTF-1 and napsin A positive by "
        "immunohistochemistry. No pleural invasion. Bronchial margin free of tumor. "
        "Lymph node: 0/12 positive for malignancy."
    ),
    "Kidney Cancer (KIRC)": (
        "SPECIMEN: Right kidney, radical nephrectomy. CLINICAL HISTORY: 58-year-old male with "
        "right renal mass discovered incidentally on CT. GROSS DESCRIPTION: Right kidney 11 x 7 "
        "x 5 cm. A well-circumscribed 5.5 x 4.8 cm golden-yellow mass with areas of hemorrhage "
        "in the upper pole. MICROSCOPIC DESCRIPTION: Clear cell renal cell carcinoma, Fuhrman "
        "grade 2. Tumor cells have abundant clear cytoplasm with small round nuclei arranged in "
        "nests surrounded by delicate vasculature. No sarcomatoid differentiation. Renal vein "
        "not involved."
    ),
    "Brain Cancer (GBM)": (
        "SPECIMEN: Left frontal lobe, stereotactic biopsy. CLINICAL HISTORY: 55-year-old "
        "presenting with headaches and right-sided weakness. MRI shows ring-enhancing lesion "
        "left frontal lobe. GROSS DESCRIPTION: Multiple tan-pink fragments of soft tissue. "
        "MICROSCOPIC DESCRIPTION: High grade glial neoplasm consistent with glioblastoma "
        "multiforme, WHO Grade IV. Sections show marked nuclear pleomorphism, brisk mitotic "
        "activity, microvascular proliferation, and geographic necrosis with pseudopalisading. "
        "GFAP positive, IDH1 R132H negative. Ki-67 labeling index approximately 35%."
    ),
    "Thyroid Cancer (THCA)": (
        "SPECIMEN: Thyroid gland, total thyroidectomy. CLINICAL HISTORY: 42-year-old female "
        "with thyroid nodule. GROSS DESCRIPTION: Total thyroid gland 6 x 4 x 3 cm. Right lobe "
        "contains a 2.1 cm well-circumscribed firm white nodule. MICROSCOPIC DESCRIPTION: "
        "Papillary thyroid carcinoma, classic variant. Tumor cells show characteristic nuclear "
        "features including nuclear clearing, nuclear grooves, and intranuclear pseudoinclusions. "
        "Papillary architecture with fibrovascular cores. No capsular invasion. Lymphovascular "
        "invasion absent. Surgical margins negative."
    ),
}

# ── CSS styling ───────────────────────────────────────────────────────────────
def apply_css():
    st.markdown("""
    <style>
    .header-box {
        background: linear-gradient(135deg, #0d2137 0%, #1e3a5f 100%);
        padding: 22px 30px; border-radius: 12px; margin-bottom: 20px;
    }
    .header-box h1 { color: #ffffff; margin: 0; font-size: 2.4rem; letter-spacing: 1px; }
    .header-box p  { color: #a8c8e8; margin: 6px 0 0 0; font-size: 0.95rem; }
    .metric-card {
        background: #f8fafc; border: 1px solid #e2e8f0;
        border-left: 5px solid #1e3a5f; border-radius: 10px;
        padding: 16px 20px; text-align: center;
    }
    .metric-card h3 { color: #1e3a5f; font-size: 2rem; margin: 0; font-weight: 700; }
    .metric-card p  { color: #64748b; font-size: 0.82rem; margin: 6px 0 0 0; }
    .method-row {
        border: 1px solid #e2e8f0; border-radius: 8px;
        padding: 12px 16px; margin-bottom: 8px;
        display: flex; align-items: center; gap: 12px;
    }
    .badge-done    { background:#dcfce7; color:#166534; padding:3px 12px; border-radius:12px; font-size:0.78rem; font-weight:700; }
    .badge-next    { background:#dbeafe; color:#1e40af; padding:3px 12px; border-radius:12px; font-size:0.78rem; font-weight:700; }
    .badge-planned { background:#fef3c7; color:#92400e; padding:3px 12px; border-radius:12px; font-size:0.78rem; font-weight:700; }
    .pred-box {
        background: linear-gradient(135deg, #0d2137, #1e3a5f);
        border-radius: 12px; padding: 20px 24px; color: white; text-align: center;
        margin: 12px 0;
    }
    .pred-box h2 { margin: 0; font-size: 1.6rem; }
    .pred-box p  { margin: 8px 0 0 0; color: #a8c8e8; }
    </style>
    """, unsafe_allow_html=True)

# ── Data loading (cached) ─────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading TCGA dataset...")
def load_data():
    df_r = pd.read_csv(REPORTS_PATH)
    df_l = pd.read_csv(LABELS_PATH)
    df_r["patient_id"] = df_r["patient_filename"].apply(lambda x: x.split(".")[0])
    df = df_r.merge(df_l, on="patient_id", how="inner")
    df["word_count"]  = df["text"].apply(lambda x: len(str(x).split()))
    df["cancer_name"] = df["cancer_type"].map(CANCER_NAMES).fillna(df["cancer_type"])
    return df

# ── Model training (cached) ───────────────────────────────────────────────────
@st.cache_resource(show_spinner="Training TF-IDF model (first run only — takes ~60 sec)...")
def train_model(_df):
    X_tr, X_te, y_tr, y_te = train_test_split(
        _df["text"], _df["cancer_type"],
        test_size=0.3, random_state=42, stratify=_df["cancer_type"]
    )
    vec = TfidfVectorizer(max_features=15000, ngram_range=(1, 2),
                          sublinear_tf=True, min_df=2)
    X_tr_v = vec.fit_transform(X_tr)
    X_te_v = vec.transform(X_te)

    model = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced",
                               solver="lbfgs", random_state=42)
    model.fit(X_tr_v, y_tr)

    y_pred = model.predict(X_te_v)
    acc = accuracy_score(y_te, y_pred)
    f1  = f1_score(y_te, y_pred, average="weighted")
    return model, vec, acc, f1, y_te.values, y_pred


# ── Main app ──────────────────────────────────────────────────────────────────
def main():
    apply_css()

    # Header
    st.markdown("""
    <div class="header-box">
        <h1>🔬 MEDREx</h1>
        <p>Medical Evidence &amp; Decision Reasoning eXchange &nbsp;|&nbsp;
           CSC 590 Master's Project &nbsp;|&nbsp; CSUDH Spring 2026</p>
    </div>
    """, unsafe_allow_html=True)

    # Load data + train model
    df = load_data()
    model, vec, acc, f1, y_test, y_pred = train_model(df)

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## MEDREx")
        st.markdown("**LLM-Based Diagnostic Insight Extraction from Pathology Reports**")
        st.divider()
        st.markdown("**Student:** Kingslee Dominic Savio Velu")
        st.markdown("**Course:** CSC 590 – Master's Project")
        st.markdown("**Mentor:** Dr. Bin Tang")
        st.markdown("**Sponsor:** Cedars-Sinai AI Campus")
        st.divider()
        st.markdown("### Dataset")
        st.metric("Total Reports",  f"{len(df):,}")
        st.metric("Cancer Types",   df["cancer_type"].nunique())
        st.metric("Avg Words/Report", f"{int(df['word_count'].mean()):,}")
        st.divider()
        st.markdown("### Model Status")
        st.success(f"✅ TF-IDF + LR — {acc*100:.1f}%")
        st.info("🔄 TF-IDF + SVM — In Progress")
        st.warning("📋 BERT — Planned")
        st.warning("📋 Embedding k-NN — Planned")
        st.warning("📋 RAG + LLM — Planned")

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Data Explorer",
        "🏗️ System Architecture",
        "⚗️ Method Comparison",
        "🔬 Live Prediction"
    ])

    # ════════════════════════════════════════════════════════════════════════
    # TAB 1 — DATA EXPLORER
    # ════════════════════════════════════════════════════════════════════════
    with tab1:
        st.subheader("TCGA Pathology Report Dataset")

        c1, c2, c3, c4 = st.columns(4)
        for col, val, label in zip(
            [c1, c2, c3, c4],
            [f"{len(df):,}", df['cancer_type'].nunique(), f"{int(df['word_count'].mean()):,}", f"{df['word_count'].median():.0f}"],
            ["Total Reports", "Cancer Types", "Avg Words/Report", "Median Words/Report"]
        ):
            col.markdown(f'<div class="metric-card"><h3>{val}</h3><p>{label}</p></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        col_l, col_r = st.columns([3, 2])

        with col_l:
            st.markdown("#### All 35 Cancer Types — Report Count")
            ct = df["cancer_type"].value_counts().reset_index()
            ct.columns = ["code", "count"]
            ct["label"] = ct["code"] + " — " + ct["code"].map(CANCER_NAMES).fillna(ct["code"])
            fig = px.bar(ct, x="count", y="label", orientation="h",
                         color="count", color_continuous_scale="Blues",
                         labels={"count": "Reports", "label": ""},
                         height=680)
            fig.update_layout(
                yaxis=dict(autorange="reversed"),
                coloraxis_showscale=False,
                plot_bgcolor="white", paper_bgcolor="white",
                margin=dict(l=5, r=5, t=5, b=5)
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_r:
            st.markdown("#### Report Length Distribution")
            fig2 = px.histogram(df, x="word_count", nbins=50, height=280,
                                color_discrete_sequence=["#1e3a5f"],
                                labels={"word_count": "Words per Report"})
            fig2.add_vline(x=512, line_dash="dash", line_color="orange",
                           annotation_text="BERT 512-token limit")
            fig2.update_layout(
                plot_bgcolor="white", paper_bgcolor="white",
                margin=dict(l=5, r=5, t=5, b=5)
            )
            st.plotly_chart(fig2, use_container_width=True)

            st.markdown("#### Class Imbalance")
            top5 = ct.head(5)
            bot5 = ct.tail(5)
            ratio = ct["count"].max() / ct["count"].min()
            st.metric("Imbalance Ratio", f"{ratio:.0f}x")
            st.markdown("**Largest classes:**")
            for _, r in top5.iterrows():
                st.markdown(f"- **{r['code']}**: {r['count']:,}")
            st.markdown("**Smallest classes:**")
            for _, r in bot5.iterrows():
                st.markdown(f"- **{r['code']}**: {r['count']:,}")

            pct512 = (df["word_count"] > 512).mean() * 100
            st.info(f"**{pct512:.1f}%** of reports exceed 512 words → chunking needed for BERT/RAG")

    # ════════════════════════════════════════════════════════════════════════
    # TAB 2 — SYSTEM ARCHITECTURE
    # ════════════════════════════════════════════════════════════════════════
    with tab2:
        st.subheader("MEDREx System Architecture — RAG Pipeline")

        st.markdown("""
        MEDREx is built on **Retrieval-Augmented Generation (RAG)** to solve a fundamental
        clinical problem: LLMs hallucinate and patient data cannot be sent to external APIs (HIPAA).

        > *"The system separates retrieval from generation to reduce hallucination and improve reliability."*
        """)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("""
            <div style="background:#1e3a5f; color:white; border-radius:10px; padding:20px; text-align:center; height:260px;">
            <h3 style="color:#7dd3fc;">① DATA INGESTION</h3>
            <p style="color:#94a3b8; font-size:0.8rem;">(Offline)</p>
            <hr style="border-color:#334155;">
            <p>📄 <b>Pathology Reports</b><br><small>TCGA — 9,523 reports</small></p>
            <p>📚 <b>Biomedical Papers</b><br><small>PubMed / arXiv</small></p>
            <p>✂️ <b>Preprocess & Chunk</b><br><small>Text splitting by section</small></p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <div style="background:#14532d; color:white; border-radius:10px; padding:20px; text-align:center; height:260px;">
            <h3 style="color:#86efac;">② RETRIEVAL</h3>
            <p style="color:#94a3b8; font-size:0.8rem;">(Online)</p>
            <hr style="border-color:#334155;">
            <p>🧮 <b>Embed + Vector DB</b><br><small>FAISS / ChromaDB</small></p>
            <p>🔍 <b>Semantic Retriever</b><br><small>Top-k similarity search</small></p>
            <p>❓ <b>User Query</b><br><small>Clinical question</small></p>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown("""
            <div style="background:#7c2d12; color:white; border-radius:10px; padding:20px; text-align:center; height:260px;">
            <h3 style="color:#fdba74;">③ GENERATION</h3>
            <p style="color:#94a3b8; font-size:0.8rem;">(Online)</p>
            <hr style="border-color:#334155;">
            <p>🤖 <b>LLM Generator</b><br><small>GPT-4 / LLaMA (local)</small></p>
            <p>📋 <b>Structured Response</b><br><small>With source citations</small></p>
            <p>✅ <b>Evidence-grounded</b><br><small>Traceable reasoning</small></p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div style="background:#f8fafc; border:1px solid #e2e8f0; border-radius:8px; padding:12px 20px; margin-top:16px; text-align:center; font-size:0.9rem; color:#475569;">
        <b>Tech Stack:</b> Python &nbsp;•&nbsp; LangChain &nbsp;•&nbsp; HuggingFace Embeddings &nbsp;•&nbsp; FAISS / ChromaDB &nbsp;•&nbsp; GPT-4 / LLaMA &nbsp;•&nbsp; FastAPI &nbsp;•&nbsp; Streamlit
        </div>
        """, unsafe_allow_html=True)

        st.divider()
        st.markdown("#### Why RAG over plain LLM?")
        comp = pd.DataFrame([
            {"Approach": "Plain LLM (ChatGPT API)", "Hallucination": "High", "HIPAA Safe": "❌ No", "Explainable": "❌ No", "Updateable": "❌ No"},
            {"Approach": "Fine-tuned LLM",           "Hallucination": "Medium", "HIPAA Safe": "✅ Yes", "Explainable": "❌ No", "Updateable": "Hard"},
            {"Approach": "MEDREx (RAG + LLM)",        "Hallucination": "Low",  "HIPAA Safe": "✅ Yes", "Explainable": "✅ Yes", "Updateable": "✅ Yes"},
        ])
        st.dataframe(comp, use_container_width=True, hide_index=True)

    # ════════════════════════════════════════════════════════════════════════
    # TAB 3 — METHOD COMPARISON
    # ════════════════════════════════════════════════════════════════════════
    with tab3:
        st.subheader("5-Method Evaluation Framework")
        st.markdown("""
        We test 5 fundamentally different NLP approaches on the same 9,523-report dataset
        to scientifically justify why MEDREx uses RAG. Each answers: **"Is added complexity worth the accuracy gain?"**
        """)

        methods = [
            ("1", "TF-IDF + Logistic Regression", "Traditional ML",   "✅ Done",    f"{acc*100:.1f}%", "No",      "Yes"),
            ("2", "TF-IDF + SVM",                 "Traditional ML",   "🔄 Next",    "—",                "No",      "Yes"),
            ("3", "BERT (Bio_ClinicalBERT)",       "Transformer",      "📋 Planned", "—",                "Yes",     "Partial"),
            ("4", "Embedding Similarity (k-NN)",  "Semantic Search",  "📋 Planned", "—",                "No",      "Yes"),
            ("5", "RAG + LLM (MEDREx Core)",       "LLM + Retrieval",  "📋 Planned", "—",                "Yes",     "Partial"),
        ]
        df_m = pd.DataFrame(methods, columns=["#", "Method", "Category", "Status", "Accuracy", "Needs GPU", "Explainable"])
        st.dataframe(df_m, use_container_width=True, hide_index=True)

        st.divider()

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Accuracy Progress")
            method_names = ["TF-IDF\n+ LR", "TF-IDF\n+ SVM", "BERT", "Embedding\nk-NN", "RAG\n+ LLM"]
            values = [acc * 100, 0, 0, 0, 0]
            bar_colors = ["#1e3a5f" if v > 0 else "#e2e8f0" for v in values]
            labels = [f"{v:.1f}%" if v > 0 else "TBD" for v in values]

            fig3 = go.Figure(go.Bar(
                x=method_names, y=values,
                marker_color=bar_colors,
                text=labels, textposition="outside"
            ))
            fig3.update_layout(
                yaxis=dict(range=[0, 105], title="Test Accuracy (%)"),
                plot_bgcolor="white", paper_bgcolor="white",
                height=320, margin=dict(l=5, r=5, t=10, b=5)
            )
            st.plotly_chart(fig3, use_container_width=True)

        with c2:
            st.markdown("#### Industry Applications")
            st.markdown("""
            | Method | Used In |
            |--------|---------|
            | TF-IDF + LR | FDA adverse events, hospital triage |
            | TF-IDF + SVM | Legal document classification |
            | BERT | Epic (77% US hospitals), Google Health |
            | Embedding k-NN | PubMed semantic search, Mayo Clinic |
            | RAG + LLM | Suki AI, Azure Health Bot, Epic |
            """)
            st.info(
                f"**Reference (Cedars-Sinai):** BoW + LR = **95.31%** on 32 classes\n\n"
                f"**Our TF-IDF + LR:** **{acc*100:.1f}%** on all 35 classes"
            )

    # ════════════════════════════════════════════════════════════════════════
    # TAB 4 — LIVE PREDICTION
    # ════════════════════════════════════════════════════════════════════════
    with tab4:
        st.subheader("Live Cancer Type Prediction")
        st.markdown(
            "Paste a pathology report below and MEDREx will predict the cancer type "
            "using the TF-IDF model. Try the quick examples to see it in action."
        )

        col_a, col_b = st.columns([1, 3])
        with col_a:
            choice = st.selectbox("Load example:", ["— Type your own —"] + list(EXAMPLE_REPORTS.keys()))

        report_text = EXAMPLE_REPORTS.get(choice, "")

        user_input = st.text_area(
            "Pathology Report Text:",
            value=report_text,
            height=210,
            placeholder="Paste a pathology report here, or select an example above..."
        )

        predict_btn = st.button("🔬 Analyze Report", type="primary", use_container_width=False)

        if predict_btn:
            if not user_input.strip():
                st.warning("Please enter or select a pathology report first.")
            else:
                with st.spinner("Analyzing..."):
                    x_vec = vec.transform([user_input])
                    probs   = model.predict_proba(x_vec)[0]
                    classes = model.classes_
                    top5_i  = np.argsort(probs)[::-1][:5]

                top_code = classes[top5_i[0]]
                top_name = CANCER_NAMES.get(top_code, top_code)
                top_prob = probs[top5_i[0]]

                st.divider()

                # Primary prediction box
                st.markdown(f"""
                <div class="pred-box">
                    <h2>Predicted: {top_code}</h2>
                    <p>{top_name}</p>
                </div>
                """, unsafe_allow_html=True)

                m1, m2, m3 = st.columns(3)
                m1.metric("Confidence",  f"{top_prob*100:.1f}%")
                m2.metric("Cancer Code", top_code)
                m3.metric("Model",       "TF-IDF + LR")

                st.markdown("#### Top 5 Predictions")
                for idx in top5_i:
                    code  = classes[idx]
                    name  = CANCER_NAMES.get(code, code)
                    prob  = float(probs[idx])
                    label = f"**{code}** — {name}"
                    col_bar, col_pct = st.columns([5, 1])
                    with col_bar:
                        st.progress(prob, text=label)
                    with col_pct:
                        st.markdown(f"**{prob*100:.1f}%**")

                st.divider()
                st.caption(
                    "This prediction uses TF-IDF + Logistic Regression (Method 1 of 5). "
                    "Future phases will add BERT and RAG-based reasoning with source citations."
                )


if __name__ == "__main__":
    main()
