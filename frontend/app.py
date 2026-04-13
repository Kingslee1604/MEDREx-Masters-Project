"""
MEDREx — Frontend UI
Pure Streamlit app. No ML logic here — all data comes from the FastAPI backend.

Run: streamlit run app.py
Requires backend running at http://localhost:8000
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests

API_BASE = "http://localhost:8000"

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MEDREx",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Example reports for live demo ─────────────────────────────────────────────
EXAMPLE_REPORTS = {
    "Breast Cancer (BRCA)": (
        "SPECIMEN: Right breast, lumpectomy. CLINICAL NOTES: Palpable mass, right breast, "
        "2 o'clock position. GROSS DESCRIPTION: Fibrofatty tissue with a firm white-tan area "
        "measuring 2.1 x 1.8 cm. MICROSCOPIC DESCRIPTION: Sections show an invasive ductal "
        "carcinoma, grade 2, with tubule formation and nuclear pleomorphism. Estrogen receptor "
        "positive (90%), progesterone receptor positive (70%), HER2 negative by IHC. "
        "No lymphovascular invasion. Surgical margins free of tumor."
    ),
    "Lung Adenocarcinoma (LUAD)": (
        "SPECIMEN: Right upper lobe, lobectomy. CLINICAL HISTORY: 67-year-old male, former "
        "smoker, 3.2 cm mass on CT scan. GROSS DESCRIPTION: 3.2 x 2.8 cm gray-white firm mass "
        "in the peripheral parenchyma. MICROSCOPIC DESCRIPTION: Adenocarcinoma with acinar "
        "pattern. TTF-1 and napsin A positive. No pleural invasion. Lymph node: 0/12 positive."
    ),
    "Kidney Cancer (KIRC)": (
        "SPECIMEN: Right kidney, radical nephrectomy. GROSS DESCRIPTION: 5.5 x 4.8 cm "
        "golden-yellow mass with areas of hemorrhage in the upper pole. MICROSCOPIC DESCRIPTION: "
        "Clear cell renal cell carcinoma, Fuhrman grade 2. Tumor cells have abundant clear "
        "cytoplasm with small round nuclei in nests surrounded by delicate vasculature. "
        "No sarcomatoid differentiation. Renal vein not involved."
    ),
    "Brain Cancer (GBM)": (
        "SPECIMEN: Left frontal lobe, stereotactic biopsy. CLINICAL HISTORY: Ring-enhancing "
        "lesion on MRI. MICROSCOPIC DESCRIPTION: High grade glial neoplasm consistent with "
        "glioblastoma multiforme, WHO Grade IV. Marked nuclear pleomorphism, brisk mitotic "
        "activity, microvascular proliferation, and geographic necrosis with pseudopalisading. "
        "GFAP positive, IDH1 R132H negative. Ki-67 approximately 35%."
    ),
    "Thyroid Cancer (THCA)": (
        "SPECIMEN: Thyroid gland, total thyroidectomy. GROSS DESCRIPTION: 2.1 cm well-"
        "circumscribed firm white nodule in right lobe. MICROSCOPIC DESCRIPTION: Papillary "
        "thyroid carcinoma, classic variant. Nuclear clearing, nuclear grooves, and "
        "intranuclear pseudoinclusions. Papillary architecture with fibrovascular cores. "
        "No capsular invasion. Surgical margins negative."
    ),
}

# ── API helpers ───────────────────────────────────────────────────────────────
def api_get(path: str, params: dict = None):
    try:
        r = requests.get(f"{API_BASE}{path}", params=params, timeout=10)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to MEDREx backend. Make sure the API is running: `uvicorn main:app --port 8000`")
        st.stop()
    except Exception as e:
        st.error(f"API error: {e}")
        st.stop()


def api_post(path: str, body: dict):
    try:
        r = requests.post(f"{API_BASE}{path}", json=body, timeout=60)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to MEDREx backend.")
        st.stop()
    except Exception as e:
        st.error(f"API error: {e}")
        st.stop()


# ── CSS ───────────────────────────────────────────────────────────────────────
def apply_css():
    st.markdown("""
    <style>
    .header-box {
        background: linear-gradient(135deg, #0d2137 0%, #1e3a5f 100%);
        padding: 22px 30px; border-radius: 12px; margin-bottom: 20px;
    }
    .header-box h1 { color:#fff; margin:0; font-size:2.4rem; letter-spacing:1px; }
    .header-box p  { color:#a8c8e8; margin:6px 0 0 0; font-size:0.95rem; }

    .metric-card {
        background:#f8fafc; border:1px solid #e2e8f0;
        border-left:5px solid #1e3a5f; border-radius:10px;
        padding:16px 20px; text-align:center;
    }
    .metric-card h3 { color:#1e3a5f; font-size:2rem; margin:0; font-weight:700; }
    .metric-card p  { color:#64748b; font-size:0.82rem; margin:6px 0 0 0; }

    .pred-box {
        background: linear-gradient(135deg, #0d2137, #1e3a5f);
        border-radius:12px; padding:20px 24px; color:white; text-align:center; margin:12px 0;
    }
    .pred-box h2 { margin:0; font-size:1.7rem; }
    .pred-box p  { margin:8px 0 0 0; color:#a8c8e8; font-size:1rem; }

    .endpoint-card {
        background:#f1f5f9; border:1px solid #cbd5e1; border-radius:8px;
        padding:10px 14px; margin-bottom:8px; font-family:monospace; font-size:0.85rem;
    }
    </style>
    """, unsafe_allow_html=True)


# ── Main ──────────────────────────────────────────────────────────────────────
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

    # Check backend health
    health = api_get("/health")

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## MEDREx")
        st.markdown("**LLM-Based Diagnostic Insight Extraction from Pathology Reports**")
        st.divider()
        st.markdown("**Student:** Kingslee Dominic Savio Velu")
        st.markdown("**Course:** CSC 590 – Master's Project")
        st.markdown("**Mentor:** Dr. Bin Tang")
        st.divider()

        # Pull live stats from API
        stats = api_get("/stats")
        st.markdown("### Dataset")
        st.metric("Total Reports",    f"{stats['total_reports']:,}")
        st.metric("Cancer Types",      stats["cancer_types"])
        st.metric("Avg Words/Report",  stats["avg_word_count"])
        st.divider()

        st.markdown("### Model Status")
        if health.get("tfidf_ready"):
            st.success(f"✅ TF-IDF + LR — {stats['model_accuracy']*100:.1f}%")
        else:
            st.warning("⏳ TF-IDF model loading...")
        st.info("🔄 TF-IDF + SVM — Next")
        st.warning("📋 BERT — Planned")

        # Vector store status
        st.divider()
        st.markdown("### Vector Store (ChromaDB)")
        vs = health
        if vs.get("vector_ready"):
            st.success(f"✅ {vs['vector_count']:,} embeddings ready")
        elif vs.get("vector_building"):
            pct = round(vs["vector_count"] / vs["vector_total"] * 100) if vs["vector_total"] else 0
            st.info(f"⏳ Building... {vs['vector_count']:,}/{vs['vector_total']:,} ({pct}%)")
            st.progress(pct / 100)
        else:
            st.warning("📋 Not built yet")
        st.caption("all-MiniLM-L6-v2 · 384-dim · cosine")

        st.warning("📋 RAG + LLM — Planned")
        st.divider()
        st.caption("Backend: http://localhost:8000")
        st.caption("API Docs: http://localhost:8000/docs")

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Data Explorer",
        "🏗️ System Architecture",
        "⚗️ Method Comparison",
        "🔬 Live Prediction",
        "🔌 API Reference",
    ])

    # ════════════════════════════════════════════════════════════════════════
    # TAB 1 — DATA EXPLORER
    # ════════════════════════════════════════════════════════════════════════
    with tab1:
        st.subheader("TCGA Pathology Report Dataset")

        # Metric cards (from /stats)
        c1, c2, c3, c4 = st.columns(4)
        cards = [
            (f"{stats['total_reports']:,}",   "Total Reports"),
            (stats["cancer_types"],           "Cancer Types"),
            (f"{stats['avg_word_count']:.0f}", "Avg Words / Report"),
            (f"{stats['pct_over_512']}%",      "Reports > 512 words"),
        ]
        for col, (val, label) in zip([c1, c2, c3, c4], cards):
            col.markdown(
                f'<div class="metric-card"><h3>{val}</h3><p>{label}</p></div>',
                unsafe_allow_html=True
            )

        st.markdown("<br>", unsafe_allow_html=True)

        col_l, col_r = st.columns([3, 2])

        with col_l:
            # Cancer type distribution from /distribution
            st.markdown("#### All 35 Cancer Types — Report Count")
            dist = api_get("/distribution")
            df_dist = pd.DataFrame(dist)
            fig = px.bar(
                df_dist, x="count", y="label", orientation="h",
                color="count", color_continuous_scale="Blues",
                labels={"count": "Reports", "label": ""},
                height=700
            )
            fig.update_layout(
                yaxis=dict(autorange="reversed", color="#e2e8f0", tickfont=dict(color="#e2e8f0")),
                xaxis=dict(color="#e2e8f0", tickfont=dict(color="#e2e8f0")),
                coloraxis_showscale=False,
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e2e8f0"),
                margin=dict(l=5, r=5, t=5, b=5)
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_r:
            # Text length histogram from /text-lengths
            st.markdown("#### Report Length Distribution")
            tl = api_get("/text-lengths")
            fig2 = px.histogram(
                x=tl["word_counts"], nbins=tl["bins"],
                labels={"x": "Words per Report"},
                color_discrete_sequence=["#1e3a5f"], height=280
            )
            fig2.add_vline(x=512, line_dash="dash", line_color="orange",
                           annotation_text="BERT 512-token limit",
                           annotation_font_color="#e2e8f0")
            fig2.update_layout(
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e2e8f0"),
                xaxis=dict(color="#e2e8f0", tickfont=dict(color="#e2e8f0")),
                yaxis=dict(color="#e2e8f0", tickfont=dict(color="#e2e8f0")),
                margin=dict(l=5, r=5, t=5, b=5)
            )
            st.plotly_chart(fig2, use_container_width=True)

            # Class imbalance summary
            st.markdown("#### Class Imbalance")
            top5 = df_dist.head(5)
            bot5 = df_dist.tail(5)
            ratio = int(df_dist["count"].max() / df_dist["count"].min())
            st.metric("Imbalance Ratio", f"{ratio}x")

            col_t, col_b = st.columns(2)
            with col_t:
                st.markdown("**Largest:**")
                for _, r in top5.iterrows():
                    st.markdown(f"- **{r['code']}**: {r['count']:,}")
            with col_b:
                st.markdown("**Smallest:**")
                for _, r in bot5.iterrows():
                    st.markdown(f"- **{r['code']}**: {r['count']:,}")

            # Random sample report
            st.markdown("#### Sample Report")
            if st.button("🔄 Load Random Report"):
                sample = api_get("/sample")
                st.info(f"**{sample['cancer_type']}** — {sample['cancer_name']}  |  {sample['word_count']} words")
                st.text_area("Preview:", sample["text_preview"] + "...", height=140)

    # ════════════════════════════════════════════════════════════════════════
    # TAB 2 — SYSTEM ARCHITECTURE
    # ════════════════════════════════════════════════════════════════════════
    with tab2:
        st.subheader("MEDREx System Architecture — RAG Pipeline")
        st.markdown("""
        MEDREx is built on **Retrieval-Augmented Generation (RAG)** to solve a core clinical problem:
        LLMs hallucinate, and patient data cannot be sent to external APIs (HIPAA).

        > *"The system separates retrieval from generation — reducing hallucination and improving reliability."*
        """)

        c1, c2, c3 = st.columns(3)
        boxes = [
            ("#1e3a5f", "#7dd3fc", "① DATA INGESTION", "(Offline)",
             "📄 Pathology Reports\nTCGA — 9,523 reports",
             "📚 Biomedical Papers\nPubMed / arXiv",
             "✂️ Preprocess & Chunk\nText splitting by section"),
            ("#14532d", "#86efac", "② RETRIEVAL", "(Online)",
             "🧮 Embed + Vector DB\nFAISS / ChromaDB",
             "🔍 Semantic Retriever\nTop-k similarity search",
             "❓ User Query\nClinical question"),
            ("#7c2d12", "#fdba74", "③ GENERATION", "(Online)",
             "🤖 LLM Generator\nGPT-4 / LLaMA (local)",
             "📋 Structured Response\nWith source citations",
             "✅ Evidence-grounded\nTraceable reasoning"),
        ]
        for col, (bg, title_color, title, sub, l1, l2, l3) in zip([c1, c2, c3], boxes):
            col.markdown(f"""
            <div style="background:{bg};color:white;border-radius:10px;padding:20px;
                        text-align:center;min-height:260px;">
              <h3 style="color:{title_color};margin-top:0;">{title}</h3>
              <p style="color:#94a3b8;font-size:0.78rem;">{sub}</p>
              <hr style="border-color:#334155;">
              <p style="white-space:pre-line;font-size:0.85rem;">{l1}</p>
              <p style="white-space:pre-line;font-size:0.85rem;">{l2}</p>
              <p style="white-space:pre-line;font-size:0.85rem;">{l3}</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("""
        <div style="background:#1e293b;border:1px solid #334155;border-radius:8px;
                    padding:12px 20px;text-align:center;
                    font-size:0.9rem;color:#cbd5e1;">
        <b style="color:#7dd3fc;">Tech Stack:</b>&nbsp; Python &nbsp;•&nbsp; FastAPI &nbsp;•&nbsp; LangChain &nbsp;•&nbsp;
        HuggingFace Embeddings &nbsp;•&nbsp; FAISS / ChromaDB &nbsp;•&nbsp;
        GPT-4 / LLaMA &nbsp;•&nbsp; Streamlit
        </div>
        """, unsafe_allow_html=True)

        st.divider()
        st.markdown("#### Why RAG over plain LLM?")
        df_why = pd.DataFrame([
            {"Approach": "Plain LLM (ChatGPT API)", "Hallucination": "High",   "HIPAA Safe": "❌ No",  "Explainable": "❌ No",  "Updateable": "❌ No"},
            {"Approach": "Fine-tuned LLM",           "Hallucination": "Medium","HIPAA Safe": "✅ Yes", "Explainable": "❌ No",  "Updateable": "Hard"},
            {"Approach": "MEDREx (RAG + LLM)",        "Hallucination": "Low",  "HIPAA Safe": "✅ Yes", "Explainable": "✅ Yes", "Updateable": "✅ Yes"},
        ])
        st.dataframe(df_why, use_container_width=True, hide_index=True)

    # ════════════════════════════════════════════════════════════════════════
    # TAB 3 — METHOD COMPARISON
    # ════════════════════════════════════════════════════════════════════════
    with tab3:
        st.subheader("5-Method Evaluation Framework")
        st.markdown("""
        We test 5 NLP approaches on the same dataset to scientifically justify why MEDREx uses RAG.
        Each answers: **"Is added complexity worth the accuracy gain?"**
        """)

        methods = api_get("/methods")
        df_m = pd.DataFrame(methods)
        df_m["accuracy_display"] = df_m["accuracy"].apply(
            lambda x: f"{x:.1f}%" if x is not None else "TBD"
        )
        df_m["status_display"] = df_m["status"].map({
            "done": "✅ Done", "next": "🔄 Next", "planned": "📋 Planned"
        })
        display_cols = {
            "rank": "#", "name": "Method", "category": "Category",
            "status_display": "Status", "accuracy_display": "Accuracy",
            "needs_gpu": "Needs GPU", "explainable": "Explainable"
        }
        st.dataframe(
            df_m.rename(columns=display_cols)[list(display_cols.values())],
            use_container_width=True, hide_index=True
        )

        st.divider()
        col_chart, col_info = st.columns(2)

        with col_chart:
            st.markdown("#### Accuracy Progress")
            names  = [m["name"].replace(" (MEDREx Core)", "").replace(" (Bio_ClinicalBERT)", "") for m in methods]
            values = [m["accuracy"] if m["accuracy"] else 0 for m in methods]
            colors = ["#1e3a5f" if v > 0 else "#e2e8f0" for v in values]
            labels = [f"{v:.1f}%" if v > 0 else "TBD" for v in values]

            fig3 = go.Figure(go.Bar(
                x=names, y=values,
                marker_color=colors,
                text=labels, textposition="outside"
            ))
            fig3.update_layout(
                yaxis=dict(range=[0, 105], title="Test Accuracy (%)", color="#e2e8f0", tickfont=dict(color="#e2e8f0")),
                xaxis=dict(color="#e2e8f0", tickfont=dict(color="#e2e8f0")),
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e2e8f0"),
                height=320, margin=dict(l=5, r=5, t=10, b=5)
            )
            st.plotly_chart(fig3, use_container_width=True)

        with col_info:
            st.markdown("#### Industry Applications")
            for m in methods:
                icon = "✅" if m["status"] == "done" else "🔄" if m["status"] == "next" else "📋"
                st.markdown(f"**{icon} {m['name']}**")
                st.caption(m["industry"])

            st.divider()
            ref_acc = 95.31
            our_acc = stats["model_accuracy"] * 100
            st.info(
                f"**Reference (Cedars-Sinai):** BoW + LR = **{ref_acc}%** on 32 classes\n\n"
                f"**MEDREx TF-IDF + LR:** **{our_acc:.1f}%** on all 35 classes"
            )

        # Per-class F1 chart
        st.divider()
        st.markdown("#### Per-Class F1 — Which Cancer Types Are Hardest to Predict?")
        f1_data = api_get("/class-f1")
        df_f1   = pd.DataFrame(f1_data)
        df_f1["color"] = df_f1["f1"].apply(
            lambda x: "#ef4444" if x < 0.7 else "#f97316" if x < 0.85 else "#22c55e"
        )
        fig4 = go.Figure(go.Bar(
            x=df_f1["f1"], y=df_f1["cancer_code"],
            orientation="h",
            marker_color=df_f1["color"].tolist(),
            text=[f"{v:.2f}" for v in df_f1["f1"]],
            textposition="inside",
        ))
        fig4.update_traces(insidetextfont=dict(color="white", size=12))
        fig4.update_layout(
            xaxis=dict(range=[0, 1.05], title=dict(text="F1 Score"),
                       color="#e2e8f0", tickfont=dict(color="#e2e8f0")),
            yaxis=dict(autorange="reversed", color="#e2e8f0",
                       tickfont=dict(color="#e2e8f0", size=13)),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e2e8f0"),
            height=750, margin=dict(l=60, r=20, t=10, b=40),
        )
        st.plotly_chart(fig4, use_container_width=True)

    # ════════════════════════════════════════════════════════════════════════
    # TAB 4 — LIVE PREDICTION
    # ════════════════════════════════════════════════════════════════════════
    with tab4:
        st.subheader("Live Cancer Type Prediction")
        st.markdown(
            "Paste a pathology report and MEDREx predicts the cancer type. "
            "All processing happens in the backend API — the UI just shows results."
        )

        col_a, _ = st.columns([2, 3])
        with col_a:
            choice = st.selectbox("Quick example:", ["— Paste your own —"] + list(EXAMPLE_REPORTS.keys()))

        report_text = EXAMPLE_REPORTS.get(choice, "")

        user_input = st.text_area(
            "Pathology Report Text:",
            value=report_text,
            height=210,
            placeholder="Paste pathology report text here, or pick an example above..."
        )

        predict_btn = st.button("🔬 Analyze Report", type="primary")

        if predict_btn:
            if not user_input.strip():
                st.warning("Please enter or select a report first.")
            else:
                with st.spinner("Sending to API..."):
                    result = api_post("/predict", {"text": user_input})

                pred = result["predicted"]
                top5 = result["top5"]

                st.divider()

                # Main result
                st.markdown(f"""
                <div class="pred-box">
                    <h2>Predicted: {pred['cancer_code']}</h2>
                    <p>{pred['cancer_name']}</p>
                </div>
                """, unsafe_allow_html=True)

                m1, m2, m3 = st.columns(3)
                m1.metric("Confidence",   f"{pred['confidence']*100:.1f}%")
                m2.metric("Cancer Code",  pred["cancer_code"])
                m3.metric("Model",        result["model_used"])

                st.markdown("#### Top 5 Predictions")
                for item in top5:
                    label = f"**{item['cancer_code']}** — {item['cancer_name']}"
                    col_bar, col_pct = st.columns([5, 1])
                    with col_bar:
                        st.progress(float(item["confidence"]), text=label)
                    with col_pct:
                        st.markdown(f"**{item['confidence']*100:.1f}%**")

                # Similar cases from ChromaDB
                similar = result.get("similar_cases")
                if similar:
                    st.divider()
                    st.markdown("#### Similar Cases Found in ChromaDB")
                    st.caption("These are the most semantically similar real patient reports from the TCGA dataset.")
                    for i, case in enumerate(similar):
                        with st.expander(
                            f"#{i+1} — {case['cancer_type']} ({case['cancer_name']})  "
                            f"· Similarity: {case['similarity']*100:.1f}%"
                        ):
                            st.markdown(f"**Patient:** `{case['patient_id']}`")
                            st.markdown(f"**Cancer Type:** {case['cancer_type']} — {case['cancer_name']}")
                            st.progress(float(case["similarity"]),
                                        text=f"Semantic similarity: {case['similarity']*100:.1f}%")
                            st.text(case["text_preview"] + "...")
                elif health.get("vector_building"):
                    pct = round(health["vector_count"] / health["vector_total"] * 100) if health["vector_total"] else 0
                    st.info(f"⏳ ChromaDB is building in background ({pct}% done). Similar cases will appear once ready.")
                else:
                    st.caption("Similar cases will appear once ChromaDB finishes indexing.")

                st.caption(
                    f"API endpoint called: POST {API_BASE}/predict  |  "
                    "Future phases will add BERT and RAG-based reasoning."
                )

    # ════════════════════════════════════════════════════════════════════════
    # TAB 5 — API REFERENCE
    # ════════════════════════════════════════════════════════════════════════
    with tab5:
        st.subheader("MEDREx API Reference")
        st.markdown(
            f"Backend running at **{API_BASE}**  |  "
            f"[Interactive docs →]({API_BASE}/docs)"
        )

        endpoints = [
            ("GET",  "/health",       "Check if API and model are ready"),
            ("GET",  "/stats",        "Dataset statistics (total reports, accuracy, etc.)"),
            ("GET",  "/distribution", "Cancer type counts for bar chart"),
            ("GET",  "/text-lengths", "Word counts per report for histogram"),
            ("GET",  "/sample",       "Random pathology report (optional: ?cancer_type=BRCA)"),
            ("POST", "/predict",      'Predict cancer type — body: { "text": "..." }'),
            ("GET",  "/methods",      "All 5 methods with status and accuracy"),
            ("GET",  "/class-f1",     "Per-class F1 scores sorted worst to best"),
        ]

        for method, path, desc in endpoints:
            color = "#22c55e" if method == "GET" else "#3b82f6"
            st.markdown(
                f'<div class="endpoint-card">'
                f'<span style="color:{color};font-weight:700;">{method}</span> '
                f'<span style="color:#1e3a5f;">{API_BASE}{path}</span> '
                f'<span style="color:#64748b;"> — {desc}</span>'
                f'</div>',
                unsafe_allow_html=True
            )

        st.divider()
        st.markdown("#### Test /predict right here")
        test_text = st.text_area("Enter text to test:", height=100,
                                  placeholder="Invasive ductal carcinoma, breast tissue...")
        if st.button("Send to API"):
            if test_text.strip():
                res = api_post("/predict", {"text": test_text})
                st.json(res)
            else:
                st.warning("Enter some text first.")


if __name__ == "__main__":
    main()
