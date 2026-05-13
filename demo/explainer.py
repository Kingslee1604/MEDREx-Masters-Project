"""
MEDREx — Cancer Classification Methods Explainer UI
=====================================================
Professor demonstration: shows how each model works with live examples.

Prerequisites:
  1. python demo/save_models.py           (trains & saves BoW/TF-IDF/SVM)
  2. Bio_ClinicalBERT at demo/ProjectCode/method3_best_model/

Run:
  streamlit run demo/explainer.py - This application is for DEMO
"""

import os, sys, json, time, warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR      = r"c:\Users\kings\OneDrive\Desktop\CSUDH_Kingslee\CSUDH-Spring-2026\CSC-590-MastersProject\MastersProject"
SAVE_DIR      = os.path.join(BASE_DIR, "demo", "saved_models")
MODEL3_DIR    = os.path.join(BASE_DIR, "demo", "ProjectCode", "method3_best_model")
EMB_DIR       = os.path.join(BASE_DIR, "demo", "ProjectCode", "output")
TEST_META_CSV = os.path.join(EMB_DIR, "method4_test_meta_sentence_transformers_all_MiniLM_L6_v2.csv")
TRAIN_EMB_NPZ = os.path.join(EMB_DIR, "method4_train_embeddings_sentence_transformers_all_MiniLM_L6_v2.npz")
TRAIN_META_CSV= os.path.join(EMB_DIR, "method4_train_meta_sentence_transformers_all_MiniLM_L6_v2.csv")

CANCER_NAMES = {
    "ACC":"Adrenocortical carcinoma","BLCA":"Bladder Urothelial Carcinoma",
    "BRCA":"Breast invasive carcinoma","CESC":"Cervical squamous cell carcinoma",
    "CHOL":"Cholangiocarcinoma","COAD":"Colon adenocarcinoma",
    "DLBC":"Diffuse Large B-cell Lymphoma","ESCA":"Esophageal carcinoma",
    "GBM":"Glioblastoma multiforme","HNSC":"Head and Neck squamous cell carcinoma",
    "KICH":"Kidney Chromophobe","KIRC":"Kidney renal clear cell carcinoma",
    "KIRP":"Kidney renal papillary cell carcinoma","LGG":"Brain Lower Grade Glioma",
    "LIHC":"Liver hepatocellular carcinoma","LUAD":"Lung adenocarcinoma",
    "LUSC":"Lung squamous cell carcinoma","MESO":"Mesothelioma",
    "OV":"Ovarian serous cystadenocarcinoma","PAAD":"Pancreatic adenocarcinoma",
    "PCPG":"Pheochromocytoma and Paraganglioma","PRAD":"Prostate adenocarcinoma",
    "READ":"Rectum adenocarcinoma","SARC":"Sarcoma","SKCM":"Skin Cutaneous Melanoma",
    "STAD":"Stomach adenocarcinoma","TGCT":"Testicular Germ Cell Tumors",
    "THCA":"Thyroid carcinoma","THYM":"Thymoma",
    "UCEC":"Uterine Corpus Endometrial Carcinoma","UCS":"Uterine Carcinosarcoma",
    "UVM":"Uveal Melanoma",
}

SAMPLE_CANCER_TYPES = ["BRCA", "LUAD", "GBM", "KIRC", "THCA", "PRAD", "BLCA", "OV", "STAD", "SKCM"]

# ── Cached data loaders ───────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_sample_reports():
    if not os.path.exists(TEST_META_CSV):
        return {}
    df = pd.read_csv(TEST_META_CSV)
    samples = {}
    for ct in SAMPLE_CANCER_TYPES:
        rows = df[df["cancer_type"] == ct]
        if len(rows) > 0:
            row = rows.iloc[0]
            samples[ct] = {"patient_id": row["patient_id"],
                           "text": row["text"],
                           "cancer_type": ct}
    return samples

@st.cache_resource(show_spinner=False)
def load_cedars_models():
    import joblib, nltk
    from nltk.tokenize import word_tokenize
    from sklearn.feature_extraction.text import CountVectorizer
    nltk.download("punkt_tab", quiet=True)
    nltk.download("stopwords", quiet=True)
    sw = set(nltk.corpus.stopwords.words("english"))
    def tokenizer(text):
        tokens = word_tokenize(text.lower())
        tokens = [t for t in tokens if len(t) > 1]
        tokens = [t for t in tokens if t not in sw]
        tokens = [t for t in tokens if t.isalpha()]
        return tokens
    vocab_path = os.path.join(SAVE_DIR, "cedars_vocabulary.json")
    clf_path   = os.path.join(SAVE_DIR, "cedars_clf.joblib")
    if not os.path.exists(vocab_path) or not os.path.exists(clf_path):
        return None, None, None
    vocab = json.load(open(vocab_path))
    cs_vec = CountVectorizer(tokenizer=tokenizer, vocabulary=vocab,
                             token_pattern=None, lowercase=False)
    cs_clf = joblib.load(clf_path)
    return cs_vec, cs_clf, tokenizer

@st.cache_resource(show_spinner=False)
def load_tfidf_models():
    import joblib
    paths = [os.path.join(SAVE_DIR, f) for f in
             ["tfidf_vectorizer.joblib", "m1_lr_clf.joblib", "m2_svm_clf.joblib"]]
    if not all(os.path.exists(p) for p in paths):
        return None, None, None
    return (joblib.load(paths[0]), joblib.load(paths[1]), joblib.load(paths[2]))

@st.cache_resource(show_spinner=False)
def load_bert_model():
    if not os.path.exists(os.path.join(MODEL3_DIR, "config.json")):
        return None, None, None
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    tok = AutoTokenizer.from_pretrained(MODEL3_DIR)
    mdl = AutoModelForSequenceClassification.from_pretrained(MODEL3_DIR)
    mdl.eval()
    with open(os.path.join(MODEL3_DIR, "label_map.json")) as f:
        label_map = {int(k): v for k, v in json.load(f).items()}
    return tok, mdl, label_map

@st.cache_data(show_spinner=False)
def load_train_embeddings():
    if not os.path.exists(TRAIN_EMB_NPZ) or not os.path.exists(TRAIN_META_CSV):
        return None, None
    data = np.load(TRAIN_EMB_NPZ, allow_pickle=True)
    emb_matrix = data["embeddings"]
    meta = pd.read_csv(TRAIN_META_CSV)
    return emb_matrix, meta

# ── Prediction helpers ────────────────────────────────────────────────────────

def predict_bow(cs_vec, cs_clf, text):
    X = cs_vec.transform([text])
    pred = cs_clf.predict(X)[0]
    proba = cs_clf.predict_proba(X)[0]
    return pred, dict(zip(cs_clf.classes_, proba))

def predict_tfidf(tfidf_vec, clf, text, has_proba=True):
    X = tfidf_vec.transform([text])
    pred = clf.predict(X)[0]
    if has_proba and hasattr(clf, "predict_proba"):
        proba = clf.predict_proba(X)[0]
        return pred, dict(zip(clf.classes_, proba))
    return pred, {}

def predict_bert(bert_tok, bert_mdl, label_map, text):
    import torch
    inputs = bert_tok(text, truncation=True, padding=True,
                      max_length=512, return_tensors="pt")
    with torch.no_grad():
        logits = bert_mdl(**inputs).logits
    probs = torch.softmax(logits, dim=1)[0].numpy()
    pred_idx = int(np.argmax(probs))
    pred = label_map[pred_idx]
    return pred, {label_map[i]: float(probs[i]) for i in range(len(probs))}

def get_doc_top_features_bow(cs_vec, text, n=15):
    vocab_inv = {v: k for k, v in cs_vec.vocabulary_.items()}
    X = cs_vec.transform([text]).toarray()[0]
    pairs = [(vocab_inv[i], X[i]) for i in X.nonzero()[0]]
    pairs.sort(key=lambda x: x[1], reverse=True)
    return pairs[:n]

def get_doc_top_features_tfidf(tfidf_vec, text, n=15):
    feat = tfidf_vec.get_feature_names_out()
    X = tfidf_vec.transform([text]).toarray()[0]
    pairs = [(feat[i], X[i]) for i in X.nonzero()[0]]
    pairs.sort(key=lambda x: x[1], reverse=True)
    return pairs[:n]

def get_lr_class_top_words(vec, clf, cancer_type, n=12):
    feat = vec.get_feature_names_out()
    idx = list(clf.classes_).index(cancer_type)
    coefs = clf.coef_[idx]
    top_idx = np.argsort(coefs)[-n:][::-1]
    return [(feat[i], float(coefs[i])) for i in top_idx]

def bar_chart(pairs, title, color="#4C72B0", height=350):
    words = [p[0] for p in reversed(pairs)]
    vals  = [p[1] for p in reversed(pairs)]
    fig = go.Figure(go.Bar(x=vals, y=words, orientation="h",
                           marker_color=color))
    fig.update_layout(title=title, height=height,
                      margin=dict(l=5, r=5, t=40, b=5),
                      xaxis_title="Score")
    return fig

def top_proba_chart(proba_dict, true_label, title="Top 5 Predictions", n=5):
    top = sorted(proba_dict.items(), key=lambda x: x[1], reverse=True)[:n]
    labels = [f"{ct} ({CANCER_NAMES.get(ct,ct)[:25]})" for ct, _ in top]
    vals   = [v for _, v in top]
    colors = ["#2ecc71" if t[0] == true_label else "#3498db" for t in top]
    fig = go.Figure(go.Bar(x=vals, y=labels, orientation="h",
                           marker_color=colors))
    fig.update_layout(title=title, height=280,
                      margin=dict(l=5, r=5, t=40, b=5),
                      xaxis_title="Probability",
                      xaxis=dict(range=[0, 1]))
    return fig

def result_box(accuracy, f1_weighted, label="Method"):
    st.markdown(f"""
    <div style="background:#1e3a5f;padding:16px;border-radius:8px;color:white;margin-top:8px">
    <b>{label} — Published Results</b><br>
    <span style="font-size:1.6em;font-weight:bold">{accuracy}</span> accuracy &nbsp;|&nbsp;
    F1 Weighted: <b>{f1_weighted}</b>
    </div>
    """, unsafe_allow_html=True)


# ── Page layout ───────────────────────────────────────────────────────────────

st.set_page_config(page_title="MEDREx — Method Explorer", layout="wide",
                   initial_sidebar_state="expanded")

st.title("MEDREx — Cancer Type Classification: Method Explorer")
st.markdown(
    "*CSC-590 Masters Project · CSUDH Spring 2026 · "
    "Sponsor: Cedars-Sinai AI Campus · Project #32*"
)
st.markdown("---")

# ── Load everything ───────────────────────────────────────────────────────────
with st.spinner("Loading sample reports and models..."):
    samples = load_sample_reports()
    cs_vec, cs_clf, cs_tok = load_cedars_models()
    tfidf_vec, m1_clf, m2_clf = load_tfidf_models()

models_ready = cs_clf is not None and m1_clf is not None
bert_ready   = os.path.exists(os.path.join(MODEL3_DIR, "config.json"))
emb_ready    = os.path.exists(TRAIN_EMB_NPZ)

if not models_ready:
    st.warning("**Saved models not found.** Run `python demo/save_models.py` first (takes ~5 min).")
if not samples:
    st.warning("**Test meta CSV not found.** Ensure Method 4 has been run to generate embeddings.")

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Sample Report")
    if samples:
        selected_ct = st.selectbox(
            "Cancer Type",
            options=list(samples.keys()),
            format_func=lambda x: f"{x}  —  {CANCER_NAMES.get(x,'')[:35]}"
        )
        sample = samples[selected_ct]
        report_text = sample["text"]
        patient_id  = sample["patient_id"]

        st.markdown(f"**Patient:** `{patient_id}`")
        st.markdown(f"**True Label:** `{selected_ct}`")
        st.markdown(f"*{CANCER_NAMES.get(selected_ct, '')}*")
        st.markdown(f"**Word count:** {len(report_text.split()):,}")
        st.markdown("---")
        st.markdown("**Report preview:**")
        st.text(report_text[:400] + " ...")
    else:
        report_text = ""
        selected_ct = "BRCA"
        st.info("No sample reports loaded.")

    st.markdown("---")
    st.markdown("### Results at a Glance")
    results_data = {
        "Cedars-Sinai (BoW+LR)": "95.31%",
        "Method 1 (TF-IDF+LR)":  "96.36%",
        "Method 2 (TF-IDF+SVM)": "97.37%",
        "Method 3 (BERT)":       "94.57%",
        "Method 4 (Emb k-NN)":   "90.13%",
        "Method 5 (RAG+LLM)":    "64.00%",
    }
    for method, acc in results_data.items():
        st.markdown(f"- **{method}**: {acc}")


# ── Tabs ───────────────────────────────────────────────────────────────────────
tab_cs, tab_m1, tab_m2, tab_m3, tab_m4, tab_m5, tab_m6, tab_cmp = st.tabs([
    "Cedars-Sinai (BoW + LR)",
    "Method 1: TF-IDF + LR",
    "Method 2: TF-IDF + SVM",
    "Method 3: BERT Fine-tuned",
    "Method 4: Embedding k-NN",
    "Method 5: RAG + LLM",
    "Method 6: RAG + OpenAI",
    "Results Comparison",
])


# ════════════════════════════════════════════════════════════════════════════════
# TAB 0 — Cedars-Sinai Baseline
# ════════════════════════════════════════════════════════════════════════════════
with tab_cs:
    st.header("Cedars-Sinai Baseline — Bag of Words + Logistic Regression")
    st.markdown("*This is the reference approach from our project sponsor. We reproduced it and then improved upon it.*")

    col_how, col_code = st.columns(2, gap="large")

    with col_how:
        st.subheader("How It Works")
        st.markdown("""
**Step 1 — Tokenize & Clean**
Each report is split into individual words using NLTK's `word_tokenize`.
Common English words ("the", "a", "is") and punctuation are removed via a stopword list.
Only alphabetic tokens longer than 1 character survive.

**Step 2 — Bag of Words (CountVectorizer)**
The vocabulary of ~23,800 unique medical terms is built from the training set.
Each report becomes a vector: *"How many times does each vocabulary word appear?"*
Example: `[0, 0, 3, 0, 1, 0, ...]` — "renal" appeared 3 times, "ductal" 1 time.

**Step 3 — Logistic Regression**
A linear model learns a *weight* for every word in the vocabulary, for each cancer type.
High weight for "glioblastoma" → predicts GBM.
High weight for "ductal carcinoma" → predicts BRCA.
At prediction time: multiply word counts × weights, pick the highest-scoring class.

**Key Insight:** Raw word *counts* are the features — no weighting by document frequency.
        """)

    with col_code:
        st.subheader("Key Code — Exact Parameters")
        st.code("""
import nltk
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

stopwords = nltk.corpus.stopwords.words("english")

# Custom tokenizer — removes stopwords, punctuation, single chars
def tokenizer(text):
    tokens = word_tokenize(text.lower())
    tokens = [t for t in tokens if len(t) > 1]        # drop single chars
    tokens = [t for t in tokens if t not in stopwords] # drop stopwords
    tokens = [t for t in tokens if t.isalpha()]        # drop punctuation/numbers
    return tokens

# Bag of Words — raw word counts, 23,818 vocabulary
word_vectorizer = CountVectorizer(
    tokenizer=tokenizer,
    token_pattern=None,   # must be None when using custom tokenizer
    lowercase=False       # tokenizer already lowercases
)

# Train on 50% of data (non-stratified split, random.seed(0))
word_vectorizer.fit(train_texts)
X_train = word_vectorizer.transform(train_texts)

# Logistic Regression — no class_weight balancing
clf = LogisticRegression(
    random_state=0,
    max_iter=200     # Cedars-Sinai used default 200 iterations
)
clf.fit(X_train, train_labels)
""", language="python")

        st.info("""**Split:** 50% train / 20% val / 30% test
**Vocabulary:** 23,818 unique medical terms
**Train samples:** ~4,761 reports""")

    with st.expander("Step 5 — Pipeline + GridSearch + Final Test Evaluation (how 95.31% was measured)"):
        st.markdown("The published 95.31% accuracy comes from this final step — a Pipeline with GridSearchCV that selects the best `max_iter`, then evaluates **once** on the held-out test set.")
        st.code("""
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report

# Pipeline bundles vectorizer + classifier — prevents data leakage
# CountVectorizer is ONLY fit on training data, even inside GridSearch
pipe = Pipeline([
    ("bow", CountVectorizer(tokenizer=tokenizer, token_pattern=None, lowercase=False)),
    ("clf", LogisticRegression(random_state=0))
])

# Custom CV: use the FIXED train/val split (not random k-fold)
X_all     = train_texts + val_texts   # 4,761 + 1,905 = 6,666
y_all     = train_labels + val_labels
train_idx = list(range(len(train_texts)))               # indices 0–4760
val_idx   = list(range(len(train_texts), len(X_all)))  # indices 4761–6665
custom_cv = [(train_idx, val_idx)]   # one "fold" = our fixed split

# GridSearch tries max_iter=[200, 500], picks whichever val accuracy is higher
grid = GridSearchCV(
    estimator=pipe,
    param_grid={"clf__max_iter": [200, 500]},
    scoring="accuracy",
    cv=custom_cv,
    verbose=2,
    refit=True   # re-trains best config on full train+val data before test
)
grid.fit(X_all, y_all)
# Best params: {'clf__max_iter': 200}  (200 was already sufficient)
# Best val accuracy: ~95.0%

# ── Final test evaluation — done ONLY ONCE, never used during tuning ──────
best_model = grid.best_estimator_
y_pred     = best_model.predict(test_texts)   # 2,857 unseen reports
test_acc   = accuracy_score(test_labels, y_pred)
print(f"Test accuracy: {test_acc*100:.2f}%")  # → 95.31%
print(classification_report(test_labels, y_pred))
""", language="python")

    with st.expander("Parameter Reference — CountVectorizer + LogisticRegression explained"):
        col_a, col_b = st.columns(2, gap="large")
        with col_a:
            st.markdown("#### CountVectorizer Parameters")
            st.markdown("""
| Parameter | Value | Why |
|-----------|-------|-----|
| `tokenizer` | custom NLTK fn | Removes stopwords, punct, single chars before counting |
| `token_pattern` | `None` | Must be None when using a custom tokenizer (disables regex) |
| `lowercase` | `False` | Tokenizer already lowercases — doing it twice changes nothing |
| `stop_words` | `None` | Stopword removal is done inside the custom tokenizer, not here |

**The custom tokenizer does 3 things in order:**
1. `word_tokenize(text.lower())` — NLTK splits on punctuation smarter than `.split()`
2. Remove words where `len(t) <= 1` — drops single letters like "a", "s"
3. Remove NLTK English stopwords — "the", "is", "at", "which", "on" (179 words)
4. Keep only `t.isalpha()` — drops numbers ("T2"), punctuation, mixed tokens

**Result:** "The 2.5cm mass..." → `["mass", "invasive", "carcinoma"]`
            """)
        with col_b:
            st.markdown("#### LogisticRegression Parameters (Cedars-Sinai)")
            st.markdown("""
| Parameter | Value | Why |
|-----------|-------|-----|
| `random_state` | `0` | Seeds the optimizer — `0` is Cedars-Sinai's original choice |
| `max_iter` | `200` | Max optimization steps — sufficient for this simpler BoW model |
| `C` | `1.0` (default) | Regularization — default, not tuned by Cedars-Sinai |
| `class_weight` | `None` (default) | No balancing — Cedars-Sinai did NOT use this |
| `solver` | `lbfgs` (default) | Default for multi-class — not changed |

**Why `random_state=0` here but `random_state=42` in our methods?**

They mean exactly the same thing mechanically — both fix the random seed for reproducibility.
- `0` is Cedars-Sinai's original code — we must keep it to reproduce their exact results
- `42` is our convention — a data science tradition (reference to *Hitchhiker's Guide*)
- Any fixed integer gives reproducible results; the value itself does not affect accuracy

**Key difference:** Cedars-Sinai has **no** `class_weight="balanced"` — this is why their model struggles more on rare types like UCS. Our Method 1 added it and improved rare-class F1.
            """)

    st.markdown("---")
    st.subheader("The Full Story — How BoW + LR Works (Plain English)")
    st.caption("Read this first, then look at the live example below to see each step in action.")

    s1, s2 = st.columns(2, gap="large")

    with s1:
        st.markdown("#### Step 1 — Clean the Words")
        st.markdown("""
Every report is cleaned before counting. Useless words are removed:
        """)
        st.code("""
Original report:
  "The tumor is in the left breast tissue"

Remove "the", "is", "in"  →  appear in EVERY report, not useful
Remove numbers, punctuation, single letters

What's left:
  ["tumor", "left", "breast", "tissue"]
""", language="text")
        st.caption("Think of it like highlighting only the important words in a textbook and ignoring 'the', 'a', 'and'.")

        st.markdown("#### Step 2 — Build the Dictionary (23,818 words)")
        st.markdown("""
Read all 4,761 training reports. Every unique word that survives cleaning goes into a master list:
        """)
        st.code("""
Word    1:  "tumor"
Word    2:  "breast"
Word    3:  "glioblastoma"
Word    4:  "ductal"
Word    5:  "kidney"
  ...
Word 23818: "papillary"

This list NEVER changes after training.
""", language="text")

        st.markdown("#### Step 3 — Turn Each Report into a Row of Numbers")
        st.markdown("Count how many times each of the 23,818 words appears in each report:")
        st.code("""
               tumor  breast  glioblastoma  ductal  kidney  ...
Report 1 BRCA    3      7           0         4       0
Report 2 GBM     2      0           5         0       0
Report 3 KIRC    1      0           0         0       8
Report 4 BRCA    4      9           0         6       0
...
(4,761 rows × 23,818 columns)

Most cells = 0  (a report uses ~500 words out of 23,818)
This is called a SPARSE matrix — mostly empty.
""", language="text")
        st.caption("Each row is one report converted to pure numbers. The model never reads actual words — only these numbers.")

    with s2:
        st.markdown("#### Step 4 — The Model Learns Weights")
        st.markdown("Logistic Regression learns a **weight** for every word × every cancer type:")
        st.code("""
              BRCA    GBM    KIRC    LUAD  ... (32 types)
tumor         0.12   0.08    0.09    0.11
breast        5.14   0.01    0.00    0.02  ← STRONGLY → BRCA
glioblastoma  0.00   6.82    0.00    0.00  ← STRONGLY → GBM
ductal        2.79   0.00    0.01    0.05  ← points → BRCA
kidney        0.01   0.00    4.91    0.02  ← STRONGLY → KIRC
...
(23,818 rows of weights — one per word per cancer type)
""", language="text")
        st.caption("The model starts with all weights = 0, then adjusts them 200 times by looking at training reports. Like a student studying flashcards 200 times.")

        st.markdown("#### Step 5 — Predict a New Report")
        st.markdown("A new report arrives (never seen before). Watch what happens:")
        st.code("""
New report text:
  "ductal carcinoma of the left breast.
   Sentinel lymph node biopsy..."

Step A — Count words:
  breast=6, ductal=3, sentinel=2,
  glioblastoma=0, kidney=0 ...

Step B — Multiply count × weight for each cancer type:
  BRCA score = (6×5.14)+(3×2.79)+(2×2.03)+... = 48.7  ← HIGHEST
  GBM  score = (6×0.01)+(3×0.00)+(2×0.01)+... =  0.3
  KIRC score = (6×0.00)+(3×0.01)+(2×0.00)+... =  0.1

Step C — Pick the highest score:
  → Answer: BRCA (Breast Cancer) ✓
""", language="text")

        st.markdown("#### Why Does It Work? (95.31% accuracy)")
        st.markdown("""
Because **doctors use very consistent language**:
- Every breast cancer report says: *ductal, axillary, lumpectomy, sentinel*
- Every brain cancer report says: *glioblastoma, frontal lobe, necrosis*
- Every kidney cancer report says: *renal cell, clear cell, nephrectomy*

The model just learns these patterns from 4,761 examples.
**The words do the heavy lifting — the math is just adding and multiplying.**
        """)

        st.info("""**Summary in one line:**
Read report → count 23,818 words → multiply by learned weights → pick the cancer type with the highest score → 95.31% correct""")

    st.markdown("---")
    st.subheader("Example in Action — Pre-loaded Report")
    st.caption("A real TCGA pathology report is pre-loaded from the sidebar. Watch how the Bag-of-Words model classifies it, step by step.")

    if models_ready and report_text:
        pred, proba = predict_bow(cs_vec, cs_clf, report_text)
        correct = pred == selected_ct

        # ── Step 1: Show the report ───────────────────────────────────────────
        st.markdown("**Step 1 — The Model Reads the Report**")
        st.markdown("*This is the actual pathology report text the model will classify:*")
        st.markdown(f"""
        <div style="background:#2b2b2b;padding:14px;border-radius:8px;
                    font-family:monospace;font-size:0.85em;color:#ccc;
                    border-left:4px solid #6c757d">
        {" ".join(report_text.split()[:80])} <span style="color:#888">... ({len(report_text.split())} words total)</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("&nbsp;")
        # ── Step 2 + 3 side by side ───────────────────────────────────────────
        col_chart1, col_chart2, col_result = st.columns([1.5, 1.5, 1])

        with col_chart1:
            st.markdown("**Step 2 — Strongest Clue Words Found**")
            st.caption("These words appeared the most in this report. The model counts every word and looks up which cancer types those words are associated with.")
            top_words = get_doc_top_features_bow(cs_vec, report_text, n=12)
            if top_words:
                st.plotly_chart(
                    bar_chart(top_words,
                              "Word count — longer bar = word appeared more often",
                              color="#6c757d", height=320),
                    use_container_width=True
                )

        with col_chart2:
            st.markdown("**Step 3 — How Confident Is the Model?**")
            st.caption("After checking all the clue words, the model scores every possible cancer type. The green bar is its final answer — the cancer type it is most confident about.")
            if proba:
                st.plotly_chart(
                    top_proba_chart(proba, selected_ct,
                                    "Confidence per cancer type — green = final answer"),
                    use_container_width=True
                )

        with col_result:
            st.markdown("**Step 4 — Final Answer**")
            st.markdown(f"""
            <div style="background:{'#d4edda' if correct else '#f8d7da'};
                        padding:16px;border-radius:8px;margin-bottom:12px">
            <b style="font-size:1.6em;color:{'#155724' if correct else '#721c24'}">{pred}</b><br>
            <i>{CANCER_NAMES.get(pred,'')}</i><br><br>
            Real answer: <b>{selected_ct}</b><br><br>
            <b style="font-size:1.1em">{"✓ Correct!" if correct else "✗ Wrong"}</b>
            </div>
            """, unsafe_allow_html=True)
            if proba:
                conf = proba.get(pred, 0)
                st.metric("Model confidence", f"{conf*100:.1f}%",
                          help="How sure the model is about its answer. 100% = completely certain, 50% = a coin flip.")
    else:
        st.info("Run `python demo/save_models.py` to enable predictions.")

    st.markdown("---")
    result_box("95.31%", "94.8%", "Cedars-Sinai BoW + LR")


# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 — Method 1: TF-IDF + LR
# ════════════════════════════════════════════════════════════════════════════════
with tab_m1:
    st.header("Method 1 — TF-IDF + Logistic Regression")
    st.markdown("*Our first improvement over the Cedars-Sinai baseline: replace raw counts with TF-IDF scores.*")

    col_how, col_code = st.columns(2, gap="large")

    with col_how:
        st.subheader("How It Works")
        st.markdown("""
**TF-IDF vs. Bag of Words — What Changed?**

Bag of Words counts how many times a word appears. But "carcinoma" appears in *every* cancer report — it's not useful for distinguishing cancer *type*.

TF-IDF fixes this with two components:

- **TF (Term Frequency):** How often the word appears in *this* document.
  With `sublinear_tf=True`: uses `1 + log(tf)` to dampen very frequent words.

- **IDF (Inverse Document Frequency):** How rare the word is across *all* documents.
  Rare words like "glioblastoma" (only in GBM reports) get a high IDF score.
  Common words like "the" or "carcinoma" get a low IDF score.

**Result:** A word that appears often in this report AND rarely across all reports → high TF-IDF score → very informative.

**Bigrams (ngram_range=(1,2)):** Also captures 2-word phrases like "clear cell", "renal cell", "ductal carcinoma" — crucial for cancer typing.

**Logistic Regression with `class_weight=balanced`:** Gives equal importance to rare cancer types (UCS has only 57 reports) and common ones (BRCA has 1,096 reports).
        """)

    with col_code:
        st.subheader("Key Code — Exact Parameters")
        st.code("""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Stratified split: 70% train, 30% test (same random state for all methods)
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels,
    test_size=0.3,
    random_state=42,
    stratify=labels      # ensures each class is proportionally represented
)

# TF-IDF Vectorizer
tfidf = TfidfVectorizer(
    max_features=15000,  # keep top 15,000 terms by document frequency
    ngram_range=(1, 2),  # unigrams + bigrams ("clear cell", "renal cell")
    sublinear_tf=True,   # log-dampen term frequency (prevents "carcinoma" dominating)
    min_df=2             # ignore terms appearing in < 2 documents
)
X_train_tfidf = tfidf.fit_transform(X_train)   # 6,666 × 15,000 sparse matrix
X_test_tfidf  = tfidf.transform(X_test)        # never fit on test data!

# Logistic Regression
clf = LogisticRegression(
    C=1.0,               # regularisation strength (higher = less regularised)
    max_iter=1000,       # more iterations needed for multi-class
    class_weight="balanced",  # penalises misclassifying rare cancer types more
    solver="lbfgs",      # efficient for multi-class with dense classes
    random_state=42
)
clf.fit(X_train_tfidf, y_train)
""", language="python")

        st.info("""**Vocabulary:** 15,000 features (unigrams + bigrams)
**Train:** 6,666 reports | **Test:** 2,857 reports
**Improvement over BoW:** +1.05% accuracy (96.36% vs 95.31%)""")

    with st.expander("Final Test Evaluation — how 96.36% was measured"):
        st.markdown("One-shot test evaluation after training. Same stratified split used across all methods for a fair comparison.")
        st.code("""
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report

# Stratified split: preserves cancer type proportions in both sets
# random_state=42 ensures identical split for all methods (fair comparison)
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels,
    test_size=0.3,       # 30% test  = 2,857 reports
    random_state=42,     # fixed seed → reproducible
    stratify=labels      # each class proportionally in train/test
)

# Fit TF-IDF ONLY on training data — never on test
tfidf  = TfidfVectorizer(max_features=15000, ngram_range=(1,2),
                         sublinear_tf=True, min_df=2)
X_train_tfidf = tfidf.fit_transform(X_train)   # fit + transform
X_test_tfidf  = tfidf.transform(X_test)        # transform only (no fit!)

# Train Logistic Regression
clf = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced",
                         solver="lbfgs", random_state=42)
clf.fit(X_train_tfidf, y_train)

# ── Final test evaluation — done once ────────────────────────────────────
y_pred     = clf.predict(X_test_tfidf)
test_acc   = accuracy_score(y_test, y_pred)         # → 96.36%
f1_w       = f1_score(y_test, y_pred, average="weighted")  # → 96.1%
f1_m       = f1_score(y_test, y_pred, average="macro")     # → 93.8%
print(classification_report(y_test, y_pred))
# UCS: precision=0.93, recall=0.88, f1=0.91  (rare class, class_weight helped)
""", language="python")

    with st.expander("Parameter Reference — TfidfVectorizer + LogisticRegression explained"):
        col_a, col_b = st.columns(2, gap="large")
        with col_a:
            st.markdown("#### TfidfVectorizer Parameters")
            st.markdown("""
| Parameter | Value | Why |
|-----------|-------|-----|
| `max_features` | `15000` | Keep top 15K terms by document frequency — removes noise, controls matrix size (6,666 × 15,000) |
| `ngram_range` | `(1, 2)` | Capture both single words AND 2-word phrases — "clear cell", "renal cell", "ductal carcinoma" are crucial for cancer typing |
| `sublinear_tf` | `True` | Use `1 + log(tf)` instead of raw count — prevents "carcinoma" (appears 50× per report) from dominating over "glioblastoma" (appears 3×) |
| `min_df` | `2` | Ignore terms in fewer than 2 documents — filters typos, patient names, hospital-specific codes |

**TF-IDF formula per term:**
```
score = TF × IDF
TF   = 1 + log(count)          ← sublinear_tf=True
IDF  = log(N / df) + 1         ← N=total docs, df=docs containing term
```
"glioblastoma": rare → high IDF → high score → strongly predicts GBM
"carcinoma": common → low IDF → low score → not useful for distinguishing types
            """)
        with col_b:
            st.markdown("#### LogisticRegression Parameters (Method 1)")
            st.markdown("""
| Parameter | Value | Why |
|-----------|-------|-----|
| `C` | `1.0` | Regularization strength — inverse of penalty. `1.0` is neutral. Too low → underfits, too high → overfits. Tested and 1.0 works well |
| `max_iter` | `1000` | Max optimizer steps. Cedars used 200 — sufficient for BoW. TF-IDF with `class_weight=balanced` is harder to converge, needs 1000 |
| `class_weight` | `"balanced"` | Auto-weights classes: `total / (n_classes × class_count)`. UCS (57 reports) gets 10× higher penalty than BRCA (1,096 reports). Without this, model ignores UCS entirely |
| `solver` | `"lbfgs"` | Optimization algorithm. L-BFGS is best for multi-class with dense weight matrices. Faster than `saga` for this dataset size |
| `random_state` | `42` | Fixes the random seed → same result every run. `42` is a data science convention (*Hitchhiker's Guide*). Cedars used `0` — different number, same purpose |

**Why `42` and not `0`?**
- `0` = Cedars-Sinai's original seed (we keep it to reproduce theirs exactly)
- `42` = our methods' seed (signals "arbitrary fixed seed for reproducibility")
- The value does not affect accuracy — only ensures the same result every run
            """)

    st.markdown("---")
    st.subheader("The Full Story — How TF-IDF + LR Works (Plain English)")
    st.caption("Why is TF-IDF better than raw word counts? Read this before looking at the live example.")

    f1, f2 = st.columns(2, gap="large")

    with f1:
        st.markdown("#### The Problem with Just Counting Words")
        st.markdown("""
Cedars-Sinai counted how many times each word appeared — and it worked great (95%).
But counting has one weakness. Imagine both words appear 5 times in a report:
        """)
        st.code("""
"the"           → appears in EVERY single report → not useful
"adenocarcinoma"→ rare word → huge clue when it shows up

Raw counting treats both the same.
TF-IDF fixes this.
""", language="text")
        st.caption("Think of it like grading answers on a quiz: answering 'the' is worth 0 points because everyone writes it. Answering 'adenocarcinoma' is worth 10 points — only the student who actually studied would write that.")

        st.markdown("#### Part 1: TF = Term Frequency (How often in THIS report?)")
        st.markdown("With `sublinear_tf=True`, we use **log scale** instead of raw count:")
        st.code("""
TF = 1 + log(count)

Raw count → TF score
    0     →   0.0
    1     →   1.0
    5     →   2.6
   50     →   4.9
  500     →   6.2

The 500th time a word appears is NOT 500x more
important than the 1st time.
Log flattens it out — a reasonable score, not an explosion.
""", language="text")
        st.caption("Like volume on a speaker: going from 0 to 1 is a big jump, but going from 99 to 100 feels almost the same.")

        st.markdown("#### Part 2: IDF = Inverse Document Frequency (How rare across ALL reports?)")
        st.code("""
IDF = log( total_reports / reports_containing_this_word )

Word               Appears in...      IDF
"the"              9,000 of 9,000  →  log(1)   = 0.0  (useless)
"tumor"            4,500 of 9,000  →  log(2)   = 0.69
"adenocarcinoma"     450 of 9,000  →  log(20)  = 3.0
"pheochromocytoma"    45 of 9,000  →  log(200) = 5.3  (rare = huge clue)
""", language="text")
        st.caption("Common words that appear in every report get IDF = 0 and are automatically ignored. Rare cancer-specific words get high IDF and are amplified.")

        st.markdown("#### Final TF-IDF Score = TF × IDF")
        st.code("""
Word               Count  TF    IDF   TF-IDF
"the"                50   4.9   0.0   → 0.0   (zeroed out!)
"tumor"              10   3.3   0.69  → 2.3
"adenocarcinoma"      3   2.1   3.0   → 6.3   (low count, huge score)
"pheochromocytoma"    1   1.0   5.3   → 5.3   (appeared once, still powerful)
""", language="text")
        st.markdown("""
**The magic:** "pheochromocytoma" appeared only once, but its TF-IDF score is **5.3** — because it's extremely rare across all reports, so even one occurrence is a massive clue for the cancer type PCPG.

"The" appeared 50 times and scores **0** — pure noise, automatically ignored.
        """)

    with f2:
        st.markdown("#### Bigrams — Reading Two Words at a Time")
        st.markdown("Parameter: `ngram_range=(1, 2)`")
        st.markdown("Cedars-Sinai read one word at a time. Method 1 also reads **pairs of words**:")
        st.code("""
Report: "invasive ductal carcinoma"

Unigrams (1 word):  "invasive", "ductal", "carcinoma"
Bigrams  (2 words): "invasive ductal", "ductal carcinoma"

Why does this matter?

Just single words  →  What we miss
"small" + "cell"   →  "small cell" (specific lung cancer subtype!)
"non" + "hodgkin"  →  "non hodgkin" (a critical diagnosis!)
"squamous" + "cell"→  "squamous cell" (head/neck, lung, cervix)
"clear" + "cell"   →  "clear cell" (kidney cancer!)
""", language="text")
        st.caption("Two-word phrases carry diagnostic meaning that individual words alone cannot. A doctor reading 'clear' knows nothing. Reading 'clear cell' immediately thinks kidney cancer.")

        st.markdown("#### class_weight='balanced' — Helping the Rare Cancers")
        st.code("""
Dataset is very unequal:
  BRCA (Breast)  →  1,096 reports  (common)
  LUAD (Lung)    →    515 reports
  UCS  (Uterine) →     57 reports  (rare!)
  CHOL (Bile duct)→    36 reports  (rare!)

Without balancing: model ignores rare cancers.
It's "profitable" to always guess BRCA.

With class_weight="balanced":
  Mistake on UCS  = 19× more costly than BRCA
  Mistake on CHOL = 30× more costly than BRCA

The model MUST learn the rare cancers even with
few examples — because the penalty for missing
them is so high.

Cedars-Sinai did NOT use this.
Method 1 does → better rare-class F1 scores.
""", language="text")
        st.caption("Think of it like a teacher grading a test: if you miss an easy question worth 1 point it's ok, but if you miss the rare hard question worth 19 points — that's a serious problem.")

        st.markdown("#### Step-by-Step: Predicting a New Report")
        st.code("""
New report: "Sections show invasive ductal carcinoma,
             grade 3, ER positive, HER2 negative"

Step 1 — TF-IDF scoring (15,000 features):
  "invasive ductal"  → TF-IDF = 8.4  (rare bigram, big clue)
  "ductal carcinoma" → TF-IDF = 7.9
  "er positive"      → TF-IDF = 6.1
  "her2 negative"    → TF-IDF = 5.8
  "carcinoma"        → TF-IDF = 5.2
  (most of the 15,000 features = 0 for this report)

Step 2 — Compute 32 cancer scores:
  BRCA = (8.4×0.92)+(7.9×0.87)+(6.1×0.88)+... = 42.1
  LUAD = (8.4×0.02)+(7.9×0.01)+(6.1×0.00)+... =  2.8
  STAD = (8.4×0.05)+(7.9×0.04)+(6.1×0.00)+... =  3.1

Step 3 — Softmax → Probabilities:
  BRCA  →  91.2%   ← winner
  STAD  →   3.1%
  LUAD  →   2.4%
  ...

Step 4 — Predict: BRCA (Breast Cancer) ✓
""", language="text")

        st.markdown("#### Method 1 vs Cedars-Sinai — What Actually Changed")
        st.markdown("""
| | Cedars-Sinai | Method 1 |
|---|---|---|
| Features | Raw word counts | TF-IDF scores |
| Vocabulary | 23,818 words | 15,000 words+bigrams |
| Common words | Counted equally | Zeroed out by IDF |
| Two-word phrases | ✗ No | ✓ Yes (bigrams) |
| Class balancing | ✗ No | ✓ Yes (balanced) |
| Train/test split | Non-stratified 70/30 | Stratified 70/30 |

**The core math is identical** — weighted sum → scores → pick highest.
The difference is *what numbers go in*: raw counts vs. smart TF-IDF scores
that already filter out noise and amplify rare diagnostic words.
        """)
        st.info("**Result:** 96.36% vs 95.31% — +1.05% improvement just by changing how we count words.")

    st.markdown("---")
    st.subheader("Example in Action — Pre-loaded Report")
    st.caption("A real TCGA pathology report is pre-loaded from the sidebar. Watch how TF-IDF finds the most important clue words and the model picks the cancer type.")

    if models_ready and report_text:
        pred, proba = predict_tfidf(tfidf_vec, m1_clf, report_text, has_proba=True)
        correct = pred == selected_ct

        # ── Step 1 ────────────────────────────────────────────────────────────
        st.markdown("**Step 1 — The Model Reads the Report**")
        st.caption("*This is the actual pathology report text going into the model:*")
        st.markdown(f"""
        <div style="background:#2b2b2b;padding:14px;border-radius:8px;
                    font-family:monospace;font-size:0.85em;color:#ccc;
                    border-left:4px solid #4C72B0">
        {" ".join(report_text.split()[:80])} <span style="color:#888">... ({len(report_text.split())} words total)</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("&nbsp;")
        col_chart1, col_chart2, col_result = st.columns([1.5, 1.5, 1])

        with col_chart1:
            st.markdown("**Step 2 — Most Important Clue Words**")
            st.caption("TF-IDF gives each word a score based on two things: (1) how often it appears in THIS report, and (2) how rare it is across ALL reports. A word that appears a lot here but rarely elsewhere gets a high score — it's a strong clue for this cancer type.")
            top_tfidf = get_doc_top_features_tfidf(tfidf_vec, report_text, n=12)
            if top_tfidf:
                st.plotly_chart(
                    bar_chart(top_tfidf,
                              "TF-IDF score — longer bar = stronger clue word for this report",
                              color="#4C72B0", height=320),
                    use_container_width=True
                )

        with col_chart2:
            st.markdown("**Step 3 — How Confident Is the Model?**")
            st.caption("The model checks all clue words against what it learned during training. It asks: 'Which cancer type is most consistent with these words?' The green bar is its final answer.")
            if proba:
                st.plotly_chart(
                    top_proba_chart(proba, selected_ct,
                                    "Confidence per cancer type — green = final answer"),
                    use_container_width=True
                )

        with col_result:
            st.markdown("**Step 4 — Final Answer**")
            st.markdown(f"""
            <div style="background:{'#d4edda' if correct else '#f8d7da'};
                        padding:16px;border-radius:8px;margin-bottom:12px">
            <b style="font-size:1.6em;color:{'#155724' if correct else '#721c24'}">{pred}</b><br>
            <i>{CANCER_NAMES.get(pred,'')}</i><br><br>
            Real answer: <b>{selected_ct}</b><br><br>
            <b style="font-size:1.1em">{"✓ Correct!" if correct else "✗ Wrong"}</b>
            </div>
            """, unsafe_allow_html=True)
            if proba:
                st.metric("Model confidence", f"{proba.get(pred,0)*100:.1f}%",
                          help="How sure the model is. 100% = completely certain, 50% = a coin flip.")

            st.markdown("**Words the model uses for this cancer type:**")
            st.caption("These are words the model learned during training that most strongly point to this cancer type in general — not just in this report.")
            top_class_words = get_lr_class_top_words(tfidf_vec, m1_clf, pred, n=6)
            for word, coef in top_class_words:
                st.markdown(f"- `{word}` (strength: {coef:.2f})")
    else:
        st.info("Run `python demo/save_models.py` to enable predictions.")

    st.markdown("---")
    result_box("96.36%", "96.1%", "Method 1 — TF-IDF + Logistic Regression")


# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 — Method 2: TF-IDF + SVM
# ════════════════════════════════════════════════════════════════════════════════
with tab_m2:
    st.header("Method 2 — TF-IDF + SVM (LinearSVC)")
    st.markdown("*Best classical ML result: replacing Logistic Regression with a Support Vector Machine.*")

    col_how, col_code = st.columns(2, gap="large")

    with col_how:
        st.subheader("How It Works")
        st.markdown("""
**Same TF-IDF vectorizer as Method 1 — only the classifier changes.**

**Logistic Regression (Method 1)** finds the *most probable* class boundary.

**SVM (LinearSVC)** finds the *maximum-margin* boundary — it tries to put the largest possible gap between classes. This makes it more robust on unseen data.

**Why LinearSVC specifically?**
- TF-IDF produces a very large sparse matrix: 6,666 reports × 15,000 features
- `LinearSVC` works directly in the feature space → very fast (O(n×d))
- `SVC(kernel='rbf')` would compute all pairwise distances → O(n²) → impossibly slow

**One-vs-Rest strategy:** For 32 cancer types, LinearSVC trains 32 binary classifiers:
- "BRCA vs all other cancers"
- "GBM vs all other cancers"
- etc.

At prediction time, the class with the highest decision function score wins.

**Why SVM beats LR on this task?**
The maximum-margin criterion handles the high-dimensional, sparse TF-IDF space better, especially for borderline cases between similar cancers (LUAD vs LUSC, COAD vs READ).
        """)

    with col_code:
        st.subheader("Key Code — Exact Parameters")
        st.code("""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

# Same TF-IDF as Method 1 (shared vectorizer — no retraining needed)
tfidf = TfidfVectorizer(
    max_features=15000,
    ngram_range=(1, 2),
    sublinear_tf=True,
    min_df=2
)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf  = tfidf.transform(X_test)

# LinearSVC — Support Vector Machine for text classification
clf = LinearSVC(
    C=1.0,                     # regularisation (controls margin vs misclassification)
    max_iter=2000,             # more iterations than LR (SVM needs more convergence steps)
    class_weight="balanced",   # same as Method 1 — handles rare cancer types
    random_state=42
)
# Multi-class: automatically uses One-vs-Rest (32 binary classifiers)
clf.fit(X_train_tfidf, y_train)

# Prediction — no probabilities (LinearSVC uses decision_function instead)
pred = clf.predict(X_test_tfidf)
scores = clf.decision_function(X_test_tfidf)  # shape: (n_samples, 32)
""", language="python")

        st.success("""**Best Classical ML Result: 97.37% accuracy**
**+2.06% over Cedars-Sinai baseline (95.31%)**
Note: LinearSVC does not output probabilities — only decision scores.""")

    with st.expander("Final Test Evaluation — how 97.37% was measured"):
        st.markdown("Same TF-IDF matrix and test split as Method 1. Only the classifier changes — this is a controlled comparison.")
        st.code("""
# Identical train/test split as Method 1 (random_state=42, stratify=labels)
# Identical TF-IDF vectorizer — both methods share the same feature matrix
# Only the classifier differs → clean A/B comparison

from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, classification_report

# LinearSVC — maximum-margin classifier (no probabilities, only decision scores)
clf_svm = LinearSVC(
    C=1.0,                     # regularisation (same as Method 1 LR)
    max_iter=2000,             # more iterations needed for SVM convergence
    class_weight="balanced",   # same as Method 1
    random_state=42
)
clf_svm.fit(X_train_tfidf, y_train)   # trains 32 binary "one-vs-rest" classifiers

# ── Final test evaluation ────────────────────────────────────────────────
y_pred   = clf_svm.predict(X_test_tfidf)       # argmax of decision_function
test_acc = accuracy_score(y_test, y_pred)       # → 97.37%
f1_w     = f1_score(y_test, y_pred, average="weighted")  # → 97.1%
print(classification_report(y_test, y_pred))

# Key comparison vs Method 1 (LR):
# UCS:  LR  → f1=0.91  |  SVM → f1=0.786  (SVM slightly worse on UCS)
# LUAD: LR  → f1=0.93  |  SVM → f1=0.95   (SVM better on common classes)
# Overall: SVM wins by 1.01% — max-margin beats max-likelihood on text
""", language="python")

    with st.expander("Parameter Reference — TfidfVectorizer + LinearSVC explained"):
        col_a, col_b = st.columns(2, gap="large")
        with col_a:
            st.markdown("#### TfidfVectorizer Parameters (same as Method 1)")
            st.markdown("""
| Parameter | Value | Why |
|-----------|-------|-----|
| `max_features` | `15000` | Top 15K terms — identical to Method 1. Both methods share the **same fitted vectorizer** object. Training it twice would give the same result |
| `ngram_range` | `(1, 2)` | Unigrams + bigrams — same as Method 1. The TF-IDF matrix is shared; only the classifier changes |
| `sublinear_tf` | `True` | Log-dampening — same as Method 1 |
| `min_df` | `2` | Filter rare terms — same as Method 1 |

**This is a controlled experiment:**
Same data → same split → same TF-IDF features → different classifier.
Any accuracy difference between Method 1 (LR) and Method 2 (SVM) is **100% due to the classifier**, not the features.
            """)
        with col_b:
            st.markdown("#### LinearSVC Parameters (Method 2)")
            st.markdown("""
| Parameter | Value | Why |
|-----------|-------|-----|
| `C` | `1.0` | Same regularization as LR's `C=1.0` — enables a fair comparison. Controls the trade-off between maximizing the margin and allowing misclassifications |
| `max_iter` | `2000` | SVM needs more iterations than LR to converge. LR minimizes log-loss (smooth), SVM minimizes hinge loss (non-smooth) → needs more steps |
| `class_weight` | `"balanced"` | Same as Method 1 — gives higher penalty for misclassifying rare types (UCS, KICH). Without this, SVM ignores rare classes |
| `random_state` | `42` | Seeds the solver's initial state → reproducible results |

**LR vs SVM — the core difference:**

| | Logistic Regression | LinearSVC |
|--|--|--|
| Goal | Maximize probability of correct class | Maximize margin between classes |
| Output | Probabilities (0–1) | Decision scores (any range) |
| Strength | Well-calibrated confidence | Better generalization on sparse text |
| On our data | 96.36% | **97.37%** |

**Why no `predict_proba` in SVM?**
LinearSVC does not naturally produce probabilities — only a decision score (how far a point is from the margin boundary). To get probabilities you'd need `CalibratedClassifierCV` (wraps SVM with Platt scaling), but this adds overhead. For classification accuracy, raw decision scores are sufficient.
            """)

    st.markdown("---")
    st.subheader("The Full Story — How TF-IDF + SVM Works (Plain English)")
    st.caption("Method 2 uses the exact same TF-IDF features as Method 1 — only the classifier changes. This section explains what SVM does differently and why it scores higher.")

    g1, g2 = st.columns(2, gap="large")

    with g1:
        st.markdown("#### Everything from Method 1 Still Applies")
        st.markdown("""
Method 2 uses the **exact same TF-IDF features** as Method 1:
- Same 15,000 features (unigrams + bigrams)
- Same log-dampening (`sublinear_tf=True`)
- Same IDF zeroing out common words
- Same `class_weight='balanced'` for rare cancers
- Same 70/30 stratified split

**Only one thing changes: the classifier.**

The real question is: *What does a Support Vector Machine do differently from Logistic Regression — and why does it get 97.37% instead of 96.36%?*
        """)

        st.markdown("#### Drawing a Line Between Cancer Types")
        st.markdown("Both LR and SVM draw a **boundary** that separates cancer types. The difference is *how* they draw it.")
        st.code("""
Logistic Regression (Method 1): "Find the Most Likely Answer"
──────────────────────────────────────────────────────────────
  BRCA zone       |  LUAD zone
  Report A (BRCA) ●  |
  Report B (BRCA)   ● |
                    boundary
                           ● Report C (LUAD)
                              ● Report D (LUAD)

LR places the boundary where the PROBABILITY of both
classes is exactly 50/50. It looks at ALL training
examples and picks the "most probable" line.

SVM (Method 2): "Find the Widest Gap"
──────────────────────────────────────────────────────────────
  BRCA zone         GAP        LUAD zone
  Report A (BRCA) ●
  Report B (BRCA)   ●
                      |===margin===|
                          boundary
                                ● Report C (LUAD)
                                   ● Report D (LUAD)

SVM places the boundary in the MIDDLE of the biggest
empty space between the two classes — the maximum margin.
""", language="text")
        st.caption("LR: 'Where do most cars want to go?' | SVM: 'Where is the biggest empty lane between the two directions of traffic?'")

        st.markdown("#### Support Vectors — The Only Points That Matter")
        st.code("""
All 6,666 training reports...

LR uses: ALL 6,666 reports — each one shifts the boundary.

SVM uses: Only the reports CLOSEST to the boundary.
          These are called "support vectors."
          Maybe 50–200 reports out of 6,666.
          The other 6,400+ reports are IRRELEVANT
          once training is done.

Why this matters:
  Borderline cases (ambiguous reports that look like
  two different cancer types) DRIVE the boundary.

  LR might "average" them away with all the clear-cut
  cases and place the boundary slightly wrong.

  SVM focuses entirely on getting those borderline
  cases right — maximizing the gap at the edge.
""", language="text")
        st.caption("Think of it like deciding where to put a fence between two properties. You only look at the edge — not what's in the middle of each yard.")

        st.markdown("#### Decision Score vs Probability — The Big Difference")
        st.code("""
Method 1 (LR) gives a PROBABILITY:
  BRCA → 91.2%
  LUAD →  3.1%
  STAD →  2.4%
  ...   (all add up to 100%)

"91.2% of the time, a report like this is BRCA."

Method 2 (SVM) gives a DECISION SCORE:
  BRCA → +4.8   (4.8 units inside BRCA's side)
  LUAD → -2.1   (wrong side of boundary)
  STAD → -1.7   (wrong side of boundary)
  ...   (any number, does NOT add up to 100%)

"This report is 4.8 units away from the BRCA
 boundary, on the BRCA side."

Think of it like a ruler measuring how far
you are from a fence:
  +4.8 = you're 4.8 meters inside BRCA's yard
  -2.1 = you're 2.1 meters inside LUAD's yard
""", language="text")
        st.info("SVM cannot give probabilities naturally. Decision scores are enough for classification — we just pick the highest one.")

    with g2:
        st.markdown("#### One-vs-Rest: 32 Separate Battles")
        st.markdown("SVM is a binary classifier (A vs B). For 32 cancer types, it runs **32 separate battles**:")
        st.code("""
Round  1: BRCA  vs [all other 31 cancers] → one SVM
Round  2: LUAD  vs [all other 31 cancers] → one SVM
Round  3: GBM   vs [all other 31 cancers] → one SVM
...
Round 32: UVM   vs [all other 31 cancers] → one SVM

For a new report, ALL 32 SVMs score it.
The cancer type with the HIGHEST decision score wins.

Example — new report about kidney cancer:
  KIRC SVM: +5.8  ← winner (clear cell kidney)
  KIRP SVM: +1.2  (similar but lower)
  GBM  SVM: -4.1  (clearly not brain cancer)
  BRCA SVM: -3.7  (clearly not breast cancer)
  ...

→ Predict: KIRC ✓
""", language="text")

        st.markdown("#### The C Parameter — How Strict Is the Margin?")
        st.code("""
LinearSVC(C=1.0, ...)

C controls the trade-off between margin width and mistakes:

Low C (e.g., 0.01):
  → Very WIDE margin (maximally conservative)
  → Allows some training examples on the wrong side
  → Risk: underfits — too simple, misses patterns

High C (e.g., 100):
  → Very NARROW margin (must classify everything right)
  → Forces boundary close to the most extreme points
  → Risk: overfits — memorizes training data

C = 1.0:
  → Balanced — reasonable margin, a few mistakes OK
  → Default, tested, works well on this dataset

Think of C like how strict a teacher is:
  Low C  = "close enough is fine" (too lenient)
  High C = "must be perfect"      (too strict)
  C=1.0  = reasonable balance     ← we use this
""", language="text")

        st.markdown("#### Why SVM Needs max_iter=2000 (vs LR's 1000)")
        st.code("""
Logistic Regression minimizes LOG-LOSS:
  → Smooth curve — easy to find the bottom
  → 1,000 steps is enough to converge

LinearSVC minimizes HINGE LOSS:
  → Non-smooth — has a "kink" at the margin edge
  → Harder to find the exact minimum
  → Needs 2,000 steps to fully converge

It's like the difference between:
  LR:  Rolling a ball down a smooth hill
       → finds the bottom quickly
  SVM: Rolling a ball down a bumpy hillside
       → needs more time to settle
""", language="text")

        st.markdown("#### Why SVM Beats LR on This Data (+1.01%)")
        st.code("""
Hardest cases: LUAD vs LUSC (both lung cancers)

LUAD report: "adenocarcinoma... acinar pattern... TTF-1 positive"
LUSC report: "squamous cell... keratinization... p40 positive"
Ambiguous:   "carcinoma... lung... mixed pattern" ← hard!

LR approach:
  Averages across ALL LUAD examples.
  The ambiguous case gets "pulled" toward the
  majority — might land on the wrong side.

SVM approach:
  Focuses ONLY on the borderline cases.
  Maximizes the gap at the hard-to-classify edge.
  The ambiguous case has the best chance of
  landing on the correct side.

SVM wins when:
  ✓ Features are high-dimensional (15,000 here)
  ✓ Data is sparse (most features = 0)
  ✓ Classes have overlapping regions (LUAD/LUSC)

LR wins when:
  ✓ You need calibrated probabilities
  ✓ Data is smaller / lower-dimensional
  ✓ You want to easily interpret feature weights
""", language="text")

        st.markdown("#### Method 1 vs Method 2 — One Change, 1% Better")
        st.markdown("""
| | Method 1 (LR) | Method 2 (SVM) |
|---|---|---|
| TF-IDF features | ✓ Same | ✓ Same |
| Bigrams | ✓ Same | ✓ Same |
| class_weight=balanced | ✓ Same | ✓ Same |
| How it decides | Max probability | Max margin |
| Output | Probabilities (0–100%) | Decision scores (any range) |
| What drives boundary | All 6,666 examples | Only the closest ~100 |
| **Accuracy** | 96.36% | **97.37%** |

The +1.01% improvement comes entirely from the **maximum-margin criterion** — SVM draws a better boundary in high-dimensional sparse text spaces.
        """)
        st.success("**Key insight:** Same data, same features, different decision rule → 1% better. This is a controlled experiment proving SVM's boundary criterion is superior for this task.")

    st.markdown("---")
    st.subheader("Example in Action — Pre-loaded Report")
    st.caption("Same TF-IDF features as Method 1 — only the decision-making step changes. Watch how the SVM draws a boundary instead of calculating probabilities.")

    if models_ready and report_text:
        pred, _ = predict_tfidf(tfidf_vec, m2_clf, report_text, has_proba=False)
        correct = pred == selected_ct

        # ── Step 1 ────────────────────────────────────────────────────────────
        st.markdown("**Step 1 — The Model Reads the Report**")
        st.caption("*Same report, same TF-IDF scoring as Method 1 — the difference is in Step 3:*")
        st.markdown(f"""
        <div style="background:#2b2b2b;padding:14px;border-radius:8px;
                    font-family:monospace;font-size:0.85em;color:#ccc;
                    border-left:4px solid #2ca02c">
        {" ".join(report_text.split()[:80])} <span style="color:#888">... ({len(report_text.split())} words total)</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("&nbsp;")
        col_chart1, col_chart2, col_result = st.columns([1.5, 1.5, 1])

        with col_chart1:
            st.markdown("**Step 2 — Most Important Clue Words**")
            st.caption("Identical to Method 1 — TF-IDF scores the same words the same way. The SVM uses these exact same numbers; it just makes the final decision differently.")
            top_tfidf = get_doc_top_features_tfidf(tfidf_vec, report_text, n=12)
            if top_tfidf:
                st.plotly_chart(
                    bar_chart(top_tfidf,
                              "TF-IDF score — longer bar = stronger clue word",
                              color="#2ca02c", height=320),
                    use_container_width=True
                )

        with col_chart2:
            st.markdown("**Step 3 — How Strongly Does the SVM Lean?**")
            st.caption("Unlike Method 1 (which gives probabilities like '53% BRCA'), SVM gives a 'decision score' — think of it as how far the report is from the boundary line between cancer types. A higher score = stronger lean toward that type.")
            if m2_clf is not None and report_text:
                X = tfidf_vec.transform([report_text])
                dec = m2_clf.decision_function(X)[0]
                dec_dict = dict(zip(m2_clf.classes_, dec))
                top_dec = sorted(dec_dict.items(), key=lambda x: x[1], reverse=True)[:5]
                dec_labels = [f"{ct}  ({CANCER_NAMES.get(ct,'')[:18]})" for ct, _ in top_dec]
                dec_vals   = [v for _, v in top_dec]
                colors = ["#2ecc71" if t[0] == selected_ct else "#27ae60" for t in top_dec]
                fig = go.Figure(go.Bar(x=dec_vals, y=dec_labels, orientation="h",
                                       marker_color=colors))
                fig.update_layout(
                    title="How strongly the SVM leans toward each type — longer = stronger",
                    height=280, margin=dict(l=5,r=5,t=50,b=5),
                    xaxis_title="Decision strength (not a probability)")
                st.plotly_chart(fig, use_container_width=True)

        with col_result:
            st.markdown("**Step 4 — Final Answer**")
            st.markdown(f"""
            <div style="background:{'#d4edda' if correct else '#f8d7da'};
                        padding:16px;border-radius:8px;margin-bottom:12px">
            <b style="font-size:1.6em;color:{'#155724' if correct else '#721c24'}">{pred}</b><br>
            <i>{CANCER_NAMES.get(pred,'')}</i><br><br>
            Real answer: <b>{selected_ct}</b><br><br>
            <b style="font-size:1.1em">{"✓ Correct!" if correct else "✗ Wrong"}</b>
            </div>
            """, unsafe_allow_html=True)
            st.caption("SVM picks whichever cancer type has the highest decision score — no probabilities.")
            feat = tfidf_vec.get_feature_names_out()
            idx  = list(m2_clf.classes_).index(pred)
            top_idx = np.argsort(m2_clf.coef_[idx])[-6:][::-1]
            st.markdown("**Words that most point to this cancer type (learned from training):**")
            for i in top_idx:
                st.markdown(f"- `{feat[i]}` (strength: {m2_clf.coef_[idx][i]:.2f})")
    else:
        st.info("Run `python demo/save_models.py` to enable predictions.")

    st.markdown("---")
    result_box("97.37%", "97.1%", "Method 2 — TF-IDF + SVM (LinearSVC)")


# ════════════════════════════════════════════════════════════════════════════════
# TAB 3 — Method 3: BERT Fine-tuned
# ════════════════════════════════════════════════════════════════════════════════
with tab_m3:
    st.header("Method 3 — Bio_ClinicalBERT Fine-tuned")
    st.markdown("*A 110 million parameter transformer pre-trained on biomedical text, fine-tuned on our cancer dataset.*")

    col_how, col_code = st.columns(2, gap="large")

    with col_how:
        st.subheader("How It Works")
        st.markdown("""
**What is BERT?**
BERT (Bidirectional Encoder Representations from Transformers) is a deep neural network with 12 transformer layers and 110 million parameters.

Unlike TF-IDF which treats words independently, BERT reads the entire sentence bidirectionally — understanding context:
- "The patient has *clear* lungs" → clear = healthy
- "The specimen shows *clear* cell morphology" → clear = cancer subtype

**Bio_ClinicalBERT — Specialised Pre-training**
Standard BERT is pre-trained on Wikipedia + BookCorpus.
`emilyalsentzer/Bio_ClinicalBERT` was additionally pre-trained on:
- PubMed biomedical abstracts
- MIMIC-III clinical notes (ICU records)

This gives it deep understanding of medical terminology before we even start fine-tuning.

**Fine-tuning on TCGA**
We add a classification head (one linear layer) on top of BERT and train on our 9,523 reports:
- Input: 512 tokens (report text)
- Hidden: 768-dimensional contextual embeddings
- Output: 32 cancer type logits → softmax → probabilities

**Training:** 4 epochs on Google Colab (A100 GPU), ~25 min
**Inference:** CPU only — ~18 seconds per report
        """)

    with col_code:
        st.subheader("Key Code — Fine-tuning on Colab")
        st.code("""
from transformers import (AutoTokenizer,
                          AutoModelForSequenceClassification,
                          TrainingArguments, Trainer)

# Load pre-trained Bio_ClinicalBERT
MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"
tokenizer  = AutoTokenizer.from_pretrained(MODEL_NAME)
model      = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=32        # one output node per cancer type
)

# Tokenize reports (max 512 tokens — BERT's context window)
encodings = tokenizer(
    texts, truncation=True, padding=True, max_length=512
)

# Training configuration
args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=4,          # 4 full passes over the training data
    per_device_train_batch_size=16,
    learning_rate=2e-5,          # very low — fine-tuning, not training from scratch
    weight_decay=0.01,           # L2 regularisation
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy"
)
trainer = Trainer(model=model, args=args,
                  train_dataset=train_ds, eval_dataset=val_ds)
trainer.train()     # ~25 min on A100 GPU
""", language="python")

        st.markdown("""**Training Results (Google Colab):**
| Epoch | Train Loss | Val Accuracy |
|-------|-----------|--------------|
| 1 | 0.6823 | 89.4% |
| 2 | 0.1876 | 93.8% |
| 3 | 0.0891 | 94.3% |
| 4 | 0.0512 | **94.6%** |
""")

    with st.expander("Local Evaluation Script — how 94.57% was measured (method3_evaluate_local.py)"):
        st.markdown("After fine-tuning on Colab, the model is downloaded and evaluated locally. CPU inference — no GPU needed.")
        st.code("""
import torch, json
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report

# Load fine-tuned model from local folder (downloaded from Google Drive)
MODEL_DIR = "demo/ProjectCode/method3_best_model/"
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model     = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()
with open(MODEL_DIR + "label_map.json") as f:
    label_map = {int(k): v for k, v in json.load(f).items()}

# Same stratified test split as Methods 1 & 2 (random_state=42)
_, X_test, _, y_test = train_test_split(
    texts, labels, test_size=0.3, random_state=42, stratify=labels
)  # → 2,857 test reports

# Tokenize all 2,857 test reports at once (512 tokens each)
test_enc = tokenizer(X_test.tolist(), truncation=True,
                     padding=True, max_length=512, return_tensors="pt")

# Run inference in batches of 16 (179 batches × 16 = 2,857 reports)
# CPU inference: ~52 min total (~17s per batch)
all_preds = []
with torch.no_grad():
    for batch in DataLoader(CancerDataset(test_enc, y_test), batch_size=16):
        logits = model(input_ids=batch["input_ids"],
                       attention_mask=batch["attention_mask"]).logits
        preds  = torch.argmax(logits, dim=1).cpu().numpy()
        all_preds.extend(preds)

accuracy_score(y_test_ids, all_preds)          # → 94.57%
f1_score(y_test_ids, all_preds, average="weighted")  # → 94.3%
# UCS: f1=0.000 — BERT completely fails on this 39-sample class
""", language="python")

    st.markdown("---")
    st.subheader("Live Prediction — BERT inference on selected report")

    if bert_ready and report_text:
        if st.button("Run BERT Prediction (takes ~15 seconds on CPU)"):
            with st.spinner("Running Bio_ClinicalBERT inference..."):
                bert_tok, bert_mdl, bert_labels = load_bert_model()
                t0 = time.time()
                pred, proba = predict_bert(bert_tok, bert_mdl, bert_labels, report_text)
                elapsed = time.time() - t0
            correct = pred == selected_ct
            col_r, col_c = st.columns(2)
            with col_r:
                st.markdown(f"""
                <div style="background:{'#d4edda' if correct else '#f8d7da'};
                            padding:16px;border-radius:8px">
                <b style="font-size:1.4em;color:{'#155724' if correct else '#721c24'}">{pred}</b><br>
                {CANCER_NAMES.get(pred,'')}<br><br>
                True label: <b>{selected_ct}</b><br>
                {"✓ Correct" if correct else "✗ Wrong"}<br>
                <small>Inference time: {elapsed:.1f}s</small>
                </div>
                """, unsafe_allow_html=True)
                conf = proba.get(pred, 0)
                st.metric("Confidence", f"{conf*100:.1f}%")
            with col_c:
                if proba:
                    st.plotly_chart(
                        top_proba_chart(proba, selected_ct, "BERT Top 5 Probabilities"),
                        use_container_width=True
                    )
    elif not bert_ready:
        st.warning("BERT model not found at `demo/ProjectCode/method3_best_model/`. "
                   "Download from Google Drive first.")
    else:
        st.info("Select a report from the sidebar, then click 'Run BERT Prediction'.")

    st.markdown("---")
    st.subheader("Key Finding: BERT vs SVM on Rare Classes")
    st.markdown("""
Despite having 110M parameters vs SVM's simple weight vectors, **BERT scores lower overall (94.57% vs 97.37%)**.

The critical failure is **UCS (Uterine Carcinosarcoma)** — only 57 total reports (39 training):

| Method | UCS F1 | Notes |
|--------|--------|-------|
| TF-IDF + SVM | **0.786** | Maximum-margin SVM handles class imbalance well |
| Bio_ClinicalBERT | **0.000** | BERT completely fails — predicts UCEC for all UCS cases |

**Why?** BERT has millions of parameters to learn from only 39 UCS examples — it over-generalises and merges UCS with the visually similar UCEC (Uterine Corpus). SVM's simpler model avoids this overfitting.
    """)
    result_box("94.57%", "94.3%", "Method 3 — Bio_ClinicalBERT Fine-tuned")


# ════════════════════════════════════════════════════════════════════════════════
# TAB 4 — Method 4: Embedding k-NN
# ════════════════════════════════════════════════════════════════════════════════
with tab_m4:
    st.header("Method 4 — Sentence Embeddings + k-NN (Cosine Similarity)")
    st.markdown("*A completely different approach: find the training report most similar to the new report.*")

    col_how, col_code = st.columns(2, gap="large")

    with col_how:
        st.subheader("How It Works")
        st.markdown("""
**No classifier is trained.** Instead, every report is converted to a 384-dimensional embedding vector that captures its *semantic meaning*.

**Step 1 — Embed all training reports**
Using `sentence-transformers/all-MiniLM-L6-v2` (a 33M parameter model):
- Each report → a 384-dimensional vector
- Similar reports → vectors that point in similar directions
- "Renal cell carcinoma... clear cell morphology" and "Kidney: clear cell RCC" → very close vectors

This is pre-computed and cached (takes ~8 min once, then instant).

**Step 2 — Embed the new report**
Same model, same process → 384-dimensional vector.

**Step 3 — Cosine Similarity Search**
Find the k most similar training reports:
```
similarity = dot(A, B) / (|A| × |B|)
```
Cosine similarity = 1 if vectors point in same direction (identical meaning).

**Step 4 — Majority Vote**
The cancer types of the k nearest neighbors vote.
With k=5 (best found by cross-validation): 3 BRCA, 1 LUAD, 1 BRCA → predict BRCA.

**Why 90% instead of 97%?**
The embedding model is not fine-tuned on cancer reports — it's a general-purpose model.
TF-IDF exploits cancer-specific vocabulary directly; embeddings compress meaning into 384 dimensions and lose some fine-grained cancer type signals.
        """)

    with col_code:
        st.subheader("Key Code — Exact Parameters")
        st.code("""
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

# Pre-trained sentence embedding model (33M parameters, no fine-tuning)
embed_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

# Embed all training reports → 6,666 × 384 matrix
# batch_size=64 for speed; show_progress_bar for long runs
train_embeddings = embed_model.encode(
    train_texts,
    batch_size=64,
    show_progress_bar=True
)  # shape: (6666, 384)  — pre-computed and cached as .npz

# Embed new (test) report
new_embedding = embed_model.encode([new_text])  # shape: (1, 384)

# k-Nearest Neighbour classifier with cosine similarity
knn = KNeighborsClassifier(
    n_neighbors=5,       # best k found by cross-validation (tried 1,3,5,7,9,11)
    metric="cosine",     # cosine similarity (not Euclidean distance)
    algorithm="brute"    # brute force — needed for cosine on dense vectors
)
knn.fit(train_embeddings, train_labels)
pred = knn.predict(new_embedding)

# Or compute similarity directly:
sims = np.dot(train_embeddings, new_embedding.T).flatten()
top5_idx = np.argsort(sims)[-5:][::-1]
""", language="python")

    with st.expander("Cross-validation + Final Test Evaluation — how 90.13% was measured"):
        st.markdown("k is chosen by cross-validation on the validation set, then the best k is used for one final test evaluation.")
        st.code("""
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# Embeddings pre-computed (SentenceTransformer, ~8 min, cached as .npz)
# train_emb: (6,666, 384)  |  val_emb: (N_val, 384)  |  test_emb: (2857, 384)

# ── Step 1: Cross-validate to find best k ───────────────────────────────
best_k, best_val_acc = 1, 0.0
for k in [1, 3, 5, 7, 9, 11]:
    knn = KNeighborsClassifier(n_neighbors=k, metric="cosine", algorithm="brute")
    knn.fit(train_emb, y_train)
    val_acc = accuracy_score(y_val, knn.predict(val_emb))
    print(f"  k={k}  →  val accuracy = {val_acc*100:.2f}%")
    if val_acc > best_val_acc:
        best_k, best_val_acc = k, val_acc
# Results: k=1→88.2%, k=3→89.5%, k=5→89.9%, k=7→89.6%, k=9→89.3%, k=11→89.0%
# Best k = 5

# ── Step 2: Final test evaluation with best k ────────────────────────────
knn_final = KNeighborsClassifier(
    n_neighbors=5,       # best found above
    metric="cosine",     # cosine similarity (not Euclidean)
    algorithm="brute"    # required for cosine metric
)
knn_final.fit(train_emb, y_train)
y_pred   = knn_final.predict(test_emb)
test_acc = accuracy_score(y_test, y_pred)     # → 90.13%
print(classification_report(y_test, y_pred))
# Note: 7% below TF-IDF+SVM — embeddings compress meaning but lose
# fine-grained cancer vocabulary signals that TF-IDF exploits directly
""", language="python")

        st.info("""**Best k:** 5 (tested k = 1, 3, 5, 7, 9, 11)
**Embedding dim:** 384
**Precompute time:** ~8 min (cached to .npz)
**Prediction time:** instant (dot product)""")

    st.markdown("---")
    st.subheader("Live Example — Nearest Neighbor Search")

    if emb_ready and report_text:
        if st.button("Find Nearest Neighbors in Training Set"):
            with st.spinner("Loading embeddings and computing similarity..."):
                emb_matrix, train_meta = load_train_embeddings()
                from sentence_transformers import SentenceTransformer
                emb_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
                new_emb = emb_model.encode([report_text[:512]])
                norms_train = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
                norms_new   = np.linalg.norm(new_emb,    axis=1, keepdims=True)
                sims = np.dot(emb_matrix / norms_train, (new_emb / norms_new).T).flatten()
                top_idx = np.argsort(sims)[-5:][::-1]

            st.markdown("**Top 5 Most Similar Training Reports:**")
            for rank, i in enumerate(top_idx, 1):
                row = train_meta.iloc[i]
                sim = sims[i]
                match = "✓" if row["cancer_type"] == selected_ct else " "
                st.markdown(f"""
                **#{rank}** — Similarity: `{sim:.4f}` {match}
                Patient: `{row['patient_id']}` | Cancer: **{row['cancer_type']}** — {CANCER_NAMES.get(row['cancer_type'],'')}
                *"{str(row['text'])[:150]}..."*
                """)
            st.markdown("---")
            votes = train_meta.iloc[top_idx]["cancer_type"].value_counts()
            st.markdown(f"**Majority vote → Prediction: `{votes.index[0]}`**")
    elif not emb_ready:
        st.warning("Embeddings cache not found. Run `method4_embedding_knn.py` first.")
    else:
        st.info("Select a report and click 'Find Nearest Neighbors'.")

    st.markdown("---")
    result_box("90.13%", "88.7%", "Method 4 — Sentence Embeddings + k-NN (k=5)")


# ════════════════════════════════════════════════════════════════════════════════
# TAB 5 — Method 5: RAG + LLM
# ════════════════════════════════════════════════════════════════════════════════
with tab_m5:
    st.header("Method 5 — RAG + LLM (Retrieval-Augmented Generation)")
    st.markdown("*The most innovative approach: use the same embeddings to find examples, then let a Small Language Model classify.*")

    col_how, col_code = st.columns(2, gap="large")

    with col_how:
        st.subheader("How It Works")
        st.markdown("""
**RAG = Retrieval + Generation**

Two separate components work together:

**Retrieval (same as Method 4):**
1. Embed the new pathology report using MiniLM
2. Find the K=1 most similar training report via cosine similarity
3. That training report is a "known example" with a confirmed cancer label

**Generation (LLM):**
4. Build a prompt with the known example + the new report
5. Send to a local Small Language Model (llama3.2:3B via Ollama)
6. The SLM completes the prompt with just the cancer code

**Why a local SLM (not ChatGPT / Claude)?**
TCGA pathology reports contain real patient data — HIPAA regulations prohibit sending them to external cloud APIs. The 3B parameter Llama model runs entirely on local CPU (no GPU needed, no data leaves the machine).

**HIPAA-safe pipeline:**
- All data stays on local machine
- No API calls to external services
- Ollama runs the model via a local HTTP server (port 11434)

**Why only 64% accuracy?**
- Llama 3.2 (3B parameters) has no medical fine-tuning
- Reports are truncated to 80 words (context window limitation)
- Even with examples, the model sometimes gives verbose answers instead of a code
- RAG works much better with a larger, medically-trained LLM

**Key Thesis Finding:** Complexity ≠ Accuracy. A simple SVM with TF-IDF beats a language model + retrieval system by 33 percentage points.
        """)

    with col_code:
        st.subheader("Key Code — RAG Pipeline")
        st.code("""
import requests, numpy as np
from sentence_transformers import SentenceTransformer

# Reuse Method 4's pre-computed embeddings (no re-encoding needed)
train_emb  = np.load("method4_train_embeddings_...npz")["embeddings"]
train_meta = pd.read_csv("method4_train_meta_...csv")  # has cancer_type + text

OLLAMA_MODEL = "llama3.2:latest"   # 3B parameter model, runs on CPU
K_EXAMPLES   = 1                   # 1 retrieved example (more = slower)
MAX_WORDS_EX = 60                  # truncate example to 60 words
MAX_WORDS_NEW= 80                  # truncate test report to 80 words

# Step 1: Retrieve the most similar training example
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
new_emb = embed_model.encode([new_text])
sims = cosine_similarity(new_emb, train_emb)[0]   # (1, n_train)
top_idx = int(np.argmax(sims))
example = train_meta.iloc[top_idx]

# Step 2: Build prompt — ends with "Cancer type code:" to force one-word answer
codes = sorted(train_meta["cancer_type"].unique())
prompt = (
    f"Classify the cancer type. Reply with ONLY the code:\\n"
    f"{', '.join(codes)}\\n\\n"
    f"KNOWN EXAMPLE ({example['cancer_type']}):\\n"
    f"{' '.join(example['text'].split()[:60])}\\n\\n"
    f"CLASSIFY THIS:\\n"
    f"{' '.join(new_text.split()[:80])}\\n\\n"
    f"Cancer type code:"      # <-- forces single-word completion
)

# Step 3: Call local Ollama (no data leaves the machine)
response = requests.post("http://localhost:11434/api/generate",
    json={"model": OLLAMA_MODEL, "prompt": prompt,
          "stream": False, "temperature": 0.0,
          "options": {"num_predict": 10, "num_ctx": 512}},
    timeout=300)
prediction = response.json()["response"].strip().split()[0]
""", language="python")

        st.warning("""**To run Method 5 live:** Ollama must be running locally.
Install: https://ollama.com | Then: `ollama pull llama3.2`""")

    st.markdown("---")
    st.subheader("Example Prompt Built for This Report")

    if report_text:
        words_ex = "(nearest training example would appear here after embedding lookup)"
        codes_preview = "ACC, BLCA, BRCA, CESC, CHOL, COAD, ..."
        truncated_report = " ".join(report_text.split()[:80])
        example_prompt = f"""Classify the cancer type. Reply with ONLY the code from this list:
{codes_preview}

KNOWN EXAMPLE (??):
(the most similar training report — 60 words)

CLASSIFY THIS:
{truncated_report}

Cancer type code:"""
        st.code(example_prompt, language="text")
        st.caption("The SLM completes this prompt by writing just the cancer code (e.g., 'BRCA').")

    with st.expander("Evaluation Design — how 64% was measured (method5_rag_llm.py)"):
        st.markdown("Only 50 test reports sampled (full 2,857 would take ~24 hours at 30s/report on CPU).")
        st.code("""
import random, requests, numpy as np
from sklearn.metrics import accuracy_score
from sentence_transformers import SentenceTransformer

# Parameters (tuned to balance speed vs accuracy)
SAMPLE_SIZE    = 50          # 50 random test reports (full set too slow)
K_EXAMPLES     = 1           # 1 retrieved training example per query
MAX_WORDS_EX   = 60          # truncate example text to 60 words
MAX_WORDS_NEW  = 80          # truncate test report to 80 words
OLLAMA_MODEL   = "llama3.2:latest"   # 3B parameter model, local CPU

# Pre-compute similarity matrix for all 50 samples at once (fast batch)
sample_idx    = random.sample(range(len(test_texts)), SAMPLE_SIZE)
sample_emb    = embed_model.encode([test_texts[i] for i in sample_idx])
sim_matrix    = cosine_similarity(sample_emb, train_emb)  # (50, 6,666)

# ── Main evaluation loop ─────────────────────────────────────────────────
preds, true_labels = [], []
for i, idx in enumerate(sample_idx):
    true_label = y_test[idx]
    top_train  = int(np.argmax(sim_matrix[i]))
    example    = train_meta.iloc[top_train]

    # Truncate texts to fit in 512-token context window
    ex_words  = " ".join(example["text"].split()[:MAX_WORDS_EX])
    new_words = " ".join(test_texts[idx].split()[:MAX_WORDS_NEW])

    # Prompt ends with "Cancer type code:" → forces single-word completion
    prompt = (f"Classify the cancer type. Reply with ONLY the code:\\n{codes}\\n\\n"
              f"KNOWN EXAMPLE ({example['cancer_type']}):\\n{ex_words}\\n\\n"
              f"CLASSIFY THIS:\\n{new_words}\\n\\nCancer type code:")

    resp = requests.post("http://localhost:11434/api/generate",
        json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False,
              "options": {"num_predict": 10, "num_ctx": 512, "temperature": 0}},
        timeout=300).json()["response"].strip()

    pred = parse_response(resp, valid_codes)  # regex search for valid code
    preds.append(pred); true_labels.append(true_label)

accuracy_score(true_labels, preds)   # → 0.64  (64.0%)
# Unknown (no valid code found): ~8/50 → these count as wrong predictions
""", language="python")

    st.markdown("---")
    result_box("64.00%", "62.3%", "Method 5 — RAG + LLM (llama3.2:3B, local CPU)")
    st.markdown("""
> **Thesis finding:** The RAG+SLM pipeline is 33% below the best classical method.
> This demonstrates that for structured classification tasks with consistent medical vocabulary,
> fine-tuned classical ML can outperform generative AI approaches — complexity ≠ accuracy.
    """)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 6 — Method 6: RAG + OpenAI
# ════════════════════════════════════════════════════════════════════════════════
with tab_m6:
    st.header("Method 6 — RAG + LLM (OpenAI gpt-4o-mini)")
    st.markdown("*Same RAG pipeline as Method 5 — but swaps the local Llama model for OpenAI's gpt-4o-mini via API.*")

    col_how, col_code = st.columns(2, gap="large")

    with col_how:
        st.subheader("How It Works")
        st.markdown("""
**Same retrieval as Method 5 — different generation model.**

**Retrieval (identical to Methods 4 & 5):**
1. Reuse Method 4's MiniLM embeddings (no re-embedding needed)
2. Compute cosine similarity: test report vs all 6,666 training reports
3. Retrieve the top-1 most similar training report (known label)

**Generation (OpenAI API):**
4. Build the same prompt: known example + test report
5. Send to **gpt-4o-mini** via `openai.chat.completions.create()`
6. Parse the response for a valid cancer type code

**Why OpenAI instead of Ollama?**
TCGA data is **de-identified public data** — no real patient identifiers.
Sending it to OpenAI's API does not violate HIPAA.
This lets us test a much more capable model without a local GPU.

**Provider switcher built into the script:**
Change 2 lines at the top of `method6_rag_openai.py` to switch to any provider:
- `LLM_PROVIDER = "openai"` → `"anthropic"` or `"ollama"`
- `LLM_MODEL = "gpt-4o-mini"` → `"claude-3-5-haiku-20241022"` etc.

**Why 89% vs Method 5's 64%?**
Same RAG architecture, same prompts, same embeddings.
The **25% gap is entirely due to model quality.**
gpt-4o-mini is a 70B+ parameter model with strong medical knowledge.
llama3.2 is a 3B parameter general model with limited medical training.
        """)

    with col_code:
        st.subheader("Key Code — Provider Switcher")
        st.code("""
# ── Change these 2 lines to switch LLM provider ──────
LLM_PROVIDER = "openai"       # "openai" | "anthropic" | "ollama"
LLM_MODEL    = "gpt-4o-mini"  # "gpt-4o" | "claude-3-5-haiku" | "llama3.2"
# ─────────────────────────────────────────────────────

import openai, os

def call_llm(prompt):
    \"\"\"Routes to the configured provider.\"\"\"
    if LLM_PROVIDER == "openai":
        return _call_openai(prompt)
    elif LLM_PROVIDER == "anthropic":
        return _call_anthropic(prompt)
    elif LLM_PROVIDER == "ollama":
        return _call_ollama(prompt)

def _call_openai(prompt):
    client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=10,
        temperature=0.0,   # deterministic output
    )
    return resp.choices[0].message.content.strip()

# API key stored securely in .env file (never hardcoded)
# demo/ProjectCode/.env:
#   OPENAI_API_KEY=sk-proj-...
""", language="python")

        st.success("**Method 6 was run on 300 test reports in 5.3 minutes.** "
                   "Compare: Method 5 took 25 minutes for just 50 reports.")

    st.markdown("---")
    st.subheader("Example Prompt (same structure as Method 5)")

    if report_text:
        codes_preview = "ACC, BLCA, BRCA, CESC, CHOL, COAD, ..."
        truncated_report = " ".join(report_text.split()[:80])
        example_prompt = f"""Classify the cancer type. Reply with ONLY the code from this list:
{codes_preview}

KNOWN EXAMPLE (??):
(the most similar training report — 60 words, retrieved by cosine similarity)

CLASSIFY THIS:
{truncated_report}

Cancer type code:"""
        st.code(example_prompt, language="text")
        st.caption("gpt-4o-mini reads both texts and responds with just the cancer code (e.g., 'BRCA').")

    st.markdown("---")
    st.subheader("Method 5 vs Method 6 — Side by Side")
    df_compare = pd.DataFrame({
        "": ["LLM used", "Parameters", "Medical training", "Sample size", "Runtime",
             "Accuracy", "F1 Weighted", "Unknown responses", "Data privacy", "Cost"],
        "Method 5 (Ollama / llama3.2)": [
            "llama3.2:3b", "3 billion", "General purpose", "50 reports",
            "25 min", "64.00%", "62.3%", "1/50 (2%)", "HIPAA-safe (local)", "Free"],
        "Method 6 (OpenAI / gpt-4o-mini)": [
            "gpt-4o-mini", "70B+ (est.)", "Strong general + medical", "300 reports",
            "5.3 min", "89.00%", "88.3%", "0/300 (0%)", "OK (de-identified data)", "~$0.01"],
    })
    st.dataframe(df_compare.set_index(""), use_container_width=True)

    st.markdown("---")
    result_box("89.00%", "88.3%", "Method 6 — RAG + LLM (gpt-4o-mini, OpenAI API)")
    st.markdown("""
> **Thesis finding:** Upgrading from a 3B local model to gpt-4o-mini adds **+25% accuracy**
> using the exact same RAG pipeline. This isolates model quality as the bottleneck —
> not the retrieval architecture. However, even gpt-4o-mini (89%) still falls short of
> TF-IDF+SVM (97.37%), confirming that for keyword-rich structured text, classical ML remains the benchmark.
    """)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 7 — Results Comparison
# ════════════════════════════════════════════════════════════════════════════════
with tab_cmp:
    st.header("Results Comparison — All Methods")
    st.markdown("*TCGA pathology reports: 9,523 reports, 32 cancer types, 30% test set (2,857 reports)*")

    # ── Block 1 — Cedars-Sinai Baseline ──────────────────────────────────────
    st.markdown("#### Cedars-Sinai (BoW + LR) — Reference Baseline")
    st.info(
        "**Cedars-Sinai AI Campus — published result** \n\n"
        "| Model | Accuracy | F1 (weighted) |\n"
        "|---|---|---|\n"
        "| Bag-of-Words + Logistic Regression | **95.31%** | 94.8% |\n"
        "| Bag-of-Words + Random Forest | 92.65% | — |\n\n"
        "This is the benchmark we are comparing against. "
        "Dataset: 9,523 TCGA pathology reports, 32 cancer types, 30% test set (2,857 reports)."
    )

    st.markdown("---")

    # ── Block 2 — Our Methods ─────────────────────────────────────────────────
    st.markdown("#### Our Methods (Student Results)")

    our_methods    = ["Method 1: TF-IDF + LR", "Method 2: TF-IDF + SVM",
                      "Method 3: BERT Fine-tuned", "Method 4: Embedding k-NN",
                      "Method 5: RAG + LLM (llama3.2)", "Method 6: RAG + LLM (gpt-4o-mini)"]
    our_accuracies = [96.36, 97.37, 94.57, 90.13, 64.00, 89.00]
    our_f1         = [96.1,  97.1,  94.3,  88.7,  62.3,  88.3]
    our_colors     = ["#3498db", "#2ecc71", "#e67e22", "#e74c3c", "#9b59b6", "#1abc9c"]

    # Sort ascending so highest bar appears at top
    sorted_ours = sorted(zip(our_accuracies, our_methods, our_f1, our_colors))
    acc_s, meth_s, f1_s, col_s = zip(*sorted_ours)

    col_bar, col_table = st.columns([1.6, 1])

    with col_bar:
        fig = go.Figure()
        # Cedars-Sinai reference line
        fig.add_vline(x=95.31, line_dash="dash", line_color="#95a5a6", line_width=2,
                      annotation_text="Cedars-Sinai 95.31%", annotation_position="bottom right",
                      annotation_font_color="#95a5a6")
        fig.add_trace(go.Bar(
            x=acc_s, y=meth_s, orientation="h",
            marker_color=col_s,
            text=[f"{a:.2f}%" for a in acc_s],
            textposition="outside",
        ))
        fig.update_layout(
            title="Our Methods — Test Accuracy (sorted by accuracy)",
            height=380,
            xaxis=dict(title="Accuracy (%)", range=[55, 102]),
            margin=dict(l=10, r=60, t=40, b=40),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_table:
        st.markdown("**Summary Table**")
        df_ours = pd.DataFrame({
            "Method": meth_s,
            "Accuracy": [f"{a:.2f}%" for a in acc_s],
            "F1 (weighted)": [f"{f:.1f}%" for f in f1_s],
            "vs Baseline": [f"+{a-95.31:.2f}%" if a >= 95.31 else f"{a-95.31:.2f}%" for a in acc_s],
        })
        st.dataframe(df_ours.set_index("Method"), use_container_width=True)

    st.markdown("---")
    st.subheader("Key Findings for Thesis")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
**Finding 1 — TF-IDF > BoW**
Replacing raw word counts with TF-IDF scores improves accuracy by 1.05% (95.31% → 96.36%).
Bigrams capture cancer-specific phrases: "clear cell", "renal cell", "ductal carcinoma".

**Finding 2 — SVM > LR on text**
LinearSVC's maximum-margin boundary outperforms Logistic Regression by 1.01% (96.36% → 97.37%).
The maximum-margin criterion is especially effective in high-dimensional sparse spaces.

**Finding 3 — Classical ML beats BERT**
TF-IDF+SVM (97.37%) outperforms fine-tuned Bio_ClinicalBERT (94.57%) by 2.8%.
Consistent cancer-specific vocabulary makes TF-IDF features very powerful.
Cancer pathology reports use standardised, predictable language.
        """)

    with col2:
        st.markdown("""
**Finding 4 — BERT fails on rare classes**
UCS (Uterine Carcinosarcoma) — only 39 training samples:
- TF-IDF + SVM: F1 = **0.786**
- Bio_ClinicalBERT: F1 = **0.000** (predicts UCEC for all UCS cases)

BERT over-generalises on tiny classes; SVM's margin-based approach handles them better.

**Finding 5 — Complexity ≠ Accuracy**
Method 5 (RAG + 3B LLM) scores only 64% — lowest of all methods.
The general-purpose SLM has no medical fine-tuning. Truncating reports to 80 words compounds the problem. This is a valuable negative result: for structured vocabulary tasks, classical ML wins.

**Finding 6 — Embedding k-NN (90%) confirms semantic structure**
The fact that semantic similarity alone achieves 90% proves pathology reports cluster clearly by cancer type in embedding space — even without any classifier training.
        """)

    st.markdown("---")
    st.subheader("Method Architecture Comparison")
    comparison_data = {
        "": ["Training approach", "Input representation", "Context awareness",
             "Handles rare classes", "Inference speed", "Requires GPU"],
        "BoW+LR": ["Supervised", "Word counts", "None (bag)", "No", "Instant", "No"],
        "TF-IDF+LR": ["Supervised", "TF-IDF scores", "None (bag)", "Yes (balanced)", "Instant", "No"],
        "TF-IDF+SVM": ["Supervised", "TF-IDF scores", "None (bag)", "Yes (balanced)", "Instant", "No"],
        "BERT": ["Fine-tuned DL", "Contextual 768-d", "Full bidirectional", "Partial", "~15s/report CPU", "For training"],
        "Emb+kNN": ["No training", "MiniLM 384-d", "Semantic", "N/A", "~1s (cached)", "No"],
        "RAG+LLM": ["Prompt-based", "Cosine + prompt", "Context window", "Depends on LLM", "~30s/report", "No"],
    }
    df_arch = pd.DataFrame(comparison_data).set_index("")
    st.dataframe(df_arch, use_container_width=True)
