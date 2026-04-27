"""
Cedars-Sinai Reference Pipeline — Combined Single Script
=========================================================
Reproduces all 5 notebooks in one runnable Python file:
  1-Compile_Dataset
  2-Train_Val_Test_Split
  3-Bag_of_Words
  4-BoW_LR
  4-BoW_RF
  5-BoW_ML  (pipeline + hyperparameter tuning + test evaluation)

Run from project root:
  python reference_code/cancer_type/cedars_sinai_pipeline.py

All intermediate files are saved to:
  reference_code/cancer_type/output/
"""

import os
import sys
import random
import logging
import warnings
import time

warnings.filterwarnings("ignore")

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger(__name__)

def section(title):
    bar = "=" * 70
    log.info("")
    log.info(bar)
    log.info(f"  {title}")
    log.info(bar)

def subsection(title):
    log.info(f"\n--- {title} ---")


# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR     = r"c:\Users\kings\OneDrive\Desktop\CSUDH_Kingslee\CSUDH-Spring-2026\CSC-590-MastersProject\MastersProject"
REPORTS_PATH = os.path.join(BASE_DIR, "DataSet", "TCGA_Reports.csv", "TCGA_Reports.csv")
LABELS_PATH  = os.path.join(BASE_DIR, "DataSet", "tcga_patient_to_cancer_type.csv")
OUT_DIR      = os.path.join(BASE_DIR, "reference_code", "cancer_type", "output")
os.makedirs(OUT_DIR, exist_ok=True)

# Intermediate file paths
COMPILED_CSV  = os.path.join(OUT_DIR, "tcga_reports_cancer_type.csv")
TRAIN_CSV     = os.path.join(OUT_DIR, "train_tcga_reports_cancer_type.csv")
VAL_CSV       = os.path.join(OUT_DIR, "val_tcga_reports_cancer_type.csv")
TEST_CSV      = os.path.join(OUT_DIR, "test_tcga_reports_cancer_type.csv")

# Cancer type full names (same as tcga-tumor-types.csv from reference repo)
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


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Compile Dataset  (notebook: 1-Compile_Dataset.ipynb)
# ══════════════════════════════════════════════════════════════════════════════
def step1_compile_dataset():
    section("STEP 1 — Compile Dataset  [1-Compile_Dataset.ipynb]")
    import pandas as pd

    # ── Load reports ──────────────────────────────────────────────────────────
    subsection("Loading TCGA pathology reports")
    log.info(f"Reading: {REPORTS_PATH}")
    t0 = time.time()
    df_corpus = pd.read_csv(REPORTS_PATH)
    log.info(f"Loaded in {time.time()-t0:.1f}s")
    log.info(f"Shape: {df_corpus.shape}  (rows=reports, cols=columns)")
    log.info(f"Columns: {list(df_corpus.columns)}")

    subsection("Sample raw rows")
    log.info(f"\n{df_corpus.head(3).to_string()}")

    # ── Extract patient_id from patient_filename ───────────────────────────
    subsection("Extracting patient_id from patient_filename")
    log.info("  patient_filename example: 'TCGA-BP-5195.25c0b433-...'")
    log.info("  patient_id = everything before the first '.'  → 'TCGA-BP-5195'")
    df_corpus["patient_id"] = df_corpus["patient_filename"].apply(lambda x: x.split(".")[0])

    # Sanity checks
    assert not df_corpus["patient_filename"].duplicated().any(), "Duplicate filenames!"
    assert not df_corpus["patient_id"].duplicated().any(),       "Duplicate patient IDs!"
    log.info("  Sanity check passed: no duplicate patient IDs")

    df_corpus = df_corpus[["patient_id", "text"]]
    df_corpus.index = df_corpus["patient_id"].values
    log.info(f"  Unique patients: {len(df_corpus)}")
    log.info(f"\n{df_corpus.head(3).to_string()}")

    # ── Load cancer type labels ────────────────────────────────────────────
    subsection("Loading cancer type labels")
    log.info(f"Reading: {LABELS_PATH}")
    df_labels = pd.read_csv(LABELS_PATH)
    log.info(f"Shape: {df_labels.shape}")
    log.info(f"\n{df_labels.head(5).to_string()}")
    log.info(f"  Unique cancer types in labels file: {df_labels['cancer_type'].nunique()}")

    assert not df_labels["patient_id"].duplicated().any(), "Duplicate patient IDs in labels!"
    df_labels.index = df_labels["patient_id"].values

    # ── Merge reports + labels ────────────────────────────────────────────
    subsection("Merging reports with labels")
    log.info("  Strategy: inner join on patient_id (only keep reports that have a label)")
    assert df_corpus["patient_id"].isin(df_labels["patient_id"]).all(), "Some reports have no label!"
    df_corpus["cancer_type"] = df_labels.loc[df_corpus.index, "cancer_type"]
    log.info(f"  Merged shape: {df_corpus.shape}")

    # ── Add full cancer type name ─────────────────────────────────────────
    subsection("Adding full cancer type name")
    df_corpus["cancer_type_name"] = df_corpus["cancer_type"].apply(
        lambda x: CANCER_NAMES.get(x, x)
    )
    log.info("  Cancer types present in this dataset:")
    dist = df_corpus["cancer_type"].value_counts()
    for code, count in dist.items():
        name = CANCER_NAMES.get(code, code)
        pct  = count / len(df_corpus) * 100
        log.info(f"    {code:6s} ({name:45s}) — {count:4d} reports ({pct:.1f}%)")

    log.info(f"\n  Total reports: {len(df_corpus)}")
    log.info(f"  Total cancer types: {df_corpus['cancer_type'].nunique()}")

    # ── Text stats ────────────────────────────────────────────────────────
    subsection("Text length analysis")
    df_corpus["word_count"] = df_corpus["text"].apply(lambda x: len(str(x).split()))
    log.info(f"  Average words per report : {df_corpus['word_count'].mean():.0f}")
    log.info(f"  Median  words per report : {df_corpus['word_count'].median():.0f}")
    log.info(f"  Min     words per report : {df_corpus['word_count'].min()}")
    log.info(f"  Max     words per report : {df_corpus['word_count'].max()}")
    log.info(f"  Reports > 512 words      : {(df_corpus['word_count'] > 512).sum()} ({(df_corpus['word_count'] > 512).mean()*100:.1f}%)")

    # ── Example report ────────────────────────────────────────────────────
    subsection("Example pathology report (first 500 chars)")
    example = df_corpus["text"].iloc[0]
    log.info(f"  Patient: {df_corpus['patient_id'].iloc[0]}")
    log.info(f"  Cancer:  {df_corpus['cancer_type'].iloc[0]} — {df_corpus['cancer_type_name'].iloc[0]}")
    log.info(f"  Text preview:\n\n    {example[:500]}\n")

    # ── Save compiled dataset ─────────────────────────────────────────────
    subsection("Saving compiled dataset")
    df_corpus.drop(columns=["word_count"], inplace=True)
    df_corpus.to_csv(COMPILED_CSV, index=False)
    log.info(f"  Saved to: {COMPILED_CSV}")

    return df_corpus


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Train / Val / Test Split  (notebook: 2-Train_Val_Test_Split.ipynb)
# ══════════════════════════════════════════════════════════════════════════════
def step2_split(df_corpus):
    section("STEP 2 — Train / Val / Test Split  [2-Train_Val_Test_Split.ipynb]")
    import pandas as pd

    random.seed(0)
    VAL_SIZE  = 0.20   # 20% validation
    TEST_SIZE = 0.30   # 30% test
    # remainder → 50% train

    log.info(f"  Total reports : {len(df_corpus)}")
    log.info(f"  Test  split   : {TEST_SIZE*100:.0f}%")
    log.info(f"  Val   split   : {VAL_SIZE*100:.0f}%")
    log.info(f"  Train split   : {(1-TEST_SIZE-VAL_SIZE)*100:.0f}%")

    df_corpus.index = df_corpus["patient_id"].values
    arr_all_ids = df_corpus["patient_id"].values.tolist()

    n_test = round(TEST_SIZE * len(df_corpus))
    n_val  = round(VAL_SIZE  * len(df_corpus))

    # ── Test split ────────────────────────────────────────────────────────
    subsection("Creating test set")
    arr_test_ids = random.sample(arr_all_ids, n_test)
    df_test = df_corpus.loc[arr_test_ids].copy()
    log.info(f"  Test set  : {len(df_test)} reports  ({len(df_test)/len(df_corpus)*100:.1f}%)")

    arr_trainval_ids = [pid for pid in arr_all_ids if pid not in arr_test_ids]
    df_trainval = df_corpus.loc[arr_trainval_ids].copy()

    # ── Val split ─────────────────────────────────────────────────────────
    subsection("Creating validation set")
    arr_val_ids = random.sample(arr_trainval_ids, n_val)
    df_val = df_corpus.loc[arr_val_ids].copy()
    log.info(f"  Val   set : {len(df_val)} reports  ({len(df_val)/len(df_corpus)*100:.1f}%)")

    # ── Train split ────────────────────────────────────────────────────────
    arr_train_ids = [pid for pid in arr_trainval_ids if pid not in arr_val_ids]
    df_train = df_corpus.loc[arr_train_ids].copy()
    log.info(f"  Train set : {len(df_train)} reports  ({len(df_train)/len(df_corpus)*100:.1f}%)")

    # ── Sanity checks ─────────────────────────────────────────────────────
    subsection("Sanity checks")
    assert df_train.shape[0] + df_val.shape[0] + df_test.shape[0] == df_corpus.shape[0]
    assert len(set(df_train["patient_id"]) & set(df_val["patient_id"]))  == 0, "Train/Val overlap!"
    assert len(set(df_train["patient_id"]) & set(df_test["patient_id"])) == 0, "Train/Test overlap!"
    assert len(set(df_val["patient_id"])   & set(df_test["patient_id"])) == 0, "Val/Test overlap!"
    log.info("  All sanity checks passed — no data leakage between splits")

    # ── Label distribution across splits ─────────────────────────────────
    subsection("Cancer type distribution across splits")
    log.info(f"  {'Type':<6} {'All':>10} {'Train':>10} {'Val':>10} {'Test':>10}")
    log.info(f"  {'-'*50}")
    for code in sorted(df_corpus["cancer_type"].unique()):
        n_all   = (df_corpus["cancer_type"] == code).sum()
        n_train = (df_train["cancer_type"]  == code).sum()
        n_val_  = (df_val["cancer_type"]    == code).sum()
        n_test_ = (df_test["cancer_type"]   == code).sum()
        log.info(
            f"  {code:<6} {n_all:>5}({n_all/len(df_corpus)*100:4.1f}%)"
            f"  {n_train:>5}({n_train/len(df_train)*100:4.1f}%)"
            f"  {n_val_:>5}({n_val_/len(df_val)*100:4.1f}%)"
            f"  {n_test_:>5}({n_test_/len(df_test)*100:4.1f}%)"
        )

    # ── Save splits ────────────────────────────────────────────────────────
    subsection("Saving splits")
    df_train.to_csv(TRAIN_CSV, index=False)
    df_val.to_csv(VAL_CSV,   index=False)
    df_test.to_csv(TEST_CSV,  index=False)
    log.info(f"  Train → {TRAIN_CSV}")
    log.info(f"  Val   → {VAL_CSV}")
    log.info(f"  Test  → {TEST_CSV}")

    return df_train, df_val, df_test


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Bag of Words Analysis  (notebook: 3-Bag_of_Words.ipynb)
# ══════════════════════════════════════════════════════════════════════════════
def step3_bag_of_words(df_train):
    section("STEP 3 — Bag of Words Analysis  [3-Bag_of_Words.ipynb]")
    import pandas as pd
    import nltk
    from nltk.tokenize import word_tokenize
    from sklearn.feature_extraction.text import CountVectorizer

    log.info("Downloading NLTK data (punkt, stopwords) — skipped if already present")
    nltk.download("punkt_tab", quiet=True)
    nltk.download("stopwords", quiet=True)

    arr_train_corpus = df_train["text"].values.tolist()
    log.info(f"  Training corpus: {len(arr_train_corpus)} reports")

    # ── Example text analysis ─────────────────────────────────────────────
    subsection("Example report tokenization")
    example_text = arr_train_corpus[0]
    log.info(f"  Report length : {len(example_text)} characters")
    log.info(f"\n  Raw text (first 300 chars):\n    {example_text[:300]}\n")

    log.info("  Simple whitespace tokenizer (split):")
    simple_tokens = example_text.split()[:15]
    log.info(f"    {simple_tokens}  ... ({len(example_text.split())} total tokens)")

    log.info("\n  NLTK word_tokenize (smarter — handles punctuation):")
    nltk_tokens = word_tokenize(example_text)[:20]
    log.info(f"    {nltk_tokens}  ... ({len(word_tokenize(example_text))} total tokens)")

    # ── Corpus-level stats ─────────────────────────────────────────────────
    subsection("Corpus-level token statistics")
    log.info("  Tokenizing entire training corpus (this takes ~30 sec)...")
    t0 = time.time()
    corpus_text   = " ".join(arr_train_corpus)
    all_tokens    = word_tokenize(corpus_text)
    log.info(f"  Done in {time.time()-t0:.1f}s")
    log.info(f"  Total characters in corpus : {len(corpus_text):,}")
    log.info(f"  Total tokens               : {len(all_tokens):,}")
    log.info(f"  Unique tokens (vocab size) : {len(set(all_tokens)):,}")

    log.info("\n  Top 20 most frequent tokens (includes stopwords + punctuation):")
    import pandas as pd
    token_freq = pd.Series(all_tokens).value_counts().head(20)
    for token, count in token_freq.items():
        log.info(f"    '{token}' → {count:,}")

    # ── Tokenizer with cleaning ────────────────────────────────────────────
    subsection("Custom tokenizer — removes stopwords, punctuation, single chars")
    arr_stopwords = nltk.corpus.stopwords.words("english")
    log.info(f"  Stopwords list size: {len(arr_stopwords)}")
    log.info(f"  Example stopwords: {arr_stopwords[:10]}")

    def tokenizer(text):
        tokens = word_tokenize(text.lower())
        tokens = [t for t in tokens if len(t) > 1]           # remove single chars
        tokens = [t for t in tokens if t not in arr_stopwords] # remove stopwords
        tokens = [t for t in tokens if t.isalpha()]           # remove punctuation / numbers
        return tokens

    log.info("\n  Applying custom tokenizer to example report:")
    cleaned = tokenizer(example_text)
    log.info(f"    Before cleaning: {len(word_tokenize(example_text))} tokens")
    log.info(f"    After  cleaning: {len(cleaned)} tokens")
    log.info(f"    Sample cleaned tokens: {cleaned[:20]}")

    # ── Fit CountVectorizer (Bag of Words) ────────────────────────────────
    subsection("Fitting Bag-of-Words (CountVectorizer) on training corpus")
    log.info("  This converts each report into a vector of word counts")
    log.info("  Fitting on training data only (to prevent data leakage)...")
    t0 = time.time()
    word_vectorizer = CountVectorizer(
        tokenizer=tokenizer,
        token_pattern=None,
        lowercase=False,
        stop_words=None
    )
    word_vectorizer.fit(arr_train_corpus)
    log.info(f"  Fit complete in {time.time()-t0:.1f}s")
    log.info(f"  Vocabulary size (unique medical terms): {len(word_vectorizer.vocabulary_):,}")

    log.info("\n  Sample vocabulary entries (word → index):")
    sample_vocab = list(word_vectorizer.vocabulary_.items())[:10]
    for word, idx in sample_vocab:
        log.info(f"    '{word}' → index {idx}")

    subsection("Transforming training corpus into BoW matrix")
    log.info("  Each row = one report, each column = one word from vocabulary")
    log.info("  Cell value = how many times that word appears in that report")
    t0 = time.time()
    arr_train_bow = word_vectorizer.transform(arr_train_corpus)
    log.info(f"  Transform complete in {time.time()-t0:.1f}s")
    log.info(f"  BoW matrix shape: {arr_train_bow.shape}")
    log.info(f"    → {arr_train_bow.shape[0]} reports × {arr_train_bow.shape[1]} unique words")
    log.info(f"  Matrix is sparse: {arr_train_bow.nnz / (arr_train_bow.shape[0]*arr_train_bow.shape[1])*100:.2f}% non-zero")

    return word_vectorizer, arr_train_bow, tokenizer


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4a — BoW + Logistic Regression  (notebook: 4-BoW_LR.ipynb)
# ══════════════════════════════════════════════════════════════════════════════
def step4a_bow_lr(df_train, df_val, word_vectorizer):
    section("STEP 4a — Bag of Words + Logistic Regression  [4-BoW_LR.ipynb]")
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score, classification_report

    arr_train_corpus = df_train["text"].values.tolist()
    arr_train_labels = df_train["cancer_type"].values.tolist()
    arr_val_corpus   = df_val["text"].values.tolist()
    arr_val_labels   = df_val["cancer_type"].values.tolist()

    log.info(f"  Training samples : {len(arr_train_corpus)}")
    log.info(f"  Validation samples: {len(arr_val_corpus)}")
    log.info(f"  Classes (cancer types): {len(set(arr_train_labels))}")

    # ── Transform to BoW ──────────────────────────────────────────────────
    subsection("Transforming text to BoW vectors")
    arr_train_bow = word_vectorizer.transform(arr_train_corpus)
    arr_val_bow   = word_vectorizer.transform(arr_val_corpus)
    log.info(f"  Train BoW shape: {arr_train_bow.shape}")
    log.info(f"  Val   BoW shape: {arr_val_bow.shape}")

    # ── Train LR ──────────────────────────────────────────────────────────
    subsection("Training Logistic Regression")
    log.info("  Logistic Regression is a linear model that learns a weight for each word")
    log.info("  It predicts the most likely cancer type given the word frequencies")
    log.info("  Parameters: random_state=0, max_iter=200")
    log.info("  Training... (takes ~30-60 sec for 23,818 features × 4,761 samples)")
    t0 = time.time()
    clf_lr = LogisticRegression(random_state=0, max_iter=200)
    clf_lr.fit(arr_train_bow, arr_train_labels)
    log.info(f"  Training complete in {time.time()-t0:.1f}s")

    # ── Train set evaluation ───────────────────────────────────────────────
    subsection("Evaluation on TRAINING set")
    arr_train_pred = clf_lr.predict(arr_train_bow)
    train_acc = accuracy_score(arr_train_labels, arr_train_pred)
    log.info(f"  Train Accuracy : {train_acc*100:.2f}%")
    log.info("  (Near-perfect on training data is expected — model has 'seen' these)")

    # ── Validation set evaluation ──────────────────────────────────────────
    subsection("Evaluation on VALIDATION set  (unseen data)")
    arr_val_pred = clf_lr.predict(arr_val_bow)
    val_acc = accuracy_score(arr_val_labels, arr_val_pred)
    val_f1  = f1_score(arr_val_labels, arr_val_pred, average="weighted")
    log.info(f"  Val Accuracy   : {val_acc*100:.2f}%")
    log.info(f"  Val F1 (weighted): {val_f1*100:.2f}%")
    log.info(f"\n  Reference result from Cedars-Sinai: ~95.31% accuracy")
    log.info(f"  Our reproduction                  : {val_acc*100:.2f}% accuracy")

    log.info("\n  Full classification report:")
    log.info(classification_report(arr_val_labels, arr_val_pred, zero_division=0))

    # ── Per-class F1 ──────────────────────────────────────────────────────
    subsection("Per-class F1 scores (sorted worst → best)")
    arr_f1_per_class = f1_score(arr_val_labels, arr_val_pred,
                                average=None, labels=clf_lr.classes_, zero_division=0)
    import pandas as pd
    df_f1 = pd.DataFrame({
        "cancer_code": clf_lr.classes_,
        "cancer_name": [CANCER_NAMES.get(c, c) for c in clf_lr.classes_],
        "f1_score":    arr_f1_per_class,
        "n_train":     [len(df_train[df_train["cancer_type"]==c]) for c in clf_lr.classes_],
        "n_val":       [len(df_val[df_val["cancer_type"]==c])     for c in clf_lr.classes_],
    }).sort_values("f1_score")

    log.info(f"  {'Code':<6} {'Cancer Name':<45} {'F1':>6} {'Train':>6} {'Val':>5}")
    log.info(f"  {'-'*72}")
    for _, row in df_f1.iterrows():
        log.info(
            f"  {row['cancer_code']:<6} {row['cancer_name']:<45} "
            f"{row['f1_score']:>6.3f} {int(row['n_train']):>6} {int(row['n_val']):>5}"
        )
    log.info("\n  KEY INSIGHT: Rare cancer types (low n_train) generally have lower F1")

    # ── Feature importance (top words per class) ───────────────────────────
    subsection("Top 10 most important words per cancer type (LR coefficients)")
    log.info("  LR assigns a coefficient to each word for each cancer type")
    log.info("  High coefficient = that word strongly predicts this cancer type")
    dict_index_word = {v: k for k, v in word_vectorizer.vocabulary_.items()}

    for i, class_label in enumerate(clf_lr.classes_):
        top_indices = sorted(range(len(clf_lr.coef_[i])),
                             key=lambda j: clf_lr.coef_[i][j], reverse=True)[:10]
        top_words = [dict_index_word[j] for j in top_indices]
        top_vals  = [round(clf_lr.coef_[i][j], 3) for j in top_indices]
        name = CANCER_NAMES.get(class_label, class_label)
        log.info(f"\n  {class_label} — {name}")
        for word, val in zip(top_words, top_vals):
            log.info(f"    {word:<25} coeff={val:+.3f}")

    return clf_lr, val_acc


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4b — BoW + Random Forest  (notebook: 4-BoW_RF.ipynb)
# ══════════════════════════════════════════════════════════════════════════════
def step4b_bow_rf(df_train, df_val, word_vectorizer):
    section("STEP 4b — Bag of Words + Random Forest  [4-BoW_RF.ipynb]")
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, f1_score, classification_report

    arr_train_corpus = df_train["text"].values.tolist()
    arr_train_labels = df_train["cancer_type"].values.tolist()
    arr_val_corpus   = df_val["text"].values.tolist()
    arr_val_labels   = df_val["cancer_type"].values.tolist()

    # ── Transform to BoW ──────────────────────────────────────────────────
    subsection("Transforming text to BoW vectors")
    arr_train_bow = word_vectorizer.transform(arr_train_corpus)
    arr_val_bow   = word_vectorizer.transform(arr_val_corpus)

    log.info("  Word frequency statistics in training BoW matrix:")
    word_freq_stats = arr_train_bow.sum(axis=0)
    import numpy as np
    freq_arr = np.array(word_freq_stats)[0]
    log.info(f"    Mean  word frequency: {freq_arr.mean():.1f}")
    log.info(f"    Median word frequency: {np.median(freq_arr):.1f}")
    log.info(f"    Max   word frequency: {freq_arr.max():.0f}")
    log.info(f"    (Most words appear rarely — typical for medical text)")

    # ── Train RF ──────────────────────────────────────────────────────────
    subsection("Training Random Forest")
    log.info("  Random Forest = ensemble of 100 decision trees")
    log.info("  Each tree votes on the cancer type; majority wins")
    log.info("  Parameters: n_estimators=100, random_state=0")
    log.info("  Training... (takes ~3-5 min — 100 trees × 23,818 features)")
    t0 = time.time()
    clf_rf = RandomForestClassifier(random_state=0, n_estimators=100)
    clf_rf.fit(arr_train_bow, arr_train_labels)
    log.info(f"  Training complete in {time.time()-t0:.1f}s")

    # ── Train set evaluation ───────────────────────────────────────────────
    subsection("Evaluation on TRAINING set")
    arr_train_pred = clf_rf.predict(arr_train_bow)
    train_acc = accuracy_score(arr_train_labels, arr_train_pred)
    log.info(f"  Train Accuracy : {train_acc*100:.2f}%")

    # ── Validation set evaluation ──────────────────────────────────────────
    subsection("Evaluation on VALIDATION set  (unseen data)")
    arr_val_pred = clf_rf.predict(arr_val_bow)
    val_acc = accuracy_score(arr_val_labels, arr_val_pred)
    val_f1  = f1_score(arr_val_labels, arr_val_pred, average="weighted", zero_division=0)
    log.info(f"  Val Accuracy   : {val_acc*100:.2f}%")
    log.info(f"  Val F1 (weighted): {val_f1*100:.2f}%")
    log.info(f"\n  Reference result from Cedars-Sinai: ~92.65% accuracy (RF)")
    log.info(f"  Our reproduction                  : {val_acc*100:.2f}% accuracy")

    log.info("\n  Full classification report:")
    log.info(classification_report(arr_val_labels, arr_val_pred, zero_division=0))

    # ── Per-class F1 ──────────────────────────────────────────────────────
    subsection("Per-class F1 scores (sorted worst → best)")
    arr_f1_per_class = f1_score(arr_val_labels, arr_val_pred,
                                average=None, labels=clf_rf.classes_, zero_division=0)
    df_f1 = pd.DataFrame({
        "cancer_code": clf_rf.classes_,
        "cancer_name": [CANCER_NAMES.get(c, c) for c in clf_rf.classes_],
        "f1_score":    arr_f1_per_class,
        "n_train":     [len(df_train[df_train["cancer_type"]==c]) for c in clf_rf.classes_],
        "n_val":       [len(df_val[df_val["cancer_type"]==c])     for c in clf_rf.classes_],
    }).sort_values("f1_score")

    log.info(f"  {'Code':<6} {'Cancer Name':<45} {'F1':>6} {'Train':>6} {'Val':>5}")
    log.info(f"  {'-'*72}")
    for _, row in df_f1.iterrows():
        log.info(
            f"  {row['cancer_code']:<6} {row['cancer_name']:<45} "
            f"{row['f1_score']:>6.3f} {int(row['n_train']):>6} {int(row['n_val']):>5}"
        )

    # ── Feature importance ─────────────────────────────────────────────────
    subsection("Top 20 most important words overall (RF feature importances)")
    log.info("  RF feature importance = how much each word reduces classification error")
    log.info("  (Unlike LR, this is global — not per cancer type)")
    dict_index_word = {v: k for k, v in word_vectorizer.vocabulary_.items()}
    top20_idx = sorted(range(len(clf_rf.feature_importances_)),
                       key=lambda j: clf_rf.feature_importances_[j], reverse=True)[:20]
    for rank, idx in enumerate(top20_idx, 1):
        word = dict_index_word[idx]
        imp  = clf_rf.feature_importances_[idx]
        log.info(f"  #{rank:2d}  '{word:<20}'  importance={imp:.5f}")

    return clf_rf, val_acc


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — Pipeline + Hyperparameter Tuning + Test Evaluation (5-BoW_ML.ipynb)
# ══════════════════════════════════════════════════════════════════════════════
def step5_pipeline_and_tuning(df_train, df_val, df_test, tokenizer):
    section("STEP 5 — Pipeline + Hyperparameter Tuning + Test Evaluation  [5-BoW_ML.ipynb]")
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
    from sklearn.metrics import accuracy_score, classification_report

    arr_train_corpus = df_train["text"].values.tolist()
    arr_train_labels = df_train["cancer_type"].values.tolist()
    arr_val_corpus   = df_val["text"].values.tolist()
    arr_val_labels   = df_val["cancer_type"].values.tolist()
    arr_test_corpus  = df_test["text"].values.tolist()
    arr_test_labels  = df_test["cancer_type"].values.tolist()

    log.info(f"  Train : {len(arr_train_corpus)} | Val : {len(arr_val_corpus)} | Test : {len(arr_test_corpus)}")

    # ── Custom CV (uses fixed train/val split — no random shuffling) ───────
    X_data = arr_train_corpus + arr_val_corpus
    y_data = arr_train_labels + arr_val_labels
    train_idx = list(range(len(arr_train_corpus)))
    val_idx   = list(range(len(arr_train_corpus), len(X_data)))
    custom_cv = [(train_idx, val_idx)]

    # ══ 5A — Logistic Regression Pipeline + Grid Search ══════════════════
    subsection("5A: LR Pipeline — quick sanity check (no tuning yet)")
    log.info("  Sklearn Pipeline bundles: BoW vectorizer → LR classifier")
    log.info("  This ensures no data leakage (vectorizer is fit only on train split)")
    lr_pipeline = Pipeline([
        ("bow", CountVectorizer(tokenizer=tokenizer, token_pattern=None,
                                lowercase=False, stop_words=None)),
        ("classifier", LogisticRegression(random_state=0, max_iter=500))
    ])
    log.info("  Training LR pipeline...")
    t0 = time.time()
    lr_pipeline.fit(arr_train_corpus, arr_train_labels)
    log.info(f"  Done in {time.time()-t0:.1f}s")
    arr_val_pred = lr_pipeline.predict(arr_val_corpus)
    val_acc = accuracy_score(arr_val_labels, arr_val_pred)
    log.info(f"  Validation accuracy (max_iter=500): {val_acc*100:.2f}%")

    subsection("5B: LR Hyperparameter Tuning — GridSearchCV")
    log.info("  Trying different max_iter values: [200, 500]")
    log.info("  GridSearch tries all combinations and picks the best one")
    log.info("  Using our fixed train/val split as the 'cross-validation' fold")
    lr_grid_pipeline = Pipeline([
        ("bow", CountVectorizer(tokenizer=tokenizer, token_pattern=None,
                                lowercase=False, stop_words=None)),
        ("classifier", LogisticRegression(random_state=0))
    ])
    param_grid_lr = {"classifier__max_iter": [200, 500]}
    grid_search = GridSearchCV(
        estimator=lr_grid_pipeline,
        param_grid=param_grid_lr,
        scoring="accuracy",
        cv=custom_cv,
        verbose=2,
        refit=True
    )
    log.info("  Running GridSearch (trains 2 models, takes ~2 min)...")
    t0 = time.time()
    grid_search.fit(X_data, y_data)
    log.info(f"  GridSearch complete in {time.time()-t0:.1f}s")
    log.info(f"  Best parameters: {grid_search.best_params_}")
    log.info(f"  Best val accuracy: {grid_search.best_score_*100:.2f}%")

    import pandas as pd
    df_gs = pd.DataFrame(grid_search.cv_results_)[
        ["param_classifier__max_iter", "mean_test_score", "mean_fit_time"]
    ]
    log.info("\n  GridSearch results:")
    for _, row in df_gs.iterrows():
        log.info(f"    max_iter={int(row['param_classifier__max_iter'])} → "
                 f"val_acc={row['mean_test_score']*100:.2f}%  "
                 f"(fit time={row['mean_fit_time']:.1f}s)")

    subsection("5C: LR Test Set Evaluation (final — only done once!)")
    log.info("  IMPORTANT: Test set is only used here at the very end")
    log.info("  This gives an unbiased estimate of real-world performance")
    best_lr_model = grid_search.best_estimator_
    arr_test_pred_lr = best_lr_model.predict(arr_test_corpus)
    lr_test_acc = accuracy_score(arr_test_labels, arr_test_pred_lr)
    log.info(f"\n  LR Final Test Accuracy : {lr_test_acc*100:.2f}%")
    log.info(f"  Reference (Cedars-Sinai): ~95.31%")
    log.info("\n  Full test classification report:")
    log.info(classification_report(arr_test_labels, arr_test_pred_lr, zero_division=0))

    # ══ 5D — Random Forest Pipeline + Randomized Search ══════════════════
    subsection("5D: RF Hyperparameter Tuning — RandomizedSearchCV")
    log.info("  RF has too many hyperparameters to try all → use RandomizedSearch")
    log.info("  Trying 2 random combinations from the search space:")
    log.info("    n_estimators       : [10, 100, 500]")
    log.info("    max_features       : ['sqrt', 'log2']")
    log.info("    min_samples_split  : [2, 5, 10]")
    rf_pipeline = Pipeline([
        ("bow", CountVectorizer(tokenizer=tokenizer, token_pattern=None,
                                lowercase=False, stop_words=None)),
        ("classifier", RandomForestClassifier(random_state=0))
    ])
    param_grid_rf = {
        "classifier__n_estimators":      [10, 100, 500],
        "classifier__max_features":      ["sqrt", "log2"],
        "classifier__min_samples_split": [2, 5, 10]
    }
    rand_search = RandomizedSearchCV(
        estimator=rf_pipeline,
        param_distributions=param_grid_rf,
        scoring="accuracy",
        cv=custom_cv,
        verbose=2,
        refit=True,
        n_iter=2,
        random_state=0
    )
    log.info("  Running RandomizedSearch (trains 2 RF models, takes ~3-5 min)...")
    t0 = time.time()
    rand_search.fit(X_data, y_data)
    log.info(f"  RandomizedSearch complete in {time.time()-t0:.1f}s")
    log.info(f"  Best parameters: {rand_search.best_params_}")
    log.info(f"  Best val accuracy: {rand_search.best_score_*100:.2f}%")

    subsection("5E: RF Test Set Evaluation (final)")
    best_rf_model = rand_search.best_estimator_
    arr_test_pred_rf = best_rf_model.predict(arr_test_corpus)
    rf_test_acc = accuracy_score(arr_test_labels, arr_test_pred_rf)
    log.info(f"\n  RF Final Test Accuracy : {rf_test_acc*100:.2f}%")
    log.info(f"  Reference (Cedars-Sinai): ~92.65%")
    log.info("\n  Full test classification report:")
    log.info(classification_report(arr_test_labels, arr_test_pred_rf, zero_division=0))

    return lr_test_acc, rf_test_acc


# ══════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
def print_summary(lr_val_acc, rf_val_acc, lr_test_acc, rf_test_acc):
    section("FINAL SUMMARY — Cedars-Sinai Reference Results vs Our Reproduction")

    log.info(f"  {'Model':<35} {'Val Acc':>10} {'Test Acc':>10} {'Reference':>12}")
    log.info(f"  {'-'*72}")
    log.info(f"  {'BoW + Logistic Regression':<35} {lr_val_acc*100:>9.2f}% {lr_test_acc*100:>9.2f}% {'~95.31%':>12}")
    log.info(f"  {'BoW + Random Forest':<35} {rf_val_acc*100:>9.2f}% {rf_test_acc*100:>9.2f}% {'~92.65%':>12}")

    log.info("""
  KEY TAKEAWAYS:
  -------------
  1. Bag-of-Words works surprisingly well for cancer classification (~95%)
     because pathology reports use very consistent, cancer-specific terminology.

  2. Logistic Regression outperforms Random Forest on this task.
     LR is a linear model — it assigns a weight to each medical word (e.g.,
     'glioblastoma' → brain cancer, 'ductal' → breast cancer).

  3. Rare cancer types (e.g., UCS, KICH) have lower F1 scores because
     there are fewer training examples to learn from.

  4. This is what YOUR project improves upon:
     - TF-IDF instead of raw BoW counts (already done — 96.36%)
     - BERT fine-tuning (planned Phase 3)
     - Embedding similarity / RAG (planned Phase 4/6)
    """)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    overall_start = time.time()

    log.info("=" * 70)
    log.info("  MEDREx — Cedars-Sinai Reference Pipeline (All 5 Notebooks Combined)")
    log.info("  CSC-590 Masters Project, CSUDH Spring 2026")
    log.info("=" * 70)
    log.info(f"  Output directory: {OUT_DIR}")

    # Step 1 — Compile dataset
    df_corpus = step1_compile_dataset()

    # Step 2 — Split
    df_train, df_val, df_test = step2_split(df_corpus)

    # Step 3 — BoW analysis + fit vectorizer
    word_vectorizer, arr_train_bow, tokenizer_fn = step3_bag_of_words(df_train)

    # Step 4a — Logistic Regression
    clf_lr, lr_val_acc = step4a_bow_lr(df_train, df_val, word_vectorizer)

    # Step 4b — Random Forest
    clf_rf, rf_val_acc = step4b_bow_rf(df_train, df_val, word_vectorizer)

    # Step 5 — Pipeline + tuning + test evaluation
    lr_test_acc, rf_test_acc = step5_pipeline_and_tuning(df_train, df_val, df_test, tokenizer_fn)

    # Final summary
    print_summary(lr_val_acc, rf_val_acc, lr_test_acc, rf_test_acc)

    total_time = time.time() - overall_start
    log.info(f"\n  Total runtime: {total_time/60:.1f} minutes")
    log.info("  Pipeline complete.\n")
