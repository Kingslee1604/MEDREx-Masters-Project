"""
MEDREx — Save Trained Models for the Explainer UI
==================================================
Run this ONCE before launching the Streamlit explainer:

  python demo/save_models.py

What it does:
  1. Reproduces Cedars-Sinai BoW split + trains CountVectorizer + LR
  2. Trains TF-IDF vectorizer (shared by Methods 1 & 2)
  3. Trains Method 1: TF-IDF + Logistic Regression
  4. Trains Method 2: TF-IDF + LinearSVC
  5. Saves all artifacts to  demo/saved_models/

Outputs (demo/saved_models/):
  cedars_vocabulary.json      — BoW vocabulary dict (word → int index)
  cedars_clf.joblib           — Cedars-Sinai Logistic Regression
  tfidf_vectorizer.joblib     — TF-IDF vectorizer (Methods 1 & 2 share this)
  m1_lr_clf.joblib            — Method 1: Logistic Regression
  m2_svm_clf.joblib           — Method 2: LinearSVC (SVM)
"""

import os, sys, random, time, json
import pandas as pd
import numpy as np
import joblib
import nltk
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
import warnings
warnings.filterwarnings("ignore")

BASE_DIR     = r"c:\Users\kings\OneDrive\Desktop\CSUDH_Kingslee\CSUDH-Spring-2026\CSC-590-MastersProject\MastersProject"
REPORTS_PATH = os.path.join(BASE_DIR, "DataSet", "TCGA_Reports.csv", "TCGA_Reports.csv")
LABELS_PATH  = os.path.join(BASE_DIR, "DataSet", "tcga_patient_to_cancer_type.csv")
SAVE_DIR     = os.path.join(BASE_DIR, "demo", "saved_models")
os.makedirs(SAVE_DIR, exist_ok=True)

t_total = time.time()
print("=" * 65)
print("  MEDREx — Model Training & Save Script")
print("  Output:", SAVE_DIR)
print("=" * 65)

# ── Load dataset ──────────────────────────────────────────────────────────
print("\nLoading dataset...")
df_reports = pd.read_csv(REPORTS_PATH)
df_reports["patient_id"] = df_reports["patient_filename"].apply(lambda x: x.split(".")[0])
df_reports = df_reports[["patient_id", "text"]]
df_reports.index = df_reports["patient_id"].values

df_labels = pd.read_csv(LABELS_PATH)
df_labels.index = df_labels["patient_id"].values
df_reports["cancer_type"] = df_labels.loc[df_reports.index, "cancer_type"]
df_reports = df_reports.dropna(subset=["cancer_type"])
print(f"  {len(df_reports):,} reports | {df_reports['cancer_type'].nunique()} cancer types")

# ── NLTK setup ────────────────────────────────────────────────────────────
print("\nDownloading NLTK data...")
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)
_stopwords = set(nltk.corpus.stopwords.words("english"))

def cs_tokenizer(text):
    tokens = word_tokenize(text.lower())
    tokens = [t for t in tokens if len(t) > 1]
    tokens = [t for t in tokens if t not in _stopwords]
    tokens = [t for t in tokens if t.isalpha()]
    return tokens


# ════════════════════════════════════════════════════════════════════════
# [1/3]  Cedars-Sinai — BoW + Logistic Regression
#        Split: 50/20/30, random.seed(0), NON-stratified (matches original)
# ════════════════════════════════════════════════════════════════════════
print("\n[1/3] Cedars-Sinai BoW + LR ...")
random.seed(0)
arr_all_ids = df_reports["patient_id"].values.tolist()
n_test          = round(0.30 * len(df_reports))
n_val           = round(0.20 * len(df_reports))
arr_test_ids    = random.sample(arr_all_ids, n_test)
_test_set       = set(arr_test_ids)
arr_trainval    = [p for p in arr_all_ids if p not in _test_set]
arr_val_ids     = random.sample(arr_trainval, n_val)
_val_set        = set(arr_val_ids)
arr_train_ids   = [p for p in arr_trainval if p not in _val_set]

df_train_cs = df_reports.loc[arr_train_ids]
print(f"  Train: {len(df_train_cs):,}  |  Val: {n_val:,}  |  Test: {n_test:,}")

print("  Fitting CountVectorizer (NLTK tokenizer)...")
t0 = time.time()
cs_vec = CountVectorizer(tokenizer=cs_tokenizer, token_pattern=None, lowercase=False)
cs_vec.fit(df_train_cs["text"].tolist())
print(f"  Vocabulary: {len(cs_vec.vocabulary_):,} words  ({time.time()-t0:.1f}s)")

print("  Training LogisticRegression(random_state=0, max_iter=200)...")
t0 = time.time()
X_cs = cs_vec.transform(df_train_cs["text"].tolist())
cs_clf = LogisticRegression(random_state=0, max_iter=200)
cs_clf.fit(X_cs, df_train_cs["cancer_type"].values)
print(f"  Done in {time.time()-t0:.1f}s")

with open(os.path.join(SAVE_DIR, "cedars_vocabulary.json"), "w") as f:
    json.dump(cs_vec.vocabulary_, f)
joblib.dump(cs_clf, os.path.join(SAVE_DIR, "cedars_clf.joblib"))
print("  Saved: cedars_vocabulary.json, cedars_clf.joblib")


# ════════════════════════════════════════════════════════════════════════
# [2/3]  TF-IDF vectorizer + Method 1 LR
#        Split: 70/30 stratified, random_state=42  (same as methods 1-4)
# ════════════════════════════════════════════════════════════════════════
print("\n[2/3] Method 1 — TF-IDF + LogisticRegression ...")
X_train, X_test, y_train, y_test = train_test_split(
    df_reports["text"],
    df_reports["cancer_type"],
    test_size=0.3,
    random_state=42,
    stratify=df_reports["cancer_type"]
)
print(f"  Train: {len(X_train):,}  |  Test: {len(X_test):,}")

print("  Fitting TfidfVectorizer(max_features=15000, ngram_range=(1,2), sublinear_tf=True, min_df=2)...")
t0 = time.time()
tfidf_vec = TfidfVectorizer(max_features=15000, ngram_range=(1, 2), sublinear_tf=True, min_df=2)
X_train_tfidf = tfidf_vec.fit_transform(X_train.tolist())
print(f"  Vocabulary: {len(tfidf_vec.vocabulary_):,} features  ({time.time()-t0:.1f}s)")

print("  Training LogisticRegression(C=1.0, max_iter=1000, class_weight=balanced, solver=lbfgs)...")
t0 = time.time()
m1_clf = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced",
                             solver="lbfgs", random_state=42)
m1_clf.fit(X_train_tfidf, y_train)
print(f"  Done in {time.time()-t0:.1f}s")

joblib.dump(tfidf_vec, os.path.join(SAVE_DIR, "tfidf_vectorizer.joblib"))
joblib.dump(m1_clf,    os.path.join(SAVE_DIR, "m1_lr_clf.joblib"))
print("  Saved: tfidf_vectorizer.joblib, m1_lr_clf.joblib")


# ════════════════════════════════════════════════════════════════════════
# [3/3]  Method 2 — TF-IDF + LinearSVC  (reuses same TF-IDF matrix)
# ════════════════════════════════════════════════════════════════════════
print("\n[3/3] Method 2 — TF-IDF + LinearSVC ...")
print("  Training LinearSVC(C=1.0, max_iter=2000, class_weight=balanced)...")
t0 = time.time()
m2_clf = LinearSVC(C=1.0, max_iter=2000, class_weight="balanced", random_state=42)
m2_clf.fit(X_train_tfidf, y_train)
print(f"  Done in {time.time()-t0:.1f}s")

joblib.dump(m2_clf, os.path.join(SAVE_DIR, "m2_svm_clf.joblib"))
print("  Saved: m2_svm_clf.joblib")


print("\n" + "=" * 65)
print(f"  All models saved to: {SAVE_DIR}")
print(f"  Total time: {(time.time()-t_total)/60:.1f} min")
print("=" * 65)
print("\nNext step:")
print("  streamlit run demo/explainer.py")
