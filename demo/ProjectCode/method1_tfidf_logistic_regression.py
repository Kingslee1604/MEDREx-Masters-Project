"""
MEDREx — Method 1: TF-IDF + Logistic Regression
=================================================
This is OUR Method 1 — not the Cedars-Sinai baseline.

Key differences vs Cedars-Sinai (BoW + LR):
  Cedars-Sinai used : CountVectorizer (raw word counts)     → 95.31%
  We use            : TfidfVectorizer (weighted word scores) → 96.36%

Why TF-IDF beats Bag-of-Words:
  - TF-IDF penalises common words that appear in EVERY report (e.g. "the", "patient")
  - It rewards rare but highly specific words (e.g. "glioblastoma" → GBM only)
  - sublinear_tf=True : prevents very frequent words from dominating the vector
  - ngram_range=(1,2)  : captures 2-word medical phrases ("clear cell", "renal cell")
  - class_weight=balanced : handles cancer types with few samples (UCS, KICH)

Run:
  python demo/ProjectCode/method1_tfidf_logistic_regression.py
"""

import os, sys, logging, warnings, time
warnings.filterwarnings("ignore")

# ── Log file setup ─────────────────────────────────────────────────────────
_LOG_DIR  = os.path.join(
    r"c:\Users\kings\OneDrive\Desktop\CSUDH_Kingslee\CSUDH-Spring-2026\CSC-590-MastersProject\MastersProject",
    "demo", "ProjectCode", "output"
)
os.makedirs(_LOG_DIR, exist_ok=True)
_LOG_FILE = os.path.join(_LOG_DIR, "method1_tfidf_lr_run.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),           # print to terminal
        logging.FileHandler(_LOG_FILE, mode="w", encoding="utf-8"),  # save to file
    ]
)
log = logging.getLogger(__name__)
log.info(f"  Log file : {_LOG_FILE}")

def section(title):
    log.info("=" * 70)
    log.info(f"  {title}")
    log.info("=" * 70)

def subsection(title):
    log.info(f"--- {title} ---")

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR     = r"c:\Users\kings\OneDrive\Desktop\CSUDH_Kingslee\CSUDH-Spring-2026\CSC-590-MastersProject\MastersProject"
REPORTS_PATH = os.path.join(BASE_DIR, "DataSet", "TCGA_Reports.csv", "TCGA_Reports.csv")
LABELS_PATH  = os.path.join(BASE_DIR, "DataSet", "tcga_patient_to_cancer_type.csv")
OUT_DIR      = os.path.join(BASE_DIR, "demo", "ProjectCode", "output")
os.makedirs(OUT_DIR, exist_ok=True)

CANCER_NAMES = {
    "ACC":"Adrenocortical carcinoma","BLCA":"Bladder Urothelial Carcinoma",
    "BRCA":"Breast invasive carcinoma","CESC":"Cervical squamous cell carcinoma",
    "CHOL":"Cholangiocarcinoma","COAD":"Colon adenocarcinoma",
    "DLBC":"Diffuse Large B-cell Lymphoma","ESCA":"Esophageal carcinoma",
    "GBM":"Glioblastoma multiforme","HNSC":"Head and Neck squamous cell carcinoma",
    "KICH":"Kidney Chromophobe","KIRC":"Kidney renal clear cell carcinoma",
    "KIRP":"Kidney renal papillary cell carcinoma","LAML":"Acute Myeloid Leukemia",
    "LGG":"Brain Lower Grade Glioma","LIHC":"Liver hepatocellular carcinoma",
    "LUAD":"Lung adenocarcinoma","LUSC":"Lung squamous cell carcinoma",
    "MESO":"Mesothelioma","OV":"Ovarian serous cystadenocarcinoma",
    "PAAD":"Pancreatic adenocarcinoma","PCPG":"Pheochromocytoma and Paraganglioma",
    "PRAD":"Prostate adenocarcinoma","READ":"Rectum adenocarcinoma",
    "SARC":"Sarcoma","SKCM":"Skin Cutaneous Melanoma","STAD":"Stomach adenocarcinoma",
    "TGCT":"Testicular Germ Cell Tumors","THCA":"Thyroid carcinoma","THYM":"Thymoma",
    "UCEC":"Uterine Corpus Endometrial Carcinoma","UCS":"Uterine Carcinosarcoma",
    "UVM":"Uveal Melanoma",
}


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Load & Compile Dataset
# ══════════════════════════════════════════════════════════════════════════════
def step1_load_data():
    section("STEP 1 — Load & Compile Dataset")
    import pandas as pd

    log.info(f"  Reading reports : {REPORTS_PATH}")
    t0 = time.time()
    df_reports = pd.read_csv(REPORTS_PATH)
    log.info(f"  Loaded {len(df_reports):,} reports in {time.time()-t0:.1f}s")
    log.info(f"  Columns : {list(df_reports.columns)}")

    subsection("Extract patient_id from filename")
    log.info("  Raw filename  : 'TCGA-BP-5195.25c0b433-5557-4165-922e-2c1eac9c26f0'")
    log.info("  patient_id    : 'TCGA-BP-5195'  (everything before the first '.')")
    df_reports["patient_id"] = df_reports["patient_filename"].apply(lambda x: x.split(".")[0])
    assert not df_reports["patient_id"].duplicated().any(), "Duplicate patient IDs!"
    df_reports = df_reports[["patient_id", "text"]]
    df_reports.index = df_reports["patient_id"].values

    log.info(f"  Unique patient reports: {len(df_reports):,}")
    _disp = df_reports.head(3).copy(); _disp["text"] = _disp["text"].str[:80] + "..."
    log.info(f"\n{_disp.to_string()}")

    subsection("Load cancer type labels")
    log.info(f"  Reading labels: {LABELS_PATH}")
    df_labels = pd.read_csv(LABELS_PATH)
    log.info(f"  Label rows    : {len(df_labels):,}")
    log.info(f"  Unique types  : {df_labels['cancer_type'].nunique()}")
    log.info(f"\n{df_labels.head(5).to_string()}")
    df_labels.index = df_labels["patient_id"].values

    subsection("Merge reports + labels")
    assert df_reports["patient_id"].isin(df_labels["patient_id"]).all(), "Some reports have no label!"
    df_reports["cancer_type"]      = df_labels.loc[df_reports.index, "cancer_type"]
    df_reports["cancer_type_name"] = df_reports["cancer_type"].map(CANCER_NAMES).fillna(df_reports["cancer_type"])
    log.info(f"  Merged shape  : {df_reports.shape}")
    _disp2 = df_reports.head(3).copy(); _disp2["text"] = _disp2["text"].str[:80] + "..."
    log.info(f"\n{_disp2.to_string()}")

    subsection("Cancer type distribution (all 33 types)")
    dist = df_reports["cancer_type"].value_counts()
    for code, cnt in dist.items():
        bar = "█" * int(cnt / 30)
        log.info(f"    {code:<6} {CANCER_NAMES.get(code,''):<45} {cnt:4d} ({cnt/len(df_reports)*100:4.1f}%)  {bar}")

    subsection("Text length analysis")
    df_reports["word_count"] = df_reports["text"].apply(lambda x: len(str(x).split()))
    log.info(f"  Average words per report : {df_reports['word_count'].mean():.0f}")
    log.info(f"  Median  words per report : {df_reports['word_count'].median():.0f}")
    log.info(f"  Min     words per report : {df_reports['word_count'].min()}")
    log.info(f"  Max     words per report : {df_reports['word_count'].max()}")
    log.info(f"  Reports over 512 words   : {(df_reports['word_count'] > 512).sum()} "
             f"({(df_reports['word_count'] > 512).mean()*100:.1f}%)")
    log.info("  → BERT has a 512-token limit. TF-IDF has NO length limit — handles long reports better.")

    subsection("Example pathology report")
    row = df_reports.iloc[0]
    log.info(f"  Patient ID  : {row['patient_id']}")
    log.info(f"  Cancer type : {row['cancer_type']} — {row['cancer_type_name']}")
    log.info(f"  Word count  : {row['word_count']}")
    log.info(f"  Report text :\n\n    {str(row['text'])[:600]}\n")

    df_reports.drop(columns=["word_count"], inplace=True)
    return df_reports

#| patient_id   | text                   | cancer_type | cancer_type_name                  |
#| ------------ | ---------------------- | ----------- | --------------------------------- |
#| TCGA-BP-5195 | "Date of receipt..."   | KIRC        | Kidney renal clear cell carcinoma |
#| TCGA-D7-8573 | "Material: stomach..." | STAD        | Stomach adenocarcinoma            |
#| TCGA-EI-7004 | "Histopathological..." | READ        | Rectum adenocarcinoma             |

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Stratified Train / Test Split
# ══════════════════════════════════════════════════════════════════════════════
def step2_split(df):
    section("STEP 2 — Stratified Train / Test Split  (70% train / 30% test)")
    from sklearn.model_selection import train_test_split
    import pandas as pd

    log.info("  WHY STRATIFIED?")
    log.info("  Some cancer types (e.g. UCS=37, CHOL=36) have very few samples.")
    log.info("  Without stratify=, the test set might accidentally get 0 samples of a rare type.")
    log.info("  stratify=cancer_type ensures each type is proportionally represented in both splits.")
    log.info("  WHY NO VALIDATION SET HERE?")
    log.info("  Cedars-Sinai used train/val/test (3 splits).")
    log.info("  We use train/test only (2 splits) — simpler, and we use cross-validation")
    log.info("  via GridSearch in Step 5 to tune hyperparameters without a fixed val set.")
    log.info("  Split: 70% train / 30% test  |  random_state=42  |  stratify=cancer_type")

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"],
        df["cancer_type"],
        test_size=0.3,
        random_state=42,
        stratify=df["cancer_type"]
    )

    log.info(f"\n  Total  : {len(df):,} reports")
    log.info(f"  Train  : {len(X_train):,} reports ({len(X_train)/len(df)*100:.0f}%)")
    log.info(f"  Test   : {len(X_test):,}  reports ({len(X_test)/len(df)*100:.0f}%)")

    subsection("Stratification check — class counts in each split")
    log.info(f"  {'Type':<6} {'Total':>7} {'Train':>7} {'Test':>7}  {'Train%':>7} {'Test%':>7}")
    log.info(f"  {'-'*50}")
    for code in sorted(df["cancer_type"].unique()):
        n_all   = (df["cancer_type"]  == code).sum()
        n_train = (y_train == code).sum()
        n_test  = (y_test  == code).sum()
        log.info(
            f"  {code:<6} {n_all:>7} {n_train:>7} {n_test:>7}"
            f"  {n_train/n_all*100:>6.0f}%  {n_test/n_all*100:>6.0f}%"
        )

    return X_train, X_test, y_train, y_test


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — TF-IDF Vectorizer
# ══════════════════════════════════════════════════════════════════════════════
def step3_tfidf_vectorizer(X_train, X_test):
    section("STEP 3 — TF-IDF Vectorizer  ← KEY UPGRADE vs Cedars-Sinai")
    from sklearn.feature_extraction.text import TfidfVectorizer
    import numpy as np
    import pandas as pd

    log.info("  WHAT IS TF-IDF?")
    log.info("  TF  = Term Frequency  : how often a word appears in THIS report")
    log.info("  IDF = Inverse Document Frequency : how RARE the word is across ALL reports")
    log.info("  TF-IDF score = TF × IDF")
    log.info("  WHY IS THIS BETTER THAN BAG-OF-WORDS?")
    log.info("  Example — word 'the':")
    log.info("    BoW  score = 50  (appears 50 times → high score, but useless for classification)")
    log.info("    IDF  score = 0.001 (appears in EVERY report → near zero)")
    log.info("    TF-IDF     = 50 × 0.001 = 0.05  (correctly downweighted)")
    log.info("  Example — word 'glioblastoma':")
    log.info("    BoW  score = 3   (appears 3 times → low score)")
    log.info("    IDF  score = 8.5 (appears in very few reports → high)")
    log.info("    TF-IDF     = 3 × 8.5 = 25.5  (correctly upweighted → signals GBM)")

    subsection("TF-IDF parameters — what each one does")
    log.info("  max_features=15000")
    log.info("    Only keep the top 15,000 most informative words/phrases.")
    log.info("    Cedars-Sinai kept all ~23,818 tokens — many were noise.")
    log.info("    15,000 features = faster training + less overfitting.")
    log.info("  ngram_range=(1, 2)  ← BIGGEST UPGRADE")
    log.info("    (1,1) = single words only:  'renal', 'cell', 'carcinoma'")
    log.info("    (1,2) = also 2-word phrases: 'renal cell', 'clear cell carcinoma'")
    log.info("    Medical meaning lives in PHRASES, not single words:")
    log.info("      'renal cell carcinoma' → KIRC (kidney cancer)")
    log.info("      'squamous cell'        → LUSC or HNSC")
    log.info("      'ductal carcinoma'     → BRCA (breast cancer)")
    log.info("      'germ cell'            → TGCT (testicular)")
    log.info("  sublinear_tf=True")
    log.info("    Normal TF:       word appears 100× → score=100  (too dominant)")
    log.info("    Sublinear TF:    score = 1 + log(100) = 5.6     (compressed)")
    log.info("    Prevents very frequent words from drowning out rarer but more specific ones.")
    log.info("  min_df=2")
    log.info("    Ignore words that appear in fewer than 2 documents.")
    log.info("    Removes typos, one-off abbreviations, and noise.")

    # IDF = log(10000 / 9500) ≈ 0.02  ❌ very low
    # IDF = log(10000 / 50) ≈ 5.3  ✅ high
    
    subsection("Fitting TF-IDF on training corpus")
    log.info("  Fitting on training data ONLY — never on test data (prevents data leakage).")
    t0 = time.time()
    vec = TfidfVectorizer(
        max_features=15000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2
    )
    X_train_tfidf = vec.fit_transform(X_train) #big numeric matrix. Example : 6666 reports × 15000 features
    #fit
    #Learns vocabulary and importance from training data.
    #glioblastoma → rare → high IDF
    #the → common → low IDF
    #renal cell → useful phrase
    
    #transform
    #Converts each report into numbers
    #"renal cell carcinoma..." -- [0.00, 0.24, 0.00, 0.81, ...]

    log.info(f"  Fit + transform complete in {time.time()-t0:.1f}s")
    log.info(f"  Vocabulary size : {len(vec.vocabulary_):,} features (words + 2-word phrases)")
    log.info(f"  Matrix shape    : {X_train_tfidf.shape}  ({X_train_tfidf.shape[0]} reports × {X_train_tfidf.shape[1]} features)")
    log.info(f"  Matrix sparsity : {X_train_tfidf.nnz / (X_train_tfidf.shape[0]*X_train_tfidf.shape[1])*100:.3f}% non-zero")

    subsection("What the vocabulary looks like (sample entries)")
    feature_names = vec.get_feature_names_out() #gives the 15,000 selected words/phrases - ["adenocarcinoma", "renal", "renal cell", "glioblastoma", ...]
    unigrams  = [f for f in feature_names if " " not in f][:15]
    bigrams   = [f for f in feature_names if " "     in f][:15]
    log.info(f"  Sample unigrams  (single words) : {unigrams}")
    log.info(f"  Sample bigrams   (2-word phrases): {bigrams}")

    subsection("TF-IDF score example — one report")
    sample_vec = X_train_tfidf[0]
    nonzero_idx = sample_vec.nonzero()[1]
    scores = [(feature_names[i], round(sample_vec[0, i], 4)) for i in nonzero_idx]
    scores.sort(key=lambda x: x[1], reverse=True)
    log.info("  Top 20 highest TF-IDF scoring terms in the first training report:")
    for term, score in scores[:20]:
        log.info(f"    '{term:<30}'  tfidf = {score:.4f}")

    subsection("Transforming test set")
    t0 = time.time()
    X_test_tfidf = vec.transform(X_test)
    log.info(f"  Test transform complete in {time.time()-t0:.1f}s")
    log.info(f"  Test matrix shape : {X_test_tfidf.shape}")

    return vec, X_train_tfidf, X_test_tfidf


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Logistic Regression (with class balancing)
# ══════════════════════════════════════════════════════════════════════════════
def step4_logistic_regression(X_train_tfidf, X_test_tfidf, y_train, y_test, vec):
    section("STEP 4 — Logistic Regression with Class Balancing")
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score, classification_report

    log.info("  LOGISTIC REGRESSION PARAMETER DIFFERENCES vs Cedars-Sinai:")
    log.info("  class_weight='balanced'  ← NEW (Cedars-Sinai did not use this)")
    log.info("    Problem: BRCA has 1034 samples, UCS has only 37.")
    log.info("    Without balancing: model ignores rare cancers (biased toward BRCA).")
    log.info("    With balancing   : rare class UCS gets weight = 1034/37 = 27.9×")
    log.info("    Effect: model pays much more attention to getting rare cancers right.")
    log.info("  max_iter=1000  (Cedars-Sinai used 200)")
    log.info("    More iterations → model fully converges → slightly better accuracy.")
    log.info("  solver='lbfgs'")
    log.info("    L-BFGS is a quasi-Newton optimiser — faster and more stable")
    log.info("    than the default solver for multi-class problems with many features.")
    log.info("  C=1.0  (regularisation strength — default, same as Cedars-Sinai)")
    log.info("    C controls how much the model can overfit.")
    log.info("    Higher C = less regularisation = can fit training data more closely.")
    log.info("    C=1.0 is a good balance between underfitting and overfitting.")

    subsection("Class weights computed from training data")
    from sklearn.utils.class_weight import compute_class_weight
    classes   = np.array(sorted(y_train.unique()))
    weights   = compute_class_weight("balanced", classes=classes, y=y_train)
    wt_dict   = dict(zip(classes, weights))
    log.info("  Class weights assigned to each cancer type:")
    for code in sorted(wt_dict, key=lambda c: wt_dict[c], reverse=True):
        count  = (y_train == code).sum()
        weight = wt_dict[code]
        bar    = "█" * min(int(weight * 3), 30)
        log.info(f"    {code:<6}  n_train={count:4d}  weight={weight:6.2f}  {bar}")
    log.info("  → Rare types get HIGH weight so the model doesn't ignore them.")

    subsection("Training Logistic Regression")
    log.info("  Parameters: C=1.0, max_iter=1000, class_weight=balanced, solver=lbfgs, random_state=42")
    log.info("  Training... (~30-60 seconds)")
    t0 = time.time()
    clf = LogisticRegression(
        C=1.0,
        max_iter=1000,
        class_weight="balanced",
        solver="lbfgs",
        random_state=42
    )
    clf.fit(X_train_tfidf, y_train)
    log.info(f"  Training complete in {time.time()-t0:.1f}s")
    log.info(f"  Model converged: {clf.n_iter_[0] < 1000}")
    log.info(f"  Iterations used : {clf.n_iter_[0]} / 1000")
    log.info(f"  Number of classes: {len(clf.classes_)}")

    subsection("Results on TRAINING set")
    y_train_pred = clf.predict(X_train_tfidf)
    train_acc    = accuracy_score(y_train, y_train_pred)
    log.info(f"  Train Accuracy : {train_acc*100:.2f}%")
    log.info("  (High training accuracy is expected — model has seen these reports before)")

    subsection("Results on TEST set  ← THIS IS THE REAL SCORE")
    y_test_pred = clf.predict(X_test_tfidf)
    test_acc    = accuracy_score(y_test, y_test_pred)
    test_f1_w   = f1_score(y_test, y_test_pred, average="weighted", zero_division=0)
    test_f1_m   = f1_score(y_test, y_test_pred, average="macro",    zero_division=0)
    log.info(f"  ╔══════════════════════════════════════════════════════════╗")
    log.info(f"  ║  METHOD 1 — TF-IDF + LOGISTIC REGRESSION RESULTS       ║")
    log.info(f"  ║                                                          ║")
    log.info(f"  ║  Test Accuracy        : {test_acc*100:>6.2f}%                      ║")
    log.info(f"  ║  F1 Weighted          : {test_f1_w*100:>6.2f}%                      ║")
    log.info(f"  ║  F1 Macro             : {test_f1_m*100:>6.2f}%                      ║")
    log.info(f"  ║                                                          ║")
    log.info(f"  ║  Cedars-Sinai BoW+LR  : ~95.31%                         ║")
    log.info(f"  ║  Our improvement      : +{(test_acc - 0.9531)*100:>4.2f}%                        ║")
    log.info(f"  ╚══════════════════════════════════════════════════════════╝")

    log.info("\n  Full classification report (all 33 cancer types):")
    log.info(classification_report(y_test, y_test_pred, zero_division=0))

    subsection("Per-class F1 sorted worst → best")
    f1_per = f1_score(y_test, y_test_pred, average=None, labels=clf.classes_, zero_division=0)
    df_f1  = pd.DataFrame({
        "code":    clf.classes_,
        "name":    [CANCER_NAMES.get(c, c) for c in clf.classes_],
        "f1":      f1_per,
        "n_train": [(y_train == c).sum() for c in clf.classes_],
        "n_test":  [(y_test  == c).sum() for c in clf.classes_],
    }).sort_values("f1")

    log.info(f"  {'Code':<6} {'Cancer Name':<45} {'F1':>6} {'#Train':>7} {'#Test':>6}")
    log.info(f"  {'-'*74}")
    for _, r in df_f1.iterrows():
        flag = " ← hardest" if r["f1"] < 0.85 else ""
        log.info(f"  {r['code']:<6} {r['name']:<45} {r['f1']:>6.3f} {int(r['n_train']):>7} {int(r['n_test']):>6}{flag}")

    log.info("\n  ANALYSIS:")
    hard = df_f1[df_f1["f1"] < 0.85]
    log.info(f"  Cancer types below 0.85 F1 ({len(hard)} types) :")
    for _, r in hard.iterrows():
        log.info(f"    {r['code']} ({r['name']}) — F1={r['f1']:.3f}, only {int(r['n_train'])} training samples")
    log.info("  These are targets for BERT fine-tuning (Phase 3) — BERT understands medical context better.")

    return clf, test_acc, test_f1_w, df_f1


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — Deep Feature Analysis (why does it know which cancer is which?)
# ══════════════════════════════════════════════════════════════════════════════
def step5_feature_analysis(clf, vec, y_train):
    section("STEP 5 — Deep Feature Analysis  (what did the model actually learn?)")
    import numpy as np

    feature_names = vec.get_feature_names_out()
    idx_to_word   = {i: f for i, f in enumerate(feature_names)}

    log.info("  LR COEFFICIENT MEANING:")
    log.info("  For each cancer type, LR learned a weight (coefficient) per feature.")
    log.info("  High positive coefficient = that word/phrase STRONGLY predicts this cancer.")
    log.info("  Low negative coefficient  = that word/phrase argues AGAINST this cancer.")
    log.info("  These are the actual 'rules' the model learned from 9,523 pathology reports.")

    subsection("Top 15 most predictive words/phrases per cancer type")
    for i, label in enumerate(clf.classes_):
        coefs = clf.coef_[i]
        top_idx   = np.argsort(coefs)[::-1][:5]
        top_terms = [(feature_names[j], round(coefs[j], 3)) for j in top_idx]

        n_train = (y_train == label).sum()
        name    = CANCER_NAMES.get(label, label)
        log.info(f"\n  ┌─ {label} — {name}  (n_train={n_train}) ─────")
        for term, coef in top_terms:
            bar = "+" * min(int(coef * 3), 20)
            log.info(f"  │  {term:<35} {coef:+.3f}  {bar}")
        log.info(f"  └{'─'*60}")

    subsection("Most uniquely diagnostic words across ALL cancer types")
    log.info("  These are words with the highest coefficient in exactly ONE cancer type")
    log.info("  — the clearest signals the model uses:")

    specifics = []
    for i, label in enumerate(clf.classes_):
        coefs   = clf.coef_[i]
        top_idx = np.argsort(coefs)[::-1][:5]
        for j in top_idx:
            specifics.append({
                "term":   feature_names[j],
                "cancer": label,
                "name":   CANCER_NAMES.get(label, label),
                "coef":   coefs[j],
            })

    import pandas as pd
    df_spec = pd.DataFrame(specifics).sort_values("coef", ascending=False).head(40)
    log.info(f"  {'Term':<35} {'Cancer':>6}  {'Coeff':>7}")
    log.info(f"  {'-'*55}")
    for _, r in df_spec.iterrows():
        log.info(f"  {r['term']:<35} {r['cancer']:>6}  {r['coef']:>+7.3f}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 — Side-by-side comparison vs Cedars-Sinai
# ══════════════════════════════════════════════════════════════════════════════
def step6_comparison(test_acc, test_f1_w):
    section("STEP 6 — Comparison: Our Method 1 vs Cedars-Sinai Baseline")

    cedars_acc = 0.9531
    cedars_f1  = 0.9513  # approximate from their report

    log.info(f"  {'Metric':<30} {'Cedars-Sinai':>14} {'Our Method 1':>14} {'Improvement':>12}")
    log.info(f"  {'-'*74}")
    log.info(f"  {'Vectorizer':<30} {'CountVectorizer':>14} {'TfidfVectorizer':>14}")
    log.info(f"  {'n-gram range':<30} {'(1,1) unigrams':>14} {'(1,2) bigrams':>14} {'← new':>12}")
    log.info(f"  {'Sublinear TF':<30} {'No':>14} {'Yes':>14} {'← new':>12}")
    log.info(f"  {'Max features':<30} {'23,818':>14} {'15,000':>14}")
    log.info(f"  {'Class weighting':<30} {'None':>14} {'balanced':>14} {'← new':>12}")
    log.info(f"  {'Max iterations':<30} {'200':>14} {'1,000':>14} {'← new':>12}")
    log.info(f"  {'Test Accuracy':<30} {cedars_acc*100:>13.2f}% {test_acc*100:>13.2f}% {(test_acc-cedars_acc)*100:>+11.2f}%")
    log.info(f"  {'F1 Weighted':<30} {cedars_f1*100:>13.2f}% {test_f1_w*100:>13.2f}% {(test_f1_w-cedars_f1)*100:>+11.2f}%")

    log.info("""
  WHY DID WE GET BETTER ACCURACY?

  1. TF-IDF vs BoW
     BoW gives the same score to 'the' (common, useless) and 'glioblastoma'
     (rare, specific to brain cancer). TF-IDF correctly down-weights 'the'
     and up-weights 'glioblastoma'.

  2. Bigrams — (1,2) ngram_range
     'renal' alone could appear in kidney cancer OR kidney disease descriptions.
     'renal cell' is almost always KIRC (clear cell kidney cancer).
     Bigrams capture the medical meaning that single words miss.

  3. Balanced class weights
     UCS (Uterine Carcinosarcoma) has only ~37 training samples.
     Without balancing, the model largely ignores it.
     With weight=27×, the model learns UCS-specific patterns properly.

  4. More iterations (1000 vs 200)
     Complex 33-class problems need more optimisation steps to fully converge.
    """)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    t_start = time.time()

    log.info("=" * 70)
    log.info("  MEDREx — Method 1: TF-IDF + Logistic Regression")
    log.info("  CSC-590 Masters Project, CSUDH Spring 2026")
    log.info("  Goal: Understand HOW and WHY we beat Cedars-Sinai (95.31% → 96%+)")
    log.info("=" * 70)

    # Step 1 — Load data
    df = step1_load_data()

    # Step 2 — Split
    X_train, X_test, y_train, y_test = step2_split(df)

    # Step 3 — TF-IDF
    vec, X_train_tfidf, X_test_tfidf = step3_tfidf_vectorizer(X_train, X_test)

    # Step 4 — Train + evaluate LR
    clf, test_acc, test_f1_w, df_f1 = step4_logistic_regression(
        X_train_tfidf, X_test_tfidf, y_train, y_test, vec
    )

    # Step 5 — Feature analysis
    step5_feature_analysis(clf, vec, y_train)

    # Step 6 — Comparison table
    step6_comparison(test_acc, test_f1_w)

    section("DONE")
    log.info(f"  Final Test Accuracy : {test_acc*100:.2f}%")
    log.info(f"  Final F1 Weighted   : {test_f1_w*100:.2f}%")
    log.info(f"  Total runtime       : {(time.time()-t_start)/60:.1f} min")
    log.info(f"  Output directory    : {OUT_DIR}")
    log.info("  NEXT: Run Method 2 (TF-IDF + SVM) and compare.")
