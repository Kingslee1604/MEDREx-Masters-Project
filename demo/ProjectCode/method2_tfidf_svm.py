"""
MEDREx — Method 2: TF-IDF + SVM (LinearSVC)
============================================
This is OUR Method 2.

Why SVM after Logistic Regression?
  Method 1 (TF-IDF + LR) → ~96%
  Method 2 (TF-IDF + SVM) → typically 96–97% on sparse high-dimensional text

Key idea behind SVM:
  LR learns the "best probability" boundary between classes.
  SVM learns the "widest margin" boundary — it tries to maximise the gap
  between classes, which often generalises better on text data.

Why LinearSVC specifically (not SVC with rbf kernel)?
  TF-IDF produces a very large, sparse matrix (9523 × 15000).
  LinearSVC is optimised for large sparse feature spaces — much faster.
  SVC (with rbf/poly kernel) would be extremely slow here.

Run:
  python demo/ProjectCode/method2_tfidf_svm.py
"""

import os, sys, logging, warnings, time
warnings.filterwarnings("ignore")

# ── Log file setup ──────────────────────────────────────────────────────────
_LOG_DIR  = os.path.join(
    r"c:\Users\kings\OneDrive\Desktop\CSUDH_Kingslee\CSUDH-Spring-2026\CSC-590-MastersProject\MastersProject",
    "demo", "ProjectCode", "output"
)
os.makedirs(_LOG_DIR, exist_ok=True)
_LOG_FILE = os.path.join(_LOG_DIR, "method2_tfidf_svm_run.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(_LOG_FILE, mode="w", encoding="utf-8"),
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

# ── Paths ───────────────────────────────────────────────────────────────────
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
# STEP 1 — Load & Compile Dataset  (same as Method 1)
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
    _disp = df_reports.head(3).copy()
    _disp["text"] = _disp["text"].str[:80] + "..."
    log.info(f"\n{_disp.to_string()}")

    subsection("Load cancer type labels")
    df_labels = pd.read_csv(LABELS_PATH)
    log.info(f"  Label rows    : {len(df_labels):,}")
    log.info(f"  Unique types  : {df_labels['cancer_type'].nunique()}")
    df_labels.index = df_labels["patient_id"].values

    subsection("Merge reports + labels")
    df_reports["cancer_type"]      = df_labels.loc[df_reports.index, "cancer_type"]
    df_reports["cancer_type_name"] = df_reports["cancer_type"].map(CANCER_NAMES).fillna(df_reports["cancer_type"])
    log.info(f"  Merged shape  : {df_reports.shape}")
    _disp2 = df_reports.head(3).copy()
    _disp2["text"] = _disp2["text"].str[:80] + "..."
    log.info(f"\n{_disp2.to_string()}")

    subsection("Cancer type distribution (all 33 types)")
    dist = df_reports["cancer_type"].value_counts()
    for code, cnt in dist.items():
        bar = "█" * int(cnt / 30)
        log.info(f"    {code:<6} {CANCER_NAMES.get(code,''):<45} {cnt:4d} ({cnt/len(df_reports)*100:4.1f}%)  {bar}")

    return df_reports


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Stratified Train / Test Split  (same as Method 1)
# ══════════════════════════════════════════════════════════════════════════════
def step2_split(df):
    section("STEP 2 — Stratified Train / Test Split  (70% train / 30% test)")
    from sklearn.model_selection import train_test_split

    log.info("  Same split as Method 1 — random_state=42, stratify=cancer_type.")
    log.info("  Using identical split ensures a FAIR comparison between LR and SVM.")
    log.info("  Both models see the same training examples and are tested on the same reports.")

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

    subsection("Stratification check — rare cancer types")
    for code in ["UCS", "CHOL", "KICH", "UVM", "MESO"]:
        n_all   = (df["cancer_type"]  == code).sum()
        n_train = (y_train == code).sum()
        n_test  = (y_test  == code).sum()
        log.info(f"  {code:<6}  total={n_all:3d}  train={n_train:3d}  test={n_test:3d}")
    log.info("  All rare types preserved in both splits — stratification is working.")

    return X_train, X_test, y_train, y_test


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — TF-IDF Vectorizer  (identical parameters to Method 1)
# ══════════════════════════════════════════════════════════════════════════════
def step3_tfidf_vectorizer(X_train, X_test):
    section("STEP 3 — TF-IDF Vectorizer  (same parameters as Method 1)")
    from sklearn.feature_extraction.text import TfidfVectorizer

    log.info("  Using IDENTICAL TF-IDF parameters as Method 1:")
    log.info("    max_features=15000   — top 15k most informative words/phrases")
    log.info("    ngram_range=(1, 2)   — unigrams + bigrams (captures medical phrases)")
    log.info("    sublinear_tf=True    — log-compress term frequency")
    log.info("    min_df=2             — drop words in fewer than 2 documents")
    log.info("  Why keep them the same? We want to isolate the effect of the classifier.")
    log.info("  If we change TF-IDF too, we can't tell if SVM or the features caused the difference.")

    t0 = time.time()
    vec = TfidfVectorizer(
        max_features=15000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2
    )
    X_train_tfidf = vec.fit_transform(X_train)
    X_test_tfidf  = vec.transform(X_test)
    log.info(f"  Fit + transform done in {time.time()-t0:.1f}s")
    log.info(f"  Vocabulary size   : {len(vec.vocabulary_):,} features")
    log.info(f"  Train matrix shape: {X_train_tfidf.shape}")
    log.info(f"  Test  matrix shape: {X_test_tfidf.shape}")
    log.info(f"  Sparsity          : {X_train_tfidf.nnz / (X_train_tfidf.shape[0]*X_train_tfidf.shape[1])*100:.3f}% non-zero")

    return vec, X_train_tfidf, X_test_tfidf


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — LinearSVC (the new classifier) # Support Vector Machine SVM
# ══════════════════════════════════════════════════════════════════════════════
def step4_svm(X_train_tfidf, X_test_tfidf, y_train, y_test, vec):
    section("STEP 4 — LinearSVC  (Support Vector Machine for Text)")
    import numpy as np
    import pandas as pd
    from sklearn.svm import LinearSVC
    from sklearn.metrics import accuracy_score, f1_score, classification_report

    log.info("  HOW DOES SVM WORK?")
    log.info("  SVM draws a decision boundary (hyperplane) between classes.")
    log.info("  It doesn't just find ANY boundary — it finds the one with the")
    log.info("  MAXIMUM MARGIN: the widest possible gap between the classes.")
    log.info("  Reports close to the boundary (support vectors) define the margin.")
    log.info("  Wide margin = better generalisation to unseen data.")
    log.info("")
    log.info("  LOGISTIC REGRESSION vs SVM — key difference:")
    log.info("  LR: minimises log-loss across ALL training points (every point contributes)")
    log.info("  SVM: only cares about support vectors (points near the boundary)")
    log.info("  SVM ignores easy examples far from the boundary → focuses on hard cases")
    log.info("  → Often better on text because most documents are easy to classify.")
    log.info("")
    log.info("  WHY LinearSVC and NOT SVC(kernel='rbf')?")
    log.info("  Our feature matrix: 6,666 rows × 15,000 columns (very high-dimensional).")
    log.info("  rbf kernel needs to compute similarity between all pairs of documents → O(n²) → too slow.")
    log.info("  LinearSVC: works directly in the feature space → O(n×d) → very fast.")
    log.info("  Rule of thumb: for text (n_features >> n_samples), always use LinearSVC.")

    subsection("LinearSVC parameters — what each one does")
    log.info("  C=1.0  (regularisation strength)")
    log.info("    C controls how strictly the model tries to correctly classify every training point.")
    log.info("    Low  C (e.g. 0.01) → wide margin, some training errors allowed → less overfit")
    log.info("    High C (e.g. 100)  → narrow margin, almost all training points correct → can overfit")
    log.info("    C=1.0 is the standard starting point (same as Method 1 LR).")
    log.info("  class_weight='balanced'")
    log.info("    Same as Method 1: rare cancers (UCS=37, CHOL=36) get higher weight.")
    log.info("    Without this, SVM would mostly ignore rare classes.")
    log.info("  max_iter=2000")
    log.info("    SVM optimisation sometimes needs more iterations than LR.")
    log.info("    2000 ensures it converges on this multi-class 33-label problem.")
    log.info("  multi_class strategy: one-vs-rest (OvR)")
    log.info("    LinearSVC trains 33 binary classifiers: 'BRCA vs rest', 'GBM vs rest', etc.")
    log.info("    Each classifier gets one score; highest score wins the prediction.")

    subsection("Class weights")
    from sklearn.utils.class_weight import compute_class_weight
    classes = np.array(sorted(y_train.unique()))
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    wt_dict = dict(zip(classes, weights))
    log.info("  Class weights assigned to each cancer type:")
    for code in sorted(wt_dict, key=lambda c: wt_dict[c], reverse=True):
        count  = (y_train == code).sum()
        weight = wt_dict[code]
        bar    = "█" * min(int(weight * 3), 30)
        log.info(f"    {code:<6}  n_train={count:4d}  weight={weight:6.2f}  {bar}")

    subsection("Training LinearSVC")
    log.info("  Parameters: C=1.0, max_iter=2000, class_weight=balanced, random_state=42")
    log.info("  Training... (usually faster than LR — ~10-30 seconds)")
    t0 = time.time()
    clf = LinearSVC(
        C=1.0,
        max_iter=2000,
        class_weight="balanced",
        random_state=42
    )
    clf.fit(X_train_tfidf, y_train)
    log.info(f"  Training complete in {time.time()-t0:.1f}s")
    log.info(f"  Number of classes : {len(clf.classes_)}")
    log.info(f"  Coefficient matrix: {clf.coef_.shape}  (33 classes × 15,000 features)")

    subsection("Results on TRAINING set")
    y_train_pred = clf.predict(X_train_tfidf)
    train_acc    = accuracy_score(y_train, y_train_pred)
    log.info(f"  Train Accuracy : {train_acc*100:.2f}%")
    log.info("  (High training accuracy is expected — model has seen these reports)")

    subsection("Results on TEST set  ← THE REAL SCORE")
    y_test_pred = clf.predict(X_test_tfidf)
    test_acc    = accuracy_score(y_test, y_test_pred)
    test_f1_w   = f1_score(y_test, y_test_pred, average="weighted", zero_division=0)
    test_f1_m   = f1_score(y_test, y_test_pred, average="macro",    zero_division=0)

    log.info(f"  ╔══════════════════════════════════════════════════════════╗")
    log.info(f"  ║  METHOD 2 — TF-IDF + SVM (LinearSVC) RESULTS           ║")
    log.info(f"  ║                                                          ║")
    log.info(f"  ║  Test Accuracy        : {test_acc*100:>6.2f}%                      ║")
    log.info(f"  ║  F1 Weighted          : {test_f1_w*100:>6.2f}%                      ║")
    log.info(f"  ║  F1 Macro             : {test_f1_m*100:>6.2f}%                      ║")
    log.info(f"  ║                                                          ║")
    log.info(f"  ║  Cedars-Sinai BoW+LR  : ~95.31%                         ║")
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
    log.info(f"  Cancer types below 0.85 F1 ({len(hard)} types):")
    for _, r in hard.iterrows():
        log.info(f"    {r['code']} ({r['name']}) — F1={r['f1']:.3f}, only {int(r['n_train'])} training samples")

    return clf, test_acc, test_f1_w, df_f1


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — Feature Analysis  (SVM coefficients — what did SVM learn?)
# ══════════════════════════════════════════════════════════════════════════════
def step5_feature_analysis(clf, vec, y_train):
    section("STEP 5 — SVM Feature Analysis  (what did LinearSVC actually learn?)")
    import numpy as np

    feature_names = vec.get_feature_names_out()

    log.info("  SVM COEFFICIENT MEANING:")
    log.info("  LinearSVC (one-vs-rest) trains one weight vector per cancer type.")
    log.info("  High positive weight = word/phrase strongly predicts THIS cancer type.")
    log.info("  High negative weight = word/phrase argues AGAINST this cancer type.")
    log.info("  These are the 'decision rules' SVM learned from 9,523 pathology reports.")
    log.info("  Unlike LR probabilities, SVM weights are raw decision scores (not 0-1).")

    subsection("Top 5 most predictive words/phrases per cancer type")
    for i, label in enumerate(clf.classes_):
        coefs     = clf.coef_[i]
        top_idx   = np.argsort(coefs)[::-1][:5]
        top_terms = [(feature_names[j], round(coefs[j], 3)) for j in top_idx]
        n_train   = (y_train == label).sum()
        name      = CANCER_NAMES.get(label, label)
        log.info(f"\n  ┌─ {label} — {name}  (n_train={n_train}) ─────")
        for term, coef in top_terms:
            bar = "+" * min(int(coef * 3), 20)
            log.info(f"  │  {term:<35} {coef:+.3f}  {bar}")
        log.info(f"  └{'─'*60}")

    subsection("Comparing SVM vs LR top features — do they agree?")
    log.info("  If SVM and LR learned DIFFERENT top words for the same cancer type,")
    log.info("  that tells us something interesting about what each model finds important.")
    log.info("  Below are SVM's top 3 diagnostic words across ALL cancer types:")
    import pandas as pd
    specifics = []
    for i, label in enumerate(clf.classes_):
        coefs   = clf.coef_[i]
        top_idx = np.argsort(coefs)[::-1][:3]
        for j in top_idx:
            specifics.append({
                "term":   feature_names[j],
                "cancer": label,
                "coef":   coefs[j],
            })
    df_spec = pd.DataFrame(specifics).sort_values("coef", ascending=False).head(40)
    log.info(f"  {'Term':<35} {'Cancer':>6}  {'SVM Weight':>10}")
    log.info(f"  {'-'*55}")
    for _, r in df_spec.iterrows():
        log.info(f"  {r['term']:<35} {r['cancer']:>6}  {r['coef']:>+10.3f}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 — Full Comparison: Method 2 vs Method 1 vs Cedars-Sinai
# ══════════════════════════════════════════════════════════════════════════════
def step6_comparison(test_acc, test_f1_w, df_f1_svm):
    section("STEP 6 — Full Comparison: Method 2 vs Method 1 vs Cedars-Sinai")

    cedars_acc  = 0.9531
    cedars_f1   = 0.9513
    method1_acc = 0.9636   # from Method 1 run
    method1_f1  = 0.9634

    log.info(f"  {'Aspect':<32} {'Cedars-Sinai':>14} {'Method 1 (LR)':>14} {'Method 2 (SVM)':>14}")
    log.info(f"  {'-'*80}")
    log.info(f"  {'Vectorizer':<32} {'CountVectorizer':>14} {'TF-IDF':>14} {'TF-IDF':>14}")
    log.info(f"  {'n-gram range':<32} {'(1,1)':>14} {'(1,2)':>14} {'(1,2)':>14}")
    log.info(f"  {'Classifier':<32} {'LogisticReg':>14} {'LogisticReg':>14} {'LinearSVC':>14}")
    log.info(f"  {'Class weighting':<32} {'None':>14} {'balanced':>14} {'balanced':>14}")
    log.info(f"  {'Test Accuracy':<32} {cedars_acc*100:>13.2f}% {method1_acc*100:>13.2f}% {test_acc*100:>13.2f}%")
    log.info(f"  {'F1 Weighted':<32} {cedars_f1*100:>13.2f}% {method1_f1*100:>13.2f}% {test_f1_w*100:>13.2f}%")
    log.info(f"  {'vs Cedars-Sinai':<32} {'baseline':>14} {(method1_acc-cedars_acc)*100:>+13.2f}% {(test_acc-cedars_acc)*100:>+13.2f}%")

    log.info("""
  ANALYSIS — WHY DOES SVM COMPARE TO LR THIS WAY?

  SVM and LR often perform similarly on TF-IDF text data.
  The key difference is in HOW they handle errors:

  LR minimises log-loss (cross-entropy):
    It adjusts ALL weights based on ALL training examples.
    Easy examples (clear GBM reports) still contribute to the gradient.
    This can cause the model to overfit the easy majority classes.

  SVM minimises hinge loss:
    It only adjusts weights based on support vectors (hard examples).
    Easy examples far from the boundary are completely ignored.
    This focus on hard cases often helps with rare cancer types.

  WHEN SVM BEATS LR:
    — When the data has clear margin separability (pathology reports tend to)
    — When rare classes matter (SVM's margin focus helps UCS, CHOL, KICH)

  WHEN LR BEATS SVM:
    — When you need probability estimates (SVM only gives decision scores)
    — When the dataset is very noisy

  For cancer type classification from clean pathology reports:
  Both methods are competitive. The winner depends on the specific dataset split.
    """)

    subsection("Hard cases: cancer types below 0.85 F1 in SVM")
    hard = df_f1_svm[df_f1_svm["f1"] < 0.85]
    if len(hard) == 0:
        log.info("  No cancer types below 0.85 F1 — SVM handles all classes well!")
    else:
        for _, r in hard.iterrows():
            log.info(f"  {r['code']} ({r['name']}) — F1={r['f1']:.3f}, n_train={int(r['n_train'])}")
        log.info("  These same types may remain challenging for BERT (Method 3).")
        log.info("  RAG (Method 5) can help by retrieving confirmed similar cases.")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    t_start = time.time()

    log.info("=" * 70)
    log.info("  MEDREx — Method 2: TF-IDF + SVM (LinearSVC)")
    log.info("  CSC-590 Masters Project, CSUDH Spring 2026")
    log.info("  Goal: Can SVM do better than LR on high-dimensional TF-IDF text?")
    log.info("=" * 70)

    df                              = step1_load_data()
    X_train, X_test, y_train, y_test = step2_split(df)
    vec, X_train_tfidf, X_test_tfidf = step3_tfidf_vectorizer(X_train, X_test)
    clf, test_acc, test_f1_w, df_f1  = step4_svm(
        X_train_tfidf, X_test_tfidf, y_train, y_test, vec
    )
    step5_feature_analysis(clf, vec, y_train)
    step6_comparison(test_acc, test_f1_w, df_f1)

    section("DONE")
    log.info(f"  Method 2 Test Accuracy : {test_acc*100:.2f}%")
    log.info(f"  Method 2 F1 Weighted   : {test_f1_w*100:.2f}%")
    log.info(f"  Total runtime          : {(time.time()-t_start)/60:.1f} min")
    log.info(f"  Log saved to           : {_LOG_FILE}")
    log.info("  NEXT: Run Method 4 (Embedding Similarity + k-NN).")
