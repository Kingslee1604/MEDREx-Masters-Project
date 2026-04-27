"""
Cedars-Sinai Reference — Bag of Words + Random Forest
======================================================
Covers notebooks: 1-Compile_Dataset, 2-Train_Val_Test_Split,
                  3-Bag_of_Words, 4-BoW_RF, 5-BoW_ML (RF part only)

Run:
  python demo/CedarCinaiCode/bow_random_forest.py
"""

import os, sys, random, logging, warnings, time
warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger(__name__)

def section(title):
    log.info("")
    log.info("=" * 70)
    log.info(f"  {title}")
    log.info("=" * 70)

def subsection(title):
    log.info(f"\n--- {title} ---")

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR     = r"c:\Users\kings\OneDrive\Desktop\CSUDH_Kingslee\CSUDH-Spring-2026\CSC-590-MastersProject\MastersProject"
REPORTS_PATH = os.path.join(BASE_DIR, "DataSet", "TCGA_Reports.csv", "TCGA_Reports.csv")
LABELS_PATH  = os.path.join(BASE_DIR, "DataSet", "tcga_patient_to_cancer_type.csv")
OUT_DIR      = os.path.join(BASE_DIR, "demo", "CedarCinaiCode", "output")
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

    log.info(f"  Reading reports from: {REPORTS_PATH}")
    t0 = time.time()
    df = pd.read_csv(REPORTS_PATH)
    log.info(f"  Loaded {len(df):,} reports in {time.time()-t0:.1f}s")
    log.info(f"  Columns: {list(df.columns)}")

    df["patient_id"] = df["patient_filename"].apply(lambda x: x.split(".")[0])
    assert not df["patient_id"].duplicated().any(), "Duplicate patient IDs!"
    df = df[["patient_id", "text"]]
    df.index = df["patient_id"].values
    log.info(f"  Unique patients: {len(df)}")

    log.info(f"  Reading labels from: {LABELS_PATH}")
    df_labels = pd.read_csv(LABELS_PATH)
    log.info(f"  Label rows: {len(df_labels)} | Unique cancer types: {df_labels['cancer_type'].nunique()}")
    df_labels.index = df_labels["patient_id"].values

    assert df["patient_id"].isin(df_labels["patient_id"]).all(), "Missing labels!"
    df["cancer_type"]      = df_labels.loc[df.index, "cancer_type"]
    df["cancer_type_name"] = df["cancer_type"].map(CANCER_NAMES).fillna(df["cancer_type"])
    log.info(f"  Merged dataset shape: {df.shape}")

    subsection("Cancer type distribution")
    dist = df["cancer_type"].value_counts()
    for code, cnt in dist.items():
        log.info(f"    {code:<6} {CANCER_NAMES.get(code,''):<45} {cnt:4d} reports ({cnt/len(df)*100:.1f}%)")

    subsection("Text length stats")
    wc = df["text"].apply(lambda x: len(str(x).split()))
    log.info(f"  Avg words: {wc.mean():.0f}  |  Median: {wc.median():.0f}  |  Max: {wc.max()}")
    log.info(f"  Reports > 512 words: {(wc > 512).sum()} ({(wc > 512).mean()*100:.1f}%)")

    subsection("Example report")
    row = df.iloc[0]
    log.info(f"  Patient : {row['patient_id']}")
    log.info(f"  Cancer  : {row['cancer_type']} — {row['cancer_type_name']}")
    log.info(f"  Preview : {str(row['text'])[:400]}")

    return df


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Train / Val / Test Split
# ══════════════════════════════════════════════════════════════════════════════
def step2_split(df):
    section("STEP 2 — Train / Val / Test Split  (50% / 20% / 30%)")
    random.seed(0)

    ids = df["patient_id"].tolist()
    n_test = round(0.30 * len(ids))
    n_val  = round(0.20 * len(ids))

    test_ids     = random.sample(ids, n_test)
    trainval_ids = [i for i in ids if i not in test_ids]
    val_ids      = random.sample(trainval_ids, n_val)
    train_ids    = [i for i in trainval_ids if i not in val_ids]

    df_train = df.loc[train_ids].copy()
    df_val   = df.loc[val_ids].copy()
    df_test  = df.loc[test_ids].copy()

    assert len(set(df_train["patient_id"]) & set(df_val["patient_id"]))  == 0
    assert len(set(df_train["patient_id"]) & set(df_test["patient_id"])) == 0
    assert len(set(df_val["patient_id"])   & set(df_test["patient_id"])) == 0
    log.info("  No data leakage between splits — sanity checks passed")

    log.info(f"  Train : {len(df_train):,} reports ({len(df_train)/len(df)*100:.0f}%)")
    log.info(f"  Val   : {len(df_val):,}  reports ({len(df_val)/len(df)*100:.0f}%)")
    log.info(f"  Test  : {len(df_test):,} reports ({len(df_test)/len(df)*100:.0f}%)")

    subsection("Distribution per split")
    log.info(f"  {'Type':<6} {'All':>8} {'Train':>8} {'Val':>8} {'Test':>8}")
    log.info(f"  {'-'*44}")
    for code in sorted(df["cancer_type"].unique()):
        log.info(
            f"  {code:<6} {(df['cancer_type']==code).sum():>5}"
            f"  {(df_train['cancer_type']==code).sum():>6}"
            f"  {(df_val['cancer_type']==code).sum():>6}"
            f"  {(df_test['cancer_type']==code).sum():>6}"
        )
    return df_train, df_val, df_test


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Bag of Words Vectorizer
# ══════════════════════════════════════════════════════════════════════════════
def step3_build_vectorizer(df_train):
    section("STEP 3 — Bag of Words Vectorizer")
    import numpy as np
    import nltk
    from nltk.tokenize import word_tokenize
    from sklearn.feature_extraction.text import CountVectorizer

    nltk.download("punkt_tab", quiet=True)
    nltk.download("stopwords",  quiet=True)

    arr_stopwords = nltk.corpus.stopwords.words("english")
    corpus = df_train["text"].tolist()

    subsection("Example tokenization")
    sample = corpus[0]
    log.info(f"  Raw text (300 chars): {sample[:300]}")
    log.info(f"  Whitespace split (first 15): {sample.split()[:15]}")
    log.info(f"  NLTK tokenized   (first 15): {word_tokenize(sample)[:15]}")

    def tokenizer(text):
        tokens = word_tokenize(text.lower())
        tokens = [t for t in tokens if len(t) > 1]
        tokens = [t for t in tokens if t not in arr_stopwords]
        tokens = [t for t in tokens if t.isalpha()]
        return tokens

    cleaned = tokenizer(sample)
    log.info(f"\n  After cleaning — {len(cleaned)} tokens: {cleaned[:20]}")

    subsection("Fitting CountVectorizer on training corpus")
    log.info(f"  Corpus size: {len(corpus)} reports")
    t0 = time.time()
    vec = CountVectorizer(tokenizer=tokenizer, token_pattern=None, lowercase=False, stop_words=None)
    vec.fit(corpus)
    log.info(f"  Fit complete in {time.time()-t0:.1f}s")
    log.info(f"  Vocabulary size: {len(vec.vocabulary_):,} unique medical terms")

    X_train = vec.transform(corpus)
    log.info(f"  BoW matrix shape: {X_train.shape}")
    log.info(f"  Sparsity: {X_train.nnz/(X_train.shape[0]*X_train.shape[1])*100:.2f}% non-zero cells")

    subsection("Word frequency stats in BoW matrix")
    freq = np.array(X_train.sum(axis=0))[0]
    log.info(f"  Mean   word frequency across vocabulary: {freq.mean():.1f}")
    log.info(f"  Median word frequency: {np.median(freq):.1f}")
    log.info(f"  Max    word frequency: {freq.max():.0f}")
    log.info("  (Most medical words appear rarely — sparse matrix is normal)")

    return vec, tokenizer


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Random Forest
# ══════════════════════════════════════════════════════════════════════════════
def step4_random_forest(df_train, df_val, vec):
    section("STEP 4 — Random Forest  [4-BoW_RF.ipynb]")
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, f1_score, classification_report

    X_train = vec.transform(df_train["text"].tolist())
    X_val   = vec.transform(df_val["text"].tolist())
    y_train = df_train["cancer_type"].tolist()
    y_val   = df_val["cancer_type"].tolist()

    log.info(f"  Train shape: {X_train.shape}")
    log.info(f"  Val   shape: {X_val.shape}")
    log.info(f"  Classes    : {len(set(y_train))} cancer types")

    subsection("How Random Forest works")
    log.info("  A Random Forest builds many decision trees — each on a random subset of data.")
    log.info("  Each tree independently votes: 'I think this report is BRCA (breast cancer)'")
    log.info("  The final prediction = majority vote across all 100 trees.")
    log.info("  More trees = more stable predictions, but slower training.")
    log.info("  Each tree only looks at a random subset of words (features) at each split.")
    log.info("  This prevents any single tree from overfitting to one pattern.")

    subsection("Training  (n_estimators=100, random_state=0)")
    log.info("  Training 100 decision trees — this takes ~3-5 minutes...")
    t0 = time.time()
    clf = RandomForestClassifier(random_state=0, n_estimators=100)
    clf.fit(X_train, y_train)
    log.info(f"  Training complete in {time.time()-t0:.1f}s")
    log.info(f"  Number of trees built: {len(clf.estimators_)}")
    log.info(f"  Features per split  : {clf.max_features}")

    subsection("Results on TRAINING set")
    y_train_pred = clf.predict(X_train)
    train_acc    = accuracy_score(y_train, y_train_pred)
    log.info(f"  Train Accuracy : {train_acc*100:.2f}%")
    log.info("  (Trees overfit training data — near-perfect is expected here)")

    subsection("Results on VALIDATION set  ← real performance on unseen data")
    y_val_pred = clf.predict(X_val)
    val_acc    = accuracy_score(y_val, y_val_pred)
    val_f1     = f1_score(y_val, y_val_pred, average="weighted", zero_division=0)
    log.info(f"  Val Accuracy        : {val_acc*100:.2f}%")
    log.info(f"  Val F1 (weighted)   : {val_f1*100:.2f}%")
    log.info(f"  Reference benchmark : ~92.65%  (Cedars-Sinai original result)")
    log.info(f"  Our result          : {val_acc*100:.2f}%")
    log.info("\n  Full classification report:")
    log.info(classification_report(y_val, y_val_pred, zero_division=0))

    subsection("Per-class F1 sorted worst → best")
    f1_per = f1_score(y_val, y_val_pred, average=None, labels=clf.classes_, zero_division=0)
    df_f1  = pd.DataFrame({
        "code": clf.classes_,
        "name": [CANCER_NAMES.get(c, c) for c in clf.classes_],
        "f1":   f1_per,
        "n_train": [len(df_train[df_train["cancer_type"]==c]) for c in clf.classes_],
        "n_val":   [len(df_val[df_val["cancer_type"]==c])     for c in clf.classes_],
    }).sort_values("f1")

    log.info(f"  {'Code':<6} {'Cancer Name':<45} {'F1':>6} {'#Train':>7} {'#Val':>6}")
    log.info(f"  {'-'*74}")
    for _, r in df_f1.iterrows():
        log.info(f"  {r['code']:<6} {r['name']:<45} {r['f1']:>6.3f} {int(r['n_train']):>7} {int(r['n_val']):>6}")

    subsection("Top 20 most important words overall (RF feature importances)")
    log.info("  RF importance = how much each word reduces prediction error across all trees.")
    log.info("  Unlike LR, this is GLOBAL (not per cancer type — one ranking for all classes).")
    idx_word = {v: k for k, v in vec.vocabulary_.items()}
    top20 = sorted(range(len(clf.feature_importances_)),
                   key=lambda j: clf.feature_importances_[j], reverse=True)[:20]
    for rank, idx in enumerate(top20, 1):
        log.info(f"  #{rank:2d}  '{idx_word[idx]:<22}'  importance = {clf.feature_importances_[idx]:.5f}")

    return clf, val_acc


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — Pipeline + RandomizedSearch + Final Test Evaluation
# ══════════════════════════════════════════════════════════════════════════════
def step5_tune_and_test(df_train, df_val, df_test, tokenizer):
    section("STEP 5 — Pipeline + RandomizedSearch + Final Test Evaluation  [5-BoW_ML.ipynb — RF]")
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import RandomizedSearchCV
    from sklearn.metrics import accuracy_score, classification_report
    import pandas as pd

    X_all  = df_train["text"].tolist() + df_val["text"].tolist()
    y_all  = df_train["cancer_type"].tolist() + df_val["cancer_type"].tolist()
    X_test = df_test["text"].tolist()
    y_test = df_test["cancer_type"].tolist()

    train_idx = list(range(len(df_train)))
    val_idx   = list(range(len(df_train), len(X_all)))
    custom_cv = [(train_idx, val_idx)]

    subsection("What is RandomizedSearchCV?")
    log.info("  RF has too many hyperparameters to try every combination (GridSearch would take hours).")
    log.info("  RandomizedSearch randomly samples n_iter combinations from the search space.")
    log.info("  Search space:")
    log.info("    n_estimators      : [10, 100, 500]   (number of trees)")
    log.info("    max_features      : ['sqrt', 'log2'] (words considered per split)")
    log.info("    min_samples_split : [2, 5, 10]       (min samples to split a node)")
    log.info("  We try 2 random combos (n_iter=2) — same as Cedars-Sinai reference.")

    pipe = Pipeline([
        ("bow", CountVectorizer(tokenizer=tokenizer, token_pattern=None,
                                lowercase=False, stop_words=None)),
        ("clf", RandomForestClassifier(random_state=0))
    ])
    param_grid = {
        "clf__n_estimators":      [10, 100, 500],
        "clf__max_features":      ["sqrt", "log2"],
        "clf__min_samples_split": [2, 5, 10],
    }
    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_grid,
        scoring="accuracy",
        cv=custom_cv,
        verbose=2,
        refit=True,
        n_iter=10,
        random_state=0
    )
    log.info("  Running RandomizedSearch (2 RF models, ~5-8 min)...")
    t0 = time.time()
    search.fit(X_all, y_all)
    log.info(f"  Done in {time.time()-t0:.1f}s")
    log.info(f"  Best parameters  : {search.best_params_}")
    log.info(f"  Best val accuracy: {search.best_score_*100:.2f}%")

    df_rs = pd.DataFrame(search.cv_results_)[[
        "param_clf__n_estimators", "param_clf__max_features",
        "param_clf__min_samples_split", "mean_test_score", "mean_fit_time"
    ]]
    log.info("\n  RandomizedSearch results table:")
    for _, row in df_rs.iterrows():
        log.info(
            f"    n_estimators={row['param_clf__n_estimators']}"
            f"  max_features={row['param_clf__max_features']}"
            f"  min_samples_split={row['param_clf__min_samples_split']}"
            f"  →  val_acc={row['mean_test_score']*100:.2f}%"
            f"  (fit={row['mean_fit_time']:.1f}s)"
        )

    subsection("Final Test Set Evaluation  ← ONLY DONE ONCE")
    log.info("  The test set was never used during training or tuning.")
    log.info("  This is the unbiased real-world accuracy estimate.")
    best_model  = search.best_estimator_
    y_test_pred = best_model.predict(X_test)
    test_acc    = accuracy_score(y_test, y_test_pred)
    log.info(f"\n  ╔══════════════════════════════════════════════╗")
    log.info(f"  ║  RANDOM FOREST FINAL TEST ACCURACY          ║")
    log.info(f"  ║                                              ║")
    log.info(f"  ║  Our result   : {test_acc*100:>6.2f}%                    ║")
    log.info(f"  ║  Reference    : ~92.65%  (Cedars-Sinai)     ║")
    log.info(f"  ╚══════════════════════════════════════════════╝")
    log.info("\n  Full classification report on test set:")
    log.info(classification_report(y_test, y_test_pred, zero_division=0))

    return test_acc


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    t_start = time.time()
    log.info("=" * 70)
    log.info("  Bag of Words + RANDOM FOREST  — Cedars-Sinai Reference")
    log.info("  CSC-590 Masters Project, CSUDH Spring 2026")
    log.info("=" * 70)

    df                        = step1_load_data()
    df_train, df_val, df_test = step2_split(df)
    vec, tokenizer_fn         = step3_build_vectorizer(df_train)
    clf_rf, val_acc           = step4_random_forest(df_train, df_val, vec)
    test_acc                  = step5_tune_and_test(df_train, df_val, df_test, tokenizer_fn)

    section("SUMMARY — Random Forest")
    log.info(f"  Validation Accuracy : {val_acc*100:.2f}%")
    log.info(f"  Test Accuracy       : {test_acc*100:.2f}%")
    log.info(f"  Cedars-Sinai ref    : ~92.65%")
    log.info(f"  Total runtime       : {(time.time()-t_start)/60:.1f} min")
    log.info("")
