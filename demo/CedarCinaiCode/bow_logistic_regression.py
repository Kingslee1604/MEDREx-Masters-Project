"""
Cedars-Sinai Reference — Bag of Words + Logistic Regression
============================================================
Covers notebooks: 1-Compile_Dataset, 2-Train_Val_Test_Split,
                  3-Bag_of_Words, 4-BoW_LR, 5-BoW_ML (LR part only)

Run:
  python demo/CedarCinaiCode/bow_logistic_regression.py
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

    # Extract patient_id  (e.g. "TCGA-BP-5195.uuid..." → "TCGA-BP-5195")
    df["patient_id"] = df["patient_filename"].apply(lambda x: x.split(".")[0])
    assert not df["patient_id"].duplicated().any(), "Duplicate patient IDs!"
    df = df[["patient_id", "text"]]
    df.index = df["patient_id"].values
    log.info(f"  Unique patients: {len(df)}")

    # Load labels
    log.info(f"  Reading labels from: {LABELS_PATH}")
    df_labels = pd.read_csv(LABELS_PATH)
    log.info(f"  Label rows: {len(df_labels)} | Unique cancer types: {df_labels['cancer_type'].nunique()}")
    df_labels.index = df_labels["patient_id"].values

    # Merge
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

    # Sanity checks
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
    import nltk
    from nltk.tokenize import word_tokenize
    from sklearn.feature_extraction.text import CountVectorizer

    nltk.download("punkt_tab", quiet=True) # for tokenization 
    #Kidney tumor is detected --> ["Kidney", "tumor", "is", "detected"] 
    nltk.download("stopwords",  quiet=True) # for removing common words
    # "the tumor is in the kidney" --> ["tumor", "kidney"]

    arr_stopwords = nltk.corpus.stopwords.words("english")
    corpus = df_train["text"].tolist()

    subsection("Example tokenization")
    sample = corpus[0]
    log.info(f"  Raw text (300 chars): {sample[:300]}")
    #Raw text (300 chars): Date of Recelpt: Clinical Diagnosis & History: Incidental 3 cm left upper
    simple = sample.split()[:15]
    log.info(f"  Whitespace split (first 15): {simple}")
    #['Date', 'of', 'Recelpt:', 'Clinical', 'Diagnosis', '&', 'History:', 'Incidental', '3', 'cm']
    nltk_tok = word_tokenize(sample)[:15]
    log.info(f"  NLTK tokenized   (first 15): {nltk_tok}")
    #'Date', 'of', 'Recelpt', ':', 'Clinical', 'Diagnosis', '&']

    def tokenizer(text):
        tokens = word_tokenize(text.lower())
        tokens = [t for t in tokens if len(t) > 1] # this removes one letter words like "a"
        tokens = [t for t in tokens if t not in arr_stopwords] # removes common wods like "this", "is"
        tokens = [t for t in tokens if t.isalpha()] #removes numbers #123 and keep only letters
        return tokens

    cleaned = tokenizer(sample)
    log.info(f"\n  After cleaning — {len(cleaned)} tokens: {cleaned[:20]}")
    log.info(f"  Stopwords removed example: {arr_stopwords[:8]}")
    #Stopwords: ['a', 'about', 'above', 'after', 'again', 'against', 'ain', 'all']

    subsection("Fitting CountVectorizer on training corpus")
    log.info(f"  Corpus size: {len(corpus)} reports")
    log.info("  Fitting... (builds vocabulary from training data only)")
    t0 = time.time()
    vec = CountVectorizer(tokenizer=tokenizer, token_pattern=None, lowercase=False, stop_words=None)
    vec.fit(corpus) 
    # place where model takes unique words. Here from 4761 reports we got 23818 unique medical terms
    log.info(f"  Fit complete in {time.time()-t0:.1f}s")
    log.info(f"  Vocabulary size: {len(vec.vocabulary_):,} unique medical terms")

    subsection("BoW matrix")
    t0 = time.time()
    X_train = vec.transform(corpus) 
    # convert each report into numbers, 4761 rows and 23,818 columns: Sparse matrix
    log.info(f"  Transform complete in {time.time()-t0:.1f}s")
    log.info(f"  Matrix shape: {X_train.shape}  ({X_train.shape[0]} reports × {X_train.shape[1]} words)")
    log.info(f"  Sparsity: {X_train.nnz/(X_train.shape[0]*X_train.shape[1])*100:.2f}% non-zero cells")

    return vec, tokenizer


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Logistic Regression
# ══════════════════════════════════════════════════════════════════════════════
def step4_logistic_regression(df_train, df_val, vec):
    section("STEP 4 — Logistic Regression  [4-BoW_LR.ipynb]")
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score, classification_report

    X_train = vec.transform(df_train["text"].tolist())#Same number conversion again
    X_val   = vec.transform(df_val["text"].tolist())
    y_train = df_train["cancer_type"].tolist()
    y_val   = df_val["cancer_type"].tolist()

    log.info(f"  Train shape: {X_train.shape}")
    log.info(f"  Val   shape: {X_val.shape}")
    log.info(f"  Classes    : {len(set(y_train))} cancer types")

    subsection("How Logistic Regression works")
    log.info("  LR learns one weight per word per cancer type.")
    log.info("  Example: word 'glioblastoma' gets high weight for GBM (brain cancer)")
    log.info("           word 'ductal' gets high weight for BRCA (breast cancer)")
    log.info("  At prediction time: score = sum(word_count × word_weight) for each cancer type")
    log.info("  The cancer type with the highest score wins.")

    subsection("Training  (random_state=0, max_iter=200)")
    log.info("  Training LR — this takes ~30-60 seconds...")
    t0 = time.time()
    clf = LogisticRegression(random_state=0, max_iter=200)
    # Model will try up to 200 times (iterations) to learn best weights
    # here "clf" is called Model
    clf.fit(X_train, y_train) #learns weights: 32 cancer types × 23,818 word features
    #                word1   word2   word3   ...   word23818
    #-------------------------------------------------------
    #BRCA            1.25    0.02    0.00          0.45
    #LUAD            0.10    0.95    0.03          0.12
    #GBM             0.05    0.01    1.10          0.08
    #...
    #(32 cancers)    
    log.info(f"  Training complete in {time.time()-t0:.1f}s")

    subsection("Results on TRAINING set")
    y_train_pred = clf.predict(X_train)
    train_acc    = accuracy_score(y_train, y_train_pred)
    log.info(f"  Train Accuracy : {train_acc*100:.2f}%")
    log.info("  (Near 100% expected — model memorises what it was trained on)")

    subsection("Results on VALIDATION set  ← real performance on unseen data")
    y_val_pred = clf.predict(X_val)
    val_acc    = accuracy_score(y_val, y_val_pred)#Accuracy = correct predictions / total predictions
    val_f1     = f1_score(y_val, y_val_pred, average="weighted", zero_division=0)
    #f1 F1 = 2 × (Precision × Recall) / (Precision + Recall)
    # Precision = True Positives / (True Positives + False Positives)
    # 15 / (15 + 10) = 15 / 25 = 0.60
    # Recall = True Positives / (True Positives + False Negatives)
    # 15 / (15 + 5) = 15 / 20 = 0.75
    # F1 = 2 × (0.60 × 0.75) / (0.60 + 0.75) = 0.67
    log.info(f"  Val Accuracy        : {val_acc*100:.2f}%")
    log.info(f"  Val F1 (weighted)   : {val_f1*100:.2f}%")
    log.info(f"  Reference benchmark : ~95.31%  (Cedars-Sinai original result)")
    log.info(f"  Our result          : {val_acc*100:.2f}%")
    log.info("\n  Full classification report (precision / recall / F1 per cancer type):")
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
    log.info("\n  KEY INSIGHT: Cancer types with few training samples tend to have lower F1.")

    subsection("Top 10 most predictive words per cancer type (LR coefficients)")
    log.info("  A high coefficient means that word strongly suggests this cancer type.")
    idx_word = {v: k for k, v in vec.vocabulary_.items()}
    selected = ["GBM", "BRCA", "LUAD"]  # choose what you want
    for label in selected:
        i = list(clf.classes_).index(label)
        log.info(f"\n  {label} — {CANCER_NAMES.get(label, label)}")
        top_idx   = sorted(range(len(clf.coef_[i])), key=lambda j: clf.coef_[i][j], reverse=True)[:10]
        top_words = [(idx_word[j], round(clf.coef_[i][j], 3)) for j in top_idx]
        for word, coef in top_words:
            log.info(f"    {word:<25} coeff = {coef:+.3f}")
    return clf, val_acc

# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — Pipeline + GridSearch + Final Test Evaluation
# ══════════════════════════════════════════════════════════════════════════════
def step5_tune_and_test(df_train, df_val, df_test, tokenizer):
    section("STEP 5 — Pipeline + GridSearch + Final Test Evaluation  [5-BoW_ML.ipynb — LR]")
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import accuracy_score, classification_report
    import pandas as pd

    X_all   = df_train["text"].tolist() + df_val["text"].tolist()
    y_all   = df_train["cancer_type"].tolist() + df_val["cancer_type"].tolist()
    X_test  = df_test["text"].tolist()
    y_test  = df_test["cancer_type"].tolist()

    train_idx = list(range(len(df_train)))
    val_idx   = list(range(len(df_train), len(X_all)))
    custom_cv = [(train_idx, val_idx)]

    subsection("What is a Pipeline?")
    log.info("  A Pipeline chains steps so they run together without data leakage.")
    log.info("  Step 1: CountVectorizer  — converts raw text to BoW matrix")
    log.info("  Step 2: LogisticRegression — classifies the BoW matrix")
    log.info("  The vectorizer is ONLY fit on training data, even inside GridSearch.")

    subsection("GridSearchCV — trying max_iter values: [200, 500]")
    log.info("  GridSearch trains one model per setting, picks the best on val set.")
    pipe = Pipeline([
        ("bow", CountVectorizer(tokenizer=tokenizer, token_pattern=None,
                                lowercase=False, stop_words=None)),
        ("clf", LogisticRegression(random_state=0))
    ])
    grid = GridSearchCV(
        estimator=pipe,
        param_grid={"clf__max_iter": [200, 500]},
        scoring="accuracy",
        cv=custom_cv,
        verbose=2,
        refit=True
    )
    log.info("  Running GridSearch (2 models × 1 fold = 2 fits, ~2 min)...")
    t0 = time.time()
    grid.fit(X_all, y_all)
    log.info(f"  Done in {time.time()-t0:.1f}s")
    log.info(f"  Best parameter : {grid.best_params_}")
    log.info(f"  Best val accuracy: {grid.best_score_*100:.2f}%")

    df_gs = pd.DataFrame(grid.cv_results_)[["param_clf__max_iter", "mean_test_score", "mean_fit_time"]]
    log.info("\n  GridSearch results table:")
    for _, row in df_gs.iterrows():
        log.info(f"    max_iter={int(row['param_clf__max_iter'])}  →  "
                 f"val_acc={row['mean_test_score']*100:.2f}%  "
                 f"(fit time={row['mean_fit_time']:.1f}s)")

    subsection("Final Test Set Evaluation  ← ONLY DONE ONCE")
    log.info("  The test set was never used during training or tuning.")
    log.info("  This is the unbiased real-world accuracy estimate.")
    best_model     = grid.best_estimator_
    y_test_pred    = best_model.predict(X_test)
    test_acc       = accuracy_score(y_test, y_test_pred)
    log.info(f"\n  ╔══════════════════════════════════════════════╗")
    log.info(f"  ║  LOGISTIC REGRESSION FINAL TEST ACCURACY    ║")
    log.info(f"  ║                                              ║")
    log.info(f"  ║  Our result   : {test_acc*100:>6.2f}%                    ║")
    log.info(f"  ║  Reference    : ~95.31%  (Cedars-Sinai)     ║")
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
    log.info("  Bag of Words + LOGISTIC REGRESSION  — Cedars-Sinai Reference")
    log.info("  CSC-590 Masters Project, CSUDH Spring 2026")
    log.info("=" * 70)

    df                       = step1_load_data()
    df_train, df_val, df_test = step2_split(df)
    vec, tokenizer_fn        = step3_build_vectorizer(df_train)
    clf_lr, val_acc          = step4_logistic_regression(df_train, df_val, vec)
    test_acc                 = step5_tune_and_test(df_train, df_val, df_test, tokenizer_fn)

    section("SUMMARY — Logistic Regression")
    log.info(f"  Validation Accuracy : {val_acc*100:.2f}%")
    log.info(f"  Test Accuracy       : {test_acc*100:.2f}%")
    log.info(f"  Cedars-Sinai ref    : ~95.31%")
    log.info(f"  Total runtime       : {(time.time()-t_start)/60:.1f} min")
    log.info("")
