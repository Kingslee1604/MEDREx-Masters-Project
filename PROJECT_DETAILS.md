# MEDREx — Complete Project Reference

**Medical Report Extraction and Classification**
CSC-590 Masters Project · California State University, Dominguez Hills · Spring 2026
Student: Kingslee Dominic Savio Velu · Sponsor: Cedars-Sinai AI Campus, Project #32 — *"Natural Language Processing in Cancer: Extracting Diagnostic Insights from Pathology Reports"*
Faculty mentor: Dr. Bin Tang · Cedars-Sinai project leads: Guillermo Lopez Garcia, Takeshi Onishi
Repository: `https://github.com/Kingslee1604/MEDREx-Masters-Project` (submission branch: `submission`)

This document describes everything in the repository: the problem, the data, all six methods and the Cedars-Sinai baseline, how each script works, the results, the demo app, how to reproduce it, and the known issues. Read it without opening the code and you should still understand what the project does.

---

## Table of contents

1. [The problem](#1-the-problem)
2. [Results at a glance](#2-results-at-a-glance)
3. [Repository layout](#3-repository-layout)
4. [Dataset](#4-dataset)
5. [Shared experimental setup](#5-shared-experimental-setup)
6. [Cedars-Sinai baseline (BoW + LR / RF)](#6-cedars-sinai-baseline)
7. [Method 1 — TF-IDF + Logistic Regression](#7-method-1--tf-idf--logistic-regression)
8. [Method 2 — TF-IDF + Linear SVM](#8-method-2--tf-idf--linear-svm)
9. [Method 3 — Fine-tuned Bio_ClinicalBERT](#9-method-3--fine-tuned-bio_clinicalbert)
10. [Method 4 — Sentence embeddings + cosine k-NN](#10-method-4--sentence-embeddings--cosine-k-nn)
11. [Method 5 — RAG + local LLM (Ollama)](#11-method-5--rag--local-llm-ollama)
12. [Method 6 — RAG + hosted LLM (OpenAI / Anthropic / Ollama)](#12-method-6--rag--hosted-llm)
13. [Per-class analysis: the hard cancer types](#13-per-class-analysis-the-hard-cancer-types)
14. [Demo application (Streamlit explainer)](#14-demo-application-streamlit-explainer)
15. [Notebooks](#15-notebooks)
16. [Reference code from Cedars-Sinai and Tatonetti Lab](#16-reference-code)
17. [How to reproduce](#17-how-to-reproduce)
18. [Known issues and inconsistencies](#18-known-issues-and-inconsistencies)
19. [Conclusions](#19-conclusions)
20. [Not yet done / future work](#20-not-yet-done--future-work)

---

## 1. The problem

**Input:** the free text of one TCGA (The Cancer Genome Atlas) pathology report, as produced by OCR of a scanned clinical document. The text is noisy: OCR errors ("Date of Recelpt"), page headers, and mixed formats from many hospitals.

**Output:** the cancer type, one of **32 TCGA project codes** (BRCA, LUAD, GBM, ...).

**Goal of the project:** not just to get a high score, but to **compare families of NLP methods on the same task under the same conditions**: classical sparse models, a fine-tuned transformer, embedding similarity, and retrieval-augmented LLMs. The project asks whether the added complexity of newer methods pays for itself.

Why not just send every report to a commercial LLM?
- **Privacy.** In a real hospital, patient text generally cannot leave the institution (HIPAA). Methods that run locally (1–5) matter. TCGA data itself is de-identified and public, which is why Method 6 could be tested at all.
- **Auditability.** Clinical tools need explanations. Linear models expose per-word weights; LLMs do not.
- **Cost and speed.** Methods 1–2 train in under a minute on a laptop CPU.

A second task named in the sponsor's brief, **TNM staging** (predicting T, N and M stage from the report), has its data and reference code in the repo but **was not implemented** by this project (see §20).

---

## 2. Results at a glance

All MEDREx methods use the same stratified 70/30 split (`random_state=42`): 6,666 training reports and 2,857 test reports.

| # | Method | Test set | Accuracy | F1 weighted | F1 macro | Runtime | Hardware |
|---|---|---|---|---|---|---|---|
| Baseline | Cedars-Sinai BoW + Logistic Regression | 2,857 (their own 50/20/30 split) | 95.31% published · 95.48% our re-run | 95.13% | — | ~9 min | CPU |
| Baseline | Cedars-Sinai BoW + Random Forest | 2,857 (their split) | 92.65% published · 92.16% our re-run | — | — | ~25 min | CPU |
| 1 | TF-IDF + Logistic Regression | 2,857 | 96.36% | 96.34–96.36% | 95.73% | 0.8 min | CPU |
| **2** | **TF-IDF + LinearSVC** | 2,857 | **97.37%** | **97.35%** | **96.48%** | 0.2 min | CPU |
| 3 | Fine-tuned Bio_ClinicalBERT | 2,857 | 94.57% | 94.27% | — | ~40–60 min training | Colab T4 GPU |
| 4 | MiniLM embeddings + cosine k-NN (K=5) | 2,857 | 90.13% | 89.93% | 87.78% | 6.4 min (first run) | CPU |
| 5 | RAG + llama3.2 (3B, Ollama, local) | **50-report sample** | 64.00% | 67.07% | — | ~30 min | CPU |
| 6 | RAG + GPT-4o-mini (OpenAI API) | **300-report sample** | 89.00% | — | — | minutes | API |

**Headline:** the simplest strong model, TF-IDF + linear SVM, is the most accurate, and it beats the published Cedars-Sinai baseline by about 2 points. Fine-tuned BERT does not beat it. RAG accuracy depends almost entirely on which LLM is used (64% vs 89% with an identical retrieval pipeline).

> Caveats: Methods 5 and 6 were scored on random samples, not the full test set. The Cedars baseline uses a different split from Methods 1–6, so a ~1-point difference with it is suggestive, not exact. See §18.

---

## 3. Repository layout

```
MastersProject/
├── README.md                       Quick-start setup instructions
├── Readme.docx                     Word copy of the README
├── PROJECT_DETAILS.md              This file
├── .gitignore                      Excludes dataset, model weights, .env, logs, output/
│
├── DataSet/                        NOT in git (size + data-use terms) — see §4
│   ├── TCGA_Reports.csv/TCGA_Reports.csv
│   ├── tcga_patient_to_cancer_type.csv
│   ├── TNM_Stage/                  T / N / M stage label files (unused so far)
│   └── Clinical_Endpoints/         Survival table (unused so far)
│
├── notebooks/
│   ├── 01_EDA.ipynb                Phase 1 — exploratory data analysis
│   └── 02_TF_IDF_Baseline.ipynb    Phase 2 — first TF-IDF LR/SVM experiments
│
├── demo/
│   ├── explainer.py                Streamlit app — the main demo UI (≈2,150 lines)
│   ├── save_models.py              Trains + saves BoW, TF-IDF, LR, SVM for the UI
│   ├── requirements.txt            pip dependencies for the demo
│   ├── pipeline_diagram.html       Diagram: overall MEDREx pipeline
│   ├── demo_flow_diagram.html      Diagram: demo flow (untracked at time of writing)
│   ├── Pipeline_process.jpg        Pipeline image
│   │
│   ├── saved_models/               Committed small model artifacts (joblib/json)
│   │   ├── cedars_vocabulary.json  BoW vocabulary (word → index)
│   │   ├── cedars_clf.joblib       Cedars BoW Logistic Regression
│   │   ├── tfidf_vectorizer.joblib TF-IDF vectorizer shared by Methods 1 & 2
│   │   ├── m1_lr_clf.joblib        Method 1 classifier
│   │   └── m2_svm_clf.joblib       Method 2 classifier
│   │
│   ├── CedarCinaiCode/             Baseline re-implemented as runnable scripts
│   │   ├── cedars_sinai_pipeline.py   All 6 reference notebooks in one script
│   │   ├── bow_logistic_regression.py BoW + LR only
│   │   ├── bow_random_forest.py       BoW + RF only
│   │   ├── LoggisticRegressionResults.sql  Console log of an LR run (plain text)
│   │   └── RandomForestResults.sql         Console log of an RF run (plain text)
│   │
│   ├── ProjectCode/                MEDREx methods — one standalone script each
│   │   ├── method1_tfidf_logistic_regression.py
│   │   ├── method2_tfidf_svm.py
│   │   ├── method3_evaluate_local.py      Evaluates the downloaded BERT model
│   │   ├── method4_embedding_knn.py
│   │   ├── method5_rag_llm.py
│   │   ├── method6_rag_openai.py
│   │   ├── Method1-TFIDF_LogisticRegResults.sql   Run logs (plain text, .sql extension)
│   │   ├── Method2-TFIDFSVM_LogisticRegResults.sql
│   │   ├── Method4-EmbeddingSimilarity.sql
│   │   ├── method3_best_model/     NOT in git — fine-tuned BERT weights (~414 MB)
│   │   └── output/                 NOT in git — logs, embedding caches (.npz/.csv)
│   │
│   └── BertModel_Code/
│       ├── method3_bert_finetune_colab.ipynb   BERT training notebook (Google Colab)
│       └── method3_bert_finetune_colab.py      Same, as a script with cell markers
│
└── reference_code/                 Original sponsor / lab code, kept for comparison
    ├── utils.py, llm_utils.py
    ├── cancer_type/                Cedars-Sinai BoW notebooks 1–5 (cancer type)
    ├── tnm_stage/                  Cedars-Sinai BoW → BERT → LLM notebooks (TNM) + prompts/
    ├── tatonetti_cancer_type/      Tatonetti Lab Bio_ClinicalBERT scripts
    └── tatonetti_tnm/              Tatonetti Lab TNM dataset split
```

> The `.sql` files are **not SQL**. They are plain-text console logs saved with a `.sql` extension. They contain the full run output, including classification reports and confusion details, and are the evidence for the numbers in §2.

---

## 4. Dataset

### Files

| File | Rows | Columns | Role |
|---|---|---|---|
| `TCGA_Reports.csv/TCGA_Reports.csv` (~34 MB) | 9,523 | `patient_filename`, `text` | Input text, one report per patient |
| `tcga_patient_to_cancer_type.csv` | 11,160 | `patient_id`, `cancer_type` | Labels (33 distinct codes) |
| `TNM_Stage/TCGA_T14_patients.csv` | 6,966 | `ajcc_pathologic_t`, `project_id`, `case_id`, `case_submitter_id` | T-stage labels (not used yet) |
| `TNM_Stage/TCGA_N03_patients.csv` | 5,678 | `ajcc_pathologic_n`, ... | N-stage labels (not used yet) |
| `TNM_Stage/TCGA_M01_patients.csv` | 4,608 | `ajcc_pathologic_m`, ... | M-stage labels (not used yet) |
| `TNM_Stage/TCGA_cancer_types_binary.p` | — | pickle | From the Tatonetti repo (not used) |
| `Clinical_Endpoints/Survival_SupplementalTable_S1_20171025_xena_sp.tsv` | 12,591 | OS, DSS, DFI, PFI + demographics | Survival data (not used yet) |

Sources: the reports and labels come from the Tatonetti Lab `tcga-path-reports` release; the TNM and survival tables come from the Tatonetti TNM classifier repo and UCSC Xena (TCGA Pan-Cancer). None of the data is committed to git.

### Joining reports to labels

```python
df_reports["patient_id"] = df_reports["patient_filename"].str.split(".").str[0]
# 'TCGA-BP-5195.25c0b433-5557-...'  →  'TCGA-BP-5195'
df = df_reports.merge(df_labels, on="patient_id")
```

All 9,523 reports find a label. The label file has 33 codes, but **LAML (Acute Myeloid Leukemia) has no reports**, so the working dataset has **32 classes**.

### Class distribution (9,523 reports, 32 classes)

| Code | Cancer | Reports | | Code | Cancer | Reports |
|---|---|---|---|---|---|---|
| BRCA | Breast invasive carcinoma | 1,034 | | PAAD | Pancreatic adenocarcinoma | 176 |
| UCEC | Uterine corpus endometrial carcinoma | 546 | | PCPG | Pheochromocytoma & paraganglioma | 174 |
| KIRC | Kidney renal clear cell carcinoma | 525 | | READ | Rectum adenocarcinoma | 162 |
| HNSC | Head & neck squamous cell carcinoma | 520 | | ESCA | Esophageal carcinoma | 146 |
| LUAD | Lung adenocarcinoma | 488 | | THYM | Thymoma | 114 |
| THCA | Thyroid carcinoma | 487 | | KICH | Kidney chromophobe | 112 |
| LGG | Brain lower grade glioma | 469 | | SKCM | Skin cutaneous melanoma | 102 |
| LUSC | Lung squamous cell carcinoma | 468 | | ACC | Adrenocortical carcinoma | 90 |
| PRAD | Prostate adenocarcinoma | 446 | | TGCT | Testicular germ cell tumors | 87 |
| COAD | Colon adenocarcinoma | 418 | | MESO | Mesothelioma | 79 |
| GBM | Glioblastoma multiforme | 399 | | UVM | Uveal melanoma | 65 |
| BLCA | Bladder urothelial carcinoma | 379 | | UCS | Uterine carcinosarcoma | 56 |
| OV | Ovarian serous cystadenocarcinoma | 371 | | DLBC | Diffuse large B-cell lymphoma | 47 |
| STAD | Stomach adenocarcinoma | 361 | | CHOL | Cholangiocarcinoma | 43 |
| LIHC | Liver hepatocellular carcinoma | 341 | | | | |
| CESC | Cervical squamous cell carcinoma | 289 | | | | |
| KIRP | Kidney renal papillary cell carcinoma | 280 | | | | |
| SARC | Sarcoma | 249 | | | | |

The classes are **strongly imbalanced**: BRCA has 24× as many reports as CHOL. Methods 1 and 2 counter this with `class_weight="balanced"`.

### Report length (words, whitespace-split)

| Mean | Median | 25th pct | 75th pct | Min | Max |
|---|---|---|---|---|---|
| 560 | 432 | 180 | 808 | 1 | 4,046 |

This matters for models with input limits: BERT sees at most 512 *tokens* (fewer words than that), and MiniLM truncates at 256 tokens by default. So a large share of each long report is never seen by Methods 3–6.

---

## 5. Shared experimental setup

All six MEDREx methods use the same setup, so differences come from the method, not the data split:

- **Split:** `train_test_split(test_size=0.3, random_state=42, stratify=cancer_type)` gives 6,666 train and 2,857 test reports. No separate validation set; hyperparameters are mostly defaults.
- **Metrics:** accuracy, weighted F1, macro F1, and per-class precision/recall/F1 (`sklearn.metrics.classification_report`).
- **Logging:** every script writes timestamped, step-by-step logs to the terminal and to `demo/ProjectCode/output/<method>_run.log`. The logs explain each step in plain language (they double as teaching material for the presentation).
- **Script structure:** each method is a single file with `step1_load_data()`, `step2_split()`, `step3_...`, and so on, run in order from `if __name__ == "__main__":`.
- **Paths:** each script has an absolute `BASE_DIR` hard-coded to the author's Windows machine (see §18).

---

## 6. Cedars-Sinai baseline

**What it is:** the sponsor's reference pipeline from `reference_code/cancer_type/` (notebooks 1–5), re-implemented as runnable scripts in `demo/CedarCinaiCode/`.

**Pipeline:**
1. Compile dataset (join reports and labels as in §4).
2. Stratified **50 / 20 / 30** split: train 4,761, validation 1,905, test 2,857.
3. **Bag of words:** `CountVectorizer` with an NLTK tokenizer that lowercases, runs `word_tokenize`, drops English stopwords, punctuation, numbers and 1-character tokens. Vocabulary: **23,818** terms. Values are raw counts.
4. **Logistic Regression** `(random_state=0, max_iter=200)`, tuned with `GridSearchCV` over `max_iter`.
5. **Random Forest** tuned with randomized search over `n_estimators`, `max_features ∈ {sqrt, log2}`, `min_samples_split`. Best: 500 trees, `sqrt`, split=2.

**Results:**

| Model | Published (Cedars) | Our re-run (val / test) |
|---|---|---|
| BoW + LR | 95.31% test | 95.38% / 95.48% |
| BoW + RF | 92.65% test | 91.39% / 92.16% |

Our re-runs land within about 0.5 points of the published figures, which confirms that the reproduction is faithful. In the published results the weakest LR classes were READ (F1 0.76), KICH (0.85) and COAD (0.88). RF failed badly on KICH (F1 0.19).

---

## 7. Method 1 — TF-IDF + Logistic Regression

**File:** `demo/ProjectCode/method1_tfidf_logistic_regression.py`

**Idea:** keep the linear classifier but replace raw word counts with TF-IDF weights and add two-word phrases.

```python
TfidfVectorizer(max_features=15000, ngram_range=(1, 2), sublinear_tf=True, min_df=2)
LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced", solver="lbfgs", random_state=42)
```

**What each change does:**
- **TF-IDF** down-weights words that appear in almost every report ("specimen", "patient") and up-weights rare, specific words ("glioblastoma" points to GBM).
- **Bigrams** (`ngram_range=(1,2)`) capture phrases such as "clear cell" and "renal cell", which separate kidney subtypes.
- **`sublinear_tf`** uses `1 + log(tf)`, so a word repeated 50 times does not dominate.
- **`class_weight="balanced"`** raises the loss weight of small classes (UCS, CHOL, KICH).
- The script also prints the top-weighted words per cancer type from the LR coefficients, which makes the model explainable.

**Results:** train 98.57% · **test 96.36%** · F1 weighted 96.36% · F1 macro 95.73% · runtime 0.8 min. That is +1.05 points accuracy over the published Cedars LR.

---

## 8. Method 2 — TF-IDF + Linear SVM

**File:** `demo/ProjectCode/method2_tfidf_svm.py`

**Idea:** the same features as Method 1 (an identical vectorizer), with only the classifier changed. Any difference in score is therefore due to the classifier alone.

```python
LinearSVC(C=1.0, max_iter=2000, class_weight="balanced", random_state=42)
```

**Why a linear SVM:** it maximizes the margin between classes, which tends to generalize well on high-dimensional sparse text (6,666 × 15,000). A kernel SVM (RBF) would be far slower on this size of sparse matrix and is rarely better for text.

**Results:** train 99.74% · **test 97.37%** · F1 weighted 97.35% · F1 macro 96.48% · runtime 0.2 min. **This is the best method in the project.** The script compares the SVM's top features against those of the LR to check whether both models rely on the same words.

---

## 9. Method 3 — Fine-tuned Bio_ClinicalBERT

**Files:**
- Training: `demo/BertModel_Code/method3_bert_finetune_colab.ipynb` (and the `.py` mirror). Runs on Google Colab.
- Evaluation: `demo/ProjectCode/method3_evaluate_local.py`. Runs locally on the downloaded model.

**Model:** `emilyalsentzer/Bio_ClinicalBERT` (BERT-base pretrained on PubMed and MIMIC-III clinical notes) with a 32-way classification head (`AutoModelForSequenceClassification`).

**Training configuration:**

| Setting | Value |
|---|---|
| Max sequence length | 512 tokens (truncation: longer text is cut) |
| Batch size | 16 (fits a 16 GB T4) |
| Epochs | 4 |
| Optimizer | `torch.optim.AdamW`, lr 2e-5, weight decay 0.01 |
| Scheduler | Linear warmup (100 steps) then linear decay |
| Labels | `LabelEncoder`; mapping saved as `label_map.json` |
| Checkpointing | Best epoch saved to Google Drive with `save_pretrained` |
| Hardware / time | Colab T4 GPU, ~40–60 min |

**Workflow:**
1. Upload `TCGA_Reports.csv` and `tcga_patient_to_cancer_type.csv` to Google Drive.
2. Run the notebook on a GPU runtime. It mounts Drive, reproduces the same 70/30 split, tokenizes, trains, and saves the best model.
3. Download the model folder to `demo/ProjectCode/method3_best_model/` (`config.json`, `model.safetensors`, `tokenizer.json`, `tokenizer_config.json`, `label_map.json`).
4. Run `method3_evaluate_local.py` to score the 2,857 test reports on CPU.

**Results:** **test 94.57%** · F1 weighted 94.27%. Epoch 1 alone reached 83.27%.

**Interpretation:** BERT is below both TF-IDF models. Likely reasons: (a) 512-token truncation discards much of long reports, whereas TF-IDF reads every word; (b) cancer type is mostly signalled by distinctive keywords (organ and histology terms), which sparse models capture directly; (c) the rare classes have very few examples to fine-tune on. BERT scored **0.00 F1 on UCS** but **1.00 on CHOL**.

Implementation note: `AdamW` must be imported from `torch.optim`, because newer `transformers` versions removed `transformers.AdamW`.

---

## 10. Method 4 — Sentence embeddings + cosine k-NN

**File:** `demo/ProjectCode/method4_embedding_knn.py`

**Idea:** no training at all. Embed every report as a dense vector, then label a test report by majority vote among its K most similar training reports (cosine similarity). This method was proposed by the student; the sponsor's reference code does not include it.

```python
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"   # 384-dim, ~80 MB, CPU-friendly
BATCH_SIZE = 64
K_VALUES   = [1, 3, 5, 7, 11]                            # all tested, best reported
```

**Pipeline:**
1. Encode train and test reports with `SentenceTransformer.encode` (about 6 minutes on CPU for 9,523 reports).
2. Cache the embeddings in `.npz` files **keyed by `patient_id`**, with matching metadata CSVs, under `demo/ProjectCode/output/`:
   - `method4_train_embeddings_sentence_transformers_all_MiniLM_L6_v2.npz`
   - `method4_test_embeddings_..._.npz`, `method4_train_meta_....csv`, `method4_test_meta_....csv`

   Keying by ID rather than row order means a re-ordered dataframe cannot silently misalign vectors and labels. Later runs load the cache instantly.
3. L2-normalize the vectors and compute the cosine similarity matrix (test × train) with a matrix product.
4. Take the top-K neighbours and use their majority label as the prediction.

**Results by K:**

| K | 1 | 3 | **5** | 7 | 11 |
|---|---|---|---|---|---|
| Accuracy | 89.99% | 89.92% | **90.13%** | 89.88% | 89.60% |

Best: **K=5, 90.13% accuracy**, F1 weighted 89.93%, F1 macro 87.78%.

**Model comparison:** the medical NLI/STS model `pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb` (768-dim) scored **84.07%**, worse than general-purpose MiniLM. Domain pretraining does not help when the model was tuned for sentence-pair inference rather than document similarity, and with a 256-token input limit both models see only the start of each report.

**Why this method still matters:** it needs no labels to build the index, new classes can be added by inserting examples, and every prediction can be justified by showing the five similar past cases. Its embedding cache is also the retrieval index for Methods 5 and 6.

---

## 11. Method 5 — RAG + local LLM (Ollama)

**File:** `demo/ProjectCode/method5_rag_llm.py`

**Idea:** Retrieval-Augmented Generation. Retrieve the most similar labeled training report (using Method 4's embeddings), put it in a prompt as a worked example, and ask a local LLM to classify the new report. Unlike Method 4, a language model reads the text and decides; no votes are counted.

**Configuration:**

```python
OLLAMA_MODEL   = "llama3.2:latest"     # ~3B parameters, runs on CPU
OLLAMA_URL     = "http://localhost:11434"
SAMPLE_SIZE    = 50                    # random test reports evaluated (seed 42)
K_EXAMPLES     = 1                     # retrieved examples per prompt
MAX_WORDS_EX   = 60                    # example report truncated to 60 words
MAX_WORDS_NEW  = 80                    # query report truncated to 80 words
OLLAMA_TIMEOUT = 300
```

**Prompt** (shared with Method 6):

```
Classify the cancer type. Reply with ONLY the code from this list:
ACC, BLCA, BRCA, ... , UVM

KNOWN EXAMPLE (<label of retrieved report>):
<first 60 words of the most similar training report>

CLASSIFY THIS:
<first 80 words of the test report>

Cancer type code:
```

**Response parsing:** uppercase the reply and take an exact code match, otherwise the first valid code found as a whole word (`\bCODE\b`), otherwise `UNKNOWN`, which counts as wrong.

**Prerequisites:** install Ollama, run `ollama pull llama3.2`, start `ollama serve`, and run Method 4 first so the embedding cache exists.

**Results:** **64.00% accuracy** (F1 weighted 67.07%) on 50 sampled reports.

**Why so low:** the small 3B model, aggressive truncation (80 words of a ~560-word report), only one example, and CPU limits that forced a small sample. The prompt was kept short deliberately so CPU inference would finish in reasonable time.

---

## 12. Method 6 — RAG + hosted LLM

**File:** `demo/ProjectCode/method6_rag_openai.py`

**Idea:** the **identical** retrieval, prompt and parsing as Method 5, with a stronger LLM. This isolates the effect of the model.

**Provider switch:** two lines at the top of the file select the provider. A dispatcher (`call_llm` → `_call_openai` / `_call_anthropic` / `_call_ollama`) handles the rest.

```python
LLM_PROVIDER = "openai"        # "openai" | "anthropic" | "ollama"
LLM_MODEL    = "gpt-4o-mini"   # e.g. "gpt-4o", "claude-3-5-haiku-20241022", "llama3.2:latest"
SAMPLE_SIZE  = 300             # random test reports (seed 42)
# K_EXAMPLES = 1, MAX_WORDS_EX = 60, MAX_WORDS_NEW = 80 — same as Method 5
```

Calls use `temperature=0` and `max_tokens=10`. API keys are read from `demo/ProjectCode/.env` (`OPENAI_API_KEY=...` or `ANTHROPIC_API_KEY=...`) by a small built-in `.env` loader and are never hard-coded. `.env` is git-ignored.

**Results:** **89.00% accuracy** on 300 sampled reports with GPT-4o-mini.

**Interpretation:** a 25-point gain over Method 5 from changing only the LLM. With the same retrieval, **the model's built-in medical knowledge is the dominant factor**. It still trails TF-IDF + SVM by about 8 points, while costing money per call and sending text outside the institution.

---

## 13. Per-class analysis: the hard cancer types

The same few classes are hard for every method. They are rare, histologically similar to a neighbour, or both.

| Class | Test n | Confused with | M2 SVM F1 | M4 k-NN F1 | M3 BERT F1 |
|---|---|---|---|---|---|
| **UCS** Uterine carcinosarcoma | 17 | OV, UCEC (gynecologic) | 0.79 (recall 0.65) | 0.43 | **0.00** |
| **CHOL** Cholangiocarcinoma | 13 | LIHC, PAAD (hepatobiliary) | 0.83 | 0.81 | 1.00 |
| **READ** Rectum adenocarcinoma | 49 | COAD (same histology) | 0.86 | 0.72 | — |
| **LUSC / LUAD** Lung | 140 / 146 | each other | 0.92 / 0.94 | 0.73 / 0.76 | — |
| **KICH** Kidney chromophobe | 34 | KIRC, KIRP | 0.97 | 0.63 | — |

Observations:
- **COAD vs READ** is partly a labelling problem: reports for colorectal tumours near the rectosigmoid junction describe the same disease, and the text often does not settle which TCGA project the case was assigned to.
- **TF-IDF + SVM** is the most robust on rare classes, thanks to `class_weight="balanced"` and the fact that it reads every word.
- **Embeddings** blur the lung and kidney subtypes, because semantically "lung carcinoma" reports look alike; the distinguishing terms ("squamous" vs "adeno", "chromophobe") carry little weight in a sentence embedding.

---

## 14. Demo application (Streamlit explainer)

**File:** `demo/explainer.py` — the primary demo, built to walk a professor or audience through how each method works on real reports.

**Run:**
```bash
python demo/save_models.py         # once — trains and saves BoW / TF-IDF / LR / SVM (~1–2 min)
streamlit run demo/explainer.py    # opens http://localhost:8501
```

**Layout:**
- **Sidebar:** pick a sample test report from 10 cancer types (BRCA, LUAD, GBM, KIRC, THCA, PRAD, BLCA, OV, STAD, SKCM). Samples come from the Method 4 test metadata CSV.
- **Tabs:**
  1. *Cedars-Sinai (BoW + LR)*
  2. *Method 1: TF-IDF + LR*
  3. *Method 2: TF-IDF + SVM*
  4. *Method 3: BERT fine-tuned*
  5. *Method 4: Embedding k-NN*
  6. *Method 5: RAG + LLM*
  7. *Method 6: RAG + OpenAI*
  8. *Results comparison*

Each tab explains the method, shows the key code, runs a live prediction on the selected report, and visualizes the result with Plotly: top-5 class probabilities, the report's highest-weighted terms, the class's most predictive words, similar retrieved cases, and the final metrics.

**What each tab needs:**

| Tab | Requires |
|---|---|
| Cedars, M1, M2 | `demo/saved_models/*` (committed, or regenerate with `save_models.py`) |
| M3 | `demo/ProjectCode/method3_best_model/` (not in git) and `torch` + `transformers` |
| M4 | Embedding cache in `demo/ProjectCode/output/` (run Method 4 once) |
| M5 | Ollama running with `llama3.2` |
| M6 | `OPENAI_API_KEY` in `.env` |

Tabs whose prerequisites are missing show an explanation instead of crashing, so the static parts of the demo still work without the heavy artifacts.

**`save_models.py`** reproduces the Cedars BoW split and LR model, and the Method 1/2 TF-IDF vectorizer and classifiers with exactly the hyperparameters from §6–§8, then writes them to `demo/saved_models/`.

**Diagrams:** `demo/pipeline_diagram.html` ("MEDREx Overall Pipeline") and `demo/demo_flow_diagram.html` ("MEDREx Demo Flow") are standalone HTML pages for slides. Open them in a browser.

**History:** an earlier version (April 2026) was a FastAPI backend plus Streamlit frontend with a ChromaDB vector store. It was removed in favour of standalone method scripts plus this single explainer app; the leftover `backend/` entries in `.gitignore` date from that version.

---

## 15. Notebooks

| Notebook | Contents |
|---|---|
| `notebooks/01_EDA.ipynb` | Load data → join on `patient_id` → missing-value check → cancer-type distribution → class-imbalance analysis → text-length analysis → sample reports → summary |
| `notebooks/02_TF_IDF_Baseline.ipynb` | Load → stratified train/val/test split → TF-IDF → LR (Method 1) → Linear SVM (Method 2) → comparison table → LR classification report → confusion matrix (top 15 classes) → per-class F1 ("which cancers are hard?") → save model for the demo app |

These are the early exploratory versions. The authoritative numbers come from the scripts in `demo/ProjectCode/`.

---

## 16. Reference code

`reference_code/` holds unmodified copies of external code, kept for comparison. Paths inside assume the original authors' `../../data/` layout.

| Folder | Source | Contents |
|---|---|---|
| `cancer_type/` | `guilopgar/AI-Campus-Project-7-NLP` (Cedars-Sinai) | Notebooks 1–5: compile, split, BoW, LR, RF, tuning. **The baseline this project beats.** |
| `tnm_stage/` | same | TNM staging: BoW+LR, **Clinical-BigBird** BERT for T/N/M (T-stage 79.6%), **Llama-3.1-8B zero-shot** via vLLM with JSON-schema guided decoding (T 54.1%, N 84.3%, M 90.9%), eval notebooks, and `prompts/` (system prompt, T/N/M prompts, `tnm_zs.json`) |
| `tatonetti_cancer_type/` | `tatonetti-lab/tcga-path-reports` | Bio_ClinicalBERT training/testing scripts, report cleaning, split, eval metrics (512 tokens, 10 epochs, batch 16, ROC-AUC) |
| `tatonetti_tnm/` | `tatonetti-lab/tnm-stage-classifier` | TNM dataset split notebook, requirements |
| `utils.py`, `llm_utils.py` | Cedars-Sinai | Shared helpers used by the notebooks |

Gaps in the reference code that MEDREx addresses: no TF-IDF (only raw counts), no SVM, no BERT for *cancer type* (only for TNM), no class-imbalance handling, no embedding-similarity or RAG approaches, and limited per-class error analysis.

---

## 17. How to reproduce

### Environment
- Python 3.9+ (developed on 3.11.9, Windows 11)
- `pip install -r demo/requirements.txt`, which installs: streamlit, pandas, numpy, scikit-learn, plotly, joblib, nltk, sentence-transformers, requests, openai, python-dotenv
- Additionally for Method 3: `torch`, `transformers` (not in `requirements.txt`; `sentence-transformers` pulls in both)
- Optional: `anthropic` (Method 6 with Anthropic), Ollama (Method 5)

### Steps
1. Clone: `git clone -b submission https://github.com/Kingslee1604/MEDREx-Masters-Project.git`
2. Put the data in `DataSet/` with the exact paths in §3–§4.
3. **Edit `BASE_DIR`** at the top of every script you run (and of `explainer.py` and `save_models.py`) to your clone's path (see §18).
4. Run, in this order (later steps reuse earlier outputs):

```bash
python demo/CedarCinaiCode/cedars_sinai_pipeline.py      # baseline (~10–25 min)
python demo/ProjectCode/method1_tfidf_logistic_regression.py
python demo/ProjectCode/method2_tfidf_svm.py
python demo/ProjectCode/method4_embedding_knn.py         # builds embedding cache — needed by 5, 6, UI
python demo/ProjectCode/method5_rag_llm.py               # needs `ollama serve`
python demo/ProjectCode/method6_rag_openai.py            # needs demo/ProjectCode/.env
# Method 3: run the Colab notebook, download the model, then:
python demo/ProjectCode/method3_evaluate_local.py
python demo/save_models.py
streamlit run demo/explainer.py
```

Seeds are fixed (`random_state=42` for MEDREx, `0` for the Cedars baseline), so Methods 1, 2 and 4 should reproduce exactly. BERT varies slightly between GPU runs, and LLM outputs can vary between API model versions even at temperature 0.

---

## 18. Known issues and inconsistencies

Listed so that anyone reusing this repo is not caught out.

| # | Issue | Where | Impact / fix |
|---|---|---|---|
| 1 | **Hard-coded absolute Windows paths.** `BASE_DIR = r"c:\Users\kings\OneDrive\..."` | Every script, `explainer.py`, `save_models.py` | Fails on any other machine. Replace with `Path(__file__).resolve().parents[N]`. |
| 2 | **Class count** | Early project notes said 35 | The real number is **32** (the label file has 33 codes; LAML has no reports). The README now says 32. |
| 3 | **`.env` location** | `method6_rag_openai.py` | The script reads its key from `demo/ProjectCode/.env`, not `demo/.env`. The README now says this. |
| 4 | **Method 5 docstring says 300 reports** | `method5_rag_llm.py` header | The config is `SAMPLE_SIZE = 50`, and the reported 64% comes from 50 reports with `llama3.2:latest`. Only the docstring is out of date. |
| 5 | **Methods 5 and 6 not on the full test set** | 50 and 300 samples | Their accuracy has wider confidence intervals (roughly ±13 and ±3.5 points at 95%) and is not strictly comparable to Methods 1–4. |
| 6 | **Different split from the baseline** | Cedars: 50/20/30, seed 0. MEDREx: 70/30, seed 42 | MEDREx trains on 6,666 reports vs 4,761. Part of the +1–2 point gain could come from the extra training data. For a strict comparison, rerun the baseline on the 70/30 split. |
| 7 | **Google Drive folder case mismatch** | BERT notebook reads from `MyDrive/MedRex/` but saves to `MyDrive/MEDREx/` | Drive paths are case-sensitive in Colab, so these are two folders. Pick one. |
| 8 | **Truncation in Methods 3–6** | BERT 512 tokens, MiniLM 256 tokens, RAG prompts 60/80 words | Much of each long report is never seen. This is the main reason dense models trail TF-IDF. |
| 9 | **No validation set in MEDREx methods** | Methods 1–4 | Fine as long as no hyperparameters are tuned on the test set. Method 4 chooses K on the test set, a mild optimistic bias (the spread across K is only 0.5 points). |
| 10 | **`.sql` extension on log files** | `demo/**/*.sql` | They are text logs, not SQL. Rename to `.log`/`.txt` if they confuse tools. |
| 11 | **Stale `.gitignore` entries** | `backend/models/`, `backend/vector_store/` | Left over from the removed FastAPI version; harmless. |
| 12 | **`torch`/`transformers` not pinned** | `demo/requirements.txt` | Installed indirectly through sentence-transformers. List them explicitly for Method 3. |

---

## 19. Conclusions

1. **Classical sparse models win this task.** TF-IDF + linear SVM reaches 97.37%, trains in 12 seconds on a CPU, and is fully explainable through per-word weights. For cancer *type*, the answer is usually stated in a few distinctive words (organ, histology), which bag-of-n-grams models capture directly.
2. **Feature engineering beat the baseline.** Switching from raw counts to TF-IDF with bigrams, sublinear TF and balanced class weights added ~1 point with the same classifier; switching LR to SVM added another ~1 point.
3. **Fine-tuned BERT did not justify its cost here** (94.57%, needs a GPU). Truncation and rare classes hurt it. It is likely to matter more for tasks that need reasoning over context, such as TNM staging.
4. **Embedding k-NN is a useful training-free baseline** (90.13%) that gives case-based explanations, and it provides the retrieval layer for RAG. Domain-specific embeddings were *not* automatically better.
5. **For RAG, the LLM matters more than the retrieval.** The same pipeline scored 64% with a 3B local model and 89% with GPT-4o-mini.
6. **The practical recommendation for a hospital:** deploy TF-IDF + SVM for cancer-type triage (local, cheap, auditable). Keep LLM-based methods for tasks where keyword models fall short, run them on in-house models for privacy, and route low-confidence or rare-class cases (UCS, CHOL, READ) to human review.

---

## 20. Not yet done / future work

- **TNM staging (Task 2).** The labels (`DataSet/TNM_Stage/`) and the sponsor's BERT and LLM reference code are present, but no MEDREx method has been trained for T/N/M.
- **Survival / clinical-endpoint analysis.** `Clinical_Endpoints/` is unused, so prediction errors are not yet linked to outcomes.
- **Long-document handling.** Chunk-and-pool BERT, Clinical-BigBird or Longformer (4,096 tokens), or embedding several chunks per report.
- **Better RAG.** K>1 retrieved examples, full-length text with a larger-context model, few-shot or chain-of-thought prompting, and full-test-set evaluation.
- **Fair baseline comparison.** Rerun the Cedars BoW pipeline on the MEDREx 70/30 split.
- **Statistical testing.** McNemar's test between methods on the shared test set; bootstrap confidence intervals.
- **Portability.** Relative paths, a single config file, and a pinned `requirements.txt`.
- **Thesis write-up** (Phase 9).
