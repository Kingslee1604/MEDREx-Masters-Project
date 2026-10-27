# MEDREx — Medical Report Extraction and Classification

**Masters Project | CSC-590 | CSUDH Spring 2026**
**Student:** Kingslee Dominic Savio Velu
**Mentor:** Dr. Bin Tang
**Sponsor:** Cedars-Sinai AI Campus — Project #32

> For the full technical write-up (dataset, every method's settings, per-class results, known issues), see [PROJECT_DETAILS.md](PROJECT_DETAILS.md).

---

## Project Overview

MEDREx compares six NLP methods for classifying cancer types from TCGA pathology reports. Each method predicts one of **32 cancer types** from the free text of a report, using approaches that range from classical TF-IDF to Retrieval-Augmented Generation (RAG).

**Dataset:** 9,523 TCGA pathology reports covering 32 cancer types. The label file lists 33 codes, but LAML has no matching reports.

---

## Six Methods Implemented

All methods use the same stratified 70/30 split (`random_state=42`): 6,666 training reports and 2,857 test reports.

| Method | Approach | Accuracy | Evaluated on |
|--------|----------|----------|--------------|
| Method 1 | TF-IDF + Logistic Regression | 96.36% | 2,857 test reports |
| **Method 2** | **TF-IDF + Linear SVM** | **97.37%** | 2,857 test reports |
| Method 3 | Fine-tuned BERT (Bio_ClinicalBERT) | 94.57% | 2,857 test reports |
| Method 4 | Sentence Embeddings (MiniLM) + k-NN, K=5 | 90.13% | 2,857 test reports |
| Method 5 | RAG + Ollama (llama3.2, local) | 64.00% | 50-report random sample |
| Method 6 | RAG + OpenAI GPT-4o-mini | 89.00% | 300-report random sample |

**Baseline (Cedars-Sinai):** BoW + Logistic Regression = 95.31% (published), 95.48% in our re-run. The baseline uses its own 50/20/30 split.

---

## How to Run the Demo

### Requirements
- Python 3.9 or higher (developed on 3.11)
- Git

### Step 1 — Clone the submission branch

```bash
git clone -b submission https://github.com/Kingslee1604/MEDREx-Masters-Project.git
cd MEDREx-Masters-Project
```

### Step 2 — (Optional) Create a virtual environment

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate
```

### Step 3 — Install dependencies

```bash
pip install -r demo/requirements.txt
```

Then download the NLTK data used by the Cedars-Sinai baseline, `save_models.py` and the demo UI (one time only):

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords')"
```

> The scripts also try to download this data silently when they start. That download fails without internet access or write permission to the NLTK data folder, and tokenization then errors out. Running the command above once avoids this.

### Step 4 — Set the project path

Every script, `demo/explainer.py` and `demo/save_models.py` sets `BASE_DIR` near the top to the original author's Windows path. Change it to the folder where you cloned the repo:

```python
BASE_DIR = r"C:\path\to\MEDREx-Masters-Project"
```

### Step 5 — Set up API key (Method 6 only)

Create a file named `.env` inside the `demo/ProjectCode/` folder:

```
OPENAI_API_KEY=your_openai_api_key_here
```

> If you do not have an OpenAI key, the other five methods still run normally. Only Method 6 needs the key.

### Step 6 — Launch the demo

```bash
streamlit run demo/explainer.py
```

Open your browser and go to: **http://localhost:8501**

The Cedars-Sinai, Method 1 and Method 2 tabs use the models already saved in `demo/saved_models/`. To rebuild those models, run `python demo/save_models.py`; this needs the dataset.

---

## Running Individual Method Scripts

Each method can also be run as a standalone script from the repo root. Every script prints a step-by-step log and saves it to `demo/ProjectCode/output/`. All of them need the dataset in `DataSet/` (see [Dataset Note](#dataset-note)).

```bash
# Cedars-Sinai baseline — BoW + Logistic Regression and Random Forest (~10–25 min)
python demo/CedarCinaiCode/cedars_sinai_pipeline.py

# Method 1 — TF-IDF + Logistic Regression
python demo/ProjectCode/method1_tfidf_logistic_regression.py

# Method 2 — TF-IDF + SVM
python demo/ProjectCode/method2_tfidf_svm.py

# Method 3 — BERT (evaluates the downloaded model in demo/ProjectCode/method3_best_model/)
python demo/ProjectCode/method3_evaluate_local.py

# Method 4 — Embedding similarity k-NN (also builds the embedding cache for Methods 5 and 6)
python demo/ProjectCode/method4_embedding_knn.py

# Method 5 — RAG + Ollama (needs Ollama: https://ollama.com, then `ollama pull llama3.2`)
python demo/ProjectCode/method5_rag_llm.py

# Method 6 — RAG + OpenAI (needs OPENAI_API_KEY in demo/ProjectCode/.env)
python demo/ProjectCode/method6_rag_openai.py
```

---

## Project Structure

```
MEDREx-Masters-Project/
├── README.md
├── PROJECT_DETAILS.md                       # Full technical documentation
├── demo/
│   ├── explainer.py                         # Main Streamlit UI (run this)
│   ├── save_models.py                       # Trains + saves models used by the UI
│   ├── requirements.txt                     # Python dependencies
│   ├── saved_models/                        # Saved BoW / TF-IDF / LR / SVM models
│   ├── CedarCinaiCode/                      # Cedars-Sinai baseline as runnable scripts
│   │   ├── cedars_sinai_pipeline.py
│   │   ├── bow_logistic_regression.py
│   │   └── bow_random_forest.py
│   ├── ProjectCode/
│   │   ├── method1_tfidf_logistic_regression.py   # Method 1
│   │   ├── method2_tfidf_svm.py                   # Method 2
│   │   ├── method3_evaluate_local.py              # Method 3 (evaluation)
│   │   ├── method4_embedding_knn.py               # Method 4
│   │   ├── method5_rag_llm.py                     # Method 5
│   │   ├── method6_rag_openai.py                  # Method 6
│   │   └── *.sql                                  # Saved run logs (plain text, not SQL)
│   └── BertModel_Code/
│       ├── method3_bert_finetune_colab.ipynb      # BERT training (Google Colab)
│       └── method3_bert_finetune_colab.py
├── notebooks/
│   ├── 01_EDA.ipynb                         # Exploratory Data Analysis
│   └── 02_TF_IDF_Baseline.ipynb             # TF-IDF baseline experiments
├── reference_code/                          # Original Cedars-Sinai / Tatonetti Lab code
└── DataSet/                                 # TCGA data (not in repo — see below)
```

---

## Dataset Note

The TCGA dataset is not included in the repository because of its size (~34 MB) and data-use terms. Place the files here:

```
DataSet/TCGA_Reports.csv/TCGA_Reports.csv
DataSet/tcga_patient_to_cancer_type.csv
```

The demo's sample reports and the Method 4 tab come from the embedding cache in `demo/ProjectCode/output/`. That folder is not in git; running `python demo/ProjectCode/method4_embedding_knn.py` once (with the dataset in place) creates it.

---

## Notes

- **Run order:** run Method 4 before Methods 5 and 6, because they reuse its embedding cache.
- **Method 3 (BERT):** the model weights (~414 MB) are not included. Train with `demo/BertModel_Code/method3_bert_finetune_colab.ipynb` on a Google Colab GPU, then download the model folder to `demo/ProjectCode/method3_best_model/`. Running this method also needs `torch` and `transformers`; installing `sentence-transformers` brings both.
- **Method 5 (Ollama):** needs Ollama installed and running locally (`ollama serve`) with the model pulled (`ollama pull llama3.2`).
- **Method 6 (OpenAI):** needs a valid OpenAI API key in `demo/ProjectCode/.env`. To use Anthropic or Ollama instead, change `LLM_PROVIDER` and `LLM_MODEL` at the top of the script.
- **Methods 1, 2 and 4** run fully offline on a CPU. After the first download of the MiniLM model, Method 4 also needs no internet.
