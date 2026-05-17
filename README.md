# MEDREx — Medical Report Extraction and Classification

**Masters Project | CSC-590 | CSUDH Spring 2026**
**Student:** Kingslee Vinoth Victor
**Sponsor:** Cedars-Sinai AI Campus — Project #32

---

## Project Overview

MEDREx compares six NLP methods for classifying cancer types from TCGA pathology reports. The system predicts one of 35 cancer types from free-text medical reports using approaches ranging from classical TF-IDF to modern Retrieval-Augmented Generation (RAG).

**Dataset:** 9,523 TCGA pathology reports across 35 cancer types

---

## Six Methods Implemented

| Method | Approach | Accuracy |
|--------|----------|----------|
| Method 1 | TF-IDF + Logistic Regression | 96.36% |
| Method 2 | TF-IDF + SVM | 97.37% |
| Method 3 | Fine-tuned BERT (Bio_ClinicalBERT) | 94.57% |
| Method 4 | Sentence Embeddings + k-NN | 90.13% |
| Method 5 | RAG + Ollama (Local LLM) | 64.00% |
| Method 6 | RAG + OpenAI GPT-4o-mini | 89.00% |

**Baseline (Cedars-Sinai):** BoW + Logistic Regression = 95.31%

---

## How to Run the Demo

### Requirements
- Python 3.9 or higher
- Git

### Step 1 — Clone the submission branch

```bash
git clone -b submission https://github.com/Kingslee1604/MastersProject.git
cd MastersProject
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

### Step 4 — Set up API key (Method 6 only)

Create a file named `.env` inside the `demo/` folder:

```
OPENAI_API_KEY=your_openai_api_key_here
```

> If you do not have an OpenAI key, all other 5 methods will still run normally. Method 6 will be skipped.

### Step 5 — Launch the demo

```bash
streamlit run demo/explainer.py
```

Open your browser and go to: **http://localhost:8501**

---

## Project Structure

```
MastersProject/
├── demo/
│   ├── explainer.py              # Main Streamlit UI (run this)
│   ├── requirements.txt          # All Python dependencies
│   ├── ProjectCode/
│   │   ├── method1_tfidf_lr.py          # Method 1
│   │   ├── method2_tfidf_svm.py         # Method 2
│   │   ├── method3_evaluate_local.py    # Method 3 (inference)
│   │   ├── method4_embedding_knn.py     # Method 4
│   │   ├── method5_rag_ollama.py        # Method 5
│   │   └── method6_rag_openai.py        # Method 6
│   └── BertModel_Code/
│       └── method3_bert_finetune_colab.ipynb  # BERT training (Colab)
├── notebooks/
│   ├── 01_EDA.ipynb              # Exploratory Data Analysis
│   └── 02_TF_IDF_Baseline.ipynb # TF-IDF baseline experiments
├── reference_code/               # Cedars-Sinai baseline reference
└── DataSet/                      # TCGA reports (not in repo — see below)
```

---

## Dataset Note

The TCGA pathology reports dataset is not included in the repository due to size (34 MB) and data use agreements. The demo UI runs with pre-computed results and does not require the raw dataset to explore the interface.

To reproduce training results, place `TCGA_Reports.csv` in `DataSet/TCGA_Reports.csv/`.

---

## Notes

- **Method 3 (BERT):** Model weights are not included (414 MB). The training notebook (`demo/BertModel_Code/method3_bert_finetune_colab.ipynb`) is designed to run on Google Colab with GPU.
- **Method 5 (Ollama):** Requires Ollama installed locally with `llama3` model pulled (`ollama pull llama3`).
- **Method 6 (OpenAI):** Requires a valid OpenAI API key in `demo/.env`.
- All other methods (1, 2, 4) run fully offline with no external dependencies beyond pip packages.
