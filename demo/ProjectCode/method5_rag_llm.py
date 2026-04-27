"""
MEDREx — Method 5: RAG + LLM (Ollama SLM)
==========================================
RAG = Retrieval Augmented Generation

How it works:
  1. Reuse Method 4's MiniLM embedding cache — no re-embedding needed
  2. For each test report: find top-3 most similar TRAINING reports
     (cosine similarity on embeddings — same math as Method 4)
  3. Build a prompt: "Here are 3 confirmed cases [examples]. Classify this new report."
  4. Send prompt to local Ollama SLM (llama3.2:3b)
  5. Parse SLM response → predicted cancer type code

Why this is different from Method 4 (k-NN):
  Method 4: finds similar reports → COUNTS votes → majority wins (no AI reasoning)
  Method 5: finds similar reports → PASSES TEXT to LLM → LLM reads + reasons → answers

Runs on 300 randomly sampled test reports (statistically valid, ~50 min on CPU).

Before running:
  1. Install Ollama from https://ollama.com
  2. Run in terminal: ollama pull llama3.2:3b
  3. Make sure ollama is running: ollama serve
  4. Run Method 4 first to build embedding cache

Run:
  python demo/ProjectCode/method5_rag_llm.py
"""

import os, sys, logging, warnings, time, re
import numpy as np
import requests
warnings.filterwarnings("ignore")

# ── Configuration ─────────────────────────────────────────────────────────────
OLLAMA_MODEL  = "llama3.2:latest"      # SLM model — already installed
OLLAMA_URL    = "http://localhost:11434"
SAMPLE_SIZE   = 50                     # test reports to evaluate (50 = ~30 min, enough for thesis)
RANDOM_SEED   = 42                     # same seed as all other methods
K_EXAMPLES    = 1                      # number of similar cases to show LLM (1 = shortest prompt)
MAX_WORDS_EX  = 60                     # truncate each example to this many words
MAX_WORDS_NEW = 80                     # truncate the test report to this many words
OLLAMA_TIMEOUT = 300                   # seconds to wait for each LLM response

# ── Embedding model — must match what Method 4 used ───────────────────────────
EMB_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_MODEL_TAG     = EMB_MODEL_NAME.replace("/", "_").replace("-", "_")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR       = r"c:\Users\kings\OneDrive\Desktop\CSUDH_Kingslee\CSUDH-Spring-2026\CSC-590-MastersProject\MastersProject"
OUT_DIR        = os.path.join(BASE_DIR, "demo", "ProjectCode", "output")
TRAIN_EMB_PATH = os.path.join(OUT_DIR, f"method4_train_embeddings_{_MODEL_TAG}.npz")
TEST_EMB_PATH  = os.path.join(OUT_DIR, f"method4_test_embeddings_{_MODEL_TAG}.npz")
TRAIN_META_PATH= os.path.join(OUT_DIR, f"method4_train_meta_{_MODEL_TAG}.csv")
TEST_META_PATH = os.path.join(OUT_DIR, f"method4_test_meta_{_MODEL_TAG}.csv")
_LOG_FILE      = os.path.join(OUT_DIR, "method5_rag_llm_run.log")

os.makedirs(OUT_DIR, exist_ok=True)

# ── Logging ───────────────────────────────────────────────────────────────────
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

def section(title):
    log.info("=" * 70)
    log.info(f"  {title}")
    log.info("=" * 70)

def subsection(title):
    log.info(f"--- {title} ---")

# ── Cancer type reference ──────────────────────────────────────────────────────
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
VALID_CODES = sorted(CANCER_NAMES.keys())


# ── Helper: load .npz cache ────────────────────────────────────────────────────
def load_npz(path):
    """Load .npz → dict {patient_id: embedding_vector}"""
    data = np.load(path, allow_pickle=True)
    return dict(zip(data["patient_ids"], data["embeddings"]))


def build_matrix(emb_dict, ordered_ids):
    """Return embedding matrix in the order of ordered_ids."""
    missing = [pid for pid in ordered_ids if pid not in emb_dict]
    if missing:
        raise KeyError(f"Missing patient_ids in cache: {missing[:5]}")
    return np.array([emb_dict[pid] for pid in ordered_ids])


# ── Helper: cosine similarity ──────────────────────────────────────────────────
def cosine_similarity_matrix(query_matrix, key_matrix):
    """
    query_matrix : (n_queries, dim)
    key_matrix   : (n_keys, dim)
    Returns      : (n_queries, n_keys) similarity scores in [-1, 1]
    """
    q_norm = query_matrix / (np.linalg.norm(query_matrix, axis=1, keepdims=True) + 1e-10)
    k_norm = key_matrix   / (np.linalg.norm(key_matrix,   axis=1, keepdims=True) + 1e-10)
    return q_norm @ k_norm.T


# ── Helper: build RAG prompt ───────────────────────────────────────────────────
def build_prompt(test_text, examples):
    """
    examples : list of dicts with keys 'text', 'cancer_type', 'cancer_name'
    """
    codes_str = ", ".join(VALID_CODES)

    ex = examples[0]
    ex_words   = " ".join(ex["text"].split()[:MAX_WORDS_EX])
    test_words = " ".join(test_text.split()[:MAX_WORDS_NEW])

    prompt = (
        f"Classify the cancer type. Reply with ONLY the code from this list:\n"
        f"{codes_str}\n\n"
        f"KNOWN EXAMPLE ({ex['cancer_type']}):\n{ex_words}\n\n"
        f"CLASSIFY THIS:\n{test_words}\n\n"
        f"Cancer type code:"
    )
    return prompt


# ── Helper: call Ollama ────────────────────────────────────────────────────────
def call_ollama(prompt):
    """Send prompt to local Ollama, return raw response string."""
    try:
        r = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.0, "num_predict": 10, "num_ctx": 512},
            },
            timeout=OLLAMA_TIMEOUT,
        )
        r.raise_for_status()
        return r.json().get("response", "").strip()
    except requests.exceptions.ConnectionError:
        log.error("  Ollama is not running! Start it with: ollama serve")
        sys.exit(1)
    except Exception as e:
        log.warning(f"  Ollama call failed: {e}")
        return ""


# ── Helper: parse cancer code from LLM response ───────────────────────────────
def parse_response(response):
    """
    Extract a valid cancer type code from the LLM's response.
    Tries exact match first, then searches for any valid code in the text.
    Returns 'UNKNOWN' if nothing valid found.
    """
    cleaned = response.strip().upper()

    # Try exact match first (model responded with just the code)
    if cleaned in VALID_CODES:
        return cleaned

    # Search for any valid code in the response
    for code in VALID_CODES:
        if re.search(r'\b' + code + r'\b', cleaned):
            return code

    return "UNKNOWN"


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import pandas as pd
    from sklearn.metrics import accuracy_score, f1_score, classification_report

    t_start = time.time()

    section("METHOD 5 — RAG + LLM (Ollama SLM)")
    log.info(f"  Ollama model   : {OLLAMA_MODEL}")
    log.info(f"  Sample size    : {SAMPLE_SIZE} test reports")
    log.info(f"  K examples     : {K_EXAMPLES} similar cases per prompt")
    log.info(f"  Embedding model: {EMB_MODEL_NAME}")
    log.info(f"  Log file       : {_LOG_FILE}")

    # ── STEP 1 — Check Ollama is running ──────────────────────────────────────
    section("STEP 1 — Check Ollama is running")
    try:
        resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        available = [m["name"] for m in resp.json().get("models", [])]
        log.info(f"  Ollama is running.")
        log.info(f"  Available models: {available}")
        if not any(OLLAMA_MODEL in m for m in available):
            log.error(f"  Model '{OLLAMA_MODEL}' not found!")
            log.error(f"  Run this in your terminal: ollama pull {OLLAMA_MODEL}")
            sys.exit(1)
        log.info(f"  Model '{OLLAMA_MODEL}' is ready.")
    except requests.exceptions.ConnectionError:
        log.error("  Ollama is NOT running.")
        log.error("  Start it with: ollama serve")
        log.error("  Then pull the model: ollama pull llama3.2:3b")
        sys.exit(1)

    # ── STEP 2 — Load Method 4 embedding cache ────────────────────────────────
    section("STEP 2 — Load embedding cache from Method 4")

    for path, label in [(TRAIN_EMB_PATH, "Train embeddings"),
                        (TEST_EMB_PATH,  "Test  embeddings"),
                        (TRAIN_META_PATH,"Train meta CSV"),
                        (TEST_META_PATH, "Test  meta CSV")]:
        if not os.path.exists(path):
            log.error(f"  MISSING: {path}")
            log.error("  Run method4_embedding_knn.py first to build the cache.")
            sys.exit(1)
        log.info(f"  Found: {label}")

    subsection("Loading train cache")
    train_emb_dict = load_npz(TRAIN_EMB_PATH)
    train_meta     = pd.read_csv(TRAIN_META_PATH)
    train_meta.index = train_meta["patient_id"].values
    log.info(f"  Train embeddings : {len(train_emb_dict):,} reports")
    log.info(f"  Train meta rows  : {len(train_meta):,}")

    subsection("Loading test cache")
    test_emb_dict = load_npz(TEST_EMB_PATH)
    test_meta     = pd.read_csv(TEST_META_PATH)
    test_meta.index = test_meta["patient_id"].values
    log.info(f"  Test embeddings  : {len(test_emb_dict):,} reports")
    log.info(f"  Test meta rows   : {len(test_meta):,}")

    # ── STEP 3 — Build matrices ───────────────────────────────────────────────
    section("STEP 3 — Build embedding matrices")

    train_ids    = train_meta["patient_id"].values
    train_matrix = build_matrix(train_emb_dict, train_ids)
    log.info(f"  Train matrix : {train_matrix.shape}  (n_reports × embedding_dim)")

    # ── STEP 4 — Sample test reports ─────────────────────────────────────────
    section(f"STEP 4 — Sample {SAMPLE_SIZE} test reports (seed={RANDOM_SEED})")

    rng          = np.random.default_rng(RANDOM_SEED)
    test_ids_all = test_meta["patient_id"].values
    sample_idx   = rng.choice(len(test_ids_all), size=min(SAMPLE_SIZE, len(test_ids_all)), replace=False)
    sample_ids   = test_ids_all[sample_idx]
    sample_meta  = test_meta.loc[sample_ids]
    sample_matrix = build_matrix(test_emb_dict, sample_ids)

    log.info(f"  Sampled {len(sample_ids)} test reports")
    log.info(f"  Cancer types in sample : {sample_meta['cancer_type'].nunique()} unique types")

    # ── STEP 5 — RAG + LLM loop ───────────────────────────────────────────────
    section("STEP 5 — RAG retrieval + LLM classification")
    log.info(f"  For each test report:")
    log.info(f"    1. Find top-{K_EXAMPLES} similar training reports (cosine similarity)")
    log.info(f"    2. Build prompt with those {K_EXAMPLES} examples + test report")
    log.info(f"    3. Call Ollama ({OLLAMA_MODEL}) → get cancer type prediction")
    log.info(f"  Estimated time: {SAMPLE_SIZE * 15 // 60}–{SAMPLE_SIZE * 25 // 60} min on CPU")
    log.info("")

    # Pre-compute all cosine similarities at once (fast numpy batch operation)
    subsection("Pre-computing cosine similarities (batch)")
    sim_matrix = cosine_similarity_matrix(sample_matrix, train_matrix)
    log.info(f"  Similarity matrix: {sim_matrix.shape}  (sample × train)")

    all_preds    = []
    all_labels   = []
    all_raw_resp = []
    unknown_count = 0

    t_loop = time.time()

    for i, pid in enumerate(sample_ids):
        true_label = sample_meta.loc[pid, "cancer_type"]
        test_text  = str(sample_meta.loc[pid, "text"])

        # Find top-K similar training reports
        sim_scores   = sim_matrix[i]
        top_k_idx    = np.argsort(sim_scores)[::-1][:K_EXAMPLES]
        top_k_pids   = train_ids[top_k_idx]
        top_k_scores = sim_scores[top_k_idx]

        examples = []
        for k_pid, score in zip(top_k_pids, top_k_scores):
            row = train_meta.loc[k_pid]
            examples.append({
                "text"       : str(row["text"]),
                "cancer_type": row["cancer_type"],
                "cancer_name": CANCER_NAMES.get(row["cancer_type"], row["cancer_type"]),
                "similarity" : score,
            })

        # Build prompt and call LLM
        prompt   = build_prompt(test_text, examples)
        raw_resp = call_ollama(prompt)
        pred     = parse_response(raw_resp)

        if pred == "UNKNOWN":
            unknown_count += 1

        all_preds.append(pred)
        all_labels.append(true_label)
        all_raw_resp.append(raw_resp)

        # Progress update every 10 reports
        if (i + 1) % 10 == 0:
            elapsed   = time.time() - t_loop
            per_report = elapsed / (i + 1)
            remaining = per_report * (len(sample_ids) - i - 1)
            correct_so_far = sum(p == l for p, l in zip(all_preds, all_labels))
            acc_so_far = correct_so_far / (i + 1) * 100
            log.info(
                f"  [{i+1:>3}/{len(sample_ids)}] "
                f"Acc={acc_so_far:.1f}%  "
                f"Unknown={unknown_count}  "
                f"ETA={remaining/60:.1f}min  "
                f"Last: true={true_label} pred={pred} raw='{raw_resp[:30]}'"
            )

    total_time = (time.time() - t_start) / 60
    log.info(f"\n  Loop complete in {(time.time()-t_loop)/60:.1f} min")
    log.info(f"  Unparseable responses (UNKNOWN): {unknown_count}/{len(sample_ids)}")

    # ── STEP 6 — Evaluation ────────────────────────────────────────────────────
    section("STEP 6 — Results")

    # Filter out UNKNOWN for metric calculation (standard practice)
    valid_mask   = [p != "UNKNOWN" for p in all_preds]
    valid_preds  = [p for p, m in zip(all_preds,  valid_mask) if m]
    valid_labels = [l for l, m in zip(all_labels, valid_mask) if m]

    if len(valid_preds) == 0:
        log.error("  No valid predictions — check model output above.")
        sys.exit(1)

    # Accuracy on valid predictions
    acc_valid = accuracy_score(valid_labels, valid_preds) * 100
    # Accuracy including UNKNOWN as wrong (conservative)
    acc_all   = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_preds) * 100

    f1_w = f1_score(valid_labels, valid_preds, average="weighted", zero_division=0) * 100
    f1_m = f1_score(valid_labels, valid_preds, average="macro",    zero_division=0) * 100

    log.info(f"  Sample size          : {len(sample_ids)}")
    log.info(f"  Valid predictions    : {len(valid_preds)} ({len(valid_preds)/len(sample_ids)*100:.1f}%)")
    log.info(f"  Unknown (unparsed)   : {unknown_count}")
    log.info("")
    log.info(f"  ╔══════════════════════════════════════════════════════════╗")
    log.info(f"  ║  METHOD 5 — RAG + LLM ({OLLAMA_MODEL}) RESULTS")
    log.info(f"  ║                                                          ║")
    log.info(f"  ║  Accuracy (valid only) : {acc_valid:>6.2f}%                    ║")
    log.info(f"  ║  Accuracy (all, strict): {acc_all:>6.2f}%                    ║")
    log.info(f"  ║  F1 Weighted           : {f1_w:>6.2f}%                    ║")
    log.info(f"  ║  F1 Macro              : {f1_m:>6.2f}%                    ║")
    log.info(f"  ╚══════════════════════════════════════════════════════════╝")

    subsection("Full comparison — all methods")
    log.info(f"  {'Method':<40} {'Accuracy':>10}")
    log.info(f"  {'-'*52}")
    log.info(f"  {'Cedars-Sinai (BoW + LR)':<40} {'95.31%':>10}")
    log.info(f"  {'Method 1 (TF-IDF + LR)':<40} {'96.36%':>10}")
    log.info(f"  {'Method 2 (TF-IDF + SVM)':<40} {'97.37%':>10}")
    log.info(f"  {'Method 4 (Embedding k-NN, MiniLM)':<40} {'90.13%':>10}")
    log.info(f"  {'Method 3 (Bio_ClinicalBERT)':<40} {'TBD':>10}")
    log.info(f"  {'Method 5 (RAG+LLM, '+OLLAMA_MODEL+')':<40} {acc_all:>9.2f}%")

    subsection("Per-class results (valid predictions only)")
    log.info(classification_report(valid_labels, valid_preds, zero_division=0))

    subsection("Sample of predictions (first 20)")
    log.info(f"  {'#':<4} {'True':<8} {'Pred':<8} {'Match':<6} {'Raw LLM response':<40}")
    log.info(f"  {'-'*68}")
    for i in range(min(20, len(all_preds))):
        match  = "OK" if all_preds[i] == all_labels[i] else "WRONG"
        raw_30 = all_raw_resp[i][:40].replace("\n", " ")
        log.info(f"  {i+1:<4} {all_labels[i]:<8} {all_preds[i]:<8} {match:<6} {raw_30}")

    section("DONE")
    log.info(f"  Total runtime : {total_time:.1f} min")
    log.info(f"  Log saved to  : {_LOG_FILE}")
