"""
MEDREx — Method 6: RAG + LLM (Multi-Provider)
==============================================
Same RAG pipeline as Method 5 but with swappable LLM providers.

To switch provider: change LLM_PROVIDER + LLM_MODEL at the top.

  Provider    | LLM_PROVIDER  | LLM_MODEL examples
  ------------|---------------|------------------------------------
  OpenAI      | "openai"      | "gpt-4o-mini"  "gpt-4o"
  Anthropic   | "anthropic"   | "claude-3-5-haiku-20241022"
  Ollama      | "ollama"      | "llama3.2:latest"

API keys go in demo/ProjectCode/.env — never hardcoded.

Before running:
  - Run method4_embedding_knn.py first (builds embedding cache)
  - Ensure demo/ProjectCode/.env has your OPENAI_API_KEY

Run:
  python demo/ProjectCode/method6_rag_openai.py
"""

import os, sys, logging, warnings, time, re
import numpy as np
warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# PROVIDER CONFIG — change these 2 lines to switch LLM
# ══════════════════════════════════════════════════════════════════════════════
LLM_PROVIDER = "openai"        # "openai" | "anthropic" | "ollama"
LLM_MODEL    = "gpt-4o-mini"   # openai:    "gpt-4o-mini" | "gpt-4o"
                                # anthropic: "claude-3-5-haiku-20241022"
                                # ollama:    "llama3.2:latest"
# ══════════════════════════════════════════════════════════════════════════════

SAMPLE_SIZE   = 300            # reports to evaluate (300 = statistically strong)
RANDOM_SEED   = 42             # same seed as all other methods
K_EXAMPLES    = 1              # similar training cases shown per prompt
MAX_WORDS_EX  = 60             # truncate example report to this many words
MAX_WORDS_NEW = 80             # truncate test report to this many words
OLLAMA_URL    = "http://localhost:11434"

# ── Embedding model — must match what Method 4 used ──────────────────────────
EMB_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_MODEL_TAG     = EMB_MODEL_NAME.replace("/", "_").replace("-", "_")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR        = r"c:\Users\kings\OneDrive\Desktop\CSUDH_Kingslee\CSUDH-Spring-2026\CSC-590-MastersProject\MastersProject"
OUT_DIR         = os.path.join(BASE_DIR, "demo", "ProjectCode", "output")
ENV_FILE        = os.path.join(BASE_DIR, "demo", "ProjectCode", ".env")
TRAIN_EMB_PATH  = os.path.join(OUT_DIR, f"method4_train_embeddings_{_MODEL_TAG}.npz")
TEST_EMB_PATH   = os.path.join(OUT_DIR, f"method4_test_embeddings_{_MODEL_TAG}.npz")
TRAIN_META_PATH = os.path.join(OUT_DIR, f"method4_train_meta_{_MODEL_TAG}.csv")
TEST_META_PATH  = os.path.join(OUT_DIR, f"method4_test_meta_{_MODEL_TAG}.csv")
_LOG_FILE       = os.path.join(OUT_DIR, "method6_rag_openai_run.log")

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


# ── Load .env file (no external package needed) ───────────────────────────────
def load_env_file(path):
    if not os.path.exists(path):
        return
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, val = line.partition("=")
                os.environ.setdefault(key.strip(), val.strip())


# ── Embedding helpers ─────────────────────────────────────────────────────────
def load_npz(path):
    data = np.load(path, allow_pickle=True)
    return dict(zip(data["patient_ids"], data["embeddings"]))

def build_matrix(emb_dict, ordered_ids):
    missing = [pid for pid in ordered_ids if pid not in emb_dict]
    if missing:
        raise KeyError(f"Missing patient_ids in cache: {missing[:5]}")
    return np.array([emb_dict[pid] for pid in ordered_ids])

def cosine_similarity_matrix(query_matrix, key_matrix):
    q_norm = query_matrix / (np.linalg.norm(query_matrix, axis=1, keepdims=True) + 1e-10)
    k_norm = key_matrix   / (np.linalg.norm(key_matrix,   axis=1, keepdims=True) + 1e-10)
    return q_norm @ k_norm.T


# ── Build RAG prompt ──────────────────────────────────────────────────────────
def build_prompt(test_text, examples):
    codes_str  = ", ".join(VALID_CODES)
    ex         = examples[0]
    ex_words   = " ".join(ex["text"].split()[:MAX_WORDS_EX])
    test_words = " ".join(test_text.split()[:MAX_WORDS_NEW])
    return (
        f"Classify the cancer type. Reply with ONLY the code from this list:\n"
        f"{codes_str}\n\n"
        f"KNOWN EXAMPLE ({ex['cancer_type']}):\n{ex_words}\n\n"
        f"CLASSIFY THIS:\n{test_words}\n\n"
        f"Cancer type code:"
    )


# ══════════════════════════════════════════════════════════════════════════════
# LLM PROVIDER DISPATCHER
# ══════════════════════════════════════════════════════════════════════════════
def call_llm(prompt):
    """Route to the configured LLM provider. Returns raw response string."""
    if LLM_PROVIDER == "openai":
        return _call_openai(prompt)
    elif LLM_PROVIDER == "anthropic":
        return _call_anthropic(prompt)
    elif LLM_PROVIDER == "ollama":
        return _call_ollama(prompt)
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: '{LLM_PROVIDER}'. Use 'openai', 'anthropic', or 'ollama'.")


def _call_openai(prompt):
    try:
        import openai
        client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0.0,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        log.warning(f"  OpenAI call failed: {e}")
        return ""


def _call_anthropic(prompt):
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        resp = client.messages.create(
            model=LLM_MODEL,
            max_tokens=10,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text.strip()
    except Exception as e:
        log.warning(f"  Anthropic call failed: {e}")
        return ""


def _call_ollama(prompt):
    import requests
    try:
        r = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": LLM_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.0, "num_predict": 10, "num_ctx": 512},
            },
            timeout=120,
        )
        r.raise_for_status()
        return r.json().get("response", "").strip()
    except Exception as e:
        log.warning(f"  Ollama call failed: {e}")
        return ""


# ── Validate provider + connectivity ─────────────────────────────────────────
def check_provider():
    section(f"STEP 1 — Check {LLM_PROVIDER.upper()} provider")
    log.info(f"  Provider : {LLM_PROVIDER}")
    log.info(f"  Model    : {LLM_MODEL}")

    if LLM_PROVIDER == "openai":
        key = os.environ.get("OPENAI_API_KEY", "")
        if not key:
            log.error("  OPENAI_API_KEY not set.")
            log.error(f"  Add this line to: {ENV_FILE}")
            log.error("  OPENAI_API_KEY=sk-proj-...")
            sys.exit(1)
        log.info(f"  API key  : {key[:12]}... (loaded)")
        try:
            import openai
            client = openai.OpenAI(api_key=key)
            models = [m.id for m in client.models.list().data[:3]]
            log.info(f"  OpenAI connected. Sample models: {models}")
        except Exception as e:
            log.error(f"  OpenAI connection failed: {e}")
            sys.exit(1)

    elif LLM_PROVIDER == "anthropic":
        key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not key:
            log.error("  ANTHROPIC_API_KEY not set.")
            log.error(f"  Add ANTHROPIC_API_KEY=sk-ant-... to: {ENV_FILE}")
            sys.exit(1)
        log.info(f"  API key  : {key[:12]}... (loaded)")

    elif LLM_PROVIDER == "ollama":
        import requests
        try:
            resp      = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
            available = [m["name"] for m in resp.json().get("models", [])]
            if not any(LLM_MODEL in m for m in available):
                log.error(f"  Model '{LLM_MODEL}' not found.")
                log.error(f"  Run: ollama pull {LLM_MODEL}")
                sys.exit(1)
            log.info(f"  Ollama running. Model '{LLM_MODEL}' ready.")
        except Exception:
            log.error("  Ollama not running. Start it with: ollama serve")
            sys.exit(1)


# ── Parse cancer code from LLM response ──────────────────────────────────────
def parse_response(response):
    cleaned = response.strip().upper()
    if cleaned in VALID_CODES:
        return cleaned
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
    load_env_file(ENV_FILE)

    section(f"METHOD 6 — RAG + LLM ({LLM_PROVIDER.upper()} / {LLM_MODEL})")
    log.info(f"  Provider       : {LLM_PROVIDER}")
    log.info(f"  Model          : {LLM_MODEL}")
    log.info(f"  Sample size    : {SAMPLE_SIZE} test reports")
    log.info(f"  K examples     : {K_EXAMPLES} similar case per prompt")
    log.info(f"  Embedding model: {EMB_MODEL_NAME}")
    log.info(f"  Log file       : {_LOG_FILE}")

    # ── STEP 1 — Check provider ───────────────────────────────────────────────
    check_provider()

    # ── STEP 2 — Load Method 4 embedding cache ────────────────────────────────
    section("STEP 2 — Load embedding cache from Method 4")
    for path, label in [(TRAIN_EMB_PATH, "Train embeddings"),
                        (TEST_EMB_PATH,  "Test  embeddings"),
                        (TRAIN_META_PATH,"Train meta CSV"),
                        (TEST_META_PATH, "Test  meta CSV")]:
        if not os.path.exists(path):
            log.error(f"  MISSING: {path}")
            log.error("  Run method4_embedding_knn.py first.")
            sys.exit(1)
        log.info(f"  Found: {label}")

    train_emb_dict = load_npz(TRAIN_EMB_PATH)
    train_meta     = pd.read_csv(TRAIN_META_PATH)
    train_meta.index = train_meta["patient_id"].values
    log.info(f"  Train: {len(train_emb_dict):,} embeddings loaded")

    test_emb_dict = load_npz(TEST_EMB_PATH)
    test_meta     = pd.read_csv(TEST_META_PATH)
    test_meta.index = test_meta["patient_id"].values
    log.info(f"  Test : {len(test_emb_dict):,} embeddings loaded")

    # ── STEP 3 — Build embedding matrices ────────────────────────────────────
    section("STEP 3 — Build embedding matrices")
    train_ids    = train_meta["patient_id"].values
    train_matrix = build_matrix(train_emb_dict, train_ids)
    log.info(f"  Train matrix : {train_matrix.shape}  (n_reports × embedding_dim)")

    # ── STEP 4 — Sample test reports ─────────────────────────────────────────
    section(f"STEP 4 — Sample {SAMPLE_SIZE} test reports (seed={RANDOM_SEED})")
    rng           = np.random.default_rng(RANDOM_SEED)
    test_ids_all  = test_meta["patient_id"].values
    sample_idx    = rng.choice(len(test_ids_all), size=min(SAMPLE_SIZE, len(test_ids_all)), replace=False)
    sample_ids    = test_ids_all[sample_idx]
    sample_meta   = test_meta.loc[sample_ids]
    sample_matrix = build_matrix(test_emb_dict, sample_ids)
    log.info(f"  Sampled {len(sample_ids)} reports, {sample_meta['cancer_type'].nunique()} cancer types")

    # ── STEP 5 — RAG + LLM loop ───────────────────────────────────────────────
    section("STEP 5 — RAG retrieval + LLM classification")
    subsection("Pre-computing cosine similarities (batch)")
    sim_matrix = cosine_similarity_matrix(sample_matrix, train_matrix)
    log.info(f"  Similarity matrix: {sim_matrix.shape}  (sample × train)")
    log.info(f"  Starting LLM loop — {len(sample_ids)} reports...")
    log.info("")

    all_preds     = []
    all_labels    = []
    all_raw_resp  = []
    unknown_count = 0
    t_loop        = time.time()

    for i, pid in enumerate(sample_ids):
        true_label = sample_meta.loc[pid, "cancer_type"]
        test_text  = str(sample_meta.loc[pid, "text"])

        # Retrieve top-K similar training reports
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

        # Build prompt → call LLM → parse response
        prompt   = build_prompt(test_text, examples)
        raw_resp = call_llm(prompt)
        pred     = parse_response(raw_resp)

        if pred == "UNKNOWN":
            unknown_count += 1

        all_preds.append(pred)
        all_labels.append(true_label)
        all_raw_resp.append(raw_resp)

        # Progress update every 25 reports
        if (i + 1) % 25 == 0 or (i + 1) == len(sample_ids):
            elapsed    = time.time() - t_loop
            per_report = elapsed / (i + 1)
            remaining  = per_report * (len(sample_ids) - i - 1)
            acc_so_far = sum(p == l for p, l in zip(all_preds, all_labels)) / (i + 1) * 100
            log.info(
                f"  [{i+1:>3}/{len(sample_ids)}] "
                f"Acc={acc_so_far:.1f}%  "
                f"Unknown={unknown_count}  "
                f"ETA={remaining/60:.1f}min  "
                f"Last: true={true_label} pred={pred} raw='{raw_resp[:25]}'"
            )

    log.info(f"\n  Loop complete in {(time.time()-t_loop)/60:.1f} min")
    log.info(f"  Unparseable (UNKNOWN): {unknown_count}/{len(sample_ids)}")

    # ── STEP 6 — Evaluation ───────────────────────────────────────────────────
    section("STEP 6 — Results")

    valid_mask   = [p != "UNKNOWN" for p in all_preds]
    valid_preds  = [p for p, m in zip(all_preds,  valid_mask) if m]
    valid_labels = [l for l, m in zip(all_labels, valid_mask) if m]

    if len(valid_preds) == 0:
        log.error("  No valid predictions — check LLM output above.")
        sys.exit(1)

    acc_valid = accuracy_score(valid_labels, valid_preds) * 100
    acc_all   = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_preds) * 100
    f1_w      = f1_score(valid_labels, valid_preds, average="weighted", zero_division=0) * 100
    f1_m      = f1_score(valid_labels, valid_preds, average="macro",    zero_division=0) * 100

    total_time = (time.time() - t_start) / 60

    log.info(f"  Sample size          : {len(sample_ids)}")
    log.info(f"  Valid predictions    : {len(valid_preds)} ({len(valid_preds)/len(sample_ids)*100:.1f}%)")
    log.info(f"  Unknown (unparsed)   : {unknown_count}")
    log.info("")
    log.info(f"  +----------------------------------------------------------+")
    log.info(f"  |  METHOD 6 -- RAG + LLM ({LLM_PROVIDER} / {LLM_MODEL})")
    log.info(f"  |                                                          |")
    log.info(f"  |  Accuracy (valid only) : {acc_valid:>6.2f}%                    |")
    log.info(f"  |  Accuracy (all, strict): {acc_all:>6.2f}%                    |")
    log.info(f"  |  F1 Weighted           : {f1_w:>6.2f}%                    |")
    log.info(f"  |  F1 Macro              : {f1_m:>6.2f}%                    |")
    log.info(f"  +----------------------------------------------------------+")

    subsection("Full comparison — all methods")
    log.info(f"  {'Method':<45} {'Accuracy':>10}")
    log.info(f"  {'-'*57}")
    log.info(f"  {'Cedars-Sinai baseline (BoW + LR)':<45} {'95.31%':>10}")
    log.info(f"  {'Method 1 — TF-IDF + Logistic Regression':<45} {'96.36%':>10}")
    log.info(f"  {'Method 2 — TF-IDF + SVM':<45} {'97.37%':>10}")
    log.info(f"  {'Method 3 — Bio_ClinicalBERT fine-tuned':<45} {'94.57%':>10}")
    log.info(f"  {'Method 4 — Embedding k-NN (MiniLM, K=5)':<45} {'90.13%':>10}")
    log.info(f"  {'Method 5 — RAG + LLM (llama3.2:3b, n=50)':<45} {'64.00%':>10}")
    label6 = f"Method 6 — RAG + LLM ({LLM_PROVIDER}/{LLM_MODEL}, n={len(sample_ids)})"
    log.info(f"  {label6:<45} {acc_all:>9.2f}%")

    subsection("Per-class results (valid predictions only)")
    log.info(classification_report(valid_labels, valid_preds, zero_division=0))

    subsection("Sample of predictions (first 30)")
    log.info(f"  {'#':<4} {'True':<8} {'Pred':<8} {'Match':<6} {'Raw LLM response'}")
    log.info(f"  {'-'*65}")
    for i in range(min(30, len(all_preds))):
        match  = "OK" if all_preds[i] == all_labels[i] else "WRONG"
        raw_30 = all_raw_resp[i][:40].replace("\n", " ")
        log.info(f"  {i+1:<4} {all_labels[i]:<8} {all_preds[i]:<8} {match:<6} {raw_30}")

    section("DONE")
    log.info(f"  Total runtime : {total_time:.1f} min")
    log.info(f"  Log saved to  : {_LOG_FILE}")


