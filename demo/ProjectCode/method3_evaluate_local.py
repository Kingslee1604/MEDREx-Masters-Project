"""
MEDREx — Method 3: Bio_ClinicalBERT — Local Evaluation
=======================================================
Run this AFTER you have:
  1. Fine-tuned the model on Google Colab
  2. Downloaded the model folder from Google Drive
  3. Placed it at: demo/ProjectCode/method3_best_model/

Run:
  python demo/ProjectCode/method3_evaluate_local.py
"""

import os, sys, logging, warnings, time, json
warnings.filterwarnings("ignore")

_LOG_DIR  = os.path.join(
    r"c:\Users\kings\OneDrive\Desktop\CSUDH_Kingslee\CSUDH-Spring-2026\CSC-590-MastersProject\MastersProject",
    "demo", "ProjectCode", "output"
)
os.makedirs(_LOG_DIR, exist_ok=True)
_LOG_FILE = os.path.join(_LOG_DIR, "method3_bert_results.log")

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

BASE_DIR      = r"c:\Users\kings\OneDrive\Desktop\CSUDH_Kingslee\CSUDH-Spring-2026\CSC-590-MastersProject\MastersProject"
REPORTS_PATH  = os.path.join(BASE_DIR, "DataSet", "TCGA_Reports.csv", "TCGA_Reports.csv")
LABELS_PATH   = os.path.join(BASE_DIR, "DataSet", "tcga_patient_to_cancer_type.csv")
MODEL_DIR     = os.path.join(BASE_DIR, "demo", "ProjectCode", "method3_best_model")

if not os.path.exists(MODEL_DIR):
    print(f"ERROR: Model folder not found at {MODEL_DIR}")
    print("Please download the trained model from Google Drive first.")
    print("Expected folder structure:")
    print(f"  {MODEL_DIR}/")
    print(f"    config.json")
    print(f"    pytorch_model.bin")
    print(f"    tokenizer_config.json")
    print(f"    vocab.txt")
    print(f"    label_map.json")
    sys.exit(1)

if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    import torch
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, f1_score, classification_report
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from torch.utils.data import Dataset, DataLoader

    t_start = time.time()

    section("METHOD 3 — Bio_ClinicalBERT Fine-Tuned — Local Evaluation")
    log.info(f"  Model directory : {MODEL_DIR}")
    log.info(f"  Log file        : {_LOG_FILE}")

    # ── Load label map ─────────────────────────────────────────────────────
    subsection("Loading label map")
    with open(os.path.join(MODEL_DIR, "label_map.json")) as f:
        label_map = json.load(f)
    label_map = {int(k): v for k, v in label_map.items()}
    log.info(f"  {len(label_map)} cancer type labels loaded")
    for i, name in sorted(label_map.items())[:5]:
        log.info(f"    {i} → {name}")
    log.info(f"    ...")

    # ── Load dataset ───────────────────────────────────────────────────────
    subsection("Loading dataset")
    df_reports = pd.read_csv(REPORTS_PATH)
    df_reports["patient_id"] = df_reports["patient_filename"].apply(lambda x: x.split(".")[0])
    df_reports = df_reports[["patient_id", "text"]]
    df_reports.index = df_reports["patient_id"].values

    df_labels = pd.read_csv(LABELS_PATH)
    df_labels.index = df_labels["patient_id"].values
    df_reports["cancer_type"] = df_labels.loc[df_reports.index, "cancer_type"]
    df_reports = df_reports.dropna(subset=["cancer_type"])

    reverse_label_map = {v: k for k, v in label_map.items()}
    df_reports["label"] = df_reports["cancer_type"].map(reverse_label_map)
    log.info(f"  Dataset shape   : {df_reports.shape}")

    # ── Same split as all other methods ───────────────────────────────────
    subsection("Stratified split (same as Methods 1, 2, 4)")
    _, X_test, _, y_test = train_test_split(
        df_reports["text"],
        df_reports["label"],
        test_size=0.3,
        random_state=42,
        stratify=df_reports["label"]
    )
    log.info(f"  Test set : {len(X_test):,} reports")

    # ── Tokenize ──────────────────────────────────────────────────────────
    subsection("Tokenizing test reports")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    test_encodings = tokenizer(
        X_test.tolist(),
        truncation=True,
        padding=True,
        max_length=512,
        return_tensors="pt"
    )
    log.info(f"  Tokenized shape : {test_encodings['input_ids'].shape}")

    # ── Dataset & DataLoader ──────────────────────────────────────────────
    class CancerDataset(Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels    = torch.tensor(labels.values, dtype=torch.long)
        def __len__(self):
            return len(self.labels)
        def __getitem__(self, idx):
            item = {k: v[idx] for k, v in self.encodings.items()}
            item["labels"] = self.labels[idx]
            return item

    test_loader = DataLoader(CancerDataset(test_encodings, y_test),
                             batch_size=16, shuffle=False)

    # ── Load model ────────────────────────────────────────────────────────
    subsection("Loading fine-tuned model")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"  Device : {device}  (CPU inference is fine — no training here)")
    model  = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model  = model.to(device)
    model.eval()
    log.info("  Model loaded successfully.")

    # ── Run predictions ───────────────────────────────────────────────────
    subsection("Running predictions on test set")
    log.info(f"  {len(test_loader)} batches × 16 reports each")
    log.info("  This runs on CPU — may take 5-10 min...")

    all_preds  = []
    all_labels = []
    t0 = time.time()

    try:
        with torch.no_grad():
            for i, batch in enumerate(test_loader, 1):
                input_ids      = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels         = batch["labels"]
                outputs        = model(input_ids=input_ids,
                                       attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.numpy())
                if i % 10 == 0:
                    log.info(f"  Batch {i}/{len(test_loader)} done...")
    except Exception as e:
        import traceback
        log.error(f"  CRASH in inference loop: {e}")
        log.error(traceback.format_exc())
        sys.exit(1)

    log.info(f"  Inference complete in {(time.time()-t0)/60:.1f} min")

    # ── Evaluation ────────────────────────────────────────────────────────
    pred_names  = [label_map[p] for p in all_preds]
    label_names = [label_map[l] for l in all_labels]

    final_acc  = accuracy_score(all_labels, all_preds)
    final_f1_w = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    final_f1_m = f1_score(all_labels, all_preds, average="macro",    zero_division=0)

    log.info(f"  ╔══════════════════════════════════════════════════════════╗")
    log.info(f"  ║  METHOD 3 — Bio_ClinicalBERT FINE-TUNED RESULTS        ║")
    log.info(f"  ║                                                          ║")
    log.info(f"  ║  Test Accuracy  : {final_acc*100:>6.2f}%                        ║")
    log.info(f"  ║  F1 Weighted    : {final_f1_w*100:>6.2f}%                        ║")
    log.info(f"  ║  F1 Macro       : {final_f1_m*100:>6.2f}%                        ║")
    log.info(f"  ╚══════════════════════════════════════════════════════════╝")

    subsection("Full comparison — all methods")
    log.info(f"  {'Method':<35} {'Accuracy':>10}")
    log.info(f"  {'-'*48}")
    log.info(f"  {'Cedars-Sinai (BoW + LR)':<35} {'95.31%':>10}")
    log.info(f"  {'Method 1 (TF-IDF + LR)':<35} {'96.36%':>10}")
    log.info(f"  {'Method 2 (TF-IDF + SVM)':<35} {'~96.0%':>10}")
    log.info(f"  {'Method 4 (Embedding k-NN)':<35} {'90.13%':>10}")
    log.info(f"  {'Method 3 (BERT fine-tuned)':<35} {final_acc*100:>9.2f}%")

    log.info("\n  Full classification report:")
    log.info(classification_report(label_names, pred_names, zero_division=0))

    section("DONE")
    log.info(f"  Total runtime : {(time.time()-t_start)/60:.1f} min")
    log.info(f"  Log saved to  : {_LOG_FILE}")
