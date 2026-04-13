"""
MEDREx — Backend API
FastAPI server that:
  1. Loads TCGA data and trains TF-IDF model
  2. Stores report embeddings in ChromaDB (vector database)
  3. Exposes endpoints for the Streamlit frontend

Run: uvicorn main:app --port 8000 --reload
Docs: http://localhost:8000/docs
"""

import os
import warnings
import threading
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR      = r"c:\Users\kings\OneDrive\Desktop\CSUDH_Kingslee\CSUDH-Spring-2026\CSC-590-MastersProject\MastersProject"
REPORTS_PATH  = os.path.join(BASE_DIR, "DataSet", "TCGA_Reports.csv", "TCGA_Reports.csv")
LABELS_PATH   = os.path.join(BASE_DIR, "DataSet", "tcga_patient_to_cancer_type.csv")
MODEL_DIR     = os.path.join(BASE_DIR, "backend", "models")
VECTOR_DIR    = os.path.join(BASE_DIR, "backend", "vector_store")

# ── Cancer type names ─────────────────────────────────────────────────────────
CANCER_NAMES = {
    "ACC":  "Adrenocortical carcinoma",
    "BLCA": "Bladder Urothelial Carcinoma",
    "BRCA": "Breast invasive carcinoma",
    "CESC": "Cervical squamous cell carcinoma",
    "CHOL": "Cholangiocarcinoma",
    "COAD": "Colon adenocarcinoma",
    "DLBC": "Diffuse Large B-cell Lymphoma",
    "ESCA": "Esophageal carcinoma",
    "GBM":  "Glioblastoma multiforme",
    "HNSC": "Head and Neck squamous cell carcinoma",
    "KICH": "Kidney Chromophobe",
    "KIRC": "Kidney renal clear cell carcinoma",
    "KIRP": "Kidney renal papillary cell carcinoma",
    "LAML": "Acute Myeloid Leukemia",
    "LGG":  "Brain Lower Grade Glioma",
    "LIHC": "Liver hepatocellular carcinoma",
    "LUAD": "Lung adenocarcinoma",
    "LUSC": "Lung squamous cell carcinoma",
    "MESO": "Mesothelioma",
    "OV":   "Ovarian serous cystadenocarcinoma",
    "PAAD": "Pancreatic adenocarcinoma",
    "PCPG": "Pheochromocytoma and Paraganglioma",
    "PRAD": "Prostate adenocarcinoma",
    "READ": "Rectum adenocarcinoma",
    "SARC": "Sarcoma",
    "SKCM": "Skin Cutaneous Melanoma",
    "STAD": "Stomach adenocarcinoma",
    "TGCT": "Testicular Germ Cell Tumors",
    "THCA": "Thyroid carcinoma",
    "THYM": "Thymoma",
    "UCEC": "Uterine Corpus Endometrial Carcinoma",
    "UCS":  "Uterine Carcinosarcoma",
    "UVM":  "Uveal Melanoma",
}

# ── Shared state ──────────────────────────────────────────────────────────────
state: dict = {}


# ── Data loading ──────────────────────────────────────────────────────────────
def load_data() -> pd.DataFrame:
    df_r = pd.read_csv(REPORTS_PATH)
    df_l = pd.read_csv(LABELS_PATH)
    df_r["patient_id"] = df_r["patient_filename"].apply(lambda x: x.split(".")[0])
    df = df_r.merge(df_l, on="patient_id", how="inner")
    df["word_count"]  = df["text"].apply(lambda x: len(str(x).split()))
    df["cancer_name"] = df["cancer_type"].map(CANCER_NAMES).fillna(df["cancer_type"])
    return df


# ── TF-IDF model ──────────────────────────────────────────────────────────────
def train_or_load_model(df: pd.DataFrame):
    model_path = os.path.join(MODEL_DIR, "tfidf_lr.pkl")
    vec_path   = os.path.join(MODEL_DIR, "tfidf_vec.pkl")

    X_tr, X_te, y_tr, y_te = train_test_split(
        df["text"], df["cancer_type"],
        test_size=0.3, random_state=42, stratify=df["cancer_type"]
    )

    if os.path.exists(model_path) and os.path.exists(vec_path):
        print("[TF-IDF] Loading saved model from disk...")
        model = joblib.load(model_path)
        vec   = joblib.load(vec_path)
    else:
        print("[TF-IDF] Training model (first run ~60 sec)...")
        os.makedirs(MODEL_DIR, exist_ok=True)
        vec = TfidfVectorizer(max_features=15000, ngram_range=(1, 2),
                              sublinear_tf=True, min_df=2)
        X_tr_v = vec.fit_transform(X_tr)
        model  = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced",
                                    solver="lbfgs", random_state=42)
        model.fit(X_tr_v, y_tr)
        joblib.dump(model, model_path)
        joblib.dump(vec,   vec_path)
        print("[TF-IDF] Model saved.")

    X_te_v    = vec.transform(X_te)
    y_pred    = model.predict(X_te_v)
    acc       = accuracy_score(y_te, y_pred)
    f1        = f1_score(y_te, y_pred, average="weighted")
    classes   = sorted(y_te.unique())
    f1_scores = f1_score(y_te, y_pred, labels=classes, average=None, zero_division=0)
    per_class = {c: round(float(s), 4) for c, s in zip(classes, f1_scores)}

    print(f"[TF-IDF] Ready — Accuracy: {acc*100:.2f}%  F1: {f1*100:.2f}%")
    return model, vec, round(acc, 4), round(f1, 4), per_class


# ── ChromaDB vector store ─────────────────────────────────────────────────────
def init_vector_store():
    """
    Initialize ChromaDB with persistent storage.
    ChromaDB saves everything to disk at VECTOR_DIR.
    On restart it loads from disk instantly — no re-embedding needed.
    """
    os.makedirs(VECTOR_DIR, exist_ok=True)
    client = chromadb.PersistentClient(path=VECTOR_DIR)

    # Collection = a named group of embeddings (like a table in SQL)
    collection = client.get_or_create_collection(
        name="tcga_reports",
        metadata={"hnsw:space": "cosine"}   # use cosine similarity for search
    )
    return client, collection


def build_vector_store(df: pd.DataFrame, collection, embed_model):
    """
    Embed all 9,523 reports and store in ChromaDB.
    Only runs if the collection is empty (i.e., first time).
    Runs in a background thread so the API stays responsive.
    """
    existing = collection.count()
    if existing >= len(df):
        print(f"[ChromaDB] Already have {existing} embeddings. Skipping rebuild.")
        state["vector_ready"] = True
        state["vector_count"] = existing
        return

    print(f"[ChromaDB] Building vector store for {len(df)} reports...")
    print("[ChromaDB] This runs once and saves to disk. Future restarts are instant.")

    state["vector_ready"]    = False
    state["vector_building"] = True
    state["vector_count"]    = existing
    state["vector_total"]    = len(df)

    BATCH = 100
    rows  = df.reset_index(drop=True)

    for start in range(existing, len(rows), BATCH):
        batch = rows.iloc[start : start + BATCH]

        texts = batch["text"].apply(lambda x: str(x)[:1000]).tolist()  # first 1000 chars
        ids   = batch["patient_id"].tolist()
        metas = [
            {
                "cancer_type": r["cancer_type"],
                "cancer_name": CANCER_NAMES.get(r["cancer_type"], r["cancer_type"]),
                "word_count":  int(r["word_count"]),
            }
            for _, r in batch.iterrows()
        ]

        embeddings = embed_model.encode(texts, show_progress_bar=False).tolist()

        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metas,
        )

        state["vector_count"] = start + len(batch)
        pct = (state["vector_count"] / len(rows)) * 100
        if state["vector_count"] % 500 == 0 or state["vector_count"] >= len(rows):
            print(f"[ChromaDB] Embedded {state['vector_count']}/{len(rows)} ({pct:.0f}%)")

    state["vector_ready"]    = True
    state["vector_building"] = False
    print("[ChromaDB] Vector store complete. Saved to disk.")


# ── App lifespan ──────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("\n=== MEDREx API starting ===")

    # 1. Load data
    print("[Data] Loading TCGA dataset...")
    state["df"] = load_data()
    print(f"[Data] Loaded {len(state['df']):,} reports")

    # 2. TF-IDF model
    model, vec, acc, f1, per_class = train_or_load_model(state["df"])
    state["model"]        = model
    state["vec"]          = vec
    state["accuracy"]     = acc
    state["f1"]           = f1
    state["per_class_f1"] = per_class
    state["tfidf_ready"]  = True

    # 3. Embedding model (for ChromaDB)
    print("[Embeddings] Loading sentence transformer model...")
    # all-MiniLM-L6-v2: small (80MB), fast, good quality
    # To use medical model instead: pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    state["embed_model"] = embed_model
    print("[Embeddings] Model ready.")

    # 4. ChromaDB — init and start building in background
    client, collection = init_vector_store()
    state["chroma_client"]     = client
    state["chroma_collection"] = collection
    state["vector_count"]      = collection.count()
    state["vector_total"]      = len(state["df"])
    state["vector_ready"]      = collection.count() >= len(state["df"])
    state["vector_building"]   = False

    if not state["vector_ready"]:
        print("[ChromaDB] Starting background embedding...")
        t = threading.Thread(
            target=build_vector_store,
            args=(state["df"], collection, embed_model),
            daemon=True
        )
        t.start()
    else:
        print(f"[ChromaDB] Loaded {state['vector_count']} embeddings from disk.")

    print("=== MEDREx API ready ===\n")
    yield
    state.clear()


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="MEDREx API",
    description="Medical Evidence & Decision Reasoning eXchange — CSC 590 Masters Project",
    version="2.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


# ── Request / Response schemas ────────────────────────────────────────────────
class PredictRequest(BaseModel):
    text: str

class PredictionResult(BaseModel):
    cancer_code: str
    cancer_name: str
    confidence:  float

class SimilarReport(BaseModel):
    patient_id:   str
    cancer_type:  str
    cancer_name:  str
    similarity:   float
    text_preview: str

class PredictResponse(BaseModel):
    predicted:      PredictionResult
    top5:           List[PredictionResult]
    model_used:     str
    similar_cases:  Optional[List[SimilarReport]] = None


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
def health():
    return {
        "status":          "ok",
        "tfidf_ready":     state.get("tfidf_ready", False),
        "vector_ready":    state.get("vector_ready", False),
        "vector_building": state.get("vector_building", False),
        "vector_count":    state.get("vector_count", 0),
        "vector_total":    state.get("vector_total", 0),
        "accuracy":        state.get("accuracy"),
    }


@app.get("/stats", tags=["Data"])
def get_stats():
    df = state["df"]
    return {
        "total_reports":     int(len(df)),
        "cancer_types":      int(df["cancer_type"].nunique()),
        "avg_word_count":    round(float(df["word_count"].mean()), 1),
        "median_word_count": round(float(df["word_count"].median()), 1),
        "max_word_count":    int(df["word_count"].max()),
        "pct_over_512":      round(float((df["word_count"] > 512).mean() * 100), 1),
        "missing_values":    int(df.isnull().sum().sum()),
        "model_accuracy":    state.get("accuracy"),
        "model_f1":          state.get("f1"),
        "vector_ready":      state.get("vector_ready", False),
        "vector_count":      state.get("vector_count", 0),
        "vector_total":      state.get("vector_total", 0),
    }


@app.get("/distribution", tags=["Data"])
def get_distribution():
    df  = state["df"]
    cts = df["cancer_type"].value_counts().reset_index()
    cts.columns = ["code", "count"]
    cts["name"]  = cts["code"].map(CANCER_NAMES).fillna(cts["code"])
    cts["label"] = cts["code"] + " — " + cts["name"]
    return cts.to_dict(orient="records")


@app.get("/text-lengths", tags=["Data"])
def get_text_lengths():
    return {"word_counts": state["df"]["word_count"].tolist(), "bins": 50}


@app.get("/sample", tags=["Data"])
def get_sample(cancer_type: str = None):
    df = state["df"]
    subset = df[df["cancer_type"] == cancer_type.upper()] if cancer_type else df
    if subset.empty:
        raise HTTPException(404, f"No reports for {cancer_type}")
    row = subset.sample(1).iloc[0]
    return {
        "patient_id":   row["patient_id"],
        "cancer_type":  row["cancer_type"],
        "cancer_name":  CANCER_NAMES.get(row["cancer_type"], row["cancer_type"]),
        "word_count":   int(row["word_count"]),
        "text_preview": str(row["text"])[:600],
        "text_full":    str(row["text"]),
    }


@app.post("/predict", tags=["Model"])
def predict(req: PredictRequest) -> PredictResponse:
    """
    Predict cancer type from report text.
    If ChromaDB is ready, also returns the top-3 most similar real cases.
    """
    if not state.get("tfidf_ready"):
        raise HTTPException(503, "Model not ready yet.")
    if not req.text.strip():
        raise HTTPException(400, "Text cannot be empty.")

    # ── TF-IDF prediction ──
    vec   = state["vec"]
    model = state["model"]
    x_v   = vec.transform([req.text])
    probs = model.predict_proba(x_v)[0]
    top5i = np.argsort(probs)[::-1][:5]

    def make_result(idx):
        code = model.classes_[idx]
        return PredictionResult(
            cancer_code=code,
            cancer_name=CANCER_NAMES.get(code, code),
            confidence=round(float(probs[idx]), 4),
        )

    # ── ChromaDB similar cases (only when fully ready, not while building) ──
    # Skipped during build: background thread and encode() compete for CPU → timeouts
    similar = None
    if state.get("vector_ready") and not state.get("vector_building"):
        try:
            embed_model = state["embed_model"]
            collection  = state["chroma_collection"]
            query_vec   = embed_model.encode([req.text[:1000]]).tolist()

            results = collection.query(
                query_embeddings=query_vec,
                n_results=3,
                include=["documents", "metadatas", "distances"],
            )

            similar = []
            for i in range(len(results["ids"][0])):
                meta = results["metadatas"][0][i]
                dist = results["distances"][0][i]
                # ChromaDB cosine distance: 0 = identical, 2 = opposite
                # Convert to similarity score 0–1
                similarity = round(1 - (dist / 2), 3)
                similar.append(SimilarReport(
                    patient_id=results["ids"][0][i],
                    cancer_type=meta["cancer_type"],
                    cancer_name=meta["cancer_name"],
                    similarity=similarity,
                    text_preview=results["documents"][0][i][:300],
                ))
        except Exception as e:
            print(f"[ChromaDB] Similar search failed: {e}")
            similar = None

    return PredictResponse(
        predicted=make_result(top5i[0]),
        top5=[make_result(i) for i in top5i],
        model_used="TF-IDF + Logistic Regression",
        similar_cases=similar,
    )


@app.get("/similar", tags=["Vector Store"])
def find_similar(text: str, n: int = 5):
    """
    Find top-n most similar reports using ChromaDB semantic search.
    Returns similar reports with cancer types — this IS the embedding similarity method.
    """
    if not state.get("vector_ready"):
        raise HTTPException(503, f"Vector store not ready yet. Embedded {state.get('vector_count',0)}/{state.get('vector_total',0)} so far.")

    embed_model = state["embed_model"]
    collection  = state["chroma_collection"]
    query_vec   = embed_model.encode([text[:1000]]).tolist()

    results = collection.query(
        query_embeddings=query_vec,
        n_results=min(n, 10),
        include=["documents", "metadatas", "distances"],
    )

    similar = []
    for i in range(len(results["ids"][0])):
        meta = results["metadatas"][0][i]
        dist = results["distances"][0][i]
        similar.append({
            "patient_id":   results["ids"][0][i],
            "cancer_type":  meta["cancer_type"],
            "cancer_name":  meta["cancer_name"],
            "similarity":   round(1 - (dist / 2), 3),
            "text_preview": results["documents"][0][i][:300],
        })

    # Majority vote prediction from similar cases
    if similar:
        from collections import Counter
        votes   = Counter(s["cancer_type"] for s in similar)
        top_vote = votes.most_common(1)[0]
        prediction = {
            "cancer_code": top_vote[0],
            "cancer_name": CANCER_NAMES.get(top_vote[0], top_vote[0]),
            "votes":       top_vote[1],
            "total":       len(similar),
        }
    else:
        prediction = None

    return {
        "similar_cases": similar,
        "prediction_by_majority_vote": prediction,
        "method": "Embedding Similarity (cosine) via ChromaDB",
    }


@app.get("/vector-store/status", tags=["Vector Store"])
def vector_store_status():
    """How many reports are embedded and stored in ChromaDB."""
    count = state.get("vector_count", 0)
    total = state.get("vector_total", 0)
    pct   = round((count / total * 100), 1) if total > 0 else 0
    return {
        "ready":       state.get("vector_ready", False),
        "building":    state.get("vector_building", False),
        "embedded":    count,
        "total":       total,
        "percent":     pct,
        "storage_path": VECTOR_DIR,
        "model_used":  "all-MiniLM-L6-v2 (384-dim embeddings)",
    }


@app.get("/methods", tags=["Model"])
def get_methods():
    acc = state.get("accuracy", 0)
    return [
        {"rank":1, "name":"TF-IDF + Logistic Regression", "category":"Traditional ML",   "status":"done",    "accuracy":round(acc*100,2), "needs_gpu":False, "explainable":True,      "industry":"FDA adverse events, hospital triage"},
        {"rank":2, "name":"TF-IDF + SVM",                 "category":"Traditional ML",   "status":"next",    "accuracy":None,             "needs_gpu":False, "explainable":True,      "industry":"Legal document classification"},
        {"rank":3, "name":"BERT (Bio_ClinicalBERT)",       "category":"Transformer",      "status":"planned", "accuracy":None,             "needs_gpu":True,  "explainable":"partial", "industry":"Epic Systems (77% US hospitals)"},
        {"rank":4, "name":"Embedding Similarity (k-NN)",  "category":"Semantic Search",  "status":"active",  "accuracy":None,             "needs_gpu":False, "explainable":True,      "industry":"PubMed search, Mayo Clinic"},
        {"rank":5, "name":"RAG + LLM (MEDREx Core)",       "category":"LLM + Retrieval",  "status":"planned", "accuracy":None,             "needs_gpu":True,  "explainable":"partial", "industry":"Suki AI, Azure Health Bot, Epic"},
    ]


@app.get("/class-f1", tags=["Model"])
def get_class_f1():
    per_class = state.get("per_class_f1", {})
    return [
        {"cancer_code":code, "cancer_name":CANCER_NAMES.get(code,code), "f1":score}
        for code, score in sorted(per_class.items(), key=lambda x: x[1])
    ]
