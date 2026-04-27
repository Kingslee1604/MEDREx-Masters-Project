00:27:36  [INFO]    Log file : c:\Users\kings\OneDrive\Desktop\CSUDH_Kingslee\CSUDH-Spring-2026\CSC-590-MastersProject\MastersProject\demo\ProjectCode\output\method4_embedding_knn_run.log
00:27:36  [INFO]  ======================================================================
00:27:36  [INFO]    MEDREx — Method 4: Embedding Similarity + cosine k-NN
00:27:36  [INFO]    CSC-590 Masters Project, CSUDH Spring 2026
00:27:36  [INFO]    Goal: Does semantic meaning beat word counting for cancer classification?
00:27:36  [INFO]    Embedding model: sentence-transformers/all-MiniLM-L6-v2
00:27:36  [INFO]  ======================================================================
00:27:36  [INFO]  ======================================================================
00:27:36  [INFO]    STEP 1 — Load & Compile Dataset
00:27:36  [INFO]  ======================================================================
00:27:36  [INFO]    Reading reports : c:\Users\kings\OneDrive\Desktop\CSUDH_Kingslee\CSUDH-Spring-2026\CSC-590-MastersProject\MastersProject\DataSet\TCGA_Reports.csv\TCGA_Reports.csv
00:27:37  [INFO]    Loaded 9,523 reports in 0.5s
00:27:37  [INFO]    Merged shape    : (9523, 4)
00:27:37  [INFO]    Cancer types    : 32 unique types
00:27:37  [INFO]  
                patient_id                                                                                 text cancer_type                   cancer_type_name
TCGA-BP-5195  TCGA-BP-5195  Date of Recelpt: Clinical Diagnosis & History: Incidental 3 cm left upper pole r...        KIRC  Kidney renal clear cell carcinoma
TCGA-D7-8573  TCGA-D7-8573  Material: 1) Material: stomach, Method of collection: Lesion resection. Histopat...        STAD             Stomach adenocarcinoma
TCGA-EI-7004  TCGA-EI-7004  page 1 / 1. copy No. 3. Examination: Histopathological examination. Cost of diag...        READ              Rectum adenocarcinoma
00:27:37  [INFO]  ======================================================================
00:27:37  [INFO]    STEP 2 — Stratified Train / Test Split  (70% / 30%)
00:27:37  [INFO]  ======================================================================
00:27:40  [INFO]    Same split parameters as Methods 1 & 2 (random_state=42, stratify=cancer_type).
00:27:40  [INFO]    Identical split = fair comparison across all methods.
00:27:40  [INFO]    Train : 6,666 reports
00:27:40  [INFO]    Test  : 2,857  reports
00:27:40  [INFO]  ======================================================================
00:27:40  [INFO]    STEP 3 — Generate Embeddings  (convert text → vectors)
00:27:40  [INFO]  ======================================================================
00:27:40  [INFO]    WHAT IS AN EMBEDDING?
00:27:40  [INFO]    A pre-trained language model reads a pathology report and outputs
00:27:40  [INFO]    a fixed-size list of numbers (a vector) that captures the meaning.
00:27:40  [INFO]    Model: sentence-transformers/all-MiniLM-L6-v2
00:27:40  [INFO]    Output: 384 numbers per report  (the embedding dimension)
00:27:40  [INFO]    Key property: similar reports → similar vectors (small angle between them)
00:27:40  [INFO]    Example:
00:27:40  [INFO]      'clear cell renal carcinoma, right kidney...' → [0.23, -0.81, 0.44, ...]
00:27:40  [INFO]      'renal cell carcinoma with clear cell features...' → [0.24, -0.79, 0.46, ...]
00:27:40  [INFO]      → These two vectors are nearly identical (same cancer, different words)
00:27:40  [INFO]    Contrast with TF-IDF:
00:27:40  [INFO]      'kidney' and 'renal' = completely different TF-IDF features
00:27:40  [INFO]      'kidney' and 'renal' = very close in embedding space (model knows they mean the same thing)
00:27:40  [INFO]  
00:27:40  [INFO]    BERT 512-token limit:
00:27:40  [INFO]    BERT can only read 512 tokens (~380 words) at a time.
00:27:40  [INFO]    For longer reports, the model uses the first 512 tokens.
00:27:40  [INFO]    This is a known limitation — TF-IDF has NO length limit.
00:27:40  [INFO]    Reports with key diagnosis at the END may lose information.
00:27:40  [INFO]  --- Loading sentence-transformer model ---
00:27:40  [INFO]    Model: sentence-transformers/all-MiniLM-L6-v2
00:27:40  [INFO]    First run: model will download (~80MB). Subsequent runs: loaded from cache.
00:27:51  [INFO]  No device provided, using cpu
00:27:52  [INFO]  HTTP Request: HEAD https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/modules.json "HTTP/1.1 307 Temporary Redirect"
00:27:52  [WARNING]  Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
00:27:52  [INFO]  HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/sentence-transformers/all-MiniLM-L6-v2/c9745ed1d9f207416be6d2e6f8de32d1f16199bf/modules.json "HTTP/1.1 200 OK"
00:27:52  [INFO]  HTTP Request: HEAD https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/config_sentence_transformers.json "HTTP/1.1 307 Temporary Redirect"
00:27:52  [INFO]  HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/sentence-transformers/all-MiniLM-L6-v2/c9745ed1d9f207416be6d2e6f8de32d1f16199bf/config_sentence_transformers.json "HTTP/1.1 200 OK"
00:27:52  [INFO]  Loading SentenceTransformer model from sentence-transformers/all-MiniLM-L6-v2.
00:27:52  [INFO]  HTTP Request: HEAD https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/config_sentence_transformers.json "HTTP/1.1 307 Temporary Redirect"
00:27:52  [INFO]  HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/sentence-transformers/all-MiniLM-L6-v2/c9745ed1d9f207416be6d2e6f8de32d1f16199bf/config_sentence_transformers.json "HTTP/1.1 200 OK"
00:27:52  [INFO]  HTTP Request: HEAD https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/README.md "HTTP/1.1 307 Temporary Redirect"
00:27:52  [INFO]  HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/sentence-transformers/all-MiniLM-L6-v2/c9745ed1d9f207416be6d2e6f8de32d1f16199bf/README.md "HTTP/1.1 200 OK"
00:27:52  [INFO]  HTTP Request: HEAD https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/modules.json "HTTP/1.1 307 Temporary Redirect"
00:27:52  [INFO]  HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/sentence-transformers/all-MiniLM-L6-v2/c9745ed1d9f207416be6d2e6f8de32d1f16199bf/modules.json "HTTP/1.1 200 OK"
00:27:52  [INFO]  HTTP Request: HEAD https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/sentence_bert_config.json "HTTP/1.1 307 Temporary Redirect"
00:27:52  [INFO]  HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/sentence-transformers/all-MiniLM-L6-v2/c9745ed1d9f207416be6d2e6f8de32d1f16199bf/sentence_bert_config.json "HTTP/1.1 200 OK"
00:27:53  [INFO]  HTTP Request: HEAD https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/adapter_config.json "HTTP/1.1 404 Not Found"
00:27:53  [INFO]  HTTP Request: HEAD https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/config.json "HTTP/1.1 307 Temporary Redirect"
00:27:53  [INFO]  HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/sentence-transformers/all-MiniLM-L6-v2/c9745ed1d9f207416be6d2e6f8de32d1f16199bf/config.json "HTTP/1.1 200 OK"
00:27:53  [INFO]  HTTP Request: HEAD https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/processor_config.json "HTTP/1.1 404 Not Found"
00:27:53  [INFO]  HTTP Request: HEAD https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/preprocessor_config.json "HTTP/1.1 404 Not Found"
00:27:53  [INFO]  HTTP Request: HEAD https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/video_preprocessor_config.json "HTTP/1.1 404 Not Found"
00:27:53  [INFO]  HTTP Request: HEAD https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/preprocessor_config.json "HTTP/1.1 404 Not Found"
00:27:53  [INFO]  HTTP Request: HEAD https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/tokenizer_config.json "HTTP/1.1 307 Temporary Redirect"
00:27:53  [INFO]  HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/sentence-transformers/all-MiniLM-L6-v2/c9745ed1d9f207416be6d2e6f8de32d1f16199bf/tokenizer_config.json "HTTP/1.1 200 OK"
00:27:54  [INFO]  HTTP Request: HEAD https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/config.json "HTTP/1.1 307 Temporary Redirect"
00:27:54  [INFO]  HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/sentence-transformers/all-MiniLM-L6-v2/c9745ed1d9f207416be6d2e6f8de32d1f16199bf/config.json "HTTP/1.1 200 OK"
00:27:54  [INFO]  HTTP Request: HEAD https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/config.json "HTTP/1.1 307 Temporary Redirect"
00:27:54  [INFO]  HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/sentence-transformers/all-MiniLM-L6-v2/c9745ed1d9f207416be6d2e6f8de32d1f16199bf/config.json "HTTP/1.1 200 OK"
00:27:54  [INFO]  HTTP Request: HEAD https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/tokenizer_config.json "HTTP/1.1 307 Temporary Redirect"
00:27:54  [INFO]  HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/sentence-transformers/all-MiniLM-L6-v2/c9745ed1d9f207416be6d2e6f8de32d1f16199bf/tokenizer_config.json "HTTP/1.1 200 OK"
00:27:54  [INFO]  HTTP Request: GET https://huggingface.co/api/models/sentence-transformers/all-MiniLM-L6-v2/tree/main/additional_chat_templates?recursive=false&expand=false "HTTP/1.1 404 Not Found"
00:27:54  [INFO]  HTTP Request: GET https://huggingface.co/api/models/sentence-transformers/all-MiniLM-L6-v2/tree/main?recursive=true&expand=false "HTTP/1.1 200 OK"
00:27:54  [INFO]  HTTP Request: HEAD https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/1_Pooling/config.json "HTTP/1.1 307 Temporary Redirect"
00:27:54  [INFO]  HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/sentence-transformers/all-MiniLM-L6-v2/c9745ed1d9f207416be6d2e6f8de32d1f16199bf/1_Pooling%2Fconfig.json "HTTP/1.1 200 OK"
00:27:54  [INFO]  HTTP Request: GET https://huggingface.co/api/models/sentence-transformers/all-MiniLM-L6-v2 "HTTP/1.1 200 OK"
00:27:54  [INFO]    Model loaded in 3.2s
00:27:54  [INFO]    Embedding dimension : 384
00:27:54  [INFO]  --- Embedding training reports ---
00:27:54  [INFO]    6,666 reports × batch_size=64
00:27:54  [INFO]    Progress bar will appear below. This takes ~10-15 min on CPU.
00:32:07  [INFO]    Training embeddings done in 4.2 min
00:32:07  [INFO]    Shape : (6666, 384)  (6666 reports × 384 dims)
00:32:07  [INFO]  --- Embedding test reports ---
00:32:07  [INFO]    2,857 reports
00:33:54  [INFO]    Test embeddings done in 1.8 min
00:33:54  [INFO]    Shape : (2857, 384)
00:33:54  [INFO]  --- Saving embeddings to disk ---
00:33:54  [INFO]    WHAT ARE WE SAVING?
00:33:54  [INFO]    method4_train_embeddings.npy
00:33:54  [INFO]      → numpy array  shape=(6666, 384)
00:33:54  [INFO]      → Each row = one pathology report = 384 float32 numbers
00:33:54  [INFO]      → File size ~ 10.2 MB
00:33:54  [INFO]    method4_test_embeddings.npy
00:33:54  [INFO]      → numpy array  shape=(2857, 384)
00:33:54  [INFO]      → File size ~ 4.4 MB
00:33:54  [INFO]    method4_train_meta.csv
00:33:54  [INFO]      → CSV with patient_id + cancer_type for each training row
00:33:54  [INFO]      → Row i in this CSV matches row i in train_embeddings.npy
00:33:54  [INFO]    method4_test_meta.csv
00:33:54  [INFO]      → CSV with patient_id + cancer_type for each test row
00:33:54  [INFO]    All 4 files saved to: c:\Users\kings\OneDrive\Desktop\CSUDH_Kingslee\CSUDH-Spring-2026\CSC-590-MastersProject\MastersProject\demo\ProjectCode\output
00:33:54  [INFO]    Next run will load from cache — no re-embedding needed.
00:33:54  [INFO]  ======================================================================
00:33:54  [INFO]    STEP 4 — cosine k-NN Classification
00:33:54  [INFO]  ======================================================================
00:33:55  [INFO]    HOW DOES cosine k-NN WORK?
00:33:55  [INFO]    For each test report embedding:
00:33:55  [INFO]      1. Compute cosine similarity to ALL 6,666 training embeddings
00:33:55  [INFO]      2. Pick the K most similar (nearest neighbors)
00:33:55  [INFO]      3. Majority vote of their cancer types = prediction
00:33:55  [INFO]  
00:33:55  [INFO]    WHY COSINE SIMILARITY (not Euclidean distance)?
00:33:55  [INFO]    Cosine similarity measures the ANGLE between vectors, not their length.
00:33:55  [INFO]    Two reports with the same content but different lengths → same angle.
00:33:55  [INFO]    Euclidean distance would penalise longer reports unfairly.
00:33:55  [INFO]    For text embeddings, cosine is always preferred.
00:33:55  [INFO]  
00:33:55  [INFO]    COSINE SIMILARITY FORMULA:
00:33:55  [INFO]    similarity = (A · B) / (|A| × |B|)
00:33:55  [INFO]    Since we normalized embeddings at encode time: similarity = A · B (dot product)
00:33:55  [INFO]    Range: -1 (opposite) to +1 (identical)
00:33:55  [INFO]    In practice for same-domain reports: 0.7 = somewhat similar, 0.95 = very similar
00:33:55  [INFO]  --- Testing different K values to find the best ---
00:33:55  [INFO]    K=1  → predict using only the single closest training report
00:33:55  [INFO]    K=5  → majority vote among 5 closest → more robust, less noise-sensitive
00:33:55  [INFO]    K=11 → larger vote pool → better for rare cancer types
00:33:55  [INFO]    We test K = [1, 3, 5, 7, 11] and pick the winner.
00:33:55  [INFO]    K=1    Accuracy=89.99%  F1-weighted=89.95%  ████████████████████████████████████████████
00:33:56  [INFO]    K=3    Accuracy=89.92%  F1-weighted=89.78%  ████████████████████████████████████████████
00:33:56  [INFO]    K=5    Accuracy=90.13%  F1-weighted=89.93%  █████████████████████████████████████████████
00:33:56  [INFO]    K=7    Accuracy=89.88%  F1-weighted=89.58%  ████████████████████████████████████████████
00:33:57  [INFO]    K=11   Accuracy=89.60%  F1-weighted=89.23%  ████████████████████████████████████████████
00:33:57  [INFO]  
  Best K = 5  →  Accuracy = 90.13%
00:33:57  [INFO]  --- Full evaluation with best K=5 ---
00:33:57  [INFO]    ╔══════════════════════════════════════════════════════════╗
00:33:57  [INFO]    ║  METHOD 4 — EMBEDDING SIMILARITY (cosine k-NN) RESULTS ║
00:33:57  [INFO]    ║                                                          ║
00:33:57  [INFO]    ║  Best K               : 5                             ║
00:33:57  [INFO]    ║  Test Accuracy        :  90.13%                      ║
00:33:57  [INFO]    ║  F1 Weighted          :  89.93%                      ║
00:33:57  [INFO]    ║  F1 Macro             :  87.78%                      ║
00:33:57  [INFO]    ║                                                          ║
00:33:57  [INFO]    ║  Cedars-Sinai BoW+LR  : ~95.31%                         ║
00:33:57  [INFO]    ╚══════════════════════════════════════════════════════════╝
00:33:57  [INFO]  
  Full classification report (all 33 cancer types):
00:33:57  [INFO]                precision    recall  f1-score   support

         ACC       1.00      0.74      0.85        27
        BLCA       0.98      0.97      0.98       114
        BRCA       0.98      1.00      0.99       310
        CESC       0.95      0.85      0.90        87
        CHOL       0.79      0.85      0.81        13
        COAD       0.83      0.89      0.86       125
        DLBC       1.00      0.71      0.83        14
        ESCA       0.91      0.91      0.91        44
         GBM       0.85      0.97      0.91       120
        HNSC       0.99      0.97      0.98       156
        KICH       0.78      0.53      0.63        34
        KIRC       0.76      0.97      0.85       157
        KIRP       0.94      0.61      0.74        84
         LGG       0.96      0.86      0.91       141
        LIHC       0.94      0.99      0.97       102
        LUAD       0.74      0.78      0.76       146
        LUSC       0.73      0.74      0.73       140
        MESO       1.00      0.88      0.93        24
          OV       0.89      0.92      0.91       111
        PAAD       0.96      0.91      0.93        53
        PCPG       0.95      1.00      0.97        52
        PRAD       0.98      1.00      0.99       134
        READ       0.80      0.65      0.72        49
        SARC       1.00      0.85      0.92        75
        SKCM       0.94      0.94      0.94        31
        STAD       0.93      0.92      0.93       108
        TGCT       1.00      1.00      1.00        26
        THCA       0.97      0.99      0.98       146
        THYM       1.00      0.91      0.95        34
        UCEC       0.83      0.94      0.88       164
         UCS       0.83      0.29      0.43        17
         UVM       1.00      1.00      1.00        19

    accuracy                           0.90      2857
   macro avg       0.91      0.86      0.88      2857
weighted avg       0.91      0.90      0.90      2857

00:33:57  [INFO]  --- Per-class F1 sorted worst → best ---
00:33:57  [INFO]    Code   Cancer Name                                       F1  #Train  #Test
00:33:57  [INFO]    --------------------------------------------------------------------------
00:33:57  [INFO]    UCS    Uterine Carcinosarcoma                         0.435      39     17 ← hardest
00:33:57  [INFO]    KICH   Kidney Chromophobe                             0.632      78     34 ← hardest
00:33:57  [INFO]    READ   Rectum adenocarcinoma                          0.719     113     49 ← hardest
00:33:57  [INFO]    LUSC   Lung squamous cell carcinoma                   0.730     328    140 ← hardest
00:33:57  [INFO]    KIRP   Kidney renal papillary cell carcinoma          0.739     196     84 ← hardest
00:33:57  [INFO]    LUAD   Lung adenocarcinoma                            0.757     342    146 ← hardest
00:33:57  [INFO]    CHOL   Cholangiocarcinoma                             0.815      30     13 ← hardest
00:33:57  [INFO]    DLBC   Diffuse Large B-cell Lymphoma                  0.833      33     14 ← hardest
00:33:57  [INFO]    ACC    Adrenocortical carcinoma                       0.851      63     27
00:33:57  [INFO]    KIRC   Kidney renal clear cell carcinoma              0.854     368    157
00:33:57  [INFO]    COAD   Colon adenocarcinoma                           0.860     293    125
00:33:57  [INFO]    UCEC   Uterine Corpus Endometrial Carcinoma           0.883     382    164
00:33:57  [INFO]    CESC   Cervical squamous cell carcinoma               0.897     202     87
00:33:57  [INFO]    GBM    Glioblastoma multiforme                        0.906     279    120
00:33:57  [INFO]    LGG    Brain Lower Grade Glioma                       0.906     328    141
00:33:57  [INFO]    OV     Ovarian serous cystadenocarcinoma              0.907     260    111
00:33:57  [INFO]    ESCA   Esophageal carcinoma                           0.909     102     44
00:33:57  [INFO]    SARC   Sarcoma                                        0.921     174     75
00:33:57  [INFO]    STAD   Stomach adenocarcinoma                         0.925     253    108
00:33:57  [INFO]    PAAD   Pancreatic adenocarcinoma                      0.932     123     53
00:33:57  [INFO]    MESO   Mesothelioma                                   0.933      55     24
00:33:57  [INFO]    SKCM   Skin Cutaneous Melanoma                        0.935      71     31
00:33:57  [INFO]    THYM   Thymoma                                        0.954      80     34
00:33:57  [INFO]    LIHC   Liver hepatocellular carcinoma                 0.967     239    102
00:33:57  [INFO]    PCPG   Pheochromocytoma and Paraganglioma             0.972     122     52
00:33:57  [INFO]    HNSC   Head and Neck squamous cell carcinoma          0.977     364    156
00:33:57  [INFO]    BLCA   Bladder Urothelial Carcinoma                   0.978     265    114
00:33:57  [INFO]    THCA   Thyroid carcinoma                              0.983     341    146
00:33:57  [INFO]    PRAD   Prostate adenocarcinoma                        0.989     312    134
00:33:57  [INFO]    BRCA   Breast invasive carcinoma                      0.990     724    310
00:33:57  [INFO]    TGCT   Testicular Germ Cell Tumors                    1.000      61     26
00:33:57  [INFO]    UVM    Uveal Melanoma                                 1.000      46     19
00:33:57  [INFO]  ======================================================================
00:33:57  [INFO]    STEP 5 — Nearest Neighbor Inspection  (what does similarity look like?)
00:33:57  [INFO]  ======================================================================
00:33:57  [INFO]    This step shows REAL examples of what k-NN is doing.
00:33:57  [INFO]    For 5 test reports, we show their top 3 nearest training neighbors.
00:33:57  [INFO]    This makes the method transparent and explainable.
00:33:57  [INFO]  
  ┌─────────────────────────────────────────────────────────────┐
00:33:57  [INFO]    │ TEST REPORT  patient=TCGA-A2-A0SX  true_label=BRCA
00:33:57  [INFO]    │ Text preview: Specimen #: Race : WHITE. SPECIMEN: A: LEFT BREAST LUMPECTOMY B: LEFT AXILLARY CONTENTS. C: SENTINEL NODE #1. FINAL DIAG...
00:33:57  [INFO]    ├─────────────────────────────────────────────────────────────┤
00:33:57  [INFO]    │ TOP 3 NEAREST TRAINING NEIGHBORS:
00:33:57  [INFO]    │  #1  patient=TCGA-A2-A0ET  label=BRCA  sim=0.8859  ✓ MATCH
00:33:57  [INFO]    │      Text: Specimen #: Race: WHITE. Physician (s) : SPECIMEN: A: RIGHT BREAST LUMPECTOMY. B: RIGHT AXILLARY NOD...
00:33:57  [INFO]    │  #2  patient=TCGA-A2-A0EY  label=BRCA  sim=0.8673  ✓ MATCH
00:33:57  [INFO]    │      Text: Specimen #: : WHITE. SPECIMEN: A: RIGHT BREAST LUMPECTOMY B: ADDITIONAL INFERIOR MARGIN. C: ADDITION...
00:33:57  [INFO]    │  #3  patient=TCGA-A2-A0EV  label=BRCA  sim=0.8649  ✓ MATCH
00:33:57  [INFO]    │      Text: 1. M. Specimen # : Race: WHITE. Locatio. SPECIMEN: A: LEFT BREAST LUMPECTOMY B: SENTINEL LYMPH NODE ...
00:33:57  [INFO]    └─────────────────────────────────────────────────────────────┘
00:33:57  [INFO]  
  ┌─────────────────────────────────────────────────────────────┐
00:33:57  [INFO]    │ TEST REPORT  patient=TCGA-06-6390  true_label=GBM
00:33:57  [INFO]    │ Text preview: CLINICAL HISTORY. Brain tumor. OPERATIVE DIAGNOSES. Operation/Specimen A: Right frontal tumor, resection. B: Corpus coll...
00:33:57  [INFO]    ├─────────────────────────────────────────────────────────────┤
00:33:57  [INFO]    │ TOP 3 NEAREST TRAINING NEIGHBORS:
00:33:57  [INFO]    │  #1  patient=TCGA-16-1047  label=GBM  sim=0.7946  ✓ MATCH
00:33:57  [INFO]    │      Text: (Age: Visit # : Resulted : Gender: F. Facility: SPECIMEN (S) RECEIVED. 1. BRN: RIGHT FRONTAL PARIETA...
00:33:57  [INFO]    │  #2  patient=TCGA-06-0165  label=GBM  sim=0.7931  ✓ MATCH
00:33:57  [INFO]    │      Text: Anatomic athology/Cytology. Patient Name: 4. Sex: Male. NAME : 4. SURGERY DATE: RECEIVE DATE : PATHO...
00:33:57  [INFO]    │  #3  patient=TCGA-76-4927  label=GBM  sim=0.7871  ✓ MATCH
00:33:57  [INFO]    │      Text: Pq nt: FINAL DIAGNOSIS: 1. Left frontal brain lesion: glioblastoma multiforme, grade IV of IV. (WHO ...
00:33:57  [INFO]    └─────────────────────────────────────────────────────────────┘
00:33:57  [INFO]  
  ┌─────────────────────────────────────────────────────────────┐
00:33:57  [INFO]    │ TEST REPORT  patient=TCGA-B0-4823  true_label=KIRC
00:33:57  [INFO]    │ Text preview: FINAL DIAGNOSIS. KIONEY, RIGHT, HAND-ASSISTED LAPAROSCOPIC RADICAL NEPHRECTOMY -. A. RENAL CELL CARCINOMA, CLEAR CELL (C...
00:33:57  [INFO]    ├─────────────────────────────────────────────────────────────┤
00:33:57  [INFO]    │ TOP 3 NEAREST TRAINING NEIGHBORS:
00:33:57  [INFO]    │  #1  patient=TCGA-B0-4838  label=KIRC  sim=0.8106  ✓ MATCH
00:33:57  [INFO]    │      Text: FINAL DIAGNOSIS: KIDNEY, RIGHT, LAPAROSCOPIC NEPHRECTOMY: A. RENAL CELL CARCINOMA, CONVENTIONAL (CLE...
00:33:57  [INFO]    │  #2  patient=TCGA-B0-4703  label=KIRC  sim=0.8084  ✓ MATCH
00:33:57  [INFO]    │      Text: FINAL DIAGNOSIS. LEFT KIDNEY, NEPHRECTOMY -. A. RENAL CELL CARCINOMA, CONVENTIONAL (CLEAR) CELL TYPE...
00:33:57  [INFO]    │  #3  patient=TCGA-BP-4975  label=KIRC  sim=0.8063  ✓ MATCH
00:33:57  [INFO]    │      Text: Clinical Diagnosis & History: ith incidentally diagnosed right renal mass. Specimens Submitted: 1: S...
00:33:57  [INFO]    └─────────────────────────────────────────────────────────────┘
00:33:57  [INFO]  
  ┌─────────────────────────────────────────────────────────────┐
00:33:57  [INFO]    │ TEST REPORT  patient=TCGA-N6-A4VF  true_label=UCS
00:33:57  [INFO]    │ Text preview: FINAL DIAGNOSIS: PART 1: OMENTUM, OMENTECTOMY -. BENIGN ADIPOSE TISSUE. PART 2: UTERUS WITH CERVIX, BILATERAL ADNEX, TOT...
00:33:57  [INFO]    ├─────────────────────────────────────────────────────────────┤
00:33:57  [INFO]    │ TOP 3 NEAREST TRAINING NEIGHBORS:
00:33:57  [INFO]    │  #1  patient=TCGA-BG-A0M0  label=UCEC  sim=0.8650  ✗ WRONG
00:33:57  [INFO]    │      Text: FINAL DIAGNOSIS: PART 1: UTERUS, CERVIX, BILATERAL ADNEXA, TOTAL ABDOMINAL HYSTERECTOMY AND BILATERA...
00:33:57  [INFO]    │  #2  patient=TCGA-BG-A0M3  label=UCEC  sim=0.8570  ✗ WRONG
00:33:57  [INFO]    │      Text: FINAL DIAGNOSIS: PART 1: OMENTUM, BIOPSY -. FIBROADIPOSE TISSUE, NEGATIVE FOR TUMOR. PART 2: UTERUS ...
00:33:57  [INFO]    │  #3  patient=TCGA-N9-A4Q1  label=UCS  sim=0.8459  ✓ MATCH
00:33:57  [INFO]    │      Text: DIAGNOSIS. (A) UTERUS, CERVIX, RIGHT FALLOPIAN TUBE, AND RIGHT OVARY: UTERUS, MALIGNANT MIXED MULLER...
00:33:57  [INFO]    └─────────────────────────────────────────────────────────────┘
00:33:57  [INFO]  
  ┌─────────────────────────────────────────────────────────────┐
00:33:57  [INFO]    │ TEST REPORT  patient=TCGA-EM-A22P  true_label=THCA
00:33:57  [INFO]    │ Text preview: M. Facility: MD: Specimen(s) Received. 1. Thyroid: Total thyroid stitch mark left superior pole. 2. Lymph node: mediasti...
00:33:57  [INFO]    ├─────────────────────────────────────────────────────────────┤
00:33:57  [INFO]    │ TOP 3 NEAREST TRAINING NEIGHBORS:
00:33:57  [INFO]    │  #1  patient=TCGA-EM-A1CU  label=THCA  sim=0.8914  ✓ MATCH
00:33:57  [INFO]    │      Text: M. Facility: Specimen(s) Received. 1. Thyroid: total thyroid central compartment and superior medias...
00:33:57  [INFO]    │  #2  patient=TCGA-EM-A22K  label=THCA  sim=0.8888  ✓ MATCH
00:33:57  [INFO]    │      Text: F. Facility: Specimen(s) Received. 1. Thyroid: total, stitch - left upper lobe, level 6 neck. 2. Par...
00:33:57  [INFO]    │  #3  patient=TCGA-EM-A3AO  label=THCA  sim=0.8858  ✓ MATCH
00:33:57  [INFO]    │      Text: Facility: 1. Neck: Right neck level 2. 2. Right neck level 3. 3. Right neck level 4. 4. Central comp...
00:33:57  [INFO]    └─────────────────────────────────────────────────────────────┘
00:33:57  [INFO]  
  INTERPRETATION:
00:33:57  [INFO]    sim > 0.90 : very similar reports (likely same cancer, similar language)
00:33:57  [INFO]    sim ~ 0.80 : moderately similar (same organ system, different subtype)
00:33:57  [INFO]    sim < 0.70 : different cancer families
00:33:57  [INFO]    When top neighbors are all correct → model is confident and right.
00:33:57  [INFO]    When neighbors are mixed labels → ambiguous region → model may be wrong.
00:33:57  [INFO]  ======================================================================
00:33:57  [INFO]    STEP 6 — Full Comparison: All Methods So Far
00:33:57  [INFO]  ======================================================================
00:33:57  [INFO]    Method                                Accuracy     F1-W Notes
00:33:57  [INFO]    ---------------------------------------------------------------------------
00:33:57  [INFO]    Cedars-Sinai (BoW + LR)                 95.31%   95.13%  baseline
00:33:57  [INFO]    Method 1 (TF-IDF + LR)                  96.36%   ~96.3%  +bigrams +class_weight
00:33:57  [INFO]    Method 2 (TF-IDF + SVM)                 96.36%   ~96.3%  max-margin classifier
00:33:57  [INFO]    Method 4 (Embedding k-NN, K=5)          90.13%   89.93%  semantic similarity
00:33:57  [INFO]  
  ANALYSIS — WHY DOES EMBEDDING k-NN COMPARE THIS WAY?

  ADVANTAGES of embedding k-NN:
    - Captures semantic meaning (synonyms, paraphrasing)
    - No training required — uses pre-trained medical knowledge
    - Fully interpretable: you can literally show which training reports
      were used to make each prediction (the nearest neighbors)
    - Adding new cancer types = just add new embeddings, no retraining

  DISADVANTAGES vs TF-IDF methods:
    - 512 token BERT limit cuts off long pathology reports
    - TF-IDF can use all words in every report (no length limit)
    - k-NN is slower at prediction time (compares to all training points)
    - General-purpose model may not know medical jargon as well as
      a fine-tuned model (addressed in Method 3 — BERT fine-tuning)

  WHY METHOD 4 IS STILL IMPORTANT:
    - It's the foundation for Method 5 (RAG):
      RAG = find similar reports (Method 4) → pass them to an LLM as context
    - It proves that semantic similarity alone can classify cancer types
    - It can handle new cancer types with ZERO retraining
    
00:33:57  [INFO]  --- Hard cancer types comparison ---
00:33:57  [INFO]    Cancer types below 0.85 F1 in Method 4 (8 types):
00:33:57  [INFO]      UCS (Uterine Carcinosarcoma) — F1=0.435, n_train=39
00:33:57  [INFO]      KICH (Kidney Chromophobe) — F1=0.632, n_train=78
00:33:57  [INFO]      READ (Rectum adenocarcinoma) — F1=0.719, n_train=113
00:33:57  [INFO]      LUSC (Lung squamous cell carcinoma) — F1=0.730, n_train=328
00:33:57  [INFO]      KIRP (Kidney renal papillary cell carcinoma) — F1=0.739, n_train=196
00:33:57  [INFO]      LUAD (Lung adenocarcinoma) — F1=0.757, n_train=342
00:33:57  [INFO]      CHOL (Cholangiocarcinoma) — F1=0.815, n_train=30
00:33:57  [INFO]      DLBC (Diffuse Large B-cell Lymphoma) — F1=0.833, n_train=33
00:33:57  [INFO]    → Method 3 (BERT fine-tuning) is the next step to push these further.
00:33:57  [INFO]  ======================================================================
00:33:57  [INFO]    DONE
00:33:57  [INFO]  ======================================================================
00:33:57  [INFO]    Method 4 Best K            : 5
00:33:57  [INFO]    Method 4 Test Accuracy     : 90.13%
00:33:57  [INFO]    Method 4 F1 Weighted       : 89.93%
00:33:57  [INFO]    Total runtime              : 6.4 min
00:33:57  [INFO]    Embeddings cached at       : c:\Users\kings\OneDrive\Desktop\CSUDH_Kingslee\CSUDH-Spring-2026\CSC-590-MastersProject\MastersProject\demo\ProjectCode\output
00:33:57  [INFO]    Log saved to               : c:\Users\kings\OneDrive\Desktop\CSUDH_Kingslee\CSUDH-Spring-2026\CSC-590-MastersProject\MastersProject\demo\ProjectCode\output\method4_embedding_knn_run.log
00:33:57  [INFO]    NEXT: Run Method 5 (RAG + LLM) which BUILDS ON these embeddings.
