23:08:30  [INFO]    Log file : c:\Users\kings\OneDrive\Desktop\CSUDH_Kingslee\CSUDH-Spring-2026\CSC-590-MastersProject\MastersProject\demo\ProjectCode\output\method2_tfidf_svm_run.log
23:08:30  [INFO]  ======================================================================
23:08:30  [INFO]    MEDREx — Method 2: TF-IDF + SVM (LinearSVC)
23:08:30  [INFO]    CSC-590 Masters Project, CSUDH Spring 2026
23:08:30  [INFO]    Goal: Can SVM do better than LR on high-dimensional TF-IDF text?
23:08:30  [INFO]  ======================================================================
23:08:30  [INFO]  ======================================================================
23:08:30  [INFO]    STEP 1 — Load & Compile Dataset
23:08:30  [INFO]  ======================================================================
23:08:32  [INFO]    Reading reports : c:\Users\kings\OneDrive\Desktop\CSUDH_Kingslee\CSUDH-Spring-2026\CSC-590-MastersProject\MastersProject\DataSet\TCGA_Reports.csv\TCGA_Reports.csv
23:08:32  [INFO]    Loaded 9,523 reports in 0.4s
23:08:32  [INFO]    Columns : ['patient_filename', 'text']
23:08:32  [INFO]  --- Extract patient_id from filename ---
23:08:32  [INFO]    Raw filename  : 'TCGA-BP-5195.25c0b433-5557-4165-922e-2c1eac9c26f0'
23:08:32  [INFO]    patient_id    : 'TCGA-BP-5195'  (everything before the first '.')
23:08:32  [INFO]    Unique patient reports: 9,523
23:08:32  [INFO]  
                patient_id                                                                                 text
TCGA-BP-5195  TCGA-BP-5195  Date of Recelpt: Clinical Diagnosis & History: Incidental 3 cm left upper pole r...
TCGA-D7-8573  TCGA-D7-8573  Material: 1) Material: stomach, Method of collection: Lesion resection. Histopat...
TCGA-EI-7004  TCGA-EI-7004  page 1 / 1. copy No. 3. Examination: Histopathological examination. Cost of diag...
23:08:32  [INFO]  --- Load cancer type labels ---
23:08:32  [INFO]    Label rows    : 11,160
23:08:32  [INFO]    Unique types  : 33
23:08:32  [INFO]  --- Merge reports + labels ---
23:08:32  [INFO]    Merged shape  : (9523, 4)
23:08:32  [INFO]  
                patient_id                                                                                 text cancer_type                   cancer_type_name
TCGA-BP-5195  TCGA-BP-5195  Date of Recelpt: Clinical Diagnosis & History: Incidental 3 cm left upper pole r...        KIRC  Kidney renal clear cell carcinoma
TCGA-D7-8573  TCGA-D7-8573  Material: 1) Material: stomach, Method of collection: Lesion resection. Histopat...        STAD             Stomach adenocarcinoma
TCGA-EI-7004  TCGA-EI-7004  page 1 / 1. copy No. 3. Examination: Histopathological examination. Cost of diag...        READ              Rectum adenocarcinoma
23:08:32  [INFO]  --- Cancer type distribution (all 33 types) ---
23:08:32  [INFO]      BRCA   Breast invasive carcinoma                     1034 (10.9%)  ██████████████████████████████████
23:08:32  [INFO]      UCEC   Uterine Corpus Endometrial Carcinoma           546 ( 5.7%)  ██████████████████
23:08:32  [INFO]      KIRC   Kidney renal clear cell carcinoma              525 ( 5.5%)  █████████████████
23:08:32  [INFO]      HNSC   Head and Neck squamous cell carcinoma          520 ( 5.5%)  █████████████████
23:08:32  [INFO]      LUAD   Lung adenocarcinoma                            488 ( 5.1%)  ████████████████
23:08:32  [INFO]      THCA   Thyroid carcinoma                              487 ( 5.1%)  ████████████████
23:08:32  [INFO]      LGG    Brain Lower Grade Glioma                       469 ( 4.9%)  ███████████████
23:08:32  [INFO]      LUSC   Lung squamous cell carcinoma                   468 ( 4.9%)  ███████████████
23:08:32  [INFO]      PRAD   Prostate adenocarcinoma                        446 ( 4.7%)  ██████████████
23:08:32  [INFO]      COAD   Colon adenocarcinoma                           418 ( 4.4%)  █████████████
23:08:32  [INFO]      GBM    Glioblastoma multiforme                        399 ( 4.2%)  █████████████
23:08:32  [INFO]      BLCA   Bladder Urothelial Carcinoma                   379 ( 4.0%)  ████████████
23:08:32  [INFO]      OV     Ovarian serous cystadenocarcinoma              371 ( 3.9%)  ████████████
23:08:32  [INFO]      STAD   Stomach adenocarcinoma                         361 ( 3.8%)  ████████████
23:08:32  [INFO]      LIHC   Liver hepatocellular carcinoma                 341 ( 3.6%)  ███████████
23:08:32  [INFO]      CESC   Cervical squamous cell carcinoma               289 ( 3.0%)  █████████
23:08:32  [INFO]      KIRP   Kidney renal papillary cell carcinoma          280 ( 2.9%)  █████████
23:08:32  [INFO]      SARC   Sarcoma                                        249 ( 2.6%)  ████████
23:08:32  [INFO]      PAAD   Pancreatic adenocarcinoma                      176 ( 1.8%)  █████
23:08:32  [INFO]      PCPG   Pheochromocytoma and Paraganglioma             174 ( 1.8%)  █████
23:08:32  [INFO]      READ   Rectum adenocarcinoma                          162 ( 1.7%)  █████
23:08:32  [INFO]      ESCA   Esophageal carcinoma                           146 ( 1.5%)  ████
23:08:32  [INFO]      THYM   Thymoma                                        114 ( 1.2%)  ███
23:08:32  [INFO]      KICH   Kidney Chromophobe                             112 ( 1.2%)  ███
23:08:32  [INFO]      SKCM   Skin Cutaneous Melanoma                        102 ( 1.1%)  ███
23:08:32  [INFO]      ACC    Adrenocortical carcinoma                        90 ( 0.9%)  ███
23:08:32  [INFO]      TGCT   Testicular Germ Cell Tumors                     87 ( 0.9%)  ██
23:08:32  [INFO]      MESO   Mesothelioma                                    79 ( 0.8%)  ██
23:08:32  [INFO]      UVM    Uveal Melanoma                                  65 ( 0.7%)  ██
23:08:32  [INFO]      UCS    Uterine Carcinosarcoma                          56 ( 0.6%)  █
23:08:32  [INFO]      DLBC   Diffuse Large B-cell Lymphoma                   47 ( 0.5%)  █
23:08:32  [INFO]      CHOL   Cholangiocarcinoma                              43 ( 0.5%)  █
23:08:32  [INFO]  ======================================================================
23:08:32  [INFO]    STEP 2 — Stratified Train / Test Split  (70% train / 30% test)
23:08:32  [INFO]  ======================================================================
23:08:35  [INFO]    Same split as Method 1 — random_state=42, stratify=cancer_type.
23:08:35  [INFO]    Using identical split ensures a FAIR comparison between LR and SVM.
23:08:35  [INFO]    Both models see the same training examples and are tested on the same reports.
23:08:35  [INFO]  
  Total  : 9,523 reports
23:08:35  [INFO]    Train  : 6,666 reports (70%)
23:08:35  [INFO]    Test   : 2,857  reports (30%)
23:08:35  [INFO]  --- Stratification check — rare cancer types ---
23:08:35  [INFO]    UCS     total= 56  train= 39  test= 17
23:08:35  [INFO]    CHOL    total= 43  train= 30  test= 13
23:08:35  [INFO]    KICH    total=112  train= 78  test= 34
23:08:35  [INFO]    UVM     total= 65  train= 46  test= 19
23:08:35  [INFO]    MESO    total= 79  train= 55  test= 24
23:08:35  [INFO]    All rare types preserved in both splits — stratification is working.
23:08:35  [INFO]  ======================================================================
23:08:35  [INFO]    STEP 3 — TF-IDF Vectorizer  (same parameters as Method 1)
23:08:35  [INFO]  ======================================================================
23:08:35  [INFO]    Using IDENTICAL TF-IDF parameters as Method 1:
23:08:35  [INFO]      max_features=15000   — top 15k most informative words/phrases
23:08:35  [INFO]      ngram_range=(1, 2)   — unigrams + bigrams (captures medical phrases)
23:08:35  [INFO]      sublinear_tf=True    — log-compress term frequency
23:08:35  [INFO]      min_df=2             — drop words in fewer than 2 documents
23:08:35  [INFO]    Why keep them the same? We want to isolate the effect of the classifier.
23:08:35  [INFO]    If we change TF-IDF too, we can't tell if SVM or the features caused the difference.
23:08:42  [INFO]    Fit + transform done in 7.0s
23:08:42  [INFO]    Vocabulary size   : 15,000 features
23:08:42  [INFO]    Train matrix shape: (6666, 15000)
23:08:42  [INFO]    Test  matrix shape: (2857, 15000)
23:08:42  [INFO]    Sparsity          : 2.509% non-zero
23:08:42  [INFO]  ======================================================================
23:08:42  [INFO]    STEP 4 — LinearSVC  (Support Vector Machine for Text)
23:08:42  [INFO]  ======================================================================
23:08:42  [INFO]    HOW DOES SVM WORK?
23:08:42  [INFO]    SVM draws a decision boundary (hyperplane) between classes.
23:08:42  [INFO]    It doesn't just find ANY boundary — it finds the one with the
23:08:42  [INFO]    MAXIMUM MARGIN: the widest possible gap between the classes.
23:08:42  [INFO]    Reports close to the boundary (support vectors) define the margin.
23:08:42  [INFO]    Wide margin = better generalisation to unseen data.
23:08:42  [INFO]  
23:08:42  [INFO]    LOGISTIC REGRESSION vs SVM — key difference:
23:08:42  [INFO]    LR: minimises log-loss across ALL training points (every point contributes)
23:08:42  [INFO]    SVM: only cares about support vectors (points near the boundary)
23:08:42  [INFO]    SVM ignores easy examples far from the boundary → focuses on hard cases
23:08:42  [INFO]    → Often better on text because most documents are easy to classify.
23:08:42  [INFO]  
23:08:42  [INFO]    WHY LinearSVC and NOT SVC(kernel='rbf')?
23:08:42  [INFO]    Our feature matrix: 6,666 rows × 15,000 columns (very high-dimensional).
23:08:42  [INFO]    rbf kernel needs to compute similarity between all pairs of documents → O(n²) → too slow.
23:08:42  [INFO]    LinearSVC: works directly in the feature space → O(n×d) → very fast.
23:08:42  [INFO]    Rule of thumb: for text (n_features >> n_samples), always use LinearSVC.
23:08:42  [INFO]  --- LinearSVC parameters — what each one does ---
23:08:42  [INFO]    C=1.0  (regularisation strength)
23:08:42  [INFO]      C controls how strictly the model tries to correctly classify every training point.
23:08:42  [INFO]      Low  C (e.g. 0.01) → wide margin, some training errors allowed → less overfit
23:08:42  [INFO]      High C (e.g. 100)  → narrow margin, almost all training points correct → can overfit
23:08:42  [INFO]      C=1.0 is the standard starting point (same as Method 1 LR).
23:08:42  [INFO]    class_weight='balanced'
23:08:42  [INFO]      Same as Method 1: rare cancers (UCS=37, CHOL=36) get higher weight.
23:08:42  [INFO]      Without this, SVM would mostly ignore rare classes.
23:08:42  [INFO]    max_iter=2000
23:08:42  [INFO]      SVM optimisation sometimes needs more iterations than LR.
23:08:42  [INFO]      2000 ensures it converges on this multi-class 33-label problem.
23:08:42  [INFO]    multi_class strategy: one-vs-rest (OvR)
23:08:42  [INFO]      LinearSVC trains 33 binary classifiers: 'BRCA vs rest', 'GBM vs rest', etc.
23:08:42  [INFO]      Each classifier gets one score; highest score wins the prediction.
23:08:42  [INFO]  --- Class weights ---
23:08:42  [INFO]    Class weights assigned to each cancer type:
23:08:42  [INFO]      CHOL    n_train=  30  weight=  6.94  ████████████████████
23:08:42  [INFO]      DLBC    n_train=  33  weight=  6.31  ██████████████████
23:08:42  [INFO]      UCS     n_train=  39  weight=  5.34  ████████████████
23:08:42  [INFO]      UVM     n_train=  46  weight=  4.53  █████████████
23:08:42  [INFO]      MESO    n_train=  55  weight=  3.79  ███████████
23:08:42  [INFO]      TGCT    n_train=  61  weight=  3.41  ██████████
23:08:42  [INFO]      ACC     n_train=  63  weight=  3.31  █████████
23:08:42  [INFO]      SKCM    n_train=  71  weight=  2.93  ████████
23:08:42  [INFO]      KICH    n_train=  78  weight=  2.67  ████████
23:08:42  [INFO]      THYM    n_train=  80  weight=  2.60  ███████
23:08:42  [INFO]      ESCA    n_train= 102  weight=  2.04  ██████
23:08:42  [INFO]      READ    n_train= 113  weight=  1.84  █████
23:08:42  [INFO]      PCPG    n_train= 122  weight=  1.71  █████
23:08:42  [INFO]      PAAD    n_train= 123  weight=  1.69  █████
23:08:42  [INFO]      SARC    n_train= 174  weight=  1.20  ███
23:08:42  [INFO]      KIRP    n_train= 196  weight=  1.06  ███
23:08:42  [INFO]      CESC    n_train= 202  weight=  1.03  ███
23:08:42  [INFO]      LIHC    n_train= 239  weight=  0.87  ██
23:08:42  [INFO]      STAD    n_train= 253  weight=  0.82  ██
23:08:42  [INFO]      OV      n_train= 260  weight=  0.80  ██
23:08:42  [INFO]      BLCA    n_train= 265  weight=  0.79  ██
23:08:42  [INFO]      GBM     n_train= 279  weight=  0.75  ██
23:08:42  [INFO]      COAD    n_train= 293  weight=  0.71  ██
23:08:42  [INFO]      PRAD    n_train= 312  weight=  0.67  ██
23:08:42  [INFO]      LGG     n_train= 328  weight=  0.64  █
23:08:42  [INFO]      LUSC    n_train= 328  weight=  0.64  █
23:08:42  [INFO]      THCA    n_train= 341  weight=  0.61  █
23:08:42  [INFO]      LUAD    n_train= 342  weight=  0.61  █
23:08:42  [INFO]      HNSC    n_train= 364  weight=  0.57  █
23:08:42  [INFO]      KIRC    n_train= 368  weight=  0.57  █
23:08:42  [INFO]      UCEC    n_train= 382  weight=  0.55  █
23:08:42  [INFO]      BRCA    n_train= 724  weight=  0.29  
23:08:42  [INFO]  --- Training LinearSVC ---
23:08:42  [INFO]    Parameters: C=1.0, max_iter=2000, class_weight=balanced, random_state=42
23:08:42  [INFO]    Training... (usually faster than LR — ~10-30 seconds)
23:08:45  [INFO]    Training complete in 2.1s
23:08:45  [INFO]    Number of classes : 32
23:08:45  [INFO]    Coefficient matrix: (32, 15000)  (33 classes × 15,000 features)
23:08:45  [INFO]  --- Results on TRAINING set ---
23:08:45  [INFO]    Train Accuracy : 99.74%
23:08:45  [INFO]    (High training accuracy is expected — model has seen these reports)
23:08:45  [INFO]  --- Results on TEST set  ← THE REAL SCORE ---
23:08:45  [INFO]    ╔══════════════════════════════════════════════════════════╗
23:08:45  [INFO]    ║  METHOD 2 — TF-IDF + SVM (LinearSVC) RESULTS           ║
23:08:45  [INFO]    ║                                                          ║
23:08:45  [INFO]    ║  Test Accuracy        :  97.37%                      ║
23:08:45  [INFO]    ║  F1 Weighted          :  97.35%                      ║
23:08:45  [INFO]    ║  F1 Macro             :  96.48%                      ║
23:08:45  [INFO]    ║                                                          ║
23:08:45  [INFO]    ║  Cedars-Sinai BoW+LR  : ~95.31%                         ║
23:08:45  [INFO]    ╚══════════════════════════════════════════════════════════╝
23:08:45  [INFO]  
  Full classification report (all 33 cancer types):
23:08:45  [INFO]                precision    recall  f1-score   support

         ACC       1.00      0.96      0.98        27
        BLCA       1.00      1.00      1.00       114
        BRCA       1.00      1.00      1.00       310
        CESC       0.99      0.99      0.99        87
        CHOL       0.91      0.77      0.83        13
        COAD       0.94      0.97      0.95       125
        DLBC       1.00      1.00      1.00        14
        ESCA       0.98      0.95      0.97        44
         GBM       0.97      0.97      0.97       120
        HNSC       1.00      0.99      0.99       156
        KICH       0.97      0.97      0.97        34
        KIRC       0.98      0.99      0.99       157
        KIRP       1.00      0.96      0.98        84
         LGG       0.97      0.98      0.98       141
        LIHC       0.97      0.99      0.98       102
        LUAD       0.93      0.95      0.94       146
        LUSC       0.92      0.92      0.92       140
        MESO       1.00      1.00      1.00        24
          OV       0.95      0.96      0.96       111
        PAAD       0.98      1.00      0.99        53
        PCPG       0.98      1.00      0.99        52
        PRAD       1.00      1.00      1.00       134
        READ       0.89      0.84      0.86        49
        SARC       0.99      0.93      0.96        75
        SKCM       0.97      1.00      0.98        31
        STAD       0.97      0.97      0.97       108
        TGCT       1.00      1.00      1.00        26
        THCA       0.99      1.00      1.00       146
        THYM       0.97      0.97      0.97        34
        UCEC       0.95      0.98      0.96       164
         UCS       1.00      0.65      0.79        17
         UVM       1.00      1.00      1.00        19

    accuracy                           0.97      2857
   macro avg       0.97      0.96      0.96      2857
weighted avg       0.97      0.97      0.97      2857

23:08:45  [INFO]  --- Per-class F1 sorted worst → best ---
23:08:45  [INFO]    Code   Cancer Name                                       F1  #Train  #Test
23:08:45  [INFO]    --------------------------------------------------------------------------
23:08:45  [INFO]    UCS    Uterine Carcinosarcoma                         0.786      39     17 ← hardest
23:08:45  [INFO]    CHOL   Cholangiocarcinoma                             0.833      30     13 ← hardest
23:08:45  [INFO]    READ   Rectum adenocarcinoma                          0.863     113     49
23:08:45  [INFO]    LUSC   Lung squamous cell carcinoma                   0.921     328    140
23:08:45  [INFO]    LUAD   Lung adenocarcinoma                            0.939     342    146
23:08:45  [INFO]    COAD   Colon adenocarcinoma                           0.953     293    125
23:08:45  [INFO]    OV     Ovarian serous cystadenocarcinoma              0.955     260    111
23:08:45  [INFO]    SARC   Sarcoma                                        0.959     174     75
23:08:45  [INFO]    UCEC   Uterine Corpus Endometrial Carcinoma           0.964     382    164
23:08:45  [INFO]    ESCA   Esophageal carcinoma                           0.966     102     44
23:08:45  [INFO]    KICH   Kidney Chromophobe                             0.971      78     34
23:08:45  [INFO]    THYM   Thymoma                                        0.971      80     34
23:08:45  [INFO]    GBM    Glioblastoma multiforme                        0.971     279    120
23:08:45  [INFO]    STAD   Stomach adenocarcinoma                         0.972     253    108
23:08:45  [INFO]    LGG    Brain Lower Grade Glioma                       0.975     328    141
23:08:45  [INFO]    LIHC   Liver hepatocellular carcinoma                 0.981     239    102
23:08:45  [INFO]    ACC    Adrenocortical carcinoma                       0.981      63     27
23:08:45  [INFO]    KIRP   Kidney renal papillary cell carcinoma          0.982     196     84
23:08:45  [INFO]    SKCM   Skin Cutaneous Melanoma                        0.984      71     31
23:08:45  [INFO]    KIRC   Kidney renal clear cell carcinoma              0.987     368    157
23:08:45  [INFO]    CESC   Cervical squamous cell carcinoma               0.989     202     87
23:08:45  [INFO]    PCPG   Pheochromocytoma and Paraganglioma             0.990     122     52
23:08:45  [INFO]    PAAD   Pancreatic adenocarcinoma                      0.991     123     53
23:08:45  [INFO]    HNSC   Head and Neck squamous cell carcinoma          0.994     364    156
23:08:45  [INFO]    THCA   Thyroid carcinoma                              0.997     341    146
23:08:45  [INFO]    BRCA   Breast invasive carcinoma                      1.000     724    310
23:08:45  [INFO]    DLBC   Diffuse Large B-cell Lymphoma                  1.000      33     14
23:08:45  [INFO]    BLCA   Bladder Urothelial Carcinoma                   1.000     265    114
23:08:45  [INFO]    PRAD   Prostate adenocarcinoma                        1.000     312    134
23:08:45  [INFO]    MESO   Mesothelioma                                   1.000      55     24
23:08:45  [INFO]    TGCT   Testicular Germ Cell Tumors                    1.000      61     26
23:08:45  [INFO]    UVM    Uveal Melanoma                                 1.000      46     19
23:08:45  [INFO]  
  ANALYSIS:
23:08:45  [INFO]    Cancer types below 0.85 F1 (2 types):
23:08:45  [INFO]      UCS (Uterine Carcinosarcoma) — F1=0.786, only 39 training samples
23:08:45  [INFO]      CHOL (Cholangiocarcinoma) — F1=0.833, only 30 training samples
23:08:45  [INFO]  ======================================================================
23:08:45  [INFO]    STEP 5 — SVM Feature Analysis  (what did LinearSVC actually learn?)
23:08:45  [INFO]  ======================================================================
23:08:45  [INFO]    SVM COEFFICIENT MEANING:
23:08:45  [INFO]    LinearSVC (one-vs-rest) trains one weight vector per cancer type.
23:08:45  [INFO]    High positive weight = word/phrase strongly predicts THIS cancer type.
23:08:45  [INFO]    High negative weight = word/phrase argues AGAINST this cancer type.
23:08:45  [INFO]    These are the 'decision rules' SVM learned from 9,523 pathology reports.
23:08:45  [INFO]    Unlike LR probabilities, SVM weights are raw decision scores (not 0-1).
23:08:45  [INFO]  --- Top 5 most predictive words/phrases per cancer type ---
23:08:45  [INFO]  
  ┌─ ACC — Adrenocortical carcinoma  (n_train=63) ─────
23:08:45  [INFO]    │  adrenocortical                      +1.766  +++++
23:08:45  [INFO]    │  adrenocortical carcinoma            +1.639  ++++
23:08:45  [INFO]    │  adrenal cortical                    +1.591  ++++
23:08:45  [INFO]    │  cortical carcinoma                  +1.545  ++++
23:08:45  [INFO]    │  weiss                               +1.363  ++++
23:08:45  [INFO]    └────────────────────────────────────────────────────────────
23:08:45  [INFO]  
  ┌─ BLCA — Bladder Urothelial Carcinoma  (n_train=265) ─────
23:08:45  [INFO]    │  bladder                             +2.546  +++++++
23:08:45  [INFO]    │  urothelial                          +2.366  +++++++
23:08:45  [INFO]    │  urothelial carcinoma                +1.778  +++++
23:08:45  [INFO]    │  the bladder                         +1.124  +++
23:08:45  [INFO]    │  transitional                        +1.082  +++
23:08:45  [INFO]    └────────────────────────────────────────────────────────────
23:08:45  [INFO]  
  ┌─ BRCA — Breast invasive carcinoma  (n_train=724) ─────
23:08:45  [INFO]    │  breast                              +3.224  +++++++++
23:08:45  [INFO]    │  ductal                              +1.469  ++++
23:08:45  [INFO]    │  ductal carcinoma                    +1.274  +++
23:08:45  [INFO]    │  right breast                        +1.247  +++
23:08:45  [INFO]    │  left breast                         +1.171  +++
23:08:45  [INFO]    └────────────────────────────────────────────────────────────
23:08:45  [INFO]  
  ┌─ CESC — Cervical squamous cell carcinoma  (n_train=202) ─────
23:08:45  [INFO]    │  cervix                              +3.132  +++++++++
23:08:45  [INFO]    │  cervical                            +2.362  +++++++
23:08:45  [INFO]    │  radical hysterectomy                +1.625  ++++
23:08:45  [INFO]    │  site cervix                         +1.532  ++++
23:08:45  [INFO]    │  vaginal                             +1.490  ++++
23:08:45  [INFO]    └────────────────────────────────────────────────────────────
23:08:45  [INFO]  
  ┌─ CHOL — Cholangiocarcinoma  (n_train=30) ─────
23:08:45  [INFO]    │  cholangiocarcinoma                  +3.011  +++++++++
23:08:45  [INFO]    │  intrahepatic                        +0.938  ++
23:08:45  [INFO]    │  hepatic                             +0.913  ++
23:08:45  [INFO]    │  date diagnosis                      +0.887  ++
23:08:45  [INFO]    │  surgery date                        +0.804  ++
23:08:45  [INFO]    └────────────────────────────────────────────────────────────
23:08:45  [INFO]  
  ┌─ COAD — Colon adenocarcinoma  (n_train=293) ─────
23:08:45  [INFO]    │  colon                               +2.617  +++++++
23:08:45  [INFO]    │  sigmoid colon                       +1.603  ++++
23:08:45  [INFO]    │  hemicolectomy                       +1.428  ++++
23:08:45  [INFO]    │  cecum                               +1.304  +++
23:08:45  [INFO]    │  colon tumor                         +1.229  +++
23:08:45  [INFO]    └────────────────────────────────────────────────────────────
23:08:45  [INFO]  
  ┌─ DLBC — Diffuse Large B-cell Lymphoma  (n_train=33) ─────
23:08:45  [INFO]    │  lymphoma                            +1.920  +++++
23:08:45  [INFO]    │  cell lymphoma                       +1.523  ++++
23:08:45  [INFO]    │  large cell                          +1.467  ++++
23:08:45  [INFO]    │  diffuse large                       +1.402  ++++
23:08:45  [INFO]    │  cd3                                 +1.020  +++
23:08:45  [INFO]    └────────────────────────────────────────────────────────────
23:08:45  [INFO]  
  ┌─ ESCA — Esophageal carcinoma  (n_train=102) ─────
23:08:45  [INFO]    │  esophagus                           +3.860  +++++++++++
23:08:45  [INFO]    │  esophageal                          +1.887  +++++
23:08:45  [INFO]    │  adventitia                          +1.213  +++
23:08:45  [INFO]    │  barrett                             +1.178  +++
23:08:45  [INFO]    │  esophagectomy                       +1.171  +++
23:08:45  [INFO]    └────────────────────────────────────────────────────────────
23:08:45  [INFO]  
  ┌─ GBM — Glioblastoma multiforme  (n_train=279) ─────
23:08:45  [INFO]    │  glioblastoma                        +4.123  ++++++++++++
23:08:45  [INFO]    │  grade iv                            +1.717  +++++
23:08:45  [INFO]    │  multiforme                          +1.677  +++++
23:08:45  [INFO]    │  glioblastoma multiforme             +1.672  +++++
23:08:45  [INFO]    │  brain                               +1.402  ++++
23:08:45  [INFO]    └────────────────────────────────────────────────────────────
23:08:45  [INFO]  
  ┌─ HNSC — Head and Neck squamous cell carcinoma  (n_train=364) ─────
23:08:45  [INFO]    │  tongue                              +1.844  +++++
23:08:45  [INFO]    │  larynx                              +1.439  ++++
23:08:45  [INFO]    │  squamous                            +1.414  ++++
23:08:45  [INFO]    │  tonsil                              +1.313  +++
23:08:45  [INFO]    │  squamous cell                       +1.102  +++
23:08:45  [INFO]    └────────────────────────────────────────────────────────────
23:08:45  [INFO]  
  ┌─ KICH — Kidney Chromophobe  (n_train=78) ─────
23:08:45  [INFO]    │  chromophobe                         +4.709  ++++++++++++++
23:08:45  [INFO]    │  carcinoma chromophobe               +2.599  +++++++
23:08:45  [INFO]    │  chromophobe type                    +2.298  ++++++
23:08:45  [INFO]    │  chromophobe renal                   +2.291  ++++++
23:08:45  [INFO]    │  renal mass                          +0.911  ++
23:08:45  [INFO]    └────────────────────────────────────────────────────────────
23:08:45  [INFO]  
  ┌─ KIRC — Kidney renal clear cell carcinoma  (n_train=368) ─────
23:08:45  [INFO]    │  clear cell                          +2.996  ++++++++
23:08:45  [INFO]    │  clear                               +2.693  ++++++++
23:08:45  [INFO]    │  cell type                           +2.103  ++++++
23:08:45  [INFO]    │  conventional                        +1.864  +++++
23:08:45  [INFO]    │  carcinoma clear                     +1.674  +++++
23:08:45  [INFO]    └────────────────────────────────────────────────────────────
23:08:45  [INFO]  
  ┌─ KIRP — Kidney renal papillary cell carcinoma  (n_train=196) ─────
23:08:45  [INFO]    │  papillary renal                     +3.860  +++++++++++
23:08:45  [INFO]    │  papillary                           +3.690  +++++++++++
23:08:45  [INFO]    │  carcinoma papillary                 +2.267  ++++++
23:08:45  [INFO]    │  papillary type                      +2.249  ++++++
23:08:45  [INFO]    │  renal                               +1.779  +++++
23:08:45  [INFO]    └────────────────────────────────────────────────────────────
23:08:45  [INFO]  
  ┌─ LGG — Brain Lower Grade Glioma  (n_train=328) ─────
23:08:45  [INFO]    │  anaplastic                          +2.014  ++++++
23:08:45  [INFO]    │  oligodendroglioma                   +1.983  +++++
23:08:45  [INFO]    │  astrocytoma                         +1.816  +++++
23:08:45  [INFO]    │  glioma                              +1.797  +++++
23:08:45  [INFO]    │  oligoastrocytoma                    +1.359  ++++
23:08:45  [INFO]    └────────────────────────────────────────────────────────────
23:08:45  [INFO]  
  ┌─ LIHC — Liver hepatocellular carcinoma  (n_train=239) ─────
23:08:45  [INFO]    │  hepatocellular                      +3.191  +++++++++
23:08:45  [INFO]    │  hepatocellular carcinoma            +3.071  +++++++++
23:08:45  [INFO]    │  liver                               +2.258  ++++++
23:08:45  [INFO]    │  of liver                            +0.897  ++
23:08:45  [INFO]    │  cirrhosis                           +0.834  ++
23:08:45  [INFO]    └────────────────────────────────────────────────────────────
23:08:45  [INFO]  
  ┌─ LUAD — Lung adenocarcinoma  (n_train=342) ─────
23:08:45  [INFO]    │  adenocarcinoma                      +3.016  +++++++++
23:08:45  [INFO]    │  lung                                +2.212  ++++++
23:08:45  [INFO]    │  differentiated adenocarcinoma       +1.391  ++++
23:08:45  [INFO]    │  type adenocarcinoma                 +1.324  +++
23:08:45  [INFO]    │  lobe                                +1.224  +++
23:08:45  [INFO]    └────────────────────────────────────────────────────────────
23:08:45  [INFO]  
  ┌─ LUSC — Lung squamous cell carcinoma  (n_train=328) ─────
23:08:45  [INFO]    │  squamous cell                       +3.448  ++++++++++
23:08:45  [INFO]    │  squamous                            +3.313  +++++++++
23:08:45  [INFO]    │  cell carcinoma                      +2.169  ++++++
23:08:45  [INFO]    │  lung                                +2.153  ++++++
23:08:45  [INFO]    │  type squamous                       +1.910  +++++
23:08:45  [INFO]    └────────────────────────────────────────────────────────────
23:08:45  [INFO]  
  ┌─ MESO — Mesothelioma  (n_train=55) ─────
23:08:45  [INFO]    │  mesothelioma                        +3.225  +++++++++
23:08:45  [INFO]    │  malignant mesothelioma              +2.065  ++++++
23:08:45  [INFO]    │  epithelioid                         +1.250  +++
23:08:45  [INFO]    │  malignant                           +1.222  +++
23:08:45  [INFO]    │  pleural                             +1.103  +++
23:08:45  [INFO]    └────────────────────────────────────────────────────────────
23:08:45  [INFO]  
  ┌─ OV — Ovarian serous cystadenocarcinoma  (n_train=260) ─────
23:08:45  [INFO]    │  serous                              +2.115  ++++++
23:08:45  [INFO]    │  ovarian                             +1.791  +++++
23:08:45  [INFO]    │  omentum                             +1.583  ++++
23:08:45  [INFO]    │  serous carcinoma                    +1.059  +++
23:08:45  [INFO]    │  serous adenocarcinoma               +1.005  +++
23:08:45  [INFO]    └────────────────────────────────────────────────────────────
23:08:45  [INFO]  
  ┌─ PAAD — Pancreatic adenocarcinoma  (n_train=123) ─────
23:08:45  [INFO]    │  pancreas                            +2.136  ++++++
23:08:45  [INFO]    │  pancreatic                          +2.073  ++++++
23:08:45  [INFO]    │  ductal adenocarcinoma               +1.354  ++++
23:08:45  [INFO]    │  the pancreas                        +1.200  +++
23:08:45  [INFO]    │  head                                +1.178  +++
23:08:45  [INFO]    └────────────────────────────────────────────────────────────
23:08:45  [INFO]  
  ┌─ PCPG — Pheochromocytoma and Paraganglioma  (n_train=122) ─────
23:08:45  [INFO]    │  pheochromocytoma                    +3.516  ++++++++++
23:08:45  [INFO]    │  adrenal                             +1.877  +++++
23:08:45  [INFO]    │  paraganglioma                       +1.838  +++++
23:08:45  [INFO]    │  right adrenal                       +1.005  +++
23:08:45  [INFO]    │  adrenalectomy                       +0.945  ++
23:08:45  [INFO]    └────────────────────────────────────────────────────────────
23:08:45  [INFO]  
  ┌─ PRAD — Prostate adenocarcinoma  (n_train=312) ─────
23:08:45  [INFO]    │  gleason                             +1.543  ++++
23:08:45  [INFO]    │  prostate                            +1.530  ++++
23:08:45  [INFO]    │  seminal                             +1.357  ++++
23:08:45  [INFO]    │  prostatectomy                       +0.926  ++
23:08:45  [INFO]    │  vesicles                            +0.821  ++
23:08:45  [INFO]    └────────────────────────────────────────────────────────────
23:08:45  [INFO]  
  ┌─ READ — Rectum adenocarcinoma  (n_train=113) ─────
23:08:45  [INFO]    │  rectum                              +4.280  ++++++++++++
23:08:45  [INFO]    │  rectal                              +3.662  ++++++++++
23:08:45  [INFO]    │  rectosigmoid                        +2.642  +++++++
23:08:45  [INFO]    │  perirectal                          +2.210  ++++++
23:08:45  [INFO]    │  anastomosis                         +0.990  ++
23:08:45  [INFO]    └────────────────────────────────────────────────────────────
23:08:45  [INFO]  
  ┌─ SARC — Sarcoma  (n_train=174) ─────
23:08:45  [INFO]    │  leiomyosarcoma                      +4.188  ++++++++++++
23:08:45  [INFO]    │  sarcoma                             +3.403  ++++++++++
23:08:45  [INFO]    │  liposarcoma                         +1.431  ++++
23:08:45  [INFO]    │  desmin                              +1.152  +++
23:08:45  [INFO]    │  retroperitoneal                     +1.091  +++
23:08:45  [INFO]    └────────────────────────────────────────────────────────────
23:08:45  [INFO]  
  ┌─ SKCM — Skin Cutaneous Melanoma  (n_train=71) ─────
23:08:45  [INFO]    │  melanoma                            +4.307  ++++++++++++
23:08:45  [INFO]    │  skin                                +1.739  +++++
23:08:45  [INFO]    │  site skin                           +0.972  ++
23:08:45  [INFO]    │  malignant melanoma                  +0.953  ++
23:08:45  [INFO]    │  clark                               +0.806  ++
23:08:45  [INFO]    └────────────────────────────────────────────────────────────
23:08:45  [INFO]  
  ┌─ STAD — Stomach adenocarcinoma  (n_train=253) ─────
23:08:45  [INFO]    │  stomach                             +3.012  +++++++++
23:08:45  [INFO]    │  gastric                             +1.804  +++++
23:08:45  [INFO]    │  gastrectomy                         +1.722  +++++
23:08:45  [INFO]    │  intestinal                          +1.478  ++++
23:08:45  [INFO]    │  intestinal type                     +1.027  +++
23:08:45  [INFO]    └────────────────────────────────────────────────────────────
23:08:45  [INFO]  
  ┌─ TGCT — Testicular Germ Cell Tumors  (n_train=61) ─────
23:08:45  [INFO]    │  testis                              +2.018  ++++++
23:08:45  [INFO]    │  seminoma                            +1.278  +++
23:08:45  [INFO]    │  testicle                            +0.972  ++
23:08:45  [INFO]    │  spermatic cord                      +0.774  ++
23:08:45  [INFO]    │  spermatic                           +0.773  ++
23:08:45  [INFO]    └────────────────────────────────────────────────────────────
23:08:45  [INFO]  
  ┌─ THCA — Thyroid carcinoma  (n_train=341) ─────
23:08:45  [INFO]    │  thyroid                             +2.181  ++++++
23:08:45  [INFO]    │  papillary                           +1.302  +++
23:08:45  [INFO]    │  papillary carcinoma                 +1.242  +++
23:08:45  [INFO]    │  papillary thyroid                   +1.194  +++
23:08:45  [INFO]    │  isthmus                             +1.171  +++
23:08:45  [INFO]    └────────────────────────────────────────────────────────────
23:08:45  [INFO]  
  ┌─ THYM — Thymoma  (n_train=80) ─────
23:08:45  [INFO]    │  thymoma                             +3.187  +++++++++
23:08:45  [INFO]    │  thymus                              +2.113  ++++++
23:08:45  [INFO]    │  thymic                              +1.600  ++++
23:08:45  [INFO]    │  thymectomy                          +1.231  +++
23:08:45  [INFO]    │  mediastinal                         +1.088  +++
23:08:45  [INFO]    └────────────────────────────────────────────────────────────
23:08:45  [INFO]  
  ┌─ UCEC — Uterine Corpus Endometrial Carcinoma  (n_train=382) ─────
23:08:45  [INFO]    │  endometrioid                        +2.547  +++++++
23:08:45  [INFO]    │  endometrioid adenocarcinoma         +1.785  +++++
23:08:45  [INFO]    │  endometrial                         +1.772  +++++
23:08:45  [INFO]    │  myometrial                          +1.705  +++++
23:08:45  [INFO]    │  myometrial invasion                 +1.526  ++++
23:08:45  [INFO]    └────────────────────────────────────────────────────────────
23:08:45  [INFO]  
  ┌─ UCS — Uterine Carcinosarcoma  (n_train=39) ─────
23:08:45  [INFO]    │  carcinosarcoma                      +3.069  +++++++++
23:08:45  [INFO]    │  malignant mixed                     +2.169  ++++++
23:08:45  [INFO]    │  mullerian tumor                     +1.789  +++++
23:08:45  [INFO]    │  mullerian                           +1.712  +++++
23:08:45  [INFO]    │  mixed mullerian                     +1.684  +++++
23:08:45  [INFO]    └────────────────────────────────────────────────────────────
23:08:45  [INFO]  
  ┌─ UVM — Uveal Melanoma  (n_train=46) ─────
23:08:45  [INFO]    │  optic                               +1.090  +++
23:08:45  [INFO]    │  eye                                 +1.084  +++
23:08:45  [INFO]    │  optic nerve                         +1.015  +++
23:08:45  [INFO]    │  uveal                               +0.834  ++
23:08:45  [INFO]    │  uveal melanoma                      +0.759  ++
23:08:45  [INFO]    └────────────────────────────────────────────────────────────
23:08:45  [INFO]  --- Comparing SVM vs LR top features — do they agree? ---
23:08:45  [INFO]    If SVM and LR learned DIFFERENT top words for the same cancer type,
23:08:45  [INFO]    that tells us something interesting about what each model finds important.
23:08:45  [INFO]    Below are SVM's top 3 diagnostic words across ALL cancer types:
23:08:45  [INFO]    Term                                Cancer  SVM Weight
23:08:45  [INFO]    -------------------------------------------------------
23:08:45  [INFO]    chromophobe                           KICH      +4.709
23:08:45  [INFO]    melanoma                              SKCM      +4.307
23:08:45  [INFO]    rectum                                READ      +4.280
23:08:45  [INFO]    leiomyosarcoma                        SARC      +4.188
23:08:45  [INFO]    glioblastoma                           GBM      +4.123
23:08:45  [INFO]    papillary renal                       KIRP      +3.860
23:08:45  [INFO]    esophagus                             ESCA      +3.860
23:08:45  [INFO]    papillary                             KIRP      +3.690
23:08:45  [INFO]    rectal                                READ      +3.662
23:08:45  [INFO]    pheochromocytoma                      PCPG      +3.516
23:08:45  [INFO]    squamous cell                         LUSC      +3.448
23:08:45  [INFO]    sarcoma                               SARC      +3.403
23:08:45  [INFO]    squamous                              LUSC      +3.313
23:08:45  [INFO]    mesothelioma                          MESO      +3.225
23:08:45  [INFO]    breast                                BRCA      +3.224
23:08:45  [INFO]    hepatocellular                        LIHC      +3.191
23:08:45  [INFO]    thymoma                               THYM      +3.187
23:08:45  [INFO]    cervix                                CESC      +3.132
23:08:45  [INFO]    hepatocellular carcinoma              LIHC      +3.071
23:08:45  [INFO]    carcinosarcoma                         UCS      +3.069
23:08:45  [INFO]    adenocarcinoma                        LUAD      +3.016
23:08:45  [INFO]    stomach                               STAD      +3.012
23:08:45  [INFO]    cholangiocarcinoma                    CHOL      +3.011
23:08:45  [INFO]    clear cell                            KIRC      +2.996
23:08:45  [INFO]    clear                                 KIRC      +2.693
23:08:45  [INFO]    rectosigmoid                          READ      +2.642
23:08:45  [INFO]    colon                                 COAD      +2.617
23:08:45  [INFO]    carcinoma chromophobe                 KICH      +2.599
23:08:45  [INFO]    endometrioid                          UCEC      +2.547
23:08:45  [INFO]    bladder                               BLCA      +2.546
23:08:45  [INFO]    urothelial                            BLCA      +2.366
23:08:45  [INFO]    cervical                              CESC      +2.362
23:08:45  [INFO]    chromophobe type                      KICH      +2.298
23:08:45  [INFO]    carcinoma papillary                   KIRP      +2.267
23:08:45  [INFO]    liver                                 LIHC      +2.258
23:08:45  [INFO]    lung                                  LUAD      +2.212
23:08:45  [INFO]    thyroid                               THCA      +2.181
23:08:45  [INFO]    cell carcinoma                        LUSC      +2.169
23:08:45  [INFO]    malignant mixed                        UCS      +2.169
23:08:45  [INFO]    pancreas                              PAAD      +2.136
23:08:45  [INFO]  ======================================================================
23:08:45  [INFO]    STEP 6 — Full Comparison: Method 2 vs Method 1 vs Cedars-Sinai
23:08:45  [INFO]  ======================================================================
23:08:45  [INFO]    Aspect                             Cedars-Sinai  Method 1 (LR) Method 2 (SVM)
23:08:45  [INFO]    --------------------------------------------------------------------------------
23:08:45  [INFO]    Vectorizer                       CountVectorizer         TF-IDF         TF-IDF
23:08:45  [INFO]    n-gram range                              (1,1)          (1,2)          (1,2)
23:08:45  [INFO]    Classifier                          LogisticReg    LogisticReg      LinearSVC
23:08:45  [INFO]    Class weighting                            None       balanced       balanced
23:08:45  [INFO]    Test Accuracy                            95.31%         96.36%         97.37%
23:08:45  [INFO]    F1 Weighted                              95.13%         96.34%         97.35%
23:08:45  [INFO]    vs Cedars-Sinai                        baseline         +1.05%         +2.06%
23:08:45  [INFO]  
  ANALYSIS — WHY DOES SVM COMPARE TO LR THIS WAY?

  SVM and LR often perform similarly on TF-IDF text data.
  The key difference is in HOW they handle errors:

  LR minimises log-loss (cross-entropy):
    It adjusts ALL weights based on ALL training examples.
    Easy examples (clear GBM reports) still contribute to the gradient.
    This can cause the model to overfit the easy majority classes.

  SVM minimises hinge loss:
    It only adjusts weights based on support vectors (hard examples).
    Easy examples far from the boundary are completely ignored.
    This focus on hard cases often helps with rare cancer types.

  WHEN SVM BEATS LR:
    — When the data has clear margin separability (pathology reports tend to)
    — When rare classes matter (SVM's margin focus helps UCS, CHOL, KICH)

  WHEN LR BEATS SVM:
    — When you need probability estimates (SVM only gives decision scores)
    — When the dataset is very noisy

  For cancer type classification from clean pathology reports:
  Both methods are competitive. The winner depends on the specific dataset split.
    
23:08:45  [INFO]  --- Hard cases: cancer types below 0.85 F1 in SVM ---
23:08:45  [INFO]    UCS (Uterine Carcinosarcoma) — F1=0.786, n_train=39
23:08:45  [INFO]    CHOL (Cholangiocarcinoma) — F1=0.833, n_train=30
23:08:45  [INFO]    These same types may remain challenging for BERT (Method 3).
23:08:45  [INFO]    RAG (Method 5) can help by retrieving confirmed similar cases.
23:08:45  [INFO]  ======================================================================
23:08:45  [INFO]    DONE
23:08:45  [INFO]  ======================================================================
23:08:45  [INFO]    Method 2 Test Accuracy : 97.37%
23:08:45  [INFO]    Method 2 F1 Weighted   : 97.35%
23:08:45  [INFO]    Total runtime          : 0.2 min
23:08:45  [INFO]    Log saved to           : c:\Users\kings\OneDrive\Desktop\CSUDH_Kingslee\CSUDH-Spring-2026\CSC-590-MastersProject\MastersProject\demo\ProjectCode\output\method2_tfidf_svm_run.log
23:08:45  [INFO]    NEXT: Run Method 4 (Embedding Similarity + k-NN).
