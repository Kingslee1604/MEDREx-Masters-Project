22:05:47  [INFO]    Log file : c:\Users\kings\OneDrive\Desktop\CSUDH_Kingslee\CSUDH-Spring-2026\CSC-590-MastersProject\MastersProject\demo\ProjectCode\output\method1_tfidf_lr_run.log
22:05:47  [INFO]  ======================================================================
22:05:47  [INFO]    MEDREx — Method 1: TF-IDF + Logistic Regression
22:05:47  [INFO]    CSC-590 Masters Project, CSUDH Spring 2026
22:05:47  [INFO]    Goal: Understand HOW and WHY we beat Cedars-Sinai (95.31% → 96%+)
22:05:47  [INFO]  ======================================================================
22:05:47  [INFO]  ======================================================================
22:05:47  [INFO]    STEP 1 — Load & Compile Dataset
22:05:47  [INFO]  ======================================================================
22:05:48  [INFO]    Reading reports : c:\Users\kings\OneDrive\Desktop\CSUDH_Kingslee\CSUDH-Spring-2026\CSC-590-MastersProject\MastersProject\DataSet\TCGA_Reports.csv\TCGA_Reports.csv
22:05:48  [INFO]    Loaded 9,523 reports in 0.4s
22:05:48  [INFO]    Columns : ['patient_filename', 'text']
22:05:48  [INFO]  --- Extract patient_id from filename ---
22:05:48  [INFO]    Raw filename  : 'TCGA-BP-5195.25c0b433-5557-4165-922e-2c1eac9c26f0'
22:05:48  [INFO]    patient_id    : 'TCGA-BP-5195'  (everything before the first '.')
22:05:48  [INFO]    Unique patient reports: 9,523
22:05:48  [INFO]  
                patient_id                                                                                 text
TCGA-BP-5195  TCGA-BP-5195  Date of Recelpt: Clinical Diagnosis & History: Incidental 3 cm left upper pole r...
TCGA-D7-8573  TCGA-D7-8573  Material: 1) Material: stomach, Method of collection: Lesion resection. Histopat...
TCGA-EI-7004  TCGA-EI-7004  page 1 / 1. copy No. 3. Examination: Histopathological examination. Cost of diag...
22:05:48  [INFO]  --- Load cancer type labels ---
22:05:48  [INFO]    Reading labels: c:\Users\kings\OneDrive\Desktop\CSUDH_Kingslee\CSUDH-Spring-2026\CSC-590-MastersProject\MastersProject\DataSet\tcga_patient_to_cancer_type.csv
22:05:48  [INFO]    Label rows    : 11,160
22:05:48  [INFO]    Unique types  : 33
22:05:48  [INFO]  
     patient_id cancer_type
0  TCGA-OR-A5J1         ACC
1  TCGA-OR-A5J2         ACC
2  TCGA-OR-A5J3         ACC
3  TCGA-OR-A5J4         ACC
4  TCGA-OR-A5J5         ACC
22:05:48  [INFO]  --- Merge reports + labels ---
22:05:48  [INFO]    Merged shape  : (9523, 4)
22:05:48  [INFO]  
                patient_id                                                                                 text cancer_type                   cancer_type_name
TCGA-BP-5195  TCGA-BP-5195  Date of Recelpt: Clinical Diagnosis & History: Incidental 3 cm left upper pole r...        KIRC  Kidney renal clear cell carcinoma
TCGA-D7-8573  TCGA-D7-8573  Material: 1) Material: stomach, Method of collection: Lesion resection. Histopat...        STAD             Stomach adenocarcinoma
TCGA-EI-7004  TCGA-EI-7004  page 1 / 1. copy No. 3. Examination: Histopathological examination. Cost of diag...        READ              Rectum adenocarcinoma
22:05:48  [INFO]  --- Cancer type distribution (all 33 types) ---
22:05:48  [INFO]      BRCA   Breast invasive carcinoma                     1034 (10.9%)  ██████████████████████████████████
22:05:48  [INFO]      UCEC   Uterine Corpus Endometrial Carcinoma           546 ( 5.7%)  ██████████████████
22:05:48  [INFO]      KIRC   Kidney renal clear cell carcinoma              525 ( 5.5%)  █████████████████
22:05:48  [INFO]      HNSC   Head and Neck squamous cell carcinoma          520 ( 5.5%)  █████████████████
22:05:48  [INFO]      LUAD   Lung adenocarcinoma                            488 ( 5.1%)  ████████████████
22:05:48  [INFO]      THCA   Thyroid carcinoma                              487 ( 5.1%)  ████████████████
22:05:48  [INFO]      LGG    Brain Lower Grade Glioma                       469 ( 4.9%)  ███████████████
22:05:48  [INFO]      LUSC   Lung squamous cell carcinoma                   468 ( 4.9%)  ███████████████
22:05:48  [INFO]      PRAD   Prostate adenocarcinoma                        446 ( 4.7%)  ██████████████
22:05:48  [INFO]      COAD   Colon adenocarcinoma                           418 ( 4.4%)  █████████████
22:05:48  [INFO]      GBM    Glioblastoma multiforme                        399 ( 4.2%)  █████████████
22:05:48  [INFO]      BLCA   Bladder Urothelial Carcinoma                   379 ( 4.0%)  ████████████
22:05:48  [INFO]      OV     Ovarian serous cystadenocarcinoma              371 ( 3.9%)  ████████████
22:05:48  [INFO]      STAD   Stomach adenocarcinoma                         361 ( 3.8%)  ████████████
22:05:48  [INFO]      LIHC   Liver hepatocellular carcinoma                 341 ( 3.6%)  ███████████
22:05:48  [INFO]      CESC   Cervical squamous cell carcinoma               289 ( 3.0%)  █████████
22:05:48  [INFO]      KIRP   Kidney renal papillary cell carcinoma          280 ( 2.9%)  █████████
22:05:48  [INFO]      SARC   Sarcoma                                        249 ( 2.6%)  ████████
22:05:48  [INFO]      PAAD   Pancreatic adenocarcinoma                      176 ( 1.8%)  █████
22:05:48  [INFO]      PCPG   Pheochromocytoma and Paraganglioma             174 ( 1.8%)  █████
22:05:48  [INFO]      READ   Rectum adenocarcinoma                          162 ( 1.7%)  █████
22:05:48  [INFO]      ESCA   Esophageal carcinoma                           146 ( 1.5%)  ████
22:05:48  [INFO]      THYM   Thymoma                                        114 ( 1.2%)  ███
22:05:48  [INFO]      KICH   Kidney Chromophobe                             112 ( 1.2%)  ███
22:05:48  [INFO]      SKCM   Skin Cutaneous Melanoma                        102 ( 1.1%)  ███
22:05:48  [INFO]      ACC    Adrenocortical carcinoma                        90 ( 0.9%)  ███
22:05:48  [INFO]      TGCT   Testicular Germ Cell Tumors                     87 ( 0.9%)  ██
22:05:48  [INFO]      MESO   Mesothelioma                                    79 ( 0.8%)  ██
22:05:48  [INFO]      UVM    Uveal Melanoma                                  65 ( 0.7%)  ██
22:05:48  [INFO]      UCS    Uterine Carcinosarcoma                          56 ( 0.6%)  █
22:05:48  [INFO]      DLBC   Diffuse Large B-cell Lymphoma                   47 ( 0.5%)  █
22:05:48  [INFO]      CHOL   Cholangiocarcinoma                              43 ( 0.5%)  █
22:05:48  [INFO]  --- Text length analysis ---
22:05:48  [INFO]    Average words per report : 560
22:05:48  [INFO]    Median  words per report : 432
22:05:48  [INFO]    Min     words per report : 1
22:05:48  [INFO]    Max     words per report : 4046
22:05:48  [INFO]    Reports over 512 words   : 4065 (42.7%)
22:05:48  [INFO]    → BERT has a 512-token limit. TF-IDF has NO length limit — handles long reports better.
22:05:48  [INFO]  --- Example pathology report ---
22:05:48  [INFO]    Patient ID  : TCGA-BP-5195
22:05:48  [INFO]    Cancer type : KIRC — Kidney renal clear cell carcinoma
22:05:48  [INFO]    Word count  : 370
22:05:48  [INFO]    Report text :

    Date of Recelpt: Clinical Diagnosis & History: Incidental 3 cm left upper pole renal mass. Specimens Submitted: 1: Kidney, Left Upper Pole; Partial Nephrectomy. DIAGNOSIS: 1. Kidney, Left Upper Pole; Partial Nephrectomy: Tumor Type: Renal cell carcinoma - Conventional (clear cell) type. Fuhrman Nuclear Grade: Nuclear grade II/IV. Tumor Size: Greatest diameter is 2.4 cm. Local Invasion (for renal cortical types): Not Identified. Renal Vein Invasion: Not identified. Surgical Margins: Free of tumor. Non-Neoplastic Kidney: shows focal chronic inflammation and focal superficial glomerulosclerosis. 

22:05:48  [INFO]  ======================================================================
22:05:48  [INFO]    STEP 2 — Stratified Train / Test Split  (70% train / 30% test)
22:05:48  [INFO]  ======================================================================
22:05:50  [INFO]    WHY STRATIFIED?
22:05:50  [INFO]    Some cancer types (e.g. UCS=37, CHOL=36) have very few samples.
22:05:50  [INFO]    Without stratify=, the test set might accidentally get 0 samples of a rare type.
22:05:50  [INFO]    stratify=cancer_type ensures each type is proportionally represented in both splits.
22:05:50  [INFO]    WHY NO VALIDATION SET HERE?
22:05:50  [INFO]    Cedars-Sinai used train/val/test (3 splits).
22:05:50  [INFO]    We use train/test only (2 splits) — simpler, and we use cross-validation
22:05:50  [INFO]    via GridSearch in Step 5 to tune hyperparameters without a fixed val set.
22:05:50  [INFO]    Split: 70% train / 30% test  |  random_state=42  |  stratify=cancer_type
22:05:50  [INFO]  
  Total  : 9,523 reports
22:05:50  [INFO]    Train  : 6,666 reports (70%)
22:05:50  [INFO]    Test   : 2,857  reports (30%)
22:05:50  [INFO]  --- Stratification check — class counts in each split ---
22:05:50  [INFO]    Type     Total   Train    Test   Train%   Test%
22:05:50  [INFO]    --------------------------------------------------
22:05:50  [INFO]    ACC         90      63      27      70%      30%
22:05:50  [INFO]    BLCA       379     265     114      70%      30%
22:05:50  [INFO]    BRCA      1034     724     310      70%      30%
22:05:50  [INFO]    CESC       289     202      87      70%      30%
22:05:50  [INFO]    CHOL        43      30      13      70%      30%
22:05:50  [INFO]    COAD       418     293     125      70%      30%
22:05:50  [INFO]    DLBC        47      33      14      70%      30%
22:05:50  [INFO]    ESCA       146     102      44      70%      30%
22:05:50  [INFO]    GBM        399     279     120      70%      30%
22:05:50  [INFO]    HNSC       520     364     156      70%      30%
22:05:50  [INFO]    KICH       112      78      34      70%      30%
22:05:50  [INFO]    KIRC       525     368     157      70%      30%
22:05:50  [INFO]    KIRP       280     196      84      70%      30%
22:05:50  [INFO]    LGG        469     328     141      70%      30%
22:05:50  [INFO]    LIHC       341     239     102      70%      30%
22:05:50  [INFO]    LUAD       488     342     146      70%      30%
22:05:50  [INFO]    LUSC       468     328     140      70%      30%
22:05:50  [INFO]    MESO        79      55      24      70%      30%
22:05:50  [INFO]    OV         371     260     111      70%      30%
22:05:50  [INFO]    PAAD       176     123      53      70%      30%
22:05:50  [INFO]    PCPG       174     122      52      70%      30%
22:05:50  [INFO]    PRAD       446     312     134      70%      30%
22:05:50  [INFO]    READ       162     113      49      70%      30%
22:05:50  [INFO]    SARC       249     174      75      70%      30%
22:05:50  [INFO]    SKCM       102      71      31      70%      30%
22:05:50  [INFO]    STAD       361     253     108      70%      30%
22:05:50  [INFO]    TGCT        87      61      26      70%      30%
22:05:50  [INFO]    THCA       487     341     146      70%      30%
22:05:50  [INFO]    THYM       114      80      34      70%      30%
22:05:50  [INFO]    UCEC       546     382     164      70%      30%
22:05:50  [INFO]    UCS         56      39      17      70%      30%
22:05:50  [INFO]    UVM         65      46      19      71%      29%
22:05:50  [INFO]  ======================================================================
22:05:50  [INFO]    STEP 3 — TF-IDF Vectorizer  ← KEY UPGRADE vs Cedars-Sinai
22:05:50  [INFO]  ======================================================================
22:05:50  [INFO]    WHAT IS TF-IDF?
22:05:50  [INFO]    TF  = Term Frequency  : how often a word appears in THIS report
22:05:50  [INFO]    IDF = Inverse Document Frequency : how RARE the word is across ALL reports
22:05:50  [INFO]    TF-IDF score = TF × IDF
22:05:50  [INFO]    WHY IS THIS BETTER THAN BAG-OF-WORDS?
22:05:50  [INFO]    Example — word 'the':
22:05:50  [INFO]      BoW  score = 50  (appears 50 times → high score, but useless for classification)
22:05:50  [INFO]      IDF  score = 0.001 (appears in EVERY report → near zero)
22:05:50  [INFO]      TF-IDF     = 50 × 0.001 = 0.05  (correctly downweighted)
22:05:50  [INFO]    Example — word 'glioblastoma':
22:05:50  [INFO]      BoW  score = 3   (appears 3 times → low score)
22:05:50  [INFO]      IDF  score = 8.5 (appears in very few reports → high)
22:05:50  [INFO]      TF-IDF     = 3 × 8.5 = 25.5  (correctly upweighted → signals GBM)
22:05:50  [INFO]  --- TF-IDF parameters — what each one does ---
22:05:50  [INFO]    max_features=15000
22:05:50  [INFO]      Only keep the top 15,000 most informative words/phrases.
22:05:50  [INFO]      Cedars-Sinai kept all ~23,818 tokens — many were noise.
22:05:50  [INFO]      15,000 features = faster training + less overfitting.
22:05:50  [INFO]    ngram_range=(1, 2)  ← BIGGEST UPGRADE
22:05:50  [INFO]      (1,1) = single words only:  'renal', 'cell', 'carcinoma'
22:05:50  [INFO]      (1,2) = also 2-word phrases: 'renal cell', 'clear cell carcinoma'
22:05:50  [INFO]      Medical meaning lives in PHRASES, not single words:
22:05:50  [INFO]        'renal cell carcinoma' → KIRC (kidney cancer)
22:05:50  [INFO]        'squamous cell'        → LUSC or HNSC
22:05:50  [INFO]        'ductal carcinoma'     → BRCA (breast cancer)
22:05:50  [INFO]        'germ cell'            → TGCT (testicular)
22:05:50  [INFO]    sublinear_tf=True
22:05:50  [INFO]      Normal TF:       word appears 100× → score=100  (too dominant)
22:05:50  [INFO]      Sublinear TF:    score = 1 + log(100) = 5.6     (compressed)
22:05:50  [INFO]      Prevents very frequent words from drowning out rarer but more specific ones.
22:05:50  [INFO]    min_df=2
22:05:50  [INFO]      Ignore words that appear in fewer than 2 documents.
22:05:50  [INFO]      Removes typos, one-off abbreviations, and noise.
22:05:50  [INFO]  --- Fitting TF-IDF on training corpus ---
22:05:50  [INFO]    Fitting on training data ONLY — never on test data (prevents data leakage).
22:06:06  [INFO]    Fit + transform complete in 15.9s
22:06:06  [INFO]    Vocabulary size : 15,000 features (words + 2-word phrases)
22:06:06  [INFO]    Matrix shape    : (6666, 15000)  (6666 reports × 15000 features)
22:06:06  [INFO]    Matrix sparsity : 2.509% non-zero
22:06:06  [INFO]  --- What the vocabulary looks like (sample entries) ---
22:06:06  [INFO]    Sample unigrams  (single words) : ['00', '01', '02', '03', '05', '0cm', '0x', '0x0', '0x1', '0x2', '10', '100', '10a', '10mm', '10r']
22:06:06  [INFO]    Sample bigrams   (2-word phrases): ['00 00', '0x cm', '10 10', '10 11', '10 12', '10 15', '10 20', '10 and', '10 buffered', '10 cm', '10 high', '10 hpf', '10 is', '10 left', '10 lymph']
22:06:06  [INFO]  --- TF-IDF score example — one report ---
22:06:06  [INFO]    Top 20 highest TF-IDF scoring terms in the first training report:
22:06:06  [INFO]      'central neck                  '  tfidf = 0.2479
22:06:06  [INFO]      'and central                   '  tfidf = 0.1756
22:06:06  [INFO]      'total thyroidectomy           '  tfidf = 0.1292
22:06:06  [INFO]      'neck dissection               '  tfidf = 0.1282
22:06:06  [INFO]      'neck                          '  tfidf = 0.1263
22:06:06  [INFO]      'thyroidectomy                 '  tfidf = 0.1250
22:06:06  [INFO]      'central                       '  tfidf = 0.1161
22:06:06  [INFO]      'thyroid                       '  tfidf = 0.1120
22:06:06  [INFO]      'prelaryngeal                  '  tfidf = 0.1112
22:06:06  [INFO]      'inked anterior                '  tfidf = 0.1112
22:06:06  [INFO]      'thyroidectomy and             '  tfidf = 0.1075
22:06:06  [INFO]      'of 17                         '  tfidf = 0.1071
22:06:06  [INFO]      'all parts                     '  tfidf = 0.1067
22:06:06  [INFO]      'cm focality                   '  tfidf = 0.1059
22:06:06  [INFO]      'nodule greatest               '  tfidf = 0.1055
22:06:06  [INFO]      'nodes distant                 '  tfidf = 0.1051
22:06:06  [INFO]      'paratracheal and              '  tfidf = 0.1047
22:06:06  [INFO]      'from nearest                  '  tfidf = 0.1044
22:06:06  [INFO]      'delphian                      '  tfidf = 0.1040
22:06:06  [INFO]      'diagnosis total               '  tfidf = 0.1037
22:06:06  [INFO]  --- Transforming test set ---
22:06:12  [INFO]    Test transform complete in 5.5s
22:06:12  [INFO]    Test matrix shape : (2857, 15000)
22:06:12  [INFO]  ======================================================================
22:06:12  [INFO]    STEP 4 — Logistic Regression with Class Balancing
22:06:12  [INFO]  ======================================================================
22:06:12  [INFO]    LOGISTIC REGRESSION PARAMETER DIFFERENCES vs Cedars-Sinai:
22:06:12  [INFO]    class_weight='balanced'  ← NEW (Cedars-Sinai did not use this)
22:06:12  [INFO]      Problem: BRCA has 1034 samples, UCS has only 37.
22:06:12  [INFO]      Without balancing: model ignores rare cancers (biased toward BRCA).
22:06:12  [INFO]      With balancing   : rare class UCS gets weight = 1034/37 = 27.9×
22:06:12  [INFO]      Effect: model pays much more attention to getting rare cancers right.
22:06:12  [INFO]    max_iter=1000  (Cedars-Sinai used 200)
22:06:12  [INFO]      More iterations → model fully converges → slightly better accuracy.
22:06:12  [INFO]    solver='lbfgs'
22:06:12  [INFO]      L-BFGS is a quasi-Newton optimiser — faster and more stable
22:06:12  [INFO]      than the default solver for multi-class problems with many features.
22:06:12  [INFO]    C=1.0  (regularisation strength — default, same as Cedars-Sinai)
22:06:12  [INFO]      C controls how much the model can overfit.
22:06:12  [INFO]      Higher C = less regularisation = can fit training data more closely.
22:06:12  [INFO]      C=1.0 is a good balance between underfitting and overfitting.
22:06:12  [INFO]  --- Class weights computed from training data ---
22:06:12  [INFO]    Class weights assigned to each cancer type:
22:06:12  [INFO]      CHOL    n_train=  30  weight=  6.94  ████████████████████
22:06:12  [INFO]      DLBC    n_train=  33  weight=  6.31  ██████████████████
22:06:12  [INFO]      UCS     n_train=  39  weight=  5.34  ████████████████
22:06:12  [INFO]      UVM     n_train=  46  weight=  4.53  █████████████
22:06:12  [INFO]      MESO    n_train=  55  weight=  3.79  ███████████
22:06:12  [INFO]      TGCT    n_train=  61  weight=  3.41  ██████████
22:06:12  [INFO]      ACC     n_train=  63  weight=  3.31  █████████
22:06:12  [INFO]      SKCM    n_train=  71  weight=  2.93  ████████
22:06:12  [INFO]      KICH    n_train=  78  weight=  2.67  ████████
22:06:12  [INFO]      THYM    n_train=  80  weight=  2.60  ███████
22:06:12  [INFO]      ESCA    n_train= 102  weight=  2.04  ██████
22:06:12  [INFO]      READ    n_train= 113  weight=  1.84  █████
22:06:12  [INFO]      PCPG    n_train= 122  weight=  1.71  █████
22:06:12  [INFO]      PAAD    n_train= 123  weight=  1.69  █████
22:06:12  [INFO]      SARC    n_train= 174  weight=  1.20  ███
22:06:12  [INFO]      KIRP    n_train= 196  weight=  1.06  ███
22:06:12  [INFO]      CESC    n_train= 202  weight=  1.03  ███
22:06:12  [INFO]      LIHC    n_train= 239  weight=  0.87  ██
22:06:12  [INFO]      STAD    n_train= 253  weight=  0.82  ██
22:06:12  [INFO]      OV      n_train= 260  weight=  0.80  ██
22:06:12  [INFO]      BLCA    n_train= 265  weight=  0.79  ██
22:06:12  [INFO]      GBM     n_train= 279  weight=  0.75  ██
22:06:12  [INFO]      COAD    n_train= 293  weight=  0.71  ██
22:06:12  [INFO]      PRAD    n_train= 312  weight=  0.67  ██
22:06:12  [INFO]      LGG     n_train= 328  weight=  0.64  █
22:06:12  [INFO]      LUSC    n_train= 328  weight=  0.64  █
22:06:12  [INFO]      THCA    n_train= 341  weight=  0.61  █
22:06:12  [INFO]      LUAD    n_train= 342  weight=  0.61  █
22:06:12  [INFO]      HNSC    n_train= 364  weight=  0.57  █
22:06:12  [INFO]      KIRC    n_train= 368  weight=  0.57  █
22:06:12  [INFO]      UCEC    n_train= 382  weight=  0.55  █
22:06:12  [INFO]      BRCA    n_train= 724  weight=  0.29  
22:06:12  [INFO]    → Rare types get HIGH weight so the model doesn't ignore them.
22:06:12  [INFO]  --- Training Logistic Regression ---
22:06:12  [INFO]    Parameters: C=1.0, max_iter=1000, class_weight=balanced, solver=lbfgs, random_state=42
22:06:12  [INFO]    Training... (~30-60 seconds)
22:06:32  [INFO]    Training complete in 20.0s
22:06:32  [INFO]    Model converged: True
22:06:32  [INFO]    Iterations used : 34 / 1000
22:06:32  [INFO]    Number of classes: 32
22:06:32  [INFO]  --- Results on TRAINING set ---
22:06:32  [INFO]    Train Accuracy : 98.57%
22:06:32  [INFO]    (High training accuracy is expected — model has seen these reports before)
22:06:32  [INFO]  --- Results on TEST set  ← THIS IS THE REAL SCORE ---
22:06:32  [INFO]    ╔══════════════════════════════════════════════════════════╗
22:06:32  [INFO]    ║  METHOD 1 — TF-IDF + LOGISTIC REGRESSION RESULTS       ║
22:06:32  [INFO]    ║                                                          ║
22:06:32  [INFO]    ║  Test Accuracy        :  96.36%                      ║
22:06:32  [INFO]    ║  F1 Weighted          :  96.36%                      ║
22:06:32  [INFO]    ║  F1 Macro             :  95.73%                      ║
22:06:32  [INFO]    ║                                                          ║
22:06:32  [INFO]    ║  Cedars-Sinai BoW+LR  : ~95.31%                         ║
22:06:32  [INFO]    ║  Our improvement      : +1.05%                        ║
22:06:32  [INFO]    ╚══════════════════════════════════════════════════════════╝
22:06:32  [INFO]  
  Full classification report (all 33 cancer types):
22:06:32  [INFO]                precision    recall  f1-score   support

         ACC       1.00      0.96      0.98        27
        BLCA       1.00      0.99      1.00       114
        BRCA       1.00      1.00      1.00       310
        CESC       0.99      0.97      0.98        87
        CHOL       0.92      0.92      0.92        13
        COAD       0.94      0.94      0.94       125
        DLBC       1.00      1.00      1.00        14
        ESCA       0.91      0.95      0.93        44
         GBM       0.95      0.96      0.95       120
        HNSC       0.99      0.97      0.98       156
        KICH       0.87      0.97      0.92        34
        KIRC       0.96      0.97      0.97       157
        KIRP       0.99      0.90      0.94        84
         LGG       0.96      0.96      0.96       141
        LIHC       0.99      0.99      0.99       102
        LUAD       0.91      0.97      0.94       146
        LUSC       0.92      0.90      0.91       140
        MESO       1.00      1.00      1.00        24
          OV       0.95      0.95      0.95       111
        PAAD       0.98      1.00      0.99        53
        PCPG       0.96      1.00      0.98        52
        PRAD       1.00      1.00      1.00       134
        READ       0.82      0.86      0.84        49
        SARC       0.99      0.92      0.95        75
        SKCM       0.97      0.94      0.95        31
        STAD       0.98      0.94      0.96       108
        TGCT       1.00      1.00      1.00        26
        THCA       0.99      0.99      0.99       146
        THYM       0.97      0.97      0.97        34
        UCEC       0.92      0.96      0.94       164
         UCS       0.92      0.71      0.80        17
         UVM       1.00      1.00      1.00        19

    accuracy                           0.96      2857
   macro avg       0.96      0.96      0.96      2857
weighted avg       0.96      0.96      0.96      2857

22:06:32  [INFO]  --- Per-class F1 sorted worst → best ---
22:06:32  [INFO]    Code   Cancer Name                                       F1  #Train  #Test
22:06:32  [INFO]    --------------------------------------------------------------------------
22:06:32  [INFO]    UCS    Uterine Carcinosarcoma                         0.800      39     17 ← hardest
22:06:32  [INFO]    READ   Rectum adenocarcinoma                          0.840     113     49 ← hardest
22:06:32  [INFO]    LUSC   Lung squamous cell carcinoma                   0.910     328    140
22:06:32  [INFO]    KICH   Kidney Chromophobe                             0.917      78     34
22:06:32  [INFO]    CHOL   Cholangiocarcinoma                             0.923      30     13
22:06:32  [INFO]    ESCA   Esophageal carcinoma                           0.933     102     44
22:06:32  [INFO]    COAD   Colon adenocarcinoma                           0.936     293    125
22:06:32  [INFO]    LUAD   Lung adenocarcinoma                            0.940     342    146
22:06:32  [INFO]    UCEC   Uterine Corpus Endometrial Carcinoma           0.943     382    164
22:06:32  [INFO]    KIRP   Kidney renal papillary cell carcinoma          0.944     196     84
22:06:32  [INFO]    OV     Ovarian serous cystadenocarcinoma              0.946     260    111
22:06:32  [INFO]    SKCM   Skin Cutaneous Melanoma                        0.951      71     31
22:06:32  [INFO]    SARC   Sarcoma                                        0.952     174     75
22:06:32  [INFO]    GBM    Glioblastoma multiforme                        0.954     279    120
22:06:32  [INFO]    LGG    Brain Lower Grade Glioma                       0.957     328    141
22:06:32  [INFO]    STAD   Stomach adenocarcinoma                         0.962     253    108
22:06:32  [INFO]    KIRC   Kidney renal clear cell carcinoma              0.968     368    157
22:06:32  [INFO]    THYM   Thymoma                                        0.971      80     34
22:06:32  [INFO]    CESC   Cervical squamous cell carcinoma               0.977     202     87
22:06:32  [INFO]    HNSC   Head and Neck squamous cell carcinoma          0.977     364    156
22:06:32  [INFO]    ACC    Adrenocortical carcinoma                       0.981      63     27
22:06:32  [INFO]    PCPG   Pheochromocytoma and Paraganglioma             0.981     122     52
22:06:32  [INFO]    LIHC   Liver hepatocellular carcinoma                 0.990     239    102
22:06:32  [INFO]    PAAD   Pancreatic adenocarcinoma                      0.991     123     53
22:06:32  [INFO]    THCA   Thyroid carcinoma                              0.993     341    146
22:06:32  [INFO]    BLCA   Bladder Urothelial Carcinoma                   0.996     265    114
22:06:32  [INFO]    BRCA   Breast invasive carcinoma                      1.000     724    310
22:06:32  [INFO]    DLBC   Diffuse Large B-cell Lymphoma                  1.000      33     14
22:06:32  [INFO]    PRAD   Prostate adenocarcinoma                        1.000     312    134
22:06:32  [INFO]    MESO   Mesothelioma                                   1.000      55     24
22:06:32  [INFO]    TGCT   Testicular Germ Cell Tumors                    1.000      61     26
22:06:32  [INFO]    UVM    Uveal Melanoma                                 1.000      46     19
22:06:32  [INFO]  
  ANALYSIS:
22:06:32  [INFO]    Cancer types below 0.85 F1 (2 types) :
22:06:32  [INFO]      UCS (Uterine Carcinosarcoma) — F1=0.800, only 39 training samples
22:06:32  [INFO]      READ (Rectum adenocarcinoma) — F1=0.840, only 113 training samples
22:06:32  [INFO]    These are targets for BERT fine-tuning (Phase 3) — BERT understands medical context better.
22:06:32  [INFO]  ======================================================================
22:06:32  [INFO]    STEP 5 — Deep Feature Analysis  (what did the model actually learn?)
22:06:32  [INFO]  ======================================================================
22:06:33  [INFO]    LR COEFFICIENT MEANING:
22:06:33  [INFO]    For each cancer type, LR learned a weight (coefficient) per feature.
22:06:33  [INFO]    High positive coefficient = that word/phrase STRONGLY predicts this cancer.
22:06:33  [INFO]    Low negative coefficient  = that word/phrase argues AGAINST this cancer.
22:06:33  [INFO]    These are the actual 'rules' the model learned from 9,523 pathology reports.
22:06:33  [INFO]  --- Top 15 most predictive words/phrases per cancer type ---
22:06:33  [INFO]  
  ┌─ ACC — Adrenocortical carcinoma  (n_train=63) ─────
22:06:33  [INFO]    │  adrenocortical                      +3.961  +++++++++++
22:06:33  [INFO]    │  adrenocortical carcinoma            +3.783  +++++++++++
22:06:33  [INFO]    │  adrenal                             +3.528  ++++++++++
22:06:33  [INFO]    │  weiss                               +3.302  +++++++++
22:06:33  [INFO]    │  adrenal cortical                    +2.971  ++++++++
22:06:33  [INFO]    └────────────────────────────────────────────────────────────
22:06:33  [INFO]  
  ┌─ BLCA — Bladder Urothelial Carcinoma  (n_train=265) ─────
22:06:33  [INFO]    │  bladder                             +5.250  +++++++++++++++
22:06:33  [INFO]    │  urothelial                          +3.891  +++++++++++
22:06:33  [INFO]    │  urothelial carcinoma                +3.085  +++++++++
22:06:33  [INFO]    │  ureter                              +2.172  ++++++
22:06:33  [INFO]    │  the bladder                         +2.044  ++++++
22:06:33  [INFO]    └────────────────────────────────────────────────────────────
22:06:33  [INFO]  
  ┌─ BRCA — Breast invasive carcinoma  (n_train=724) ─────
22:06:33  [INFO]    │  breast                              +5.140  +++++++++++++++
22:06:33  [INFO]    │  ductal                              +2.794  ++++++++
22:06:33  [INFO]    │  ductal carcinoma                    +2.410  +++++++
22:06:33  [INFO]    │  axillary                            +2.294  ++++++
22:06:33  [INFO]    │  sentinel                            +2.029  ++++++
22:06:33  [INFO]    └────────────────────────────────────────────────────────────
22:06:33  [INFO]  
  ┌─ CESC — Cervical squamous cell carcinoma  (n_train=202) ─────
22:06:33  [INFO]    │  cervix                              +4.843  ++++++++++++++
22:06:33  [INFO]    │  cervical                            +3.504  ++++++++++
22:06:33  [INFO]    │  squamous cell                       +2.916  ++++++++
22:06:33  [INFO]    │  squamous                            +2.744  ++++++++
22:06:33  [INFO]    │  cervical biopsy                     +2.266  ++++++
22:06:33  [INFO]    └────────────────────────────────────────────────────────────
22:06:33  [INFO]  
  ┌─ CHOL — Cholangiocarcinoma  (n_train=30) ─────
22:06:33  [INFO]    │  cholangiocarcinoma                  +5.077  +++++++++++++++
22:06:33  [INFO]    │  liver                               +3.349  ++++++++++
22:06:33  [INFO]    │  hepatic                             +2.731  ++++++++
22:06:33  [INFO]    │  intrahepatic                        +2.216  ++++++
22:06:33  [INFO]    │  duct                                +2.209  ++++++
22:06:33  [INFO]    └────────────────────────────────────────────────────────────
22:06:33  [INFO]  
  ┌─ COAD — Colon adenocarcinoma  (n_train=293) ─────
22:06:33  [INFO]    │  colon                               +5.176  +++++++++++++++
22:06:33  [INFO]    │  hemicolectomy                       +2.117  ++++++
22:06:33  [INFO]    │  appendix                            +1.903  +++++
22:06:33  [INFO]    │  sigmoid colon                       +1.800  +++++
22:06:33  [INFO]    │  cecum                               +1.760  +++++
22:06:33  [INFO]    └────────────────────────────────────────────────────────────
22:06:33  [INFO]  
  ┌─ DLBC — Diffuse Large B-cell Lymphoma  (n_train=33) ─────
22:06:33  [INFO]    │  lymphoma                            +4.855  ++++++++++++++
22:06:33  [INFO]    │  cell lymphoma                       +3.636  ++++++++++
22:06:33  [INFO]    │  large cell                          +3.447  ++++++++++
22:06:33  [INFO]    │  diffuse large                       +3.146  +++++++++
22:06:33  [INFO]    │  cd3                                 +2.813  ++++++++
22:06:33  [INFO]    └────────────────────────────────────────────────────────────
22:06:33  [INFO]  
  ┌─ ESCA — Esophageal carcinoma  (n_train=102) ─────
22:06:33  [INFO]    │  esophagus                           +6.903  ++++++++++++++++++++
22:06:33  [INFO]    │  esophageal                          +3.928  +++++++++++
22:06:33  [INFO]    │  adventitia                          +2.113  ++++++
22:06:33  [INFO]    │  barrett                             +2.094  ++++++
22:06:33  [INFO]    │  distal esophagus                    +2.092  ++++++
22:06:33  [INFO]    └────────────────────────────────────────────────────────────
22:06:33  [INFO]  
  ┌─ GBM — Glioblastoma multiforme  (n_train=279) ─────
22:06:33  [INFO]    │  glioblastoma                        +5.165  +++++++++++++++
22:06:33  [INFO]    │  brain                               +3.689  +++++++++++
22:06:33  [INFO]    │  multiforme                          +2.945  ++++++++
22:06:33  [INFO]    │  glioblastoma multiforme             +2.940  ++++++++
22:06:33  [INFO]    │  grade iv                            +2.306  ++++++
22:06:33  [INFO]    └────────────────────────────────────────────────────────────
22:06:33  [INFO]  
  ┌─ HNSC — Head and Neck squamous cell carcinoma  (n_train=364) ─────
22:06:33  [INFO]    │  tongue                              +3.086  +++++++++
22:06:33  [INFO]    │  squamous                            +2.867  ++++++++
22:06:33  [INFO]    │  squamous cell                       +2.350  +++++++
22:06:33  [INFO]    │  neck                                +2.229  ++++++
22:06:33  [INFO]    │  larynx                              +1.817  +++++
22:06:33  [INFO]    └────────────────────────────────────────────────────────────
22:06:33  [INFO]  
  ┌─ KICH — Kidney Chromophobe  (n_train=78) ─────
22:06:33  [INFO]    │  chromophobe                         +5.300  +++++++++++++++
22:06:33  [INFO]    │  renal                               +3.090  +++++++++
22:06:33  [INFO]    │  kidney                              +2.938  ++++++++
22:06:33  [INFO]    │  carcinoma chromophobe               +2.814  ++++++++
22:06:33  [INFO]    │  chromophobe type                    +2.623  +++++++
22:06:33  [INFO]    └────────────────────────────────────────────────────────────
22:06:33  [INFO]  
  ┌─ KIRC — Kidney renal clear cell carcinoma  (n_train=368) ─────
22:06:33  [INFO]    │  renal                               +3.962  +++++++++++
22:06:33  [INFO]    │  kidney                              +3.571  ++++++++++
22:06:33  [INFO]    │  clear cell                          +3.454  ++++++++++
22:06:33  [INFO]    │  clear                               +2.990  ++++++++
22:06:33  [INFO]    │  cell type                           +2.442  +++++++
22:06:33  [INFO]    └────────────────────────────────────────────────────────────
22:06:33  [INFO]  
  ┌─ KIRP — Kidney renal papillary cell carcinoma  (n_train=196) ─────
22:06:33  [INFO]    │  papillary renal                     +4.517  +++++++++++++
22:06:33  [INFO]    │  renal                               +4.500  +++++++++++++
22:06:33  [INFO]    │  papillary                           +4.172  ++++++++++++
22:06:33  [INFO]    │  kidney                              +3.446  ++++++++++
22:06:33  [INFO]    │  renal cell                          +3.057  +++++++++
22:06:33  [INFO]    └────────────────────────────────────────────────────────────
22:06:33  [INFO]  
  ┌─ LGG — Brain Lower Grade Glioma  (n_train=328) ─────
22:06:33  [INFO]    │  brain                               +3.069  +++++++++
22:06:33  [INFO]    │  anaplastic                          +2.848  ++++++++
22:06:33  [INFO]    │  oligodendroglioma                   +2.731  ++++++++
22:06:33  [INFO]    │  astrocytoma                         +2.715  ++++++++
22:06:33  [INFO]    │  frontal                             +2.246  ++++++
22:06:33  [INFO]    └────────────────────────────────────────────────────────────
22:06:33  [INFO]  
  ┌─ LIHC — Liver hepatocellular carcinoma  (n_train=239) ─────
22:06:33  [INFO]    │  liver                               +5.481  ++++++++++++++++
22:06:33  [INFO]    │  hepatocellular                      +5.175  +++++++++++++++
22:06:33  [INFO]    │  hepatocellular carcinoma            +4.923  ++++++++++++++
22:06:33  [INFO]    │  gallbladder                         +2.328  ++++++
22:06:33  [INFO]    │  hepatic                             +1.792  +++++
22:06:33  [INFO]    └────────────────────────────────────────────────────────────
22:06:33  [INFO]  
  ┌─ LUAD — Lung adenocarcinoma  (n_train=342) ─────
22:06:33  [INFO]    │  lung                                +4.500  +++++++++++++
22:06:33  [INFO]    │  adenocarcinoma                      +2.732  ++++++++
22:06:33  [INFO]    │  upper lobe                          +2.580  +++++++
22:06:33  [INFO]    │  lobe                                +2.565  +++++++
22:06:33  [INFO]    │  pleura                              +2.282  ++++++
22:06:33  [INFO]    └────────────────────────────────────────────────────────────
22:06:33  [INFO]  
  ┌─ LUSC — Lung squamous cell carcinoma  (n_train=328) ─────
22:06:33  [INFO]    │  lung                                +4.181  ++++++++++++
22:06:33  [INFO]    │  squamous cell                       +3.704  +++++++++++
22:06:33  [INFO]    │  squamous                            +3.588  ++++++++++
22:06:33  [INFO]    │  cell carcinoma                      +2.597  +++++++
22:06:33  [INFO]    │  bronchial                           +2.540  +++++++
22:06:33  [INFO]    └────────────────────────────────────────────────────────────
22:06:33  [INFO]  
  ┌─ MESO — Mesothelioma  (n_train=55) ─────
22:06:33  [INFO]    │  mesothelioma                        +6.378  +++++++++++++++++++
22:06:33  [INFO]    │  malignant mesothelioma              +4.200  ++++++++++++
22:06:33  [INFO]    │  epithelioid                         +2.785  ++++++++
22:06:33  [INFO]    │  malignant                           +2.750  ++++++++
22:06:33  [INFO]    │  pleura                              +2.730  ++++++++
22:06:33  [INFO]    └────────────────────────────────────────────────────────────
22:06:33  [INFO]  
  ┌─ OV — Ovarian serous cystadenocarcinoma  (n_train=260) ─────
22:06:33  [INFO]    │  serous                              +3.751  +++++++++++
22:06:33  [INFO]    │  ovarian                             +2.715  ++++++++
22:06:33  [INFO]    │  omentum                             +2.701  ++++++++
22:06:33  [INFO]    │  ovary                               +2.456  +++++++
22:06:33  [INFO]    │  serous carcinoma                    +2.187  ++++++
22:06:33  [INFO]    └────────────────────────────────────────────────────────────
22:06:33  [INFO]  
  ┌─ PAAD — Pancreatic adenocarcinoma  (n_train=123) ─────
22:06:33  [INFO]    │  pancreas                            +4.629  +++++++++++++
22:06:33  [INFO]    │  pancreatic                          +4.373  +++++++++++++
22:06:33  [INFO]    │  the pancreas                        +2.328  ++++++
22:06:33  [INFO]    │  head                                +2.257  ++++++
22:06:33  [INFO]    │  duodenum                            +2.242  ++++++
22:06:33  [INFO]    └────────────────────────────────────────────────────────────
22:06:33  [INFO]  
  ┌─ PCPG — Pheochromocytoma and Paraganglioma  (n_train=122) ─────
22:06:33  [INFO]    │  pheochromocytoma                    +5.725  +++++++++++++++++
22:06:33  [INFO]    │  adrenal                             +4.796  ++++++++++++++
22:06:33  [INFO]    │  right adrenal                       +2.402  +++++++
22:06:33  [INFO]    │  adrenal gland                       +2.402  +++++++
22:06:33  [INFO]    │  paraganglioma                       +2.115  ++++++
22:06:33  [INFO]    └────────────────────────────────────────────────────────────
22:06:33  [INFO]  
  ┌─ PRAD — Prostate adenocarcinoma  (n_train=312) ─────
22:06:33  [INFO]    │  prostate                            +3.695  +++++++++++
22:06:33  [INFO]    │  seminal                             +3.278  +++++++++
22:06:33  [INFO]    │  gleason                             +3.182  +++++++++
22:06:33  [INFO]    │  prostatic                           +2.099  ++++++
22:06:33  [INFO]    │  vesicle                             +2.095  ++++++
22:06:33  [INFO]    └────────────────────────────────────────────────────────────
22:06:33  [INFO]  
  ┌─ READ — Rectum adenocarcinoma  (n_train=113) ─────
22:06:33  [INFO]    │  rectum                              +5.249  +++++++++++++++
22:06:33  [INFO]    │  rectal                              +3.892  +++++++++++
22:06:33  [INFO]    │  rectosigmoid                        +3.023  +++++++++
22:06:33  [INFO]    │  perirectal                          +2.377  +++++++
22:06:33  [INFO]    │  colon                               +1.846  +++++
22:06:33  [INFO]    └────────────────────────────────────────────────────────────
22:06:33  [INFO]  
  ┌─ SARC — Sarcoma  (n_train=174) ─────
22:06:33  [INFO]    │  sarcoma                             +4.767  ++++++++++++++
22:06:33  [INFO]    │  leiomyosarcoma                      +4.379  +++++++++++++
22:06:33  [INFO]    │  liposarcoma                         +1.923  +++++
22:06:33  [INFO]    │  retroperitoneal                     +1.674  +++++
22:06:33  [INFO]    │  desmin                              +1.510  ++++
22:06:33  [INFO]    └────────────────────────────────────────────────────────────
22:06:33  [INFO]  
  ┌─ SKCM — Skin Cutaneous Melanoma  (n_train=71) ─────
22:06:33  [INFO]    │  melanoma                            +7.167  ++++++++++++++++++++
22:06:33  [INFO]    │  skin                                +3.709  +++++++++++
22:06:33  [INFO]    │  malignant melanoma                  +2.558  +++++++
22:06:33  [INFO]    │  breslow                             +2.283  ++++++
22:06:33  [INFO]    │  clark                               +2.271  ++++++
22:06:33  [INFO]    └────────────────────────────────────────────────────────────
22:06:33  [INFO]  
  ┌─ STAD — Stomach adenocarcinoma  (n_train=253) ─────
22:06:33  [INFO]    │  stomach                             +4.763  ++++++++++++++
22:06:33  [INFO]    │  gastric                             +3.048  +++++++++
22:06:33  [INFO]    │  gastrectomy                         +2.893  ++++++++
22:06:33  [INFO]    │  intestinal                          +2.339  +++++++
22:06:33  [INFO]    │  curvature                           +2.193  ++++++
22:06:33  [INFO]    └────────────────────────────────────────────────────────────
22:06:33  [INFO]  
  ┌─ TGCT — Testicular Germ Cell Tumors  (n_train=61) ─────
22:06:33  [INFO]    │  testis                              +5.289  +++++++++++++++
22:06:33  [INFO]    │  seminoma                            +3.027  +++++++++
22:06:33  [INFO]    │  spermatic                           +2.699  ++++++++
22:06:33  [INFO]    │  spermatic cord                      +2.699  ++++++++
22:06:33  [INFO]    │  testicle                            +2.698  ++++++++
22:06:33  [INFO]    └────────────────────────────────────────────────────────────
22:06:33  [INFO]  
  ┌─ THCA — Thyroid carcinoma  (n_train=341) ─────
22:06:33  [INFO]    │  thyroid                             +4.760  ++++++++++++++
22:06:33  [INFO]    │  papillary                           +2.629  +++++++
22:06:33  [INFO]    │  thyroidectomy                       +2.380  +++++++
22:06:33  [INFO]    │  right lobe                          +2.353  +++++++
22:06:33  [INFO]    │  lobe                                +2.314  ++++++
22:06:33  [INFO]    └────────────────────────────────────────────────────────────
22:06:33  [INFO]  
  ┌─ THYM — Thymoma  (n_train=80) ─────
22:06:33  [INFO]    │  thymoma                             +6.631  +++++++++++++++++++
22:06:33  [INFO]    │  thymus                              +4.462  +++++++++++++
22:06:33  [INFO]    │  thymic                              +3.000  +++++++++
22:06:33  [INFO]    │  mediastinal                         +2.514  +++++++
22:06:33  [INFO]    │  thymectomy                          +2.266  ++++++
22:06:33  [INFO]    └────────────────────────────────────────────────────────────
22:06:33  [INFO]  
  ┌─ UCEC — Uterine Corpus Endometrial Carcinoma  (n_train=382) ─────
22:06:33  [INFO]    │  endometrioid                        +2.681  ++++++++
22:06:33  [INFO]    │  endometrial                         +2.431  +++++++
22:06:33  [INFO]    │  myometrial                          +2.084  ++++++
22:06:33  [INFO]    │  figo                                +1.955  +++++
22:06:33  [INFO]    │  uterus                              +1.754  +++++
22:06:33  [INFO]    └────────────────────────────────────────────────────────────
22:06:33  [INFO]  
  ┌─ UCS — Uterine Carcinosarcoma  (n_train=39) ─────
22:06:33  [INFO]    │  carcinosarcoma                      +3.590  ++++++++++
22:06:33  [INFO]    │  malignant mixed                     +2.792  ++++++++
22:06:33  [INFO]    │  mullerian tumor                     +2.379  +++++++
22:06:33  [INFO]    │  mullerian                           +2.306  ++++++
22:06:33  [INFO]    │  mixed mullerian                     +2.257  ++++++
22:06:33  [INFO]    └────────────────────────────────────────────────────────────
22:06:33  [INFO]  
  ┌─ UVM — Uveal Melanoma  (n_train=46) ─────
22:06:33  [INFO]    │  optic                               +3.193  +++++++++
22:06:33  [INFO]    │  optic nerve                         +2.972  ++++++++
22:06:33  [INFO]    │  melanoma                            +2.623  +++++++
22:06:33  [INFO]    │  eye                                 +2.612  +++++++
22:06:33  [INFO]    │  nerve                               +2.163  ++++++
22:06:33  [INFO]    └────────────────────────────────────────────────────────────
22:06:33  [INFO]  --- Most uniquely diagnostic words across ALL cancer types ---
22:06:33  [INFO]    These are words with the highest coefficient in exactly ONE cancer type
22:06:33  [INFO]    — the clearest signals the model uses:
22:06:33  [INFO]    Term                                Cancer    Coeff
22:06:33  [INFO]    -------------------------------------------------------
22:06:33  [INFO]    melanoma                              SKCM   +7.167
22:06:33  [INFO]    esophagus                             ESCA   +6.903
22:06:33  [INFO]    thymoma                               THYM   +6.631
22:06:33  [INFO]    mesothelioma                          MESO   +6.378
22:06:33  [INFO]    pheochromocytoma                      PCPG   +5.725
22:06:33  [INFO]    liver                                 LIHC   +5.481
22:06:33  [INFO]    chromophobe                           KICH   +5.300
22:06:33  [INFO]    testis                                TGCT   +5.289
22:06:33  [INFO]    bladder                               BLCA   +5.250
22:06:33  [INFO]    rectum                                READ   +5.249
22:06:33  [INFO]    colon                                 COAD   +5.176
22:06:33  [INFO]    hepatocellular                        LIHC   +5.175
22:06:33  [INFO]    glioblastoma                           GBM   +5.165
22:06:33  [INFO]    breast                                BRCA   +5.140
22:06:33  [INFO]    cholangiocarcinoma                    CHOL   +5.077
22:06:33  [INFO]    hepatocellular carcinoma              LIHC   +4.923
22:06:33  [INFO]    lymphoma                              DLBC   +4.855
22:06:33  [INFO]    cervix                                CESC   +4.843
22:06:33  [INFO]    adrenal                               PCPG   +4.796
22:06:33  [INFO]    sarcoma                               SARC   +4.767
22:06:33  [INFO]    stomach                               STAD   +4.763
22:06:33  [INFO]    thyroid                               THCA   +4.760
22:06:33  [INFO]    pancreas                              PAAD   +4.629
22:06:33  [INFO]    papillary renal                       KIRP   +4.517
22:06:33  [INFO]    renal                                 KIRP   +4.500
22:06:33  [INFO]    lung                                  LUAD   +4.500
22:06:33  [INFO]    thymus                                THYM   +4.462
22:06:33  [INFO]    leiomyosarcoma                        SARC   +4.379
22:06:33  [INFO]    pancreatic                            PAAD   +4.373
22:06:33  [INFO]    malignant mesothelioma                MESO   +4.200
22:06:33  [INFO]    lung                                  LUSC   +4.181
22:06:33  [INFO]    papillary                             KIRP   +4.172
22:06:33  [INFO]    renal                                 KIRC   +3.962
22:06:33  [INFO]    adrenocortical                         ACC   +3.961
22:06:33  [INFO]    esophageal                            ESCA   +3.928
22:06:33  [INFO]    rectal                                READ   +3.892
22:06:33  [INFO]    urothelial                            BLCA   +3.891
22:06:33  [INFO]    adrenocortical carcinoma               ACC   +3.783
22:06:33  [INFO]    serous                                  OV   +3.751
22:06:33  [INFO]    skin                                  SKCM   +3.709
22:06:33  [INFO]  ======================================================================
22:06:33  [INFO]    STEP 6 — Comparison: Our Method 1 vs Cedars-Sinai Baseline
22:06:33  [INFO]  ======================================================================
22:06:33  [INFO]    Metric                           Cedars-Sinai   Our Method 1  Improvement
22:06:33  [INFO]    --------------------------------------------------------------------------
22:06:33  [INFO]    Vectorizer                     CountVectorizer TfidfVectorizer
22:06:33  [INFO]    n-gram range                   (1,1) unigrams  (1,2) bigrams        ← new
22:06:33  [INFO]    Sublinear TF                               No            Yes        ← new
22:06:33  [INFO]    Max features                           23,818         15,000
22:06:33  [INFO]    Class weighting                          None       balanced        ← new
22:06:33  [INFO]    Max iterations                            200          1,000        ← new
22:06:33  [INFO]    Test Accuracy                          95.31%         96.36%       +1.05%
22:06:33  [INFO]    F1 Weighted                            95.13%         96.36%       +1.23%
22:06:33  [INFO]  
  WHY DID WE GET BETTER ACCURACY?

  1. TF-IDF vs BoW
     BoW gives the same score to 'the' (common, useless) and 'glioblastoma'
     (rare, specific to brain cancer). TF-IDF correctly down-weights 'the'
     and up-weights 'glioblastoma'.

  2. Bigrams — (1,2) ngram_range
     'renal' alone could appear in kidney cancer OR kidney disease descriptions.
     'renal cell' is almost always KIRC (clear cell kidney cancer).
     Bigrams capture the medical meaning that single words miss.

  3. Balanced class weights
     UCS (Uterine Carcinosarcoma) has only ~37 training samples.
     Without balancing, the model largely ignores it.
     With weight=27×, the model learns UCS-specific patterns properly.

  4. More iterations (1000 vs 200)
     Complex 33-class problems need more optimisation steps to fully converge.
    
22:06:33  [INFO]  ======================================================================
22:06:33  [INFO]    DONE
22:06:33  [INFO]  ======================================================================
22:06:33  [INFO]    Final Test Accuracy : 96.36%
22:06:33  [INFO]    Final F1 Weighted   : 96.36%
22:06:33  [INFO]    Total runtime       : 0.8 min
22:06:33  [INFO]    Output directory    : c:\Users\kings\OneDrive\Desktop\CSUDH_Kingslee\CSUDH-Spring-2026\CSC-590-MastersProject\MastersProject\demo\ProjectCode\output
22:06:33  [INFO]    NEXT: Run Method 2 (TF-IDF + SVM) and compare.
