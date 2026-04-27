(venv) PS C:\Users\kings\OneDrive\Desktop\CSUDH_Kingslee\CSUDH-Spring-2026\CSC-590-MastersProject\MastersProject\demo\CedarCinaiCode> python bow_logistic_regression.py
18:45:13  [INFO]  ======================================================================
18:45:13  [INFO]    Bag of Words + LOGISTIC REGRESSION  — Cedars-Sinai Reference
18:45:13  [INFO]    CSC-590 Masters Project, CSUDH Spring 2026
18:45:13  [INFO]  ======================================================================
18:45:13  [INFO]  
18:45:13  [INFO]  ======================================================================
18:45:13  [INFO]    STEP 1 — Load & Compile Dataset
18:45:13  [INFO]  ======================================================================
18:45:19  [INFO]    Reading reports from: c:\Users\kings\OneDrive\Desktop\CSUDH_Kingslee\CSUDH-Spring-2026\CSC-590-MastersProject\MastersProject\DataSet\TCGA_Reports.csv\TCGA_Reports.csv
18:45:20  [INFO]    Loaded 9,523 reports in 0.4s
18:45:20  [INFO]    Columns: ['patient_filename', 'text']
18:45:20  [INFO]    Unique patients: 9523
18:45:20  [INFO]    Reading labels from: c:\Users\kings\OneDrive\Desktop\CSUDH_Kingslee\CSUDH-Spring-2026\CSC-590-MastersProject\MastersProject\DataSet\tcga_patient_to_cancer_type.csv
18:45:20  [INFO]    Label rows: 11160 | Unique cancer types: 33
18:45:20  [INFO]    Merged dataset shape: (9523, 4)
18:45:20  [INFO]  
--- Cancer type distribution ---
18:45:20  [INFO]      BRCA   Breast invasive carcinoma                     1034 reports (10.9%)
18:45:20  [INFO]      UCEC   Uterine Corpus Endometrial Carcinoma           546 reports (5.7%)
18:45:20  [INFO]      KIRC   Kidney renal clear cell carcinoma              525 reports (5.5%)
18:45:20  [INFO]      HNSC   Head and Neck squamous cell carcinoma          520 reports (5.5%)
18:45:20  [INFO]      LUAD   Lung adenocarcinoma                            488 reports (5.1%)
18:45:20  [INFO]      THCA   Thyroid carcinoma                              487 reports (5.1%)
18:45:20  [INFO]      LGG    Brain Lower Grade Glioma                       469 reports (4.9%)
18:45:20  [INFO]      LUSC   Lung squamous cell carcinoma                   468 reports (4.9%)
18:45:20  [INFO]      PRAD   Prostate adenocarcinoma                        446 reports (4.7%)
18:45:20  [INFO]      COAD   Colon adenocarcinoma                           418 reports (4.4%)
18:45:20  [INFO]      GBM    Glioblastoma multiforme                        399 reports (4.2%)
18:45:20  [INFO]      BLCA   Bladder Urothelial Carcinoma                   379 reports (4.0%)
18:45:20  [INFO]      OV     Ovarian serous cystadenocarcinoma              371 reports (3.9%)
18:45:20  [INFO]      STAD   Stomach adenocarcinoma                         361 reports (3.8%)
18:45:20  [INFO]      LIHC   Liver hepatocellular carcinoma                 341 reports (3.6%)
18:45:20  [INFO]      CESC   Cervical squamous cell carcinoma               289 reports (3.0%)
18:45:20  [INFO]      KIRP   Kidney renal papillary cell carcinoma          280 reports (2.9%)
18:45:20  [INFO]      SARC   Sarcoma                                        249 reports (2.6%)
18:45:20  [INFO]      PAAD   Pancreatic adenocarcinoma                      176 reports (1.8%)
18:45:20  [INFO]      PCPG   Pheochromocytoma and Paraganglioma             174 reports (1.8%)
18:45:20  [INFO]      READ   Rectum adenocarcinoma                          162 reports (1.7%)
18:45:20  [INFO]      ESCA   Esophageal carcinoma                           146 reports (1.5%)
18:45:20  [INFO]      THYM   Thymoma                                        114 reports (1.2%)
18:45:20  [INFO]      KICH   Kidney Chromophobe                             112 reports (1.2%)
18:45:20  [INFO]      SKCM   Skin Cutaneous Melanoma                        102 reports (1.1%)
18:45:20  [INFO]      ACC    Adrenocortical carcinoma                        90 reports (0.9%)
18:45:20  [INFO]      TGCT   Testicular Germ Cell Tumors                     87 reports (0.9%)
18:45:20  [INFO]      MESO   Mesothelioma                                    79 reports (0.8%)
18:45:20  [INFO]      UVM    Uveal Melanoma                                  65 reports (0.7%)
18:45:20  [INFO]      UCS    Uterine Carcinosarcoma                          56 reports (0.6%)
18:45:20  [INFO]      DLBC   Diffuse Large B-cell Lymphoma                   47 reports (0.5%)
18:45:20  [INFO]      CHOL   Cholangiocarcinoma                              43 reports (0.5%)
18:45:20  [INFO]  
--- Text length stats ---
18:45:20  [INFO]    Avg words: 560  |  Median: 432  |  Max: 4046
18:45:20  [INFO]    Reports > 512 words: 4065 (42.7%)
18:45:20  [INFO]  
--- Example report ---
18:45:20  [INFO]    Patient : TCGA-BP-5195
18:45:20  [INFO]    Cancer  : KIRC — Kidney renal clear cell carcinoma
18:45:20  [INFO]    Preview : Date of Recelpt: Clinical Diagnosis & History: Incidental 3 cm left upper pole renal mass. Specimens Submitted: 1: Kidney, Left Upper Pole; Partial Nephrectomy. DIAGNOSIS: 1. Kidney, Left Upper Pole; Partial Nephrectomy: Tumor Type: Renal cell carcinoma - Conventional (clear cell) type. Fuhrman Nuclear Grade: Nuclear grade II/IV. Tumor Size: Greatest diameter is 2.4 cm. Local Invasion (for renal c
18:45:20  [INFO]  
18:45:20  [INFO]  ======================================================================
18:45:20  [INFO]    STEP 2 — Train / Val / Test Split  (50% / 20% / 30%)
18:45:20  [INFO]  ======================================================================
18:45:21  [INFO]    No data leakage between splits — sanity checks passed
18:45:21  [INFO]    Train : 4,761 reports (50%)
18:45:21  [INFO]    Val   : 1,905  reports (20%)
18:45:21  [INFO]    Test  : 2,857 reports (30%)
18:45:21  [INFO]  
--- Distribution per split ---
18:45:21  [INFO]    Type        All    Train      Val     Test
18:45:21  [INFO]    --------------------------------------------
18:45:21  [INFO]    ACC       90      42      16      32
18:45:21  [INFO]    BLCA     379     180      70     129
18:45:21  [INFO]    BRCA    1034     532     232     270
18:45:21  [INFO]    CESC     289     155      45      89
18:45:21  [INFO]    CHOL      43      21       9      13
18:45:21  [INFO]    COAD     418     213      85     120
18:45:21  [INFO]    DLBC      47      23       8      16
18:45:21  [INFO]    ESCA     146      82      28      36
18:45:21  [INFO]    GBM      399     184      80     135
18:45:21  [INFO]    HNSC     520     254     118     148
18:45:21  [INFO]    KICH     112      54      19      39
18:45:21  [INFO]    KIRC     525     255     111     159
18:45:21  [INFO]    KIRP     280     143      55      82
18:45:21  [INFO]    LGG      469     230     103     136
18:45:21  [INFO]    LIHC     341     183      56     102
18:45:21  [INFO]    LUAD     488     230     104     154
18:45:21  [INFO]    LUSC     468     232      97     139
18:45:21  [INFO]    MESO      79      34      18      27
18:45:21  [INFO]    OV       371     191      65     115
18:45:21  [INFO]    PAAD     176      83      39      54
18:45:21  [INFO]    PCPG     174      91      35      48
18:45:21  [INFO]    PRAD     446     225      87     134
18:45:21  [INFO]    READ     162      84      31      47
18:45:21  [INFO]    SARC     249     125      43      81
18:45:21  [INFO]    SKCM     102      47      16      39
18:45:21  [INFO]    STAD     361     197      67      97
18:45:21  [INFO]    TGCT      87      44      19      24
18:45:21  [INFO]    THCA     487     237      93     157
18:45:21  [INFO]    THYM     114      56      22      36
18:45:21  [INFO]    UCEC     546     278     111     157
18:45:21  [INFO]    UCS       56      29       8      19
18:45:21  [INFO]    UVM       65      27      15      23
18:45:21  [INFO]  
18:45:21  [INFO]  ======================================================================
18:45:21  [INFO]    STEP 3 — Bag of Words Vectorizer
18:45:21  [INFO]  ======================================================================
18:45:38  [INFO]  
--- Example tokenization ---
18:45:38  [INFO]    Raw text (300 chars): Date of Recelpt: Clinical Diagnosis & History: Incidental 3 cm left upper pole renal mass. Specimens Submitted: 1: Kidney, Left Upper Pole; Partial Nephrectomy. DIAGNOSIS: 1. Kidney, Left Upper Pole; Partial Nephrectomy: Tumor Type: Renal cell carcinoma - Conventional (clear cell) type. Fuhrman Nucl
18:45:38  [INFO]    Whitespace split (first 15): ['Date', 'of', 'Recelpt:', 'Clinical', 'Diagnosis', '&', 'History:', 'Incidental', '3', 'cm', 'left', 'upper', 'pole', 'renal', 'mass.']
18:45:39  [INFO]    NLTK tokenized   (first 15): ['Date', 'of', 'Recelpt', ':', 'Clinical', 'Diagnosis', '&', 'History', ':', 'Incidental', '3', 'cm', 'left', 'upper', 'pole']
18:45:39  [INFO]  
  After cleaning — 233 tokens: ['date', 'recelpt', 'clinical', 'diagnosis', 'history', 'incidental', 'cm', 'left', 'upper', 'pole', 'renal', 'mass', 'specimens', 'submitted', 'kidney', 'left', 'upper', 'pole', 'partial', 'nephrectomy']
18:45:39  [INFO]    Stopwords removed example: ['a', 'about', 'above', 'after', 'again', 'against', 'ain', 'all']
18:45:39  [INFO]  
--- Fitting CountVectorizer on training corpus ---
18:45:39  [INFO]    Corpus size: 4761 reports
18:45:39  [INFO]    Fitting... (builds vocabulary from training data only)
18:46:15  [INFO]    Fit complete in 36.1s
18:46:15  [INFO]    Vocabulary size: 23,818 unique medical terms
18:46:15  [INFO]  
--- BoW matrix ---
18:46:45  [INFO]    Transform complete in 30.7s
18:46:45  [INFO]    Matrix shape: (4761, 23818)  (4761 reports × 23818 words)
18:46:45  [INFO]    Sparsity: 0.61% non-zero cells
18:46:45  [INFO]  
18:46:45  [INFO]  ======================================================================
18:46:45  [INFO]    STEP 4 — Logistic Regression  [4-BoW_LR.ipynb]
18:46:45  [INFO]  ======================================================================
18:47:28  [INFO]    Train shape: (4761, 23818)
18:47:28  [INFO]    Val   shape: (1905, 23818)
18:47:28  [INFO]    Classes    : 32 cancer types
18:47:28  [INFO]  
--- How Logistic Regression works ---
18:47:28  [INFO]    LR learns one weight per word per cancer type.
18:47:28  [INFO]    Example: word 'glioblastoma' gets high weight for GBM (brain cancer)
18:47:28  [INFO]             word 'ductal' gets high weight for BRCA (breast cancer)
18:47:28  [INFO]    At prediction time: score = sum(word_count × word_weight) for each cancer type
18:47:28  [INFO]    The cancer type with the highest score wins.
18:47:28  [INFO]  
--- Training  (random_state=0, max_iter=200) ---
18:47:28  [INFO]    Training LR — this takes ~30-60 seconds...
18:47:43  [INFO]    Training complete in 15.5s
18:47:43  [INFO]  
--- Results on TRAINING set ---
18:47:43  [INFO]    Train Accuracy : 99.92%
18:47:43  [INFO]    (Near 100% expected — model memorises what it was trained on)
18:47:43  [INFO]  
--- Results on VALIDATION set  ← real performance on unseen data ---
18:47:43  [INFO]    Val Accuracy        : 95.38%
18:47:43  [INFO]    Val F1 (weighted)   : 95.31%
18:47:43  [INFO]    Reference benchmark : ~95.31%  (Cedars-Sinai original result)
18:47:43  [INFO]    Our result          : 95.38%
18:47:43  [INFO]  
  Full classification report (precision / recall / F1 per cancer type):
18:47:43  [INFO]                precision    recall  f1-score   support

         ACC       0.93      0.88      0.90        16
        BLCA       0.97      0.99      0.98        70
        BRCA       1.00      1.00      1.00       232
        CESC       0.93      0.93      0.93        45
        CHOL       1.00      0.78      0.88         9
        COAD       0.91      0.94      0.92        85
        DLBC       1.00      0.75      0.86         8
        ESCA       0.97      1.00      0.98        28
         GBM       0.96      0.94      0.95        80
        HNSC       0.98      0.99      0.99       118
        KICH       0.79      0.58      0.67        19
        KIRC       0.91      0.98      0.94       111
        KIRP       0.98      0.91      0.94        55
         LGG       0.92      0.97      0.94       103
        LIHC       0.95      0.98      0.96        56
        LUAD       0.93      0.93      0.93       104
        LUSC       0.90      0.96      0.93        97
        MESO       1.00      0.89      0.94        18
          OV       0.95      0.95      0.95        65
        PAAD       1.00      0.97      0.99        39
        PCPG       0.89      0.94      0.92        35
        PRAD       0.99      1.00      0.99        87
        READ       0.89      0.77      0.83        31
        SARC       0.97      0.86      0.91        43
        SKCM       1.00      0.94      0.97        16
        STAD       0.95      0.94      0.95        67
        TGCT       1.00      1.00      1.00        19
        THCA       0.99      0.99      0.99        93
        THYM       0.96      1.00      0.98        22
        UCEC       0.96      0.94      0.95       111
         UCS       0.71      0.62      0.67         8
         UVM       1.00      1.00      1.00        15

    accuracy                           0.95      1905
   macro avg       0.95      0.92      0.93      1905
weighted avg       0.95      0.95      0.95      1905

18:47:43  [INFO]  
--- Per-class F1 sorted worst → best ---
18:47:43  [INFO]    Code   Cancer Name                                       F1  #Train   #Val
18:47:43  [INFO]    --------------------------------------------------------------------------
18:47:43  [INFO]    KICH   Kidney Chromophobe                             0.667      54     19
18:47:43  [INFO]    UCS    Uterine Carcinosarcoma                         0.667      29      8
18:47:43  [INFO]    READ   Rectum adenocarcinoma                          0.828      84     31
18:47:43  [INFO]    DLBC   Diffuse Large B-cell Lymphoma                  0.857      23      8
18:47:43  [INFO]    CHOL   Cholangiocarcinoma                             0.875      21      9
18:47:43  [INFO]    ACC    Adrenocortical carcinoma                       0.903      42     16
18:47:43  [INFO]    SARC   Sarcoma                                        0.914     125     43
18:47:43  [INFO]    PCPG   Pheochromocytoma and Paraganglioma             0.917      91     35
18:47:43  [INFO]    COAD   Colon adenocarcinoma                           0.925     213     85
18:47:43  [INFO]    LUSC   Lung squamous cell carcinoma                   0.930     232     97
18:47:43  [INFO]    LUAD   Lung adenocarcinoma                            0.933     230    104
18:47:43  [INFO]    CESC   Cervical squamous cell carcinoma               0.933     155     45
18:47:43  [INFO]    MESO   Mesothelioma                                   0.941      34     18
18:47:43  [INFO]    KIRP   Kidney renal papillary cell carcinoma          0.943     143     55
18:47:43  [INFO]    LGG    Brain Lower Grade Glioma                       0.943     230    103
18:47:43  [INFO]    KIRC   Kidney renal clear cell carcinoma              0.944     255    111
18:47:43  [INFO]    STAD   Stomach adenocarcinoma                         0.947     197     67
18:47:43  [INFO]    GBM    Glioblastoma multiforme                        0.949     184     80
18:47:43  [INFO]    UCEC   Uterine Corpus Endometrial Carcinoma           0.950     278    111
18:47:43  [INFO]    OV     Ovarian serous cystadenocarcinoma              0.954     191     65
18:47:43  [INFO]    LIHC   Liver hepatocellular carcinoma                 0.965     183     56
18:47:43  [INFO]    SKCM   Skin Cutaneous Melanoma                        0.968      47     16
18:47:43  [INFO]    THYM   Thymoma                                        0.978      56     22
18:47:43  [INFO]    BLCA   Bladder Urothelial Carcinoma                   0.979     180     70
18:47:43  [INFO]    ESCA   Esophageal carcinoma                           0.982      82     28
18:47:43  [INFO]    PAAD   Pancreatic adenocarcinoma                      0.987      83     39
18:47:43  [INFO]    HNSC   Head and Neck squamous cell carcinoma          0.987     254    118
18:47:43  [INFO]    THCA   Thyroid carcinoma                              0.989     237     93
18:47:43  [INFO]    PRAD   Prostate adenocarcinoma                        0.994     225     87
18:47:43  [INFO]    BRCA   Breast invasive carcinoma                      0.998     532    232
18:47:43  [INFO]    TGCT   Testicular Germ Cell Tumors                    1.000      44     19
18:47:43  [INFO]    UVM    Uveal Melanoma                                 1.000      27     15
18:47:43  [INFO]  
  KEY INSIGHT: Cancer types with few training samples tend to have lower F1.
18:47:43  [INFO]  
--- Top 10 most predictive words per cancer type (LR coefficients) ---
18:47:43  [INFO]    A high coefficient means that word strongly suggests this cancer type.
18:47:43  [INFO]  
  GBM — Glioblastoma multiforme
18:47:43  [INFO]      glioblastoma              coeff = +1.480
18:47:43  [INFO]      multiforme                coeff = +0.977
18:47:43  [INFO]      birth                     coeff = +0.601
18:47:43  [INFO]      histological              coeff = +0.600
18:47:43  [INFO]      brain                     coeff = +0.476
18:47:43  [INFO]      date                      coeff = +0.369
18:47:43  [INFO]      frontal                   coeff = +0.359
18:47:43  [INFO]      site                      coeff = +0.355
18:47:43  [INFO]      diagnosis                 coeff = +0.340
18:47:43  [INFO]      right                     coeff = +0.309
18:47:43  [INFO]  
  BRCA — Breast invasive carcinoma
18:47:43  [INFO]      breast                    coeff = +1.362
18:47:43  [INFO]      ductal                    coeff = +0.871
18:47:43  [INFO]      invasive                  coeff = +0.728
18:47:43  [INFO]      classification            coeff = +0.642
18:47:43  [INFO]      nos                       coeff = +0.562
18:47:43  [INFO]      diameter                  coeff = +0.552
18:47:43  [INFO]      carcinoma                 coeff = +0.519
18:47:43  [INFO]      iii                       coeff = +0.450
18:47:43  [INFO]      lobular                   coeff = +0.408
18:47:43  [INFO]      concluding                coeff = +0.404
18:47:43  [INFO]  
  LUAD — Lung adenocarcinoma
18:47:44  [INFO]      lung                      coeff = +1.135
18:47:44  [INFO]      adenocarcinoma            coeff = +0.991
18:47:44  [INFO]      lobe                      coeff = +0.543
18:47:44  [INFO]      nos                       coeff = +0.504
18:47:44  [INFO]      pleura                    coeff = +0.502
18:47:44  [INFO]      rocurement                coeff = +0.465
18:47:44  [INFO]      pneumonectomy             coeff = +0.451
18:47:44  [INFO]      pmx                       coeff = +0.399
18:47:44  [INFO]      bronchopulmonary          coeff = +0.396
18:47:44  [INFO]      hilar                     coeff = +0.387
18:47:44  [INFO]  
18:47:44  [INFO]  ======================================================================
18:47:44  [INFO]    STEP 5 — Pipeline + GridSearch + Final Test Evaluation  [5-BoW_ML.ipynb — LR]
18:47:44  [INFO]  ======================================================================
18:47:44  [INFO]  
--- What is a Pipeline? ---
18:47:44  [INFO]    A Pipeline chains steps so they run together without data leakage.
18:47:44  [INFO]    Step 1: CountVectorizer  — converts raw text to BoW matrix
18:47:44  [INFO]    Step 2: LogisticRegression — classifies the BoW matrix
18:47:44  [INFO]    The vectorizer is ONLY fit on training data, even inside GridSearch.
18:47:44  [INFO]  
--- GridSearchCV — trying max_iter values: [200, 500] ---
18:47:44  [INFO]    GridSearch trains one model per setting, picks the best on val set.
18:47:44  [INFO]    Running GridSearch (2 models × 1 fold = 2 fits, ~2 min)...
Fitting 1 folds for each of 2 candidates, totalling 2 fits
[CV] END ..................................clf__max_iter=200; total time=  57.2s
[CV] END ..................................clf__max_iter=500; total time= 1.0min
18:53:07  [INFO]    Done in 323.7s
18:53:07  [INFO]    Best parameter : {'clf__max_iter': 200}
18:53:07  [INFO]    Best val accuracy: 95.38%
18:53:07  [INFO]  
  GridSearch results table:
18:53:07  [INFO]      max_iter=200  →  val_acc=95.38%  (fit time=45.2s)
18:53:07  [INFO]      max_iter=500  →  val_acc=95.38%  (fit time=47.6s)
18:53:07  [INFO]  
--- Final Test Set Evaluation  ← ONLY DONE ONCE ---
18:53:07  [INFO]    The test set was never used during training or tuning.
18:53:07  [INFO]    This is the unbiased real-world accuracy estimate.
18:54:17  [INFO]  
  ╔══════════════════════════════════════════════╗
18:54:17  [INFO]    ║  LOGISTIC REGRESSION FINAL TEST ACCURACY    ║
18:54:17  [INFO]    ║                                              ║
18:54:17  [INFO]    ║  Our result   :  95.48%                    ║
18:54:17  [INFO]    ║  Reference    : ~95.31%  (Cedars-Sinai)     ║
18:54:17  [INFO]    ╚══════════════════════════════════════════════╝
18:54:17  [INFO]  
  Full classification report on test set:
18:54:17  [INFO]                precision    recall  f1-score   support

         ACC       0.97      0.94      0.95        32
        BLCA       0.98      0.99      0.99       129
        BRCA       0.99      1.00      0.99       270
        CESC       0.94      0.94      0.94        89
        CHOL       0.92      0.85      0.88        13
        COAD       0.85      0.91      0.88       120
        DLBC       1.00      0.88      0.93        16
        ESCA       0.97      0.92      0.94        36
         GBM       0.99      0.98      0.98       135
        HNSC       0.97      0.99      0.98       148
        KICH       0.89      0.82      0.85        39
        KIRC       0.94      0.96      0.95       159
        KIRP       0.90      0.89      0.90        82
         LGG       0.97      0.99      0.98       136
        LIHC       0.97      0.99      0.98       102
        LUAD       0.93      0.92      0.92       154
        LUSC       0.91      0.94      0.92       139
        MESO       0.96      0.96      0.96        27
          OV       0.97      0.91      0.94       115
        PAAD       0.96      0.98      0.97        54
        PCPG       0.98      1.00      0.99        48
        PRAD       0.99      0.99      0.99       134
        READ       0.78      0.74      0.76        47
        SARC       0.94      0.93      0.93        81
        SKCM       0.97      0.97      0.97        39
        STAD       0.98      0.98      0.98        97
        TGCT       1.00      1.00      1.00        24
        THCA       0.99      0.98      0.99       157
        THYM       1.00      0.94      0.97        36
        UCEC       0.94      0.96      0.95       157
         UCS       0.94      0.79      0.86        19
         UVM       1.00      0.96      0.98        23

    accuracy                           0.95      2857
   macro avg       0.95      0.94      0.94      2857
weighted avg       0.96      0.95      0.95      2857

18:54:17  [INFO]  
18:54:17  [INFO]  ======================================================================
18:54:17  [INFO]    SUMMARY — Logistic Regression
18:54:17  [INFO]  ======================================================================
18:54:17  [INFO]    Validation Accuracy : 95.38%
18:54:17  [INFO]    Test Accuracy       : 95.48%
18:54:17  [INFO]    Cedars-Sinai ref    : ~95.31%
18:54:17  [INFO]    Total runtime       : 9.1 min
18:54:17  [INFO]  
(venv) PS C:\Users\kings\OneDrive\Desktop\CSUDH_Kingslee\CSUDH-Spring-2026\CSC-590-MastersProject\MastersProject\demo\CedarCinaiCode> 