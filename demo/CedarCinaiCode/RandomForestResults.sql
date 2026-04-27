(venv) PS C:\Users\kings\OneDrive\Desktop\CSUDH_Kingslee\CSUDH-Spring-2026\CSC-590-MastersProject\MastersProject\reference_code\final> python bow_random_forest.py 
08:27:27  [INFO]  ======================================================================
08:27:27  [INFO]    Bag of Words + RANDOM FOREST  — Cedars-Sinai Reference
08:27:27  [INFO]    CSC-590 Masters Project, CSUDH Spring 2026
08:27:27  [INFO]  ======================================================================
08:27:27  [INFO]  
08:27:27  [INFO]  ======================================================================
08:27:27  [INFO]    STEP 1 — Load & Compile Dataset
08:27:27  [INFO]  ======================================================================
08:27:28  [INFO]    Reading reports from: c:\Users\kings\OneDrive\Desktop\CSUDH_Kingslee\CSUDH-Spring-2026\CSC-590-MastersProject\MastersProject\DataSet\TCGA_Reports.csv\TCGA_Reports.csv
08:27:28  [INFO]    Loaded 9,523 reports in 0.5s
08:27:28  [INFO]    Columns: ['patient_filename', 'text']
08:27:28  [INFO]    Unique patients: 9523
08:27:28  [INFO]    Reading labels from: c:\Users\kings\OneDrive\Desktop\CSUDH_Kingslee\CSUDH-Spring-2026\CSC-590-MastersProject\MastersProject\DataSet\tcga_patient_to_cancer_type.csv
08:27:28  [INFO]    Label rows: 11160 | Unique cancer types: 33
08:27:28  [INFO]    Merged dataset shape: (9523, 4)
08:27:28  [INFO]  
--- Cancer type distribution ---
08:27:28  [INFO]      BRCA   Breast invasive carcinoma                     1034 reports (10.9%)
08:27:28  [INFO]      UCEC   Uterine Corpus Endometrial Carcinoma           546 reports (5.7%)
08:27:28  [INFO]      KIRC   Kidney renal clear cell carcinoma              525 reports (5.5%)
08:27:28  [INFO]      HNSC   Head and Neck squamous cell carcinoma          520 reports (5.5%)
08:27:28  [INFO]      LUAD   Lung adenocarcinoma                            488 reports (5.1%)
08:27:28  [INFO]      THCA   Thyroid carcinoma                              487 reports (5.1%)
08:27:28  [INFO]      LGG    Brain Lower Grade Glioma                       469 reports (4.9%)
08:27:28  [INFO]      LUSC   Lung squamous cell carcinoma                   468 reports (4.9%)
08:27:28  [INFO]      PRAD   Prostate adenocarcinoma                        446 reports (4.7%)
08:27:28  [INFO]      COAD   Colon adenocarcinoma                           418 reports (4.4%)
08:27:28  [INFO]      GBM    Glioblastoma multiforme                        399 reports (4.2%)
08:27:28  [INFO]      BLCA   Bladder Urothelial Carcinoma                   379 reports (4.0%)
08:27:28  [INFO]      OV     Ovarian serous cystadenocarcinoma              371 reports (3.9%)
08:27:28  [INFO]      STAD   Stomach adenocarcinoma                         361 reports (3.8%)
08:27:28  [INFO]      LIHC   Liver hepatocellular carcinoma                 341 reports (3.6%)
08:27:28  [INFO]      CESC   Cervical squamous cell carcinoma               289 reports (3.0%)
08:27:28  [INFO]      KIRP   Kidney renal papillary cell carcinoma          280 reports (2.9%)
08:27:28  [INFO]      SARC   Sarcoma                                        249 reports (2.6%)
08:27:28  [INFO]      PAAD   Pancreatic adenocarcinoma                      176 reports (1.8%)
08:27:28  [INFO]      PCPG   Pheochromocytoma and Paraganglioma             174 reports (1.8%)
08:27:28  [INFO]      READ   Rectum adenocarcinoma                          162 reports (1.7%)
08:27:28  [INFO]      ESCA   Esophageal carcinoma                           146 reports (1.5%)
08:27:28  [INFO]      THYM   Thymoma                                        114 reports (1.2%)
08:27:28  [INFO]      KICH   Kidney Chromophobe                             112 reports (1.2%)
08:27:28  [INFO]      SKCM   Skin Cutaneous Melanoma                        102 reports (1.1%)
08:27:28  [INFO]      ACC    Adrenocortical carcinoma                        90 reports (0.9%)
08:27:28  [INFO]      TGCT   Testicular Germ Cell Tumors                     87 reports (0.9%)
08:27:28  [INFO]      MESO   Mesothelioma                                    79 reports (0.8%)
08:27:28  [INFO]      UVM    Uveal Melanoma                                  65 reports (0.7%)
08:27:28  [INFO]      UCS    Uterine Carcinosarcoma                          56 reports (0.6%)
08:27:28  [INFO]      DLBC   Diffuse Large B-cell Lymphoma                   47 reports (0.5%)
08:27:28  [INFO]      CHOL   Cholangiocarcinoma                              43 reports (0.5%)
08:27:28  [INFO]  
--- Text length stats ---
08:27:29  [INFO]    Avg words: 560  |  Median: 432  |  Max: 4046
08:27:29  [INFO]    Reports > 512 words: 4065 (42.7%)
08:27:29  [INFO]  
--- Example report ---
08:27:29  [INFO]    Patient : TCGA-BP-5195
08:27:29  [INFO]    Cancer  : KIRC — Kidney renal clear cell carcinoma
08:27:29  [INFO]    Preview : Date of Recelpt: Clinical Diagnosis & History: Incidental 3 cm left upper pole renal mass. Specimens Submitted: 1: Kidney, Left Upper Pole; Partial Nephrectomy. DIAGNOSIS: 1. Kidney, Left Upper Pole; Partial Nephrectomy: Tumor Type: Renal cell carcinoma - Conventional (clear cell) type. Fuhrman Nuclear Grade: Nuclear grade II/IV. Tumor Size: Greatest diameter is 2.4 cm. Local Invasion (for renal c
08:27:29  [INFO]  
08:27:29  [INFO]  ======================================================================
08:27:29  [INFO]    STEP 2 — Train / Val / Test Split  (50% / 20% / 30%)
08:27:29  [INFO]  ======================================================================
08:27:29  [INFO]    No data leakage between splits — sanity checks passed
08:27:29  [INFO]    Train : 4,761 reports (50%)
08:27:29  [INFO]    Val   : 1,905  reports (20%)
08:27:29  [INFO]    Test  : 2,857 reports (30%)
08:27:29  [INFO]  
--- Distribution per split ---
08:27:29  [INFO]    Type        All    Train      Val     Test
08:27:29  [INFO]    --------------------------------------------
08:27:29  [INFO]    ACC       90      42      16      32
08:27:29  [INFO]    BLCA     379     180      70     129
08:27:29  [INFO]    BRCA    1034     532     232     270
08:27:29  [INFO]    CESC     289     155      45      89
08:27:29  [INFO]    CHOL      43      21       9      13
08:27:29  [INFO]    COAD     418     213      85     120
08:27:29  [INFO]    DLBC      47      23       8      16
08:27:29  [INFO]    ESCA     146      82      28      36
08:27:29  [INFO]    GBM      399     184      80     135
08:27:29  [INFO]    HNSC     520     254     118     148
08:27:29  [INFO]    KICH     112      54      19      39
08:27:29  [INFO]    KIRC     525     255     111     159
08:27:29  [INFO]    KIRP     280     143      55      82
08:27:29  [INFO]    LGG      469     230     103     136
08:27:29  [INFO]    LIHC     341     183      56     102
08:27:29  [INFO]    LUAD     488     230     104     154
08:27:29  [INFO]    LUSC     468     232      97     139
08:27:29  [INFO]    MESO      79      34      18      27
08:27:29  [INFO]    OV       371     191      65     115
08:27:29  [INFO]    PAAD     176      83      39      54
08:27:29  [INFO]    PCPG     174      91      35      48
08:27:30  [INFO]    PRAD     446     225      87     134
08:27:30  [INFO]    READ     162      84      31      47
08:27:30  [INFO]    SARC     249     125      43      81
08:27:30  [INFO]    SKCM     102      47      16      39
08:27:30  [INFO]    STAD     361     197      67      97
08:27:30  [INFO]    TGCT      87      44      19      24
08:27:30  [INFO]    THCA     487     237      93     157
08:27:30  [INFO]    THYM     114      56      22      36
08:27:30  [INFO]    UCEC     546     278     111     157
08:27:30  [INFO]    UCS       56      29       8      19
08:27:30  [INFO]    UVM       65      27      15      23
08:27:30  [INFO]  
08:27:30  [INFO]  ======================================================================
08:27:30  [INFO]    STEP 3 — Bag of Words Vectorizer
08:27:30  [INFO]  ======================================================================
08:27:34  [INFO]  
--- Example tokenization ---
08:27:34  [INFO]    Raw text (300 chars): Date of Recelpt: Clinical Diagnosis & History: Incidental 3 cm left upper pole renal mass. Specimens Submitted: 1: Kidney, Left Upper Pole; Partial Nephrectomy. DIAGNOSIS: 1. Kidney, Left Upper Pole; Partial Nephrectomy: Tumor Type: Renal cell carcinoma - Conventional (clear cell) type. Fuhrman Nucl
08:27:34  [INFO]    Whitespace split (first 15): ['Date', 'of', 'Recelpt:', 'Clinical', 'Diagnosis', '&', 'History:', 'Incidental', '3', 'cm', 'left', 'upper', 'pole', 'renal', 'mass.']
08:27:34  [INFO]    NLTK tokenized   (first 15): ['Date', 'of', 'Recelpt', ':', 'Clinical', 'Diagnosis', '&', 'History', ':', 'Incidental', '3', 'cm', 'left', 'upper', 'pole']
08:27:34  [INFO]  
  After cleaning — 233 tokens: ['date', 'recelpt', 'clinical', 'diagnosis', 'history', 'incidental', 'cm', 'left', 'upper', 'pole', 'renal', 'mass', 'specimens', 'submitted', 'kidney', 'left', 'upper', 'pole', 'partial', 'nephrectomy']
08:27:34  [INFO]  
--- Fitting CountVectorizer on training corpus ---
08:27:34  [INFO]    Corpus size: 4761 reports
08:28:21  [INFO]    Fit complete in 46.4s
08:28:21  [INFO]    Vocabulary size: 23,818 unique medical terms
08:30:26  [INFO]    BoW matrix shape: (4761, 23818)
08:30:26  [INFO]    Sparsity: 0.61% non-zero cells
08:30:26  [INFO]  
--- Word frequency stats in BoW matrix ---
08:30:26  [INFO]    Mean   word frequency across vocabulary: 68.1
08:30:26  [INFO]    Median word frequency: 1.0
08:30:26  [INFO]    Max    word frequency: 45522
08:30:26  [INFO]    (Most medical words appear rarely — sparse matrix is normal)
08:30:26  [INFO]  
08:30:26  [INFO]  ======================================================================
08:30:26  [INFO]    STEP 4 — Random Forest  [4-BoW_RF.ipynb]
08:30:26  [INFO]  ======================================================================
08:31:41  [INFO]    Train shape: (4761, 23818)
08:31:41  [INFO]    Val   shape: (1905, 23818)
08:31:41  [INFO]    Classes    : 32 cancer types
08:31:41  [INFO]  
--- How Random Forest works ---
08:31:41  [INFO]    A Random Forest builds many decision trees — each on a random subset of data.
08:31:41  [INFO]    Each tree independently votes: 'I think this report is BRCA (breast cancer)'
08:31:41  [INFO]    The final prediction = majority vote across all 100 trees.
08:31:41  [INFO]    More trees = more stable predictions, but slower training.
08:31:41  [INFO]    Each tree only looks at a random subset of words (features) at each split.
08:31:41  [INFO]    This prevents any single tree from overfitting to one pattern.
08:31:41  [INFO]  
--- Training  (n_estimators=100, random_state=0) ---
08:31:41  [INFO]    Training 100 decision trees — this takes ~3-5 minutes...
08:31:55  [INFO]    Training complete in 13.5s
08:31:55  [INFO]    Number of trees built: 100
08:31:55  [INFO]    Features per split  : sqrt
08:31:55  [INFO]  
--- Results on TRAINING set ---
08:31:56  [INFO]    Train Accuracy : 99.96%
08:31:56  [INFO]    (Trees overfit training data — near-perfect is expected here)
08:31:56  [INFO]  
--- Results on VALIDATION set  ← real performance on unseen data ---
08:31:56  [INFO]    Val Accuracy        : 91.39%
08:31:56  [INFO]    Val F1 (weighted)   : 90.55%
08:31:56  [INFO]    Reference benchmark : ~92.65%  (Cedars-Sinai original result)
08:31:56  [INFO]    Our result          : 91.39%
08:31:56  [INFO]  
  Full classification report:
08:31:56  [INFO]                precision    recall  f1-score   support

         ACC       1.00      0.62      0.77        16
        BLCA       0.99      0.96      0.97        70
        BRCA       0.98      1.00      0.99       232
        CESC       0.95      0.80      0.87        45
        CHOL       1.00      0.44      0.62         9
        COAD       0.78      0.95      0.86        85
        DLBC       1.00      0.62      0.77         8
        ESCA       0.92      0.86      0.89        28
         GBM       0.94      0.94      0.94        80
        HNSC       0.99      1.00      1.00       118
        KICH       1.00      0.05      0.10        19
        KIRC       0.75      0.99      0.85       111
        KIRP       0.95      0.71      0.81        55
         LGG       0.91      0.97      0.94       103
        LIHC       0.92      1.00      0.96        56
        LUAD       0.89      0.90      0.90       104
        LUSC       0.88      0.92      0.90        97
        MESO       1.00      0.78      0.88        18
          OV       0.91      0.97      0.94        65
        PAAD       0.97      0.92      0.95        39
        PCPG       0.94      0.94      0.94        35
        PRAD       0.97      1.00      0.98        87
        READ       0.73      0.35      0.48        31
        SARC       1.00      0.72      0.84        43
        SKCM       1.00      0.75      0.86        16
        STAD       0.88      0.91      0.90        67
        TGCT       1.00      1.00      1.00        19
        THCA       0.98      0.99      0.98        93
        THYM       0.95      0.86      0.90        22
        UCEC       0.84      0.96      0.90       111
         UCS       0.00      0.00      0.00         8
         UVM       1.00      1.00      1.00        15

    accuracy                           0.91      1905
   macro avg       0.91      0.81      0.83      1905
weighted avg       0.92      0.91      0.91      1905

IF "breast" present → go left
  IF "ductal" present → go left
    IF "grade" high → BRCA
	
08:31:56  [INFO]  
--- Per-class F1 sorted worst → best ---
08:31:57  [INFO]    Code   Cancer Name                                       F1  #Train   #Val
08:31:57  [INFO]    --------------------------------------------------------------------------
08:31:57  [INFO]    UCS    Uterine Carcinosarcoma                         0.000      29      8
08:31:57  [INFO]    KICH   Kidney Chromophobe                             0.100      54     19
08:31:57  [INFO]    READ   Rectum adenocarcinoma                          0.478      84     31
08:31:57  [INFO]    CHOL   Cholangiocarcinoma                             0.615      21      9
08:31:57  [INFO]    DLBC   Diffuse Large B-cell Lymphoma                  0.769      23      8
08:31:57  [INFO]    ACC    Adrenocortical carcinoma                       0.769      42     16
08:31:57  [INFO]    KIRP   Kidney renal papillary cell carcinoma          0.812     143     55
08:31:57  [INFO]    SARC   Sarcoma                                        0.838     125     43
08:31:57  [INFO]    KIRC   Kidney renal clear cell carcinoma              0.853     255    111
08:31:57  [INFO]    SKCM   Skin Cutaneous Melanoma                        0.857      47     16
08:31:57  [INFO]    COAD   Colon adenocarcinoma                           0.857     213     85
08:31:57  [INFO]    CESC   Cervical squamous cell carcinoma               0.867     155     45
08:31:57  [INFO]    MESO   Mesothelioma                                   0.875      34     18
08:31:57  [INFO]    ESCA   Esophageal carcinoma                           0.889      82     28
08:31:57  [INFO]    LUAD   Lung adenocarcinoma                            0.895     230    104
08:31:57  [INFO]    UCEC   Uterine Corpus Endometrial Carcinoma           0.895     278    111
08:31:57  [INFO]    STAD   Stomach adenocarcinoma                         0.897     197     67
08:31:57  [INFO]    LUSC   Lung squamous cell carcinoma                   0.899     232     97
08:31:57  [INFO]    THYM   Thymoma                                        0.905      56     22
08:31:57  [INFO]    GBM    Glioblastoma multiforme                        0.938     184     80
08:31:57  [INFO]    LGG    Brain Lower Grade Glioma                       0.939     230    103
08:31:57  [INFO]    OV     Ovarian serous cystadenocarcinoma              0.940     191     65
08:31:57  [INFO]    PCPG   Pheochromocytoma and Paraganglioma             0.943      91     35
08:31:57  [INFO]    PAAD   Pancreatic adenocarcinoma                      0.947      83     39
08:31:57  [INFO]    LIHC   Liver hepatocellular carcinoma                 0.957     183     56
08:31:57  [INFO]    BLCA   Bladder Urothelial Carcinoma                   0.971     180     70
08:31:57  [INFO]    PRAD   Prostate adenocarcinoma                        0.983     225     87
08:31:57  [INFO]    THCA   Thyroid carcinoma                              0.984     237     93
08:31:57  [INFO]    BRCA   Breast invasive carcinoma                      0.991     532    232
08:31:57  [INFO]    HNSC   Head and Neck squamous cell carcinoma          0.996     254    118
08:31:57  [INFO]    TGCT   Testicular Germ Cell Tumors                    1.000      44     19
08:31:57  [INFO]    UVM    Uveal Melanoma                                 1.000      27     15
08:31:57  [INFO]  
--- Top 20 most important words overall (RF feature importances) ---
08:31:57  [INFO]    RF importance = how much each word reduces prediction error across all trees.
08:31:57  [INFO]    Unlike LR, this is GLOBAL (not per cancer type — one ranking for all classes).
08:47:23  [INFO]    # 1  'breast                '  importance = 0.00840
08:47:23  [INFO]    # 2  'adenocarcinoma        '  importance = 0.00821
08:47:23  [INFO]    # 3  'lung                  '  importance = 0.00747
08:47:23  [INFO]    # 4  'squamous              '  importance = 0.00732
08:47:24  [INFO]    # 5  'carcinoma             '  importance = 0.00687
08:47:24  [INFO]    # 6  'ductal                '  importance = 0.00671
08:47:24  [INFO]    # 7  'cell                  '  importance = 0.00586
08:47:24  [INFO]    # 8  'papillary             '  importance = 0.00575
08:47:24  [INFO]    # 9  'stomach               '  importance = 0.00561
08:47:24  [INFO]    #10  'hepatocellular        '  importance = 0.00512
08:47:24  [INFO]    #11  'nuclear               '  importance = 0.00469
08:47:24  [INFO]    #12  'kidney                '  importance = 0.00461
08:47:24  [INFO]    #13  'renal                 '  importance = 0.00449
08:47:24  [INFO]    #14  'cervix                '  importance = 0.00447
08:47:25  [INFO]    #15  'lymph                 '  importance = 0.00444
08:47:25  [INFO]    #16  'glioblastoma          '  importance = 0.00442
08:47:25  [INFO]    #17  'urothelial            '  importance = 0.00388
08:47:25  [INFO]    #18  'thyroidectomy         '  importance = 0.00387
08:47:25  [INFO]    #19  'cm                    '  importance = 0.00383
08:47:25  [INFO]    #20  'pleura                '  importance = 0.00382
08:47:25  [INFO]  
08:47:25  [INFO]  ======================================================================
08:47:25  [INFO]    STEP 5 — Pipeline + RandomizedSearch + Final Test Evaluation  [5-BoW_ML.ipynb — RF]
08:47:25  [INFO]  ======================================================================
08:47:25  [INFO]  
--- What is RandomizedSearchCV? ---
08:47:25  [INFO]    RF has too many hyperparameters to try every combination (GridSearch would take hours).
08:47:25  [INFO]    RandomizedSearch randomly samples n_iter combinations from the search space.
08:47:25  [INFO]    Search space:
08:47:25  [INFO]      n_estimators      : [10, 100, 500]   (number of trees)
08:47:25  [INFO]      max_features      : ['sqrt', 'log2'] (words considered per split)
08:47:25  [INFO]      min_samples_split : [2, 5, 10]       (min samples to split a node)
08:47:25  [INFO]    We try 2 random combos (n_iter=2) — same as Cedars-Sinai reference.
08:47:25  [INFO]    Running RandomizedSearch (2 RF models, ~5-8 min)...
Fitting 1 folds for each of 2 candidates, totalling 2 fits
[CV] END clf__max_features=sqrt, clf__min_samples_split=2, clf__n_estimators=100; total time= 1.3min
[CV] END clf__max_features=sqrt, clf__min_samples_split=10, clf__n_estimators=10; total time= 1.2min
08:51:31  [INFO]    Done in 245.7s
08:51:31  [INFO]    Best parameters  : {'clf__n_estimators': 100, 'clf__min_samples_split': 2, 'clf__max_features': 'sqrt'}
08:51:31  [INFO]    Best val accuracy: 91.39%
08:51:31  [INFO]  
  RandomizedSearch results table:
08:51:31  [INFO]      n_estimators=100  max_features=sqrt  min_samples_split=2  →  val_acc=91.39%  (fit=56.5s)
08:51:31  [INFO]      n_estimators=10  max_features=sqrt  min_samples_split=10  →  val_acc=89.13%  (fit=50.8s)
08:51:31  [INFO]  
--- Final Test Set Evaluation  ← ONLY DONE ONCE ---
08:51:31  [INFO]    The test set was never used during training or tuning.
08:51:31  [INFO]    This is the unbiased real-world accuracy estimate.
08:52:06  [INFO]  
  ╔══════════════════════════════════════════════╗
08:52:06  [INFO]    ║  RANDOM FOREST FINAL TEST ACCURACY          ║
08:52:06  [INFO]    ║                                              ║
08:52:06  [INFO]    ║  Our result   :  92.16%                    ║
08:52:06  [INFO]    ║  Reference    : ~92.65%  (Cedars-Sinai)     ║
08:52:06  [INFO]    ╚══════════════════════════════════════════════╝
08:52:06  [INFO]  
  Full classification report on test set:
08:52:06  [INFO]                precision    recall  f1-score   support

         ACC       1.00      0.84      0.92        32
        BLCA       1.00      0.98      0.99       129
        BRCA       0.98      1.00      0.99       270
        CESC       0.94      0.85      0.89        89
        CHOL       1.00      0.54      0.70        13
        COAD       0.79      0.97      0.87       120
        DLBC       1.00      0.50      0.67        16
        ESCA       0.94      0.86      0.90        36
         GBM       0.93      0.95      0.94       135
        HNSC       0.97      0.99      0.98       148
        KICH       1.00      0.15      0.27        39
        KIRC       0.76      0.97      0.85       159
        KIRP       0.86      0.78      0.82        82
         LGG       0.92      0.94      0.93       136
        LIHC       0.94      1.00      0.97       102
        LUAD       0.91      0.93      0.92       154
        LUSC       0.91      0.91      0.91       139
        MESO       1.00      0.96      0.98        27
          OV       0.95      0.96      0.95       115
        PAAD       0.95      0.98      0.96        54
        PCPG       1.00      0.94      0.97        48
        PRAD       0.98      1.00      0.99       134
        READ       0.83      0.43      0.56        47
        SARC       0.99      0.81      0.89        81
        SKCM       1.00      0.95      0.97        39
        STAD       0.95      0.96      0.95        97
        TGCT       1.00      1.00      1.00        24
        THCA       0.99      0.99      0.99       157
        THYM       1.00      0.92      0.96        36
        UCEC       0.79      0.97      0.87       157
         UCS       0.00      0.00      0.00        19
         UVM       1.00      0.96      0.98        23

    accuracy                           0.92      2857
   macro avg       0.91      0.84      0.86      2857
weighted avg       0.92      0.92      0.91      2857

08:52:06  [INFO]  
08:52:06  [INFO]  ======================================================================
08:52:06  [INFO]    SUMMARY — Random Forest
08:52:06  [INFO]  ======================================================================
08:52:06  [INFO]    Validation Accuracy : 91.39%
08:52:06  [INFO]    Test Accuracy       : 92.16%
08:52:06  [INFO]    Cedars-Sinai ref    : ~92.65%
08:52:06  [INFO]    Total runtime       : 24.6 min
08:52:06  [INFO]  
(venv) PS C:\Users\kings\OneDrive\Desktop\CSUDH_Kingslee\CSUDH-Spring-2026\CSC-590-MastersProject\MastersProject\reference_code\final> 