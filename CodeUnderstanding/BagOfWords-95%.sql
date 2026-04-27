Step 1: Import Libraries
pandas → read CSV files
nltk → process text (split words)
CountVectorizer → convert text → numbers
RandomForestClassifier → ML model

Step 2: Load Data
df_train = pd.read_csv("train_tcga_reports_cancer_type.csv")
df_val = pd.read_csv("val_tcga_reports_cancer_type.csv")

Step 3: Extract Text & Labels
arr_train_corpus = df_train["text"].values.tolist()
arr_train_labels = df_train["cancer_type"].values.tolist()

Step 4: Tokenizer
def tokenizer(text):
    arr_tokens = word_tokenize(text.lower())
    arr_tokens = [token for token in arr_tokens if len(token) > 1]
    arr_tokens = [token for token in arr_tokens if token not in arr_stopwords]
    arr_tokens = [token for token in arr_tokens if token.isalpha()]
    return arr_tokens
	
Input: "The tumor is in the lung!"
Output: ["tumor", "lung"]

It removes:

small words (“a”, “is”)
punctuation
numbers

Step 5: Step 5: Convert Text → Numbers (BoW-Bag of Words)
word_vectorizer = CountVectorizer(tokenizer=tokenizer)

Vocabulary:
["tumor", "lung", "breast"]

Text: "tumor lung lung"
→ [1, 2, 0]

Step 6: Train Vectorizer
word_vectorizer.fit(arr_train_corpus)
all unique words (≈ 23,818 words)

Step 7: Convert Data
arr_train_bow = word_vectorizer.transform(arr_train_corpus)
arr_val_bow = word_vectorizer.transform(arr_val_corpus)

(4761 rows, 23818 columns)


Step 8: Train Model
clf = RandomForestClassifier(n_estimators=100)
clf.fit(arr_train_bow, arr_train_labels)

Step 9: Training Accuracy
accuracy_score(...)

Step 10: Validation Accuracy
accuracy_score(...)
~ 91.39%

Step 11: Classification Report
classification_report(...)

Precision
Recall
F1-score

