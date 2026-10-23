import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union

df = pd.read_csv("data/processed/reviews_clean.csv")

X_train, X_test, y_train, y_test = train_test_split(
    df["text_clean"], df["label"],
    test_size=0.2, stratify=df["label"], random_state=42,
)

word = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True,
                       token_pattern=r"(?u)\b\w+\b")
char = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3,
                       sublinear_tf=True)

model = make_pipeline(
    make_union(word, char),
    LogisticRegression(class_weight="balanced", max_iter=2000, C=5),
)
model.fit(X_train, y_train)

pred = model.predict(X_test)
print("macro-F1:", round(f1_score(y_test, pred, average="macro"), 4))
print(classification_report(y_test, pred, target_names=["negatif", "positif"]))
print(confusion_matrix(y_test, pred))

joblib.dump(model, "models/tfidf_word_char_lr.joblib")