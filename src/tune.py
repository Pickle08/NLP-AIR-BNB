import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline, make_union

df = pd.read_csv("data/processed/reviews_clean.csv")
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)

def word():
    return TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True,
                           token_pattern=r"(?u)\b\w+\b")

def char():
    return TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3,
                           sublinear_tf=True)

for name, feats in [("word", word), ("word+char", lambda: make_union(word(), char()))]:
    for C in [0.5, 1, 2, 5, 10]:
        m = make_pipeline(feats(), LogisticRegression(class_weight="balanced",
                                                      max_iter=2000, C=C))
        s = cross_val_score(m, df["text_clean"], df["label"], cv=cv,
                            scoring="f1_macro", n_jobs=-1)
        print(f"{name:10s} C={C:<4} mean={s.mean():.4f} std={s.std():.4f}")