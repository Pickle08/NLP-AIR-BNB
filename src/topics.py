import sys
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

# pakai: python src\topics.py          -> bandingin K=6,8,10,12
#        python src\topics.py 10       -> fit final dengan K=10
df = pd.read_csv("data/processed/reviews_clean.csv")
neg = df[(df.label == 0) & (df.text_clean.str.split().str.len() >= 4)].reset_index(drop=True)
print("review negatif >=4 kata:", len(neg))

extra = {"app", "airbnb", "air", "bnb", "just", "like", "get", "got", "use", "used",
         "would", "one", "even", "bad", "worst", "terrible", "horrible", "awful",
         "worse", "poor", "experience", "really", "good", "time", "way", "que"}
sw = list(set(ENGLISH_STOP_WORDS) | extra)

vec = TfidfVectorizer(stop_words=sw, ngram_range=(1, 2), min_df=5, max_df=0.5, sublinear_tf=True)
X = vec.fit_transform(neg.text_clean)
terms = vec.get_feature_names_out()
print("vocab:", X.shape[1])


def fit(k):
    m = NMF(n_components=k, init="nndsvd", random_state=42, max_iter=500)
    return m, m.fit_transform(X)


if len(sys.argv) == 1:                       # mode eksplorasi
    for k_try in [6, 8, 10, 12]:
        m, W = fit(k_try)
        share = pd.Series(W.argmax(axis=1)).value_counts(normalize=True).sort_index()
        print(f"\n##### K={k_try}")
        for k in range(k_try):
            top = [terms[i] for i in m.components_[k].argsort()[::-1][:10]]
            print(f"T{k} ({share.get(k, 0):.1%}): {', '.join(top)}")
else:                                        # mode final
    K = int(sys.argv[1])
    nmf, W = fit(K)
    neg["topic"] = W.argmax(axis=1)
    share = neg["topic"].value_counts(normalize=True).sort_index()
    for k in range(K):
        top = [terms[i] for i in nmf.components_[k].argsort()[::-1][:12]]
        print(f"\n=== Topik {k} ({share.get(k, 0):.1%}) ===")
        print("kata:", ", ".join(top))
        ex = neg[neg.topic == k].assign(w=W[neg.topic == k, k]).nlargest(3, "w")
        for t in ex.text_clean:
            print("  -", t[:110])
    neg.to_csv("data/processed/negative_topics.csv", index=False)