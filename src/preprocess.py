import re
import pandas as pd
from langdetect import DetectorFactory, detect_langs
from nltk.corpus import stopwords

DetectorFactory.seed = 0
EN_SW = set(stopwords.words("english")) - {"no", "not", "nor"}

URL = re.compile(r"https?://\S+|www\.\S+")
REPEAT = re.compile(r"(.)\1{2,}")
NON_ALNUM = re.compile(r"[^a-z0-9\s]")
SPACES = re.compile(r"\s+")


def clean_text(s) -> str:
    s = str(s).lower().replace("’", "'")
    s = URL.sub(" ", s)
    s = s.replace("can't", "can not").replace("won't", "will not")
    s = re.sub(r"n't\b", " not", s)      # don't -> do not, isn't -> is not
    s = REPEAT.sub(r"\1\1", s)           # gooood -> good
    s = NON_ALNUM.sub(" ", s)            # buang emoji, simbol, non-ASCII
    return SPACES.sub(" ", s).strip()


def non_en_confident(s: str) -> bool:
    w = s.split()
    if len(w) < 6:
        return False
    if sum(x in EN_SW for x in w) / len(w) > 0.15:
        return False
    try:
        top = detect_langs(s)[0]
        return top.lang != "en" and top.prob > 0.99
    except Exception:
        return False


def build_dataset(raw_path: str, out_path: str) -> pd.DataFrame:
    df = pd.read_csv(raw_path)
    print("raw:", len(df))

    df["text_clean"] = df["review_text"].fillna("").map(clean_text)
    df = df[df["text_clean"] != ""]
    print("setelah buang teks kosong:", len(df))

    df = df[~df["text_clean"].map(non_en_confident)]
    print("setelah buang non-Inggris:", len(df))

    df = df[df["rating"] != 3].copy()
    df["label"] = (df["rating"] >= 4).astype(int)   # 1=positif, 0=negatif
    print("setelah buang rating 3:", len(df))

    agg = (df.groupby("text_clean")
             .agg(label_mean=("label", "mean"), n=("label", "size"))
             .reset_index())
    print("teks unik:", len(agg))
    print("teks dengan label bentrok:", ((agg.label_mean > 0) & (agg.label_mean < 1)).sum())

    agg = agg[agg.label_mean != 0.5].copy()         # tie = ambigu, buang
    agg["label"] = (agg.label_mean > 0.5).astype(int)
    out = agg[["text_clean", "label", "n"]]

    print("final:", len(out))
    print(out["label"].value_counts(normalize=True).rename({1: "positif", 0: "negatif"}))

    out.to_csv(out_path, index=False)
    return out


if __name__ == "__main__":
    build_dataset("data/raw/airbnb_app_reviews_raw.csv",
                  "data/processed/reviews_clean.csv")