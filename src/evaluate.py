import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score

df = pd.read_csv("data/processed/reviews_clean.csv")
model = joblib.load("models/baseline_tfidf_lr.joblib")

X_train, X_test, y_train, y_test = train_test_split(
    df["text_clean"], df["label"], test_size=0.2, stratify=df["label"], random_state=42
)

# 1. salah prediksi
res = X_test.to_frame()
res["true"] = y_test
res["pred"] = model.predict(X_test)
res["p_pos"] = model.predict_proba(X_test)[:, 1]
wrong = res[res.true != res.pred]

pd.set_option("display.max_colwidth", 120)
print("=== positif tapi diprediksi negatif (sample 20) ===")
print(wrong[wrong.true == 1].sample(20, random_state=1)[["text_clean", "p_pos"]])
print("\n=== negatif tapi diprediksi positif (sample 20) ===")
print(wrong[wrong.true == 0].sample(20, random_state=1)[["text_clean", "p_pos"]])

# 2. fitur terkuat
vec = model.named_steps["tfidfvectorizer"]
coef = model.named_steps["logisticregression"].coef_[0]
feat = pd.Series(coef, index=vec.get_feature_names_out()).sort_values()
print("\nTop negatif:", list(feat.head(20).index))
print("Top positif:", list(feat.tail(20).index))

# 3. cross-validation 5-fold di seluruh data
cv = StratifiedKFold(5, shuffle=True, random_state=42)
scores = cross_val_score(model, df["text_clean"], df["label"], cv=cv, scoring="f1_macro")
print("\nCV macro-F1:", scores.round(4), "mean:", scores.mean().round(4), "std:", scores.std().round(4))