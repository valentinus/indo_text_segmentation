"""Experiments for the lecture: classic NLP (2022-style) vs LLM annotation.

1. Export the dataset to CSV.
2. Classic baseline A: seed-word (keyword) topic classifier, English seed words
   (the "topics / labels / seed words" idea from the 2022 study).
3. Classic baseline B: LDA (scikit-learn) with 6 topics on the raw comments.
4. Agreement: Cohen's kappa between each method and the LLM labels.
5. Summary tables used in the slides (topic distribution, topic x sentiment).

Run:  python scripts/run_experiments.py   (from rdsp_2026_lecture/)
"""
import json
import os
import re
import sys
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import cohen_kappa_score

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "data"))
from comments import ANNOTATION_SET, ROWS, TOPICS  # noqa: E402

OUT = os.path.join(ROOT, "results")
os.makedirs(OUT, exist_ok=True)

df = pd.DataFrame(ROWS, columns=["id", "comment", "lang", "gloss_en",
                                 "llm_topic", "llm_topic_secondary", "llm_sentiment"])
df["in_annotation_set"] = df["id"].isin(ANNOTATION_SET)

# public dataset without labels (what students / the LLM receive)
df[["id", "comment", "lang", "gloss_en"]].to_csv(
    os.path.join(ROOT, "data", "fushimi_inari_comments.csv"), index=False, encoding="utf-8-sig")
df.to_csv(os.path.join(OUT, "llm_labels.csv"), index=False, encoding="utf-8-sig")

# ---------------------------------------------------------------- seed words
SEEDS = {
    "SCN": ["view", "beautiful", "breathtaking", "stunning", "light", "sunset", "peaceful",
            "quiet", "moss", "trees", "gates", "magical", "leaves", "atmosphere"],
    "CRW": ["crowd", "crowded", "packed", "tourists", "tour groups", "overtourism",
            "people", "traffic", "busy", "disneyland", "selfie"],
    "CUL": ["shrine", "kitsune", "fox", "foxes", "god", "gods", "pray", "praying",
            "sacred", "spiritual", "history", "torii", "bow", "rude", "respect", "ad"],
    "PRC": ["tip", "station", "train", "bus", "hike", "hours", "shoes", "open",
            "free", "lockers", "vending", "rain", "visit", "early"],
    "FDS": ["food", "udon", "sushi", "mochi", "senbei", "matcha", "souvenir",
            "shop", "prices", "yen", "bought", "ate", "eat"],
    "VID": ["video", "drone", "camera", "editing", "music", "narration", "channel",
            "lighting", "4k", "shots", "you included"],
    "OTH": ["giveaway", "crypto", "subscribe", "check out my", "honeymoon", "link"],
}


def seed_classify(text: str) -> str:
    t = text.lower()
    scores = {}
    for lab, words in SEEDS.items():
        scores[lab] = sum(len(re.findall(r"(?<![a-z])" + re.escape(w) + r"(?![a-z])", t))
                          for w in words)
    best = max(scores.values())
    if best == 0:
        return "OTH"  # nothing matched -> unknown
    # tie-break in a fixed order
    for lab in ["OTH", "VID", "FDS", "CRW", "PRC", "CUL", "SCN"]:
        if scores[lab] == best:
            return lab


df["seed_topic"] = df["comment"].apply(seed_classify)
en = df["lang"] == "en"

# ---------------------------------------------------------------- LDA
vec = CountVectorizer(stop_words="english", min_df=1, token_pattern=r"(?u)\b[a-zA-Z]{3,}\b")
X = vec.fit_transform(df["comment"])
lda = LatentDirichletAllocation(n_components=6, random_state=42, max_iter=200,
                                learning_method="batch")
theta = lda.fit_transform(X)
vocab = np.array(vec.get_feature_names_out())
lda_top_words = [list(vocab[np.argsort(-comp)[:6]]) for comp in lda.components_]
df["lda_topic"] = theta.argmax(axis=1)
# docs with no English token get a flat distribution -> flag them
empty = np.asarray(X.sum(axis=1)).ravel() == 0
df.loc[empty, "lda_topic"] = -1
# map each LDA topic to the LLM label it overlaps most (best case for LDA)
mapping = {}
for k in range(6):
    sub = df[df["lda_topic"] == k]["llm_topic"]
    mapping[k] = sub.value_counts().idxmax() if len(sub) else "OTH"
df["lda_topic_mapped"] = df["lda_topic"].map(lambda k: mapping.get(k, "OTH"))


def kappa(a, b):
    return round(float(cohen_kappa_score(a, b)), 3)


def acc(a, b):
    return round(float((a.values == b.values).mean()), 3)


res = {
    "n_comments": int(len(df)),
    "languages": df["lang"].value_counts().to_dict(),
    "seed_vs_llm_all": {"kappa": kappa(df.llm_topic, df.seed_topic), "agreement": acc(df.llm_topic, df.seed_topic)},
    "seed_vs_llm_english": {"kappa": kappa(df[en].llm_topic, df[en].seed_topic), "agreement": acc(df[en].llm_topic, df[en].seed_topic)},
    "seed_vs_llm_non_english": {"agreement": acc(df[~en].llm_topic, df[~en].seed_topic), "n": int((~en).sum())},
    "lda_vs_llm_all": {"kappa": kappa(df.llm_topic, df.lda_topic_mapped), "agreement": acc(df.llm_topic, df.lda_topic_mapped)},
    "lda_top_words": lda_top_words,
    "lda_mapping": {int(k): v for k, v in mapping.items()},
    "lda_docs_without_tokens": int(empty.sum()),
    "llm_topic_counts": df["llm_topic"].value_counts().reindex(TOPICS.keys(), fill_value=0).to_dict(),
    "llm_sentiment_counts": df["llm_sentiment"].value_counts().to_dict(),
    "multi_topic_share": round(float((df["llm_topic_secondary"] != "").mean()), 3),
}

ct = pd.crosstab(df["llm_topic"], df["llm_sentiment"]).reindex(
    index=list(TOPICS.keys()), columns=["Positive", "Neutral", "Negative"], fill_value=0)
res["topic_x_sentiment"] = ct.to_dict(orient="index")
ct.to_csv(os.path.join(OUT, "topic_x_sentiment.csv"))

df.to_csv(os.path.join(OUT, "all_method_labels.csv"), index=False, encoding="utf-8-sig")
with open(os.path.join(OUT, "summary.json"), "w", encoding="utf-8") as f:
    json.dump(res, f, indent=2, ensure_ascii=False)

print(json.dumps(res, indent=2, ensure_ascii=False))
print("\nSeed-word disagreements (English):")
print(df[en & (df.seed_topic != df.llm_topic)][["id", "llm_topic", "seed_topic", "comment"]].to_string())
