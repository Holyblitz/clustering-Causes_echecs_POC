import pandas as pd, numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly.express as px, json, re

LABELS = json.load(open("artifacts/cluster_labels.json"))
LABMAP = {int(k):v["label"] for k,v in LABELS.items()}
df = pd.read_csv("artifacts/poc_clusters.csv")
df["label"] = df["cluster"].map(LABMAP)

TARGET = "Poor workflow integration / IT alignment"  # <- sous-thème à creuser
sub = df[df["label"]==TARGET].copy()
print(f"{len(sub)} rows for '{TARGET}'")

texts = sub["paragraph"].astype(str).tolist()
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
emb = model.encode(texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True)

k=6  # essaie 5–8
km = KMeans(n_clusters=k, n_init=20, random_state=42)
sub["subcluster"] = km.fit_predict(emb)

# Viz 2D
pca2 = PCA(n_components=2, random_state=42).fit_transform(emb)
sub["x"], sub["y"] = pca2[:,0], pca2[:,1]
fig = px.scatter(sub, x="x", y="y", color="subcluster", hover_data=["url"])
Path("artifacts").mkdir(exist_ok=True, parents=True)
fig.write_html("artifacts/drilldown_integration.html", include_plotlyjs="cdn")

# Top termes par sous-cluster (rapide)
from sklearn.feature_extraction.text import TfidfVectorizer
vec = TfidfVectorizer(max_features=6000, stop_words="english", ngram_range=(1,2),
                      token_pattern=r"(?u)\b[\w\-]{2,}\b")
Xtf = vec.fit_transform(sub["paragraph"])
vocab = vec.get_feature_names_out()
terms={}
for c in sorted(sub["subcluster"].unique()):
    idx = sub.index[sub["subcluster"]==c]
    top = Xtf[idx].sum(axis=0).A1.argsort()[::-1][:8]
    terms[int(c)] = [vocab[i] for i in top]
Path("artifacts/integration_subterms.json").write_text(json.dumps(terms, indent=2), encoding="utf-8")
sub[["url","paragraph","subcluster"]].to_csv("artifacts/integration_subclusters.csv", index=False)
print("ok -> artifacts/drilldown_integration.html ; integration_subclusters.csv ; integration_subterms.json")

