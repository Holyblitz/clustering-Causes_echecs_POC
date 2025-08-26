# embed_reduce_cluster.py
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
import plotly.express as px

IN = Path("data/corpus_balanced.jsonl")
OUT_DIR = Path("artifacts")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Force un nb de clusters lisible (sinon HDBSCAN)
FORCE_KMEANS = 12

def load_corpus(max_n=5000) -> pd.DataFrame:
    if not IN.exists():
        raise FileNotFoundError("data/corpus.jsonl manquant. Lance d'abord la collecte.")
    rows = []
    with IN.open("r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            if i >= max_n:
                break
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    df = pd.DataFrame(rows)
    if "paragraph" not in df.columns:
        raise ValueError("corpus.jsonl ne contient pas la colonne 'paragraph'.")
    return df

def compute_embeddings(texts):
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")  # CPU OK
    emb = model.encode(texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True)
    return emb

def reduce_dims(emb):
    # PCA -> 50D
    k = int(min(50, emb.shape[1], max(2, emb.shape[0] - 1)))
    pca50 = PCA(n_components=k, random_state=42)
    X50 = pca50.fit_transform(emb)
    # UMAP -> 2D (fallback PCA-2D)
    try:
        import umap.umap_ as umap
        reducer = umap.UMAP(
            n_components=2, n_neighbors=20, min_dist=0.03,
            metric="cosine", random_state=42
        )
        X2 = reducer.fit_transform(X50)
        meth = "UMAP"
    except Exception:
        pca2 = PCA(n_components=2, random_state=42)
        X2 = pca2.fit_transform(X50)
        meth = "PCA-2D"
    return X50, X2, meth

def clusterize(X50):
    if isinstance(globals().get("FORCE_KMEANS"), int):
        from sklearn.cluster import KMeans
        k = FORCE_KMEANS
        km = KMeans(n_clusters=k, n_init=20, random_state=42)
        labels = km.fit_predict(X50)
        return labels, f"KMeans(k={k})"
    try:
        import hdbscan
        clusterer = hdbscan.HDBSCAN(min_cluster_size=18, min_samples=5, metric="euclidean")
        labels = clusterer.fit_predict(X50)
        return labels, "HDBSCAN"
    except Exception:
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=12, n_init=20, random_state=42)
        labels = km.fit_predict(X50)
        return labels, "KMeans(k=12)"

def build_tfidf(texts):
    # stopwords custom (liste, pas set)
    DOMAIN_STOP |= {
  "deploy","deployment","prod","production","pipeline","devops","mlops","kubernetes",
  "airflow","api","sso","iam","legacy","orchestrator","cloud","aws","azure","gcp",
  "server","infrastructure","infra","latency","throughput","gpu","compute",
  "billing","quota","capacity"
    }

    
    FR_STOP = {"les","des","dans","pour","avec","sans","entreprises","projet","projets"}
    CUSTOM_STOP = sorted(set(ENGLISH_STOP_WORDS) | DOMAIN_STOP | FR_STOP)

    vec = TfidfVectorizer(
        max_features=12000,
        stop_words=CUSTOM_STOP,
        ngram_range=(1, 2),
        token_pattern=r"(?u)\b[\w\-]{2,}\b"  # garde les tokens avec tirets (use-case)
    )
    Xtf = vec.fit_transform(texts)
    vocab = vec.get_feature_names_out()
    return Xtf, vocab

def top_terms_per_cluster(labels, Xtf, vocab, n=7):
    terms = {}
    labels = np.asarray(labels)
    for c in sorted(set(labels)):
        idx = np.where(labels == c)[0]
        if len(idx) == 0:
            terms[int(c)] = []
            continue
        sub = Xtf[idx].sum(axis=0).A1  # somme tf-idf
        top_idx = sub.argsort()[::-1][:n]
        terms[int(c)] = [vocab[i] for i in top_idx]
    return terms

# --- exports cluster stats & samples ---
def export_cluster_summaries(df, out_dir=OUT_DIR, per_cluster=5):
    counts = df["cluster"].value_counts().sort_index()
    (out_dir / "cluster_counts.txt").write_text(
        "\n".join([f"{int(c)}\t{int(n)}" for c, n in counts.items()]),
        encoding="utf-8"
    )
    for c in sorted(df["cluster"].unique()):
        cols = ["url", "paragraph"] if "url" in df.columns else ["paragraph"]
        df[df["cluster"] == c].head(per_cluster)[cols].to_csv(
            out_dir / f"samples_cluster_{int(c)}.csv", index=False
        )

def main():
    df = load_corpus()
    print(f"Loaded {len(df)} paragraphs from {IN}")

    texts = df["paragraph"].astype(str).tolist()
    emb = compute_embeddings(texts)
    X50, X2, meth = reduce_dims(emb)
    labels, algo = clusterize(X50)
    df["cluster"] = labels

    # TF-IDF top terms by cluster
    Xtf, vocab = build_tfidf(df["paragraph"].astype(str).tolist())
    terms = top_terms_per_cluster(labels, Xtf, vocab, n=7)
    df["cluster_terms"] = df["cluster"].map(lambda c: ", ".join(terms.get(int(c), [])))

    # Scatter 2D
    df_plot = df.copy()
    df_plot["x"] = X2[:, 0]
    df_plot["y"] = X2[:, 1]

    hover_cols = {"cluster": True}
    if "url" in df_plot.columns:
        hover_cols["url"] = True

    fig = px.scatter(
        df_plot, x="x", y="y", color="cluster_terms",
        hover_data=hover_cols,
        title=f"POC Failures — {meth} + {algo}"
    )
    html = OUT_DIR / "poc_failures_map.html"
    fig.write_html(html, include_plotlyjs="cdn")
    print(f"✅ Carte écrite : {html}")

    # Exports utiles
    out_csv = OUT_DIR / "poc_clusters.csv"
    cols = [c for c in ["id","src","query","sub","url","paragraph","cluster","cluster_terms"] if c in df.columns]
    df[cols].to_csv(out_csv, index=False)
    print(f"✅ CSV écrit : {out_csv}")

    out_terms = OUT_DIR / "cluster_terms.json"
    out_terms.write_text(json.dumps(terms, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✅ Terms écrit : {out_terms}")

    export_cluster_summaries(df)
    print("✅ Exemples par cluster écrits dans artifacts/")

if __name__ == "__main__":
    main()
