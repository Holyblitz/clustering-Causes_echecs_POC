# label_clusters_hybrid.py — rules-first, LLM-on-ties
import json, re, requests, argparse
from pathlib import Path
import pandas as pd
from collections import Counter

ASSIGN = Path("artifacts/poc_clusters.csv")
OUT    = Path("artifacts/cluster_labels.json")          # on écrase le précédent
COUNTS = Path("artifacts/label_counts.csv")
SAMPLES= Path("artifacts/samples_by_label.csv")

TAXONOMY = [
  "No business case / unclear ROI",                 # 0
  "Poor workflow integration / IT alignment",       # 1
  "Data issues (quality, access, governance)",      # 2
  "Change management / training / adoption",        # 3
  "Cost & infrastructure constraints",              # 4
  "Risk, compliance & legal",                       # 5
  "Measurement & KPIs issues",                      # 6
  "Skills gap / team organization",                 # 7
  "Vendor/solution mismatch",                       # 8
  "Scoping & expectations (overpromising)"          # 9
]

# vocab minimal par cause (tu peux l’étoffer)
KW = {
  0: r"\b(business case|roi|p&l|revenue|profit|value proposition|unit economics)\b",
  1: r"\b(integration|integrat(e|ion)|workflow|devops|mlops|deploy(ment)?|prod(uction)?|pipeline|api|sso|legacy|orchestrator|infra team|ops)\b",
  2: r"\b(data quality|data governance|lineage|catalog|ground truth|label(ing|led)|missing data|bias|drift|privacy-preserving data)\b",
  3: r"\b(adoption|change management|buy-in|training|enablement|stakeholder(s)?|culture|resistance)\b",
  4: r"\b(cost|budget|gpu|compute|latency|throughput|cloud|aws|azure|gcp|billing|quota|capacity|server|infra(structure)?)\b",
  5: r"\b(risk|compliance|gdpr|privacy|legal|security|governance board|audit|policy)\b",
  6: r"\b(kpi(s)?|measure(ment)?|metrics|baseline|success criteria|a/b test|impact measurement)\b",
  7: r"\b(skill(s)? gap|hiring|reskilling|roles|org(anization)?|product owner|data steward|team)\b",
  8: r"\b(vendor|supplier|tool|platform|lock-?in|fit|mismatch|rfi|rfp)\b",
  9: r"\b(scope|scoping|overpromis(e|ing)|expectation(s)?|mvp|prototype|pilot requirement(s)?)\b",
}

# --- LLM config (Ollama) ---
MODEL = "llama3.2:1b"
URL   = "http://localhost:11434/api/generate"
SYSTEM = (
  "You classify the *reason* an enterprise AI POC failed. "
  "Choose exactly ONE label index from the provided taxonomy. "
  "Prefer the most *specific* label; avoid defaulting to 'Cost & infrastructure' "
  "unless explicit cost/infra terms appear. Return ONLY JSON: "
  "{\"label_index\": int, \"label\": \"...\", \"confidence\": 0-100, \"rationale\": \"...\"}"
)

def call_llm(prompt: str) -> dict:
    payload = {
        "model": MODEL, "prompt": prompt, "system": SYSTEM,
        "format": "json",
        "options": {"temperature": 0, "num_predict": 180, "num_ctx": 1024},
        "stream": False
    }
    r = requests.post(URL, json=payload, timeout=(12, 50))
    r.raise_for_status()
    txt = r.json()["response"]
    m = re.search(r"\{.*\}", txt, flags=re.S)
    return json.loads(m.group(0)) if m else {"label_index": -1, "label":"Unlabeled","confidence":0,"rationale":""}

def score_cluster(paragraphs: list[str]) -> Counter:
    scores = Counter()
    joined = "\n".join(paragraphs).lower()
    for idx, rx in KW.items():
        hits = re.findall(rx, joined, flags=re.I)
        scores[idx] = len(hits)
    return scores

def label_with_rules(paragraphs: list[str], min_hits=3, margin=2):
    scores = score_cluster(paragraphs)
    top = scores.most_common(2)
    if not top: 
        return None, scores
    (lab1, n1), (lab2, n2) = (top[0], top[1]) if len(top) > 1 else (top[0], (None, 0))
    if n1 >= min_hits and (n1 - n2) >= margin:
        return lab1, scores
    return None, scores  # ambigu → LLM

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_cluster", type=int, default=6, help="#paragraphes à échantillonner par cluster pour le LLM")
    ap.add_argument("--no_llm", action="store_true", help="désactiver LLM (règles seules)")
    args = ap.parse_args()

    df = pd.read_csv(ASSIGN)
    labels = {}

    for cid in sorted(df["cluster"].unique()):
        paras = (
            df[df["cluster"]==cid]["paragraph"]
            .dropna().astype(str).tolist()
        )
        # règles d'abord
        rule_label, scores = label_with_rules(paras, min_hits=3, margin=2)

        if rule_label is not None or args.no_llm:
            idx = rule_label if rule_label is not None else (scores.most_common(1)[0][0] if scores else -1)
            label = TAXONOMY[idx] if 0 <= idx < len(TAXONOMY) else "Unlabeled"
            labels[int(cid)] = {"label_index": int(idx), "label": label, "method": "rules", "scores": dict(scores)}
            print(f"Cluster {cid} -> {label} (rules)")
            continue

        # sinon LLM sur quelques snippets représentatifs (les plus longs)
        sample = sorted(paras, key=len, reverse=True)[:args.per_cluster]
        snippets = "\n- ".join([p[:300].replace("\n"," ") for p in sample]) or "(no snippets)"
        prompt = (
            "Taxonomy:\n" + "\n".join([f"[{i}] {t}" for i,t in enumerate(TAXONOMY)]) +
            f"\n\nRepresentative snippets ({len(sample)}):\n- {snippets}\n\n"
            "Pick the single best label index. Return JSON only."
        )
        try:
            obj = call_llm(prompt)
            idx = int(obj.get("label_index", -1))
            label = TAXONOMY[idx] if 0 <= idx < len(TAXONOMY) else "Unlabeled"
            labels[int(cid)] = {
                "label_index": idx, "label": label, "method": "llm",
                "confidence": obj.get("confidence", 0),
                "rationale": obj.get("rationale",""), "scores": dict(scores)
            }
            print(f"Cluster {cid} -> {label} (llm)")
        except Exception as e:
            # fallback règles si l'appel LLM plante
            idx = scores.most_common(1)[0][0] if scores else -1
            label = TAXONOMY[idx] if 0 <= idx < len(TAXONOMY) else "Unlabeled"
            labels[int(cid)] = {"label_index": idx, "label": label, "method": "rules-fallback", "error": str(e), "scores": dict(scores)}
            print(f"Cluster {cid} -> {label} (rules-fallback, error: {e})")

    # Sauvegardes
    OUT.write_text(json.dumps(labels, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✅ Labels écrits: {OUT}")

    # Récap fréquences + échantillons
    labs = {int(k): v["label"] for k, v in labels.items()}
    df["label"] = df["cluster"].map(labs).fillna("Unlabeled")
    df["label"].value_counts().to_csv(COUNTS, header=["count"])
    out_rows = []
    for lab, grp in df.groupby("label"):
        cols = ["label","url","paragraph"] if "url" in df.columns else ["label","paragraph"]
        out_rows.append(grp.head(3)[cols])
    pd.concat(out_rows).to_csv(SAMPLES, index=False)
    print(f"✅ Récap écrit : {COUNTS} ; {SAMPLES}")

if __name__ == "__main__":
    main()
