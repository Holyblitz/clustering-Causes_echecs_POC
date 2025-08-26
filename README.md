# clustering-Causes_echecs_POC

# Why GenAI POCs Fail — A Small-Scale Evidence Map

**TL;DR** — Les échecs ne viennent pas des modèles mais du **cadrage** : *quoi* demander au LLM, *avec quelles données*, et *selon quels critères de succès*.  
Ici, on : (1) mappe ~500 posts (Reddit/tech) en **causes d’échec**, (2) compare **RPA vs LLM** sur 3 tâches d’automatisation, (3) propose des **templates** pour cadrer les futurs POC.

---

## Key Findings (corpus Reddit/tech)
> Interactive map: `artifacts/poc_failures_map.html`  
> Exemples par cause: `artifacts/samples_by_label.csv`

[summary.md](https://github.com/user-attachments/files/21988178/summary.md)
| Cause | Count | Percent |
|---|---:|---:|
| Poor workflow integration / IT alignment | 118 | 29.1% |
| Risk, compliance & legal | 106 | 26.1% |
| Cost & infrastructure constraints | 78 | 19.2% |
| No business case / unclear ROI | 47 | 11.6% |
| Change management / training / adoption | 35 | 8.6% |
| Data issues (quality, access, governance) | 22 | 5.4% |

**Lecture rapide**
- **Integration / IT alignment** domine côté communautés techniques (déploiement, CI/CD, SSO, legacy…).  
- Fortes présences aussi : **Risk/Compliance**, **Data issues**, **Change/Adoption**, **Cost/Infra**, **Unclear ROI**.  
- Insight central : beaucoup de POC échouent car **l’“Ask” au LLM est flou** → symptômes en cascade (data/KPI/intégration).

---

## RPA vs LLM: automation results (résumé)
- **Web extraction & factures** : **RPA > LLM** (latence + exactitude).  
- **Tri d’emails** : règles > LLM, mais **Hybride** (règles+LLM) = meilleur compromis.  
- Reco générale : **LLM pour lire/décider**, **RPA pour agir** (orchestration).

> Détails & scripts : voir dossier `AB_testing_rpa_llm` de ton portfolio si public, sinon résumer ici les métriques clés (succès/latence).

---

## How it works (repro rapide)

1) **Collecte**
- Articles/JS : `scripts/collect_poc_playwright.py`  
- Reddit (sans API) : `scripts/collect_reddit_playwright.py`  
→ sortie : `data/corpus.jsonl`

2) **Embed → Reduce → Cluster**
```bash
python3 scripts/embed_reduce_cluster.py   # outputs: artifacts/poc_failures_map.html, poc_clusters.csv, cluster_terms.json

