# collect_poc_articles.py — robuste (requests + trafilatura.extract), PDF/AMP + fallback
import json, re, hashlib, requests, io
from pathlib import Path
from langdetect import detect
from pdfminer.high_level import extract_text as pdf_extract
import trafilatura

IN  = Path("data/urls.txt")
OUT = Path("data/corpus.jsonl"); OUT.parent.mkdir(parents=True, exist_ok=True)

# Mots-clés larges (POC/ROI/intégration/risques/mesure…)
KEEP_PAT = re.compile(
    r"\b(poc|pilot|proof of concept|p&l|roi|revenue|impact|adoption|integration|workflow|"
    r"production|scale|scaling|deployment|go-live|business case|use case|budget|stakeholder|"
    r"compliance|privacy|security|risk|governance|data quality|mlops|change management|training|"
    r"procurement|vendor|sla|kpi|measurement)\b",
    re.I
)

UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"}

def sid(s): return hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]

def fetch(url: str):
    # 1) GET direct
    r = requests.get(url, headers=UA, timeout=20, allow_redirects=True)
    ct = (r.headers.get("content-type") or "").lower()
    if "pdf" in ct or url.lower().endswith(".pdf"):
        return {"kind":"pdf", "content": r.content, "url": r.url, "title": url}
    return {"kind":"html", "content": r.text, "url": r.url, "title": ""}

def fetch_with_amp(url: str):
    try:
        return fetch(url)
    except Exception:
        # 2) tentative /amp si 1ère requête échoue
        amp = url.rstrip("/") + "/amp"
        return fetch(amp)

def extract_article(payload: dict):
    if payload["kind"] == "pdf":
        try:
            text = pdf_extract(io.BytesIO(payload["content"])) or ""
            return {"title": payload["title"], "text": text}
        except Exception:
            return None
    # HTML → trafilatura
    try:
        js = trafilatura.extract(
            payload["content"], output="json",
            include_comments=False, include_tables=False,
            favor_precision=True, url=payload["url"]
        )
        if js:
            meta = json.loads(js)
            title = (meta.get("title") or "").strip()
            text  = (meta.get("raw_text") or meta.get("text") or "").strip()
            if text:
                return {"title": title, "text": text}
    except Exception:
        pass
    # Fallback texte brut
    try:
        plain = trafilatura.extract(
            payload["content"], output=None,
            include_comments=False, include_tables=False,
            favor_precision=True, url=payload["url"]
        ) or ""
        return {"title": "", "text": plain}
    except Exception:
        return None

def paragraphs(text: str):
    # paragraphes > 60 chars ; coupe sur doubles sauts de ligne
    return [p.strip() for p in re.split(r"\n{2,}", text) if len(p.strip()) > 60]

def main():
    if not IN.exists():
        print("⚠️ data/urls.txt manquant.")
        return
    OUT.write_text("", encoding="utf-8")
    seen = set()
    urls = [u.strip() for u in IN.read_text(encoding="utf-8").splitlines() if u.strip()]
    for url in urls:
        try:
            pay = fetch_with_amp(url)
            art = extract_article(pay)
            if not art or not art["text"]:
                print(f"× {url} (empty)")
                continue
            paras = paragraphs(art["text"])
            # filtre mots-clés ; si rien, garde les 2 plus longs (fallback)
            hits = [p for p in paras if KEEP_PAT.search(p)]
            if not hits:
                hits = sorted(paras, key=len, reverse=True)[:2]
            kept = 0
            for p in hits:
                try:
                    lang = detect(p)
                except Exception:
                    lang = "en"
                if lang not in ("en","fr"):
                    continue
                key = sid((art["title"] or "") + p[:200])
                if key in seen:
                    continue
                seen.add(key)
                rec = {"id": key, "url": url, "title": art["title"], "lang": lang, "paragraph": p}
                OUT.open("a", encoding="utf-8").write(json.dumps(rec, ensure_ascii=False) + "\n")
                kept += 1
            print(f"✓ {url} ({kept} paras)")
        except Exception as e:
            print(f"× {url} -> {e}")

if __name__ == "__main__":
    main()


