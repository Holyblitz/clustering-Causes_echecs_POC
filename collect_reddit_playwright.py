# collect_reddit_playwright.py — cherche sur old.reddit, prend N posts/query, extrait 2–3 paragraphes par post
import json, re, time, hashlib, urllib.parse, argparse
from pathlib import Path
from playwright.sync_api import sync_playwright

UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")

KEEP_PAT = re.compile(
    r"\b(poc|pilot|proof of concept|p&l|roi|revenue|impact|adoption|integration|workflow|"
    r"production|scale|scaling|deployment|go-live|business case|use case|budget|stakeholder|"
    r"compliance|privacy|security|risk|governance|data quality|mlops|change management|training|"
    r"procurement|vendor|sla|kpi|measurement)\b", re.I
)

def sid(s): return hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]

def accept_cookies(page):
    for txt in ["Accept all","I agree","Agree","Tout accepter","J'accepte"]:
        try:
            b = page.get_by_role("button", name=txt)
            if b.count(): b.first.click(timeout=1200); return
        except Exception: pass

def get_text(page):
    # tente article/main, sinon body
    for sel in ["article","[role='main']","main"]:
        try:
            node = page.locator(sel).first
            if node.count():
                t = node.inner_text(timeout=1000)
                if t and len(t.strip())>120: return t
        except Exception: pass
    try:
        return page.locator("body").inner_text(timeout=1000)
    except Exception:
        return ""

def split_paras(text):
    parts = re.split(r"\n{2,}|(?<!\.)\.\s+(?=[A-Z])", text)
    paras = [re.sub(r"[ \t]+\n","\n", p.strip()) for p in parts if len(p.strip())>60]
    return paras

def collect_post(page, url, topk=3):
    page.goto(url, timeout=30000)
    page.wait_for_load_state("domcontentloaded")
    accept_cookies(page)
    page.wait_for_timeout(400)
    txt = get_text(page)
    if not txt or len(txt)<80: return []
    paras = split_paras(txt)
    hits = [p for p in paras if KEEP_PAT.search(p)]
    if not hits:
        hits = sorted(paras, key=len, reverse=True)[:topk]
    return hits[:topk]

def search_links(page, q, subreddit=None, per_query=10):
    if subreddit:
        url = f"https://old.reddit.com/r/{subreddit}/search/?q={urllib.parse.quote(q)}&restrict_sr=1&sort=top&t=all"
    else:
        url = f"https://old.reddit.com/search/?q={urllib.parse.quote(q)}&sort=top&t=all"
    page.goto(url, timeout=30000)
    page.wait_for_load_state("domcontentloaded")
    time.sleep(0.6)
    # liens de posts (comments)
    anchors = page.locator("a[href*='/comments/']").all()
    hrefs = []
    seen=set()
    for a in anchors:
        try:
            h = a.get_attribute("href") or ""
            if "/comments/" in h and h.startswith("http") and h not in seen:
                seen.add(h); hrefs.append(h)
        except Exception: pass
        if len(hrefs)>=per_query: break
    return hrefs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", default="data/reddit_queries.txt")
    ap.add_argument("--subs", default="", help="liste comma-separated (ex: datascience,MachineLearning,devops) ; vide = all reddit")
    ap.add_argument("--limit", type=int, default=8, help="posts par requête et par subreddit")
    ap.add_argument("--topk", type=int, default=3, help="paragraphes max par post")
    ap.add_argument("--out", default="data/corpus.jsonl")
    args = ap.parse_args()

    qpath = Path(args.queries)
    if not qpath.exists():
        print(f"⚠️ {qpath} introuvable"); return
    queries = [l.strip() for l in qpath.read_text(encoding="utf-8").splitlines() if l.strip()]
    subs = [s.strip() for s in args.subs.split(",") if s.strip()] or [None]

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    # append mode (on garde ce que tu as déjà)
    seen = set()
    if out.exists():
        for l in out.read_text(encoding="utf-8").splitlines():
            try:
                o = json.loads(l); seen.add(o.get("id"))
            except Exception: pass

    kept_total=0
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context(user_agent=UA)
        page = ctx.new_page()

        for q in queries:
            for sub in subs:
                try:
                    links = search_links(page, q, subreddit=sub, per_query=args.limit)
                except Exception as e:
                    print(f"× search '{q}'/{sub} -> {e}"); continue
                for url in links:
                    try:
                        paras = collect_post(page, url, topk=args.topk)
                        for ptxt in paras:
                            _id = sid(url + ptxt[:200])
                            if _id in seen: continue
                            seen.add(_id)
                            rec = {"id": _id, "src":"reddit", "query": q, "sub": sub or "all",
                                   "url": url, "paragraph": ptxt}
                            out.open("a", encoding="utf-8").write(json.dumps(rec, ensure_ascii=False)+"\n")
                            kept_total += 1
                        print(f"✓ {url} ({len(paras)} paras)")
                    except Exception as e:
                        print(f"× {url} -> {e}")

        browser.close()
    print(f"Done. +{kept_total} paragraphs appended to {out}")

if __name__ == "__main__":
    main()

