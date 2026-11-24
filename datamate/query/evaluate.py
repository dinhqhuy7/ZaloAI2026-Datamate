"""Score submission.jsonl on train per §9: score = (right - wrong) / number of questions, split into dev / held-out shops, by type and by shop.

Every query approach (SQL templates, LLM composing a plan, LLM writing SQL...) is compared with each other by this yardstick. HOLDOUT must not contribute
examples to the prompt and its errors must not be examined to fix things: the score there is closer to an unseen shop.

    python -m datamate.query.evaluate ../train/train sub.jsonl
"""
import argparse
import json
import os
from collections import Counter

from datamate.submit import score

HOLDOUT = {"biz_107", "biz_111", "biz_113", "biz_123", "biz_126"}


def evaluate(split_root: str, sub_path: str) -> dict:
    qs = {q["id"]: q for q in map(json.loads, open(os.path.join(split_root, "questions.jsonl"), encoding="utf-8"))}
    gold = {r["id"]: r for r in map(json.loads, open(os.path.join(split_root, "answers.jsonl"), encoding="utf-8"))}
    recs = {r["id"]: r for r in map(json.loads, open(sub_path, encoding="utf-8"))}
    out = {"dev": Counter(), "holdout": Counter(), "type": {}, "shop": {}}
    for qid, g in gold.items():
        r = recs.get(qid, {"id": qid, "type": "abstain"})
        s = score(r, g)
        shop = qs[qid]["business_id"]
        for c in (out["holdout" if shop in HOLDOUT else "dev"], out["type"].setdefault(g["type"], Counter()),
                  out["shop"].setdefault(shop, Counter())):
            c[s] += 1
    return out


def _line(name: str, c: Counter) -> str:
    n = c[1] + c[-1] + c[0]
    return f"{name:<14} đúng {c[1]:>5}  sai {c[-1]:>4}  trống {c[0]:>5}  điểm {(c[1] - c[-1]) / n if n else 0:+.3f}  (n={n})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("split_root")
    ap.add_argument("submission")
    ap.add_argument("--shops", action="store_true", help="also print each shop")
    a = ap.parse_args()
    r = evaluate(a.split_root, a.submission)
    print(_line("dev", r["dev"]))
    print(_line("shop giữ riêng", r["holdout"]))
    print(_line("tất cả", r["dev"] + r["holdout"]))
    for t, c in sorted(r["type"].items(), key=lambda x: -sum(x[1].values())):
        print(_line(t, c))
    if a.shops:
        for s, c in sorted(r["shop"].items()):
            print(_line(s, c))


if __name__ == "__main__":
    main()
