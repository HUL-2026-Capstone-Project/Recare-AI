"""
사례 결론 예측 실험(eval/results/case/*.json) 비교표를 만든다.

  python eval/compare_case.py
"""
import glob
import json
import os
import sys

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, EVAL_DIR)

from scoring import wilson  # noqa: E402


def pct(k: int, n: int) -> str:
    if n == 0:
        return "-"
    lo, hi = wilson(k, n)
    return f"{k / n:.2f} ({lo:.2f}~{hi:.2f}, n={n})"


if __name__ == "__main__":
    runs = {}
    for path in sorted(glob.glob(os.path.join(EVAL_DIR, "results", "case", "*.json")), key=os.path.getmtime):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        runs[data["label"]] = data["results"]

    print("### 전체 (괄호: 95% 신뢰구간)\n")
    print("| 실험 | 정확도 | 판단률 | 방향을 정한 답의 정확도 | 실제 인정 적중 | 실제 불인정 적중 |")
    print("|---|---|---|---|---|---|")
    for label, rs in runs.items():
        decided = [r for r in rs if r["예측"] != "중립"]
        pos = [r for r in rs if r["실제결론"] == "인정"]
        neg = [r for r in rs if r["실제결론"] == "불인정"]
        print(f"| {label} | {pct(sum(r['정답'] for r in rs), len(rs))} | {len(decided) / len(rs):.2f} "
              f"| {pct(sum(r['정답'] for r in decided), len(decided))} "
              f"| {sum(r['정답'] for r in pos)}/{len(pos)} | {sum(r['정답'] for r in neg)}/{len(neg)} |")

    print("\n### 재해유형별 방향을 정한 답의 정확도\n")
    types = sorted({r["재해유형"] for rs in runs.values() for r in rs})
    print("| 실험 | " + " | ".join(types) + " |")
    print("|---|" + "---|" * len(types))
    for label, rs in runs.items():
        cells = []
        for t in types:
            decided = [r for r in rs if r["재해유형"] == t and r["예측"] != "중립"]
            cells.append(pct(sum(r["정답"] for r in decided), len(decided)))
        print(f"| {label} | " + " | ".join(cells) + " |")
