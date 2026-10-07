"""평가 결과 요약. run_eval.py와 compare.py가 함께 쓴다."""
from collections import defaultdict


def mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def summarize(results: list[dict]) -> dict:
    groups = defaultdict(list)
    for r in results:
        groups[r["유형"]].append(r)
    groups["전체"] = results
    return {
        name: {
            "문항수": len(rs),
            "Hit@5": mean([None if r["hit"] is None else float(r["hit"]) for r in rs]),
            "Recall@5": mean([r["recall"] for r in rs]),
            "핵심포인트_충족률": mean([r["충족률"] for r in rs]),
            "인용_근거율": mean([r["인용_근거율"] for r in rs]),
            "평균_응답시간": mean([r["응답시간"] for r in rs]),
        }
        for name, rs in groups.items()
    }


def print_summary(summary: dict):
    fmt = lambda v: "-" if v is None else f"{v:.2f}"  # noqa: E731
    print(f"\n{'유형':<8}{'문항':>5}{'Hit@5':>8}{'Recall@5':>10}{'충족률':>8}{'인용근거':>9}{'응답(s)':>9}")
    for name, s in summary.items():
        print(f"{name:<8}{s['문항수']:>5}{fmt(s['Hit@5']):>8}{fmt(s['Recall@5']):>10}"
              f"{fmt(s['핵심포인트_충족률']):>8}{fmt(s['인용_근거율']):>9}{fmt(s['평균_응답시간']):>9}")
