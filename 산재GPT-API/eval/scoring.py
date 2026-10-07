"""검색 채점 로직. run_eval.py(채점)와 compare.py(저장된 결과 재채점)가 함께 쓴다."""
import re
from math import sqrt

REF = re.compile(r"(.+?)\s+(제\d+조(?:의\d+)?|별표\s*\d+(?:의\d+)?)$")


def parse_ref(ref: str) -> tuple[str, str | None]:
    """'산업재해보상보험법 시행령 별표 3' -> ('산업재해보상보험법 시행령', '별표 3'), 고시명 -> (고시명, None)"""
    ref = ref.strip()
    match = REF.match(ref)
    if not match:
        return (ref, None)
    return (match.group(1), re.sub(r"별표\s*", "별표 ", match.group(2)))


def parse_gold(raw: str) -> list[tuple[str, str | None]]:
    return [parse_ref(x) for x in raw.split(";") if x.strip()]


def is_retrieved(gold: tuple[str, str | None], retrieved: list[tuple[str, str | None]]) -> bool:
    law, article = gold
    if article is None:
        return any(r_law == law for r_law, _ in retrieved)
    # 조문 단위로 잘리지 않은 문서(예: 텍스트가 깨진 요양업무처리규정)는 법령명만 맞으면 인정
    return any(r_law == law and r_article in (article, None) for r_law, r_article in retrieved)


def score_retrieval(gold_raw: str, retrieved: list[tuple[str, str | None]]) -> tuple[bool | None, float | None]:
    """(Hit, Recall). 정답 근거가 없는 문항(범위밖)은 (None, None)."""
    gold = parse_gold(gold_raw)
    if not gold:
        return None, None
    hits = [is_retrieved(g, retrieved) for g in gold]
    return any(hits), sum(hits) / len(hits)


def wilson(k: int, n: int, z: float = 1.96) -> list[float] | None:
    """비율의 95% 신뢰구간 (Wilson). 표본이 작을 때 정규근사보다 정확하다."""
    if n == 0:
        return None
    p, d = k / n, 1 + z * z / n
    center, margin = p + z * z / (2 * n), z * sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return [round((center - margin) / d, 3), round((center + margin) / d, 3)]
