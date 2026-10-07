"""
사례 결론 예측 실험: 실제 판결 사례(eval/case_holdout.csv)를 질문으로 넣고,
시스템 답변이 기운 방향(인정/불인정)이 실제 판결 결론과 맞는지 잰다.

  cd 산재GPT-API
  PRECEDENT_K=0 python eval/run_case_eval.py --label case_no_prec   # 판례 없이
  PRECEDENT_K=3 python eval/run_case_eval.py --label case_with_prec # 판례 포함

시험 판례와 그 연관 판결은 판례 DB에서 빠져 있다 (precedents/build_index.py).
그래도 혹시 검색되면 유출로 보고 결과에 표시한다.
"""
import argparse
import csv
import json
import os
import sys
from collections import Counter
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(EVAL_DIR))
os.chdir(os.path.dirname(EVAL_DIR))  # main.py가 상대경로 vector_db를 읽는다

from fastapi.testclient import TestClient  # noqa: E402
from langchain_openai import ChatOpenAI  # noqa: E402

import main  # noqa: E402


class Lean(BaseModel):
    방향: Literal["인정", "불인정", "중립"] = Field(
        description="답변이 산재 인정 가능성을 높게 보면 '인정', 낮게 보면 '불인정', 어느 쪽으로도 기울지 않으면 '중립'")
    근거: str = Field(description="그렇게 판단한 답변 속 문장 인용")


judge = ChatOpenAI(model="gpt-4o", temperature=0).with_structured_output(Lean)

JUDGE_PROMPT = """산재 상담 챗봇의 답변이 이 사안의 산재 인정 가능성을 어느 쪽으로 보고 있는지 판단하세요.
- "인정될 가능성이 높다", "해당한다고 볼 수 있다" → 인정
- "인정되기 어렵다", "불리하다", "비슷한 판례에서 불인정됐다" → 불인정
- 요건만 나열하고 가능성에 대한 판단이 없거나 양쪽을 대등하게 말하면 → 중립
일반적인 '전문가 상담 권장' 문구는 판단에서 무시하세요.

[질문]
{question}

[답변]
{answer}"""


def evaluate(row: dict, client: TestClient) -> dict:
    res = client.post("/chat", json={"question": row["질문"]})
    res.raise_for_status()
    data = res.json()
    lean = judge.invoke(JUDGE_PROMPT.format(question=row["질문"], answer=data["answer"]))
    cited = [s["article"] for s in data["sources"] if s["law"] == "판례"]
    leaked_cases = {row["사건번호"], *filter(None, row["연관사건번호"].split(";"))}
    return {
        "id": row["id"],
        "사건번호": row["사건번호"],
        "재해유형": row["재해유형"],
        "실제결론": row["실제결론"],
        "예측": lean.방향,
        "정답": lean.방향 == row["실제결론"],
        "판정근거": lean.근거,
        "검색된_판례": cited,
        "유출": bool(leaked_cases & set(cited)),
        "질문": row["질문"],
        "답변": data["answer"],
    }


def summarize(results: list[dict]) -> dict:
    n = len(results)
    decided = [r for r in results if r["예측"] != "중립"]
    by_class = {}
    for label in ("인정", "불인정"):
        rows = [r for r in results if r["실제결론"] == label]
        by_class[label] = {"문항수": len(rows), "적중": sum(r["정답"] for r in rows),
                           "예측분포": dict(Counter(r["예측"] for r in rows))}
    return {
        "문항수": n,
        "정확도": sum(r["정답"] for r in results) / n,
        "판단률": len(decided) / n,
        "판단한_문항_정확도": (sum(r["정답"] for r in decided) / len(decided)) if decided else None,
        "평균_검색_판례수": sum(len(r["검색된_판례"]) for r in results) / n,
        "유출": sum(r["유출"] for r in results),
        "결론별": by_class,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    args = parser.parse_args()

    with open(os.path.join(EVAL_DIR, "case_holdout.csv"), encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    results = []
    with TestClient(main.app) as client:
        for i, row in enumerate(rows, 1):
            results.append(evaluate(row, client))
            r = results[-1]
            print(f"[{i}/{len(rows)}] {r['id']} 실제={r['실제결론']} 예측={r['예측']} 판례{len(r['검색된_판례'])}건", flush=True)

    summary = summarize(results)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    out_dir = os.path.join(EVAL_DIR, "results", "case")  # compare.py가 읽는 조문 평가 결과와 분리
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{args.label}_{datetime.now():%Y%m%d_%H%M}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"label": args.label, "precedent_k": main.PRECEDENT_K, "precedent_mode": main.PRECEDENT_MODE, "summary": summary, "results": results},
                  f, ensure_ascii=False, indent=2)
    print(f"결과 저장: {out_path}")
