"""
평가셋(questions.csv)으로 /chat API를 채점한다.

  cd 산재GPT-API
  python eval/run_eval.py --label baseline

지표
- 검색 Hit@5     : 정답 근거 중 하나라도 검색된 비율
- 검색 Recall@5  : 정답 근거 중 검색된 비율의 평균
- 핵심포인트 충족률: 답변이 핵심포인트를 담고 있는 비율 (GPT-4o 채점)
- 인용 근거율     : 답변에 적힌 '제N조' 중 실제로 검색된 문서에 있는 조문의 비율
"""
import argparse
import csv
import json
import os
import re
import sys
import time
from datetime import datetime

from pydantic import BaseModel, Field

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(EVAL_DIR))
sys.path.insert(0, EVAL_DIR)
os.chdir(os.path.dirname(EVAL_DIR))  # main.py가 상대경로 vector_db를 읽는다

from fastapi.testclient import TestClient  # noqa: E402
from langchain_openai import ChatOpenAI  # noqa: E402

import main  # noqa: E402
from scoring import score_retrieval  # noqa: E402
from summary import print_summary, summarize  # noqa: E402

ARTICLE = re.compile(r"제\d+조(?:의\d+)?")


class PointCheck(BaseModel):
    evidence: str = Field(description="포인트와 같은 주장을 하는 답변 속 문장을 그대로 인용. 없으면 빈 문자열")
    covered: bool = Field(description="evidence가 포인트와 같은 주장을 할 때만 true")
    reason: str = Field(description="판단 근거 한 문장 (한국어)")


judge = ChatOpenAI(model="gpt-4o", temperature=0).with_structured_output(PointCheck)

# 포인트를 한꺼번에 채점시키면 관련 있는 문장만 보여도 true를 주는 경향이 있어 하나씩 채점한다
JUDGE_PROMPT = """당신은 산재보험법 QA 답변을 채점하는 엄격한 평가자입니다.
답변이 아래 [핵심포인트]와 '같은 주장'을 명시적으로 하는지 판단하세요.

규칙
- 포인트와 같은 주장을 하는 답변 문장을 그대로 인용하세요. 없으면 evidence는 빈 문자열, covered=false.
- 같은 주제를 다루기만 하고 주장이 다르면 false입니다.
  예) 포인트 "장해등급 재판정은 2년 후" ↔ 답변 "장해등급은 대통령령으로 정한다" → false (같은 주제, 다른 주장)
  예) 포인트 "원칙적으로 지급하지 않는다" ↔ 답변 "예외적으로 지급할 수 있다" → false (원칙을 말하지 않음)
- 수치·기간·조건이 다르면 false입니다. 표현만 다르고 의미가 같으면 true입니다.
- 질문 속 내용이나 상식으로 추론되는 것은 인정하지 않습니다.

[질문]
{question}

[핵심포인트]
{point}

[답변]
{answer}"""


def ask(client: TestClient, question: str, session_id: str | None = None) -> dict:
    payload = {"question": question}
    if session_id:
        payload["session_id"] = session_id
    res = client.post("/chat", json=payload)
    res.raise_for_status()
    return res.json()


def evaluate(row: dict, by_id: dict, client: TestClient) -> dict:
    session_id = None
    if row["선행질문"]:
        session_id = ask(client, by_id[row["선행질문"]]["질문"])["session_id"]

    started = time.time()
    res = ask(client, row["질문"], session_id)
    latency = time.time() - started
    answer, sources = res["answer"], res["sources"]

    hit, recall = score_retrieval(row["정답근거"], [(s["law"], s["article"]) for s in sources])

    points = [p.strip() for p in row["핵심포인트"].split("/") if p.strip()]
    checks = [judge.invoke(JUDGE_PROMPT.format(question=row["질문"], point=p, answer=answer)) for p in points]
    covered = [c.covered and bool(c.evidence.strip()) for c in checks]

    retrieved_articles = {s["article"] for s in sources if s.get("article")}
    cited = set(ARTICLE.findall(answer))

    return {
        "id": row["id"],
        "유형": row["유형"],
        "질문": row["질문"],
        "답변": answer,
        "검색된_문서": [f"{s['law']} {s['article'] or ''}".strip() for s in sources],
        "정답근거": row["정답근거"],
        "hit": hit,
        "recall": recall,
        "핵심포인트": points,
        "충족": covered,
        "충족률": sum(covered) / len(points) if points else None,
        "채점": [{"포인트": p, "충족": c, "인용": k.evidence, "근거": k.reason}
                 for p, c, k in zip(points, covered, checks)],
        "인용_조문": sorted(cited),
        "인용_근거율": len(cited & retrieved_articles) / len(cited) if cited else None,
        "응답시간": round(latency, 2),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="run", help="결과 파일 이름 (예: baseline, +판례)")
    parser.add_argument("--only", help="특정 id만 실행 (쉼표로 구분, 예: A01,B03)")
    args = parser.parse_args()

    with open(os.path.join(EVAL_DIR, "questions.csv"), encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    by_id = {r["id"]: r for r in rows}
    if args.only:
        rows = [by_id[i.strip()] for i in args.only.split(",")]

    results = []
    # with 블록으로 이벤트 루프 하나를 유지해야 한다. 요청마다 루프가 바뀌면
    # OpenAI 비동기 클라이언트가 이전 루프의 연결을 재사용하려다 멈춘다.
    with TestClient(main.app) as client:
        for i, row in enumerate(rows, 1):
            result = evaluate(row, by_id, client)
            results.append(result)
            print(f"[{i}/{len(rows)}] {row['id']} hit={result['hit']} 충족률={result['충족률']:.2f}", flush=True)

    summary = summarize(results)
    print_summary(summary)

    out_dir = os.path.join(EVAL_DIR, "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{args.label}_{datetime.now():%Y%m%d_%H%M}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        config = {k: getattr(main, k) for k in ("RETRIEVAL_K", "FETCH_K", "ROUTING", "ROUTER_MODEL", "EXCLUDE_LAWS", "EMBEDDING_MODEL")}
        json.dump({"label": args.label, "config": config, "summary": summary, "results": results},
                  f, ensure_ascii=False, indent=2)
    print(f"\n결과 저장: {out_path}")
