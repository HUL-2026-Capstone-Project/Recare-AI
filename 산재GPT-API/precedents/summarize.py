"""
수집한 판례(precedents/data/raw)를 검색용 구조화 요약으로 만든다 → precedents/data/summaries.jsonl

  cd 산재GPT-API
  python precedents/summarize.py            # 전체 (이미 요약한 판례는 건너뜀)
  python precedents/summarize.py --limit 5  # 일부만 시험

사용자는 "택배 일 하다 허리를 다쳤어요"처럼 사실관계로 묻기 때문에, 판례도 사실관계·직종·상병
중심으로 요약해 두어야 비슷한 사례를 찾을 수 있다. 판결 전문은 길어서 주문·앞부분(사실관계)·
뒷부분(판단)만 잘라 넣는다.
"""
import argparse
import asyncio
import glob
import json
import os
import re
from typing import Literal

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

HERE = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(HERE, "data", "raw")
OUT_PATH = os.path.join(HERE, "data", "summaries.jsonl")
MODEL = "gpt-4o-mini"
CONCURRENCY = 8
HEAD_CHARS, TAIL_CHARS = 3000, 2000

load_dotenv(os.path.join(os.path.dirname(HERE), ".env"))


class PrecedentSummary(BaseModel):
    재해유형: Literal["업무상 사고", "업무상 질병", "출퇴근 재해", "보험급여·절차", "기타"]
    상병: str = Field(description="진단명 또는 사망 원인. 없으면 빈 문자열 (예: 뇌출혈, 요추간판탈출증)")
    직종: str = Field(description="재해자의 직종·업무 내용. 회사명이 아니라 하는 일 (예: 택배 배송기사, 건설현장 형틀목공). 알 수 없으면 빈 문자열")
    사실관계: str = Field(description="재해 경위를 2~4문장으로. 날짜·고유명사보다 업무 내용과 상황 위주")
    쟁점: str = Field(description="법원이 판단한 핵심 쟁점 한 문장")
    판단: str = Field(description="법원이 그렇게 판단한 근거(사실 인정과 법리) 2~3문장. 결론을 되풀이하지 말 것")
    결론: Literal["인정", "불인정", "일부 인정", "파기환송", "기타"] = Field(
        description="근로자(원고) 입장에서 업무상 재해·보험급여가 인정됐는지. 대법원 파기환송은 '파기환송'")
    관련조문: list[str] = Field(description="판단에 쓰인 조문. '산업재해보상보험법 제37조' 같은 형식")


PROMPT = """다음은 산업재해보상보험 관련 판결입니다. 산재 상담 시스템이 비슷한 사례를 찾을 수 있도록 요약하세요.
판결에 없는 내용은 지어내지 마세요.

결론은 '근로자(재해자·유족) 쪽에 유리한 결과인가'로 판단합니다.
- 하급심: 공단 처분 취소 → 인정, 청구 기각 → 불인정
- 대법원: 주문이 '원심판결을 파기하고 환송'이면 '파기환송'.
  '상고를 기각'이면 원심 결론이 확정된 것이므로, 판결문에서 원심이 근로자 승소였는지 확인해 인정/불인정으로 쓰세요.
- 근로자와 공단 사이의 다툼이 아니면(예: 사업주 간 구상금 소송) '기타'

[사건] {사건명} / {법원명} {사건번호}
[판시사항] {판시사항}
[판결요지] {판결요지}
[참조조문] {참조조문}
[판결문 앞부분]
{head}
[판결문 뒷부분]
{tail}"""


def outcome_from_order(text: str, case_name: str = "") -> str | None:
    """판결 주문으로 결론을 판별한다 (근로자가 원고, 근로복지공단이 피고인 처분취소 소송 기준).

    가장 확실한 단서는 소송비용 부담자다. 진 쪽이 비용을 부담하므로
    '비용은 피고가 부담' → 근로자 승소, '원고가 부담' → 근로자 패소.
    항소·상고 기각처럼 문장만으로는 누가 이겼는지 모호한 경우도 이걸로 판별된다.
    """
    match = re.search(r"【주문】(.{0,500}?)(【|$)", text, re.S)
    if not match:
        return None
    order = match.group(1)
    # 처분취소 소송이 아니면(사업주 간 구상금 등) 원고·피고가 근로자·공단이 아니다
    if "처분" not in case_name + order:
        return None
    if "파기" in order and "환송" in order:
        return "파기환송"
    if "나머지 청구를 기각" in order or re.search(r"비용은?.{0,20}(각자|나누어|분의)", order):
        return "일부 인정"
    if re.search(r"비용은?\s*피고(들)?가?\s*부담", order):
        return "인정"
    if re.search(r"비용은?\s*원고(들)?가?\s*부담", order):
        return "불인정"
    if re.search(r"처분[^.]{0,80}?취소한다", order):
        return "인정"
    if re.search(r"원고(들)?의 청구를 (모두 )?기각", order):
        return "불인정"
    return None


def case_year(case_no: str) -> int | None:
    match = re.search(r"((?:19|20)\d{2})[가-힣]", case_no or "")
    return int(match.group(1)) if match else None


async def summarize(llm, doc: dict, semaphore: asyncio.Semaphore) -> dict:
    body = doc.get("판례내용", "")
    async with semaphore:
        summary = await llm.ainvoke(PROMPT.format(
            **{k: doc.get(k, "") or "-" for k in ["사건명", "법원명", "사건번호", "판시사항", "판결요지", "참조조문"]},
            head=body[:HEAD_CHARS],
            tail=body[-TAIL_CHARS:] if len(body) > HEAD_CHARS else "",
        ))
    result = summary.model_dump()
    # 주문으로 판별되면 LLM 판단보다 우선한다
    if rule := outcome_from_order(body, doc.get("사건명", "")):
        result["결론"] = rule
    return {
        "판례일련번호": doc["판례일련번호"],
        "사건번호": doc.get("사건번호", ""),
        "사건명": doc.get("사건명", ""),
        "법원명": doc.get("법원명", ""),
        "선고일자": doc.get("선고일자", ""),
        "연도": case_year(doc.get("사건번호", "")),
        "데이터출처명": doc.get("데이터출처명", ""),
        "참조조문_원문": doc.get("참조조문", ""),
        **result,
    }


async def main(limit: int | None):
    done = set()
    if os.path.exists(OUT_PATH):
        with open(OUT_PATH, encoding="utf-8") as f:
            done = {json.loads(line)["판례일련번호"] for line in f}

    paths = sorted(glob.glob(os.path.join(RAW_DIR, "*.json")))
    todo = []
    for path in paths:
        with open(path, encoding="utf-8") as f:
            doc = json.load(f)
        if doc["판례일련번호"] not in done:
            todo.append(doc)
    todo = todo[:limit] if limit else todo
    print(f"요약 대상 {len(todo)}건 (이미 요약 {len(done)}건)")

    llm = ChatOpenAI(model=MODEL, temperature=0).with_structured_output(PrecedentSummary)
    semaphore = asyncio.Semaphore(CONCURRENCY)
    tasks = [asyncio.create_task(summarize(llm, doc, semaphore)) for doc in todo]
    with open(OUT_PATH, "a", encoding="utf-8") as f:
        for i, task in enumerate(asyncio.as_completed(tasks), 1):
            try:
                row = await task
            except Exception as e:  # 한 건 실패로 전체를 멈추지 않는다. 다시 실행하면 이어서 요약한다
                print(f"  실패: {e}")
                continue
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            if i % 50 == 0:
                print(f"{i}/{len(todo)}", flush=True)
    print(f"완료 → {OUT_PATH}")


def relabel():
    """주문 규칙을 고쳤을 때 LLM을 다시 부르지 않고 저장된 요약의 결론만 다시 매긴다."""
    with open(OUT_PATH, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    changed = 0
    for row in rows:
        with open(os.path.join(RAW_DIR, f"{row['판례일련번호']}.json"), encoding="utf-8") as f:
            doc = json.load(f)
        rule = outcome_from_order(doc.get("판례내용", ""), doc.get("사건명", ""))
        if rule and rule != row["결론"]:
            row["결론"], changed = rule, changed + 1
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"결론 재판별: {changed}건 변경")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int)
    parser.add_argument("--relabel", action="store_true", help="주문 규칙으로 결론만 다시 매기기")
    args = parser.parse_args()
    if args.relabel:
        relabel()
    else:
        asyncio.run(main(args.limit))
