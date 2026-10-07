"""
사례 결론 예측 실험용 시험 문항을 만든다 → eval/case_holdout.csv

  cd 산재GPT-API
  python eval/make_case_holdout.py

근로복지공단 판례 중 인정 N건 + 불인정 N건을 뽑아, 판결문의 '처분의 경위'(사실관계)만으로
재해자가 상담 챗봇에 묻는 질문을 만든다. 이 판례들(과 연관 1·2심 판결)은 판례 벡터 DB에서
제외되므로(precedents/build_index.py) 시스템은 정답 판결을 볼 수 없다.

답안 유출 방지
- 질문 생성에는 '1. 처분의 경위' 부분만 넣는다 (법원의 판단·결론 부분은 넣지 않음)
- 같은 사건의 연관 판결(【연관판결】에 적힌 사건번호)도 함께 DB에서 뺀다
"""
import csv
import json
import os
import random
import re

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(EVAL_DIR)
SUMMARY_PATH = os.path.join(ROOT, "precedents", "data", "summaries.jsonl")
RAW_DIR = os.path.join(ROOT, "precedents", "data", "raw")
OUT_PATH = os.path.join(EVAL_DIR, "case_holdout.csv")
PER_CLASS = 20
SEED = 42
FACT_CHARS = 2000
MIN_FACT_CHARS = 200

load_dotenv(os.path.join(ROOT, ".env"))

PROMPT = """아래는 산재 소송 판결문 중 '처분의 경위' 부분입니다.
재해자 본인(사망 사건이면 유족)이 산재 상담 챗봇에 상담하는 질문을 만들어 주세요.

- 1인칭 일상 말투로 4~6문장. 직종, 하던 일, 사고·발병 경위, 진단명, 근무 시간 등 사실관계를 담으세요.
- 근로복지공단이 불승인·부지급했다는 사실은 넣어도 됩니다.
- 법원의 판단, 감정 결과의 결론, 소송 결과를 암시하는 내용은 절대 넣지 마세요.
- 이름·회사명·날짜 같은 고유정보는 빼세요.
- 아래 내용에 없는 사실(근무시간, 진단명, 사고 경위 등)은 절대 지어내지 마세요. 정보가 적으면 짧게 써도 됩니다.
- 마지막 문장은 "산재로 인정받을 수 있을까요?"로 끝내세요.

[처분의 경위]
{facts}"""


def related_case_numbers(body: str) -> list[str]:
    """'【연관판결】서울고등법원,2021누31087,2심-대법원,2021두51102,3심' → ['2021누31087', '2021두51102']"""
    match = re.search(r"【연관판결】([^【]*)", body)
    return re.findall(r"(?:19|20)\d{2}[가-힣]{1,3}\d+", match.group(1)) if match else []


def facts_section(body: str) -> str | None:
    """【이유】의 '1. 처분의 경위'부터 '2.' 전까지만 잘라낸다.
    이 구간을 정확히 찾지 못하면 None (판결 이유 전체를 쓰면 법원 판단이 새어 나간다)."""
    reason = body.split("【이유】", 1)[-1]
    # 다음 단락은 제목으로 찾는다. 그냥 "2."로 찾으면 날짜("2019. 1. 2.")에서 잘린다
    match = re.search(r"1\.\s*(?:처분|이 사건 처분|청구취지 기재 처분)[^\n]{0,40}?경위(.*?)\s2\.\s*"
                      r"(?:이 사건 처분의|처분의|원고의 주장|당사자의 주장|관계 법령|관련 법령|판단|쟁점)", reason, re.S)
    if not match or len(match.group(1).strip()) < MIN_FACT_CHARS:
        return None
    return match.group(1)[:FACT_CHARS].strip()


if __name__ == "__main__":
    with open(SUMMARY_PATH, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    # 1심 행정소송(구단·구합)만 쓴다. 항소심·상고심은 '처분의 경위'가 없거나 1심을 인용만 한다
    pool = []
    for r in rows:
        if not (r["데이터출처명"] == "근로복지공단산재판례" and (r["연도"] or 0) >= 2021
                and r["재해유형"] in ("업무상 사고", "업무상 질병", "출퇴근 재해")
                and re.search(r"구[단합]", r["사건번호"])):
            continue
        with open(os.path.join(RAW_DIR, f"{r['판례일련번호']}.json"), encoding="utf-8") as f:
            if facts_section(json.load(f)["판례내용"]):
                pool.append(r)
    random.seed(SEED)
    picked = []
    for label in ("인정", "불인정"):
        candidates = [r for r in pool if r["결론"] == label]
        picked += random.sample(candidates, PER_CLASS)

    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    out = []
    for i, row in enumerate(picked, 1):
        with open(os.path.join(RAW_DIR, f"{row['판례일련번호']}.json"), encoding="utf-8") as f:
            body = json.load(f)["판례내용"]
        question = llm.invoke(PROMPT.format(facts=facts_section(body))).content.strip()
        out.append({
            "id": f"P{i:02d}",
            "판례일련번호": row["판례일련번호"],
            "사건번호": row["사건번호"],
            "연관사건번호": ";".join(related_case_numbers(body)),
            "재해유형": row["재해유형"],
            "실제결론": row["결론"],
            "질문": question,
        })
        print(f"{i}/{len(picked)} {row['사건번호']} ({row['결론']})", flush=True)

    with open(OUT_PATH, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=out[0].keys())
        writer.writeheader()
        writer.writerows(out)
    print(f"저장 → {OUT_PATH}")
