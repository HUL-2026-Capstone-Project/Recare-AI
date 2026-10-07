"""
판례 요약(precedents/data/summaries.jsonl)으로 판례 전용 벡터 DB를 만든다 → vector_db_prec/

  cd 산재GPT-API
  python precedents/build_index.py                    # 서비스용: 판례 전부
  python precedents/build_index.py --exclude-holdout  # 사례 결론 예측 실험용: 시험 판례 제외

법령 벡터 DB와 따로 두는 이유: 판례가 법령 조각보다 훨씬 많아 한 인덱스에 섞으면
조문이 검색 결과에서 밀려난다. 검색할 때 법령 K개 + 판례 K개를 따로 뽑아 합친다.
"""
import csv
import json
import os
import re
import sys

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SUMMARY_PATH = os.path.join(HERE, "data", "summaries.jsonl")
RAW_DIR = os.path.join(HERE, "data", "raw")
DB_PATH = os.path.join(ROOT, "vector_db_prec")
HOLDING_CHARS = 800  # 대법원 판결요지는 법리 설명이라 일부를 함께 넣는다
# 사례 결론 예측 실험(eval/make_case_holdout.py)의 시험 판례. --exclude-holdout일 때 정답이 새지 않게 뺀다
HOLDOUT_PATH = os.path.join(ROOT, "eval", "case_holdout.csv")

load_dotenv(os.path.join(ROOT, ".env"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")

# "산업재해보상보험법 시행령 제34조 제3항", "구 산업재해보상보험법(2010. 1. 27. 개정 전) 제37조의2"
STATUTE_REF = re.compile(r"산업재해보상보험법(?:\([^)]*\))?\s*(시행령|시행규칙)?(?:\([^)]*\))?\s*제\s*(\d+)\s*조(?:\s*의\s*(\d+))?")


def normalize_statutes(*texts: str) -> list[str]:
    """판례의 관련 조문을 법령 DB의 (법령명, 조문) 표기로 맞춘다. 산재보험법령만 남긴다."""
    found = []
    for text in texts:
        for sub, num, sub_num in STATUTE_REF.findall(text or ""):
            law = "산업재해보상보험법" + (f" {sub}" if sub else "")
            article = f"제{num}조" + (f"의{sub_num}" if sub_num else "")
            if (ref := f"{law} {article}") not in found:
                found.append(ref)
    return found


def to_document(row: dict, raw: dict) -> Document:
    court_case = f"{row['법원명']} {row['사건번호']}".strip()
    lines = [
        f"[판례 {court_case}" + (f" ({row['연도']}년)" if row.get("연도") else "") + f" {row['사건명']}]",
        f"결론: {row['결론']} | 재해유형: {row['재해유형']} | 상병: {row['상병'] or '-'} | 직종: {row['직종'] or '-'}",
        f"사실관계: {row['사실관계']}",
        f"쟁점: {row['쟁점']}",
        f"판단: {row['판단']}",
    ]
    if raw.get("판결요지"):
        lines.append(f"판결요지: {raw['판결요지'][:HOLDING_CHARS]}")
    statutes = normalize_statutes(" ".join(row.get("관련조문", [])), row.get("참조조문_원문", ""))
    metadata = {
        # 법령 문서와 같은 키(law/article/title)를 써서 API 응답의 sources에 그대로 실린다
        "law": "판례",
        "article": row["사건번호"],
        "title": f"{court_case} {row['사건명']} ({row['결론']})",
        "source": f"판례일련번호 {row['판례일련번호']}",
        "판례일련번호": row["판례일련번호"],
        "법원명": row["법원명"],
        "연도": row.get("연도"),
        "결론": row["결론"],
        "재해유형": row["재해유형"],
        "관련조문": statutes,
    }
    return Document(page_content="\n".join(lines), metadata=metadata)


def load_holdout() -> tuple[set[str], set[str]]:
    """(제외할 판례일련번호, 제외할 사건번호). 시험 판례와 그 1·2·3심 연관 판결을 모두 뺀다."""
    if not os.path.exists(HOLDOUT_PATH):
        return set(), set()
    with open(HOLDOUT_PATH, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    ids = {r["판례일련번호"] for r in rows}
    case_numbers = {r["사건번호"] for r in rows}
    case_numbers |= {c for r in rows for c in r["연관사건번호"].split(";") if c}
    return ids, case_numbers


if __name__ == "__main__":
    with open(SUMMARY_PATH, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    holdout_ids, holdout_cases = load_holdout() if "--exclude-holdout" in sys.argv else (set(), set())
    docs, excluded = [], 0
    for row in rows:
        with open(os.path.join(RAW_DIR, f"{row['판례일련번호']}.json"), encoding="utf-8") as f:
            raw = json.load(f)
        related = set(re.findall(r"(?:19|20)\d{2}[가-힣]{1,3}\d+", raw.get("판례내용", "")[:300]))  # 【연관판결】
        if row["판례일련번호"] in holdout_ids or row["사건번호"] in holdout_cases or related & holdout_cases:
            excluded += 1
            continue
        docs.append(to_document(row, raw))
    print(f"판례 {len(docs)}건 (시험용·연관 판결 {excluded}건 제외)")
    if "--dry-run" in sys.argv:
        print(docs[0].page_content, "\n", docs[0].metadata)
        sys.exit(0)
    # 임베딩 API는 요청당 30만 토큰 제한이 있어 200건씩 나눠 보낸다
    FAISS.from_documents(docs, OpenAIEmbeddings(model=EMBEDDING_MODEL, chunk_size=200)).save_local(DB_PATH)
    print(f"✅ 판례 벡터 DB 저장 → {DB_PATH}")
