"""
국가법령정보 OPEN API에서 산재보험법 관련 판례를 수집한다.

  cd 산재GPT-API
  python precedents/collect.py --list     # 1) 목록 수집 → precedents/data/list.jsonl
  python precedents/collect.py --detail   # 2) 선별한 판례 본문 수집 → precedents/data/raw/{판례일련번호}.json

수집 범위 (A안)
- 참조법령에 산업재해보상보험법이 있는 판례 중
- 대법원: 일반행정·민사 전체 (판시사항·판결요지가 있는 법리 판례)
- 근로복지공단 산재판례: 판례일련번호가 큰(최신) 순으로 MAX_COMWEL건
  목록에 사건번호·선고일자가 비어 있지만 일련번호가 클수록 최신 판례다.
  판결요지는 없고 판결문 전문(사실관계·판단·주문)만 있어 사례 데이터로 쓴다.
- 그 밖의 하급심: 일반행정, 출퇴근 재해 개정(2018) 이후 선고

목록 API의 날짜 필터(prncYd)가 제대로 동작하지 않아 목록을 전부 받은 뒤 코드에서 거른다.
과도한 호출은 이용 제한 대상이라 요청마다 REQUEST_INTERVAL초 쉰다.
"""
import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import httpx
from dotenv import load_dotenv

BASE_URL = "https://www.law.go.kr/DRF/"
LAW_NAME = "산업재해보상보험법"
REQUEST_INTERVAL = 0.6
WORKERS = 4
MAX_COMWEL = 800
LOWER_FROM = "2018.01.01"
COMWEL_SOURCE = "근로복지공단산재판례"
CASE_TYPES = {"일반행정", "민사"}

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
LIST_PATH = os.path.join(DATA_DIR, "list.jsonl")
RAW_DIR = os.path.join(DATA_DIR, "raw")

load_dotenv(os.path.join(os.path.dirname(HERE), ".env"))
OC = os.getenv("LAW_OC")


def call(path: str, **params) -> dict:
    for attempt in range(3):
        try:
            res = httpx.get(BASE_URL + path, params={"OC": OC, "type": "JSON", **params}, timeout=30)
            res.raise_for_status()
            time.sleep(REQUEST_INTERVAL)
            return res.json()
        except (httpx.HTTPError, json.JSONDecodeError) as e:
            print(f"  재시도 {attempt + 1}/3: {e}", file=sys.stderr)
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"요청 실패: {path} {params}")


def as_list(value) -> list:
    """결과가 1건이면 리스트가 아니라 dict로 오는 API 특성 처리"""
    if not value:
        return []
    return [value] if isinstance(value, dict) else value


def collect_list():
    os.makedirs(DATA_DIR, exist_ok=True)
    rows = {}
    for org, label in [("400201", "대법원"), ("400202", "하급심")]:
        page, total = 1, None
        while total is None or (page - 1) * 100 < total:
            data = call("lawSearch.do", target="prec", JO=LAW_NAME, org=org, display=100, page=page)["PrecSearch"]
            total = int(data.get("totalCnt", 0))
            for item in as_list(data.get("prec")):
                rows[item["판례일련번호"]] = {**item, "구분": label}
            print(f"{label} {min(page * 100, total)}/{total}", flush=True)
            page += 1
    with open(LIST_PATH, "w", encoding="utf-8") as f:
        for row in rows.values():
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"목록 저장: {len(rows)}건 → {LIST_PATH}")


def select_targets() -> list[dict]:
    with open(LIST_PATH, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    supreme = [r for r in rows if r["구분"] == "대법원" and r["사건종류명"] in CASE_TYPES]
    comwel = [r for r in rows if r["데이터출처명"] == COMWEL_SOURCE]
    comwel = sorted(comwel, key=lambda r: int(r["판례일련번호"]), reverse=True)[:MAX_COMWEL]
    lower = [r for r in rows if r["구분"] == "하급심" and r["데이터출처명"] != COMWEL_SOURCE
             and r["사건종류명"] == "일반행정" and r["선고일자"] >= LOWER_FROM]
    print(f"선별: 대법원 {len(supreme)}건 + 근로복지공단 {len(comwel)}건 + 기타 하급심 {len(lower)}건")
    return supreme + comwel + lower


def clean(text) -> str:
    text = re.sub(r"<br\s*/?>", "\n", str(text or ""))
    text = re.sub(r"<[^>]+>", "", text)
    return re.sub(r"[ \t]+", " ", text).strip()


def fetch_one(row: dict) -> str:
    """판례 하나를 받아 저장한다. 반환값: saved | cached | skipped"""
    path = os.path.join(RAW_DIR, f"{row['판례일련번호']}.json")
    if os.path.exists(path):  # 이어받기
        return "cached"
    data = call("lawService.do", target="prec", ID=row["판례일련번호"])
    detail = data.get("PrecService", data)
    doc = {k: clean(detail.get(k)) for k in
           ["사건명", "사건번호", "선고일자", "법원명", "사건종류명", "판결유형",
            "판시사항", "판결요지", "참조조문", "참조판례", "판례내용"]}
    doc["판례일련번호"] = row["판례일련번호"]
    doc["데이터출처명"] = row["데이터출처명"]
    if not (doc["판시사항"] or doc["판결요지"] or doc["판례내용"]):
        return "skipped"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=1)
    return "saved"


def collect_detail():
    os.makedirs(RAW_DIR, exist_ok=True)
    targets = select_targets()
    counts = {"saved": 0, "cached": 0, "skipped": 0}
    # 서버 응답이 건당 4~5초라 동시에 WORKERS건씩 요청한다 (초당 1건 안팎으로 과도한 호출은 아님)
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for i, status in enumerate(pool.map(fetch_one, targets), 1):
            counts[status] += 1
            if i % 50 == 0:
                print(f"{i}/{len(targets)} {counts}", flush=True)
    print(f"완료: {counts} → {RAW_DIR}")


if __name__ == "__main__":
    if not OC:
        sys.exit(".env에 LAW_OC를 설정하세요.")
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", action="store_true", help="판례 목록 수집")
    parser.add_argument("--detail", action="store_true", help="선별한 판례 본문 수집")
    args = parser.parse_args()
    if args.list:
        collect_list()
    if args.detail:
        collect_detail()
