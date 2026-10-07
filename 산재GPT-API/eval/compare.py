"""
eval/results/*.json 을 모아 실험별 비교표(마크다운)를 만든다.

  python eval/compare.py                 # 전체 지표
  python eval/compare.py --type 사례형    # 특정 유형만
"""
import argparse
import csv
import glob
import json
import os
import sys

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, EVAL_DIR)

from scoring import parse_ref, score_retrieval  # noqa: E402
from summary import summarize  # noqa: E402

METRICS = ["Hit@5", "Recall@5", "핵심포인트_충족률", "인용_근거율", "평균_응답시간"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", default="전체", help="유형 (전체, 법령형, 사례형, 멀티턴, 범위밖, 데이터누락)")
    args = parser.parse_args()

    # 정답근거는 결과 파일에 저장된 값이 아니라 현재 평가셋 기준으로 채점한다 (평가셋 수정 반영)
    with open(os.path.join(EVAL_DIR, "questions.csv"), encoding="utf-8-sig") as f:
        gold = {row["id"]: row["정답근거"] for row in csv.DictReader(f)}

    # 같은 label을 여러 번 돌렸으면 가장 최근 결과만 쓴다
    latest = {}
    for path in sorted(glob.glob(os.path.join(EVAL_DIR, "results", "*.json")), key=os.path.getmtime):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        # 채점 규칙이 바뀌어도 비교가 공정하도록, 저장된 검색 결과로 Hit/Recall을 다시 계산한다
        for r in data["results"]:
            r["hit"], r["recall"] = score_retrieval(gold.get(r["id"], r["정답근거"]),
                                                    [parse_ref(x) for x in r["검색된_문서"]])
        data["summary"] = summarize(data["results"])
        latest[data["label"]] = data

    print(f"### {args.type}\n")
    print("| 실험 | " + " | ".join(METRICS) + " |")
    print("|---|" + "---|" * len(METRICS))
    for label, data in latest.items():
        row = data["summary"].get(args.type, {})
        cells = ["-" if row.get(m) is None else f"{row[m]:.2f}" for m in METRICS]
        print(f"| {label} | " + " | ".join(cells) + " |")
