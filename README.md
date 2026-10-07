# 산재GPT API

산업재해보상보험법 기반 AI 법령 질의응답 REST API 서버입니다.

## 개요

산재보험법, 시행령, 시행규칙, 고시, 요양업무처리규정을 조문 단위로 벡터 DB에 저장하고, 산재 관련 판례(대법원·근로복지공단 산재판례)를 별도 DB에 저장합니다. 사용자 질문에 대해 관련 조문과 유사 판례를 검색하여 GPT-4o가 근거와 함께 답변합니다. 멀티턴 대화와 스트리밍 응답을 지원합니다.

## 기술 스택

- **FastAPI** — REST API 서버
- **LangChain** — LLM 파이프라인 구성
- **FAISS** — 벡터 유사도 검색 (RAG)
- **OpenAI GPT-4o** — 답변 생성, 조문 목차 라우팅
- **Redis** (선택) — 대화 세션 저장
- **Pydantic** — 요청/응답 스키마 검증

## 아키텍처

```
사용자 질문 (+ 대화 이력이 있으면 독립 질문으로 재구성)
    ↓
법령 검색: 벡터 검색 + LLM이 조문 목차에서 고른 조문 → RRF로 합쳐 5개
판례 검색: 벡터 검색 + 검색된 조문을 참조하는 판례 가산 → 인정 2건 + 불인정 2건
    ↓
조문 + 판례 + 질문 → GPT-4o (판례로 결과를 예측하지 않고 판단 요소를 안내)
    ↓
답변 + 출처 반환 (검색되지 않은 사건번호는 가림)
```

## 프로젝트 구조

```
산재GPT-API/
├── main.py              # FastAPI 앱
├── retrieval.py         # 법령·판례 검색기 (목차 라우팅, RRF, 판례 균형 선택)
├── sessions.py          # 대화 세션 저장소 (메모리 / Redis)
├── build_vector_db.py   # 법령 벡터 DB 생성 (조문·별표 단위 청킹)
├── precedents/          # 판례 수집(collect.py) → 요약(summarize.py) → 판례 DB 생성(build_index.py)
├── eval/                # 평가셋, 채점·실험 스크립트, 실험 결과
├── requirements.txt
├── .env                 # API 키 (.env.example 참고, 직접 생성)
├── docs/                # 법령 PDF
├── vector_db/           # 법령 FAISS 인덱스 (자동 생성)
└── vector_db_prec/      # 판례 FAISS 인덱스 (자동 생성)
```

## 시작하기

### 1. 패키지 설치

```bash
pip install -r requirements.txt
```

### 2. 환경 변수 설정

```bash
cp .env.example .env   # 값을 채운다
```

| 변수 | 필수 | 설명 |
|---|---|---|
| `OPENAI_API_KEY` | O | OpenAI API 키 |
| `LAW_OC` | 판례 수집 시 | 국가법령정보 OPEN API 인증키 (open.law.go.kr) |
| `REDIS_URL` | | 있으면 세션을 Redis에 저장 (예: `redis://localhost:6379/0`) |

### 3. 벡터 DB 생성 (최초 1회)

`docs/` 폴더에 PDF 또는 TXT 문서를 넣은 후 실행합니다.

```bash
python build_vector_db.py            # 조문 단위로 청킹 후 임베딩
python build_vector_db.py --dry-run  # 임베딩 없이 청킹 결과만 확인 (vector_db/chunks.jsonl)
```

법령 PDF는 `제N조(제목)` 단위로 잘리며, 각 청크에 `[법령명 제N조(제목)]` 머리말과 메타데이터(법령명, 조문, 시행일)가 붙습니다. 임베딩 모델(`EMBEDDING_MODEL`, 기본 `text-embedding-3-large`)이나 청킹 방식을 바꾸면 벡터 DB를 다시 만들어야 합니다.

### 4. 판례 DB 생성 (선택)

판례 DB(`vector_db_prec/`)가 없으면 법령만으로 답변합니다.

```bash
python precedents/collect.py --list --detail   # 판례 수집 (약 20분)
python precedents/summarize.py                 # 판례별 구조화 요약 (gpt-4o-mini)
python precedents/build_index.py               # 판례 벡터 DB 생성
```

### 5. 서버 실행

```bash
uvicorn main:app --reload --port 8000
```

서버 실행 후 Swagger UI에서 바로 테스트할 수 있습니다.
- Swagger UI: `http://localhost:8000/docs`
- 헬스 체크: `http://localhost:8000/health`

## API 엔드포인트

| Method | Path | 설명 |
|--------|------|------|
| `POST` | `/chat` | 질문 전송 및 AI 답변 수신 |
| `POST` | `/chat/stream` | 같은 기능을 스트리밍(SSE)으로. `status` → `sources` → `token`… → `done` 순서 |
| `GET` | `/sessions/{session_id}` | 대화 이력 조회 |
| `DELETE` | `/sessions/{session_id}` | 세션 삭제 |
| `GET` | `/health` | 서버 상태 확인 |

## 사용 예시

### 새 대화 시작

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "업무상 재해 인정 기준이 무엇인가요?"}'
```

```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "answer": "업무상 재해는 근로자가 업무상의 사유로...",
  "sources": [
    {"law": "산업재해보상보험법", "article": "제37조", "title": "업무상의 재해의 인정 기준", "source": "산업재해보상보험법(법률)(제21375호)(20260701).pdf"}
  ],
  "turn": 1
}
```

### 이어서 질문 (멀티턴)

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "question": "출퇴근 중 사고도 해당되나요?",
    "session_id": "550e8400-e29b-41d4-a716-446655440000"
  }'
```

## 배포

```bash
# 운영 환경 실행
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

> `REDIS_URL`이 없으면 세션이 프로세스 메모리에 저장되어 `--workers`를 2 이상으로 띄울 때 멀티턴 대화가 끊깁니다. 운영 환경에서는 `REDIS_URL`을 설정하세요 (세션은 마지막 대화 후 7일 보관, 세션당 최근 50턴). CORS `allow_origins`도 실제 도메인으로 제한하는 것을 권장합니다.
