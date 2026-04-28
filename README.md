# 산재GPT API

산업재해보상보험법 기반 AI 법령 질의응답 REST API 서버입니다.

## 개요

산재보험법, 시행령, 시행규칙 등 관련 문서를 벡터 DB에 저장하고, 사용자의 질문에 대해 관련 조문을 검색하여 GPT-4o가 답변을 생성합니다. 멀티턴 대화를 지원하여 이전 맥락을 유지한 채로 대화할 수 있습니다.

## 기술 스택

- **FastAPI** — REST API 서버
- **LangChain** — LLM 파이프라인 구성
- **FAISS** — 벡터 유사도 검색 (RAG)
- **OpenAI GPT-4o** — 답변 생성
- **Pydantic** — 요청/응답 스키마 검증

## 아키텍처

```
사용자 질문
    ↓
FAISS 벡터 DB에서 유사 문서 검색 (RAG)
    ↓
검색된 문서 + 대화 이력 + 질문 → GPT-4o
    ↓
답변 반환
```

## 프로젝트 구조

```
산재GPT-API/
├── main.py              # FastAPI 앱
├── build_vector_db.py   # 벡터 DB 생성 스크립트
├── requirements.txt
├── .env                 # OPENAI_API_KEY (직접 생성)
├── docs/                # 산재 관련 PDF/TXT 문서
└── vector_db/           # FAISS 인덱스 (자동 생성)
```

## 시작하기

### 1. 패키지 설치

```bash
pip install -r requirements.txt
```

### 2. 환경 변수 설정

```bash
# .env 파일 생성
echo "OPENAI_API_KEY=your_openai_api_key" > .env
```

### 3. 벡터 DB 생성 (최초 1회)

`docs/` 폴더에 PDF 또는 TXT 문서를 넣은 후 실행합니다.

```bash
python build_vector_db.py
```

### 4. 서버 실행

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

> 운영 환경에서는 세션 저장소를 Redis로 교체하고, CORS `allow_origins`를 실제 도메인으로 제한하는 것을 권장합니다.
