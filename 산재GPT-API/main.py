from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import AsyncIterator, Optional
from dotenv import load_dotenv
import json
import os
import re
import uuid

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate

from retrieval import CombinedRetriever, LawRetriever
from sessions import create_session_store

load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError(".env에 OPENAI_API_KEY를 설정하세요.")

# build_vector_db.py와 같은 임베딩 모델을 써야 한다
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
MAX_HISTORY_TURNS = 5
# 검색 설정 (평가 실험에서 환경변수로 바꿔가며 비교한다)
RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "5"))
FETCH_K = int(os.getenv("FETCH_K", "20"))
ROUTING = os.getenv("ROUTING", "toc")  # none | query | toc (retrieval.py 참고)
ROUTER_MODEL = os.getenv("ROUTER_MODEL", "gpt-4o")
EXCLUDE_LAWS = [x for x in os.getenv("EXCLUDE_LAWS", "").split(",") if x]
PRECEDENT_K = int(os.getenv("PRECEDENT_K", "4"))  # 0이면 판례를 쓰지 않는다
PRECEDENT_MODE = os.getenv("PRECEDENT_MODE", "balanced")  # top | balanced (retrieval.py 참고)
PRECEDENT_FETCH_K = int(os.getenv("PRECEDENT_FETCH_K", "40"))
PRECEDENT_DB_PATH = "vector_db_prec"
FALLBACK_ANSWER = "관련 규정을 문서에서 찾을 수 없습니다. 근로복지공단(1588-0075) 또는 전문 노무사에게 문의하세요."

SYSTEM_PROMPT = """
당신은 산업재해보상보험법(산재보험법) 전문 AI 어시스턴트입니다.
주어진 문서(산재보험법·시행령·시행규칙·고시 조문과 판례)를 기반으로만 답변하세요.

답변 형식:
1. 질문에 대한 결론. 원칙과 예외가 있으면 둘 다 말하세요.
2. 근거 조문 (각 문서 첫 줄의 [법령명 제○조(제목)] 표기를 그대로 사용)
3. 유사 판례 ([판례 ...] 문서가 있고 질문과 사실관계·쟁점이 비슷할 때만)
   - 사건번호, 결론(인정/불인정 등), 질문과 비슷한 점을 한 줄로
   - 인정된 판례와 불인정된 판례를 비교해 결과를 가른 요소(업무시간 기록, 기저질환·퇴행성 여부,
     의학적 소견, 사고 경위 등)를 설명하고, 사용자가 준비하면 좋은 자료를 안내
   - 판례를 근거로 이 사안이 인정될지 불인정될지 예측하거나 가능성을 단정하지 말 것.
     실제 결과는 의학적 감정과 증거에 따라 갈리기 때문 (eval/README.md 사례 결론 예측 실험 참고)
   - 선고 연도가 오래된 판례는 이후 법령 개정으로 기준이 달라졌을 수 있다고 밝히기
4. 필요 시 실무 절차 안내

주의사항:
- 문서에 없는 내용은 '관련 규정을 찾지 못했습니다'라고 명확히 안내하세요.
- 법적 판단이 필요한 사안은 반드시 '근로복지공단 또는 전문 노무사 상담을 권장합니다'를 덧붙이세요.
- 문서에 없는 조문 번호나 사건번호를 만들어내지 마세요.
- 법령상 요건에 해당하는지 설명하는 것은 좋지만, "인정될 가능성이 높다/낮다"처럼 결과를 예측하지 마세요.
- 제도·절차 이름은 문서에 나온 용어를 그대로 쓰세요. 특히 공단의 보험급여 결정에 대한 불복은
  '심사청구'(법 제103조) → '재심사청구'(법 제106조) → 행정소송 순서이며, '이의신청'이라고 쓰지 마세요.
- 답변은 항상 한국어로 작성하세요.
"""

# 사건번호 형식: 2019두44330, 2023구단64112, 2021누31087 등
CASE_NUMBER = re.compile(r"\b(?:19|20)\d{2}[가-힣]{1,3}\d{2,6}\b")

# 시스템 프롬프트는 답변 생성 단계에만 넣는다 (질문에 붙이면 검색어가 오염된다)
QA_PROMPT = PromptTemplate.from_template(
    SYSTEM_PROMPT + "\n[참고 문서]\n{context}\n\n질문: {question}\n답변:"
)

# 멀티턴: 후속 질문("그럼 61세가 넘으면요?")을 대화 없이도 이해되는 질문으로 바꾼 뒤 검색한다
CONDENSE_PROMPT = PromptTemplate.from_template(
    "다음 대화와 후속 질문을 보고, 후속 질문을 대화 없이도 이해할 수 있는 독립된 질문으로 바꾸세요.\n"
    "질문만 출력하세요.\n\n[대화]\n{chat_history}\n\n후속 질문: {question}\n독립된 질문:"
)

app = FastAPI(
    title="산재GPT API",
    description="""
산업재해보상보험법 기반 AI 질의응답 API입니다.

## 사용 방법
1. `POST /chat` — 새 대화 시작 또는 기존 세션에 메시지 전송
   - `POST /chat/stream` — 같은 기능을 스트리밍(Server-Sent Events)으로. 답변이 줄 단위로 전송됩니다
2. `GET /sessions/{session_id}` — 특정 세션의 대화 이력 조회
3. `DELETE /sessions/{session_id}` — 세션(대화 이력) 삭제

## 멀티턴 대화
- 첫 요청 시 session_id를 비워두면 새 세션 ID가 자동 발급됩니다.
- 이후 요청에 발급된 session_id를 포함하면 이전 대화 맥락을 유지합니다.
    """,
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

print("⏳ 벡터 DB 로딩 중...")
vectordb = FAISS.load_local(
    "vector_db",
    OpenAIEmbeddings(model=EMBEDDING_MODEL),
    allow_dangerous_deserialization=True,
)
llm = ChatOpenAI(model="gpt-4o", temperature=0)
law_retriever = LawRetriever(
    vectordb=vectordb,
    routing=ROUTING,
    router_llm=ChatOpenAI(model=ROUTER_MODEL, temperature=0),
    k=RETRIEVAL_K,
    fetch_k=FETCH_K,
    exclude_laws=EXCLUDE_LAWS,
)
precedent_db = None
if PRECEDENT_K and os.path.exists(PRECEDENT_DB_PATH):
    precedent_db = FAISS.load_local(
        PRECEDENT_DB_PATH,
        OpenAIEmbeddings(model=EMBEDDING_MODEL),
        allow_dangerous_deserialization=True,
    )
retriever = CombinedRetriever(
    law_retriever=law_retriever,
    precedent_db=precedent_db,
    precedent_k=PRECEDENT_K,
    precedent_fetch_k=PRECEDENT_FETCH_K,
    precedent_mode=PRECEDENT_MODE,
)
print("✅ 벡터 DB 로딩 완료!")

# REDIS_URL이 있으면 Redis, 없으면 메모리 (sessions.py 참고)
session_store = create_session_store()


class ChatRequest(BaseModel):
    question: str = Field(
        ...,
        description="사용자 질문",
        examples=["업무상 재해 인정 기준이 무엇인가요?"]
    )
    session_id: Optional[str] = Field(
        default=None,
        description="대화 세션 ID. 비워두면 새 세션이 자동 생성됩니다.",
        examples=["550e8400-e29b-41d4-a716-446655440000"]
    )


class Source(BaseModel):
    law: Optional[str] = Field(default=None, description="법령명. 판례는 '판례'")
    article: Optional[str] = Field(default=None, description="조문 번호 (예: 제37조). 판례는 사건번호")
    title: Optional[str] = Field(default=None, description="조문 제목. 판례는 '법원 사건번호 사건명 (결론)'")
    source: Optional[str] = Field(default=None, description="원본 파일명")


class ChatResponse(BaseModel):
    session_id: str = Field(description="현재 세션 ID (이후 요청에 재사용하세요)")
    answer: str = Field(description="AI 답변")
    sources: list[Source] = Field(default_factory=list, description="답변 생성에 참고한 문서")
    turn: int = Field(description="현재 대화 턴 수")


class SessionHistoryResponse(BaseModel):
    session_id: str
    history: list[dict[str, str]] = Field(
        description="대화 이력. 각 항목은 {'role': 'user'|'assistant', 'content': '...'} 형태"
    )
    turn: int


async def retrieve(question: str, history: list[tuple[str, str]]) -> tuple[str, list[Document]]:
    """(검색에 쓴 독립 질문, 검색된 문서). 대화 이력이 있으면 후속 질문을 독립 질문으로 바꾼다."""
    standalone = question
    if history:
        transcript = "\n".join(f"사용자: {q}\nAI: {a}" for q, a in history[-MAX_HISTORY_TURNS:])
        standalone = (await llm.ainvoke(CONDENSE_PROMPT.format(chat_history=transcript, question=question))).content.strip()
    return standalone, await retriever.ainvoke(standalone)


def answer_prompt(question: str, docs: list[Document]) -> str:
    return QA_PROMPT.format(context="\n\n".join(d.page_content for d in docs), question=question)


def mask_unknown_cases(text: str, docs: list[Document]) -> str:
    """검색된 판례에 없는 사건번호는 지어낸 것으로 보고 가린다."""
    known = {d.metadata["article"] for d in docs if d.metadata.get("law") == "판례"}
    return CASE_NUMBER.sub(lambda m: m.group() if m.group() in known else "(확인되지 않은 판례)", text)


def to_sources(docs: list[Document]) -> list[Source]:
    # 같은 조문이 여러 청크로 나뉘어 검색될 수 있으므로 중복 제거
    sources, seen = [], set()
    for doc in docs:
        source = Source(**{k: doc.metadata.get(k) for k in Source.model_fields})
        key = (source.law, source.article, source.source)
        if key not in seen:
            seen.add(key)
            sources.append(source)
    return sources


@app.post(
    "/chat",
    response_model=ChatResponse,
    summary="질문 전송",
    description="산재보험법 관련 질문을 전송하고 AI 답변을 받습니다. session_id를 유지하면 이전 대화 맥락이 반영됩니다.",
)
async def chat(req: ChatRequest):
    session_id = req.session_id or str(uuid.uuid4())

    try:
        history = await session_store.get(session_id)
        standalone, docs = await retrieve(req.question, history)
        answer = (await llm.ainvoke(answer_prompt(standalone, docs))).content.strip()
        answer = mask_unknown_cases(answer or FALLBACK_ANSWER, docs)
        turn = await session_store.append(session_id, req.question, answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI 처리 중 오류 발생: {str(e)}")

    return ChatResponse(session_id=session_id, answer=answer, sources=to_sources(docs), turn=turn)


def sse(event: str, data) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.post(
    "/chat/stream",
    summary="질문 전송 (스트리밍)",
    description="""/chat과 같지만 Server-Sent Events로 응답합니다. 이벤트 순서:
- `status`: 진행 상황 안내 (요청 직후, 검색 중)
- `sources`: 참고 문서 목록 (검색이 끝나면 바로)
- `token`: 답변 텍스트 조각 (문장·줄 단위)
- `done`: {session_id, turn, answer} (answer는 최종 답변 전체)
- `error`: 처리 중 오류""",
)
async def chat_stream(req: ChatRequest):
    session_id = req.session_id or str(uuid.uuid4())

    async def events() -> AsyncIterator[str]:
        try:
            history = await session_store.get(session_id)
            # 검색(목차 라우팅)이 2초가량 걸려서, 화면이 비어 있지 않도록 먼저 알린다
            yield sse("status", "관련 조문과 판례를 찾는 중입니다")
            standalone, docs = await retrieve(req.question, history)
            yield sse("sources", [s.model_dump() for s in to_sources(docs)])
            # 사건번호 가림 처리를 하려면 사건번호가 조각 사이에 걸리지 않아야 해서
            # 문장 끝('. ')이나 줄바꿈 단위로 내보낸다 (사건번호 안에는 둘 다 없다)
            buffer, parts = "", []
            async for chunk in llm.astream(answer_prompt(standalone, docs)):
                buffer += chunk.content
                cut = max(buffer.rfind("\n"), buffer.rfind(". "))
                if cut >= 0:
                    piece, buffer = buffer[:cut + 1], buffer[cut + 1:]
                    parts.append(mask_unknown_cases(piece, docs))
                    yield sse("token", parts[-1])
            if buffer:
                parts.append(mask_unknown_cases(buffer, docs))
                yield sse("token", parts[-1])
            answer = "".join(parts).strip() or FALLBACK_ANSWER
            turn = await session_store.append(session_id, req.question, answer)
            yield sse("done", {"session_id": session_id, "turn": turn, "answer": answer})
        except Exception as e:
            yield sse("error", {"detail": f"AI 처리 중 오류 발생: {e}"})

    return StreamingResponse(events(), media_type="text/event-stream")


@app.get(
    "/sessions/{session_id}",
    response_model=SessionHistoryResponse,
    summary="대화 이력 조회",
    description="session_id로 저장된 대화 이력 전체를 반환합니다.",
)
async def get_session(session_id: str):
    turns = await session_store.get(session_id)
    if not turns:
        raise HTTPException(status_code=404, detail="해당 세션을 찾을 수 없습니다.")

    history = []
    for user_msg, ai_msg in turns:
        history.append({"role": "user", "content": user_msg})
        history.append({"role": "assistant", "content": ai_msg})

    return SessionHistoryResponse(session_id=session_id, history=history, turn=len(turns))


@app.delete(
    "/sessions/{session_id}",
    summary="세션 삭제",
    description="대화 이력을 초기화합니다. 새 대화를 시작하고 싶을 때 사용하세요.",
)
async def delete_session(session_id: str):
    if not await session_store.delete(session_id):
        raise HTTPException(status_code=404, detail="해당 세션을 찾을 수 없습니다.")
    return {"message": f"세션 {session_id} 삭제 완료"}


@app.get("/health", summary="헬스 체크", description="서버 상태를 확인합니다.")
async def health():
    return {
        "status": "ok",
        "vector_db": "loaded",
        "precedent_db": "loaded" if precedent_db else "off",
        "session_store": await session_store.ping(),
    }
