from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
from dotenv import load_dotenv
import os
import uuid

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain_core.prompts import PromptTemplate

from retrieval import LawRetriever

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
FALLBACK_ANSWER = "관련 규정을 문서에서 찾을 수 없습니다. 근로복지공단(1588-0075) 또는 전문 노무사에게 문의하세요."

SYSTEM_PROMPT = """
당신은 산업재해보상보험법(산재보험법) 전문 AI 어시스턴트입니다.
주어진 문서(산재보험법, 시행령, 시행규칙 등)를 기반으로만 답변하세요.

답변 형식:
1. 질문에 직접 답변
2. 해당 근거 조문 또는 기준 명시 (각 문서 첫 줄의 [법령명 제○조(제목)] 표기를 그대로 사용)
3. 필요 시 실무 절차 안내

주의사항:
- 문서에 없는 내용은 '관련 규정을 찾지 못했습니다'라고 명확히 안내하세요.
- 법적 판단이 필요한 사안은 반드시 '근로복지공단 또는 전문 노무사 상담을 권장합니다'를 덧붙이세요.
- 문서에 없는 조문 번호를 만들어내지 마세요.
- 답변은 항상 한국어로 작성하세요.
"""

# 시스템 프롬프트는 답변 생성 단계에만 넣는다 (질문에 붙이면 검색어가 오염된다)
QA_PROMPT = PromptTemplate.from_template(
    SYSTEM_PROMPT + "\n[참고 문서]\n{context}\n\n질문: {question}\n답변:"
)

app = FastAPI(
    title="산재GPT API",
    description="""
산업재해보상보험법 기반 AI 질의응답 API입니다.

## 사용 방법
1. `POST /chat` — 새 대화 시작 또는 기존 세션에 메시지 전송
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
retriever = LawRetriever(
    vectordb=vectordb,
    routing=ROUTING,
    router_llm=ChatOpenAI(model=ROUTER_MODEL, temperature=0),
    k=RETRIEVAL_K,
    fetch_k=FETCH_K,
    exclude_laws=EXCLUDE_LAWS,
)
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    combine_docs_chain_kwargs={"prompt": QA_PROMPT},
    return_source_documents=True,
    verbose=False,
)
print("✅ 벡터 DB 로딩 완료!")

# 구조: { session_id: [ (user_msg, ai_msg), ... ] }
session_store: dict[str, list[tuple[str, str]]] = {}


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
    law: Optional[str] = Field(default=None, description="법령명")
    article: Optional[str] = Field(default=None, description="조문 번호 (예: 제37조)")
    title: Optional[str] = Field(default=None, description="조문 제목")
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


@app.post(
    "/chat",
    response_model=ChatResponse,
    summary="질문 전송",
    description="산재보험법 관련 질문을 전송하고 AI 답변을 받습니다. session_id를 유지하면 이전 대화 맥락이 반영됩니다.",
)
async def chat(req: ChatRequest):
    session_id = req.session_id or str(uuid.uuid4())
    if session_id not in session_store:
        session_store[session_id] = []

    chat_history = session_store[session_id]

    try:
        result = await qa_chain.ainvoke({
            "question": req.question,
            "chat_history": chat_history[-MAX_HISTORY_TURNS:],
        })
        answer = result["answer"].strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI 처리 중 오류 발생: {str(e)}")

    if not answer:
        answer = FALLBACK_ANSWER

    # 같은 조문이 여러 청크로 나뉘어 검색될 수 있으므로 중복 제거
    sources, seen = [], set()
    for doc in result.get("source_documents", []):
        source = Source(**{k: doc.metadata.get(k) for k in Source.model_fields})
        key = (source.law, source.article, source.source)
        if key not in seen:
            seen.add(key)
            sources.append(source)

    session_store[session_id].append((req.question, answer))

    return ChatResponse(
        session_id=session_id,
        answer=answer,
        sources=sources,
        turn=len(session_store[session_id]),
    )


@app.get(
    "/sessions/{session_id}",
    response_model=SessionHistoryResponse,
    summary="대화 이력 조회",
    description="session_id로 저장된 대화 이력 전체를 반환합니다.",
)
async def get_session(session_id: str):
    if session_id not in session_store:
        raise HTTPException(status_code=404, detail="해당 세션을 찾을 수 없습니다.")

    history = []
    for user_msg, ai_msg in session_store[session_id]:
        history.append({"role": "user", "content": user_msg})
        history.append({"role": "assistant", "content": ai_msg})

    return SessionHistoryResponse(
        session_id=session_id,
        history=history,
        turn=len(session_store[session_id]),
    )


@app.delete(
    "/sessions/{session_id}",
    summary="세션 삭제",
    description="대화 이력을 초기화합니다. 새 대화를 시작하고 싶을 때 사용하세요.",
)
async def delete_session(session_id: str):
    if session_id not in session_store:
        raise HTTPException(status_code=404, detail="해당 세션을 찾을 수 없습니다.")
    del session_store[session_id]
    return {"message": f"세션 {session_id} 삭제 완료"}


@app.get("/health", summary="헬스 체크", description="서버 상태를 확인합니다.")
async def health():
    return {"status": "ok", "vector_db": "loaded"}
