"""
법령 검색기.

검색 결과 목록을 여러 개 만든 뒤 순위 기반으로 합친다 (Reciprocal Rank Fusion).
  - 벡터 검색: 원래 질문으로 FETCH_K개
  - ROUTING=query : LLM이 질문을 법률 용어 검색어로 바꿔 추가 벡터 검색
  - ROUTING=toc   : LLM이 조문 제목 목차를 보고 필요한 조문을 직접 고름
같은 조문의 조각은 하나만 남겨 서로 다른 조문 K개를 돌려준다.
"""
import re
from collections import defaultdict

import numpy as np

from langchain_core.callbacks import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.retrievers import BaseRetriever
from langchain_community.vectorstores import FAISS
from pydantic import BaseModel, Field, PrivateAttr

RRF_K = 60  # RRF 표준 상수. 순위가 낮은 문서의 점수를 완만하게 깎는다

# 주의: 프롬프트 예시에 평가셋 질문과 겹치는 표현을 넣지 말 것 (평가 점수가 부풀려진다)
QUERY_PROMPT = """다음은 산업재해보상보험(산재보험) 관련 사용자 질문입니다.
산재보험법·시행령·시행규칙·고시의 조문을 검색하기 위한 검색어를 만드세요.

- 일상 표현을 조문에 실제로 쓰이는 법률 용어로 바꾸세요.
  예) "통근버스" → "사업주가 제공한 교통수단", "회사가 산재보험을 안 들었어요" → "보험관계의 성립"
- 질문의 쟁점이 여러 개면 쟁점마다 검색어를 하나씩 만드세요.
- 검색어는 1~3개, 각각 한 줄의 명사구로 쓰세요.

질문: {question}"""

TOC_PROMPT = """다음은 산업재해보상보험 관련 법령의 조문 목차입니다.

{toc}

사용자 질문에 답하려면 어떤 조문을 읽어야 하는지 위 목차에서 고르세요.
- 일상 표현을 법률 개념으로 바꿔 생각하세요. 질문에 나온 단어와 조문 제목의 단어가 달라도 됩니다.
- 원칙 조문과 그 세부 기준(시행령·시행규칙·고시)이 함께 필요하면 둘 다 고르세요.
- 관련성이 높은 순서로 최대 {n}개를 목차에 적힌 그대로 쓰세요.

질문: {question}"""


class RewrittenQueries(BaseModel):
    queries: list[str] = Field(description="법률 용어로 바꾼 검색어 1~3개")


class SelectedArticles(BaseModel):
    articles: list[str] = Field(description="목차에서 고른 항목을 적힌 그대로. 예: '산업재해보상보험법 제37조'")


def doc_key(doc: Document) -> tuple:
    """같은 조문의 조각을 하나로 묶는 키.
    조문 정보가 없는 문서(고시 등)와 별표는 조각마다 내용이 달라 청크 단위로 구분한다
    (예: 별표 6 장해등급은 손가락 항목이 5급~14급 조각에 흩어져 있다)."""
    md = doc.metadata
    if md.get("article") and not md["article"].startswith("별표"):
        return (md.get("law"), md["article"])
    return (md.get("law"), md.get("article"), doc.page_content[:80])


def fuse(ranked_lists: list[list[Document]], k: int) -> list[Document]:
    scores, first_doc = defaultdict(float), {}
    for docs in ranked_lists:
        seen = set()
        for rank, doc in enumerate(docs):
            key = doc_key(doc)
            if key in seen:  # 한 검색 결과 안에서는 조문당 가장 높은 순위만 반영
                continue
            seen.add(key)
            scores[key] += 1 / (RRF_K + rank + 1)
            first_doc.setdefault(key, doc)
    top = sorted(scores, key=scores.get, reverse=True)[:k]
    return [first_doc[key] for key in top]


class LawRetriever(BaseRetriever):
    vectordb: FAISS
    routing: str = "none"  # none | query | toc
    router_llm: BaseChatModel | None = None
    k: int = 5
    fetch_k: int = 20
    exclude_laws: list[str] = []

    _positions: dict = PrivateAttr(default_factory=dict)  # (법령명, 조문) -> FAISS 인덱스 위치들
    _vectors: np.ndarray = PrivateAttr(default=None)
    _toc: str = PrivateAttr(default="")

    def model_post_init(self, __context) -> None:
        # 목차 라우팅용 준비: 조문별 청크 위치, 저장된 벡터, 목차 문자열
        index, docstore = self.vectordb.index, self.vectordb.docstore
        self._vectors = index.reconstruct_n(0, index.ntotal)
        titles = set()
        for pos, doc_id in self.vectordb.index_to_docstore_id.items():
            md = docstore.search(doc_id).metadata
            if md.get("law") in self.exclude_laws:
                continue
            self._positions.setdefault((md.get("law"), md.get("article")), []).append(pos)
            label = md["law"]
            if md.get("article"):
                label += f" {md['article']}" + (f"({md['title']})" if md.get("title") else "")
            titles.add(label)
        self._toc = "\n".join(sorted(titles))

    def _search_kwargs(self) -> dict:
        kwargs = {"k": self.fetch_k}
        if self.exclude_laws:
            kwargs.update(filter=lambda md: md.get("law") not in self.exclude_laws, fetch_k=self.fetch_k * 4)
        return kwargs

    def _lookup(self, label: str, query_vector: np.ndarray) -> list[Document]:
        """목차 항목('산업재해보상보험법 시행령 별표 6(장해등급의 기준)')을 그 조문에서
        질문과 가장 가까운 청크로 바꾼다. 긴 조문·별표에서 엉뚱한 조각을 고르지 않기 위해서다.
        별표는 조각마다 내용이 달라 가까운 순으로 2개를 돌려준다."""
        label = re.sub(r"\(.*$", "", label.strip())
        match = re.match(r"(.+?)\s+(제\d+조(?:의\d+)?|별표\s*\d+(?:의\d+)?)$", label)
        key = (match.group(1), re.sub(r"별표\s*", "별표 ", match.group(2))) if match else (label, None)
        positions = self._positions.get(key)
        if not positions:
            return []
        distances = ((self._vectors[positions] - query_vector) ** 2).sum(axis=1)
        n = 2 if key[1] and key[1].startswith("별표") else 1
        best = [positions[i] for i in distances.argsort()[:n]]
        return [self.vectordb.docstore.search(self.vectordb.index_to_docstore_id[p]) for p in best]

    def _routing_prompt(self, query: str):
        if self.routing == "query":
            return RewrittenQueries, QUERY_PROMPT.format(question=query)
        return SelectedArticles, TOC_PROMPT.format(toc=self._toc, question=query, n=self.k)

    def _search(self, query_vector: list[float]) -> list[Document]:
        return self.vectordb.similarity_search_by_vector(query_vector, **self._search_kwargs())

    def _merge(self, query_vector, routed, extra_searches: list[list[Document]]) -> list[Document]:
        ranked = [self._search(query_vector), *extra_searches]
        if isinstance(routed, SelectedArticles):
            vec = np.array(query_vector, dtype="float32")
            ranked.append([d for label in routed.articles for d in self._lookup(label, vec)])
        return fuse(ranked, self.k)

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        query_vector = self.vectordb.embeddings.embed_query(query)
        if self.routing == "none":
            return fuse([self._search(query_vector)], self.k)
        schema, prompt = self._routing_prompt(query)
        routed = self.router_llm.with_structured_output(schema).invoke(prompt)
        extra = [self.vectordb.similarity_search(q, **self._search_kwargs())
                 for q in getattr(routed, "queries", [])]
        return self._merge(query_vector, routed, extra)

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun
    ) -> list[Document]:
        query_vector = await self.vectordb.embeddings.aembed_query(query)
        if self.routing == "none":
            return fuse([self._search(query_vector)], self.k)
        schema, prompt = self._routing_prompt(query)
        routed = await self.router_llm.with_structured_output(schema).ainvoke(prompt)
        extra = [await self.vectordb.asimilarity_search(q, **self._search_kwargs())
                 for q in getattr(routed, "queries", [])]
        return self._merge(query_vector, routed, extra)
