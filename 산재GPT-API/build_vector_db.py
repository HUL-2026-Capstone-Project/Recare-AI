"""
docs/ 폴더의 법령 문서를 조문 단위로 잘라 FAISS 벡터 DB를 만든다.

  python build_vector_db.py            # 벡터 DB 생성 (OPENAI_API_KEY 필요)
  python build_vector_db.py --dry-run  # 임베딩 없이 청킹 결과만 확인
"""
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from pypdf import PdfReader
from dotenv import load_dotenv
import hashlib
import json
import os
import re
import sys

load_dotenv()

DOCS_PATH = "docs"
DB_PATH = "vector_db"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
MAX_ARTICLE_CHARS = 1500  # 이보다 긴 조문은 항 단위로 다시 자른다

# 법제처 PDF 머리글: "법제처   3   국가법령정보센터" / 공단 규정 PDF 쪽번호: "- 3 -"
PAGE_HEADER = re.compile(r"^\s*(법제처\s+\d+\s+국가법령정보센터|-\s*\d+\s*-)\s*$")
# 줄 맨 앞에서 시작하는 조문 제목: 제37조(업무상의 재해의 인정 기준), 제91조의2(...)
ARTICLE_HEAD = re.compile(r"^(제\d+조(?:의\d+)?)\(([^)\n]*)\)", re.MULTILINE)
# 법제처 PDF: "부칙 <제35947호,...>", 공단 규정: "부 칙" 한 줄
ADDENDA_HEAD = re.compile(r"^\s*부\s*칙\s*(<|$)", re.MULTILINE)
# 조문 사이에 끼어 있는 장/절 제목: "제7장 보칙", "제2절 요양급여"
CHAPTER_LINE = re.compile(r"^\s*제\d+(?:장|절)(?:의\d+)?\s.*$\n?", re.MULTILINE)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=800, chunk_overlap=100,
    # 항(①~⑳) 경계를 하나의 정규식으로 묶어야 ①, ② 단위의 자투리 청크가 생기지 않는다
    separators=[r"\n[①-⑳]", r"\n\n", r"\n", r" "],
    is_separator_regex=True,
    keep_separator="start",
)


def parse_filename(filename: str) -> tuple[str, str]:
    """'산업재해보상보험법 시행령(대통령령)(제35947호)(20260102).pdf' -> ('산업재해보상보험법 시행령', '20260102')"""
    stem = os.path.splitext(filename)[0]
    name = stem.split("(")[0].strip()
    dates = re.findall(r"\((\d{8})\)", stem)
    return name, dates[-1] if dates else ""


def load_text(file_path: str) -> str:
    if file_path.endswith(".pdf"):
        pages = [page.extract_text() or "" for page in PdfReader(file_path).pages]
    else:
        try:
            with open(file_path, encoding="utf-8") as f:
                pages = [f.read()]
        except UnicodeDecodeError:
            with open(file_path, encoding="cp949") as f:
                pages = [f.read()]

    cleaned = []
    for page in pages:
        lines = page.split("\n")
        if lines and PAGE_HEADER.match(lines[0]):
            lines = lines[1:]
            # 법제처 PDF는 머리글 다음 줄에 문서 제목이 반복된다
            if lines and cleaned and lines[0].strip() == cleaned[0].split("\n")[0].strip():
                lines = lines[1:]
        cleaned.append("\n".join(lines))
    # 페이지를 줄바꿈으로 이어야 페이지 경계에서 단어가 붙지 않는다
    text = "\n".join(cleaned)
    # HWP에서 변환한 규정은 "제 1조", "제 3항"처럼 띄어 쓰는 경우가 있어 법제처 표기로 맞춘다
    return re.sub(r"제\s+(\d+)\s*(조|항|호|장|절)", r"제\1\2", text)


def split_articles(text: str, law: str, base_meta: dict) -> list[Document]:
    """조문 제목을 기준으로 자르고, 각 청크 앞에 [법령명 제N조(제목)] 머리말을 붙인다."""
    # 부칙(개정 이력·경과조치)은 답변에 거의 쓰이지 않고 검색 자리만 차지해서 색인하지 않는다.
    # 시행일은 파일명에서 effective_date 메타데이터로 따로 남긴다.
    addenda = ADDENDA_HEAD.search(text)
    body = text[:addenda.start()] if addenda else text

    heads = list(ARTICLE_HEAD.finditer(body))
    docs = []
    for i, head in enumerate(heads):
        end = heads[i + 1].start() if i + 1 < len(heads) else len(body)
        article_text = CHAPTER_LINE.sub("", body[head.start():end]).strip()
        article, title = head.group(1), head.group(2)
        label = f"[{law} {article}({title})]"
        meta = {**base_meta, "article": article, "title": title}

        parts = [article_text] if len(article_text) <= MAX_ARTICLE_CHARS else splitter.split_text(article_text)
        for part in parts:
            docs.append(Document(page_content=f"{label}\n{part}", metadata=meta))

    return docs


# 별표 본문 시작: "■ 산업재해보상보험법 시행령 [별표 3]" / 표 형태는 "■산업재해보상보험법시행령[별표6]"
ANNEX_HEAD = re.compile(r"■[^\[\n]*\[별표\s*(\d+(?:의\d+)?)\]")
# 부칙 뒤의 별표 목록: "[별표 3] 업무상 질병에 대한 구체적인 인정 기준(제34조제3항 관련)"
ANNEX_TOC = re.compile(r"^\s*\[별표\s*(\d+(?:의\d+)?)\]\s*(.+?)(?:\(제\d+조[^)]*관련\s*\)?)?\s*$", re.MULTILINE)


def restore_structure(text: str) -> str:
    """표에서 추출되어 띄어쓰기가 사라진 별표에 항목 단위 줄바꿈을 넣는다.
    "제1급1.두눈이실명된사람2.말하는..." -> "제1급\n1.두눈이실명된사람\n2.말하는..." (0.02 같은 소수는 그대로)"""
    if text.count(" ") / max(len(text), 1) >= 0.05:
        return text
    text = re.sub(r"(제\d+급)", r"\n\1\n", text)
    return re.sub(r"(?<=[가-힣)])(\d{1,2})\.(?=[가-힣])", r"\n\1.", text)


SPACING_CACHE = ".cache/annex_spacing.json"
SPACING_PROMPT = """다음은 표에서 추출되면서 띄어쓰기가 모두 사라진 한국어 법령 별표입니다.
한국어 맞춤법에 맞게 띄어쓰기만 넣으세요. 글자·숫자·기호·줄바꿈은 하나도 바꾸거나 빼거나 더하지 마세요.
설명 없이 결과만 출력하세요.

{text}"""


def respace(text: str) -> str:
    """띄어쓰기가 사라진 별표에 LLM으로 띄어쓰기를 되살린다.

    모델이 내용을 바꾸면 안 되므로(장해등급표는 숫자 하나도 중요), 결과에서 공백을 모두 지웠을 때
    원문과 같을 때만 받아들이고 아니면 원문을 쓴다. 결과는 캐시해 재빌드 때 다시 호출하지 않는다.
    --dry-run에서는 캐시만 쓴다.
    """
    cache = {}
    if os.path.exists(SPACING_CACHE):
        with open(SPACING_CACHE, encoding="utf-8") as f:
            cache = json.load(f)
    # 줄 단위로 1500자 이하 조각을 만들어 보낸다 (길면 모델이 내용을 빠뜨리기 쉽다)
    pieces, current = [], ""
    for line in text.split("\n"):
        if current and len(current) + len(line) > 1500:
            pieces.append(current)
            current = ""
        current += line + "\n"
    pieces.append(current)

    keys = [hashlib.sha1(p.encode()).hexdigest() for p in pieces]
    todo = [(k, p) for k, p in zip(keys, pieces) if k not in cache]
    if todo and "--dry-run" not in sys.argv:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        outputs = llm.batch([SPACING_PROMPT.format(text=p) for _, p in todo], config={"max_concurrency": 8})
        for (key, piece), out in zip(todo, outputs):
            spaced = out.content.strip("\n") + "\n"
            same = re.sub(r"\s", "", spaced) == re.sub(r"\s", "", piece)
            cache[key] = spaced if same else piece
            if not same:
                print(f"  ⚠️ 띄어쓰기 복원 중 내용이 바뀌어 원문 유지: {piece[:30]!r}")
        os.makedirs(os.path.dirname(SPACING_CACHE), exist_ok=True)
        with open(SPACING_CACHE, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False)
    return "".join(cache.get(k, p) for k, p in zip(keys, pieces))


def split_annexes(text: str, law: str, base_meta: dict) -> list[Document]:
    """시행령 PDF 끝에 붙은 별표 본문을 별표 단위로 자른다. 별표가 없으면 빈 목록."""
    heads = list(ANNEX_HEAD.finditer(text))
    if not heads:
        return []
    titles = {m.group(1): m.group(2).strip() for m in ANNEX_TOC.finditer(text[:heads[0].start()])}

    docs = []
    for i, head in enumerate(heads):
        end = heads[i + 1].start() if i + 1 < len(heads) else len(text)
        number = head.group(1)
        title = titles.get(number, "")
        article = f"별표 {number}"
        raw = re.sub(r"<[^>]*>", "", text[head.end():end])
        body = restore_structure(raw)
        if raw.count(" ") / max(len(raw), 1) < 0.05:  # 표에서 추출돼 띄어쓰기가 사라진 별표
            body = respace(body)
        label = f"[{law} {article}({title})]" if title else f"[{law} {article}]"
        meta = {**base_meta, "article": article, "title": title}
        docs += [Document(page_content=f"{label}\n{chunk}", metadata=meta)
                 for chunk in splitter.split_text(body)]
    return docs


def split_plain(text: str, label: str, base_meta: dict) -> list[Document]:
    """조문 구조가 없거나 추출이 깨진 문서용: 일반 분할 + 문서명 머리말."""
    return [
        Document(page_content=f"[{label}]\n{chunk}", metadata=base_meta)
        for chunk in splitter.split_text(text)
    ]


def build_documents() -> list[Document]:
    all_docs = []
    for filename in sorted(os.listdir(DOCS_PATH)):
        if not filename.endswith((".pdf", ".txt")):
            continue
        law, effective = parse_filename(filename)
        text = load_text(os.path.join(DOCS_PATH, filename))
        base_meta = {"source": filename, "law": law, "effective_date": effective}

        docs = split_articles(text, law, base_meta)
        # 조문 제목이 거의 없으면(고시, 별표, 추출이 깨진 규정 등) 일반 분할로 처리
        if len(docs) < 5:
            docs = split_plain(text, law, base_meta)
        docs += split_annexes(text, law, base_meta)
        print(f"  {law}: {len(docs)}개 청크")
        all_docs.extend(docs)
    return all_docs


if __name__ == "__main__":
    documents = build_documents()
    print(f"총 {len(documents)}개 청크")

    os.makedirs(DB_PATH, exist_ok=True)
    # 검색 결과 디버깅/평가셋 작성용으로 청크 원문을 함께 저장한다
    with open(os.path.join(DB_PATH, "chunks.jsonl"), "w", encoding="utf-8") as f:
        for doc in documents:
            f.write(json.dumps({"text": doc.page_content, **doc.metadata}, ensure_ascii=False) + "\n")

    if "--dry-run" in sys.argv:
        print(f"🧪 dry-run: 임베딩 생략 ({DB_PATH}/chunks.jsonl 확인)")
        sys.exit(0)

    vectordb = FAISS.from_documents(documents, OpenAIEmbeddings(model=EMBEDDING_MODEL))
    vectordb.save_local(DB_PATH)
    print(f"✅ 벡터 DB 저장 완료! (임베딩: {EMBEDDING_MODEL})")
