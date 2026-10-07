FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

# 서버 실행에 필요한 코드만 넣는다. 벡터 DB(vector_db, vector_db_prec)는 이미지에 넣지 않고
# docker-compose.yml에서 볼륨으로 연결한다 (데이터를 바꿀 때 이미지를 다시 빌드하지 않아도 되게)
COPY main.py retrieval.py sessions.py ./

RUN useradd --create-home app
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=4)"

# 세션이 Redis에 있으므로 워커를 여러 개 띄워도 멀티턴이 유지된다
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
