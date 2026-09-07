# 스켈레톤: 이번 세션에서는 빌드/실행 검증을 하지 않는다.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# lightgbm/xgboost 런타임 의존성
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

# 기본 커맨드는 docker-compose 의 각 서비스에서 덮어쓴다.
CMD ["python", "-c", "print('credit-scoring image: set a command in docker-compose.yml')"]
