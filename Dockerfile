# 대안신용평가 시스템 공용 이미지.
# api / dashboard / mlflow / bootstrap 이 모두 이 이미지를 쓰고 command 로 역할을 나눈다.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app \
    MLFLOW_DISABLE_AGENT_HINT=1

WORKDIR /app

# lightgbm/xgboost 런타임 의존성 + 데이터 내려받기용 curl + 한글 폰트
# (그림 라벨은 영문이지만, 한글 라벨로 바꿀 때 컨테이너에서 글리프가 깨지지 않도록 넣어둔다)
# fonts-nanum 만 넣으면 fc-cache 가 없어 빌드가 exit 127 로 죽는다. fontconfig 를 함께 설치한다.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        curl \
        fontconfig \
        fonts-nanum \
    && rm -rf /var/lib/apt/lists/* \
    && fc-cache -f

# 의존성 레이어를 먼저 굳혀 소스 변경 시 재설치를 피한다.
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

# 기본 커맨드는 docker-compose 의 각 서비스에서 덮어쓴다.
CMD ["python", "-c", "print('command 를 지정하세요. docker-compose.yml 참고')"]
