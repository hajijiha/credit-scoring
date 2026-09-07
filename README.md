# credit-scoring — 대안신용평가 ML 시스템

전통 신용 데이터만으로는 평가가 어려운 **씬파일러(thin filer)** 를 대상으로,
통신·공과금 납부 이력 같은 **대안 데이터**를 결합해 신용위험을 평가하고,
그 예측의 **설명가능성(SHAP)** 과 **집단 공정성(Fairlearn)** 까지 함께 검증하는 시스템이다.

> **면책 조항**
> 본 시스템은 교육 목적으로 개발되었으며, 실제 금융 의사결정에 사용할 수 없습니다.
> 대안 데이터는 실제 관측치가 아니라 시뮬레이션으로 합성된 값입니다.

## 무엇을 하는가

| 축 | 내용 |
|---|---|
| 데이터 | GMSC(Give Me Some Credit) 150,000행 + 대안 5변수 합성, 씬파일러/편향 시뮬레이션 |
| 모델 | 전통 피처 vs 전통+대안 피처 비교, 불균형 처리, MLflow 실험 추적 |
| 판정 | `score = 1000 × (1 − p_default)`, 등급 A/B/C/D/E, 승인 임계 `p_default < 0.30` |
| 설명 | SHAP 기반 전역·개별 설명 |
| 공정성 | Fairlearn 지표(DI ratio, Equalized Odds) 측정 및 ThresholdOptimizer 완화 |
| 통계 | RepeatedStratifiedKFold(5×4) 기반 ANOVA 및 사후검정 |
| 서빙 | FastAPI(8000) + Streamlit 대시보드(8501) |

설계 결정의 근거는 [`docs/decisions/`](docs/decisions/) 의 ADR-001 ~ ADR-006 에 정리돼 있다.

## 디렉터리 구조

```
credit-scoring/
├── CLAUDE.md              # 작업 규칙 (Claude Code 세션 시작 시 참조)
├── PROGRESS.md            # 작업 로그 / 다음 할 일 / 미해결 이슈
├── README.md
├── docker-compose.yml     # api / dashboard / mlflow (현재 스켈레톤, 빌드 미검증)
├── Dockerfile             # python:3.12-slim 기반
├── requirements.txt       # 버전 고정
├── requirements.lock      # pip freeze 스냅샷 (전이 의존성 포함)
├── .env.example
├── config/
│   └── config.yaml        # 모든 설정값의 단일 출처 (포트·경로·임계값·분할·CV)
├── data/raw/              # cs-training.csv (git 제외)
├── src/
│   ├── data/              # 로더, 대안 데이터 시뮬레이터, 전처리
│   ├── models/            # 학습 파이프라인
│   ├── explainer/         # SHAP
│   ├── fairness/          # Fairlearn 측정 · 완화
│   ├── statistics/        # 반복 CV, ANOVA
│   ├── api/               # FastAPI
│   └── dashboard/         # Streamlit
├── models/                # 학습 산출물 (git 제외)
├── notebooks/
├── tests/
└── docs/decisions/        # ADR
```

## 실행 방법

현재 단계에서는 **환경 세팅과 데이터 확보까지**만 실행 가능하다. 모델·API·대시보드는 아직 구현 전이다.

### 1. 가상환경

```bash
cd ~/credit-scoring
python3 -m venv .venv            # 이미 생성돼 있다면 생략
source .venv/bin/activate
pip install -r requirements.txt  # 완전히 동일한 환경이 필요하면 requirements.lock
```

Python 3.12 기준이다. 다른 프로젝트의 venv 를 재사용하지 않는다.

### 2. 데이터 내려받기

```bash
mkdir -p data/raw
curl -L -o data/raw/cs-training.csv \
  "https://raw.githubusercontent.com/JLZml/Credit-Scoring-Data-Sets/master/3.%20Kaggle/Give%20Me%20Some%20Credit/cs-training.csv"
```

검증 기준: 150,000행 / `SeriousDlqin2yrs` 양성률 약 6.7% /
`MonthlyIncome`·`NumberOfDependents` 에 결측 존재. (원본은 git 에 커밋하지 않는다.)

### 3. 환경 변수

```bash
cp .env.example .env
```

| 변수 | 기본값 | 설명 |
|---|---|---|
| `API_BASE_URL` | `http://localhost:8000` | 대시보드가 추론을 요청할 FastAPI 주소. 대시보드는 모델 파일을 직접 로드하지 않고 반드시 이 API 를 경유한다. |
| `MLFLOW_TRACKING_URI` | `http://localhost:5000` | MLflow 트래킹 서버. 로컬 파일 백엔드는 `file:./mlruns`. |
| `APP_ENV` | `local` | 실행 환경 (`local` / `docker`). |
| `LOG_LEVEL` | `INFO` | 로그 레벨. |
| `CONFIG_PATH` | `config/config.yaml` | 설정 파일 경로 오버라이드(선택). |

`.env` 는 커밋하지 않는다 (`.gitignore` 에 포함).

### 4. 컨테이너 (예정)

`Dockerfile` 과 `docker-compose.yml` 은 서비스 이름과 포트만 잡아둔 **스켈레톤**이며 아직 빌드 검증을 하지 않았다.
현재 개발 환경(WSL2)에는 docker CLI 가 설치돼 있지 않다.

## 설정 규칙

임계값·비율·포트·경로·`random_state` 등 모든 설정값은 `config/config.yaml` 에만 둔다.
코드에 상수를 하드코딩하지 않으며, 경로는 프로젝트 루트 기준 상대경로만 사용한다.
자세한 작업 규칙은 [`CLAUDE.md`](CLAUDE.md) 참조.
