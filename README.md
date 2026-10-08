# credit-scoring — XAI 기반 대안신용평가 시스템


**프로젝트 자료:** [실험 보고서 PDF](docs/experiment_report.pdf) ·
[실험 보고서 HTML](docs/experiment_report.html) · [설계 명세](docs/SPEC.md) ·
[설계 결정 12건](docs/decisions/)

전통 신용 데이터만으로는 평가가 어려운 **씬파일러(thin filer)** 를 대상으로,
통신·공과금 납부 이력 같은 **대안 데이터**를 결합해 신용위험을 평가하고,
그 예측의 **설명가능성(SHAP)** 과 **집단 공정성(Fairlearn)** 까지 함께 검증하는 시스템이다.

> ## ⚠️ 면책 조항
> **본 시스템은 교육 목적으로 개발되었으며, 실제 금융 의사결정에 사용할 수 없습니다.**
> 대안 데이터는 실제 관측치가 아니라 시뮬레이션으로 합성된 값이며,
> 아래 성능 수치는 시뮬레이터 구조에 기인한 상한이다([한계](#한계) 참조).

## 개발 범위

데이터 시뮬레이션부터 모델링, 설명가능성·공정성 분석, API와 대시보드까지 구현한 개인 프로젝트다.

| 구현 구성 | 주요 경로·도구 |
|---|---|
| 데이터 파이프라인 · 시뮬레이터 | `src/data/` |
| 모델링 · 실험 추적 | `src/models/`, MLflow |
| XAI · 공정성 · 통계 | `src/explainer/`, `src/fairness/`, `src/statistics/` |
| 서빙 · 대시보드 | `src/api/`, `src/dashboard/` |
| 설계 문서 · 리포트 | `docs/decisions/`(ADR 12건), `docs/report_assets/` |

---

## 성능 요약

챔피언 모델: **`xgboost` / `combined` / `class_weight`** (test 22,500행 기준)

| 지표 | 값 | SPEC 목표 | 판정 |
|---|---:|---|---|
| Test AUC-ROC | **0.9453** | ≥ 0.78 | ✅ |
| Test KS | **0.7502** | ≥ 0.28 | ✅ |
| PSI (train↔test) | **0.00054** | < 0.1 | ✅ |
| 씬파일러 AUC uplift | **+0.1518** | ≥ +0.03 | ✅ |
| Precision / Recall / F1 | 0.2899 / 0.8936 / 0.4378 | — | [주의](#한계) |
| `/predict` 응답 시간 | **20.3ms** (평균, 50회) | ≤ 2초 | ✅ |

**피처 그룹별 Test AUC** — 대안 데이터의 기여

| 피처 그룹 | 최고 Test AUC |
|---|---:|
| traditional_only | 0.8355 |
| alternative_only | 0.8902 |
| **combined** | **0.9460** |

**씬파일러 유형별 uplift** (XGBoost, traditional_only → combined)

| 세그먼트 | uplift |
|---|---:|
| 일반 | +0.0946 |
| 씬파일러 (이력부족만) | +0.1013 |
| 씬파일러 (연체기록 없음) | +0.1681 |
| **씬파일러 (둘 다)** | **+0.1883** |

전통 피처의 결손이 클수록 대안 데이터의 기여가 커진다.

**공정성** (승인 임계값 0.30, 전체 승인율 0.7940)

| 보호 속성 | DI (SPEC 정의) | DI (min/max) | EO TPR차 | EO FPR차 | 완화 후 DI | 완화 후 승인율 |
|---|---:|---:|---:|---:|---:|---:|
| gender | 0.9191 | 0.9191 | 0.0350 | 0.0696 | **0.9942** | 0.8196 |
| age_group | 0.8375 | **0.7747** ❌ | 0.0862 | **0.1487** ❌ | **0.9433** | 0.8492 |

보호 속성을 학습 피처에서 제외했는데도(ADR-008) 편향이 대안 피처를 경유해 전파된다.
자세한 분석은 [`docs/report_assets/fairness_narrative.md`](docs/report_assets/fairness_narrative.md).

**ANOVA** (ADR-011 통제, 그룹당 n=20)

| 검증 | F | p | η² | 해석 |
|---|---:|---:|---:|---|
| 검증 1 (피처 그룹) | 4245.65 | 9.64e-63 | 0.9933 | Large |
| 검증 2 (알고리즘) | 804.38 | 1.68e-42 | 0.9658 | Large |

검증 2 의 Tukey HSD 에서 **XGBoost vs LightGBM 은 유의차가 없다**(p_adj=0.6468). ADR-012 참조.

---

## 실행 방법

### 사전 준비

- Docker 및 Docker Compose (검증 환경: Docker 29.2.1 / Compose v5.1.0)
- 또는 Python 3.12 + `pip`

### 1) 학습 산출물 생성 (최초 1회, 약 15분)

`models/`, `data/`, `mlruns/` 는 `.gitignore` 대상이라 **클론 직후에는 존재하지 않는다.**
따라서 서비스를 띄우기 전에 산출물을 한 번 만들어야 한다.

```bash
docker compose --profile setup run --rm bootstrap
```

로컬 파이썬으로 하려면:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt      # 완전히 동일한 환경이 필요하면 requirements.lock
bash scripts/bootstrap.sh
```

`scripts/bootstrap.sh` 가 하는 일: 데이터 다운로드 → 시뮬레이션 → 전처리 → 학습(18조합 × 20 fold)
→ 씬파일러 uplift → ANOVA → SHAP → 공정성 → 리포트 자료.
이미 있는 산출물은 건너뛴다(`FORCE=1` 로 강제 재생성).

### 2) 서비스 기동

```bash
docker compose up
```

| 서비스 | 주소 | 내용 |
|---|---|---|
| API | http://localhost:8000/docs | FastAPI Swagger UI (7개 엔드포인트) |
| 대시보드 | http://localhost:8501 | Streamlit 심사 대시보드 (5개 탭) |
| MLflow | http://localhost:5000 | 실험 추적 · Model Registry |

### 3) 리포트 수치 대조 (선택)

산출물이 준비된 뒤, 리포트(`docs/report/*.md`)에 인용한 수치를 산출물에서 다시 계산해 대조한다.

```bash
python scripts/verify_report_numbers.py          # 전체
python scripts/verify_report_numbers.py --chapters 3 6   # 일부 장만
```

### 학습 산출물 관리

`docker-compose.yml`은 프로젝트 루트를 `/app`에 바인드 마운트한다.
모델과 분석 산출물은 bootstrap으로 생성하며, 이미지 재빌드 후에도 재사용된다.
산출물이 없으면 `/health`는 `degraded`를 반환하고 대시보드에 초기화 안내가 표시된다.

---

## 데이터 획득

기본 데이터셋은 **Give Me Some Credit (GMSC)** 이다. 150,000행 / 11컬럼, 부도율 6.68%.

```bash
# 방법 1: 부트스트랩 스크립트가 자동으로 받는다 (공개 미러)
bash scripts/bootstrap.sh

# 방법 2: Kaggle 원본 (계정과 API 토큰 필요)
pip install kaggle
kaggle competitions download -c GiveMeSomeCredit
unzip GiveMeSomeCredit.zip -d data/raw/
```

> Kaggle 원본[^1]은 대회 규정 동의와 API 인증이 필요해 무인 실행이 불가능하다.
> 그래서 부트스트랩은 동일한 파일을 제공하는 공개 미러[^2]에서 내려받는다.
> 두 파일의 내용은 같다(150,000행, `SeriousDlqin2yrs` 양성 6.684%).

[^1]: https://www.kaggle.com/c/GiveMeSomeCredit/data
[^2]: https://github.com/JLZml/Credit-Scoring-Data-Sets — `3. Kaggle/Give Me Some Credit/cs-training.csv`

German Credit(UCI id=144)은 `src/data/loader.py::load_german_credit()` 에 구현돼 있으나
표본이 1,000건으로 부족해 메인 분석에는 쓰지 않는다(ADR-006).

검증 기준: 150,000행 / `SeriousDlqin2yrs` 양성률 6.684% /
`MonthlyIncome`(19.82%)·`NumberOfDependents`(2.62%) 결측.

---

## 디렉터리 구조

```
credit-scoring/
├── CLAUDE.md                  # 작업 규칙
├── PROGRESS.md                # 작업 로그 / 미해결 이슈
├── README.md
├── Dockerfile                 # python:3.12-slim, 4개 서비스 공용
├── docker-compose.yml         # api / dashboard / mlflow / bootstrap
├── scripts/bootstrap.sh       # 산출물 생성 (데이터 → 학습 → 분석)
├── scripts/verify_report_numbers.py  # 리포트 인용 수치를 산출물에서 재계산해 대조
├── requirements.txt           # 버전 고정 (+ requirements.lock 스냅샷)
├── .env.example
├── config/
│   ├── config.yaml            # 모든 설정값의 단일 출처
│   └── feature_names.yaml     # 피처 한글명 · 표시 단위 (거절 사유 생성용)
├── data/raw/                  # cs-training.csv (git 제외)
├── src/
│   ├── config.py              # 설정 로딩 · 경로 해석
│   ├── data/                  # loader / simulator / preprocessor / run_preprocess
│   ├── models/                # train / evaluate / report_assets
│   ├── explainer/             # SHAP 전역·지역 해석, 거절 사유 생성
│   ├── fairness/              # DI·EO 지표, ThresholdOptimizer, bias_ratio 민감도
│   ├── statistics/            # ANOVA + Tukey HSD
│   ├── api/                   # FastAPI (main.py, schemas.py)
│   └── dashboard/             # Streamlit (app.py, api_client.py)
├── models/                    # 학습 산출물 (git 제외)
├── tests/                     # pytest 171건
└── docs/
    ├── SPEC.md                # 과제 원문
    ├── decisions/             # ADR 12건
    └── report_assets/         # 리포트용 표·그림·서술
```

---

## 환경 변수

`.env.example` 를 복사해 쓴다(`cp .env.example .env`). `.env` 는 커밋하지 않는다.

| 변수 | 기본값 | 설명 |
|---|---|---|
| `API_BASE_URL` | `http://localhost:8000` | 대시보드가 추론을 요청할 FastAPI 주소. **대시보드는 모델 파일을 직접 로드하지 않고 반드시 이 API 를 경유한다**(SPEC §7). compose 에서는 `http://api:8000` 이 자동 주입된다. |
| `MLFLOW_TRACKING_URI` | `sqlite:///mlflow.db` | MLflow 백엔드. MLflow 3.x 가 파일 저장소를 막아 sqlite 를 쓴다. 컨테이너에서는 `sqlite:////app/mlflow.db`. |
| `APP_ENV` | `local` | 실행 환경 (`local` / `docker`). |
| `LOG_LEVEL` | `INFO` | 로그 레벨. |
| `CONFIG_PATH` | `config/config.yaml` | 설정 파일 경로 오버라이드. |
| `MLFLOW_DISABLE_AGENT_HINT` | `1` | MLflow 안내 문구 억제. |

임계값·포트·경로·`random_state` 등 **모든 설정값은 `config/config.yaml` 에만** 둔다.
코드에 상수를 하드코딩하지 않으며, 경로는 프로젝트 루트 기준 상대경로만 쓴다.

---

## 테스트

```bash
pytest                 # 171건
pytest -q tests/test_api.py tests/test_dashboard.py
```

---

## 설계 결정 (ADR)

| ADR | 주제 |
|---|---|
| [ADR-001](docs/decisions/ADR-001-simulator.md) | 대안 데이터 시뮬레이터 (**v2**: 공유 잠재요인으로 결합 누수 제거) |
| [ADR-002](docs/decisions/ADR-002-bias-injection.md) | 편향 주입 (**v3**: 대상 female/20-34, `bias_ratio` 0.3) |
| [ADR-003](docs/decisions/ADR-003-score-mapping.md) | 점수 매핑 `1000×(1−p)` · 등급 컷 · 승인 임계값 0.30 |
| [ADR-004](docs/decisions/ADR-004-mitigation.md) | ThresholdOptimizer (**v2**: `balanced_accuracy_score`) |
| [ADR-005](docs/decisions/ADR-005-anova-unit.md) | ANOVA 관측 단위 = RepeatedStratifiedKFold 5×4 |
| [ADR-006](docs/decisions/ADR-006-main-dataset.md) | 메인 데이터셋 = GMSC |
| [ADR-007](docs/decisions/ADR-007-age-filter.md) | `age < 18` 행 제거 |
| [ADR-008](docs/decisions/ADR-008-protected-attributes.md) | 보호 속성을 학습 피처에서 제외 |
| [ADR-009](docs/decisions/ADR-009-credit-history-months.md) | `credit_history_months` 추가, 씬파일러 판정 완성 |
| [ADR-010](docs/decisions/ADR-010-thin-filer-types.md) | 씬파일러 유형 분화 |
| [ADR-011](docs/decisions/ADR-011-anova-control.md) | ANOVA 통제 방식 (그룹당 n=20) |
| [ADR-012](docs/decisions/ADR-012-champion-algorithm.md) | 챔피언 알고리즘 선정 근거 = SHAP 지원 성숙도 |

---

## 한계

1. **절대 AUC 수치는 실무 성능의 추정치가 아니다.** 실제 신용평가 모델의 AUC 는 통상 0.85~0.88 이다.
   본 프로젝트의 0.94 대는 SPEC §4 가 요구하는 "대안 변수와 타겟의 상관 0.3~0.5" 를
   부도율 6.68% 인 이진 타겟에 적용했을 때 나타나는 **시뮬레이터 구조상 상한**이다.
   **피처 그룹 간·집단 간 상대 비교**에만 의미를 두어야 한다. (ADR-001 "한계")
2. **승인 임계값 0.30 은 재현율 우선 설정이다.** 정밀도 0.2899 는 실무 기준으로 낮다.
   임계값을 바꾸면 공정성 지표와 Mitigation 결과가 모두 연동돼 바뀌므로 ADR-003 값으로 동결했다.
3. **η² > 0.95 는 실험 설계의 산물이다.** 집단이 서로 다른 피처 구성으로 고정돼 있고
   동일 데이터에 반복 CV 를 적용해 집단 내 변동이 작다. 반복 CV fold 는 완전 독립이 아니므로
   p-value 는 보수적으로 읽는다. (ADR-011)
4. **완화 후 AUC 하락(0.9453 → 0.86)은 성능 저하가 아니다.** 출력이 확률에서 이진 결정으로
   바뀌어 순위 정보가 소실된 결과다. 전/후 비교는 DI·EO·승인율로 판단한다.
5. **대안 피처의 target 상관이 SPEC 하한(0.30)에 근접해 있다**(최소 여유 train 0.0099 / 전체 0.0094).
   F1~F5 결과는 현재 데이터로 동결했다.


## 재현 및 검증 범위

원본 데이터, 학습 모델, MLflow DB와 가상환경은 Git에 포함되지 않는다.
데이터와 학습 산출물은 bootstrap으로 생성한다.

2026-10-08 산출물이 없는 환경에서 실행한 단위 검사는 **78 passed / 6 skipped**다.
데이터가 필요한 검사는 skip되었으며, README의 모델 성능은 기존 실험 결과다.
이 검사에서는 전체 재학습이나 성능 재측정을 수행하지 않았다.

[검증 기록](docs/verification.md)
