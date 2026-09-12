# 2. 시스템 개요

이 장은 1장의 세 질문을 검증하는 시스템이 어떻게 구성돼 있는지를 한눈에 정리한다.
세부 구현은 각 장과 부록 B(API 명세) · C(재현 절차)에 있다.

---

## 2.1 아키텍처

```
                               docker compose up
 +---------------------+    HTTP (API_BASE_URL)    +----------------------+
 |  dashboard  :8501   | ------------------------> |      api  :8000      |
 |  Streamlit, 5 tabs  |  /predict /explain /...   |  FastAPI, 7 routes   |
 |  (no model files)   |                           |  load once at start  |
 +---------------------+                           +----------+-----------+
                                                              | read
 +---------------------+          read             +----------v-----------+
 |   mlflow  :5000     | <---- mlflow.db, mlruns/  |  models/             |
 |  runs + Registry    |                           |  data/processed/     |
 +---------------------+                           +----------^-----------+
                                                              | write (once)
                                                   +----------+-----------+
                                                   |  bootstrap (profile) |
                                                   |  scripts/bootstrap.sh|
                                                   +----------------------+
```

네 서비스는 같은 이미지(`python:3.12-slim`)를 쓰고 `command` 로 역할만 나눈다. 프로젝트 루트를 컨테이너에
바인드 마운트하므로 학습 산출물은 호스트에 한 번 만들어 두고 재사용한다.

**대시보드는 모델 파일을 읽지 않는다.** 모든 추론과 지표 조회는 API 를 HTTP 로 호출한다(명세 §7 MSA 제약).
이 제약은 문서가 아니라 테스트로 강제한다. 대시보드 코드의 import 를 AST 로 검사해 모델 라이브러리와
`joblib` · `pickle` 로드가 없음을 확인한다(`tests/test_dashboard.py`).

---

## 2.2 파이프라인

`scripts/bootstrap.sh` 가 클론 직후의 저장소에서 모든 산출물을 만든다. 이미 있는 산출물은 건너뛴다.

| 단계 | 모듈 | 산출물 | 장 |
|---|---|---|---|
| 1 데이터 확보 | `curl` (공개 미러) | `data/raw/cs-training.csv` | 3 |
| 2 시뮬레이션 + 전처리 | `src.data.run_preprocess` | `data/processed/{train,valid,test}.parquet`, `preprocessor.joblib` | 3, 4 |
| 3 학습 | `src.models.train` | `models/*.pkl`, `model_results.csv`, `fold_auc.csv`, MLflow run 18개 | 5 |
| 4 uplift + ANOVA | `src.models.train --uplift-only`, `src.statistics.anova` | `thin_filer_uplift.csv`, `anova_results.json` | 5, 8 |
| 5 SHAP + 공정성 | `src.explainer.run_explain`, `src.fairness.run_fairness` | `explain/`, `fairness/` | 6, 7 |
| 6 리포트 자료 | `src.models.report_assets` | `docs/report_assets/*.png` | 5, 8 |

각주: 학습 단계는 18조합 × 20 fold = 360 fits 에 최종 학습 18 fits 를 더해 616.4초가 걸렸다(5.1절).
EDA 그림(`src.analysis.eda`)과 `bias_ratio` 민감도(`src.fairness.sensitivity`)는 데이터를 여러 번
재생성하므로 부트스트랩에 넣지 않고 별도로 실행한다.

API 는 이 산출물을 **기동 시 한 번** 읽고, 공정성 · ANOVA · 지표 응답은 사전 계산 결과를 그대로 서빙한다.
요청마다 재계산하지 않으므로 응답이 결정론적이다. 대신 재학습 뒤 분석 단계를 다시 돌리지 않으면 낡은 결과를
서빙한다(9장 #4).

---

## 2.3 서비스

| 서비스 | 포트 | 역할 | 주요 사항 |
|---|---:|---|---|
| **api** | 8000 | 심사 · 설명 · 분석 결과 서빙 | 7개 엔드포인트, Pydantic 스키마, Swagger `/docs`. `/predict` 평균 **20.3ms**, `/explain` 평균 36.9ms |
| **dashboard** | 8501 | 심사역용 UI | 탭 5개: 고객 심사 · XAI 분석 · 공정성 · 모델 성능 · 통계 검증. 위험도 색상(저 #2E7D32 / 중 #F9A825 / 고 #C62828) |
| **mlflow** | 5000 | 실험 추적 | SQLite 백엔드, run 18개, Registry `credit-scoring-champion` (alias `champion`, tag `stage=Production`) |
| bootstrap | — | 산출물 생성 | `setup` 프로파일. `docker compose up` 에서는 실행되지 않는다 |

각주: 응답 시간은 FastAPI TestClient 단일 요청 50회(`/predict`) · 20회(`/explain`) 평균, 12세션 측정.

API 가 산출물 없이 기동되면 `/health` 가 `degraded` 를 반환하고, 대시보드는 크래시 없이 실행 방법을 안내한다.

---

## 2.4 모듈 구성

| 모듈 | 주요 파일 | 줄 수 | 역할 |
|---|---|---:|---|
| `src/data/` | `loader`, `simulator`, `preprocessor`, `run_preprocess` | 763 | GMSC 로드, 대안 데이터 · 씬파일러 · 보호 속성 합성, 전처리 · 분할 |
| `src/models/` | `train`, `evaluate`, `report_assets` | 798 | 18조합 학습 · CV · MLflow, AUC · KS · PSI, uplift |
| `src/explainer/` | `explainer`, `run_explain` | 618 | SHAP 전역 · 지역 해석, 거절 사유 자연어 생성 |
| `src/fairness/` | `metrics`, `mitigation`, `run_fairness`, `sensitivity` | 739 | DI · EO, ThresholdOptimizer, `bias_ratio` 민감도 |
| `src/statistics/` | `anova` | 288 | One-way ANOVA, Tukey HSD, 가정 진단 |
| `src/api/` | `main`, `schemas` | 608 | FastAPI 서버 |
| `src/dashboard/` | `app`, `api_client` | 595 | Streamlit 대시보드, HTTP 클라이언트 |
| `src/analysis/` | `eda` | 513 | EDA 그림 |
| `src/config.py` | — | 42 | 설정 로딩 · 경로 해석의 유일한 진입점 |
| **합계** | | **4,964** | |

각주: `__init__.py` 포함, 빈 줄 · 주석 포함 줄 수. 테스트는 별도로 171개(부록 D).

---

## 2.5 기술 스택

| 영역 | 라이브러리 (버전) |
|---|---|
| 런타임 | Python 3.12, Docker · Docker Compose |
| 데이터 | pandas 3.0.5, numpy 2.5.3, scipy 1.18.1 |
| 모델링 | scikit-learn 1.9.0, XGBoost 3.4.1, LightGBM 4.7.0, imbalanced-learn 0.14.2 |
| 설명가능성 · 공정성 | SHAP 0.52.0, Fairlearn 0.14.0 |
| 통계 | statsmodels 0.15.0 |
| 실험 추적 | MLflow 3.16.0 |
| 서빙 · UI | FastAPI 0.141.1, uvicorn 0.52.4, Pydantic 2.13.5, Streamlit 1.63.0 |
| 테스트 | pytest 9.1.1 |

각주: `requirements.txt` 에 버전 고정, 전이 의존성 전체는 `requirements.lock`.

공정성 라이브러리는 **Fairlearn 하나만** 쓴다. 명세는 AIF360 도 허용하지만 Python 3.12 를 지원하지 않는다.

---

## 2.6 설계 원칙

| 원칙 | 구현 | 이유 |
|---|---|---|
| 설정의 단일 출처 | 임계값 · 비율 · 포트 · `random_state` · 하이퍼파라미터를 `config/config.yaml` 에만 둔다 | 5~7장의 수치가 하나의 설정 위에서 정합돼야 한다. 임계값 0.30 하나가 공정성 결과 전체와 연동된다(10.3절) |
| 상대 경로 | 모든 경로는 프로젝트 루트 기준, `src/config.py` 가 해석 | 호스트와 컨테이너에서 같은 설정이 동작한다 |
| 결정 기록 | 설계 결정 12건을 ADR 로 남기고, 개정 시 버전 표를 갱신 | 결정이 뒤집힌 이유를 추적할 수 있어야 한다(부록 A) |
| 테스트 라벨 격리 | test 라벨은 평가에만. 임계값 조정은 valid 로 | 명세 §7. 공정성 완화의 전/후 비교가 유효하려면 필수 |
| 보호 속성 격리 | 학습 X 는 `features.traditional` + `features.alternative` 로만 구성 | 간접 편향을 관측하려면 직접 경로를 막아야 한다(ADR-008, 7.1절) |

---

## 이 장의 결론

1. **네 서비스가 한 이미지를 공유하고, 산출물은 볼륨으로 주고받는다.** 대시보드 → API → 산출물의
   단방향 구조이며, 대시보드의 모델 파일 미로드는 AST 검사 테스트로 강제한다.
2. **부트스트랩 6단계가 클론 직후 저장소에서 모든 분석 산출물을 만든다.** API 는 이를 기동 시 한 번 읽어
   서빙하므로 응답은 빠르고(`/predict` 20.3ms) 결정론적이지만, 재학습 후 분석 재실행을 빠뜨리면 낡은 결과를 낸다.
3. **모든 설정값을 `config.yaml` 한 곳에 둔 것이 수치 정합성의 기반이다.** 소스 4,964줄, 테스트 171개,
   ADR 12건으로 구성된다.
