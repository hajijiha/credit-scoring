# 대안신용평가 시스템

금융 이력이 부족한 씬파일러를 대상으로 통신·공과금 납부 이력 등의 대안 데이터를 결합해 신용위험을 예측한다. SHAP으로 예측 근거를 설명하고, Fairlearn으로 집단별 공정성을 분석한다. FastAPI API와 Streamlit 대시보드를 제공한다.

교육용 프로젝트이며 대안 데이터는 시뮬레이션으로 생성한 값이다. 아래 성능은 합성 데이터 실험 결과로, 실제 금융 의사결정에 사용할 수 없다.

[실험 보고서 PDF](docs/experiment_report.pdf) · [HTML](docs/experiment_report.html) · [과제 명세](docs/SPEC.md) · [설계 결정](docs/decisions/)

## 실행

Docker와 Docker Compose 또는 Python 3.12가 필요하다.

데이터, 학습 모델, MLflow 기록은 Git에 포함되지 않는다. 클론 후 처음 실행할 때 학습 산출물을 생성한다. 기존 검증 환경에서 약 15분이 걸렸다.

```bash
git clone https://github.com/hajijiha/credit-scoring.git
cd credit-scoring
docker compose --profile setup run --rm bootstrap
docker compose up
```

| 서비스 | 주소 |
|---|---|
| API 문서 | http://localhost:8000/docs |
| 대시보드 | http://localhost:8501 |
| MLflow | http://localhost:5000 |

Python으로 학습 산출물을 만들려면 다음 명령을 사용한다.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
bash scripts/bootstrap.sh
```

`bootstrap.sh`는 데이터 다운로드, 시뮬레이션, 전처리, 학습과 분석을 순서대로 실행한다. 기존 산출물은 건너뛰며 `FORCE=1`로 재생성할 수 있다. 패키지 버전 스냅샷은 `requirements.lock`에 있다.

Compose는 프로젝트 루트를 `/app`에 마운트하므로 이미지 재빌드 후에도 산출물을 재사용한다. 모델이 없으면 `/health`는 `degraded`를 반환하고 대시보드에 초기화 안내가 표시된다.

## 데이터와 설정

기본 데이터는 Give Me Some Credit의 150,000행이다. 부도율은 6.684%이며, 대안 피처는 시뮬레이터로 생성한다. Bootstrap은 [공개 미러](https://github.com/JLZml/Credit-Scoring-Data-Sets)의 데이터를 사용한다. [Kaggle 원본](https://www.kaggle.com/c/GiveMeSomeCredit/data)을 받으려면 대회 규정 동의와 API 인증이 필요하다.

```bash
pip install kaggle
kaggle competitions download -c GiveMeSomeCredit
unzip GiveMeSomeCredit.zip -d data/raw/
```

`.env.example`을 `.env`로 복사해 접속 설정을 바꿀 수 있다. `.env`는 커밋하지 않는다.

| 변수 | 기본값 | 용도 |
|---|---|---|
| `API_BASE_URL` | `http://localhost:8000` | 대시보드의 추론 API 주소. Compose에서는 `http://api:8000` |
| `MLFLOW_TRACKING_URI` | `sqlite:///mlflow.db` | MLflow 저장소. 컨테이너에서는 `sqlite:////app/mlflow.db` |
| `APP_ENV` | `local` | 실행 환경 (`local` / `docker`) |
| `LOG_LEVEL` | `INFO` | 로그 레벨 |
| `CONFIG_PATH` | `config/config.yaml` | 설정 파일 경로 |
| `MLFLOW_DISABLE_AGENT_HINT` | `1` | MLflow 안내 문구 억제 |

임계값, 피처 구성과 학습 설정은 `config/config.yaml`에서 관리한다. 대시보드의 추론은 FastAPI를 경유한다.

## 실험 결과

선정 모델은 XGBoost / combined / class_weight이며, 테스트 세트는 22,500행이다.

| 지표 | 결과 |
|---|---:|
| AUC-ROC | 0.9453 |
| KS | 0.7502 |
| PSI (train ↔ test) | 0.00054 |
| 씬파일러 AUC uplift | +0.1518 |
| Precision / Recall / F1 | 0.2899 / 0.8936 / 0.4378 |
| `/predict` 평균 응답 시간 | 20.3 ms (단일 요청 50회) |

피처 그룹별 비교와 ANOVA 결과는 [실험 보고서](docs/experiment_report.pdf), 집단별 분석은 [공정성 분석](docs/report_assets/fairness_narrative.md)에 있다.

결과를 해석할 때 다음 조건을 고려해야 한다.

- 대안 피처와 타겟의 상관을 설정한 합성 데이터다. 절대 AUC보다 피처 구성과 집단 간 상대 비교에 의미가 있다.
- 승인 임계값 0.30은 재현율을 우선한다. 정밀도가 낮으며, 임계값을 바꾸면 공정성 지표도 달라진다.
- 보호 속성을 학습에서 제외해도 대안 피처를 통해 편향이 남는다. `age_group`의 DI(min/max)는 0.7747, EO FPR 차이는 0.1487이다.
- 반복 교차검증의 fold는 완전히 독립적이지 않다. ANOVA의 p-value와 효과 크기는 이 실험 조건 안에서 해석한다.
- 공정성 완화 전 확률 출력과 완화 후 이진 결정의 AUC를 직접 비교하기 어렵다. 완화 효과는 DI, EO와 승인율로 확인한다.

## 코드 구성

| 경로 | 역할 |
|---|---|
| `src/data/` | 데이터 로딩, 시뮬레이션, 전처리 |
| `src/models/` | 학습, 평가, MLflow 기록 |
| `src/explainer/` | SHAP 해석과 거절 사유 |
| `src/fairness/` | 공정성 지표와 임계값 완화 |
| `src/statistics/` | ANOVA와 Tukey HSD |
| `src/api/`, `src/dashboard/` | FastAPI와 Streamlit |
| `scripts/` | 산출물 생성과 보고서 수치 대조 |
| `tests/` | 단위·통합 검사 |
| `docs/` | 명세, 설계 결정, 보고서와 검증 기록 |

개발 시 설정·데이터 처리 기준은 [개발 지침](docs/development.md)을 참고한다.

## 테스트와 수치 대조

```bash
pytest
pytest -q tests/test_api.py tests/test_dashboard.py
python scripts/verify_report_numbers.py
python scripts/verify_report_numbers.py --chapters 3 6
```

보고서 수치 대조에는 학습 산출물이 필요하다. 기존 전체 개발 환경의 테스트 기록은 171건이며, 2026-10-08 산출물이 없는 환경에서는 78개가 통과하고 6개가 skip되었다. 이 검사에서 전체 재학습이나 성능 재측정은 수행하지 않았다. [검증 기록](docs/verification.md)에 범위를 정리했다.
