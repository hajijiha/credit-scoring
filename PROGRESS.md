# PROGRESS

> 세션 시작 시 이 파일을 먼저 읽는다. 작업이 하나 끝날 때마다 즉시 갱신한다(마지막에 몰아쓰지 않는다).
> 요구사항 원문은 `docs/SPEC.md`, 설계 결정은 `docs/decisions/ADR-*.md`.

## 1. 현재 상태

### F1. 데이터 파이프라인
- [x] `src/data/loader.py` — GMSC 로더, 컬럼명/dtype/결측비율 로그 (ADR-006, ADR-007)
- [x] `src/data/loader.py::load_german_credit()` — `fetch_ucirepo(id=144)`, 메인 미사용
- [x] `src/data/simulator.py` — 대안 5변수(`telecom_payment_rate`, `utility_payment_rate`,
      `spending_consistency`, `regular_payment_count`, `app_login_frequency`), target 상관 0.3~0.5
- [x] `thin_filer_ratio` 파라미터 — 조정 시 씬파일러 비율 변경
- [x] `bias_ratio` 파라미터 — 조정 시 성별/연령대 편향 변경 (ADR-002 v2)
- [x] 씬파일러 판정 함수 — 연체 관련 컬럼 2개 이상 결측 (`is_thin_filer`)
- [x] `src/data/preprocessor.py` — 결측 처리, StandardScaler, One-Hot, 70:15:15 stratify 분할
- [ ] 씬파일러 집단 AUC uplift 측정 (F2 모델 학습 후)
- [ ] EDA 노트북 (`notebooks/`, 권장 항목)

### F2. 신용평가 모델
- [x] `src/models/train.py` — 3개 알고리즘 × 피처그룹 3종 × 불균형 처리 2종 = 18개 조합
- [x] `src/models/evaluate.py` — AUC, KS, PSI, 임계값 기반 지표
- [x] RepeatedStratifiedKFold(5×4) fold AUC 360건 저장 (`data/processed/fold_auc.csv`)
- [x] **전체 학습 완료** (360 fits, 616.4초)
- [x] `models/` 에 `{algorithm}_v1.0.pkl` 3개 + `champion_xgboost_v1.0.pkl` 저장
- [x] MLflow: run 18개, 각 run 에 params·metrics·fold_auc(20 step)·모델 아티팩트 기록
- [x] Model Registry: `credit-scoring-champion` v1 (alias=champion, tag stage=Production)
- [x] 성능 목표: AUC ≥ 0.78 (16/18 조합, 최고 **0.9491**) / KS ≥ 0.28 (18/18, 최고 0.7576)
      / PSI < 0.1 (18/18, 최대 0.0014) / 씬파일러 uplift ≥ +0.03 (최소 +0.102)

### F3. 설명가능성(XAI)
- [x] `src/explainer/` — SHAP TreeExplainer (`explainer.py`, `run_explain.py`)
- [x] 전역: Summary Plot 상위 15개, Dependence Plot 2개 (대안 1 + 전통 1)
- [x] 지역: 개별 고객 Waterfall Plot (씬파일러 / 일반 각 1건)
- [x] 거절 사유 Top 5 — `feature`, `feature_name_kr`, `current_value`, `reference_value`,
      `shap_contribution`, `explanation`, `direction_consistent`
- [x] 자연어 설명 — 단위별 템플릿 분기(percent/ratio/count/amount/years/flag), 한글 조사 처리
- [x] `reference_value` = 동일 연령대 평균 (센티넬 −1 은 평균에서 제외)
- [x] 한글 피처명 매핑 `config/feature_names.yaml` 로 분리
- [ ] `/explain` API 엔드포인트 (F6 에서)

### F4. 공정성 검증 및 Bias Mitigation
- [x] `src/fairness/metrics.py` — DI Ratio, Equalized Odds, WARNING 로그
- [x] `src/fairness/mitigation.py` — Fairlearn ThresholdOptimizer(equalized_odds) (ADR-004)
- [x] `src/fairness/run_fairness.py` — 전/후 비교 + 차트 2종
- [x] 측정 결과(bias_ratio=0.3): gender DI 0.9191 / age_group DI 0.8375 (min-max **0.7747**),
      age_group **EO FPR차 0.1487 기준 위반**
- [x] Mitigation 후: gender DI 0.9942, age_group DI 0.9433, EO 전부 충족,
      승인율 0.7940 → 0.8196~0.8492
- [x] `src/fairness/sensitivity.py` — bias_ratio 민감도 (SPEC §4 요건 실증)
- [ ] `/fairness`, `/fairness/mitigation` API (F6 에서)

### F5. ANOVA 통계 검증
- [x] 관측치 확보 — fold AUC 360건
- [x] `src/statistics/anova.py`
- [x] 검증 1: 피처 그룹별 One-way ANOVA — F=4245.65, p=9.64e-63, η²=0.9933
- [x] 검증 2: 알고리즘별 One-way ANOVA — F=804.38, p=1.68e-42, η²=0.9658
- [x] F-통계량, p-value, η² + 해석 기준(0.01/0.06/0.14) 자동 적용
- [x] Tukey HSD 사후검정 (두 검정 모두 p<0.05 이므로 수행)
- [x] 정규성(Shapiro-Wilk)·등분산성(Levene) 진단 + 위반 시 Welch/Kruskal-Wallis 병기 로직
- [x] 관측 단위 그룹당 n=20 (ADR-005), 통제 방식은 ADR-011
- [ ] 리포트에 ANOVA 해석 1페이지 이상
- [ ] `/anova` API 엔드포인트 (F6 에서)

### F6. API 서버 (FastAPI, :8000)
- [x] `src/api/main.py` + `src/api/schemas.py`, 전 엔드포인트 Pydantic 스키마
- [x] `GET  /health` — 상태·모델버전·타임스탬프·챔피언·자원 로드 현황
- [x] `POST /predict` — credit_score / approval_status / default_probability / risk_grade
- [x] `POST /explain` — rejection_reasons(Top5) + shap_values + base_value
- [x] `GET  /fairness` — DI **양 정의 병기**, EO TPR/FPR, 보호속성별 상세
- [x] `GET  /fairness/mitigation` — before / after / delta / 집단별 조정 임계값
- [x] `GET  /metrics` — auc, ks, precision, recall, f1, psi
- [x] `GET  /anova` — test_1, test_2 (F, p, η², Tukey, 진단)
- [x] 모델·전처리기·SHAP 설명기 기동 시 1회 로드
- [x] 사전 계산 결과 서빙(요청 시 재계산 없음)
- [x] `/predict` 평균 **20.3ms** (목표 2초), `/explain` 평균 36.9ms
- [x] Swagger UI `/docs` 정상, OpenAPI 에 7개 엔드포인트 등록
- [ ] `docker-compose up` 통합 구동 (F8 에서)

### F7. 대시보드 (Streamlit, :8501)
- [x] `src/dashboard/app.py` + `api_client.py`, 5개 탭
- [x] 탭1 **고객 심사** — 입력 폼 17항목, 승인/거절 배너, 신용점수·등급·부도확률·임계값
- [x] 탭2 **XAI 분석** — Summary Plot(사전 생성), 개별 Waterfall(API SHAP 값으로 렌더), 거절 사유 Top 5
- [x] 탭3 **공정성** — 집단별 승인율 차트, DI **양 정의**, 보호속성별 표, Mitigation 전/후 차트
- [x] 탭4 **모델 성능** — 지표 6종 + ROC/PR Curve, 혼동행렬
- [x] 탭5 **통계 검증** — ANOVA 요약표, 집단 평균표, Tukey HSD 표, η² 경고
- [x] **`.pkl` 직접 로드 없음** — AST import 검사 + 코드 라인 grep 검사를 테스트로 강제
- [x] 위험도별 색상 (저위험 #2E7D32 / 중위험 #F9A825 / 고위험 #C62828)
- [x] `API_BASE_URL` 환경변수, API 미기동 시 크래시 없이 안내 표시
- [ ] `docker-compose up` 통합 구동 (F8 에서)

### F8. 프로젝트 구조 및 제출 요건
- [x] 역할별 `src/` 폴더 분리 (data/models/explainer/fairness/statistics/api/dashboard)
- [x] 설정값 config 분리, 하드코딩 절대경로 없음
- [x] `requirements.txt` 버전 명시 (+ `requirements.lock`)
- [x] `tests/` pytest **166건** (요건 10건 이상), 100% 통과 (요건 80% 이상)
- [x] `.env.example` 완성
- [x] `README.md` — 개요·팀 역할(1인)·실행방법·구조·환경변수·데이터 획득·ADR·성능표·한계
- [x] `models/` 에 학습된 모델 파일 (`champion_xgboost_v1.0.pkl` 외 3개)
- [x] `Dockerfile` / `docker-compose.yml` 완성 (api/dashboard/mlflow + bootstrap 프로파일)
- [x] `scripts/bootstrap.sh` — 클린 클론에서 산출물 생성
- [x] **`docker compose up` 통합 구동 검증 완료** — 3개 서비스 전부 healthy
      (api 8000 / dashboard 8501 / mlflow 5000), 대시보드→API 컨테이너 네트워크 경유 확인
- [x] commit 10개 이상 (현재 13개)
- [ ] GitHub 원격 저장소 + Issue/PR 3개 이상 → `docs/github_setup.md` 에 계획 정리
- [ ] `docs/experiment_report.pdf` 20페이지 이상 — **작성 중** (아래 진행표)

### 리포트 작성 진행 (총 32p 목표, 목차는 `docs/report_outline.md`)

| 장 | 파일 | 상태 |
|---|---|---|
| 표지·목차 | `docs/report/00_cover.md` | 대기 |
| 요약 | `01_summary.md` | 대기 |
| 1. 문제 정의 | `02_problem.md` | 대기 |
| 2. 시스템 개요 | `03_system.md` | 대기 |
| 3. 데이터와 EDA | `04_data_eda.md` | 대기 (그림 8종 준비 완료) |
| **4. 방법론: 시뮬레이터** | `05_simulator.md` | **완료** (316줄, 표 6개, 그림 2개) |
| 5. 모델 비교 | `06_models.md` | 대기 |
| 6. XAI 분석 | `07_xai.md` | 대기 |
| **7. 공정성** | `08_fairness.md` | **완료** (306줄, 표 12개, 그림 4개, 교차검증 24건) |
| 8. ANOVA | `09_anova.md` | 대기 |
| 9. 발견하고 수정한 오류 | `10_errors.md` | 대기 |
| 10. 한계 | `11_limitations.md` | 대기 |
| 11. 결론 | `12_conclusion.md` | 대기 |
| 부록 | `99_appendix.md` | 대기 |
| HTML 병합 | `scripts/build_report.py` | 대기 |

작성 순서: 4 → 7 → 8 → 5 → 6 → 9·10·11 → 3 → 1·2 → 요약·부록

## 2. 마지막 세션 요약

**2026-09-10 (15세션) — 리포트 목차 확정, EDA 그림, 4장 작성**

- 한 일: `docs/report_outline.md`(329줄) 작성 및 확정 반영, `src/analysis/eda.py` 로
  EDA 그림 8종 생성(총 24종), 리포트 **4장** 작성(`docs/report/05_simulator.md`, 316줄).
- **ADR-004 v2 누락 복구** — 11세션에서 개정했다고 보고했으나 문자열 치환이 공백 불일치로
  조용히 실패해 문서에 반영되지 않았다(코드·설정에는 반영됨). 17줄 → 40줄로 복구.
  9장 오류 목록에 "검증 없는 자동 편집의 위험 사례"로 추가.
- v1 재구성 검증: `rebuild_v1_alternative()` 로 ADR-001 v1 구조를 재현해
  클래스 내부 상관 **0.177**, 결합 d(독립가정) **4.02** → 이론 AUC 0.9978 을 얻었다.
  v1 실측(0.19 / 3.96 / 0.9975)과 일치해 재구성이 타당함을 확인.
- 결합 효과크기를 **독립가정 값과 실제 값(마할라노비스)** 두 개로 분리했다.
  v1: 4.02 vs 3.04 / v2: 3.00 vs **1.54**. 공유요인이 실제 판별력을 절반으로 낮춘다.
- 4장 인용 수치를 산출물과 교차 검증해 **불일치 3건 수정**:
  ① v1/v2 비교표가 서로 다른 조합을 섞어 씀 → LGBM class_weight 로 통일
  ② 4.5 두 실험이 `bias_ratio=0.1` 시점 측정임을 명시(최종은 0.3)
  ③ uplift 표의 통합 AUC 가 이전 실행 값이었음 → 현행 값으로 교체

**2026-09-08 (14세션) — Docker 구성·README 최종화(F8)**

- 한 일: `Dockerfile`/`docker-compose.yml` 완성(api/dashboard/mlflow + bootstrap 프로파일),
  `scripts/bootstrap.sh`, `.dockerignore`, `.env.example` 완성, README 258줄 최종화,
  `docs/github_setup.md`(Issue 8건·PR 3건 계획), MLflow `artifact_location` 상대경로화.
- 산출물 배치 결정: **볼륨 마운트**. `models/`·`data/processed/` 가 리포지토리에 없어
  이미지에 구울 대상이 존재하지 않고, 빌드 중 학습은 15분 이상 걸린다.
  근거를 README "왜 볼륨 마운트인가" 절에 표로 정리.
- **미해결**: 이미지 빌드가 이 환경의 BuildKit DNS 실패로 완료되지 않는다.

**2026-09-08 (13세션) — Streamlit 대시보드(F7)**

- 한 일: 임계값 0.30 의 대가를 ADR-003·리포트 한계에 기록, `src/dashboard/{app,api_client}.py`,
  `tests/test_dashboard.py` 18건(총 166건 통과), config `dashboard` 섹션.
- 검증 (Streamlit AppTest, API 기동 상태):
  - 5개 탭 모두 예외·에러 0
  - 탭1: 신용점수 625/1000, 등급 C, 부도확률 **37.50%** (borderline 샘플과 일치), 임계값 0.30
  - 탭2: 거절 사유 카드 5개 렌더
  - 탭3: 전체 승인율 0.7940, DI(age_group) 0.8375 / min-max 0.7747
  - 탭4: AUC 0.9453 / KS 0.7502 / precision 0.2899 / recall 0.8936 / f1 0.4378 / PSI 0.0005
  - 탭5: 표 6개, ANOVA 요약 + Tukey HSD
  - API 미기동 시: 예외 0, 탭 5개 유지, 에러 안내 4건 + 사이드바 "API 연결 실패" + 실행 명령 안내
- 버그 2건 수정:
  - **`/explain` 이 연체 '기록 없음' 입력에서 500** — `float(None)` 실패.
    이 프로젝트의 핵심 대상인 씬파일러가 그대로 걸리는 경로였다.
    센티넬 값으로 표기하고 "기록이 없습니다" 문장으로 분기.
  - **`app.py` import 만으로 앱이 실행됨** — `main()` 이 모듈 최상위에 있었다.
    테스트에서 Streamlit 전역 상태가 오염돼 "Forms cannot be nested" 가 났다. `__main__` 가드 추가.

**2026-09-08 (12세션) — FastAPI 7개 엔드포인트(F6), 결과 동결**

- 한 일: F1~F5 결과 동결(재생성 없음) 및 상관 여유를 ADR-001 한계 절에 기록,
  리포트 결론 문장 확정, `src/api/{main,schemas}.py` 구현,
  `src/data/run_preprocess.py` CLI 분리, `tests/test_api.py` 28건(총 148건 통과).
- 확인된 수치:
  - `/predict` 평균 **20.3ms** / 중앙값 20.3ms / 최대 26.2ms (50회) — SPEC 2초 대비 여유 충분
  - `/explain` 평균 36.9ms / 최대 49.3ms (20회)
  - `/health` status=healthy, 6개 자원 전부 로드
  - `/metrics`: AUC 0.9453, KS 0.7502, precision 0.2899, recall 0.8936, f1 0.4378, PSI 0.00054
- 버그 2건 수정:
  - **전처리기 역직렬화 실패** — `python -m src.data.preprocessor` 로 저장하면 클래스의
    `__module__` 이 `"__main__"` 이 되어 다른 프로세스에서 못 읽는다. CLI 를 분리해 해결.
  - **`credit_history_months` 입력이 무시됨** — `to_raw_frame` 에서 pop 후 기본값으로 덮어썼다.
  - **`/anova` 가 stale 결과 서빙** — 재학습 후 ANOVA 재실행을 빠뜨렸다. 재실행 후 수치 갱신
    (검증1 F=4245.65 η²=0.9933 / 검증2 F=804.38 η²=0.9658).

**2026-09-08 (11세션) — bias_ratio 0.3 (ADR-002 v3), 목적함수 변경 (ADR-004 v2)**

- 한 일: `bias_ratio` 0.1 → 0.3, ThresholdOptimizer 목적함수 `accuracy` → `balanced_accuracy`,
  데이터 재생성 → 재학습(639.6초) → explain/fairness/민감도 재실행,
  `src/fairness/sensitivity.py` 신규, 트레이드오프 차트, 공정성 서사 문서.
- 확인된 수치:
  - 상관 5개 0.3099~0.3285 (전부 범위 내, 단 **하한 여유 0.0099 로 얇음**)
  - combined test AUC **0.9453** (목표 0.90~0.95 유지), 챔피언 xgboost|combined|class_weight
  - 완화 전: gender DI **0.9191** (female 0.7603 / male 0.8272),
    age_group DI 0.8375 / min-max **0.7747**, EO FPR차 **0.1487 위반**
  - 완화 후: gender DI 0.9942 / age_group DI 0.9433, EO 전부 충족,
    **승인율 0.7940 → 0.8196~0.8492 (목표 0.75~0.85 유지)**, AUC 0.9453 → 0.8592~0.8736
  - 씬파일러 uplift(XGB): thin_filer +0.152 / normal +0.095 /
    both +0.188 > missing_only +0.168 > history_only +0.101
  - **민감도(핵심 발견)**: gender DI 는 bias_ratio 에 단조 반응(1.0030 → 0.9502 → 0.9293 → 0.9191)하나
    age_group 은 bias_ratio=0 에서 이미 min/max 0.7840 으로 거의 불변(0.3 에서 0.7747)
- 목적함수 변경 효과: 승인율 0.955 → 0.82~0.85, AUC 0.739 → 0.86~0.87 로 개선.

**2026-09-08 (10세션) — 공정성 검증·Mitigation(F4), 설명 샘플 보강**

- 한 일: 경계 사례 설명 샘플 추가(config `explainer.samples`), `src/fairness/` 3개 모듈,
  `tests/test_fairness.py` 15건(총 119건 통과), 공정성 차트 2종.
- 확인된 수치 (챔피언 xgboost|combined|class_weight, test 22,500행, 임계 0.30):
  - 전체 승인율 0.8011
  - **gender**: 승인율 female 0.7805 / male 0.8215 → DI **0.9502** (충족),
    EO TPR차 0.0258 / FPR차 0.0425 (충족) — **편향 미검출**
  - **age_group**: 승인율 20-34 0.6838 / 35-54 0.7508 / 55+ 0.8844 →
    DI(보호 vs 나머지) **0.8358** 충족이나 **min/max 0.7732 는 0.8 미만**,
    EO TPR차 0.0938 / **FPR차 0.1496 기준 위반**
  - Mitigation 후: gender DI 1.0000 / age_group DI 0.9759, EO 전부 충족
  - 트레이드오프: 승인율 0.8011 → 0.955 내외, AUC 0.9485 → 0.74 (이진 결정이라 순위 정보 소실)
- 버그·환경 이슈 2건: fairlearn 이 sklearn BaseEstimator 를 요구(래퍼에 `__sklearn_tags__` 없음),
  XGBoost float32 확률이 pandas 3.x 에서 대입 오류 → `Float64ProbaEstimator` 어댑터로 해결.

**2026-09-08 (9세션) — SHAP 설명가능성(F3), ADR-012**

- 한 일: ADR-012(챔피언 알고리즘 선정 근거 변경) 작성 및 관련 서술 정정,
  `config/feature_names.yaml`, `src/explainer/` 구현, `tests/test_explainer.py` 21건
  (총 104건 통과), 이미지 5종 생성 및 `docs/report_assets/` 복사.
- 확인된 수치 (챔피언 xgboost|combined|class_weight, test 2,000행 표본):
  - 전역 중요도 1위 **telecom_payment_rate 1.3460** (2위 RevolvingUtilization 0.7769 의 1.7배)
  - 상위 5개 중 대안 피처 3개, 상위 15개 구성은 traditional 10 / alternative 5
  - 피처당 평균 |SHAP|: **alternative 0.6563 vs traditional 0.2011** (3.3배)
  - 씬파일러 거절 사례(history_only, p=0.9997): 1~3위가 전부 대안 피처
  - 일반 거절 사례(p=0.9997): 1위 대안, 2위 연체 횟수로 혼재
- 버그 2건 수정:
  - **거절 사유가 워터폴과 다른 피처를 가리킴** — 역스케일한 값을 모델에 넣어 SHAP 을 계산했다.
    `rejection_reasons(row, display_row=...)` 로 모델 입력과 표시값을 분리. 회귀 테스트 추가.
  - **참조 평균에 센티넬(−1) 포함** — "동일 연령대 평균 −0회" 가 출력됐다. 결측 플래그 기준으로 제외.

**2026-09-08 (8세션) — ANOVA 검증(F5), 챔피언 재선정**

- 한 일: ADR-001 부기 정정(SMOTE 재평가 + 채택 근거), 챔피언을 class_weight 조합으로 재등록,
  Dockerfile 에 fonts-nanum 추가, ADR-011(ANOVA 통제 방식) 작성,
  `src/statistics/anova.py` 구현, `tests/test_anova.py` 22건 추가(총 83건 통과).
- 확인된 수치:
  - **검증 1 (피처 그룹, xgboost/class_weight 고정, n=20)**
    F=4514.60, p=1.69e-63, η²=0.9937 [Large] — Tukey HSD 3쌍 모두 유의
    평균 AUC: traditional 0.8355 / alternative 0.8965 / combined 0.9479
  - **검증 2 (알고리즘, combined/class_weight 고정, n=20)**
    F=660.02, p=3.82e-40, η²=0.9586 [Large]
    Tukey: LR vs XGB 유의(+0.0291), LR vs LGBM 유의(+0.0282), **XGB vs LGBM 유의하지 않음(p=0.61)**
    → 챔피언 알고리즘 선정 근거를 "성능 우위"에서 "SHAP 지원 성숙도"로 변경 (ADR-012)
    평균 AUC: xgboost 0.9479 / lightgbm 0.9470 / logistic_regression 0.9188
  - 진단: 두 검정 모두 **Shapiro-Wilk 정규성 충족 / Levene 등분산 충족** → 표준 ANOVA 사용
  - 참고(교락 있음, 360건): 검증1 F=1158.95 η²=0.8665 / 검증2 F=12.53 η²=0.0656
- 챔피언: `xgboost | combined | class_weight` (Registry v2, alias=champion). v1(smote)은 Archived.

**2026-09-08 (7세션) — 전체 학습 완료, 리포트 자료 착수**

- 한 일: ADR-001 v2 에 목표 구간 정정(0.85~0.90 → **0.90~0.95**)과 시뮬레이션 한계 항목 기록,
  MLflow 백엔드 전환(file → sqlite), Registry 등록·최종 모델 저장·유형별 uplift 구현,
  전체 학습 실행, `src/models/report_assets.py` 및 진단 문서 작성.
- 확인된 수치:
  - **총 소요 616.4초** (18조합 × 20 fold = 360 fits + 최종 18 fits)
  - 최고 조합 **xgboost | combined | smote — test AUC 0.9491, KS 0.7576, PSI 0.0004**
  - 피처그룹별 최고 test AUC: traditional 0.8355 / alternative 0.8975 / combined 0.9491
  - 씬파일러 uplift(XGBoost): thin_filer +0.156 / normal +0.097
  - 씬파일러 **유형별** uplift(XGBoost): both +0.195 / missing_only +0.173 / history_only +0.102
  - fold_auc.csv 360행, 조합당 20 fold — ANOVA 그룹당 n=120
  - MLflow run 18개, 모델 아티팩트 18개, Registry v1 등록
- 환경 이슈 2건 해결: MLflow 3.16 이 파일 저장소를 막아 sqlite 백엔드로 전환,
  skops 직렬화가 imblearn 파이프라인을 거부해 cloudpickle 로 고정.

**2026-09-08 (6세션) — ADR-001 v2 (누수 제거), ADR-010 (씬파일러 유형 분화)**

- 한 일: ADR-001 v2 개정(공유 잠재요인 구조), ADR-010 신규(씬파일러 3유형),
  시뮬레이터 재작성, config 파라미터화, 테스트 5건 추가·3건 수정(총 61건 통과).
- 확인된 수치 (latent_factor_strength=0.85, generation_correlation 0.31~0.35):
  - 개별 |corr| 0.3094 ~ 0.3497 — SPEC §4 요건 0.30~0.50 충족
  - 클래스 내부 평균 상관 **0.14~0.19 → 0.669** (공유 잠재요인 작동 확인)
  - test AUC: traditional 0.8338 / alternative **0.8963** / combined **0.9470**
    (v1: 0.834 / 0.996 / 0.9999 — 결합 누수 제거됨)
  - 씬파일러 유형: missing_only 17,879(39.7%) / history_only 13,529(30.1%) / both 13,592(30.2%)
  - 판정 조건 분리 성공: 연체결측 31,471 / 이력<12 27,121 / **교집합 13,592 / 합집합 45,000**
  - 유효 bias_ratio 범위 축소: 0 ~ 0.3 (0.4 부터 상관 0.30 미만)
- **전체 학습 미실행** — combined 0.9470 이 목표 구간(0.85~0.90) 밖. 아래 이슈 참조.

**2026-09-08 (5세션) — ADR-008/009, 학습 파이프라인, 대안 데이터 누수 발견**

- 한 일: ADR-008(보호 속성 학습 제외), ADR-009(credit_history_months + 씬파일러 판정 완성),
  `config.features` 분리, 전처리기 수정, `src/models/train.py`·`src/models/evaluate.py` 구현,
  테스트 13건 추가(총 57건), 스모크 학습 실행.
- 확인된 수치:
  - 씬파일러 판정: is_thin_filer 비율이 thin_filer_ratio 와 일치(0.0/0.1/0.3/0.5 전부).
    두 조건(연체결측 / 이력<12개월)은 **동일 집합** — 교집합 = 합집합 = 45,000건 (ADR-009 에 명시)
  - `credit_history_months`: 씬파일러 3~11(평균 7.0), 일반 12~120(평균 66.0), age 상관 +0.204
  - 대안 5변수 상관 0.3037~0.4855 (범위 유지)
  - 스모크(n_repeats=1, 18조합 × 5 fold): **408.9초**
  - traditional_only test AUC — LGBM 0.8342 / XGB 0.8308 / LR 0.7593 (목표 0.78 충족)
  - **alternative_only test AUC 0.9864~0.9959, combined 0.9901~0.9999 — 대안 데이터 누수**
- 성능 튜닝: LightGBM `n_jobs=-1` 57.5초 → `n_jobs=4` 1.4초 (스레드 경합). config `models.n_jobs=4`.

**2026-09-08 (4세션) — SPEC 반영, ADR-002 개정, 전처리기**

- 한 일: `docs/SPEC.md` 저장, PROGRESS 체크리스트를 스펙 실제 요건으로 교체,
  ADR-002 v2 개정(편향 대상 55+ → 20-34, 하향폭 0.1 → 0.3), `src/data/preprocessor.py` 구현,
  `tests/test_preprocessor.py` 16건 추가.
- 확인된 수치:
  - 편향 방향 정상화: `(20-34) − (55+)` telecom 격차가 bias 0/0.1/0.5 에서
    **−0.077 → −0.107 → −0.227** 로 단조 심화 (v1 에서는 격차가 줄어들었음)
  - 성별 격차(female − male): +0.0009 → −0.0291 → −0.1490
  - 대안 5변수 상관: 0.3037 ~ 0.4955, bias_ratio 0.5 까지 전부 0.3~0.5 유지
  - 타겟 양성 10,026건, bias_ratio 무관하게 불변
  - 분할: train 104,999(0.7000) / valid 22,500(0.1500) / test 22,500(0.1500),
    양성률 6.6839% / 6.6844% / 6.6844%
  - 전처리 결과 25열 (수치형 15 스케일링 + One-Hot 5 + 결측플래그 3 + 통과 1 + target)
  - train 중앙값: MonthlyIncome 5,400.0 / NumberOfDependents 0.0
- `pytest` 39건 전부 통과.

**2026-09-08 (3세션) — 시뮬레이터 구현** / **(2세션) — 로더** / **(1세션) — 뼈대 세팅**

## 3. 다음 할 일 (우선순위 순)

1. `src/explainer/` — SHAP 전역(Summary, Dependence 2개)/지역(Waterfall) + 거절 사유 Top 5
3. `src/fairness/` — DI Ratio, Equalized Odds, ThresholdOptimizer (ADR-004)
5. `src/explainer/` — SHAP 전역/지역 + 거절 사유 Top 5 생성
6. `src/fairness/` — DI Ratio, Equalized Odds, ThresholdOptimizer (ADR-004)
7. `src/api/main.py` — 7개 엔드포인트 + Pydantic 스키마
8. `src/dashboard/app.py` — 5개 탭
9. Dockerfile / docker-compose 실체화 후 통합 구동 검증
10. 실험 리포트(PDF) 작성, README 팀 역할 추가, GitHub 원격 생성 + Issue/PR

## 3-1. 리포트 결론 (확정)

> 전통 신용정보가 결손된 씬파일러에게 대안 데이터는 통계적으로 유의한 AUC 개선을 제공한다
> (씬파일러 +0.152, 전체 +0.110; ANOVA F=4245.65, p<0.001, η²=0.9933).
> 그러나 보호 속성을 학습에서 제외해도 편향은 대안 피처를 경유해 전파되며,
> 그 경로는 **측정 편향(성별)** 과 **피처 내재 상관(연령대)** 두 종류로 구분된다.
> 후자는 편향 원인 제거가 예측력 손실을 수반하므로 **사후 임계값 조정이 현실적 대응**이다.

전문은 `docs/report_assets/fairness_narrative.md` §5.

## 4. 미해결 이슈

- **[해소] Docker 이미지 빌드 실패** — 두 가지 원인이었고 둘 다 고쳤다.
  1. **BuildKit 빌더의 DNS 실패**: `apt-get update` 가 `Could not resolve 'deb.debian.org'` 로
     exit 100. 호스트(1.2 MB/s)와 `docker run` 컨테이너에서는 정상이었고 빌더에서만 실패했다.
     → `docker-compose.yml` 에 `build.network: host` 추가로 해소.
  2. **`fc-cache` 명령 부재**: `fonts-nanum` 만 설치하면 fontconfig 바이너리가 없어 exit 127.
     → `fontconfig` 패키지 추가.

- **[해소] `combined` AUC 목표 구간** — 목표를 0.90~0.95 로 정정하고 현행 설정을 최종 채택.
  (아래는 정정 근거로 남긴다.)
- **[해소] gender 편향 미검출** — bias_ratio 0.3 상향(ADR-002 v3)으로 DI 0.9502 → 0.9191.
  민감도 표에서 bias_ratio 에 단조 반응함을 확인해 SPEC §4 요건을 실증했다.
  다만 DI 절대값은 여전히 0.8 위라 성별에서는 "기준 위반" 수준의 편향이 아니다.
- **[핵심 발견 · 리포트 필수] age_group 격차는 주입 편향이 아니라 피처 설계에서 온다**
  `bias_ratio=0` 에서 이미 age_group min/max DI 가 **0.7840** 으로 4/5 규칙 미달이고,
  0.3 까지 올려도 0.7747 로 거의 변하지 않는다. 대안 피처가 소득·연령을 신호원으로 생성되므로
  (ADR-001 `nuisance = 0.7·소득 + 0.3·연령`) 연령 정보가 대안 피처를 통해 그대로 전파된다.
  반면 gender 격차는 bias_ratio=0 에서 DI 1.0030 으로 완전 균등 → 전적으로 주입 편향의 결과.
  **두 속성의 편향 원인이 다르다는 이 대비가 리포트의 핵심 결과다.**
  → `docs/report_assets/fairness_narrative.md` §2-1 에 기록 완료.
- **[동결 · 기록만] 상관 하한 마진 0.0099**
  `utility_payment_rate` 상관 0.3099 로 SPEC 하한 0.30 에 근접. **요건은 충족한다.**
  F1~F5 결과는 현재 데이터로 **동결**했다(시뮬레이터·모델·공정성 재생성 없음).
  지금까지의 모든 수치가 이 데이터 위에서 정합되어 있고, 재생성 시 리포트 자료 전체를
  다시 만들어야 하기 때문이다. → ADR-001 "한계" 절에 기록 완료.
- **[참고] gender 편향 미검출 (bias_ratio 0.1 시절 기록)**
  ADR-002 v2 는 female 과 20-34 의 telecom/utility 를 `bias_ratio × 0.3` 만큼 하향한다.
  현재 `bias_ratio=0.1` → 하향폭 0.03. 그 결과 **age_group 에서는 편향이 잡히지만
  (min/max DI 0.7732, EO FPR차 0.1496 위반) gender 에서는 잡히지 않는다 (DI 0.9502)**.
  연령대는 대안 피처의 연령 신호(ADR-001)와 주입 편향이 같은 방향으로 누적되는 반면,
  성별은 무작위 합성이라 주입 편향만으로 격차를 만들어야 하기 때문으로 보인다.
  선택지: (a) `bias_ratio` 0.1 → 0.2~0.3 상향 후 데이터 재생성 + 재학습(약 15분),
  (b) 현행 유지하고 "성별은 프록시 경로가 약해 편향이 전파되지 않았다"로 해석,
  (c) `bias_shift_per_ratio` 를 0.3 → 0.4 이상으로 상향(단 상관 요건 재확인 필요).
  ※ `bias_ratio` 상한은 0.3 (ADR-002 v2, 그 이상은 SPEC §4 상관 요건 위반).
- **[해석 주의] DI Ratio 정의에 따라 판정이 갈린다**
  SPEC §1 용어 정의는 "보호 집단 승인율 / 비보호 집단 승인율"이라 age_group 은 0.8358 로 충족이지만,
  규제 실무에서 흔한 min/max 정의로는 0.7732 로 **미달**이다. 두 값을 모두 산출해 리포트에 병기한다.
- **[한계 · 리포트 필수 기재] η² 가 0.95 이상으로 극단적**
  두 ANOVA 모두 η² > 0.95 다. 집단이 서로 다른 피처 구성/알고리즘으로 **고정**돼 있고
  동일 데이터에 반복 CV 를 적용해 집단 내 fold 변동이 매우 작기 때문이다(표준편차 0.0024~0.0047).
  실무적 우월성의 크기로 읽으면 안 되며 집단 평균 차이의 절대 크기로 판단해야 한다.
  또한 반복 CV fold 는 완전 독립이 아니라 p-value 가 실제보다 작게 나온다(ADR-011 한계).
  → `anova.py` 가 결과에 `effect_size_note` 로 자동 첨부한다.
- **[한계 · 리포트 필수 기재] 시뮬레이션 데이터의 AUC 상한**
  실제 신용평가 모델 AUC 는 통상 0.85~0.88 이다. 본 프로젝트의 combined AUC 0.94 대는
  SPEC §4 의 상관 요건(0.3~0.5)을 부도율 6.68% 이진 타겟에 적용한 결과인 **구조적 상한**이며
  실무 성능 추정치가 아니다. 절대 수치가 아니라 **상대 비교**(피처그룹 간, 집단 간)로만 해석한다.
  → ADR-001 v2 "한계" 절, `docs/report_assets/simulator_diagnosis.md` §6 에 기재 완료.
- **[참고 · 해소된 경위] `combined` AUC 가 초기 목표 구간(0.85~0.90)에 들어오지 않았던 건**
  ADR-001 v2(공유 잠재요인) 적용으로 v1 의 결합 누수는 제거됐다(combined 0.9999 → **0.9470**).
  그러나 지시하신 목표 구간 0.85~0.90 에는 미달이며, 지정된 조정 레버
  (`latent_factor_strength` 0.90~0.92)로는 도달하지 못한다.
  | w | traditional | alternative | combined |
  |---:|---:|---:|---:|
  | 0.85 | 0.8338 | 0.8963 | 0.9470 |
  | 0.90 | 0.8338 | 0.8870 | 0.9420 |
  | 0.92 | 0.8338 | 0.8843 | 0.9412 |
  | 0.94 | 0.8338 | 0.8798 | 0.9396 |
  남은 0.94 는 누수가 아니라 **전통(0.834)과 대안(0.896)의 정상적인 상보성**이다.
  구간에 넣으려면 생성 상관을 낮춰야 하는데 그러면 SPEC §4 를 위반한다.
  | 생성 상관 | 실제 \|corr\| | alternative | combined | SPEC |
  |---|---|---:|---:|---|
  | 0.31~0.35 (현행) | 0.309~0.346 | 0.8963 | 0.9470 | 충족 |
  | 0.26~0.30 | 0.260~0.297 | 0.8453 | 0.9206 | 위반 |
  | 0.22~0.26 | 0.220~0.259 | 0.7980 | 0.8985 | 위반 |
  | 0.18~0.22 | 0.181~0.219 | 0.7549 | 0.8788 | 위반 |
  선택지: (a) 현행 유지하고 combined 0.947 로 전체 학습 진행, (b) 상관 0.22~0.26 으로 낮추고
  스펙 위반을 리포트에 명시, (c) 목표 구간을 0.90~0.95 로 조정.
- **[해소] v1 대안 데이터 결합 누수** — ADR-001 v2 로 해소.
  개별 상관은 스펙대로 0.30~0.49 범위 안에 있으나, 다섯 변수를 함께 쓰면 타겟이 거의 복원된다.
  `alternative_only` test AUC 0.9864~0.9959, `combined` 0.9901~0.9999.
  원인은 버그가 아니라 **부도율 6.68% 인 이진 타겟에 대해 점이연상관 0.3~0.5 가 매우 강한 조건**이라는 점이다.
  단일 피처만으로도 |corr|=0.30 → AUC 0.813, |corr|=0.49 → AUC 0.944 가 나온다.
  거기에 현재 구현은 타겟을 조건으로 하면 다섯 변수가 서로 거의 독립이라 정보가 그대로 합산된다
  (조건부 독립 가정 결합 Cohen's d = 3.96 → 이론 AUC 0.9975, 관측 0.9959).
  선택지(측정치는 100,000행 시뮬레이션 기준):
  | 안 | 개별 \|corr\| | 5변수 결합 AUC | 스펙 문구 |
  |---|---|---|---|
  | A. 현행 유지 | 0.30~0.49 | 0.998 | 충족 |
  | B1. 범위 하단만 사용 | 0.30~0.35 | 0.985 | 충족 |
  | **B2. 범위 하단 + 공유 잠재요인 0.85** | 0.30~0.35 | **0.870** | 충족 |
  | C2. 상관 0.15~0.25 + 공유요인 | 0.15~0.25 | 0.789 | **위반** |
  B2 는 스펙의 0.3~0.5 조건을 지키면서 결합 판별력을 낮춘다. 공유 잠재요인은
  "성실한 사람은 통신비도 공과금도 잘 낸다"는 현실 구조이기도 하다. ADR-001 개정이 필요하다.
- **[해소] 보호 속성 학습 사용 여부** — ADR-008 로 확정: 학습 피처에서 제외, 데이터에는 유지.
- **[해소] 씬파일러 판정 두 번째 조건** — ADR-009 로 확정: `credit_history_months` 추가.
- **[해소] 씬파일러 판정 두 조건이 동일 집합** — ADR-010 으로 유형을 분화해 해소.
  교집합 13,592 < 합집합 45,000.
- **[제약] 유효 bias_ratio 범위 0 ~ 0.3** — ADR-001 v2 로 생성 상관 여유가 줄어 0.4 부터
  telecom/utility 상관이 0.30 아래로 내려간다. 실험은 0 ~ 0.3 에서 수행한다.
- **[제약] bias_ratio 상한** — `bias_shift_per_ratio=0.3` 에서 bias_ratio 가 0.6 을 넘으면
  telecom/utility 의 target 상관이 0.30 아래로 떨어진다. 실험은 0 ~ 0.5 범위에서 수행한다.
- **[예정] 승인 임계값 0.30 재검토** — PR 곡선 확인 후 조정 (ADR-003).
- **[예정] bias_shift_per_ratio 재조정** — 모델 학습 후 실제 DI Ratio 를 보고 판단 (ADR-002 v2).
- **[미검증] Docker 이미지 빌드** — Docker 29.2.1 / Compose v5.1.0 사용 가능하나
  Dockerfile / docker-compose.yml 은 아직 스켈레톤.
- **[제출요건] GitHub 원격 저장소 없음** — commit 10개·Issue/PR 3개 요건 충족을 위해 원격 생성 필요.

## 5. 세션 히스토리

### 2026-09-10 (15세션) — 리포트 목차 · EDA 그림 · 4장
- 목차 확정(32p), EDA 그림 8종 생성, 4장 작성 완료(316줄).
- ADR-004 v2 문서 누락 복구 — 조용히 실패한 자동 편집 사례.
- 4장 인용 수치 교차 검증으로 불일치 3건 수정.

### 2026-09-08 (14세션) — Docker 구성 · README 최종화(F8)
- Dockerfile/compose 완성(4개 서비스), bootstrap 스크립트, README 258줄, GitHub 준비 문서.
- 산출물은 볼륨 마운트 방식 채택(이미지에 구울 대상이 리포지토리에 없다).
- 이미지 빌드가 BuildKit DNS 실패로 막힘 — 환경 문제이며 구성 자체는 완성.

### 2026-09-08 (13세션) — Streamlit 대시보드(F7)
- 5개 탭 구현, AppTest 로 전 탭 렌더링·심사 플로우·API 미기동 동작 검증. 테스트 166건 통과.
- 모델 파일 미로드를 AST import 검사 + 코드 grep 검사로 테스트에 고정.
- 버그 2건: /explain 이 연체 '기록 없음'(씬파일러) 입력에서 500, app.py import 시 앱 실행.

### 2026-09-08 (12세션) — FastAPI(F6) · 결과 동결
- 7개 엔드포인트 구현, /predict 평균 20.3ms, Swagger UI 정상. 테스트 148건 통과.
- F1~F5 결과 동결. 리포트 결론 문장 확정(fairness_narrative.md §5).
- 버그 3건 수정: 전처리기 피클 네임스페이스, credit_history_months 무시, stale ANOVA 서빙.

### 2026-09-08 (11세션) — bias_ratio 0.3 · 목적함수 변경
- ADR-002 v3(bias_ratio 0.3), ADR-004 v2(balanced_accuracy). 재학습 639.6초, combined AUC 0.9453.
- 민감도 분석으로 gender 는 주입 편향, age_group 은 피처 설계 상관이 원인임을 분리 확인.
- 완화 후 승인율 0.82~0.85 (이전 목적함수에서는 0.955) — 심사 기능 유지.

### 2026-09-08 (10세션) — 공정성 검증 · Mitigation
- age_group 에서 proxy discrimination 확인: min/max DI 0.7732, EO FPR차 0.1496 위반.
- gender 는 DI 0.9502 로 편향 미검출 → bias_ratio 상향 여부 사용자 결정 대기.
- ThresholdOptimizer 적용 후 DI·EO 전부 기준 충족하나 승인율 0.80→0.96, AUC 0.95→0.74.

### 2026-09-08 (9세션) — SHAP 설명가능성 · ADR-012
- ADR-012: XGB vs LGBM 유의차 없음(p=0.6137)이라 챔피언 근거를 SHAP 지원 성숙도로 변경.
- `src/explainer/` 구현: 전역 중요도 1위 telecom_payment_rate(1.346), 대안 피처가 traditional 대비 3.3배.
- 거절 사유 생성 버그 2건 수정(SHAP 입력 스케일 혼동, 센티넬 포함 평균). 테스트 104건 통과.

### 2026-09-08 (8세션) — ANOVA 검증 · 챔피언 재선정
- ANOVA 검증1 F=4514.60 η²=0.9937 / 검증2 F=660.02 η²=0.9586, 둘 다 정규성·등분산 충족.
- Tukey: XGBoost vs LightGBM 만 유의하지 않음(p=0.61) — ADR-012 로 선정 근거 변경.
- 챔피언을 SMOTE → class_weight 로 교체(설명가능성 근거, ADR-001 부기). Registry v2.

### 2026-09-08 (7세션) — 전체 학습 완료
- 전체 학습 360 fits / 616.4초. 최고 xgboost|combined|smote test AUC 0.9491, KS 0.7576, PSI 0.0004.
- 씬파일러 유형별 uplift 확인: both +0.195 > missing_only +0.173 > history_only +0.102 (ADR-010 효과).
- MLflow run 18개 + Registry champion 등록, 리포트 자료(진단 문서 + 그림 5종) 생성 시작.

### 2026-09-08 (6세션) — ADR-001 v2 · ADR-010
- ADR-001 v2: 공유 잠재요인 구조로 결합 누수 제거 (combined AUC 0.9999 → 0.9470).
- ADR-010: 씬파일러 3유형 분화로 SPEC §4 의 OR 정의가 실제 작동 (교집합 13,592 < 합집합 45,000).
- 테스트 61건 통과. 전체 학습은 combined 목표 구간 미달로 사용자 확인 대기.

### 2026-09-08 (5세션) — ADR-008/009 · 학습 파이프라인 · 누수 발견
- ADR-008(보호 속성 학습 제외), ADR-009(credit_history_months, 씬파일러 판정 완성) 작성 및 반영.
- `train.py` / `evaluate.py` 구현, 스모크 18조합 × 5 fold 408.9초. traditional_only AUC 0.83 (목표 충족).
- **대안 데이터 결합 누수 발견** (alternative_only AUC 0.996). 전체 학습은 보정 결정 후로 보류.

### 2026-09-08 (4세션) — SPEC 반영 · ADR-002 개정 · 전처리기
- `docs/SPEC.md` 저장, PROGRESS 체크리스트를 F1~F8 실제 요건으로 교체.
- ADR-002 v2: 편향 대상 55+ → 20-34, 하향폭 0.1 → 0.3. 연령 격차가 단조 심화되도록 정상화.
- `preprocessor.py` 구현(연체 결측은 센티넬+플래그로 보존, train 전용 fit), 테스트 39건 통과.

### 2026-09-08 (3세션) — 시뮬레이터 구현
- `src/config.py` 분리, `simulator.py` 구현: 인구통계 / 대안 5변수 / 씬파일러 / 편향 주입.
- 대안 5변수 target 상관 0.30~0.50 범위 확보, 씬파일러 비율·결측률·타겟 불변 검증.
- `pytest` 20건 통과.

### 2026-09-08 (2세션) — 로더 구현
- SHAP 0.52 + numpy 2.5 / pandas 3.0 / sklearn 1.9 조합 정상 동작 확인, Docker 29.2.1 사용 가능.
- ADR-007(age<18 제거) 작성, `loader.py` 구현 → 1건 제거 / 149,999행 / 양성률 6.684%.
- `tests/test_loader.py` 6건 통과.

### 2026-09-08 (1세션) — 뼈대 세팅
- 프로젝트 구조·설정·ADR 6건·문서 3종 생성, `.venv` 및 의존성 고정 완료.
- GMSC 150,000행 검증 통과 (양성률 6.684%, MonthlyIncome/NumberOfDependents 결측 확인).
- 첫 커밋 `chore: project skeleton, config, ADRs`.
