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
- [x] RepeatedStratifiedKFold fold 별 AUC 저장 (`data/processed/fold_auc.csv`, ANOVA 입력)
- [x] 스모크 실행 완료 (n_repeats=1, 90 fold, 408.9초)
- [ ] **전체 실행 보류** — 아래 미해결 이슈의 대안 데이터 누수 결정 대기
- [ ] `models/` 에 `.joblib` 저장 (MLflow 로깅 시 함께 저장, 스모크는 `--no-mlflow` 로 미저장)
- [ ] MLflow: 실험 3개 이상 기록, Model Registry 등록
- [ ] 성능 목표 확정 — 현재 수치는 누수 상태 기준이라 확정 불가

### F3. 설명가능성(XAI)
- [ ] `src/explainer/` — SHAP TreeExplainer
- [ ] 전역: Summary Plot (상위 15개 변수), Dependence Plot 2개 이상
- [ ] 지역: 개별 고객 Waterfall Plot
- [ ] 거절 사유 Top 5 — 필드 `feature`, `current_value`, `shap_contribution`, `explanation`
- [ ] 자연어 설명 생성 (예: "통신비 정상납부율이 78%로, 동일 연령대 평균(92%) 대비 낮습니다 (기여도: -0.15)")

### F4. 공정성 검증 및 Bias Mitigation
- [ ] `src/fairness/` — 보호 속성 성별(male/female), 연령대(20-34/35-54/55+)
- [ ] DI Ratio 계산 함수, **0.8 미만 또는 1.25 초과 시 WARNING 로그**
- [ ] Equalized Odds 계산 함수 (TPR 차이, FPR 차이, 기준 ≤ 0.10)
- [ ] Bias Mitigation 1개 이상 — Fairlearn ThresholdOptimizer (ADR-004)
- [ ] Mitigation 전/후 비교 표 (리포트 포함)

### F5. ANOVA 통계 검증
- [ ] `src/statistics/anova.py`
- [ ] 검증 1: 피처 그룹별(전통 Only / 대안 Only / 통합) One-way ANOVA
- [ ] 검증 2: 알고리즘별(LR / XGB / LGBM) One-way ANOVA
- [ ] 각 검증에 F-통계량, p-value, η²(효과 크기) 포함
- [ ] p < 0.05 시 Tukey HSD 사후검정
- [ ] 관측 단위는 RepeatedStratifiedKFold 5×4 = 그룹당 n=20 (ADR-005)

### F6. API 서버 (FastAPI, :8000)
- [ ] `src/api/main.py`, 모든 엔드포인트에 Pydantic Request/Response 스키마
- [ ] `GET  /health` → `{"status", "model_version", "timestamp"}`
- [ ] `POST /predict` → `{"credit_score": int, "approval_status": str, "default_probability": float, "risk_grade": str}`
- [ ] `POST /explain` → `{"rejection_reasons": [...], "shap_values": {...}}`
- [ ] `GET  /fairness` → `{"di_ratio", "equalized_odds_tpr_diff", "equalized_odds_fpr_diff"}`
- [ ] `GET  /fairness/mitigation` → `{"before": {...}, "after": {...}}`
- [ ] `GET  /metrics` → `{"auc", "ks", "precision", "recall", "f1"}`
- [ ] `GET  /anova` → `{"test_1": {...}, "test_2": {...}}`
- [ ] `localhost:8000/docs` Swagger UI, `/predict` 응답 2초 이내

### F7. 대시보드 (Streamlit, :8501)
- [ ] `src/dashboard/app.py`, 5개 탭
- [ ] 탭1 **고객 심사** — 입력 폼(나이·소득·부채비율 등), 승인/거절, 신용점수, 리스크 등급
- [ ] 탭2 **XAI 분석** — Summary Plot, 개별 Waterfall Plot, 거절 사유 Top 5 텍스트
- [ ] 탭3 **공정성** — 집단별 승인율 막대 차트, DI Ratio 수치, Mitigation 전/후 비교 차트
- [ ] 탭4 **모델 성능** — ROC Curve, PR Curve, 혼동행렬 히트맵
- [ ] 탭5 **통계 검증** — ANOVA 결과 테이블(F, p, η²), Tukey HSD 결과
- [ ] `.pkl` 직접 로드 금지, 모든 추론은 FastAPI HTTP 경유
- [ ] 위험도별 색상 구분 (고위험=빨강 / 중위험=노랑 / 저위험=초록)

### F8. 프로젝트 구조 및 제출 요건
- [x] 역할별 `src/` 폴더 분리 (data/models/explainer/fairness/statistics/api/dashboard)
- [x] 설정값 config 분리, 하드코딩 절대경로 없음
- [x] `requirements.txt` 버전 명시 (+ `requirements.lock`)
- [x] `tests/` pytest **39건** (요건 10건 이상), 현재 100% 통과 (요건 80% 이상)
- [x] `.env.example`
- [ ] `README.md` — 팀 구성원 역할 항목 추가 필요
- [ ] `docker-compose up` 으로 api(8000) + dashboard(8501) + mlflow(5000) 동시 구동
- [ ] `docs/experiment_report.pdf` 20페이지 이상
- [ ] `models/` 에 학습된 모델 파일
- [ ] commit 10개 이상 (현재 3개), GitHub Issue/PR 3개 이상 (현재 0개, 원격 저장소 미생성)

## 2. 마지막 세션 요약

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

1. **[사용자 결정 필요] 대안 데이터 결합 누수 해소** — 아래 미해결 이슈 참조. 결정 후 시뮬레이터 재보정
2. 재보정된 데이터로 전체 학습 실행 (n_repeats=4, 예상 25~30분) + MLflow 로깅 + Registry 등록
3. `src/statistics/anova.py` — 검증 1·2 + Tukey HSD (ADR-005)
5. `src/explainer/` — SHAP 전역/지역 + 거절 사유 Top 5 생성
6. `src/fairness/` — DI Ratio, Equalized Odds, ThresholdOptimizer (ADR-004)
7. `src/api/main.py` — 7개 엔드포인트 + Pydantic 스키마
8. `src/dashboard/app.py` — 5개 탭
9. Dockerfile / docker-compose 실체화 후 통합 구동 검증
10. 실험 리포트(PDF) 작성, README 팀 역할 추가, GitHub 원격 생성 + Issue/PR

## 4. 미해결 이슈

- **[사용자 결정 필요 · 최우선] 대안 데이터 5변수의 결합 누수**
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
- **[성질] 씬파일러 판정 두 조건이 동일 집합** — 교집합 = 합집합 = 45,000건. ADR-009 에 기록.
  OR 정의의 의미를 데이터로 보이려면 씬파일러 내부를 세 갈래로 나눠야 한다. 현재는 보류.
- **[제약] bias_ratio 상한** — `bias_shift_per_ratio=0.3` 에서 bias_ratio 가 0.6 을 넘으면
  telecom/utility 의 target 상관이 0.30 아래로 떨어진다. 실험은 0 ~ 0.5 범위에서 수행한다.
- **[예정] 승인 임계값 0.30 재검토** — PR 곡선 확인 후 조정 (ADR-003).
- **[예정] bias_shift_per_ratio 재조정** — 모델 학습 후 실제 DI Ratio 를 보고 판단 (ADR-002 v2).
- **[미검증] Docker 이미지 빌드** — Docker 29.2.1 / Compose v5.1.0 사용 가능하나
  Dockerfile / docker-compose.yml 은 아직 스켈레톤.
- **[제출요건] GitHub 원격 저장소 없음** — commit 10개·Issue/PR 3개 요건 충족을 위해 원격 생성 필요.

## 5. 세션 히스토리

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
