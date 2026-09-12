# 부록

## A. 설계 결정(ADR) 색인

| ADR | 주제 | 결정 요지 | 개정 이력 |
|---|---|---|---|
| 001 | 대안 데이터 시뮬레이터 | 대안 5변수 = 타겟 신호 + **공유 잠재요인**(소득·연령 신호 + 공통 노이즈) 구조. 생성 상관 0.31~0.35, `w = 0.85` | v1 개별 노이즈 → **v2** 공유 잠재요인(결합 누수 제거). 부기: 불균형 처리 `class_weight` 채택. 2026-09-11 상관 측정 조건 명시 |
| 002 | 편향 주입 (`bias_ratio`) | `female` · `20-34` 의 통신·공과금 납부율을 `bias_ratio × 0.3` 하향. **타겟은 불변** | v1 대상 55+ → v2 20-34, 계수 0.1 → 0.3 → **v3** 기본 `bias_ratio` 0.3 |
| 003 | 점수 매핑 · 임계값 | `score = 1000 × (1 − p)`, 등급 A≥800 ~ E, 승인 `p < 0.30` | 사후 관찰: 임계값 0.30 의 정밀도 0.2899 기록 |
| 004 | 공정성 완화 기법 | Fairlearn `ThresholdOptimizer(equalized_odds)` — 후처리라 전/후를 같은 모델 위에서 비교 | **v2** 목적함수 `accuracy` → `balanced_accuracy_score` |
| 005 | ANOVA 관측 단위 | `RepeatedStratifiedKFold(5×4)` → 그룹당 n = 20 | — |
| 006 | 메인 데이터셋 | GMSC. German Credit 은 로더 함수만 두고 미사용(표본 1,000건) | — |
| 007 | 연령 이상치 | `age < 18` 행 제거(1행). 대체하지 않음 | — |
| 008 | 보호 속성 | `gender` · `age_group` 을 학습 피처에서 제외, 데이터에는 유지 | — |
| 009 | `credit_history_months` | 거래 이력 개월수 합성으로 씬파일러 OR 판정 완성. 판정 근거 컬럼은 학습 제외 | — |
| 010 | 씬파일러 유형 분화 | `missing_only` 40% / `history_only` 30% / `both` 30% | — |
| 011 | ANOVA 통제 방식 | 비교 요인 외 나머지를 챔피언 조합으로 고정, pooled 결과는 참고치로 병기 | — |
| 012 | 챔피언 알고리즘 근거 | XGBoost 채택, 근거는 **성능이 아니라 SHAP 지원 성숙도**(Tukey p_adj = 0.6468) | 2026-09-11 Tukey·학습시간 수치를 최종 데이터로 갱신 |

원문은 `docs/decisions/ADR-0NN-*.md`. 각 ADR 은 결정 · 근거 · **버린 대안**을 함께 적는다.

---

## B. API 명세

기본 주소 `http://localhost:8000`, 문서 `/docs`(Swagger UI). 모든 요청·응답에 Pydantic 스키마가 있다.

| # | 메서드 · 경로 | 입력 | 주요 응답 필드 |
|---:|---|---|---|
| 1 | `GET /health` | — | `status`(healthy/degraded), `model_version`, `champion`, `artifacts_loaded`, `timestamp` |
| 2 | `POST /predict` | 고객 정보 JSON | `credit_score`(0~1000), `approval_status`, `default_probability`, `risk_grade`(A~E), `threshold` |
| 3 | `POST /explain` | 고객 정보 JSON | `rejection_reasons[5]`, `total_rejection_factors`, `shap_values`(18개), `base_value` |
| 4 | `GET /metrics` | — | `auc`, `ks`, `precision`, `recall`, `f1`, `psi` |
| 5 | `GET /fairness` | — | `di_ratio`, `di_ratio_min_max`, `equalized_odds_tpr_diff`, `equalized_odds_fpr_diff`, `worst_attribute`, `by_attribute` |
| 6 | `GET /fairness/mitigation` | — | `before` / `after` / `delta`, `adjusted_thresholds`(집단별), `note` |
| 7 | `GET /anova` | — | `test_1`, `test_2` — `f_statistic`, `p_value`, `effect_size_eta_squared`, `post_hoc_tukey`, `diagnostics`, `effect_size_note` |

### 요청 스키마 (`POST /predict`, `POST /explain`)

| 그룹 | 필드 | 비고 |
|---|---|---|
| 전통 | `RevolvingUtilizationOfUnsecuredLines`, `age`, `DebtRatio`, `MonthlyIncome`, `NumberOfOpenCreditLinesAndLoans`, `NumberRealEstateLoansOrLines`, `NumberOfDependents` | `MonthlyIncome` · `NumberOfDependents` 가 `null` 이면 학습 중앙값으로 대체 |
| 전통 (연체) | `NumberOfTime30-59DaysPastDueNotWorse`, `NumberOfTime60-89DaysPastDueNotWorse`, `NumberOfTimes90DaysLate` | **`null` = 기록 없음**(씬파일러). 중앙값 대체 없이 센티넬 + 플래그로 보존 |
| 대안 | `telecom_payment_rate`, `utility_payment_rate`, `spending_consistency`(0~1), `regular_payment_count`, `app_login_frequency` | |
| 보호 속성 · 세그먼트 | `gender`, `age_group`, `credit_history_months` | **학습에 쓰이지 않는다.** `age_group` 은 생략 시 `age` 로 계산 |

필드명은 GMSC 원본 컬럼명(alias)과 snake_case 둘 다 받는다. 알 수 없는 필드는 거부한다(`extra="forbid"`).

### 거절 사유 항목 (`rejection_reasons[]`)

`rank`, `feature`, `feature_name_kr`, `current_value`, `reference_value`(동일 연령대 평균),
`shap_contribution`(신용도 관점, 음수 = 불리), `shap_value_default`(부도 관점 원본), `explanation`,
`direction_consistent`. 명세 §3 의 필수 4개 필드를 포함한다(6.5절).

---

## C. 재현 절차

```bash
# 1) 산출물 생성 (약 15분, 최초 1회)
docker compose --profile setup run --rm bootstrap     # 또는: bash scripts/bootstrap.sh

# 2) 서비스 기동 → localhost:8000 / 8501 / 5000
docker compose up

# 3) 테스트
pytest                                                # 171건

# 4) 리포트 수치 대조 · HTML 생성
python scripts/verify_report_numbers.py
python scripts/build_report.py                        # → docs/experiment_report.html
```

부트스트랩에 포함되지 않는 분석(데이터를 여러 번 재생성하므로 별도 실행):

```bash
python -m src.analysis.eda            # 3·4장 그림 8종
python -m src.fairness.sensitivity    # 7.3절 bias_ratio 민감도 (0.0 ~ 0.3 재학습)
```

모든 난수는 `config.project.random_state = 42` 로 고정돼 있어 같은 입력에서 같은 산출물이 나온다.
예외는 SHAP Summary · Dependence Plot 의 점 배치(렌더링 난수)로, 그림의 픽셀은 달라져도 수치는 같다.

PDF 는 `docs/experiment_report.html` 을 브라우저에서 열고 **인쇄 → PDF로 저장**(용지 A4,
배경 그래픽 켜기, 머리글·바닥글 끄기)으로 만든다.

---

## D. 테스트 구성

| 파일 | 개수 | 검증 대상 |
|---|---:|---|
| `test_simulator.py` | 24 | 대안 5변수 상관 요건, `thin_filer_ratio` · `bias_ratio` 반응, 씬파일러 유형 분화 |
| `test_preprocessor.py` | 19 | 결측 처리(센티넬·중앙값), 스케일링, 층화 분할 비율, train 전용 fit |
| `test_loader.py` | 6 | GMSC 로드, `age < 18` 제거, 결측률 로그 |
| `test_train.py` | 6 | 조합 생성, 불균형 처리 파이프라인, 챔피언 선택 |
| `test_evaluate.py` | 7 | AUC · KS · PSI 계산 |
| `test_explainer.py` | **25** | SHAP 형태·가산성, 거절 사유 5개와 필수 필드, 참조 평균(센티넬 제외), 방향 일치, 한글 조사, 축 절단 |
| `test_fairness.py` | 15 | DI 두 정의, Equalized Odds, WARNING 로그, ThresholdOptimizer 전/후 |
| `test_anova.py` | 22 | F · p · η², Tukey HSD, 정규성·등분산 진단, 통제 vs pooled |
| `test_api.py` | **29** | 7개 엔드포인트 스키마, 응답 시간, 연체 `null` 입력, OpenAPI · Swagger |
| `test_dashboard.py` | 18 | 탭 5개 렌더링, **모델 파일 미로드(AST 검사)**, API 미기동 시 동작 |
| **합계** | **171** | 명세 요건: 10개 이상, 80% 이상 통과 → **171개 전부 통과** |

9장의 오류에 대응하는 회귀 테스트: `test_display_row_changes_values_not_ranking`(#1),
`test_reference_value_excludes_flagged_sentinel_rows`(#2), `test_explain_accepts_missing_delinquency`(#3),
`test_sentinel_with_missing_flag_is_described_as_no_record`(#12).

---

## E. 수치 대조 스크립트

리포트에 인용한 수치가 산출물과 어긋나는 것(9.2절의 stale 수치, 5건)을 막으려고 대조를 스크립트로 고정했다.

```bash
python scripts/verify_report_numbers.py                 # 전체
python scripts/verify_report_numbers.py --chapters 3 6  # 일부 장
```

| 항목 | 내용 |
|---|---|
| 방식 | 산출물(`data/processed/`, 원본 CSV, MLflow 산출 CSV)에서 값을 **다시 계산**해 문서와 같은 형식으로 만든 뒤, 해당 문자열이 문서의 **같은 줄**에 있는지 확인 |
| 처리 | 수학 기호 마이너스(−)와 하이픈(-)을 동일 취급, 백분율은 half-up 반올림, 일부 검사는 모델 재학습·API 호출로 값을 얻음 |
| 대조 범위 | 3 · 6 · 9 · 10 · 11장 전 수치와 읽는 법 · 요약 · 1 · 2장 · 부록의 핵심 수치 |
| 제외 | 4 · 5 · 7 · 8장 — 작성 세션에서 같은 방식으로 수작업 대조했고 기록은 `PROGRESS.md` 에 있다 |
| 한계 | 문서에 값이 **있는지**만 보고 문맥의 정확성은 보지 않는다. 실행에 수 분이 걸린다. 과거 값 대조는 git 이력에 의존한다 |
| 종료 코드 | 전부 일치 0, 하나라도 불일치 1 |
| 대조 결과 | 2026-09-12 기준 **389건 전부 일치** (읽는 법 3 · 요약 38 · 1장 14 · 2장 37 · 3장 80 · 6장 102 · 9장 27 · 10장 36 · 11장 18 · 부록 34) |

---

## F. 산출물 목록

| 경로 | 내용 |
|---|---|
| `models/champion_xgboost_v1.0.pkl` 외 3개 | 최종 모델 (알고리즘별 1개 + 챔피언) |
| `data/processed/{train,valid,test}.parquet` | 전처리·분할 결과 (24열) |
| `data/processed/model_results.csv` · `fold_auc.csv` | 18조합 지표 · fold AUC 360건 |
| `data/processed/thin_filer_uplift.csv` | 세그먼트별 uplift |
| `data/processed/anova_results.json` | ANOVA 두 검정 + Tukey + 진단 + pooled 참고치 |
| `data/processed/explain/` | 전역 중요도 CSV, 거절 사유 샘플 JSON, 그림 7종 |
| `data/processed/fairness/` | 공정성 전/후 JSON, `bias_ratio` 민감도 CSV, 그림 4종 |
| `docs/report_assets/` | 리포트 그림 24종, 진단 · 공정성 서술 문서 |
| `mlflow.db` · `mlruns/` | MLflow run 18개, Model Registry |
