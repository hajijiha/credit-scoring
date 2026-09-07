# 37_XAI 기반 대안신용평가 시스템 개발

> 과제 원문. 이 문서는 출처 그대로 보존한다. 우리 구현 결정은 `docs/decisions/ADR-*.md`,
> 진행 상황은 `PROGRESS.md` 를 본다.

!4_응용_11.jpeg

### 1. 미션 소개

대안신용평가(Alternative Credit Scoring)는 기존 금융 이력 외에 통신비 납부, 소비 패턴, 공과금 납부 등 비금융 데이터를 활용하여 개인의 신용을 평가하는 기법이다. 2024년 기준 국내 금융이력부족자(씬파일러)는 약 1,100만 명으로, 카카오뱅크(AUC 87%), 케이뱅크(대손율 26.5% 감소), 토스뱅크(TSS 모델)에서 이들의 대출 심사에 대안신용평가를 활용하고 있다.

이 미션에서는 Kaggle 신용평가 데이터셋과 대안 데이터 시뮬레이터를 결합한 신용평가 ML 모델을 개발하고, SHAP 기반 설명가능성(XAI) 모듈, AIF360/Fairlearn 기반 공정성 검증 및 Bias Mitigation 모듈을 구현한다. 최종 결과물은 FastAPI 기반 대출 심사 API와 Streamlit 대시보드로 구성되며, MLflow를 통해 모든 실험을 체계적으로 추적한다.

학습자는 불균형 데이터에서의 Precision-Recall Trade-off, 모델 해석을 위한 전역/지역 SHAP 분석, Disparate Impact Ratio와 Equalized Odds 기반 공정성 검증, Bias Mitigation 기법 적용, 그리고 ANOVA 통계 검증을 통한 모델 성능 비교까지 전체 파이프라인을 경험한다. 금융위원회 금융분야 AI 가이드라인이 요구하는 설명가능성과 공정성 기준, 대출 거절 사유 고지 의무를 시스템에 반영하는 방법을 이해한다.

용어 정의 (Glossary)

씨파일러(Thin-filer) : 금융 이력이 부족하여 기존 CB(신용평가) 모델로는 적절한 신용평가가 어려운 개인. 사회초년생, 프리랜서, 주부 등이 해당된다.
본 미션에서는 금융 연체 관련 컬럼 2개 이상 결측치이거나 신용카드 거래 이력이 12개월 미만인 고객으로 정의한다.
CB(Credit Bureau) : 신용정보원. 금융 기관이 개인의 신용정보를 수집하여 신용등급을 산출하는 기관
XAI(eXplainable AI) : AI 모델의 예측 결과에 대해 인간이 이해할 수 있는 설명을 제공하는 기술
DI Ratio(Disparate Impact Ratio) : 보호 집단의 승인율을 비보호 집단의 승인율으로 나눈 비율. 0.8 미만 시 차별적 영향이 있는 것으로 판단

---

### 2. 최종 결과물

다음 4가지 기능이 정상 동작하는 시스템 1개를 완성하고,  모든 코드와 문서는 GitHub 레포지토리(Repository) URL을 제출한다.

레포지토리 필수 포함 항목

| 항목 | 설명 | 필수 여부 |
| --- | --- | --- |
| `README.md` | 프로젝트 개요, 실행 방법, 팀 구성원 역할, 디렉토리 구조 | ✅ 필수 |
| `docker-compose.yml` | API + Dashboard + MLflow 통합 실행 설정 | ✅ 필수 |
| `src/` | 전체 소스 코드 (데이터, 모델, API, 대시보드) | ✅ 필수 |
| `tests/` | pytest 기반 단위 테스트 (10개 이상) | ✅ 필수 |
| `docs/experiment_report.pdf` | 실험 리포트 (20페이지 이상) | ✅ 필수 |
| `models/` | 학습된 모델 파일 (.pkl 또는 .joblib) | ✅ 필수 |
| `notebooks/` | EDA, 모델링, 분석 Jupyter Notebook | 권장 |
| `.env.example` | 환경 변수 템플릿 | 권장 |

1. **시스템 아키텍처**
    - 입력 : `docker-compose up` 명령 실행
    - 출력 : API 서버(localhost:8000), 대시보드(localhost:8501), MLflow UI(localhost:5000)가 동시에 정상 구동됨
2. **신용평가 및 설명 API**
    - 입력 : 고객 정보 JSON (`POST /predict`)
    - 출력 : 신용 점수(0-1000), 승인 여부(APPROVED/REJECTED), 부도 확률(0-1), 리스크 등급(A-E)
3. **심사역용 웹 대시보드**
    - 입력 : 대시보드에서 고객 정보 입력
    - 출력 : 심사 결과, SHAP Waterfall Plot, 거절 사유 Top 5, 공정성 지표 차트
4. **MLflow 실험 추적 시스템**
    - 입력 : 모델 학습 실행
    - 출력 : MLflow UI에서 실험별 파라미터, 메트릭, 모델 아티팩트 비교 가능
5. **최종 결과 보고서**
    - 형식 : PDF, 20페이지 이상
    - 내용 : 문제 정의, EDA, 모델 비교, XAI 분석, 공정성 검증, ANOVA 통계 검증, 결론

---

### 3. 과제 목표

이 과제를 마친 후, 학습자는 아래를 스스로 설명할 수 있어야 한다.

- 불균형 데이터 처리의 필요성과 SMOTE, class_weight 기법의 원리를 설명할 수 있다.
- SHAP Value가 양수/음수일 때 모델 예측에 미치는 영향과 Base Value의 의미를 설명할 수 있다.
- Disparate Impact Ratio와 Equalized Odds의 계산 방법과 임계값(0.8, 0.10)의 근거를 설명할 수 있다.
- ANOVA F-통계량과 p-value의 의미, 사후 검정(Tukey HSD)이 필요한 상황을 설명할 수 있다.
- MLflow를 활용한 실험 추적과 모델 버전 관리의 필요성을 설명할 수 있다.
- 금융위원회 AI 가이드라인이 요구하는 설명가능성과 공정성 기준을 이해한다.
- 씨파일러 문제가 금융 포용성에 미치는 영향과 대안 데이터의 활용 의의를 설명할 수 있다.

---

### 4. 기능 요구 사항

다음 요구사항을 모두 만족해야 한다.

1. **데이터 파이프라인**
    - 기본 데이터셋

        | 데이터셋 | 설명 | 획득 방법 |
        | --- | --- | --- |
        | Give Me Some Credit | Kaggle 부도 예측 데이터 (150,000건) | `kaggle competitions download -c GiveMeSomeCredit` |
        | German Credit | UCI 신용 데이터 (1,000건) | `from ucimlrepo import fetch_ucirepo` |

        - `src/data/loader.py` 파일이 존재하고, Give Me Some Credit 또는 German Credit 데이터셋을 로드하는 함수가 구현되어 있다.
        - 데이터 로드 시 컬럼명, 데이터 타입, 결측치 비율이 콘솔 또는 로그에 출력된다.
    - 대안 데이터 시뮬레이터
        - `src/data/simulator.py` 파일이 존재한다.
        - 시뮬레이터가 다음 5개 대안 변수를 생성한다: `telecom_payment_rate`(통신비 정상납부율), `utility_payment_rate`(공과금 납부율), `spending_consistency`(소비 일관성 점수), `regular_payment_count`(정기결제 건수), `app_login_frequency`(앱 로그인 빈도)
        - 생성된 대안 변수와 Target 변수 간 상관계수가 0.3 ~ 0.5 범위 내에 있음을 `df.corr()` 또는 리포트에서 확인할 수 있다.
        - 시뮬레이터 함수에 `thin_filer_ratio` 파라미터가 존재하고, 이 값을 조정하면 씬파일러 비율이 변경된다.
        - 시뮬레이터 함수에 `bias_ratio` 파라미터가 존재하고, 이 값을 조정하면 성별/연령대별 승인율 편향이 변경된다.
    - 씨파일러 판정 기준 (Thin-filer Labeling)
        - 씨파일러 판정 함수가 명시적으로 구현되어 있어야 한다.
        - 판정 기준: 금융 연체 관련 컬럼 2개 이상 결측치이거나 신용카드 거래 이력 12개월 미만인 고객을 씨파일러로 분류한다.
        - 씨파일러 집단 분리 평가 방법: 대안 데이터 포함 모델 vs 미포함 모델의 AUC uplift를 측정하여 효과를 검증한다.
    - 시뮬레이터 구현 접근 방법 훈트
        - 아래는 정답이 아닌 접근 방법 후보이다. 학습자는 자유롭게 선택하여 구현한다.
            - 방법 1 : 공분산 행렬 기반 생성 - Target 변수와의 상관관계를 공분산 행렬로 정의하고 다변량 정규분포에서 샘플링
            - 방법 2 : Cholesky 분해 - 상관 구조를 유지하면서 독립적인 랜덤 변수를 상관된 변수로 변환
            - 방법 3 : Copula 기반 샘플링 - 변수 간 비선형 의존성을 모델링하여 생성
            - 검증 체크 : 생성 후corr()로 상관계수 확인, bias_ratio 변경 시 집단별 승인율 변화 확인
    - 전처리 및 분할
        - `src/data/preprocessor.py` 파일이 존재한다.
        - 결측치 처리 로직이 구현되어 있다 (중앙값/평균값 대체 또는 삭제).
        - 수치형 변수에 StandardScaler 또는 MinMaxScaler가 적용된다.
        - 범주형 변수에 One-Hot Encoding 또는 Label Encoding이 적용된다.
        - Train/Validation/Test 분할 비율이 **70:15:15** 또는 **80:10:10**이다.
        - 분할 시 `stratify=y` 옵션이 적용되어 클래스 비율이 유지된다.
2. **신용평가 모델**
    - 모델 학습
        - 최소 3개 알고리즘(Logistic Regression, XGBoost, LightGBM)을 학습한다.
        - 각 모델에 5-Fold Stratified Cross-Validation을 적용한다.
        - 불균형 데이터 처리 기법을 2개 이상 비교 실험한다 (`class_weight='balanced'`, SMOTE, Focal Loss 중 선택).
        - 최종 모델 파일이 `models/` 폴더에 `.pkl` 또는 `.joblib` 형식으로 저장되어 있다.
        - 모델 파일명에 버전 정보가 포함되어 있다 (예: `xgboost_v1.0.pkl`).
    - 성능 목표

        | 지표 | 목표값 | 검증 방법 |
        | --- | --- | --- |
        | Test Set AUC-ROC | **≥ 0.78** | 리포트 또는 `/metrics` API 응답에서 확인 |
        | Test Set KS Statistic | **≥ 0.28** | 리포트에서 확인 |
        | 씬파일러 집단 AUC 향상 | **+0.03 이상** | 대안 데이터 포함 모델 vs 미포함 모델 비교 |
        | PSI (Population Stability Index) | **< 0.1** | Train/Test 분포 안정성 검증 |

    - MLflow 실험 추적
        - `docker-compose up` 실행 후 `localhost:5000`에서 MLflow UI가 정상 표시된다.
        - MLflow UI에서 최소 3개 이상의 실험(모델별)이 기록되어 있다.
        - 각 실험에 다음 항목이 기록되어 있다: `params`(하이퍼파라미터), `metrics`(AUC, KS, F1), `artifacts`(모델 파일).
        - 최종 선택 모델이 MLflow Model Registry에 `Production` 또는 `Staging` 태그로 등록되어 있다.
3. **설명가능성(XAI) 모듈**
    - `src/explainer/` 폴더가 존재한다.
    - SHAP TreeExplainer 또는 KernelExplainer를 사용한다.
    - 전역 해석: SHAP Summary Plot 이미지가 생성되고, 상위 15개 변수가 표시된다.
    - 전역 해석: SHAP Dependence Plot 이미지가 최소 2개 변수에 대해 생성된다.
    - 지역 해석: 개별 고객의 SHAP Waterfall Plot 이미지가 생성된다.
    - `/explain` API 호출 시 거절 사유 Top 5가 JSON 형식으로 반환된다.
    - 각 거절 사유에 다음 4개 필드가 포함된다: `feature`(변수명), `current_value`(현재값), `shap_contribution`(SHAP 기여도), `explanation`(자연어 설명)
    - 거절 사유 자연어 설명 형식 예시:

        ```bash
        "통신비 정상납부율이 78%로, 동일 연령대 평균(92%) 대비 낮습니다 (기여도: -0.15)"
        ```

4. **공정성 검증 및 Bias Mitigation 모듈**
    - `src/fairness/` 폴더가 존재한다.
    - 보호 속성으로 성별(male/female)과 연령대(20-34/35-54/55+)를 사용한다.
    - *Disparate Impact Ratio(DI Ratio)**를 계산하는 함수가 존재한다.
    - DI Ratio 계산 결과가 0.8 미만 또는 1.25 초과일 때 WARNING 로그가 출력된다.
    - Equalized Odds를 계산하는 함수가 존재한다 (TPR 차이, FPR 차이).
    - Bias Mitigation 기법이 1개 이상 구현되어 있다 (Reweighting 또는 Threshold Adjustment).
    - Mitigation 적용 전/후 공정성 지표 비교 표가 리포트에 포함되어 있다.
    - `/fairness` API 호출 시 DI Ratio와 Equalized Odds 값이 JSON으로 반환된다.
    - `/fairness/mitigation` API 호출 시 Mitigation 전/후 비교 결과가 JSON으로 반환된다.
5. **ANOVA 통계 검증**
    - `src/statistics/anova.py` 파일이 존재한다.
    - 다음 2가지 ANOVA 검증을 필수로 수행한다:

        | 검증 | 유형 | 비교 대상 | 목적 |
        | --- | --- | --- | --- |
        | **검증 1** | One-way ANOVA | 피처 그룹별 모델 (전통 Only / 대안 Only / 통합) | 대안 데이터 효용성 검증 |
        | **검증 2** | One-way ANOVA | 알고리즘별 (LR / XGB / LGBM) | 최적 알고리즘 선정 근거 |

    - 각 검증 결과에 F-통계량, p-value, η²(효과 크기)**가 포함된다.
    - p < 0.05인 경우 사후 검정(Tukey HSD) 결과가 포함된다.
    - `/anova` API 호출 시 검증 결과가 JSON으로 반환된다.
    - 리포트에 ANOVA 결과 해석이 1페이지 이상 포함되어 있다.
    - ANOVA 결과 해석 가이드
        - η²(효과 크기) 해석 기준

            | **η² 범위** | **해석** | **의미** |
            | --- | --- | --- |
            | 0.01 ~ 0.06 | Small (weak) | 집단 간 차이가 작음 |
            | 0.06 ~ 0.14 | Medium (moderate) | 집단 간 차이가 중간 |
            | 0.14 이상 | Large (strong) | 집단 간 차이가 큼 |

        - Tukey HSD 해석 예시 : 통합 모델(M=0.79)이 전통 모델(M=0.72)보다 통계적으로 유의하게 높은 AUC를 보였다(p_adj=0.002). 반면, 전통 모델과 대안 모델 간에는 유의한 차이가 없었다(p_adj=0.18). 이는 대안 데이터가 단독으로는 전통 데이터와 유사한 예측력을 보이지만, 통합 시 시너지 효과가 발생함을 시사한다.
6. **API 서버 (FastAPI)**
    - `src/api/main.py` 파일이 존재한다.
    - 다음 7개 엔드포인트가 구현되어 있다:

        | 엔드포인트 | 메서드 | 성공 시 HTTP Status | 응답 형식 |
        | --- | --- | --- | --- |
        | `/health` | GET | 200 | `{"status": "healthy", "model_version": "...", "timestamp": "..."}` |
        | `/predict` | POST | 200 | `{"credit_score": int, "approval_status": str, "default_probability": float, "risk_grade": str}` |
        | `/explain` | POST | 200 | `{"rejection_reasons": [...], "shap_values": {...}}` |
        | `/fairness` | GET | 200 | `{"di_ratio": float, "equalized_odds_tpr_diff": float, "equalized_odds_fpr_diff": float}` |
        | `/fairness/mitigation` | GET | 200 | `{"before": {...}, "after": {...}}` |
        | `/metrics` | GET | 200 | `{"auc": float, "ks": float, "precision": float, "recall": float, "f1": float}` |
        | `/anova` | GET | 200 | `{"test_1": {...}, "test_2": {...}}` |

        - 모든 엔드포인트에 Pydantic Request/Response Schema가 정의되어 있다.
        - `localhost:8000/docs`에서 Swagger UI가 정상 표시되고, 모든 API를 테스트할 수 있다.
        - `/predict` API 응답 시간이 2초 이내이다 (단일 요청 기준).
7. **대시보드 (Streamlit)**
    - `src/dashboard/app.py` 파일이 존재한다.
    - `localhost:8501`에서 대시보드가 정상 표시된다.
    - 대시보드에 다음 5개 탭(또는 페이지)이 존재한다:

        | 탭 이름 | 필수 구성 요소 |
        | --- | --- |
        | **고객 심사** | 입력 폼(나이, 소득, 부채비율 등), 승인/거절 결과, 신용 점수, 리스크 등급 |
        | **XAI 분석** | SHAP Summary Plot 이미지, 개별 고객 Waterfall Plot, 거절 사유 Top 5 텍스트 |
        | **공정성** | 집단별 승인율 막대 차트, DI Ratio 수치, Mitigation 전/후 비교 차트 |
        | **모델 성능** | ROC Curve, Precision-Recall Curve, 혼동행렬 히트맵 |
        | **통계 검증** | ANOVA 검증 결과 테이블 (F값, p값, η²), Tukey HSD 결과 |

        - 대시보드 코드에서 모델 파일(.pkl)을 직접 로드하지 않는다 (MSA 구조).
        - 모든 추론 및 지표 조회는 FastAPI 서버와 HTTP 통신을 통해 수행한다.
        - 위험도에 따라 색상이 구분된다 (예: 고위험=빨강, 중위험=노랑, 저위험=초록).
8. **프로젝트 구조 및 제출 요건**
    - `docker-compose up` 명령 한 번으로 다음 3개 서비스가 동시에 실행된다:
        - API 서버: `localhost:8000`
        - 대시보드: `localhost:8501`
        - MLflow UI: `localhost:5000`
    - `README.md`에 다음 내용이 포함된다: 프로젝트 개요, 팀 구성원 역할, 실행 방법, 디렉토리 구조, 환경 변수 설명
    - `requirements.txt` 또는 `pyproject.toml`에 의존성이 버전과 함께 명시된다.
    - `tests/` 폴더에 pytest 기반 단위 테스트가 10개 이상 존재한다.
    - `pytest` 명령 실행 시 80% 이상의 테스트가 통과한다.
    - `docs/experiment_report.pdf` 파일이 존재하고, 20페이지 이상이다.
    - GitHub 레포지토리에 최소 10개 이상의 commit이 존재한다.
    - GitHub Issue 또는 Pull Request를 3개 이상 활용한 이력이 있다.

---

### 5. 보너스 과제 (선택)

1. LIME 비교 분석
    - 해야 할 작업 : LIME을 사용하여 개별 예측 설명을 생성하고, 동일 샘플에 대한 SHAP 결과와 비교한다. 두 기법의 상위 5개 중요 변수 일치율을 계산한다.
    - 배움 포인트 : 모델 불가지론적(Model-Agnostic) XAI 기법들의 특성 차이와 한계를 이해한다.
2. Counterfactual Explanation
    - 해야 할 작업 : 거절된 고객에게 "어떤 조건이 바뀌면 승인될 수 있는지"를 제시하는 Counterfactual Explanation 기능을 구현한다.
    - 배움 포인트 : Actionable AI의 개념과 고객 관점의 XAI 활용 방안을 경험한다.
3. Two-way ANOVA (심화 통계 검증)
    - 해야 할 작업 : 고객 세그먼트(씬파일러/일반) × 모델 구성(전통/통합)에 대한 Two-way ANOVA를 수행하고, 교호작용(Interaction Effect)을 분석한다.
    - 배움 포인트 : 다요인 분산분석의 원리와 교호작용 해석 방법을 경험한다.
4. 모델 드리프트 탐지
    - 해야 할 작업 : PSI와 함께 CSI(Characteristic Stability Index)를 계산하여 개별 피처 수준의 분포 변화를 탐지하는 모듈을 구현한다.
    - 배움 포인트 : 프로덕션 환경에서의 모델 모니터링 필요성과 드리프트 대응 전략을 이해한다.
5. A/B 테스트 시뮬레이션
    - 해야 할 작업 : 기존 CB 모델과 대안신용평가 모델의 승인 결과를 비교하는 A/B 테스트 시뮬레이션을 구현하고, 예상 대손율 변화를 분석한다.
    - 배움 포인트 : 금융 도메인에서의 모델 도입 의사결정 과정을 경험한다.

---

- 구현 지침

    아래는 문제 해결에 도움이 될 수 있는 검색 키워드와 권장 순서다. 정답이 아니므로 자유롭게 접근해도 된다.

    ### 권장 구현 순서

    1. Docker 기반 개발 환경 구축 (Python 3.10, 필수 라이브러리)
    2. Git 레포지토리 생성 및 프로젝트 구조 설계
    3. 공개 데이터셋 다운로드 및 EDA (Give Me Some Credit)
    4. 대안 데이터 시뮬레이터 설계 및 구현
    5. 씬파일러/씩파일러 레이블링 로직 구현
    6. 전처리 파이프라인 개발 (결측치, 정규화, 인코딩)
    7. Stratified Train/Valid/Test 분할
    8. MLflow 설정 및 실험 추적 환경 구축
    9. Logistic Regression 베이스라인 모델 학습(MLflow 로깅)
    10. XGBoost, LightGBM 모델 학습 및 하이퍼파라미터 튜닝
    11. 불균형 처리 기법 비교 (class_weight, SMOTE)
    12. 5-Fold Cross-Validation 및 성능 평가
    13. AUC-ROC, KS 통계량 계산 모듈 개발
    14. 씬파일러 집단 대안 데이터 효과 분석
    15. ANOVA 통계 검증 모듈 개발 (2가지 검증 + Tukey HSD)
    16. SHAP TreeExplainer 적용 및 전역 해석 시각화
    17. SHAP 지역 해석 (Force Plot, Waterfall) 구현
    18. 거절 사유 Top 5 자동 생성 로직 개발
    19. 공정성 지표 계산 모듈 (DI Ratio, Equalized Odds)
    20. Bias Mitigation 구현 (Reweighting, Threshold Adjustment)
    21. FastAPI 서버 구현 (7개 엔드포인트)
    22. Pydantic 스키마 정의 및 Swagger UI 설정
    23. Streamlit 대시보드 구현 (5개 탭)
    24. API ↔ 대시보드 연동 테스트
    25. pytest 유닛 테스트 10개 이상 작성
    26. docker-compose.yml 작성 및 통합 테스트(API + Dashboard + MLflow)
    27. 실험 리포트(PDF) 20페이지 이상 작성
    28. README.md 완성, GitHub Issue/PR 정리, 최종 점검

    ### 참고 키워드

    | 영역 | 키워드 |
    | --- | --- |
    | 데이터 | Give Me Some Credit, Kaggle API, 시뮬레이터, Stratified Split, Thin-filer |
    | 모델링 | XGBoost, LightGBM, SMOTE, class_weight, 5-Fold CV, GridSearchCV |
    | XAI | SHAP TreeExplainer, Summary Plot, Dependence Plot, Waterfall Plot, LIME |
    | 공정성 | Fairlearn, AIF360, Disparate Impact, Equalized Odds, Reweighting, Threshold Adjustment |
    | 통계 | scipy.stats.f_oneway, statsmodels ANOVA, Tukey HSD, eta squared, effect size |
    | MLOps | MLflow, mlflow.sklearn.log_model, MLflow Model Registry, experiment tracking |
    | 배포 | FastAPI, Streamlit, Docker Compose, Pydantic, Swagger, uvicorn |

- 추천 데이터셋 및 획득 방법
    1. 필수 활용 데이터셋

        | 데이터셋 | 설명 | 크기 | 획득 방법 | 권장 |
        | --- | --- | --- | --- | --- |
        | **Give Me Some Credit** | Kaggle 부도 예측 경진대회 데이터 | 150,000건, 11컬럼 | `kaggle competitions download -c GiveMeSomeCredit` | ✅ **기본** |
        | **German Credit** | UCI 신용 데이터 (클래식 벤치마크) | 1,000건, 20컬럼 | `from ucimlrepo import fetch_ucirepo; fetch_ucirepo(id=144)` | 대안 |

    2. 선택 데이터셋 (심화)

        | 데이터셋 | 설명 | 크기 | 획득 방법 | 용도 |
        | --- | --- | --- | --- | --- |
        | **Taiwan Credit** | 대만 신용카드 부도 데이터 | 30,000건, 24컬럼 | UCI 아카이브 엑셀 다운로드 | 추가 검증 |
        | **Home Credit Default Risk** | Kaggle 대회 (복잡한 관계형 데이터) | 300,000건, 7개 테이블 | `kaggle competitions download -c home-credit-default-risk` | 보너스 과제 |
        | **LendingClub Loan Data** | P2P 대출 데이터 | 2,000,000건+ | Kaggle 검색 후 다운로드 | 대규모 실험 |

    3. 대안 데이터 시뮬레이터

        | 변수명 | 설명 | 생성 로직 |
        | --- | --- | --- |
        | `telecom_payment_rate` | 통신비 정상납부율 (0-1) | `np.clip(0.85 + 0.15 * target_corr + noise, 0, 1)` |
        | `utility_payment_rate` | 공과금 납부율 (0-1) | `np.clip(0.80 + 0.20 * target_corr + noise, 0, 1)` |
        | `spending_consistency` | 소비 일관성 점수 (0-100) | 월별 소비 표준편차의 역수 기반 |
        | `regular_payment_count` | 정기결제 건수 (0-20) | `np.random.poisson(lam=5 + 3 * income_level)` |
        | `app_login_frequency` | 앱 로그인 빈도 (0-30) | `np.random.poisson(lam=10 + 5 * age_factor)` |

        ```python
        # 시뮬레이터 예시 코드
        def generate_alternative_data(df, target_col='SeriousDlqin2yrs',
                                       thin_filer_ratio=0.3, bias_ratio=0.1):
            """
            대안 데이터 시뮬레이터

            Args:
                df: 원본 데이터프레임
                target_col: 타겟 변수명
                thin_filer_ratio: 씬파일러 비율 (0-1)
                bias_ratio: 편향 강도 (0: 편향 없음, 1: 최대 편향)

            Returns:
                대안 변수가 추가된 데이터프레임
            """
            n = len(df)
            target = df[target_col].values

            # 상관관계 0.3~0.5 유지하며 노이즈 추가
            noise = np.random.normal(0, 0.15, n)
            df['telecom_payment_rate'] = np.clip(0.85 - 0.2 * target + noise, 0, 1)

            # 씬파일러 레이블링
            financial_cols = ['NumberOfTime30-59DaysPastDueNotWorse',
                              'NumberOfTimes90DaysLate',
                              'NumberOfTime60-89DaysPastDueNotWorse']
            df['is_thin_filer'] = (df[financial_cols].isna().sum(axis=1) >= 2).astype(int)

            return df
        ```

    4. 데이터 획득 상세 가이드
        - 방법 1: Kaggle CLI 사용

            ```bash
            # Kaggle API 설치 및 인증
            pip install kaggle
            mkdir -p ~/.kaggle
            # kaggle.json 파일을 ~/.kaggle/에 복사

            # Give Me Some Credit 다운로드
            kaggle competitions download -c GiveMeSomeCredit
            unzip GiveMeSomeCredit.zip -d data/raw/
            ```

        - 방법 2: UCI ML Repository 사용

            ```bash
            from ucimlrepo import fetch_ucirepo

            # German Credit 데이터셋
            german_credit = fetch_ucirepo(id=144)
            X = german_credit.data.features
            y = german_credit.data.targets
            ```

        - 방법 3: 직접 다운로드
            - Give Me Some Credit: https://www.kaggle.com/c/GiveMeSomeCredit/data
            - German Credit: https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data))
            - Taiwan Credit: https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients

- 참고자료

    ### 필수 문서

    | 자료 | 설명 | 링크 |
    | --- | --- | --- |
    | **금융위원회 금융분야 AI 가이드라인** | 금융 AI 설명가능성, 공정성 규제 기준 | 금융위원회 보도자료 |
    | **신용정보법 제36조의2** | 자동화평가 결과에 대한 설명 및 이의제기 | 국가법령정보센터 |
    | **SHAP 공식 문서** | SHAP 라이브러리 사용법 및 이론 | shap.readthedocs.io |
    | **Fairlearn 공식 문서** | 공정성 지표 및 Mitigation 기법 | fairlearn.org |
    | **MLflow 공식 문서** | 실험 추적, 모델 레지스트리 사용법 | mlflow.org/docs |

    ### 📖 추천 논문 및 아티클

    | 제목 | 저자/출처 | 핵심 내용 |
    | --- | --- | --- |
    | **"A Unified Approach to Interpreting Model Predictions"** | Lundberg & Lee (NeurIPS 2017) | SHAP 이론적 기반 |
    | **"Fairness Constraints: Mechanisms for Fair Classification"** | Zafar et al. (AISTATS 2017) | Disparate Impact, Equalized Odds 정의 |
    | **"Why Should I Trust You?: Explaining the Predictions of Any Classifier"** | Ribeiro et al. (KDD 2016) | LIME 설명 기법 |
    | **대안신용평가 모형의 개인신용평가 활용** | 한국신용정보원 (2023) | 국내 대안신용평가 현황 |
    | **Alternative Data and Credit Risk** | BIS Working Papers (2022) | 글로벌 대안데이터 활용 동향 |

    ### 🔧 라이브러리 공식 문서

    | 라이브러리 | 용도 | 문서 링크 |
    | --- | --- | --- |
    | **scikit-learn** | ML 모델링, 전처리 | scikit-learn.org |
    | **XGBoost** | Gradient Boosting | xgboost.readthedocs.io |
    | **LightGBM** | Gradient Boosting | lightgbm.readthedocs.io |
    | **imbalanced-learn** | SMOTE, 불균형 처리 | imbalanced-learn.org |
    | **SHAP** | 모델 해석 | shap.readthedocs.io |
    | **AIF360** | 공정성 도구 (IBM) | aif360.readthedocs.io |
    | **Fairlearn** | 공정성 도구 (MS) | fairlearn.org |
    | **FastAPI** | API 서버 | fastapi.tiangolo.com |
    | **Streamlit** | 대시보드 | docs.streamlit.io |
    | **MLflow** | 실험 추적 | mlflow.org/docs |
    | **scipy.stats** | ANOVA 통계 검정 | docs.scipy.org |
    | **statsmodels** | 고급 통계 분석 | statsmodels.org |

    ### 🎓 온라인 학습 자료

    | 주제 | 플랫폼 | 추천 강의/자료 |
    | --- | --- | --- |
    | **SHAP 이해하기** | YouTube | "SHAP Values Explained" by StatQuest |
    | **불균형 데이터 처리** | Kaggle | "Imbalanced Classification" Tutorial |
    | **Fairness in ML** | Google | "ML Fairness" Crash Course |
    | **FastAPI Tutorial** | YouTube | "FastAPI Full Course" by freeCodeCamp |
    | **MLflow 입문** | Databricks | "MLflow Quickstart" |
    | **ANOVA 통계** | Khan Academy | "ANOVA" 시리즈 |

    ### 🏢 현업 사례 자료

    | 기업/기관 | 자료 | 핵심 내용 |
    | --- | --- | --- |
    | **카카오뱅크** | 기술 블로그 | 대안신용평가 모델 개발기 (AUC 87% 달성) |
    | **케이뱅크** | IR 자료 | 대안데이터 활용 대손율 26.5% 감소 사례 |
    | **토스뱅크** | 보도자료 | TSS(Toss Scoring System) 모델 소개 |
    | **나이스평가정보** | 리서치 | 국내 CB사 대안데이터 활용 현황 |
    | **FICO** | 백서 | "Alternative Data in Credit Scoring" |

    ### 📁 GitHub 참고 레포지토리

    | 레포지토리 | 설명 |
    | --- | --- |
    | slundberg/shap | SHAP 공식 구현 및 예제 |
    | Trusted-AI/AIF360 | IBM AI Fairness 360 |
    | fairlearn/fairlearn | Microsoft Fairlearn |
    | mlflow/mlflow | MLflow 공식 레포지토리 |
    | scikit-learn-contrib/imbalanced-learn | 불균형 데이터 처리 |

---

### 6. 개발 환경

- Python 3.10 이상
- ML 라이브러리: scikit-learn, XGBoost, LightGBM
- XAI 라이브러리: SHAP (shap>=0.42.0)
- 공정성 라이브러리: Fairlearn 또는 AIF360
- 실험 추적: MLflow (mlflow>=2.0.0)
- Docker 및 Docker Compose 사용

---

### 7. 제약 사항

- 데이터
    - 기본 데이터셋은 Give Me Some Credit 또는 German Credit을 사용한다.
    - 테스트 데이터의 라벨을 학습에 사용하는 것을 금지한다.
    - 데이터 분할 시 Stratified Split을 반드시 적용해야 한다.
- 아키텍처
    - Streamlit 코드에서 모델 파일(.pkl)을 직접 로드하는 것을 금지한다.
    - 모든 추론은 FastAPI 서버와 HTTP 통신을 통해 수행해야 한다 (MSA 구조 강제).
    - 모델 파일은 pickle 또는 joblib 형식으로 저장해야 한다.
- 코드 품질
    - GitHub의 Issue와 Pull Request 기능을 활용하여 개발 이력을 남겨야 한다 (최소 3개).
    - 설정값(포트, 경로, 임계값)은 환경 변수 또는 config 파일로 분리해야 한다.
    - 코드에 하드코딩된 절대 경로가 없어야 한다.
- 성능 기준

    | 항목 | 기준 |
    | --- | --- |
    | 통합 모델 AUC-ROC | ≥ 0.78 |
    | 씬파일러 AUC 향상 | 전통 모델 대비 +3%p 이상 |
    | PSI | < 0.1 |
    | API 응답 시간 | ≤ 2초 (단일 요청) |
    | ANOVA 유의성 | 최소 1개 이상 검증에서 p < 0.05 |

- 금융 규제 대응
    - 모든 신용평가 결과에 SHAP 기반 설명을 제공해야 한다.
    - 신용 거절 시 최소 3가지 이상 주요 사유를 제시해야 한다.
    - 공정성 분석 결과를 문서화하고 편향 완화 조치를 기록해야 한다.
- 윤리 및 면책
    - 모든 결과물에 다음 면책 조항(Disclaimer)을 명시해야 한다:

        "본 시스템은 교육 목적으로 개발되었으며, 실제 금융 의사결정에 사용할 수 없습니다."

    - 민감 정보(인종, 종교, 장애 여부 등)를 피처로 사용하는 것을 금지한다.

---

### 8. 결과 예시

아래는 정답이 아니라 참고 예시다. 실제 문구, 디자인, 구현 방식은 달라도 된다.

- 신용평가 API 응답 예시 (`POST /predict`) 예시

    ```json
    {
      "customer_id": "C001234",
      "credit_score": 687,
      "risk_grade": "B",
      "approval_status": "APPROVED",
      "default_probability": 0.12,
      "model_version": "xgboost_v1.2.0",
      "timestamp": "2026-01-30T10:15:30Z"
    }
    ```

- /metrics API 응답 예시    (`POST /predict`) 예시

    ```json
    {
      "customer_id": "C001234",
      "credit_score": 687,
      "risk_grade": "B",
      "approval_status": "APPROVED",
      "default_probability": 0.12,
      "model_version": "xgboost_v1.2.0",
      "timestamp": "2026-01-30T10:15:30Z"
    }
    ```

- 거절 사유 API 응답 예시

    ```json
    {
      "customer_id": "C005678",
      "approval_status": "REJECTED",
      "rejection_reasons": [
        {
          "rank": 1,
          "feature": "telecom_payment_rate",
          "feature_name_kr": "통신비 정상납부율",
          "current_value": 0.78,
          "reference_value": 0.92,
          "shap_contribution": -0.15,
          "explanation": "통신비 정상납부율이 78%로, 동일 연령대 평균(92%) 대비 낮습니다 (기여도: -0.15)"
        },
        {
          "rank": 2,
          "feature": "NumberOfTimes90DaysLate",
          "feature_name_kr": "최근90일 연체횟수",
          "current_value": 3,
          "reference_value": 0,
          "shap_contribution": -0.12,
          "explanation": "최근 90일 연체 횟수가 3회로, 승인 기준(0회) 대비 높습니다 (기여도: -0.12)"
        }
      ],
      "total_rejection_factors": 5
    }
    ```

- ANOVA 검증 결과 API 응답 예시 (`GET /anova`) 예시

    ```json
    {
      "test_1_feature_group": {
        "test_name": "피처 그룹별 AUC 비교 (One-way ANOVA)",
        "groups": ["전통모델", "대안모델", "통합모델"],
        "group_means": {"전통모델": 0.72, "대안모델": 0.68, "통합모델": 0.79},
        "f_statistic": 12.45,
        "p_value": 0.00023,
        "effect_size_eta_squared": 0.34,
        "is_significant": true,
        "post_hoc_tukey": [
          {"comparison": "전통 vs 대안", "mean_diff": 0.04, "p_adj": 0.18, "significant": false},
          {"comparison": "전통 vs 통합", "mean_diff": -0.07, "p_adj": 0.002, "significant": true},
          {"comparison": "대안 vs 통합", "mean_diff": -0.11, "p_adj": 0.0004, "significant": true}
        ]
      },
      "test_2_algorithm": {
        "test_name": "알고리즘별 AUC 비교 (One-way ANOVA)",
        "groups": ["LogisticRegression", "XGBoost", "LightGBM"],
        "group_means": {"LogisticRegression": 0.71, "XGBoost": 0.79, "LightGBM": 0.78},
        "f_statistic": 8.92,
        "p_value": 0.0012,
        "effect_size_eta_squared": 0.28,
        "is_significant": true
      }
    ```

---

### 9. 동료 평가 질문 예시

기능 동작 검증

- [ ]  `git clone` 후 `docker-compose up` 명령으로 시스템이 에러 없이 실행되는가?
- [ ]  `localhost:8000/docs`에서 Swagger UI가 정상 표시되는가?
- [ ]  `localhost:8501`에서 Streamlit 대시보드가 정상 표시되는가?
- [ ]  `localhost:5000`에서 MLflow UI가 정상 표시되고, 3개 이상의 실험 기록이 있는가?
- [ ]  `GET /health` 호출 시 모델 버전과 상태가 JSON으로 반환되는가?
- [ ]  `POST /predict`에 샘플 고객 정보 전송 시 신용 점수, 승인 여부, 리스크 등급이 반환되는가?
- [ ]  `POST /explain`에 거절된 고객 정보 전송 시 거절 사유 Top 5가 자연어 텍스트와 함께 반환되는가?
- [ ]  `GET /fairness` 호출 시 DI Ratio와 Equalized Odds 값이 반환되는가?
- [ ]  `GET /anova` 호출 시 2가지 ANOVA 검증 결과(F값, p값, η²)가 반환되는가?
- [ ]  대시보드에서 고객 정보 입력 후 SHAP Waterfall Plot이 표시되는가?
- [ ]  대시보드 통계 검증 탭에서 ANOVA 결과 테이블이 표시되는가?

코드 구조 및 설계

- [ ]  `src/data/`, `src/models/`, `src/explainer/`, `src/fairness/`, `src/statistics/`, `src/api/`, `src/dashboard/` 폴더가 역할별로 분리되어 있는가?
- [ ]  설정값(포트, 경로, 임계값)이 코드에 하드코딩되지 않고 config 파일 또는 환경 변수로 분리되어 있는가?
- [ ]  `README.md`에 실행 방법, 환경 변수 설명, 디렉토리 구조가 명확히 기술되어 있는가?
- [ ]  `requirements.txt`에 라이브러리 버전이 명시되어 있는가?
- [ ]  `tests/` 폴더에 pytest 테스트가 10개 이상 존재하는가?
- [ ]  `pytest` 실행 시 80% 이상의 테스트가 통과하는가?
- [ ]  GitHub에 10개 이상의 commit과 3개 이상의 Issue/PR이 존재하는가?

핵심 기술 원리 적용

- [ ]  불균형 데이터 처리 기법(class_weight 또는 SMOTE)이 코드에 적용되어 있는가?
- [ ]  5-Fold Stratified Cross-Validation이 코드에 구현되어 있는가?
- [ ]  SHAP Summary Plot(전역 해석)과 Waterfall Plot(지역 해석)이 모두 구현되어 있는가?
- [ ]  거절 사유가 SHAP value 기반으로 자동 생성되고, 자연어 텍스트를 포함하는가?
- [ ]  DI Ratio ≥ 0.8 기준과 Equalized Odds 차이 ≤ 0.10 기준이 코드에 명시되어 있는가?
- [ ]  Bias Mitigation 적용 전/후 비교가 리포트에 명시되어 있는가?
- [ ]  2가지 ANOVA 검증이 수행되고, F-통계량, p-value, η²가 리포트에 기록되어 있는가?
- [ ]  p < 0.05인 검증에 대해 사후 검정(Tukey HSD)이 수행되었는가?
- [ ]  PSI < 0.1이 리포트에서 확인되는가?
- [ ]  모든 실험이 MLflow로 추적되고, 최종 모델이 Registry에 등록되어 있는가?
- [ ]  실험 리포트가 20페이지 이상이고, 문제 정의/모델 비교/XAI/공정성/ANOVA 결과가 포함되어 있는가?
