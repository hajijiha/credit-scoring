# 실험 리포트 목차 제안

> **작성 전 확인용 문서다.** 확정 후 `docs/experiment_report.pdf` 로 작성한다.
> SPEC §2 요건: PDF, **20페이지 이상**, 문제 정의 / EDA / 모델 비교 / XAI 분석 /
> 공정성 검증 / ANOVA 통계 검증 / 결론 포함.

## 1. 가용 재료 조사 결과

### 1-1. 바로 쓸 수 있는 것

| 재료 | 분량 | 대응 장 |
|---|---|---|
| `docs/report_assets/simulator_diagnosis.md` | 188줄 / 7절 | 4장 (거의 그대로) |
| `docs/report_assets/fairness_narrative.md` | 156줄 / 6절 | 7장 (거의 그대로) |
| `docs/decisions/ADR-001 ~ 012` | 12건 / 총 657줄 | 전 장의 근거 인용 |
| `data/processed/anova_results.json` | 12.5KB | 8장 (F, p, η², Tukey, 진단 전부) |
| `data/processed/fairness/fairness_results.json` | 전/후 + 조정 임계값 | 7장 |
| `data/processed/fairness/bias_ratio_sensitivity.csv` | 4수준 × 12지표 | 7장 핵심 표 |
| `data/processed/model_results.csv` | 18조합 × 11지표 | 6장 |
| `data/processed/fold_auc.csv` | 360행 | 8장 |
| `data/processed/thin_filer_uplift.csv` | 3알고리즘 × 6세그먼트 | 6장 핵심 표 |
| `data/processed/explain/global_importance.csv` | 18피처 | 6장·7장 |
| `data/processed/explain/rejection_reasons_sample.json` | 4샘플 | 6장 |

### 1-2. 그림 18종 (전부 생성 완료)

| 그림 | 배치 |
|---|---|
`feature_group_auc.png` | 5장
`cv_fold_auc_distribution.png` | 8장
`roc_pr_curves.png`, `confusion_matrices.png` | 5장
`thin_filer_uplift.png` | 5장
`shap_summary.png` | 6장
`shap_dependence_telecom_payment_rate.png`, `..._RevolvingUtilization....png` | 6장
`shap_waterfall_borderline_rejected.png`, `..._borderline_approved.png` | 6장
`shap_waterfall_extreme_rejected_thin_filer.png`, `..._normal.png` | 6장 (부록 후보)
`fairness_approval_rates.png` | 7장
`bias_ratio_sensitivity.png` | 7장 (핵심)
`fairness_mitigation_comparison.png`, `fairness_tradeoff.png` | 7장

### 1-3. ⚠️ 새로 만들어야 하는 것 (EDA 장의 재료가 없다)

| 필요 산출물 | 용도 | 예상 작업 |
|---|---|---|
| 타겟 분포 + 클래스 불균형 그림 | 3장 | 신규 스크립트 |
| 결측 패턴 그림 (MonthlyIncome 19.82%, NumberOfDependents 2.62%) | 3장 | 신규 |
| 주요 피처 분포 (`age`, `DebtRatio`, `RevolvingUtilization` 이상치) | 3장 | 신규 |
| 상관행렬 히트맵 (전통 10 + 대안 5) | 3장 | 신규 |
| 대안 5변수 × target 상관 막대그림 | 3장·4장 | 신규 |
| 씬파일러 vs 일반 집단 프로파일 비교표 | 3장 | 신규 |
| 클래스 내부(타겟 조건부) 상관 히트맵 v1 vs v2 | 4장 (핵심) | 신규 |

→ **`src/models/eda_assets.py` 를 새로 만들어 7종을 생성해야 한다.** (예상 30분)

---

## 2. 목차 및 페이지 배분 (총 32페이지)

SPEC 최소 20페이지 대비 여유를 두되, 각 장의 근거 데이터가 실제로 있는 만큼만 배분했다.

| 장 | 제목 | 쪽 | 누적 |
|---|---|---:|---:|
| — | 표지 · 면책 조항 · 목차 | 2 | 2 |
| — | 요약 (Executive Summary) | 1 | 3 |
| 1 | 문제 정의 | 2 | 5 |
| 2 | 시스템 개요 | 1 | 6 |
| 3 | 데이터와 EDA | 3 | 9 |
| 4 | **방법론: 대안 데이터 시뮬레이터 설계와 진단** | 5 | 14 |
| 5 | 모델 비교 | 3 | 17 |
| 6 | XAI 분석 | 3 | 20 |
| 7 | 공정성 검증과 완화 | 5 | 25 |
| 8 | ANOVA 통계 검증 | 3 | 28 |
| 9 | 검증 과정에서 발견하고 수정한 오류 | 1.5 | 29.5 |
| 10 | 한계 | 1.5 | 31 |
| 11 | 결론 | 1 | 32 |
| 부록 | ADR 색인 · API 명세 · 재현 절차 | 2 | 34 |

---

## 3. 장별 상세

### 요약 (1p)
- 한 문단 결론 + 성능 요약표(AUC 0.9453 / KS 0.7502 / PSI 0.00054 / 씬파일러 uplift +0.1518)
- **핵심 발견 3줄**: ① 대안 데이터의 기여는 전통 피처 결손이 클수록 커진다
  ② 보호 속성을 제외해도 편향은 대안 피처를 경유해 전파된다
  ③ 그 경로는 측정 편향(성별)과 피처 내재 상관(연령대)으로 갈리며 후자가 더 어렵다

### 1장. 문제 정의 (2p)
- 씬파일러 문제와 금융 포용성 (SPEC §1 배경, 국내 1,100만 명)
- 대안신용평가의 개념, 국내 사례 (카카오뱅크 AUC 87% 등)
- 규제 요건: 금융위 AI 가이드라인, **신용정보법 제36조의2** 자동화평가 설명 의무
- 본 프로젝트의 질문 3개 (성능 / 설명가능성 / 공정성)
- 성공 기준표 (SPEC §2 목표값)

### 2장. 시스템 개요 (1p)
- 아키텍처 그림 (api 8000 / dashboard 8501 / mlflow 5000, MSA 강제)
- 모듈 구성표, 기술 스택, 설정 단일화 원칙 (`config.yaml`)
- **신규 그림 필요 없음** — 텍스트 + 표로 구성 (또는 간단한 ASCII/도식 1개)

### 3장. 데이터와 EDA (3p)
- GMSC 150,000행 → ADR-007 `age<18` 1건 제거 → 149,999행
- 타겟 분포: 양성 10,026건 (6.684%) — 클래스 불균형
- 결측: MonthlyIncome 19.82%, NumberOfDependents 2.62%
- 이상치: `age` min=0(제거됨), `DebtRatio` 최대 z=148.9, `RevolvingUtilization` z=128.6
- 씬파일러 정의(SPEC §4)와 유형 분포: missing_only 39.7% / history_only 30.1% / both 30.2%
- 씬파일러 vs 일반 집단 프로파일 비교
- **그림 4종 신규 생성** (1-3 참조)

### 4장. 방법론: 시뮬레이터 설계와 진단 (5p) ★ 이 프로젝트의 핵심 챕터
`simulator_diagnosis.md` 를 기반으로 확장한다.

4.1 설계 목표와 제약 (0.5p)
  - SPEC §4 요건: 대안 5변수, target 상관 0.3~0.5, `thin_filer_ratio`, `bias_ratio`
  - ADR-001 의 의도: 씬파일러의 전통 피처 정보량을 **구조적으로** 줄여야
    대안 데이터 uplift 가 누수가 아닌 결과로 나온다

4.2 v1 구현과 문제 발견 (1p)
  - v1 결과: traditional 0.834 / alternative **0.996** / combined **0.9999**
  - 씬파일러 uplift **+0.238** (SPEC 목표 +0.03 의 8배)

4.3 원인 분석 (1.5p) ★
  - 개별 상관은 규격 내(0.3031~0.4855)였음을 먼저 보인다 → 규격 위반이 아니다
  - **부도율 6.68% 에서 점이연상관 0.3~0.5 는 매우 강한 조건**:
    효과크기 환산표 (|corr| 0.15~0.49 → Cohen's d, 단일 변수 AUC 0.666~0.944)
  - **조건부 독립이 정보를 합산시켰다**: 클래스 내부 평균 상관 0.19(v1)
    피처별 d = 1.27~2.22 → 결합 **d = sqrt(Σd²) = 3.96** → 이론 AUC **0.9975**
    (관측 0.9959와 일치) → 구현 버그가 아니라 **가정의 결과**
  - **클래스 내부 상관 히트맵 v1 vs v2 신규 그림**

4.4 v2: 공유 잠재요인 도입 (1p)
  - 수식과 구조: `v_i = ρ_i·t_z + √(1-ρ_i²)·m_i`, `m_i = w·shared + √(1-w²)·ε_i`
  - 도메인 근거: "성실한 사람은 통신비도 공과금도 잘 낸다"
  - 클래스 내부 상관 0.19 → **0.669**

4.5 w 레버 실험과 상보성 판정 (0.5p) ★
  - w 0.85→0.94 에서 combined 0.9470→0.9396 로 **거의 불변**
  - 생성 상관 하한 실험: 0.31~0.35(충족) 0.947 / 0.22~0.26(위반) 0.899
  - **결론: 남은 0.94 는 누수가 아니라 전통·대안 정보의 상보성**이며
    이는 프로젝트가 증명하려는 현상 자체다 → 목표 구간을 0.90~0.95 로 정정

4.6 씬파일러 유형 분화 (0.5p)
  - v1 은 두 조건이 동일 집합(교집합=합집합=45,000) → OR 정의 형해화
  - ADR-010: missing_only 40% / history_only 30% / both 30%
  - 교집합 13,592 < 합집합 45,000 으로 정상화

### 5장. 모델 비교 (3p)
- 실험 설계: 3 알고리즘 × 3 피처그룹 × 2 불균형처리 = 18조합, 총 616.4초
- 18조합 전체 표 (CV AUC±SD / test AUC / KS / PSI)
- 피처그룹별 최고: traditional 0.8355 / alternative 0.8902 / **combined 0.9460**
- **씬파일러 유형별 uplift** ★
  | 세그먼트 | uplift |
  |---|---:|
  | 일반 | +0.0946 |
  | thin: history_only | +0.1013 |
  | thin: missing_only | +0.1681 |
  | **thin: both** | **+0.1883** |
  → 전통 피처 정보량이 적을수록 대안 데이터 기여가 크다. ADR-001 설계 의도의 실증
- 불균형 처리 비교와 **채택 근거**: 성능은 CV 표준편차 미만 차이라 동등,
  SMOTE 는 합성 샘플로 학습해 거절 사유 설명에 부적합(신용정보법 §36-2) → `class_weight`
- 챔피언 선정: XGBoost vs LightGBM 유의차 없음(Tukey p_adj=0.6468, ADR-012)
  → 근거는 SHAP TreeExplainer 지원 성숙도
- 성능 목표 달성표, MLflow 실험 추적 (run 18개, Registry champion)
- 그림: `feature_group_auc.png`, `roc_pr_curves.png`, `confusion_matrices.png`, `thin_filer_uplift.png`

### 6장. XAI 분석 (3p)
- 전역: SHAP Summary(상위 15개), 유형별 비중
  1위 telecom_payment_rate **1.3460** (2위의 1.7배),
  피처당 평균 |SHAP| alternative **0.6563** vs traditional **0.2011** (3.3배)
- Dependence Plot 2종 (대안 1 + 전통 1) 해석
- 지역: **경계 사례**(p=0.375) Waterfall + 거절 사유 Top 5 자연어
  → 극단 사례(p=0.97)보다 경계 사례가 XAI 가치를 보여준다는 설계 의도 서술
- 승인 사례(p=0.25) 대비
- 거절 사유 생성 로직: 부호 규약(신용도 기여도, 음수=불리),
  reference_value = 동일 연령대 평균(센티넬 제외), 단위별 문장 템플릿, 방향 일치 검증
- 규제 대응: 거절 사유 3가지 이상 제시 요건 충족

### 7장. 공정성 검증과 완화 (5p) ★
`fairness_narrative.md` 기반.

7.1 실험 설계: 직접 경로 차단 3겹 (0.5p)
  - 편향은 피처에만(타겟 불변) / 보호 속성 학습 제외(ADR-008) / 후처리 완화(ADR-004)

7.2 완화 전 측정 (1p)
  - gender DI 0.9191, EO TPR 0.0350 / FPR 0.0696 (전부 충족)
  - age_group DI 0.8375, **min/max 0.7747 (4/5 규칙 미달)**, **EO FPR 0.1487 위반**
  - **DI 정의에 따라 판정이 갈린다**: SPEC §1 정의 0.8375 충족 vs 관행 min/max 0.7747 미달
    → 두 값 병기 원칙

7.3 bias_ratio 민감도: 두 종류의 간접 편향 (1.5p) ★★
  | bias_ratio | female | male | **gender DI** | 20-34 | 55+ | **age DI(min/max)** |
  |---:|---:|---:|---:|---:|---:|---:|
  | 0.0 | 0.8055 | 0.8031 | **1.0030** | 0.6947 | 0.8861 | **0.7840** |
  | 0.1 | 0.7805 | 0.8215 | 0.9502 | 0.6838 | 0.8844 | 0.7732 |
  | 0.2 | 0.7698 | 0.8283 | 0.9293 | 0.6803 | 0.8825 | 0.7709 |
  | 0.3 | 0.7603 | 0.8272 | **0.9191** | 0.6789 | 0.8764 | 0.7747 |
  - **성별 = 주입한 측정 편향**: bias 0 에서 DI 1.0030(완전 균등) → 단조 감소.
    SPEC §4 "bias_ratio 조정 시 승인율 편향이 변경된다" 요건의 실증
  - **연령대 = 피처 내재 상관**: bias 0 에서 이미 min/max DI **0.7840**(4/5 미달),
    0.3 까지 올려도 0.7747 로 거의 불변(−0.009).
    원인은 ADR-001 의 `nuisance = 0.7·소득 + 0.3·연령`
  - **실무적 함의**: 후자가 더 어렵다. 편향을 주입한 적이 없어도 대안 데이터가
    보호 속성과 자연 상관되어 있으면 그 상관이 그대로 결정에 실린다.
    원인 제거는 예측력 손실을 수반한다(4.5절의 w 레버 실험과 같은 구조)

7.4 Mitigation 결과 (1p)
  - 전/후 표: DI 0.9191→0.9942 / 0.8375→0.9433, EO FPR 0.1487→**0.0155**
  - 승인율 0.7940 → 0.8196~0.8492, 집단별 조정 임계값 표
  - **AUC 0.9453→0.86 은 성능 저하가 아니다**: 출력이 확률→이진 결정으로 바뀌어
    순위 정보 소실. 이진 예측의 AUC = (TPR+TNR)/2
  - **목적함수 교체 근거(ADR-004 v2)**: `accuracy` 는 승인율을 0.955 로 끌어올려
    심사 기능을 상실시켰다 → `balanced_accuracy`

7.5 종합 (1p)
  - 트레이드오프 차트 해석, 규제 관점 정리

### 8장. ANOVA 통계 검증 (3p)
8.1 관측 단위와 통제 방식 (1p) ★
  - ADR-005: RepeatedStratifiedKFold(5×4) → 그룹당 n=20
  - ADR-011: 360건을 그대로 넣으면 피처그룹 효과와 알고리즘 효과가 **교락**
    → 나머지 요인을 champion 조합으로 고정
  - **통제 vs pooled 대조표** ★
    | 검정 | 통제(n=20) F / η² | pooled(n=120) F / η² |
    |---|---|---|
    | 검증1 (피처그룹) | 4245.65 / **0.9933** | 1031.64 / 0.8525 |
    | 검증2 (알고리즘) | 804.38 / **0.9658** | 16.08 / **0.0826** |
    → 검증2 의 η² 가 0.9658 → 0.0826 으로 급락. 통제가 필요한 이유의 직접 증거

8.2 검증 1: 피처 그룹 (0.5p)
  - 집단 평균 traditional 0.8355 / alternative 0.8965 / combined 0.9479
  - F=4245.65, p=9.64e-63, η²=0.9933, Tukey 3쌍 모두 유의
  - 대안 단독으로도 전통을 앞서고(+0.056), 통합 시 추가 이득(+0.053)

8.3 검증 2: 알고리즘 (0.5p)
  - F=804.38, p=1.68e-42, η²=0.9658
  - Tukey: LR vs XGB/LGBM 유의, **XGB vs LGBM 유의하지 않음(p_adj=0.6468)**
  - → 주장 범위는 "부스팅 계열이 LR 보다 우수하다"까지 (ADR-012)

8.4 가정 진단 (0.5p)
  - Shapiro-Wilk: 두 검정 6집단 전부 정규성 충족 (W 0.9253~0.9847)
  - Levene: 검증1 p=0.0751, 검증2 p=0.1066 → 등분산 충족 (경계에 가까움을 명시)
  - 위반 시 Welch ANOVA / Kruskal-Wallis 병기 로직 구현 사실

8.5 η² > 0.95 해석 주의 (0.5p) ★
  - 집단이 서로 다른 피처 구성/알고리즘으로 **고정**돼 있고, 동일 데이터에 반복 CV 를
    적용해 집단 내 변동이 매우 작다(SD 0.0024~0.0047) → η² 과대평가
  - 반복 CV fold 는 완전 독립이 아니므로 **p-value 는 보수적으로** 읽는다
  - **실무 판단은 Tukey mean_diff 의 절대 크기로**: combined vs traditional +0.1096,
    XGB vs LGBM +0.0008(무의미)
  - 그림: `cv_fold_auc_distribution.png`

### 9장. 검증 과정에서 발견하고 수정한 오류 (1.5p) ★
> 각 오류는 자동화된 테스트 또는 교차 검증 절차에서 포착되었으며, 재발 방지 테스트를 함께 추가했다.

검증 절차가 실제로 작동했다는 증거로 제시한다. 각 항목: 증상 → 원인 → 조치 → 재발 방지.

| # | 오류 | 원인 | 재발 방지 |
|---|---|---|---|
| 1 | 거절 사유가 SHAP Waterfall 과 다른 피처를 가리킴 | 역스케일 값을 모델에 넣어 SHAP 계산 | 모델 입력/표시값 분리 + 회귀 테스트 |
| 2 | 참조 평균에 센티넬(−1) 포함 → "평균 −0회" | 씬파일러 결측 대체값이 평균에 섞임 | 결측 플래그 기준 제외 |
| 3 | `/explain` 이 연체 '기록 없음' 입력에서 500 | `float(None)` — **핵심 대상 집단이 걸리는 경로** | 센티넬 표기 + "기록 없음" 문장 분기 |
| 4 | 재학습 후 `/anova` 가 stale 결과 서빙 | ANOVA 재실행 누락 (F=4514.60 → 4245.65) | 부트스트랩 스크립트에 단계 포함 |
| 5 | 전처리기 joblib 역직렬화 불가 | `-m` 실행 시 클래스 `__module__` 이 `__main__` | CLI 분리 (`run_preprocess.py`) |
| 6 | 챔피언 근거가 통계와 불일치 | Tukey 유의차 없음인데 "성능 우위" 주장 | ADR-012 로 근거 교체 |
| 7 | 대시보드 import 만으로 앱 실행 | `main()` 이 모듈 최상위 | `__main__` 가드 |
| 8 | Docker 빌드 실패 2건 | BuildKit DNS / `fc-cache` 부재 | `network: host`, `fontconfig` |
| 9 | ADR-004 v2 개정이 문서에 반영되지 않음 | 문자열 치환이 공백 불일치로 **조용히 실패**(성공 메시지는 출력됨). 코드·설정에는 반영, 문서만 누락 | 편집 후 `assert` 로 패턴 일치 확인. **검증 없는 자동 편집의 위험 사례** |

### 10장. 한계 (1.5p)
1. **시뮬레이션 데이터의 구조적 상한** — combined AUC 0.9453 은 실제 신용평가 통상
   0.85~0.88 보다 높다. SPEC §4 의 상관 0.3~0.5 를 부도율 6.68% 이진 타겟에 적용한 결과.
   **절대 수치가 아니라 상대 비교로만** 해석
2. **상관 하한 여유 0.0099** — `utility_payment_rate` 0.3099. 설정을 조금만 바꾸면 요건 이탈.
   F1~F5 결과를 현 데이터로 동결한 이유
3. **임계값 0.30 의 정밀도 0.2899** — 재현율 0.8936 을 얻은 대가. 실무에서는
   승인율 목표·대손율 허용치를 함께 고려한 최적화 필요. 임계값 변경 시 공정성 지표와
   Mitigation 결과가 연동돼 바뀌므로 ADR-003 값으로 동결
4. **성별은 100% 합성** — GMSC 에 성별이 없어 무작위 50:50 으로 만들었다.
   실제 성별 분포·상관 구조와 무관하며, 7.3절의 "측정 편향" 해석은 이 합성 전제 위에 있다
5. **η² 과대평가와 fold 비독립성** (8.5절)
6. **ThresholdOptimizer 의 확률적 결정** — 보간 규칙이라 같은 고객에게 항상 같은 결정이
   나오지 않는다. 실서비스에서는 별도 검토 필요
7. **German Credit 미사용** — 표본 1,000건으로 부족 (ADR-006)

### 11장. 결론 (1p)
- 확정된 결론 문장 (PROGRESS §3-1 / `fairness_narrative.md` §5)
- 세 가지 발견 정리, 실무 시사점, 향후 과제

### 부록 (2p)
- A. ADR 색인 12건 (제목 + 한 줄 요약 + 개정 이력)
- B. API 명세 7개 엔드포인트 요약표
- C. 재현 절차 (`docker compose --profile setup run --rm bootstrap` → `docker compose up`)
- D. 테스트 166건 구성
- E. 극단 사례 Waterfall 2종

---

## 4. 작성 계획

| 단계 | 내용 | 예상 |
|---|---|---|
| 1 | `src/models/eda_assets.py` 작성 → EDA 그림 7종 생성 | 30분 |
| 2 | 본문 Markdown 작성 (`docs/experiment_report.md`) | 본작업 |
| 3 | PDF 변환 | 아래 참조 |

### PDF 변환 방법 — 확인 필요

한글 PDF 변환에 제약이 있다. 현재 환경 조사 결과:

| 방법 | 상태 |
|---|---|
| `pandoc` + LaTeX | **미설치** (`pandoc`, `xelatex` 없음) |
| WeasyPrint / wkhtmltopdf | **미설치** |
| matplotlib PdfPages | 사용 가능하나 문서 조판에 부적합 |
| **Markdown → HTML → 브라우저 인쇄** | 사용자가 직접 PDF 저장 |
| `reportlab` + 나눔고딕 | 설치 필요, 조판 직접 구현 |

→ **가장 현실적인 안**: 본문을 `docs/experiment_report.md` 로 작성하고,
자체 CSS 를 넣은 단일 HTML(`docs/experiment_report.html`)을 생성한다.
그림은 상대경로로 삽입하고 `@page` 규칙으로 페이지 나눔을 지정해,
브라우저에서 "PDF로 저장"하면 20페이지 이상 PDF 가 나오게 한다.
`pandoc` 설치가 가능하다면(`sudo apt install pandoc texlive-xetex fonts-nanum`)
자동 변환으로 바꿀 수 있다.

## 5. 확정 사항 (2026-09-10)

| 항목 | 결정 |
|---|---|
| 페이지 배분 | **32p 유지** (부록 포함 34p) |
| 4장 비중 | **5p 유지** |
| 9장 | **본문 유지**, 제목을 "검증 과정에서 발견하고 수정한 오류"로 변경, ADR-004 누락 추가 |
| PDF 변환 | **장별 Markdown → 단일 HTML → 브라우저 인쇄** |
| EDA 그림 | **7종 생성** (`src/analysis/eda.py`) |

### 작성 순서 (재료가 있는 장부터)

`4장 → 7장 → 8장 → 5장 → 6장 → 9·10·11장 → 3장(EDA 그림 후) → 1·2장 → 요약·부록`

장 하나가 끝날 때마다 `PROGRESS.md` 를 갱신해 세션이 끊겨도 이어갈 수 있게 한다.

### 파일 배치

```
docs/report/
├── 00_cover.md         표지 · 면책 · 목차
├── 01_summary.md       요약
├── 02_problem.md       1장 문제 정의
├── 03_system.md        2장 시스템 개요
├── 04_data_eda.md      3장 데이터와 EDA
├── 05_simulator.md     4장 방법론 (★)
├── 06_models.md        5장 모델 비교
├── 07_xai.md           6장 XAI 분석
├── 08_fairness.md      7장 공정성 (★)
├── 09_anova.md         8장 ANOVA
├── 10_errors.md        9장 발견하고 수정한 오류
├── 11_limitations.md   10장 한계
├── 12_conclusion.md    11장 결론
└── 99_appendix.md      부록
```

병합 스크립트가 위 순서대로 이어 붙이고 `@page` CSS·표지·목차·페이지 번호를 넣어
`docs/experiment_report.html` 을 만든다.
