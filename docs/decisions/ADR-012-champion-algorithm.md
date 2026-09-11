# ADR-012. 챔피언 알고리즘 선정 근거

## 결정

챔피언 알고리즘은 **XGBoost** 로 유지하되, **선정 근거를 "성능 우위"에서
"SHAP TreeExplainer 지원 성숙도와 참조 자료의 풍부함"으로 변경한다.**

성능에 대해 주장할 수 있는 범위는 다음까지다.

> 부스팅 계열(XGBoost, LightGBM)이 Logistic Regression 보다 통계적으로 유의하게 우수하다.

**XGBoost 가 LightGBM 보다 낫다고는 주장하지 않는다.**

## 근거

ANOVA 검증 2(ADR-011 통제: `feature_group=combined`, `imbalance=class_weight`, 그룹당 n=20)의
Tukey HSD 결과(최종 데이터 `bias_ratio=0.3`, `data/processed/anova_results.json`):

| 비교 | mean_diff | p_adj | 유의 |
|---|---:|---:|---|
| logistic_regression vs xgboost | +0.0331 | < 0.0001 | 예 |
| lightgbm vs logistic_regression | −0.0323 | < 0.0001 | 예 |
| **lightgbm vs xgboost** | **+0.0008** | **0.6468** | **아니오** |

> 2026-09-11 수치 갱신: 최초 작성 시의 값(mean_diff +0.0009, p_adj 0.6137, 평균 0.9479 / 0.9470)은
> `bias_ratio=0.1` 데이터 기준이었다. ADR-002 v3 재생성 후 갱신되지 않고 남아 있던 것을
> 최종 데이터 값으로 교체한다. **결론(유의차 없음)은 바뀌지 않는다.**

집단 평균 AUC 는 xgboost 0.9451, lightgbm 0.9443 으로 차이가 0.0008 이다.
이는 집단 내 표준편차(0.0024~0.0027)보다도 작다. **두 알고리즘은 통계적으로 구분되지 않는다.**

따라서 "XGBoost 가 가장 성능이 좋아서 채택했다"는 서술은 데이터가 뒷받침하지 않는다.
유의차가 없는 두 후보 중 하나를 고르는 문제이므로 성능 외 기준으로 판단한다.

### XGBoost 를 고른 이유

- **SHAP TreeExplainer 지원 성숙도.** SHAP 은 XGBoost 를 1급 지원 대상으로 삼아 왔고,
  `tree_path_dependent` 경로가 가장 오래 검증됐다. F3 의 전역·지역 해석이 이 프로젝트의
  핵심 산출물이므로 해석 도구와의 궁합을 우선한다.
- **참조 자료의 풍부함.** 거절 사유 자동 생성처럼 규격 밖의 구현을 할 때 참고할 예제와
  트러블슈팅 자료가 많다. 교육 목적 프로젝트에서 재현·검증 비용을 낮춘다.

### 뒤집힐 수 있는 결정임을 명시

학습 시간은 이 환경(`n_jobs=4`)에서 비슷하다. 최종 데이터 기준 `class_weight` 조합의 fit 시간은
traditional_only XGBoost 19.0초 / LightGBM 22.3초, combined 24.7초 / 26.3초로 오히려
XGBoost 가 약간 빠르다(`model_results.csv` 의 `fit_seconds`). 일반적으로는 LightGBM 이 대용량에서
유리하다고 알려져 있으나 본 프로젝트 규모에서는 차이가 드러나지 않았다.
성능이 동등하므로 학습·추론 비용이 병목이 되면 LightGBM 으로 교체해도 논거가 훼손되지 않는다.

## 버린 대안

- **"XGBoost 가 최고 성능"을 근거로 유지**: Tukey HSD 가 유의차 없음을 보였으므로
  데이터와 어긋나는 서술이다. 리포트 심사에서 지적될 수 있다.
- **두 모델의 앙상블**: 성능 이득이 불확실한 반면 SHAP 해석과 API 서빙이 복잡해진다.
  설명가능성이 핵심 요건인 프로젝트에서 비용이 이득보다 크다.
- **LightGBM 으로 교체**: 성능상 동등하므로 가능하지만, 위의 해석 도구 성숙도 근거를 뒤집을
  이유가 없다.
