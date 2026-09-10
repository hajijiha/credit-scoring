# ADR-004. 공정성 완화 기법

| 버전 | 날짜 | 변경 내용 |
|---|---|---|
| v1 | 2026-09-08 | 최초 작성. `ThresholdOptimizer(constraints="equalized_odds")` 채택 |
| **v2 (현행)** | 2026-09-08 | 목적함수를 `accuracy_score` → **`balanced_accuracy_score`** 로 변경 |

## 결정
Fairlearn 의 `ThresholdOptimizer(constraints="equalized_odds", objective="balanced_accuracy_score")`
를 사용한다. 공정성 라이브러리는 Fairlearn 하나로 통일한다.

### 목적함수를 balanced_accuracy 로 바꾼 이유 (v2)

`accuracy_score` 로 적합했을 때의 결과(test 22,500행):

| 지표 | 완화 전 | 완화 후 (accuracy) | 완화 후 (balanced_accuracy) |
|---|---:|---:|---:|
| 전체 승인율 | 0.7940 | **0.9546 ~ 0.9572** | **0.8196 ~ 0.8492** |
| AUC | 0.9453 | 0.7389 ~ 0.7411 | 0.8592 ~ 0.8736 |
| DI Ratio (gender / age_group) | 0.9191 / 0.8375 | 1.0000 / 0.9759 | 0.9942 / 0.9433 |

`accuracy` 목적함수에서는 공정성 지표가 개선됐지만 **승인율이 0.80 에서 0.95 로 뛰었다.**
부도율이 6.7% 인 데이터에서 정확도를 최대화하면 "거의 전부 승인"이 최적해가 되기 때문이다.
이는 공정성 개선이 아니라 **심사 기능의 상실**이며, 대출 심사 시스템으로 방어할 수 없다.
공정성 지표가 좋아 보이는 이유도 "모두를 승인하면 집단 간 승인율 격차가 사라진다"는 자명한 사실일 뿐이다.

`balanced_accuracy_score` 는 TPR 과 TNR 의 평균이라 소수 클래스(부도)를 무시할 수 없다.
전부 승인하면 TPR 이 0 이 되어 목적함수가 0.5 로 떨어진다. 따라서 승인율이 현실적인 범위에 유지된다.

## 근거
- **후처리(post-processing)** 기법이라 학습된 모델을 그대로 두고 적용된다.
  덕분에 완화 전/후를 동일 모델·동일 예측 확률 위에서 비교할 수 있어 비교가 깔끔하다.
- FastAPI 응답에 "완화 전 결정 / 완화 후 결정"을 나란히 실어 노출하기 쉽다.
- Equalized Odds 는 ADR-002 에서 주입한 측정 편향(타겟은 그대로, 피처만 왜곡)에 대응하는 지표로 적절하다.

## 버린 대안
- **Reweighting(전처리)**: 학습 파이프라인 안쪽에 들어가 CV·MLflow 실험 구성과 얽힌다.
  재학습이 필요해 전/후 비교에서 모델 자체가 달라지고, 디버깅 비용이 크다.
- **ExponentiatedGradient(in-processing)**: 성능은 좋을 수 있으나 재학습 비용과 수렴 이슈가 있다.
- **AIF360**: Python 3.12 호환 문제로 도입하지 않는다.
