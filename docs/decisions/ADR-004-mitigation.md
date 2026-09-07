# ADR-004. 공정성 완화 기법

## 결정
Fairlearn 의 `ThresholdOptimizer(constraints="equalized_odds")` 를 사용한다.
공정성 라이브러리는 Fairlearn 하나로 통일한다.

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
