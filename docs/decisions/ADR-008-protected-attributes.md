# ADR-008. 보호 속성(gender, age_group)을 학습 피처에서 제외

## 결정

`gender`(male/female)와 `age_group`(20-34/35-54/55+)은 **모델 학습 피처에서 제외**한다.
다만 데이터프레임에서는 삭제하지 않고 원본 값 그대로 유지한다.

- `config/config.yaml` 의 `features` 섹션에서 학습 대상(`traditional`, `alternative`)과
  보호 속성(`protected`), 세그먼트 전용(`segment`)을 분리한다.
- `src/data/preprocessor.py` 는 보호 속성을 One-Hot 인코딩 대상에서 제외하고
  `passthrough_columns` 로 원본 문자열 값을 유지한다.
- 학습 코드는 `features.traditional` / `features.alternative` 만으로 X 를 구성한다.

보호 속성이 데이터에 남아 있어야 하는 이유: 공정성 지표(DI Ratio, Equalized Odds) 계산과
Fairlearn `ThresholdOptimizer` 의 `sensitive_features` 인자에 필요하다.

## 근거

- **예측력이 없다.** `gender` 는 시뮬레이터가 타겟과 독립인 난수로 합성한 값이다(ADR-002).
  학습에 넣어도 노이즈만 추가된다. `age_group` 은 이미 연속형 `age` 가 피처로 들어가 있어
  구간화된 중복 정보다.
- **금융 규제.** 성별·연령을 신용평가 모델의 입력으로 직접 사용하는 것은 부적절하다.
  SPEC §7 은 민감정보의 피처 사용을 금지하며, 금융위원회 AI 가이드라인의 공정성 기준과도 맞지 않는다.
- **이 프로젝트의 핵심 검증 대상이 바뀐다.** 보호 속성을 뺐는데도 대안 피처를 경유한
  **간접 편향(proxy discrimination)** 이 DI Ratio 에 나타나는지가 이 실험의 핵심 질문이다.
  ADR-002 는 `telecom_payment_rate` / `utility_payment_rate` 에만 측정 편향을 주입하므로,
  보호 속성이 없는 모델에서 집단 간 승인율 격차가 관측된다면 그것은 전적으로 대안 피처를
  통해 전파된 간접 편향이다. 이 인과 경로를 깔끔하게 보이려면 직접 경로를 차단해야 한다.

## 버린 대안

- **보호 속성을 피처로 포함**: 모델이 성별·연령대를 직접 학습하면, 관측된 공정성 격차가
  직접 차별(보호 속성 자체)에서 온 것인지 간접 차별(대안 피처 경유)에서 온 것인지
  구분할 수 없게 된다. Mitigation 전/후 비교의 해석도 흐려진다.
- **보호 속성 컬럼을 아예 드롭**: 공정성 평가와 ThresholdOptimizer 가 불가능해진다.
