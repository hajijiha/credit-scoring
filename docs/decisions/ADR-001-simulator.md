# ADR-001. 대안 데이터 시뮬레이터 설계

## 결정
GMSC 원본에 대안 데이터 5개 변수를 합성해 붙인다.

- `telecom_payment_rate`, `utility_payment_rate`, `spending_consistency`,
  `regular_payment_count`, `app_login_frequency`
- 생성 기준: **target(SeriousDlqin2yrs) · 소득(MonthlyIncome) · 연령(age)** 을 신호원으로 사용하고
  노이즈를 더해 **target 과의 상관계수가 0.3 ~ 0.5** 범위에 들어오도록 맞춘다.

씬파일러(thin filer)는 `simulator.thin_filer_ratio`(기본 0.30) 비율로 **무작위 선정**하고, 선정된 행에 한해
전통 신용 피처의 정보량을 구조적으로 낮춘다.

- 결측 처리: `NumberOfTime30-59DaysPastDueNotWorse`,
  `NumberOfTime60-89DaysPastDueNotWorse`, `NumberOfTimes90DaysLate`
- 강한 노이즈 주입: `RevolvingUtilizationOfUnsecuredLines`,
  `NumberOfOpenCreditLinesAndLoans`

## 근거
씬파일러의 전통 피처 정보량을 실제로 줄여놓지 않으면, 대안 데이터를 추가했을 때 나오는 성능 향상이
"대안 데이터의 기여"인지 "같은 정보를 두 번 넣은 누수"인지 구분할 수 없다.
정보량 축소를 데이터 생성 단계에서 구조적으로 강제해야 uplift 가 결과로서 해석 가능해진다.
상관 0.3~0.5 구간은 신호가 있으면서도 target 을 그대로 복사하지 않는 범위로 잡은 값이다.

## 버린 대안
- **상관 0.7 이상의 강한 대안 변수**: uplift 는 크게 나오지만 사실상 target 누수라 분석 가치가 없다.
- **씬파일러를 소득·연령 조건으로 선정**: 씬파일러 여부가 민감속성과 얽혀 ADR-002 의 편향 분석과 교란된다.
  무작위 선정으로 두 축을 분리한다.
- **전통 피처를 통째로 드롭**: 모델이 씬파일러 여부를 자명하게 학습해버려 비교가 무의미해진다.
