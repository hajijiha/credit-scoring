# 개발 지침

- 임계값, 비율, 포트와 `random_state`는 `config/config.yaml`에서 읽는다. 설정 접근은 `src/config.py`를 사용한다.
- 파일 경로는 프로젝트 루트 기준 상대경로로 작성한다.
- 공정성 분석은 Fairlearn을 사용한다. AIF360은 이 프로젝트의 Python 3.12 환경과 호환되지 않는다.
- 대시보드는 모델 파일을 직접 읽지 않고 FastAPI로 추론을 요청한다.
- 학습, 튜닝과 임계값 선택에 테스트 라벨을 사용하지 않는다. 임계값 조정은 validation 세트에서 수행한다.
- `gender`, `age_group`, `is_thin_filer`, `credit_history_months`는 학습 피처에서 제외한다. 학습 입력은 `config.features`의 traditional / alternative 구성으로 제한한다.

설계 변경의 근거와 검증 조건은 [설계 결정](decisions/)에 기록한다. 기능 변경 시 관련 테스트, 데이터·모델 산출물과 보고서를 함께 갱신한다.

실행과 테스트 명령은 [README](../README.md), 구현 상태와 알려진 한계는 [PROGRESS.md](../PROGRESS.md)에 있다.
